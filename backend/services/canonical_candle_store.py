"""Durable canonical OHLCV store.

Identity is (symbol, interval, open_timestamp). Persist-now snapshots are not
candle identity. Zero-volume exchange bars are stored honestly.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
from typing import Any

from sqlalchemy import and_, asc, desc, select

from backend.config.canonical_candle_intervals import (
    CANONICAL_CANDLE_INTERVALS,
    align_open_ms,
    interval_ms,
    interval_sec,
    is_supported_interval,
    redis_window,
)
from backend.services.feature_store import FeatureOHLCV, SessionLocal

logger = logging.getLogger(__name__)

RESEARCH_TABLE_RETIRED = "day_research_klines"
RESEARCH_RETIRE_REASON = "retired_stale_september_2_use_canonical_feature_ohlcv"


def api_symbol(symbol: str) -> str:
    s = (symbol or "").strip().upper().replace(" ", "").replace("/", "").replace("-", "")
    if s.endswith("USD") and not s.endswith("USDT"):
        s = s[:-3] + "USDT"
    return s


def db_symbol(symbol: str) -> str:
    api = api_symbol(symbol)
    if api.endswith("USDT") and len(api) > 4:
        return f"{api[:-4]}-USDT"
    return api


def redis_symbol(symbol: str) -> str:
    return api_symbol(symbol)


def _dec(value: Any) -> Decimal:
    if value is None:
        return Decimal("0")
    if isinstance(value, Decimal):
        return value
    try:
        return Decimal(str(value))
    except (InvalidOperation, ValueError, TypeError):
        return Decimal("0")


def parse_ohlcv_float(value: Any) -> float:
    return float(_dec(value))


def open_dt_from_ms(open_ms: int) -> datetime:
    return datetime.fromtimestamp(int(open_ms) / 1000.0, tz=timezone.utc)


def open_ms_from_dt(ts: datetime | None) -> int | None:
    if ts is None:
        return None
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=timezone.utc)
    return int(ts.timestamp() * 1000)


def is_aligned_open_ms(open_ms: int, interval: str) -> bool:
    return int(open_ms) == align_open_ms(int(open_ms), interval)


def candle_dict(
    *,
    symbol: str,
    interval: str,
    open_ms: int,
    open_: float,
    high: float,
    low: float,
    close: float,
    volume: float,
    completed: bool,
    source: str = "binance.us",
    zero_volume_legitimate: bool | None = None,
) -> dict[str, Any]:
    zv = bool(volume == 0.0) if zero_volume_legitimate is None else bool(zero_volume_legitimate)
    return {
        "symbol": api_symbol(symbol),
        "db_symbol": db_symbol(symbol),
        "interval": interval,
        "open_ms": int(open_ms),
        "open_time": open_dt_from_ms(open_ms).isoformat(),
        "open": float(open_),
        "high": float(high),
        "low": float(low),
        "close": float(close),
        "volume": float(volume),
        "completed": bool(completed),
        "forming": not bool(completed),
        "source": source,
        "zero_volume_legitimate": zv,
    }


def row_from_binance_kline(kline: list[Any], *, symbol: str, interval: str, now_ms: int) -> dict[str, Any] | None:
    if not isinstance(kline, (list, tuple)) or len(kline) < 6:
        return None
    open_ms = int(kline[0])
    if not is_aligned_open_ms(open_ms, interval):
        open_ms = align_open_ms(open_ms, interval)
    width = interval_ms(interval)
    completed = (open_ms + width) <= int(now_ms)
    return candle_dict(
        symbol=symbol,
        interval=interval,
        open_ms=open_ms,
        open_=parse_ohlcv_float(kline[1]),
        high=parse_ohlcv_float(kline[2]),
        low=parse_ohlcv_float(kline[3]),
        close=parse_ohlcv_float(kline[4]),
        volume=parse_ohlcv_float(kline[5]),
        completed=completed,
        source="binance.us",
    )


def expected_open_ms_range(start_ms: int, end_ms: int, interval: str) -> list[int]:
    width = interval_ms(interval)
    cursor = align_open_ms(start_ms, interval)
    if cursor < start_ms:
        cursor += width
    out: list[int] = []
    last = align_open_ms(end_ms, interval)
    while cursor <= last:
        out.append(cursor)
        cursor += width
    return out


def upsert_completed_candles(symbol: str, interval: str, candles: list[dict[str, Any]]) -> dict[str, int]:
    """Insert or replace completed candles by (db_symbol, interval, open ts)."""
    if not is_supported_interval(interval):
        raise ValueError(f"unsupported interval {interval}")
    db_sym = db_symbol(symbol)
    inserted = 0
    updated = 0
    skipped_forming = 0
    with SessionLocal() as session:
        for candle in candles:
            if not candle.get("completed", True):
                skipped_forming += 1
                continue
            open_ms = int(candle["open_ms"])
            ts = open_dt_from_ms(open_ms)
            existing = (
                session.execute(
                    select(FeatureOHLCV).where(
                        and_(
                            FeatureOHLCV.symbol == db_sym,
                            FeatureOHLCV.interval == interval,
                            FeatureOHLCV.ts == ts,
                        )
                    )
                )
                .scalars()
                .first()
            )
            payload = {
                "open": parse_ohlcv_float(candle.get("open")),
                "high": parse_ohlcv_float(candle.get("high")),
                "low": parse_ohlcv_float(candle.get("low")),
                "close": parse_ohlcv_float(candle.get("close")),
                "volume": parse_ohlcv_float(candle.get("volume")),
            }
            if existing is None:
                session.add(
                    FeatureOHLCV(
                        symbol=db_sym,
                        interval=interval,
                        ts=ts,
                        **payload,
                    )
                )
                inserted += 1
            else:
                existing.open = payload["open"]
                existing.high = payload["high"]
                existing.low = payload["low"]
                existing.close = payload["close"]
                existing.volume = payload["volume"]
                updated += 1
        session.commit()
    return {"inserted": inserted, "updated": updated, "skipped_forming": skipped_forming}


def load_aligned_candles(
    symbol: str,
    interval: str,
    *,
    start_ms: int | None = None,
    end_ms: int | None = None,
    limit: int | None = None,
) -> list[dict[str, Any]]:
    db_sym = db_symbol(symbol)
    with SessionLocal() as session:
        conds = [
            FeatureOHLCV.symbol == db_sym,
            FeatureOHLCV.interval == interval,
        ]
        if start_ms is not None:
            conds.append(FeatureOHLCV.ts >= open_dt_from_ms(start_ms))
        if end_ms is not None:
            conds.append(FeatureOHLCV.ts <= open_dt_from_ms(end_ms))
        if limit is not None and limit > 0 and start_ms is None and end_ms is None:
            fetch_n = int(limit) * 4
            stmt = select(FeatureOHLCV).where(and_(*conds)).order_by(desc(FeatureOHLCV.ts)).limit(fetch_n)
            rows = list(reversed(list(session.execute(stmt).scalars().all())))
        else:
            stmt = select(FeatureOHLCV).where(and_(*conds)).order_by(asc(FeatureOHLCV.ts))
            rows = list(session.execute(stmt).scalars().all())
    out: list[dict[str, Any]] = []
    seen: set[int] = set()
    for row in rows:
        open_ms = open_ms_from_dt(row.ts)
        if open_ms is None or not is_aligned_open_ms(open_ms, interval):
            continue
        if open_ms in seen:
            continue
        seen.add(open_ms)
        out.append(
            candle_dict(
                symbol=symbol,
                interval=interval,
                open_ms=open_ms,
                open_=parse_ohlcv_float(row.open),
                high=parse_ohlcv_float(row.high),
                low=parse_ohlcv_float(row.low),
                close=parse_ohlcv_float(row.close),
                volume=parse_ohlcv_float(row.volume),
                completed=True,
                source="sqlite",
            )
        )
    if limit is not None and limit > 0:
        out = out[-int(limit) :]
    return out


def continuity_report(
    symbol: str,
    interval: str,
    *,
    start_ms: int,
    end_completed_ms: int,
) -> dict[str, Any]:
    rows = load_aligned_candles(symbol, interval, start_ms=start_ms, end_ms=end_completed_ms)
    have = [int(r["open_ms"]) for r in rows]
    expected = expected_open_ms_range(start_ms, end_completed_ms, interval)
    have_set = set(have)
    missing = [ts for ts in expected if ts not in have_set]
    extra = [ts for ts in have if ts not in set(expected)]
    duplicates = len(have) - len(have_set)
    out_of_order = sum(1 for i in range(1, len(have)) if have[i] < have[i - 1])
    latest = rows[-1] if rows else None
    return {
        "symbol": api_symbol(symbol),
        "interval": interval,
        "row_count": len(rows),
        "expected_count": len(expected),
        "missing_count": len(missing),
        "missing_timestamps": missing[:50],
        "duplicate_count": duplicates,
        "out_of_order_count": out_of_order,
        "extra_unaligned_or_outside": extra[:20],
        "oldest_open_ms": have[0] if have else None,
        "newest_completed_ms": have[-1] if have else None,
        "latest_ohlcv": latest,
    }


def earliest_aligned_1m_open_ms() -> int | None:
    with SessionLocal() as session:
        rows = session.execute(select(FeatureOHLCV.ts).where(FeatureOHLCV.interval == "1m").order_by(asc(FeatureOHLCV.ts)).limit(500)).scalars().all()
    for ts in rows:
        open_ms = open_ms_from_dt(ts)
        if open_ms is not None and is_aligned_open_ms(open_ms, "1m"):
            return open_ms
    # Persist-now history: fall back to the earliest retained persist timestamp,
    # aligned down to the 1m bucket so backfill starts at that calendar minute.
    if rows:
        open_ms = open_ms_from_dt(rows[0])
        if open_ms is not None:
            return align_open_ms(open_ms, "1m")
    return None


def delete_unaligned_persist_now_rows(symbol: str, interval: str, *, limit: int = 50000) -> int:
    """Remove persist-now snapshots that are not exchange-open timestamps."""
    db_sym = db_symbol(symbol)
    removed = 0
    with SessionLocal() as session:
        rows = session.execute(select(FeatureOHLCV).where(and_(FeatureOHLCV.symbol == db_sym, FeatureOHLCV.interval == interval))).scalars().all()
        for row in rows:
            open_ms = open_ms_from_dt(row.ts)
            if open_ms is None or not is_aligned_open_ms(open_ms, interval):
                session.delete(row)
                removed += 1
                if removed >= limit:
                    break
        session.commit()
    return removed


def aggregate_exact_from_1m(bars_1m: list[dict[str, Any]], interval: str) -> list[dict[str, Any]]:
    """Aggregate completed 1m bars into `interval` using exact UTC components only."""
    need = interval_ms(interval) // interval_ms("1m")
    if need < 2:
        return list(bars_1m)
    grouped: dict[int, list[dict[str, Any]]] = {}
    for bar in bars_1m:
        if not bar.get("completed", True):
            continue
        open_ms = int(bar["open_ms"])
        bucket = align_open_ms(open_ms, interval)
        grouped.setdefault(bucket, []).append(bar)
    out: list[dict[str, Any]] = []
    for bucket in sorted(grouped):
        chunk = sorted(grouped[bucket], key=lambda r: int(r["open_ms"]))
        expected = [bucket + i * interval_ms("1m") for i in range(need)]
        have = [int(r["open_ms"]) for r in chunk]
        if have != expected:
            continue
        out.append(
            candle_dict(
                symbol=chunk[0]["symbol"],
                interval=interval,
                open_ms=bucket,
                open_=parse_ohlcv_float(chunk[0]["open"]),
                high=max(parse_ohlcv_float(r["high"]) for r in chunk),
                low=min(parse_ohlcv_float(r["low"]) for r in chunk),
                close=parse_ohlcv_float(chunk[-1]["close"]),
                volume=sum(parse_ohlcv_float(r["volume"]) for r in chunk),
                completed=True,
                source="aggregate_1m",
            )
        )
    return out


def redis_klines_key(symbol: str, interval: str) -> str:
    return f"klines:{redis_symbol(symbol)}:{interval}"


def redis_forming_key(symbol: str, interval: str) -> str:
    return f"klines:{redis_symbol(symbol)}:{interval}:forming"


def redis_integrity_key(symbol: str, interval: str) -> str:
    return f"candle_integrity:{redis_symbol(symbol)}:{interval}"


def redis_checkpoint_key(symbol: str, interval: str) -> str:
    return f"candle_checkpoint:{redis_symbol(symbol)}:{interval}"


def to_redis_row(candle: dict[str, Any]) -> list[float]:
    return [
        float(int(candle["open_ms"]) // 1000),
        float(candle["open"]),
        float(candle["high"]),
        float(candle["low"]),
        float(candle["close"]),
        float(candle["volume"]),
    ]


def merge_redis_rows(existing: list[list[Any]], incoming: list[dict[str, Any]], interval: str) -> list[list[float]]:
    by_ts: dict[int, list[float]] = {}
    for row in existing:
        if not isinstance(row, (list, tuple)) or len(row) < 6:
            continue
        ts_sec = int(float(row[0]))
        by_ts[ts_sec] = [float(x) for x in row[:6]]
    for candle in incoming:
        if not candle.get("completed", True):
            continue
        row = to_redis_row(candle)
        by_ts[int(row[0])] = row
    ordered = [by_ts[ts] for ts in sorted(by_ts)]
    keep = redis_window(interval)
    if len(ordered) > keep:
        ordered = ordered[-keep:]
    return ordered


def parse_redis_rows(raw: Any) -> list[list[float]]:
    if not raw:
        return []
    if isinstance(raw, (bytes, bytearray)):
        raw = raw.decode()
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except (TypeError, ValueError):
            return []
    if not isinstance(raw, list):
        return []
    out: list[list[float]] = []
    for row in raw:
        if isinstance(row, (list, tuple)) and len(row) >= 6:
            out.append([float(x) for x in row[:6]])
    return out


def refuse_research_table_read() -> dict[str, Any]:
    return {
        "table": RESEARCH_TABLE_RETIRED,
        "retired": True,
        "reason": RESEARCH_RETIRE_REASON,
        "canonical": "feature_ohlcv aligned open timestamps + Redis klines:{SYM}:{tf}",
    }


__all__ = [
    "CANONICAL_CANDLE_INTERVALS",
    "RESEARCH_RETIRE_REASON",
    "RESEARCH_TABLE_RETIRED",
    "aggregate_exact_from_1m",
    "api_symbol",
    "candle_dict",
    "continuity_report",
    "db_symbol",
    "delete_unaligned_persist_now_rows",
    "earliest_aligned_1m_open_ms",
    "expected_open_ms_range",
    "is_aligned_open_ms",
    "load_aligned_candles",
    "merge_redis_rows",
    "open_dt_from_ms",
    "open_ms_from_dt",
    "parse_ohlcv_float",
    "parse_redis_rows",
    "redis_checkpoint_key",
    "redis_forming_key",
    "redis_integrity_key",
    "redis_klines_key",
    "redis_symbol",
    "refuse_research_table_read",
    "row_from_binance_kline",
    "to_redis_row",
    "upsert_completed_candles",
]
