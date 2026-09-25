"""Candle and volume contract for the four live coins.

Missing OHLC or volume stays missing. Zero is never substituted.
The forming bar is reported separately from the last completed bar.
"""

from __future__ import annotations

import sqlite3
from datetime import datetime, timezone
from typing import Any

UNIVERSE = ("BTC-USDT", "ETH-USDT", "SOL-USDT", "XRP-USDT")
INTERVALS = ("1m", "3m", "5m", "15m", "1h", "4h")
INTERVAL_SEC = {"1m": 60, "3m": 180, "5m": 300, "15m": 900, "1h": 3600, "4h": 14400}
STALE_MULT = 2.0
CONSUMERS = {
    "1m": "SCALP_V2",
    "3m": "SCALP_V2",
    "5m": "SCALP_V2,DAY_V2_CONFIRM",
    "15m": "DAY_V2",
    "1h": "DAY_V2_CONTEXT",
    "4h": "DAY_V2_CONTEXT",
}


def _parse_ts(value: object) -> float | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        num = float(value)
        return num / 1000.0 if num > 10_000_000_000 else num
    text = str(value).strip()
    if not text:
        return None
    try:
        num = float(text)
        return num / 1000.0 if num > 10_000_000_000 else num
    except ValueError:
        pass
    cleaned = text.replace("Z", "+00:00")
    if " " in cleaned and "T" not in cleaned:
        cleaned = cleaned.replace(" ", "T", 1)
    try:
        dt = datetime.fromisoformat(cleaned)
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.timestamp()


def _positive(value: object) -> float | None:
    try:
        num = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    return num if num > 0 else None


def _non_null_ohlc(row: tuple) -> bool:
    return all(row[i] is not None for i in range(1, 5))


def _completed_rows(rows: list[tuple], interval_sec: int, now: float) -> tuple[list[tuple], tuple | None]:
    forming = None
    completed: list[tuple] = []
    for row in rows:
        ts = _parse_ts(row[0])
        if ts is None:
            continue
        if ts + interval_sec > now + 2:
            forming = row
            continue
        completed.append(row)
    return completed, forming


def candle_contract_matrix(
    db_path: str,
    *,
    now: float | None = None,
    books: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    moment = float(now if now is not None else datetime.now(timezone.utc).timestamp())
    conn = sqlite3.connect(db_path, timeout=10)
    cells: list[dict[str, Any]] = []
    try:
        tables = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        has = "feature_ohlcv" in tables
        for symbol in UNIVERSE:
            book = (books or {}).get(symbol) or (books or {}).get(symbol.replace("-", "")) or {}
            book_ts = _parse_ts(book.get("ts")) if book else None
            bid = _positive(book.get("bid"))
            ask = _positive(book.get("ask"))
            book_px = _positive(book.get("price"))
            if book_px is None and bid is not None and ask is not None:
                book_px = (bid + ask) / 2.0
            spread = (ask - bid) if bid is not None and ask is not None else None
            book_age = None if book_ts is None else max(0.0, moment - book_ts)
            cells.append(
                {
                    "symbol": symbol,
                    "timeframe": "book",
                    "bid": bid,
                    "ask": ask,
                    "spread": spread,
                    "book_timestamp": book_ts,
                    "book_age_sec": book_age,
                    "last_completed_ts": book_ts,
                    "forming_ts": None,
                    "age_sec": book_age,
                    "ohlc_non_null": book_px is not None,
                    "volume_non_null": None,
                    "expected_interval_sec": None,
                    "missing_bar_count": 0 if book_px else 1,
                    "source": str(book.get("source") or "absent"),
                    "cache_key": f"price:{symbol.replace('-', '')}",
                    "consumer": "SCALP_V2,DAY_V2",
                    "stale_threshold_sec": 30,
                    "stale": book_ts is None or (book_age or 0) > 30,
                    "recovery": "fetch_canonical_mark_once_per_cycle" if book_px else "book_absent",
                }
            )
            if not has:
                for interval in INTERVALS:
                    cells.append(_absent(symbol, interval))
                continue
            for interval in INTERVALS:
                sec = INTERVAL_SEC[interval]
                rows = conn.execute(
                    """
                    SELECT ts, open, high, low, close, volume
                    FROM feature_ohlcv
                    WHERE symbol=? AND interval=?
                    ORDER BY ts DESC
                    LIMIT 20
                    """,
                    (symbol, interval),
                ).fetchall()
                chrono = list(reversed(rows))
                completed, forming = _completed_rows(chrono, sec, moment)
                last = completed[-1] if completed else None
                prev = completed[-2] if len(completed) >= 2 else None
                last_ts = _parse_ts(last[0]) if last else None
                missing = 0
                if last is not None and prev is not None:
                    gap = (_parse_ts(last[0]) or 0) - (_parse_ts(prev[0]) or 0)
                    if sec > 0 and gap > sec * 1.5:
                        missing = max(0, round(gap / sec) - 1)
                age = None if last_ts is None else max(0.0, moment - last_ts)
                close_age = None if last_ts is None else max(0.0, moment - (last_ts + sec))
                stale_limit = sec * STALE_MULT
                cells.append(
                    {
                        "symbol": symbol,
                        "timeframe": interval,
                        "last_completed_ts": last[0] if last else None,
                        "forming_ts": forming[0] if forming else None,
                        "age_sec": age,
                        "close_age_sec": close_age,
                        "ohlc_non_null": bool(last) and _non_null_ohlc(last),
                        "volume_non_null": bool(last) and last[5] is not None,
                        "expected_interval_sec": sec,
                        "missing_bar_count": missing if last else None,
                        "source": "feature_ohlcv" if last else "absent",
                        "cache_key": f"feature_ohlcv:{symbol}:{interval}",
                        "consumer": CONSUMERS[interval],
                        "stale_threshold_sec": stale_limit,
                        "stale": last_ts is None or (close_age or 0) > stale_limit,
                        "recovery": "skip_blank_row_wait_for_next_closed_bar" if last else "writer_has_not_committed",
                    }
                )
    finally:
        conn.close()
    return {"as_of": moment, "cells": cells}


def _absent(symbol: str, interval: str) -> dict[str, Any]:
    sec = INTERVAL_SEC[interval]
    return {
        "symbol": symbol,
        "timeframe": interval,
        "last_completed_ts": None,
        "forming_ts": None,
        "age_sec": None,
        "ohlc_non_null": False,
        "volume_non_null": False,
        "expected_interval_sec": sec,
        "missing_bar_count": None,
        "source": "absent",
        "cache_key": f"feature_ohlcv:{symbol}:{interval}",
        "consumer": CONSUMERS[interval],
        "stale_threshold_sec": sec * STALE_MULT,
        "stale": True,
        "recovery": "writer_has_not_committed",
    }


def load_closed_bars(db_path: str, symbol: str, interval: str, limit: int, *, as_of: float) -> list[dict]:
    """Oldest-first closed bars. Blank OHLC or volume rows are skipped, not zeroed."""
    sec = INTERVAL_SEC.get(interval, 60)
    conn = sqlite3.connect(db_path, timeout=10)
    try:
        rows = conn.execute(
            "SELECT ts, open, high, low, close, volume FROM feature_ohlcv WHERE symbol=? AND interval=? ORDER BY ts DESC LIMIT ?",
            (symbol, interval, max(limit * 2, limit)),
        ).fetchall()
    except sqlite3.OperationalError:
        return []
    finally:
        conn.close()
    out: list[dict] = []
    for row in reversed(rows):
        if any(row[i] is None for i in range(1, 6)):
            continue
        ts = _parse_ts(row[0])
        if ts is None or ts + sec > as_of + 2:
            continue
        out.append(
            {
                "ts": row[0],
                "ts_epoch": ts,
                "open": float(row[1]),
                "high": float(row[2]),
                "low": float(row[3]),
                "close": float(row[4]),
                "volume": float(row[5]),
            }
        )
    return out[-limit:]
