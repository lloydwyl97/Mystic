"""
Feature Store
-------------
Persistent storage for live ticks and OHLCV aggregates.

Notes
- SQLite (or DATABASE_URL) via backend.services.db helpers
- Minimal read/write helpers; safe defaults; no external data sources
- Back-compat: keeps DB_PATH constant used elsewhere
"""

from __future__ import annotations

import os
from collections.abc import Iterable
from datetime import datetime, timedelta, timezone
from typing import Any

from sqlalchemy import (
    Column,
    DateTime,
    Float,
    Index,
    Integer,
    String,
    and_,
    asc,
    delete,
    desc,
    func,
    select,
)
from sqlalchemy.orm import declarative_base

from backend.services.db import get_engine, get_sessionmaker

# Back-compat path used by labeler/online_trainer/paper_trader
DB_PATH = os.getenv("MYSTIC_DB_PATH", "mystic_trading.db")

ENGINE = get_engine()
SessionLocal = get_sessionmaker(ENGINE)
Base = declarative_base()

# -------------------------
# Models
# -------------------------


class FeatureTick(Base):
    __tablename__ = "feature_ticks"

    id = Column(Integer, primary_key=True)
    symbol = Column(String(32), index=True)
    price = Column(Float)  # type: ignore[assignment]
    bid = Column(Float)  # type: ignore[assignment]
    ask = Column(Float)  # type: ignore[assignment]
    volume_24h = Column(Float)  # type: ignore[assignment]
    change_24h = Column(Float)  # type: ignore[assignment]
    ts = Column(DateTime, default=lambda: datetime.now(timezone.utc), index=True)


class FeatureOHLCV(Base):
    __tablename__ = "feature_ohlcv"

    id = Column(Integer, primary_key=True)
    symbol = Column(String(32), index=True)
    interval = Column(String(8), index=True)
    open = Column(Float)  # type: ignore[assignment]
    high = Column(Float)  # type: ignore[assignment]
    low = Column(Float)  # type: ignore[assignment]
    close = Column(Float)  # type: ignore[assignment]
    volume = Column(Float)  # type: ignore[assignment]
    ts = Column(DateTime, default=lambda: datetime.now(timezone.utc), index=True)


Index("ix_feature_ticks_symbol_ts", FeatureTick.symbol, FeatureTick.ts)
Index(
    "ix_feature_ohlcv_symbol_interval_ts",
    FeatureOHLCV.symbol,
    FeatureOHLCV.interval,
    FeatureOHLCV.ts,
)

# -------------------------
# Schema init
# -------------------------


def init_feature_store() -> None:
    """Create tables if they do not exist."""
    Base.metadata.create_all(ENGINE)


# -------------------------
# Write helpers
# -------------------------


def _f(v: Any, default: float = 0.0) -> float:
    try:
        if v is None:
            return default
        if isinstance(v, (int, float)):
            return float(v)
        return float(str(v))
    except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError):
        return default


def insert_tick(symbol: str, data: dict[str, Any]) -> None:
    """
    Insert a live tick snapshot.
    Expected keys in `data` (best-effort): price, bid, ask, volume_24h|baseVolume, change_24h|percentage
    """
    with SessionLocal() as s:
        row = FeatureTick(
            symbol=str(symbol),
            price=_f(data.get("price")),
            bid=_f(data.get("bid")),
            ask=_f(data.get("ask")),
            volume_24h=_f(data.get("volume_24h", data.get("baseVolume"))),
            change_24h=_f(data.get("change_24h", data.get("percentage"))),
            ts=datetime.now(timezone.utc),
        )
        s.add(row)
        s.commit()


def insert_ticks_bulk(items: Iterable[tuple[str, dict[str, Any]]]) -> int:
    """
    Bulk insert ticks.
    items: iterable of (symbol, data_dict)
    Returns number of inserted rows.
    """
    inserted = 0
    with SessionLocal() as s:
        for symbol, data in items:
            s.add(
                FeatureTick(
                    symbol=str(symbol),
                    price=_f(data.get("price")),
                    bid=_f(data.get("bid")),
                    ask=_f(data.get("ask")),
                    volume_24h=_f(data.get("volume_24h", data.get("baseVolume"))),
                    change_24h=_f(data.get("change_24h", data.get("percentage"))),
                    ts=datetime.now(timezone.utc),
                ),
            )
            inserted += 1
        s.commit()
    return inserted


def insert_ohlcv(symbol: str, interval: str, candle: dict[str, float]) -> None:
    """
    Insert one OHLCV candle (latest).
    candle: {open, high, low, close, volume}
    """
    with SessionLocal() as s:
        row = FeatureOHLCV(
            symbol=str(symbol),
            interval=str(interval),
            open=_f(candle.get("open")),
            high=_f(candle.get("high")),
            low=_f(candle.get("low")),
            close=_f(candle.get("close")),
            volume=_f(candle.get("volume")),
            ts=datetime.now(timezone.utc),
        )
        s.add(row)
        s.commit()


def insert_ohlcv_bulk(symbol: str, interval: str, candles: Iterable[dict[str, Any]]) -> int:
    """
    Bulk insert multiple candles for a symbol/interval.
    Each dict should include: open, high, low, close, volume, and optionally ts (datetime).
    Returns number of inserted rows.
    """
    inserted = 0
    with SessionLocal() as s:
        for c in candles:
            ts = c.get("ts")
            row = FeatureOHLCV(
                symbol=str(symbol),
                interval=str(interval),
                open=_f(c.get("open")),
                high=_f(c.get("high")),
                low=_f(c.get("low")),
                close=_f(c.get("close")),
                volume=_f(c.get("volume")),
                ts=ts if isinstance(ts, datetime) else datetime.now(timezone.utc),
            )
            s.add(row)
            inserted += 1
        s.commit()
    return inserted


# -------------------------
# Read helpers
# -------------------------


def _dt(dt_like: Any) -> datetime | None:
    if dt_like is None:
        return None
    if isinstance(dt_like, datetime):
        return dt_like if dt_like.tzinfo else dt_like.replace(tzinfo=timezone.utc)
    # accept unix seconds
    try:
        return datetime.fromtimestamp(float(dt_like), tz=timezone.utc)
    except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError):
        return None


def get_latest_tick(symbol: str) -> dict[str, Any] | None:
    """Return the most recent tick for symbol (or None)."""
    with SessionLocal() as s:
        stmt = select(FeatureTick).where(FeatureTick.symbol == str(symbol)).order_by(desc(FeatureTick.ts)).limit(1)
        row = s.execute(stmt).scalars().first()
        if not row:
            return None
        return {
            "symbol": row.symbol,
            "price": row.price,
            "bid": row.bid,
            "ask": row.ask,
            "volume_24h": row.volume_24h,
            "change_24h": row.change_24h,
            "ts": row.ts.isoformat() if row.ts else None,
        }


def get_recent_ticks(symbol: str | None = None, limit: int = 100) -> list[dict[str, Any]]:
    """Return recent ticks, optionally filtered by symbol."""
    with SessionLocal() as s:
        if symbol:
            stmt = select(FeatureTick).where(FeatureTick.symbol == str(symbol)).order_by(desc(FeatureTick.ts)).limit(int(limit))
        else:
            stmt = select(FeatureTick).order_by(desc(FeatureTick.ts)).limit(int(limit))
        rows = s.execute(stmt).scalars().all()
        return [
            {
                "symbol": r.symbol,
                "price": r.price,
                "bid": r.bid,
                "ask": r.ask,
                "volume_24h": r.volume_24h,
                "change_24h": r.change_24h,
                "ts": r.ts.isoformat() if r.ts else None,
            }
            for r in rows
        ]


def get_ohlcv_recent(
    symbol: str,
    interval: str = "1m",
    limit: int = 1000,
) -> list[dict[str, Any]]:
    """
    Fetch the most recent `limit` OHLCV rows for symbol/interval, oldest-first.

    Unlike get_ohlcv(..., limit=N) with ascending order (which returns the *earliest*
    N rows in range), this returns the latest N candles — correct for ML/signal lookback.
    """
    with SessionLocal() as s:
        stmt = (
            select(FeatureOHLCV)
            .where(
                FeatureOHLCV.symbol == str(symbol),
                FeatureOHLCV.interval == str(interval),
            )
            .order_by(desc(FeatureOHLCV.ts))
            .limit(int(limit))
        )
        rows = list(s.execute(stmt).scalars().all())
        rows.reverse()

        return [
            {
                "symbol": r.symbol,
                "interval": r.interval,
                "open": r.open,
                "high": r.high,
                "low": r.low,
                "close": r.close,
                "volume": r.volume,
                "ts": r.ts.isoformat() if r.ts else None,
            }
            for r in rows
        ]


def get_ohlcv(
    symbol: str,
    interval: str = "1m",
    start: Any | None = None,
    end: Any | None = None,
    limit: int = 1000,
) -> list[dict[str, Any]]:
    """
    Fetch OHLCV rows for a symbol/interval, optionally bounded by [start, end].
    Returns list ordered by ts ascending.
    """
    ts_start = _dt(start)
    ts_end = _dt(end)

    with SessionLocal() as s:
        conds = [
            FeatureOHLCV.symbol == str(symbol),
            FeatureOHLCV.interval == str(interval),
        ]
        if ts_start:
            conds.append(FeatureOHLCV.ts >= ts_start)
        if ts_end:
            conds.append(FeatureOHLCV.ts <= ts_end)

        stmt = select(FeatureOHLCV).where(and_(*conds)).order_by(asc(FeatureOHLCV.ts)).limit(int(limit))
        rows = s.execute(stmt).scalars().all()

        return [
            {
                "symbol": r.symbol,
                "interval": r.interval,
                "open": r.open,
                "high": r.high,
                "low": r.low,
                "close": r.close,
                "volume": r.volume,
                "ts": r.ts.isoformat() if r.ts else None,
            }
            for r in rows
        ]


def get_store_stats() -> dict[str, Any]:
    """Lightweight stats about the feature store."""
    with SessionLocal() as s:
        # Counts
        ticks_count = int(s.execute(select(func.count()).select_from(FeatureTick)).scalar() or 0)
        ohlcv_count = int(s.execute(select(func.count()).select_from(FeatureOHLCV)).scalar() or 0)
        # First/last timestamps (best-effort)
        first_tick = s.execute(select(FeatureTick).order_by(asc(FeatureTick.ts)).limit(1)).scalars().first()
        last_tick = s.execute(select(FeatureTick).order_by(desc(FeatureTick.ts)).limit(1)).scalars().first()
        first_ohlcv = s.execute(select(FeatureOHLCV).order_by(asc(FeatureOHLCV.ts)).limit(1)).scalars().first()
        last_ohlcv = s.execute(select(FeatureOHLCV).order_by(desc(FeatureOHLCV.ts)).limit(1)).scalars().first()

    return {
        "ticks": {
            "rows": ticks_count,
            "first_ts": first_tick.ts.isoformat() if first_tick and first_tick.ts else None,
            "last_ts": last_tick.ts.isoformat() if last_tick and last_tick.ts else None,
        },
        "ohlcv": {
            "rows": ohlcv_count,
            "first_ts": first_ohlcv.ts.isoformat() if first_ohlcv and first_ohlcv.ts else None,
            "last_ts": last_ohlcv.ts.isoformat() if last_ohlcv and last_ohlcv.ts else None,
        },
        "db_path": DB_PATH,
    }


# -------------------------
# Retention
# -------------------------


def cleanup_retention(hours_ticks: int = 24, days_ohlcv: int = 30) -> dict[str, int]:
    """
    Remove old rows by timestamp.
    - ticks older than `hours_ticks`
    - candles older than `days_ohlcv`
    Returns dict with number of deleted rows per table.
    """
    deleted_ticks = 0
    deleted_ohlcv = 0
    now = datetime.now(timezone.utc)

    with SessionLocal() as s:
        # Ticks retention
        if hours_ticks is not None and hours_ticks >= 0:
            cutoff_ticks = now - timedelta(hours=int(hours_ticks))
            try:
                res = s.execute(delete(FeatureTick).where(FeatureTick.ts < cutoff_ticks))
                deleted_ticks = int(res.rowcount or 0)
            except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError):
                # Silent fallback: do nothing if backend doesn't support rowcount (older SQLite)
                deleted_ticks = 0

        # OHLCV retention
        if days_ohlcv is not None and days_ohlcv >= 0:
            cutoff_ohlcv = now - timedelta(days=int(days_ohlcv))
            try:
                res = s.execute(delete(FeatureOHLCV).where(FeatureOHLCV.ts < cutoff_ohlcv))
                deleted_ohlcv = int(res.rowcount or 0)
            except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError):
                deleted_ohlcv = 0

        s.commit()

    return {"ticks_deleted": deleted_ticks, "ohlcv_deleted": deleted_ohlcv}


__all__ = [
    "DB_PATH",
    "FeatureOHLCV",
    "FeatureTick",
    "cleanup_retention",
    "get_latest_tick",
    "get_ohlcv",
    "get_ohlcv_recent",
    "get_recent_ticks",
    "get_store_stats",
    "init_feature_store",
    "insert_ohlcv",
    "insert_ohlcv_bulk",
    "insert_tick",
    "insert_ticks_bulk",
]
