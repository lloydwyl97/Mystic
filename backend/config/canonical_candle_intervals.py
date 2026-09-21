"""Canonical candle intervals exposed by Mystic charts, APIs, and storage.

Binance.US is the market-data authority. Every interval listed here must have
a collector path, durable SQLite rows, Redis cache, and dashboard/API delivery.
3m is included even though it is not a DAY trade-permission timeframe.
"""

from __future__ import annotations

from typing import Final

# Ordered: finest to coarsest. Do not silently substitute one for another.
CANONICAL_CANDLE_INTERVALS: Final[tuple[str, ...]] = (
    "1m",
    "3m",
    "5m",
    "15m",
    "30m",
    "1h",
    "2h",
    "4h",
    "6h",
    "8h",
    "12h",
    "1d",
    "1w",
)

INTERVAL_MS: Final[dict[str, int]] = {
    "1m": 60_000,
    "3m": 180_000,
    "5m": 300_000,
    "15m": 900_000,
    "30m": 1_800_000,
    "1h": 3_600_000,
    "2h": 7_200_000,
    "4h": 14_400_000,
    "6h": 21_600_000,
    "8h": 28_800_000,
    "12h": 43_200_000,
    "1d": 86_400_000,
    "1w": 604_800_000,
}

# Freshness: a stream is stale when source age exceeds this multiple of its interval.
STALE_AGE_MULT: Final[float] = 2.5
MIN_STALE_SEC: Final[dict[str, int]] = {
    "1m": 90,
    "3m": 240,
    "5m": 400,
    "15m": 1_200,
    "30m": 2_400,
    "1h": 4_800,
    "2h": 9_000,
    "4h": 18_000,
    "6h": 27_000,
    "8h": 36_000,
    "12h": 54_000,
    "1d": 108_000,
    "1w": 700_000,
}

CANONICAL_SYMBOLS: Final[tuple[str, ...]] = ("BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT")

# Redis live window depth (completed bars). Higher TFs keep more calendar coverage.
REDIS_WINDOW: Final[dict[str, int]] = {
    "1m": 1440,
    "3m": 960,
    "5m": 864,
    "15m": 672,
    "30m": 672,
    "1h": 720,
    "2h": 360,
    "4h": 360,
    "6h": 240,
    "8h": 240,
    "12h": 240,
    "1d": 400,
    "1w": 120,
}

# Refresh cadence for the live collector (seconds). Interval-aware, never 1m-timeout on 4h.
REFRESH_SEC: Final[dict[str, int]] = {
    "1m": 15,
    "3m": 30,
    "5m": 30,
    "15m": 60,
    "30m": 90,
    "1h": 120,
    "2h": 180,
    "4h": 300,
    "6h": 400,
    "8h": 400,
    "12h": 600,
    "1d": 900,
    "1w": 1800,
}

TELEMETRY_ONLY_NO_TRADE_AUTHORITY: Final[str] = "TELEMETRY_ONLY_NO_TRADE_AUTHORITY"


def interval_ms(interval: str) -> int:
    key = (interval or "").strip().lower()
    if key not in INTERVAL_MS:
        msg = f"unsupported candle interval: {interval}"
        raise ValueError(msg)
    return INTERVAL_MS[key]


def interval_sec(interval: str) -> int:
    return interval_ms(interval) // 1000


def stale_after_sec(interval: str) -> int:
    key = (interval or "").strip().lower()
    base = interval_sec(key)
    return max(MIN_STALE_SEC.get(key, base * 2), int(base * STALE_AGE_MULT))


def redis_window(interval: str) -> int:
    return REDIS_WINDOW.get((interval or "").strip().lower(), 300)


def refresh_sec(interval: str) -> int:
    return REFRESH_SEC.get((interval or "").strip().lower(), 120)


def align_open_ms(ts_ms: int, interval: str) -> int:
    width = interval_ms(interval)
    ts = int(ts_ms)
    if interval == "1w":
        # Binance.US weekly klines open Monday 00:00 UTC. Unix epoch is Thursday.
        monday0 = -3 * 86_400_000
        return monday0 + ((ts - monday0) // width) * width
    return (ts // width) * width


def is_supported_interval(interval: str) -> bool:
    return (interval or "").strip().lower() in INTERVAL_MS


__all__ = [
    "CANONICAL_CANDLE_INTERVALS",
    "CANONICAL_SYMBOLS",
    "INTERVAL_MS",
    "TELEMETRY_ONLY_NO_TRADE_AUTHORITY",
    "align_open_ms",
    "interval_ms",
    "interval_sec",
    "is_supported_interval",
    "redis_window",
    "refresh_sec",
    "stale_after_sec",
]
