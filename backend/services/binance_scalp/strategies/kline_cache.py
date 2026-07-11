"""Lightweight kline cache for paper scalp strategies."""

from __future__ import annotations

import json
import time
from datetime import datetime, timedelta, timezone

# Force IPv4 for Binance.US REST — shared bootstrap patch.
from backend.utils.network_ipv4 import ensure_ipv4_only

ensure_ipv4_only()

# Regime classifier needs >=31 completed 1h bars (~31h of history).
MIN_REGIME_1H_BARS = 31
DEFAULT_1H_LOOKBACK_MINUTES = 2880  # 48h -> ~48 bars


def fetch_bars(symbol: str, interval: str, *, minutes: int = 30) -> list[dict]:
    """Fetch OHLCV bars for symbol/interval from Binance.US public REST."""
    if interval == "1h":
        minutes = max(int(minutes), MIN_REGIME_1H_BARS * 60)
    end = datetime.now(timezone.utc)
    start = end - timedelta(minutes=minutes)
    start_ms = int(start.timestamp() * 1000)
    end_ms = int(end.timestamp() * 1000)
    url = f"https://api.binance.us/api/v3/klines?symbol={symbol}&interval={interval}&startTime={start_ms}&endTime={end_ms}&limit=1000"
    try:
        import httpx

        with httpx.Client(timeout=20.0) as client:
            resp = client.get(url)
            resp.raise_for_status()
            rows = resp.json()
    except Exception:
        return []
    if not isinstance(rows, list):
        return []
    return [
        {
            "ts_ms": int(r[0]),
            "ts": int(r[0]) // 1000,
            "open": float(r[1]),
            "high": float(r[2]),
            "low": float(r[3]),
            "close": float(r[4]),
            "volume": float(r[5]),
        }
        for r in rows
    ]


def fetch_1m_bars(symbol: str, *, minutes: int = 30) -> list[dict]:
    return fetch_bars(symbol, "1m", minutes=minutes)


class KlineCache:
    def __init__(self, *, ttl_sec: float = 45.0, ttl_1h_sec: float = 300.0) -> None:
        self._ttl = ttl_sec
        self._ttl_1h = ttl_1h_sec
        self._cache: dict[str, tuple[float, list[dict]]] = {}

    def _get_interval(
        self,
        symbol: str,
        interval: str,
        *,
        minutes: int,
        ttl_sec: float | None = None,
    ) -> list[dict]:
        sym = symbol.upper()
        key = f"{sym}:{interval}:{minutes}"
        now = time.time()
        ttl = ttl_sec if ttl_sec is not None else self._ttl
        hit = self._cache.get(key)
        if hit and (now - hit[0]) < ttl:
            return hit[1]
        bars = fetch_bars(sym, interval, minutes=minutes)
        self._cache[key] = (now, bars)
        return bars

    def get(self, symbol: str, *, minutes: int = 30) -> list[dict]:
        return self._get_interval(symbol, "1m", minutes=minutes)

    def get_5m(self, symbol: str, *, minutes: int = 240) -> list[dict]:
        return self._get_interval(symbol, "5m", minutes=minutes)

    def get_15m(self, symbol: str, *, minutes: int = 720) -> list[dict]:
        return self._get_interval(symbol, "15m", minutes=minutes)

    def get_1h(self, symbol: str, *, minutes: int = DEFAULT_1H_LOOKBACK_MINUTES) -> list[dict]:
        return self._get_interval(symbol, "1h", minutes=minutes, ttl_sec=self._ttl_1h)
