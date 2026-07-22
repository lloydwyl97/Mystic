"""Lightweight kline cache for paper scalp strategies."""

from __future__ import annotations

import json
import logging
import os
import time
from datetime import datetime, timedelta, timezone

logger = logging.getLogger(__name__)

# Force IPv4 for Binance.US REST — shared bootstrap patch.
from backend.utils.network_ipv4 import ensure_ipv4_only

ensure_ipv4_only()

# Regime classifier needs >=31 completed 1h bars (~31h of history).
MIN_REGIME_1H_BARS = 31
DEFAULT_1H_LOOKBACK_MINUTES = 2880  # 48h -> ~48 bars

# Every strategy in strategies/*.py reads bars_1m through fixed tail slices
# (bars[-N:], range(-period, 0), etc.) or through common.py's _atr_pct — none
# scan the full list length-dependently — so widening this window is pure
# extra headroom, not a behavior change for any existing setup. Previously
# capped at 30 bars/30 min, which left no room for anything needing to look
# back further (e.g. candle-shape pattern matching in the spirit of the DAY
# AI pattern memory work). Configurable so it can be tuned without a redeploy.
# Default 120m (~120 1m bars). Strategies still slice bars[-N:] so widening is
# headroom for MTF confirmation / future candle-shape matching, not a silent
# behavior change for existing setup window logic.
DEFAULT_1M_LOOKBACK_MINUTES = int(os.getenv("SCALP_1M_BARS_LOOKBACK_MINUTES", "120"))

# Cross-process Redis cache for fetch_bars(), keyed by (symbol, interval, minutes).
# KlineCache's own in-process TTL cache only helps a single long-lived instance
# (e.g. paper_engine's shared instance); callers that construct a *fresh*
# KlineCache per invocation (e.g. status_snapshot.py on every rebuild) or run
# in a different process previously got zero cache benefit and refetched from
# Binance every time. This mirrors the same short-TTL Redis-backed pattern
# already used for SCALP depth (market_reader.py) and the DAY bundle.
_BARS_CACHE_KEY_PREFIX = "scalp:bars_cache:"
_BARS_CACHE_TTL_SEC: dict[str, float] = {
    "1m": float(os.getenv("SCALP_BARS_CACHE_TTL_1M_SEC", "20")),
    "5m": float(os.getenv("SCALP_BARS_CACHE_TTL_5M_SEC", "45")),
    "15m": float(os.getenv("SCALP_BARS_CACHE_TTL_15M_SEC", "45")),
    "1h": float(os.getenv("SCALP_BARS_CACHE_TTL_1H_SEC", "300")),
}


def _bars_cache_ttl(interval: str) -> float:
    return _BARS_CACHE_TTL_SEC.get(interval, 30.0)


def _bars_cache_key(symbol: str, interval: str, minutes: int) -> str:
    return f"{_BARS_CACHE_KEY_PREFIX}{symbol.upper()}:{interval}:{minutes}"


def _read_bars_cache(symbol: str, interval: str, minutes: int) -> list[dict] | None:
    try:
        from backend.config.redis_config import get_shared_redis_sync

        r = get_shared_redis_sync()
        if not r:
            return None
        raw = r.get(_bars_cache_key(symbol, interval, minutes))
        if not raw:
            return None
        payload = json.loads(raw)
        fetched_at = float(payload.get("fetched_at") or 0)
        if time.time() - fetched_at > _bars_cache_ttl(interval):
            return None
        bars = payload.get("bars")
        return bars if isinstance(bars, list) else None
    except Exception:
        return None


def _write_bars_cache(symbol: str, interval: str, minutes: int, bars: list[dict]) -> None:
    try:
        from backend.config.redis_config import get_shared_redis_sync

        r = get_shared_redis_sync()
        if not r:
            return
        payload = json.dumps({"fetched_at": time.time(), "bars": bars})
        r.set(_bars_cache_key(symbol, interval, minutes), payload, ex=max(30, int(_bars_cache_ttl(interval) * 2)))
    except Exception:
        pass


def fetch_bars(symbol: str, interval: str, *, minutes: int = 30) -> list[dict]:
    """Fetch OHLCV bars for symbol/interval from Binance.US public REST.

    Cross-process cached with a short TTL (see ``_BARS_CACHE_TTL_SEC``) so
    repeated callers — including a freshly-constructed ``KlineCache`` in a
    different process — reuse the same recent fetch instead of hitting
    Binance again for data that hasn't meaningfully changed.
    """
    cached = _read_bars_cache(symbol, interval, minutes)
    if cached is not None:
        return cached
    bars = _fetch_bars_live(symbol, interval, minutes=minutes)
    if bars:
        _write_bars_cache(symbol, interval, minutes, bars)
    return bars


def _fetch_bars_live(symbol: str, interval: str, *, minutes: int = 30) -> list[dict]:
    """Live Binance.US REST fetch — no caching (internal; use fetch_bars())."""
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
    except Exception as e:
        logger.warning("[KLINE_CACHE] Failed to fetch bars for %s/%s: %s", symbol, interval, e)
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

    def get(self, symbol: str, *, minutes: int = DEFAULT_1M_LOOKBACK_MINUTES) -> list[dict]:
        return self._get_interval(symbol, "1m", minutes=minutes)

    def get_5m(self, symbol: str, *, minutes: int = 240) -> list[dict]:
        return self._get_interval(symbol, "5m", minutes=minutes)

    def get_15m(self, symbol: str, *, minutes: int = 720) -> list[dict]:
        return self._get_interval(symbol, "15m", minutes=minutes)

    def get_1h(self, symbol: str, *, minutes: int = DEFAULT_1H_LOOKBACK_MINUTES) -> list[dict]:
        return self._get_interval(symbol, "1h", minutes=minutes, ttl_sec=self._ttl_1h)
