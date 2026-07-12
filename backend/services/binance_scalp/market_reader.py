"""Read Binance.US market state — DAY orderbook feed read-only + public depth."""

from __future__ import annotations

import json
import os
import socket
import time
from dataclasses import dataclass

import redis
from backend.services.binance_scalp.config import ScalpConfig, get_scalp_config

# Force IPv4 for Binance.US REST on this VM — shared bootstrap patch.
from backend.utils.network_ipv4 import ensure_ipv4_only

ensure_ipv4_only()

# Short-lived, cross-process depth cache (Redis-backed, same pattern as
# day_active_bundle.py / orderbook:{BASE} elsewhere in this codebase).
#
# /api/v3/depth was previously fetched fresh on *every* read() call from *every*
# caller (paper_engine tick, strategy router evaluate_symbol + evaluate_all,
# position lifecycle exit checks, dashboard status_snapshot, diagnostics) with
# zero caching and zero weight-limiter visibility — confirmed via Binance's own
# X-MBX-USED-WEIGHT-1M header bursting to 235-386/min per host. This TTL keeps
# depth data effectively real-time (far fresher than the 5s SCALP tick cadence)
# while eliminating redundant duplicate Binance calls within the same window.
# Does not change any entry/exit/ranking/scoring/sizing/learning logic.
SCALP_DEPTH_CACHE_TTL_SEC: float = float(os.getenv("SCALP_DEPTH_CACHE_TTL_SEC", "2.5"))
_DEPTH_CACHE_KEY_PREFIX = "scalp:depth_cache:"
_DEPTH_ENDPOINT_WEIGHT = 5  # Binance.US weight for GET /api/v3/depth?limit=100


@dataclass(frozen=True)
class MarketSnapshot:
    symbol: str
    symbol_bus: str
    best_bid: float
    best_ask: float
    mid: float
    spread_pct: float
    bids: list[list[float]]
    asks: list[list[float]]
    redis_spread_pct: float | None
    order_book_imbalance: float | None
    book_source: str
    orderbook_age_sec: float


def symbol_bus(symbol: str) -> str:
    s = symbol.strip().upper().replace("/", "")
    if not s.endswith("USDT"):
        s = f"{s}USDT"
    return s


def symbol_base(symbol_bus: str) -> str:
    s = symbol_bus.strip().upper().replace("/", "")
    return s[:-4] if s.endswith("USDT") else s


def fetch_depth_sync(symbol_bus: str, *, limit: int = 100) -> tuple[list[list[float]], list[list[float]]]:
    sym = symbol_bus.strip().upper()
    base_url = os.getenv("BINANCE_US_REST_BASE", "https://api.binance.us")
    url = f"{base_url.rstrip('/')}/api/v3/depth?symbol={sym}&limit={int(limit)}"
    try:
        import httpx

        with httpx.Client(timeout=15.0) as client:
            resp = client.get(url)
            resp.raise_for_status()
            data = resp.json()
    except Exception as exc:
        raise RuntimeError(f"depth fetch failed for {sym}: {exc}") from exc
    bids = [[float(p), float(q)] for p, q in data.get("bids") or []]
    asks = [[float(p), float(q)] for p, q in data.get("asks") or []]
    return bids, asks


def _read_depth_cache(r: redis.Redis, sym: str) -> tuple[list[list[float]], list[list[float]], float] | None:
    """Cross-process cache read. Returns (bids, asks, age_sec) or None on miss/stale/error."""
    try:
        raw = r.get(f"{_DEPTH_CACHE_KEY_PREFIX}{sym}")
        if not raw:
            return None
        payload = json.loads(raw)
        fetched_at = float(payload.get("fetched_at") or 0)
        age = time.time() - fetched_at
        if age > SCALP_DEPTH_CACHE_TTL_SEC or age < 0:
            return None
        bids = payload.get("bids") or []
        asks = payload.get("asks") or []
        if not bids or not asks:
            return None
        return bids, asks, age
    except Exception:
        return None


def _write_depth_cache(r: redis.Redis, sym: str, bids: list[list[float]], asks: list[list[float]]) -> None:
    try:
        payload = json.dumps({"fetched_at": time.time(), "bids": bids, "asks": asks})
        r.set(f"{_DEPTH_CACHE_KEY_PREFIX}{sym}", payload, ex=max(5, int(SCALP_DEPTH_CACHE_TTL_SEC * 2)))
    except Exception:
        pass


def _record_depth_weight_usage(r: redis.Redis) -> None:
    """Mirror BinanceWeightLimiter's usage counters for visibility (bwl:usage:*, bwl:req:*).

    Best-effort only — never gates/blocks a request; this is metrics, not enforcement.
    Existing async limiter behavior for other endpoints is unchanged.
    """
    try:
        r.incrby("bwl:usage:/api/v3/depth", _DEPTH_ENDPOINT_WEIGHT)
        r.incr("bwl:req:/api/v3/depth")
    except Exception:
        pass


class ScalpMarketReader:
    """Read-only: Redis orderbook:{BASE} features + public REST depth for walks."""

    def __init__(self, config: ScalpConfig | None = None) -> None:
        self.config = config or get_scalp_config()
        self._redis = redis.from_url(self.config.redis_url, decode_responses=True)

    def _read_redis_features(self, base: str) -> tuple[float | None, float | None]:
        key = f"orderbook:{base}"
        spread_raw = self._redis.hget(key, "bid_ask_spread")
        imb_raw = self._redis.hget(key, "order_book_imbalance")
        spread: float | None = None
        imb: float | None = None
        try:
            if spread_raw is not None:
                spread = float(spread_raw)
        except (TypeError, ValueError):
            pass
        try:
            if imb_raw is not None:
                imb = float(imb_raw)
        except (TypeError, ValueError):
            pass
        return spread, imb

    def read(self, symbol: str) -> MarketSnapshot | None:
        bus = symbol_bus(symbol)
        base = symbol_base(bus)
        redis_spread, imbalance = self._read_redis_features(base)

        cached = _read_depth_cache(self._redis, bus)
        if cached is not None:
            bids, asks, age_sec = cached
            book_source = "binance_us_public_depth_readonly_cached"
        else:
            try:
                bids, asks = fetch_depth_sync(bus)
            except Exception:
                return None
            age_sec = 0.0
            book_source = "binance_us_public_depth_readonly"
            if bids and asks:
                _write_depth_cache(self._redis, bus, bids, asks)
                _record_depth_weight_usage(self._redis)

        if not bids or not asks:
            return None
        best_bid = float(bids[0][0])
        best_ask = float(asks[0][0])
        if best_bid <= 0 or best_ask <= 0 or best_ask < best_bid:
            return None
        mid = (best_bid + best_ask) / 2.0
        spread_pct = (best_ask - best_bid) / mid if mid > 0 else 1.0
        return MarketSnapshot(
            symbol=bus,
            symbol_bus=bus,
            best_bid=best_bid,
            best_ask=best_ask,
            mid=mid,
            spread_pct=spread_pct,
            bids=bids,
            asks=asks,
            redis_spread_pct=redis_spread,
            order_book_imbalance=imbalance,
            book_source=book_source,
            orderbook_age_sec=age_sec,
        )
