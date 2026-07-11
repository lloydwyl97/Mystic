"""Read Binance.US market state — DAY orderbook feed read-only + public depth."""

from __future__ import annotations

import json
import os
import socket
from dataclasses import dataclass

import redis
from backend.services.binance_scalp.config import ScalpConfig, get_scalp_config

# Force IPv4 for Binance.US REST on this VM — shared bootstrap patch.
from backend.utils.network_ipv4 import ensure_ipv4_only

ensure_ipv4_only()


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
        try:
            bids, asks = fetch_depth_sync(bus)
        except Exception:
            return None
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
            book_source="binance_us_public_depth_readonly",
            orderbook_age_sec=0.0,
        )
