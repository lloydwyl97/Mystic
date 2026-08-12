"""
Cross-exchange informational reference layer (item p20).

Public-feed-only comparison against Coinbase Exchange's public REST API,
used purely as an informational ranking/context input: price dislocation
between the execution venue (Binance US) and a second independent public
venue, plus a coarse 24h volume-ratio comparison.

Explicitly NOT an execution-venue change: Mystic still executes exclusively
on the configured EXCHANGE_ID. This module never places, routes, or informs
order placement on any exchange — it only reads public market data from a
second venue as an additional, additive ranking signal (informational
agreement/disagreement between two independent public price feeds is useful
context; it is never a hard gate).
"""

from __future__ import annotations

import logging
import os
import time
from typing import Any

import httpx

logger = logging.getLogger(__name__)

_COINBASE_PRODUCT_OVERRIDES: dict[str, str] = {
    "BTCUSDT": "BTC-USD",
    "ETHUSDT": "ETH-USD",
    "SOLUSDT": "SOL-USD",
    "XRPUSDT": "XRP-USD",
}


def _reference_feed_enabled() -> bool:
    return os.getenv("CROSS_EXCHANGE_REFERENCE_FEED_ENABLED", "1").strip().lower() not in {"0", "false", "no"}


def _coinbase_product_id(symbol: str) -> str:
    normalized = symbol.upper().replace("/", "").replace("-", "")
    if normalized in _COINBASE_PRODUCT_OVERRIDES:
        return _COINBASE_PRODUCT_OVERRIDES[normalized]
    if normalized.endswith("USDT"):
        return f"{normalized[:-4]}-USD"
    return normalized


def _http_get(url: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
    try:
        resp = httpx.get(url, params=params or {}, headers={"User-Agent": "mystic-trading/1.0"}, timeout=8.0)
        resp.raise_for_status()
        data = resp.json()
        return data if isinstance(data, dict) else {}
    except httpx.HTTPStatusError as e:
        logger.debug("cross_exchange HTTP %s for %s", e.response.status_code, url)
        return {}
    except httpx.RequestError as e:
        logger.debug("cross_exchange request failed for %s: %s", url, e)
        return {}
    except ValueError as e:
        logger.debug("cross_exchange JSON decode failed for %s: %s", url, e)
        return {}


def fetch_coinbase_ticker(symbol: str) -> dict[str, Any]:
    """Public Coinbase Exchange ticker (no auth). Returns {} on any failure
    or unsupported product — callers must treat that as an honest degraded
    state, never as a neutral price read."""
    if not _reference_feed_enabled():
        return {}
    product_id = _coinbase_product_id(symbol)
    url = f"https://api.exchange.coinbase.com/products/{product_id}/ticker"
    data = _http_get(url)
    if not data:
        return {}
    try:
        price = float(data.get("price", 0.0))
        volume_24h = float(data.get("volume", 0.0))
    except (TypeError, ValueError):
        return {}
    if price <= 0.0:
        return {}
    return {"symbol": symbol, "product_id": product_id, "price": price, "volume_24h": volume_24h}


_CROSS_EX_CACHE: dict[str, tuple[float, dict[str, Any]]] = {}
_CROSS_EX_CACHE_TTL_SEC = 30.0


def cross_exchange_snapshot(
    symbol: str,
    *,
    own_price: float,
    own_volume_24h: float = 0.0,
) -> dict[str, Any]:
    """Compare the execution venue's own price/volume against Coinbase's
    public feed for the same symbol.

    dislocation_pct = (own_price - coinbase_price) / coinbase_price — the
    execution venue's price premium/discount vs an independent public
    reference. volume_ratio_vs_coinbase is a coarse 24h-level comparison
    (not real-time burst detection — that is covered by the same-symbol
    RVOL feature in day_feature_stack_v2.py). Always informational; never a
    gate, never routes execution to a different venue.
    """
    now = time.time()
    cache_key = symbol
    cached = _CROSS_EX_CACHE.get(cache_key)
    if cached is not None and (now - cached[0]) < _CROSS_EX_CACHE_TTL_SEC:
        result = dict(cached[1])
        result["own_price"] = own_price
        if result.get("available") and result.get("coinbase_price", 0.0) > 0:
            result["dislocation_pct"] = (own_price - result["coinbase_price"]) / result["coinbase_price"]
        return result

    if not _reference_feed_enabled():
        result = {"available": False, "degraded_reason": "reference_feed_disabled", "own_price": own_price}
        _CROSS_EX_CACHE[cache_key] = (now, result)
        return result

    cb = fetch_coinbase_ticker(symbol)
    if not cb:
        result = {"available": False, "degraded_reason": "coinbase_unreachable_or_unsupported", "own_price": own_price}
        _CROSS_EX_CACHE[cache_key] = (now, result)
        return result

    coinbase_price = float(cb["price"])
    coinbase_volume = float(cb.get("volume_24h", 0.0))
    dislocation_pct = (own_price - coinbase_price) / coinbase_price if coinbase_price > 0 else 0.0
    volume_ratio = (own_volume_24h / coinbase_volume) if coinbase_volume > 0 else 0.0

    result = {
        "available": True,
        "coinbase_price": coinbase_price,
        "own_price": own_price,
        "dislocation_pct": dislocation_pct,
        "coinbase_volume_24h": coinbase_volume,
        "volume_ratio_vs_coinbase": volume_ratio,
    }
    _CROSS_EX_CACHE[cache_key] = (now, result)
    return result


_DISLOCATION_FULL_SIGNAL_PCT = 0.002  # 0.20% dislocation vs Coinbase = full +/-1 signal


def cross_exchange_dislocation_signal(snapshot: dict[str, Any]) -> float:
    """Item p20 ranking promotion: small mean-reversion tilt from the
    execution venue trading rich/cheap vs an independent public reference
    (Coinbase) — bounded [-1, 1], zero on any degraded/unavailable state.
    Purely informational; never changes the execution venue or gates a trade."""
    if not snapshot or not snapshot.get("available"):
        return 0.0
    dislocation = snapshot.get("dislocation_pct")
    if dislocation is None:
        return 0.0
    return max(-1.0, min(1.0, -(float(dislocation) / _DISLOCATION_FULL_SIGNAL_PCT)))


__all__ = [
    "cross_exchange_dislocation_signal",
    "cross_exchange_snapshot",
    "fetch_coinbase_ticker",
]
