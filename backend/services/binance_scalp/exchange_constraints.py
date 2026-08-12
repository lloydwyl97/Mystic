"""
Exchange constraint handler for SCALP live orders.
Fetches and caches lot size / min notional from Binance.US /exchangeInfo.
"""

from __future__ import annotations

import logging
import time
from typing import Dict

import requests

logger = logging.getLogger(__name__)

_CACHE: Dict[str, dict] = {}
_CACHE_TS: float = 0.0
_CACHE_TTL: float = 3600.0  # 1 hour

BINANCE_US_BASE = "https://api.binance.us"

_FALLBACK = {
    "min_qty": 0.00001,
    "max_qty": 9_999_999.0,
    "step_size": 0.00001,
    "min_notional": 10.0,
}


def get_symbol_constraints(symbol: str) -> dict:
    """Return lot-size and min-notional constraints for a symbol. Results are cached.

    Includes an explicit ``is_fallback`` flag (item p25): when the real
    /exchangeInfo fetch has never succeeded (or this symbol wasn't in the
    last successful response), the conservative generic ``_FALLBACK``
    values are used instead of real exchange filters. ``is_fallback=True``
    lets callers log/alert on this rather than silently rounding orders to
    generic values that look identical to real per-symbol filters.
    """
    global _CACHE, _CACHE_TS
    if time.time() - _CACHE_TS > _CACHE_TTL or symbol not in _CACHE:
        _refresh_cache()
    cached = _CACHE.get(symbol)
    if cached is not None:
        result = dict(cached)
        result["is_fallback"] = False
        return result
    result = dict(_FALLBACK)
    result["is_fallback"] = True
    return result


def round_qty_to_step(symbol: str, qty: float) -> float:
    """Round qty down to the exchange-mandated step size for the given symbol."""
    c = get_symbol_constraints(symbol)
    step = c.get("step_size") or 0.0
    if step <= 0:
        return round(qty, 5)
    # Floor to nearest step (never over-sell / never over-buy)
    steps = int(qty / step)
    result = steps * step
    # Re-round to eliminate floating-point dust (8 decimal places is Binance max)
    return round(result, 8)


def _refresh_cache() -> None:
    global _CACHE, _CACHE_TS
    try:
        resp = requests.get(
            f"{BINANCE_US_BASE}/api/v3/exchangeInfo",
            timeout=10,
        )
        resp.raise_for_status()
        data = resp.json()
        new_cache: Dict[str, dict] = {}
        for sym_info in data.get("symbols", []):
            s = sym_info["symbol"]
            filters: dict[str, dict] = {f["filterType"]: f for f in sym_info.get("filters", [])}
            lot = filters.get("LOT_SIZE", {})
            notional = filters.get("MIN_NOTIONAL", {}) or filters.get("NOTIONAL", {})
            new_cache[s] = {
                "min_qty": float(lot.get("minQty", _FALLBACK["min_qty"])),
                "max_qty": float(lot.get("maxQty", _FALLBACK["max_qty"])),
                "step_size": float(lot.get("stepSize", _FALLBACK["step_size"])),
                "min_notional": float(notional.get("minNotional") or notional.get("notionalMin") or _FALLBACK["min_notional"]),
            }
        _CACHE = new_cache
        _CACHE_TS = time.time()
        logger.info("[EXCHANGE_CONSTRAINTS] Cached %d symbols", len(_CACHE))
    except Exception as e:
        logger.warning("[EXCHANGE_CONSTRAINTS] Failed to refresh: %s", e)
