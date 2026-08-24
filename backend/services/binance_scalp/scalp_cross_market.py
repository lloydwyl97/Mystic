"""Optional cross-market ranking intel. Fail-open if a venue is unavailable."""

from __future__ import annotations

import logging
import time
from typing import Any

logger = logging.getLogger(__name__)

_CACHE: dict[str, tuple[float, dict[str, Any]]] = {}
_TTL = 15.0


def cross_market_features(symbol: str, *, own_mid: float = 0.0) -> dict[str, Any]:
    """Coinbase dislocation + optional perp basis. Never raises, never gates."""
    now = time.time()
    cached = _CACHE.get(symbol)
    if cached and cached[0] > now:
        return dict(cached[1])
    out: dict[str, Any] = {
        "cross_venue_available": False,
        "cross_venue_dislocation_bps": 0.0,
        "spot_perp_available": False,
        "spot_perp_basis_bps": 0.0,
        "cross_market_stale": True,
    }
    if own_mid > 0:
        try:
            from backend.services.cross_exchange_reference import cross_exchange_snapshot

            snap = cross_exchange_snapshot(symbol if symbol.endswith("USDT") else f"{symbol}USDT", own_price=own_mid)
            if snap.get("available"):
                out["cross_venue_available"] = True
                out["cross_venue_dislocation_bps"] = round(float(snap.get("dislocation_pct") or 0.0) * 10_000.0, 4)
                out["cross_market_stale"] = False
        except Exception:
            pass
        try:
            from backend.derivatives_monitor import fetch_binance_funding_and_basis

            px = fetch_binance_funding_and_basis(symbol if str(symbol).endswith("USDT") else f"{symbol}USDT")
            if px and px.get("basis_pct") is not None:
                out["spot_perp_available"] = True
                out["spot_perp_basis_bps"] = round(float(px["basis_pct"]) * 10_000.0, 4)
                out["cross_market_stale"] = False
        except Exception:
            pass
    _CACHE[symbol] = (now + _TTL, out)
    return dict(out)


def reset_cross_market_cache() -> None:
    _CACHE.clear()


__all__ = ["cross_market_features", "reset_cross_market_cache"]
