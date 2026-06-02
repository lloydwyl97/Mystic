"""
ai_feature_v3 — same 145-dim geometry as v2, but **callers MUST pass primary-clock OHLCV**
(5m day / 15m day) into ``ohlcv=`` for ``build_feature_vector_124``. Execution / microstructure
(1m) is optional via ``ohlcv_exec_1m`` for overlays that must not become the primary signal mind.

Training and live inference both use this module when ``feature_version >= 3``.
"""

from __future__ import annotations

from typing import Any

from backend.services.ai_decision_contract import AI_FEATURE_DIM_V2
from backend.services.ai_feature_v2 import build_feature_vector_v2


def build_feature_vector_v3(
    *,
    symbol_ccxt: str,
    ohlcv_primary: list[list],
    volume_profile: dict[str, Any] | None,
    orderbook: dict[str, Any] | None,
    ohlcv_1d: list[list] | None = None,
    sentiment: dict[str, Any] | None = None,
    ai_context: dict[str, Any] | None = None,
    ai_context_mtf: dict[str, Any] | None = None,
    ohlcv_exec_1m: list[list] | None = None,
) -> list[float]:
    """
    ``ohlcv_primary`` — 5m (day) or 15m (day) aggregated series (oldest-first).
    ``ohlcv_exec_1m`` — optional tail of 1m bars for execution-only consumers inside fundamentals;
      the 124-dim technical core is still computed from ``ohlcv_primary`` only.
    """
    _ = ohlcv_exec_1m  # reserved for future 1m-only micro overlays; primary mind stays on ohlcv_primary
    out = build_feature_vector_v2(
        symbol_ccxt=symbol_ccxt,
        ohlcv=ohlcv_primary,
        volume_profile=volume_profile,
        orderbook=orderbook,
        ohlcv_1d=ohlcv_1d,
        sentiment=sentiment,
        ai_context=ai_context,
        ai_context_mtf=ai_context_mtf,
    )
    if len(out) != AI_FEATURE_DIM_V2:
        raise ValueError(f"build_feature_vector_v3 produced {len(out)} dims, expected {AI_FEATURE_DIM_V2}")
    return out


__all__ = ["build_feature_vector_v3"]
