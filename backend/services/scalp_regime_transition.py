"""SCALP micro-regime transition scores (rank nudges only)."""

from __future__ import annotations

import math
from typing import Any


def _f(dd: dict[str, Any], key: str, default: float = 0.0) -> float:
    try:
        v = float(dd.get(key))
        return v if math.isfinite(v) else default
    except (TypeError, ValueError):
        return default


def _c01(v: float) -> float:
    return max(0.0, min(1.0, v))


def compute_scalp_regime_transition_scores(data: dict[str, Any], memory: dict[str, Any] | None) -> dict[str, float]:
    dd = data or {}
    mem = memory or {}
    prev = str(mem.get("previous_micro_regime") or dd.get("previous_micro_regime") or "")
    cur = str(mem.get("current_micro_regime") or dd.get("micro_regime") or "")
    chop = prev in ("chop", "low_vol_dead", "range") or cur in ("chop", "low_vol_dead", "range")
    breakout = cur in ("vol_expansion", "high_vol_breakout", "pump_continuation")
    failed = _f(dd, "kline_range_position") < 0.15 and _f(dd, "mid_change_15s") > 0
    compression = _f(dd, "kline_atr_pct") < 0.004 and _f(dd, "realized_volatility_pct") < 0.005
    expansion = _f(dd, "realized_volatility_pct") > 0.008
    exhaustion = _f(dd, "kline_rsi_proxy") > 75 and _f(dd, "mid_change_15s") < 0
    sweep = _f(dd, "kline_range_position") < 0.1 and _f(dd, "kline_volume_ratio") > 1.5
    spread_wide = _f(dd, "spread_pct") > 0.003

    return {
        "micro_trend_to_chop_score": round(_c01(1.0 if chop and prev != cur else 0.2), 4),
        "micro_range_to_breakout_score": round(_c01(1.0 if breakout and prev in ("range", "vol_crush") else 0.0), 4),
        "micro_breakout_failure_score": round(_c01(1.0 if failed else 0.0), 4),
        "micro_compression_expansion_score": round(_c01(expansion - (0.5 if compression else 0.0)), 4),
        "micro_exhaustion_reversal_score": round(_c01(1.0 if exhaustion else 0.0), 4),
        "liquidity_sweep_score": round(_c01(1.0 if sweep else 0.0), 4),
        "spread_widening_risk_score": round(_c01(spread_wide), 4),
        "scalp_regime_transition_score": round(
            _c01(
                0.2 * (1.0 if breakout else 0.0)
                + 0.2 * (1.0 if sweep else 0.0)
                - 0.15 * spread_wide
                - 0.15 * (1.0 if chop else 0.0)
            ),
            4,
        ),
    }


def regime_transition_rank_delta(transition: dict[str, float], setup: str) -> float:
    sc = float(transition.get("scalp_regime_transition_score") or 0.0)
    adj = (sc - 0.5) * 0.08
    if setup in ("MICRO_BREAKOUT", "MOMENTUM_BURST") and sc > 0.6:
        adj += 0.01
    return round(max(-0.04, min(0.04, adj)), 4)


__all__ = [
    "compute_scalp_regime_transition_scores",
    "regime_transition_rank_delta",
]
