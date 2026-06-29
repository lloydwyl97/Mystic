"""Regime transition intelligence scores for DAY ranking (soft only)."""

from __future__ import annotations

import math
from typing import Any


def _f(dd: dict[str, Any], key: str, default: float = 0.0) -> float:
    raw = dd.get(key)
    try:
        v = float(raw)
        return v if math.isfinite(v) else default
    except (TypeError, ValueError):
        return default


def _clamp01(v: float) -> float:
    return max(0.0, min(1.0, v))


def compute_regime_transition_scores(decision_data: dict[str, Any], memory: dict[str, Any] | None = None) -> dict[str, float]:
    dd = decision_data or {}
    mem = memory or {}
    adx = _f(dd, "adx", 25.0)
    chop = _f(dd, "chop_score", 0.5)
    rsi = _f(dd, "rsi", 50.0)
    mom = _f(dd, "price_momentum", 0.0)
    vol = _f(dd, "atr", 0.0) / max(_f(dd, "current_price", 1.0), 1e-9)
    prev_regime = str(mem.get("previous_regime") or dd.get("previous_regime") or "")
    cur_regime = str(mem.get("current_regime") or dd.get("day_route_regime") or dd.get("regime") or "neutral")
    trans = _f(mem, "regime_transition_score", 0.0)

    trend_to_chop = _clamp01((40.0 - adx) / 40.0 * chop)
    range_to_breakout = _clamp01((adx - 20.0) / 30.0 * _clamp01((mom + 1.0) / 3.0))
    bear_exhaustion = _clamp01((rsi - 28.0) / 25.0 * _clamp01((mom + 0.5) / 2.0))
    compression_expansion = _clamp01(abs(vol - 0.015) / 0.02)
    bull_pullback_recovery = _clamp01(_f(dd, "ema_alignment", 0.5) * _clamp01((45.0 - abs(rsi - 45.0)) / 45.0))
    panic_dump = _clamp01((25.0 - rsi) / 25.0 * _clamp01(-mom / 3.0))
    liquidity_sweep = _clamp01(abs(_f(dd, "ctx_depth_imbalance", 0.0)) * _clamp01(vol / 0.03))

    if prev_regime and prev_regime != cur_regime:
        trans = max(trans, 0.35)

    return {
        "trend_to_chop_score": round(trend_to_chop, 4),
        "range_to_breakout_score": round(range_to_breakout, 4),
        "bear_exhaustion_score": round(bear_exhaustion, 4),
        "compression_expansion_score": round(compression_expansion, 4),
        "bull_pullback_recovery_score": round(bull_pullback_recovery, 4),
        "panic_dump_score": round(panic_dump, 4),
        "liquidity_sweep_score": round(liquidity_sweep, 4),
        "regime_transition_score": round(_clamp01(trans), 4),
    }


def regime_transition_rank_delta(scores: dict[str, float], setup: str) -> float:
    setup_u = str(setup or "").upper()
    s = scores or {}
    if setup_u == "HTF_TREND_PULLBACK":
        boost = s.get("bull_pullback_recovery_score", 0.0)
    elif setup_u == "BREAKOUT_CONTINUATION":
        boost = s.get("range_to_breakout_score", 0.0)
    elif setup_u == "FAILED_BREAKDOWN_REVERSAL":
        boost = s.get("bear_exhaustion_score", 0.0)
    elif setup_u == "RANGE_BOUNCE":
        boost = s.get("trend_to_chop_score", 0.0)
    else:
        boost = s.get("regime_transition_score", 0.0)
    return round(max(-0.04, min(0.04, (float(boost) - 0.5) * 0.10)), 4)


__all__ = ["compute_regime_transition_scores", "regime_transition_rank_delta"]
