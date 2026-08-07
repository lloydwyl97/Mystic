"""DAY higher-timeframe regime anchoring — soft rank/size demotion.

The existing day_regime_router picks a coarse regime label (bull / range /
bear / chop / neutral) and produces a hard allowed/not-allowed decision
with a coarse route_size_factor. This module adds a smooth per-candidate
"how well does this setup ride the 1h/4h current?" score in [0, 1] that
is applied as an additive rank delta and multiplicative size factor
independent of the router's coarse label.

Purpose: cure the class of trades where the router allowed the entry but
1h/4h alignment was already fighting the setup direction. Ranking + sizing
only. Never hard-blocks and never overrides the router.

Score components (weighted mean):
* h1_h4_agreement — how much 1h and 4h alignments favor the setup
* momentum         — recent price momentum sign vs setup direction
* ema_stack        — EMA alignment consistent with the setup

Feature flag: DAY_HTF_ANCHOR_ENABLED (default true).
"""

from __future__ import annotations

import os
from typing import Any

from backend.services.day_trade_thesis import (
    SETUP_BREAKOUT_CONTINUATION,
    SETUP_FAILED_BREAKDOWN_REVERSAL,
    SETUP_HTF_TREND_PULLBACK,
    SETUP_RANGE_BOUNCE,
    SETUP_VWAP_REVERSION,
    _safe_float,
    _tf_align,
    parse_mtf_json,
)

DEFAULT_WEIGHTS: dict[str, float] = {
    "h1_h4_agreement": 0.55,
    "momentum": 0.25,
    "ema_stack": 0.20,
}

RANK_DELTA_AT_ZERO = -0.10  # counter-HTF setup
RANK_DELTA_AT_ONE = 0.05    # HTF-aligned bonus
SIZE_FACTOR_AT_ZERO = 0.55
SIZE_FACTOR_AT_ONE = 1.15
SIZE_FACTOR_AT_HALF = 0.85


def htf_anchor_enabled() -> bool:
    return os.getenv("DAY_HTF_ANCHOR_ENABLED", "true").strip().lower() in ("1", "true", "yes", "on")


TREND_LONG_SETUPS = frozenset({SETUP_HTF_TREND_PULLBACK, SETUP_BREAKOUT_CONTINUATION})
RANGE_SETUPS = frozenset({SETUP_RANGE_BOUNCE, SETUP_VWAP_REVERSION})
REVERSAL_SETUPS = frozenset({SETUP_FAILED_BREAKDOWN_REVERSAL})


def _setup_family(setup: str) -> str:
    s = str(setup or "").strip()
    if s in TREND_LONG_SETUPS:
        return "trend_long"
    if s in RANGE_SETUPS:
        return "range"
    if s in REVERSAL_SETUPS:
        return "reversal"
    # Aliased legacy labels
    su = s.upper()
    if "TREND" in su or "PULLBACK" in su or "BREAKOUT" in su:
        return "trend_long"
    if "RANGE" in su or "VWAP" in su or "MEAN_REV" in su:
        return "range"
    if "REVERSAL" in su or "REVERSION" in su:
        return "reversal"
    return "unknown"


def _h1_h4_agreement_credit(h1: float | None, h4: float | None, family: str) -> tuple[float, str]:
    """Credit for higher timeframes agreeing with the setup's direction."""
    if h1 is None and h4 is None:
        return 0.5, "htf_unknown"
    h1v = float(h1) if h1 is not None else 0.5
    h4v = float(h4) if h4 is not None else h1v
    avg = (h1v + h4v) / 2.0
    if family == "trend_long":
        if avg >= 0.60:
            return 1.0, "htf_strong_bull"
        if avg >= 0.52:
            return 0.75, "htf_bull_lean"
        if avg >= 0.45:
            return 0.5, "htf_neutral"
        if avg >= 0.35:
            return 0.25, "htf_soft_bear"
        return 0.05, "htf_hard_bear"
    if family == "range":
        # Range setups prefer HTF neutrality — extremes in either direction hurt.
        deviation = abs(avg - 0.5)
        if deviation <= 0.05:
            return 1.0, "htf_range_flat"
        if deviation <= 0.10:
            return 0.75, "htf_mild_lean"
        if deviation <= 0.15:
            return 0.5, "htf_lean"
        return 0.25, "htf_strong_trend_bad_for_range"
    if family == "reversal":
        # Reversal setups thrive when HTF is bearish and something is bottoming.
        if avg <= 0.35:
            return 1.0, "htf_strong_bear_reversal_candidate"
        if avg <= 0.45:
            return 0.7, "htf_bear_lean"
        if avg <= 0.52:
            return 0.5, "htf_mixed"
        return 0.25, "htf_bull_bad_for_bear_reversal"
    return 0.5, "family_unknown"


def _momentum_credit(price_momentum: float, family: str) -> tuple[float, str]:
    """Credit for price momentum sign matching setup direction."""
    try:
        m = float(price_momentum)
    except (TypeError, ValueError):
        return 0.5, "momentum_unknown"
    if family == "trend_long":
        if m >= 0.02:
            return 1.0, "momentum_strong_up"
        if m >= 0.005:
            return 0.75, "momentum_up"
        if m >= -0.005:
            return 0.5, "momentum_flat"
        if m >= -0.02:
            return 0.25, "momentum_down"
        return 0.0, "momentum_strong_down"
    if family == "range":
        # Range setups prefer flat-ish momentum
        am = abs(m)
        if am <= 0.005:
            return 1.0, "momentum_flat_range_good"
        if am <= 0.015:
            return 0.75, "momentum_mild"
        if am <= 0.03:
            return 0.5, "momentum_moderate"
        return 0.25, "momentum_strong_trending"
    if family == "reversal":
        # Reversal wants prior down-momentum starting to turn
        if -0.02 <= m <= 0.005:
            return 1.0, "momentum_bottoming"
        if m > 0.02:
            return 0.4, "momentum_already_ripped"
        if m < -0.05:
            return 0.4, "momentum_still_falling_hard"
        return 0.6, "momentum_neutral"
    return 0.5, "family_unknown"


def _ema_stack_credit(ema_alignment: float, family: str) -> tuple[float, str]:
    try:
        e = float(ema_alignment)
    except (TypeError, ValueError):
        return 0.5, "ema_unknown"
    if family == "trend_long":
        if e >= 0.70:
            return 1.0, "ema_stacked_up"
        if e >= 0.55:
            return 0.75, "ema_leaning_up"
        if e >= 0.45:
            return 0.5, "ema_mixed"
        return 0.25, "ema_bearish_stack"
    if family == "range":
        deviation = abs(e - 0.5)
        if deviation <= 0.10:
            return 1.0, "ema_flat_range_good"
        if deviation <= 0.20:
            return 0.6, "ema_mild_lean"
        return 0.3, "ema_stacked_strong"
    if family == "reversal":
        if e <= 0.35:
            return 1.0, "ema_bearish_stack_reversal_ok"
        if e <= 0.50:
            return 0.7, "ema_slight_bear_ok"
        return 0.4, "ema_bull_bad_for_reversal"
    return 0.5, "family_unknown"


def _score_to_rank_delta(score: float) -> float:
    s = max(0.0, min(1.0, float(score)))
    # Linear from (0.0, RANK_DELTA_AT_ZERO) → (1.0, RANK_DELTA_AT_ONE)
    return RANK_DELTA_AT_ZERO + s * (RANK_DELTA_AT_ONE - RANK_DELTA_AT_ZERO)


def _score_to_size_factor(score: float) -> float:
    s = max(0.0, min(1.0, float(score)))
    if s <= 0.5:
        t = s / 0.5
        return SIZE_FACTOR_AT_ZERO + t * (SIZE_FACTOR_AT_HALF - SIZE_FACTOR_AT_ZERO)
    t = (s - 0.5) / 0.5
    return SIZE_FACTOR_AT_HALF + t * (SIZE_FACTOR_AT_ONE - SIZE_FACTOR_AT_HALF)


def compute_htf_anchor(
    decision_data: dict[str, Any],
    *,
    context_payload: dict[str, Any] | None = None,
    weights: dict[str, float] | None = None,
) -> dict[str, Any]:
    """Compute HTF anchor score/rank_delta/size_factor for the candidate."""
    if not htf_anchor_enabled():
        return {
            "htf_anchor_enabled": False,
            "htf_anchor_score": 0.5,
            "htf_anchor_state": "disabled",
            "htf_anchor_reasons": "",
            "htf_anchor_rank_delta": 0.0,
            "htf_anchor_size_factor": 1.0,
            "htf_anchor_family": "",
            "htf_anchor_components": {},
        }
    dd = dict(decision_data or {})
    w = dict(weights or DEFAULT_WEIGHTS)
    setup = str(
        dd.get("setup_type_canonical")
        or dd.get("setup_type")
        or dd.get("entry_thesis")
        or ""
    )
    family = _setup_family(setup)

    mtf = parse_mtf_json(dd)
    if context_payload and isinstance(context_payload.get("mtf"), dict):
        mtf = {**mtf, **context_payload["mtf"]}
    h1 = _tf_align(mtf, "1h") if isinstance(mtf.get("1h"), dict) else None
    h4 = _tf_align(mtf, "4h") if isinstance(mtf.get("4h"), dict) else None

    momentum = _safe_float(dd.get("price_momentum"), 0.0)
    ema = _safe_float(dd.get("ema_alignment"), _safe_float(dd.get("signal_ema_alignment"), 0.5))

    h_c, h_r = _h1_h4_agreement_credit(h1, h4, family)
    m_c, m_r = _momentum_credit(momentum, family)
    e_c, e_r = _ema_stack_credit(ema, family)

    components = {
        "h1_h4_agreement": {"credit": h_c, "reason": h_r, "weight": w["h1_h4_agreement"]},
        "momentum": {"credit": m_c, "reason": m_r, "weight": w["momentum"]},
        "ema_stack": {"credit": e_c, "reason": e_r, "weight": w["ema_stack"]},
    }
    total_weight = sum(float(v["weight"]) for v in components.values()) or 1.0
    weighted_sum = sum(float(v["credit"]) * float(v["weight"]) for v in components.values())
    score = max(0.0, min(1.0, weighted_sum / total_weight))

    if score >= 0.75:
        state = "htf_aligned"
    elif score >= 0.55:
        state = "htf_soft_aligned"
    elif score >= 0.35:
        state = "htf_neutral"
    else:
        state = "htf_counter_setup"

    reasons_joined = ",".join(str(v["reason"]) for v in components.values())

    return {
        "htf_anchor_enabled": True,
        "htf_anchor_score": round(score, 5),
        "htf_anchor_state": state,
        "htf_anchor_reasons": reasons_joined,
        "htf_anchor_rank_delta": round(_score_to_rank_delta(score), 5),
        "htf_anchor_size_factor": round(_score_to_size_factor(score), 5),
        "htf_anchor_family": family,
        "htf_anchor_components": components,
    }


def apply_htf_anchor_to_decision_data(
    decision_data: dict[str, Any],
    *,
    context_payload: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Stamp HTF anchor fields onto decision_data and compound into
    thesis_size_factor / thesis_rank_delta. Ranking-only side effects.
    """
    result = compute_htf_anchor(decision_data, context_payload=context_payload)
    dd = dict(decision_data or {})
    for k, v in result.items():
        dd[k] = v
    if result.get("htf_anchor_enabled"):
        try:
            prev_size = float(dd.get("thesis_size_factor") or 1.0)
        except (TypeError, ValueError):
            prev_size = 1.0
        anchor_size = float(result["htf_anchor_size_factor"])
        dd["thesis_size_factor"] = round(max(SIZE_FACTOR_AT_ZERO, prev_size * anchor_size), 5)
        try:
            prev_rank_delta = float(dd.get("thesis_rank_delta") or 0.0)
        except (TypeError, ValueError):
            prev_rank_delta = 0.0
        dd["thesis_rank_delta"] = round(prev_rank_delta + float(result["htf_anchor_rank_delta"]), 5)
    dd["hard_block"] = bool(dd.get("hard_block") or False)
    dd["candidate_eligible"] = True
    return dd


__all__ = [
    "apply_htf_anchor_to_decision_data",
    "compute_htf_anchor",
    "htf_anchor_enabled",
]
