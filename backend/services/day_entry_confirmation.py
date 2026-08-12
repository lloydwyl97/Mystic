"""DAY entry-confirmation soft demotion.

Post-audit observation: recent DAY BUYs closed with MFE < 0.06% but MAE
0.25–0.65% — entries triggered at bar-close on setups that never actually
confirmed a directional move. This module computes a soft "did the bar
confirm the entry direction?" penalty that lowers a candidate's rank and
size when confirmation is missing, without hard-blocking the trade.

Ranking + sizing only. Never a hard block. All checks are advisory and
compound into `entry_confirmation_score` in [0, 1]:

* 1.0 → all confirmation signals green (mark > entry_vwap, body-heavy
  candle, EMA aligned, ADX not deep in noise band).
* 0.0 → all signals red (below VWAP, top-wick rejection, EMA misaligned,
  ADX far below trend floor).

Bandit and outcome-penalty modules can then multiply the candidate's
size and rank tilt by the resulting factor. The size floor is 0.20 so
even a fully-red confirmation still allows a small exploration trade —
never a fill-time block.

Feature flag: DAY_ENTRY_CONFIRMATION_ENABLED (default true).
"""

from __future__ import annotations

import os
from typing import Any

# Score components: each check returns a 0-1 credit. Their weighted mean is
# `entry_confirmation_score`. Weights sum to 1.0 by design.
DEFAULT_WEIGHTS: dict[str, float] = {
    "mark_vs_vwap": 0.35,
    "candle_body": 0.25,
    "ema_alignment": 0.20,
    "adx_floor": 0.20,
}

# Rank/size mappings from score:
#   score 1.0 → rank_delta 0.0, size_factor 1.0
#   score 0.5 → rank_delta -0.05, size_factor 0.60
#   score 0.0 → rank_delta -0.15, size_factor 0.20 (floor)
RANK_DELTA_AT_ZERO = -0.15
SIZE_FACTOR_AT_ZERO = 0.20
SIZE_FACTOR_AT_HALF = 0.60


def entry_confirmation_enabled() -> bool:
    return os.getenv("DAY_ENTRY_CONFIRMATION_ENABLED", "true").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )


def _mark_vs_vwap_credit(mark: float, entry_vwap: float) -> tuple[float, str]:
    """Credit for price standing above the entry VWAP by a small buffer."""
    if entry_vwap <= 0 or mark <= 0:
        return 0.5, "vwap_unknown"
    diff_pct = (mark - entry_vwap) / entry_vwap
    if diff_pct >= 0.0015:  # +0.15% clear
        return 1.0, "mark_above_vwap_strong"
    if diff_pct >= 0.0002:  # +0.02% touch
        return 0.75, "mark_above_vwap"
    if diff_pct >= -0.0005:  # essentially at VWAP
        return 0.5, "mark_at_vwap"
    if diff_pct >= -0.0015:  # slightly below
        return 0.25, "mark_below_vwap"
    return 0.0, "mark_far_below_vwap"


def _candle_body_credit(body_pct: float, upper_wick_pct: float, lower_wick_pct: float) -> tuple[float, str]:
    """Credit for a mostly-body candle without top-wick rejection.

    body_pct, upper_wick_pct, lower_wick_pct are fractions of the candle's
    total range (they roughly sum to 1). Missing data returns neutral 0.5.
    """
    if body_pct <= 0 and upper_wick_pct <= 0 and lower_wick_pct <= 0:
        return 0.5, "candle_unknown"
    body = max(0.0, min(1.0, float(body_pct or 0.0)))
    upper = max(0.0, min(1.0, float(upper_wick_pct or 0.0)))
    if body >= 0.6 and upper <= 0.25:
        return 1.0, "body_dominant"
    if body >= 0.4 and upper <= 0.4:
        return 0.75, "body_ok"
    if upper >= 0.5 and body <= 0.3:
        return 0.15, "top_wick_rejection"
    if body <= 0.25:
        return 0.35, "body_thin"
    return 0.5, "candle_neutral"


def _ema_alignment_credit(ema_alignment: float) -> tuple[float, str]:
    """Credit from signal EMA alignment (already normalized to [0, 1])."""
    if ema_alignment is None:
        return 0.5, "ema_unknown"
    try:
        v = float(ema_alignment)
    except (TypeError, ValueError):
        return 0.5, "ema_unknown"
    if v >= 0.75:
        return 1.0, "ema_strong"
    if v >= 0.55:
        return 0.75, "ema_ok"
    if v >= 0.40:
        return 0.5, "ema_neutral"
    if v >= 0.25:
        return 0.25, "ema_weak"
    return 0.0, "ema_misaligned"


def _adx_floor_credit(adx: float, setup: str) -> tuple[float, str]:
    """Credit from ADX floor. Trend-family setups need higher ADX; range
    setups are OK with lower ADX. This is a *soft* preference, not a gate.
    """
    if adx is None:
        return 0.5, "adx_unknown"
    try:
        v = float(adx)
    except (TypeError, ValueError):
        return 0.5, "adx_unknown"
    s = str(setup or "").upper()
    is_trend = ("TREND" in s) or ("PULLBACK" in s and "RANGE" not in s) or ("BREAKOUT" in s)
    if is_trend:
        if v >= 22.0:
            return 1.0, "adx_trend_strong"
        if v >= 16.0:
            return 0.6, "adx_trend_ok"
        if v >= 12.0:
            return 0.35, "adx_trend_weak"
        return 0.1, "adx_trend_dead"
    else:
        # Range / mean-reversion setups: prefer ADX not too high (choppy is fine).
        if v <= 18.0:
            return 1.0, "adx_range_ok"
        if v <= 25.0:
            return 0.75, "adx_range_moderate"
        if v <= 32.0:
            return 0.5, "adx_range_trending_up"
        return 0.3, "adx_range_too_trendy"


def _score_to_rank_delta(score: float) -> float:
    s = max(0.0, min(1.0, float(score)))
    # Linear from (1.0, 0.0) → (0.0, RANK_DELTA_AT_ZERO)
    return RANK_DELTA_AT_ZERO * (1.0 - s)


def _score_to_size_factor(score: float) -> float:
    s = max(0.0, min(1.0, float(score)))
    # Piecewise: [0, 0.5] → [SIZE_FACTOR_AT_ZERO, SIZE_FACTOR_AT_HALF]
    #            [0.5, 1.0] → [SIZE_FACTOR_AT_HALF, 1.0]
    if s <= 0.5:
        t = s / 0.5
        return SIZE_FACTOR_AT_ZERO + t * (SIZE_FACTOR_AT_HALF - SIZE_FACTOR_AT_ZERO)
    t = (s - 0.5) / 0.5
    return SIZE_FACTOR_AT_HALF + t * (1.0 - SIZE_FACTOR_AT_HALF)


def compute_entry_confirmation(
    decision_data: dict[str, Any],
    *,
    current_price: float | None = None,
    weights: dict[str, float] | None = None,
) -> dict[str, Any]:
    """Compute the entry-confirmation score, reasons, rank delta, size factor.

    Reads fields from `decision_data` populated during the signal→candidate
    pipeline: `entry_vwap`, `candle_body_pct`, `candle_upper_wick_pct`,
    `candle_lower_wick_pct`, `signal_ema_alignment`, `signal_adx`,
    `setup_type` / `setup_type_canonical` / `entry_thesis`. Missing fields
    default to neutral 0.5 credit so we never punish for a data gap.

    Returns a dict with:
    * `entry_confirmation_score` — float [0, 1]
    * `entry_confirmation_state` — coarse label
    * `entry_confirmation_reasons` — per-check reason labels (joined string)
    * `entry_confirmation_rank_delta` — negative float, additive to rank
    * `entry_confirmation_size_factor` — [SIZE_FLOOR, 1.0]
    * `entry_confirmation_components` — per-check credit dict
    * `entry_confirmation_enabled` — mirrors the module flag
    """
    if not entry_confirmation_enabled():
        return {
            "entry_confirmation_enabled": False,
            "entry_confirmation_score": 1.0,
            "entry_confirmation_state": "disabled",
            "entry_confirmation_reasons": "",
            "entry_confirmation_rank_delta": 0.0,
            "entry_confirmation_size_factor": 1.0,
            "entry_confirmation_components": {},
        }

    dd = dict(decision_data or {})
    w = dict(weights or DEFAULT_WEIGHTS)

    entry_vwap = float(dd.get("entry_vwap") or 0.0)
    mark = float(current_price if current_price is not None else (dd.get("current_price") or dd.get("mark_price") or dd.get("price") or 0.0))
    body = float(dd.get("candle_body_pct") or 0.0)
    upper = float(dd.get("candle_upper_wick_pct") or 0.0)
    lower = float(dd.get("candle_lower_wick_pct") or 0.0)
    ema = dd.get("signal_ema_alignment")
    adx = dd.get("signal_adx") or dd.get("adx")
    setup = dd.get("setup_type_canonical") or dd.get("setup_type") or dd.get("entry_thesis") or ""

    vwap_c, vwap_r = _mark_vs_vwap_credit(mark, entry_vwap)
    body_c, body_r = _candle_body_credit(body, upper, lower)
    ema_c, ema_r = _ema_alignment_credit(ema if ema is not None else 0.5)
    adx_c, adx_r = _adx_floor_credit(adx if adx is not None else 20.0, setup)

    components = {
        "mark_vs_vwap": {"credit": vwap_c, "reason": vwap_r, "weight": w["mark_vs_vwap"]},
        "candle_body": {"credit": body_c, "reason": body_r, "weight": w["candle_body"]},
        "ema_alignment": {"credit": ema_c, "reason": ema_r, "weight": w["ema_alignment"]},
        "adx_floor": {"credit": adx_c, "reason": adx_r, "weight": w["adx_floor"]},
    }
    total_weight = sum(float(v["weight"]) for v in components.values()) or 1.0
    weighted_sum = sum(float(v["credit"]) * float(v["weight"]) for v in components.values())
    score = weighted_sum / total_weight
    score = max(0.0, min(1.0, score))

    if score >= 0.75:
        state = "confirmed"
    elif score >= 0.55:
        state = "weak_confirmation"
    elif score >= 0.35:
        state = "unconfirmed"
    else:
        state = "contradicted"

    reasons_joined = ",".join(str(v["reason"]) for v in components.values())

    return {
        "entry_confirmation_enabled": True,
        "entry_confirmation_score": round(score, 5),
        "entry_confirmation_state": state,
        "entry_confirmation_reasons": reasons_joined,
        "entry_confirmation_rank_delta": round(_score_to_rank_delta(score), 5),
        "entry_confirmation_size_factor": round(_score_to_size_factor(score), 5),
        "entry_confirmation_components": components,
    }


def apply_entry_confirmation_to_decision_data(
    decision_data: dict[str, Any],
    *,
    current_price: float | None = None,
) -> dict[str, Any]:
    """Stamp entry-confirmation fields onto decision_data and compound into
    thesis_size_factor / thesis_rank_delta so downstream ranking + sizing pick
    it up. Ranking-only side effects; no candidate_eligible flip.
    """
    result = compute_entry_confirmation(decision_data, current_price=current_price)
    dd = dict(decision_data or {})
    for k, v in result.items():
        if k == "entry_confirmation_components":
            # Keep the components dict on decision_data too, for logging.
            dd[k] = v
        else:
            dd[k] = v

    if result.get("entry_confirmation_enabled"):
        try:
            prev_size = float(dd.get("thesis_size_factor") or 1.0)
        except (TypeError, ValueError):
            prev_size = 1.0
        conf_size = float(result["entry_confirmation_size_factor"])
        dd["thesis_size_factor"] = round(max(SIZE_FACTOR_AT_ZERO, prev_size * conf_size), 5)
        try:
            prev_rank_delta = float(dd.get("thesis_rank_delta") or 0.0)
        except (TypeError, ValueError):
            prev_rank_delta = 0.0
        conf_rank_delta = float(result["entry_confirmation_rank_delta"])
        dd["thesis_rank_delta"] = round(prev_rank_delta + conf_rank_delta, 5)

    dd["hard_block"] = bool(dd.get("hard_block") or False)  # unchanged; ranking only
    dd["candidate_eligible"] = True  # always eligible; this is a soft demote
    return dd


__all__ = [
    "apply_entry_confirmation_to_decision_data",
    "compute_entry_confirmation",
    "entry_confirmation_enabled",
]
