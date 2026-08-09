"""DAY candle-quality soft-demotion gate.

The 145-dim ML feature vector aggregates volume into a single scalar
(`ctx_relative_volume` = 24h volume relative to baseline). It cannot see
"big red volume bar on the drop" — the pattern a human reads instantly on
a 15m chart. This gate adds an explicit read of the last few bars'
volume, wick, and price-volume divergence, and soft-demotes bullish
buy candidates whose most recent bar shows classic rejection behavior.

Read-only inputs (populated by ai_signal_generator → portfolio_engine_integration):
* `recent_last_bar_vol_ratio` — last bar vol / prior-20 SMA (spike detector)
* `recent_vol_5_vs_20` — last-5-bar mean vol / prior-20 SMA
* `recent_vp_divergence` — signed score in [-1, +1]; negative = bearish divergence
* `recent_3bar_reversal_flag` — 1 = last 3 bars form top-wick rejection pattern
* `candle_upper_wick_pct`, `candle_body_pct` — last-bar shape

Never a hard block. Ranking + size only. Feature flag:
DAY_CANDLE_QUALITY_GATE_ENABLED (default true).
"""

from __future__ import annotations

import os
from typing import Any

RANK_DELTA_AT_ZERO = -0.15
SIZE_FACTOR_AT_ZERO = 0.25
SIZE_FACTOR_AT_HALF = 0.65

DEFAULT_WEIGHTS: dict[str, float] = {
    "volume_spike_reversal": 0.35,  # big vol on a rejection bar = biggest tell
    "vp_divergence": 0.25,
    "three_bar_reversal": 0.20,
    "last_bar_shape": 0.20,
}


def candle_quality_gate_enabled() -> bool:
    return os.getenv("DAY_CANDLE_QUALITY_GATE_ENABLED", "true").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )


def _safe_float(v: Any, default: float = 0.0) -> float:
    try:
        if v is None or v == "":
            return default
        return float(v)
    except (TypeError, ValueError):
        return default


def _safe_int(v: Any, default: int = 0) -> int:
    try:
        if v is None or v == "":
            return default
        return int(float(v))
    except (TypeError, ValueError):
        return default


def _volume_spike_reversal_credit(
    last_bar_vol_ratio: float,
    upper_wick_pct: float,
    body_pct: float,
    direction: str,
) -> tuple[float, str]:
    """Only penalize BUYs. Big volume + big upper wick + small body = distribution.

    Returns credit in [0, 1]. 1 = clean (no penalty), 0 = full penalty.
    """
    if direction != "buy":
        return 1.0, "not_buy"
    is_top_reject = upper_wick_pct >= 0.45 and body_pct <= 0.35
    if not is_top_reject:
        return 1.0, "no_top_rejection"
    if last_bar_vol_ratio >= 2.5:
        return 0.05, "vol_spike_2p5x_on_rejection"
    if last_bar_vol_ratio >= 1.7:
        return 0.30, "vol_spike_1p7x_on_rejection"
    if last_bar_vol_ratio >= 1.2:
        return 0.60, "vol_elevated_on_rejection"
    return 0.80, "rejection_no_volume"


def _vp_divergence_credit(vp_divergence: float, direction: str) -> tuple[float, str]:
    """Bearish divergence (negative) on a BUY = penalty. Bullish divergence = boost neutral."""
    if direction != "buy":
        return 1.0, "not_buy"
    if vp_divergence <= -0.7:
        return 0.10, "strong_bearish_divergence"
    if vp_divergence <= -0.35:
        return 0.35, "bearish_divergence"
    if vp_divergence <= -0.15:
        return 0.70, "mild_bearish_divergence"
    return 1.0, "no_divergence"


def _three_bar_reversal_credit(flag: int, direction: str) -> tuple[float, str]:
    if direction != "buy":
        return 1.0, "not_buy"
    if flag == 1:
        return 0.15, "three_bar_top_reversal"
    return 1.0, "no_reversal"


def _last_bar_shape_credit(upper_wick_pct: float, body_pct: float, direction: str) -> tuple[float, str]:
    """Standalone last-bar top rejection (without needing volume confirmation)."""
    if direction != "buy":
        return 1.0, "not_buy"
    if upper_wick_pct >= 0.55 and body_pct <= 0.25:
        return 0.25, "top_wick_dominant"
    if upper_wick_pct >= 0.40 and body_pct <= 0.35:
        return 0.55, "top_wick_present"
    return 1.0, "shape_ok"


def _score_to_rank_delta(score: float) -> float:
    s = max(0.0, min(1.0, float(score)))
    return RANK_DELTA_AT_ZERO * (1.0 - s)


def _score_to_size_factor(score: float) -> float:
    s = max(0.0, min(1.0, float(score)))
    if s <= 0.5:
        t = s / 0.5
        return SIZE_FACTOR_AT_ZERO + t * (SIZE_FACTOR_AT_HALF - SIZE_FACTOR_AT_ZERO)
    t = (s - 0.5) / 0.5
    return SIZE_FACTOR_AT_HALF + t * (1.0 - SIZE_FACTOR_AT_HALF)


def compute_candle_quality(decision_data: dict[str, Any]) -> dict[str, Any]:
    if not candle_quality_gate_enabled():
        return {
            "candle_quality_gate_enabled": False,
            "candle_quality_score": 1.0,
            "candle_quality_state": "disabled",
            "candle_quality_reasons": "",
            "candle_quality_rank_delta": 0.0,
            "candle_quality_size_factor": 1.0,
            "candle_quality_components": {},
        }

    dd = dict(decision_data or {})
    direction = str(
        dd.get("side") or dd.get("prediction") or dd.get("action") or ""
    ).strip().lower()

    last_vol_ratio = _safe_float(dd.get("recent_last_bar_vol_ratio"), 1.0)
    vp_div = _safe_float(dd.get("recent_vp_divergence"), 0.0)
    reversal_flag = _safe_int(dd.get("recent_3bar_reversal_flag"), 0)
    upper_wick = _safe_float(dd.get("candle_upper_wick_pct"), 0.0)
    body = _safe_float(dd.get("candle_body_pct"), 0.0)

    vsr_c, vsr_r = _volume_spike_reversal_credit(last_vol_ratio, upper_wick, body, direction)
    vpd_c, vpd_r = _vp_divergence_credit(vp_div, direction)
    tbr_c, tbr_r = _three_bar_reversal_credit(reversal_flag, direction)
    lbs_c, lbs_r = _last_bar_shape_credit(upper_wick, body, direction)

    w = DEFAULT_WEIGHTS
    components = {
        "volume_spike_reversal": {"credit": vsr_c, "reason": vsr_r, "weight": w["volume_spike_reversal"]},
        "vp_divergence": {"credit": vpd_c, "reason": vpd_r, "weight": w["vp_divergence"]},
        "three_bar_reversal": {"credit": tbr_c, "reason": tbr_r, "weight": w["three_bar_reversal"]},
        "last_bar_shape": {"credit": lbs_c, "reason": lbs_r, "weight": w["last_bar_shape"]},
    }
    total_w = sum(float(v["weight"]) for v in components.values()) or 1.0
    weighted_sum = sum(float(v["credit"]) * float(v["weight"]) for v in components.values())
    score = max(0.0, min(1.0, weighted_sum / total_w))

    if score >= 0.85:
        state = "clean"
    elif score >= 0.65:
        state = "acceptable"
    elif score >= 0.40:
        state = "weak"
    else:
        state = "rejection_pattern"

    reasons_joined = ",".join(str(v["reason"]) for v in components.values())
    return {
        "candle_quality_gate_enabled": True,
        "candle_quality_score": round(score, 5),
        "candle_quality_state": state,
        "candle_quality_reasons": reasons_joined,
        "candle_quality_rank_delta": round(_score_to_rank_delta(score), 5),
        "candle_quality_size_factor": round(_score_to_size_factor(score), 5),
        "candle_quality_components": components,
        "candle_quality_last_vol_ratio": round(last_vol_ratio, 5),
        "candle_quality_vp_divergence": round(vp_div, 5),
        "candle_quality_reversal_flag": int(reversal_flag),
    }


def apply_candle_quality_to_decision_data(
    decision_data: dict[str, Any],
) -> dict[str, Any]:
    """Stamp candle-quality fields and compound thesis_size_factor + thesis_rank_delta."""
    result = compute_candle_quality(decision_data)
    dd = dict(decision_data or {})
    for k, v in result.items():
        dd[k] = v

    if result.get("candle_quality_gate_enabled"):
        try:
            prev_size = float(dd.get("thesis_size_factor") or 1.0)
        except (TypeError, ValueError):
            prev_size = 1.0
        cq_size = float(result["candle_quality_size_factor"])
        dd["thesis_size_factor"] = round(max(SIZE_FACTOR_AT_ZERO, prev_size * cq_size), 5)
        try:
            prev_rank_delta = float(dd.get("thesis_rank_delta") or 0.0)
        except (TypeError, ValueError):
            prev_rank_delta = 0.0
        cq_rank_delta = float(result["candle_quality_rank_delta"])
        dd["thesis_rank_delta"] = round(prev_rank_delta + cq_rank_delta, 5)

    dd["hard_block"] = bool(dd.get("hard_block") or False)
    dd["candidate_eligible"] = bool(dd.get("candidate_eligible", True))
    return dd


__all__ = [
    "apply_candle_quality_to_decision_data",
    "compute_candle_quality",
    "candle_quality_gate_enabled",
]
