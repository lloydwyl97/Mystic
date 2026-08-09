"""DAY model-quality soft-demotion gate.

The ML model produces a per-coin confidence score in [0, 1]. When the
model's own validation accuracy is at or below chance (0.5 for binary,
0.33 for triple), that confidence is effectively noise — trusting it
proportionally exposes bad trades.

This gate reads `model_accuracy` (stamped through by ai_signal_generator
in batch 8, propagated via portfolio_engine_integration in this ship)
and applies size + rank soft demotion when accuracy is at or below a
configurable floor. Never a hard block.

Feature flag: DAY_MODEL_QUALITY_GATE_ENABLED (default true).
Accuracy floor: DAY_MODEL_ACCURACY_FLOOR (default 0.50 — models below
this are treated as unreliable and demoted).
"""

from __future__ import annotations

import os
from typing import Any

SIZE_FACTOR_AT_ZERO = 0.20
RANK_DELTA_AT_ZERO = -0.15


def model_quality_gate_enabled() -> bool:
    return os.getenv("DAY_MODEL_QUALITY_GATE_ENABLED", "true").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )


def _accuracy_floor() -> float:
    try:
        return float(os.getenv("DAY_MODEL_ACCURACY_FLOOR", "0.50"))
    except (TypeError, ValueError):
        return 0.50


def _safe_float(v: Any, default: float = 0.0) -> float:
    try:
        if v is None or v == "":
            return default
        return float(v)
    except (TypeError, ValueError):
        return default


def compute_model_quality(decision_data: dict[str, Any]) -> dict[str, Any]:
    if not model_quality_gate_enabled():
        return {
            "model_quality_gate_enabled": False,
            "model_quality_score": 1.0,
            "model_quality_state": "disabled",
            "model_quality_reason": "",
            "model_quality_rank_delta": 0.0,
            "model_quality_size_factor": 1.0,
        }

    dd = dict(decision_data or {})
    acc = _safe_float(dd.get("model_accuracy"), 0.5)
    floor = _accuracy_floor()

    # Score curve:
    #   acc >= floor + 0.15 (e.g. 0.65) → 1.0 (no demotion)
    #   acc == floor (0.50)             → 0.6 (moderate demotion)
    #   acc == floor - 0.10 (0.40)      → 0.3 (heavy)
    #   acc <= 0.34 (SOL current)       → 0.10 (near-floor)
    if acc >= floor + 0.15:
        score = 1.0
        state = "reliable"
        reason = f"acc={acc:.2f}_ok"
    elif acc >= floor:
        # linear 0.6→1.0 as acc rises floor → floor+0.15
        t = (acc - floor) / 0.15
        score = 0.6 + 0.4 * max(0.0, min(1.0, t))
        state = "marginal"
        reason = f"acc={acc:.2f}_marginal"
    elif acc >= floor - 0.10:
        t = (acc - (floor - 0.10)) / 0.10
        score = 0.3 + 0.3 * max(0.0, min(1.0, t))
        state = "weak"
        reason = f"acc={acc:.2f}_below_floor"
    else:
        # deep sub-floor (e.g. SOL 0.33): heavy demotion
        score = max(0.10, 0.3 - 0.4 * ((floor - 0.10) - acc))
        state = "broken"
        reason = f"acc={acc:.2f}_untrusted"

    score = max(0.0, min(1.0, float(score)))
    rank_delta = round(RANK_DELTA_AT_ZERO * (1.0 - score), 5)
    size_factor = round(max(SIZE_FACTOR_AT_ZERO, SIZE_FACTOR_AT_ZERO + (1.0 - SIZE_FACTOR_AT_ZERO) * score), 5)

    return {
        "model_quality_gate_enabled": True,
        "model_quality_score": round(score, 5),
        "model_quality_state": state,
        "model_quality_reason": reason,
        "model_quality_accuracy": round(acc, 5),
        "model_quality_floor": round(floor, 5),
        "model_quality_rank_delta": rank_delta,
        "model_quality_size_factor": size_factor,
    }


def apply_model_quality_to_decision_data(
    decision_data: dict[str, Any],
) -> dict[str, Any]:
    """Stamp model-quality fields and compound thesis_size_factor + thesis_rank_delta."""
    result = compute_model_quality(decision_data)
    dd = dict(decision_data or {})
    for k, v in result.items():
        dd[k] = v

    if result.get("model_quality_gate_enabled"):
        try:
            prev_size = float(dd.get("thesis_size_factor") or 1.0)
        except (TypeError, ValueError):
            prev_size = 1.0
        mq_size = float(result["model_quality_size_factor"])
        dd["thesis_size_factor"] = round(max(SIZE_FACTOR_AT_ZERO, prev_size * mq_size), 5)
        try:
            prev_rank_delta = float(dd.get("thesis_rank_delta") or 0.0)
        except (TypeError, ValueError):
            prev_rank_delta = 0.0
        mq_rank_delta = float(result["model_quality_rank_delta"])
        dd["thesis_rank_delta"] = round(prev_rank_delta + mq_rank_delta, 5)

    dd["hard_block"] = bool(dd.get("hard_block") or False)
    dd["candidate_eligible"] = bool(dd.get("candidate_eligible", True))
    return dd


__all__ = [
    "apply_model_quality_to_decision_data",
    "compute_model_quality",
    "model_quality_gate_enabled",
]
