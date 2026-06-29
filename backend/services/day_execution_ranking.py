"""Execution-aware ranking scores (soft rank inputs — not trade blockers)."""

from __future__ import annotations

import math
import time
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


def compute_execution_ranking_scores(decision_data: dict[str, Any]) -> dict[str, float]:
    dd = decision_data or {}
    spread = abs(_f(dd, "spread_pct", 0.0))
    depth = abs(_f(dd, "ctx_depth_imbalance", 0.0))
    slip = abs(_f(dd, "entry_slippage_pct", 0.0) or _f(dd, "estimated_slippage_pct", 0.0))
    impact = abs(_f(dd, "price_impact_pct", 0.0) or _f(dd, "estimated_price_impact_pct", 0.0))

    ctx_age = _f(dd, "ctx_age_sec", -1.0)
    ob_age = _f(dd, "orderbook_age_sec", -1.0)
    freshness_age = ob_age if ob_age >= 0 else ctx_age
    ob_fresh = 1.0 if freshness_age < 0 else _clamp01(1.0 - freshness_age / 120.0)
    fresh_mod = _f(dd, "freshness_trust_modifier", 1.0)
    if fresh_mod <= 0:
        fresh_mod = 1.0
    fresh_mod = max(0.25, min(1.0, fresh_mod))

    spread_score = _clamp01(1.0 - spread / 0.004)
    depth_score = _clamp01(depth * 2.0)
    slippage_score = _clamp01(1.0 - slip / 0.003)
    price_impact_score = _clamp01(1.0 - impact / 0.006)
    orderbook_freshness_score = round(ob_fresh, 4)

    execution_quality_score = round(
        _clamp01(
            (
                0.30 * spread_score
                + 0.20 * depth_score
                + 0.20 * slippage_score
                + 0.15 * price_impact_score
                + 0.15 * orderbook_freshness_score
            )
            * fresh_mod
        ),
        4,
    )

    return {
        "spread_score": round(spread_score, 4),
        "depth_score": round(depth_score, 4),
        "slippage_score": round(slippage_score, 4),
        "price_impact_score": round(price_impact_score, 4),
        "orderbook_freshness_score": orderbook_freshness_score,
        "execution_quality_score": execution_quality_score,
    }


def execution_rank_delta(scores: dict[str, float]) -> float:
    eq = float((scores or {}).get("execution_quality_score", 0.5))
    return round(max(-0.05, min(0.05, (eq - 0.55) * 0.15)), 4)


__all__ = ["compute_execution_ranking_scores", "execution_rank_delta"]
