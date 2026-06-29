"""SCALP execution-aware ranking (soft rank — existing protections unchanged)."""

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


def compute_scalp_execution_scores(data: dict[str, Any]) -> dict[str, float]:
    dd = data or {}
    spread = abs(_f(dd, "spread_pct"))
    depth = abs(_f(dd, "order_book_imbalance"))
    slip = abs(_f(dd, "entry_slippage_pct") or _f(dd, "slippage_estimate", 0.0))
    impact = abs(_f(dd, "impact_pct"))
    ob_age = _f(dd, "orderbook_age_sec", -1.0)
    fresh_mod = _f(dd, "freshness_trust_modifier", 1.0)
    if fresh_mod <= 0:
        fresh_mod = 1.0
    ob_fresh = 1.0 if ob_age < 0 else _c01(1.0 - ob_age / 45.0)

    scalp_spread_score = _c01(1.0 - spread / 0.003)
    scalp_depth_score = _c01(depth * 2.0)
    scalp_slippage_score = _c01(1.0 - slip / 0.002)
    scalp_price_impact_score = _c01(1.0 - impact / 0.004)
    scalp_orderbook_freshness_score = round(ob_fresh * fresh_mod, 4)
    scalp_fill_quality_score = round(_c01(0.5 * scalp_spread_score + 0.5 * scalp_depth_score), 4)

    scalp_execution_quality_score = round(
        _c01(
            (
                0.35 * scalp_spread_score
                + 0.20 * scalp_depth_score
                + 0.15 * scalp_slippage_score
                + 0.15 * scalp_price_impact_score
                + 0.15 * scalp_orderbook_freshness_score
            )
            * min(1.0, fresh_mod)
        ),
        4,
    )

    return {
        "scalp_spread_score": round(scalp_spread_score, 4),
        "scalp_depth_score": round(scalp_depth_score, 4),
        "scalp_slippage_score": round(scalp_slippage_score, 4),
        "scalp_price_impact_score": round(scalp_price_impact_score, 4),
        "scalp_orderbook_freshness_score": scalp_orderbook_freshness_score,
        "scalp_fill_quality_score": scalp_fill_quality_score,
        "scalp_execution_quality_score": scalp_execution_quality_score,
        "execution_quality_score": scalp_execution_quality_score,
    }


def execution_rank_delta(scores: dict[str, float]) -> float:
    eq = float((scores or {}).get("scalp_execution_quality_score", 0.5))
    return round(max(-0.06, min(0.06, (eq - 0.55) * 0.18)), 4)


__all__ = ["compute_scalp_execution_scores", "execution_rank_delta"]
