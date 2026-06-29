"""Trust-weighted block scores from DAY v5 feature health sidecar (rank inputs only)."""

from __future__ import annotations

import json
import math
from typing import Any

from backend.services.ai_decision_contract import AI_FEATURE_DIM_V1
from backend.services.day_feature_audit import _block_for_index

BLOCK_SCORE_KEYS: tuple[str, ...] = (
    "trend_block_score",
    "momentum_block_score",
    "volatility_block_score",
    "volume_block_score",
    "sentiment_block_score",
    "orderbook_block_score",
    "context_block_score",
    "time_block_score",
    "feature_health_score",
)

_BLOCK_TO_SCORE_KEY: dict[str, str] = {
    "trend": "trend_block_score",
    "momentum": "momentum_block_score",
    "volatility": "volatility_block_score",
    "volume_profile": "volume_block_score",
    "advanced_volume": "volume_block_score",
    "market_sentiment": "sentiment_block_score",
    "microstructure": "orderbook_block_score",
    "context_125_145": "context_block_score",
    "context_day_full": "context_block_score",
    "time_based": "time_block_score",
    "basic_price": "trend_block_score",
    "technical_indicators": "trend_block_score",
    "advanced_ta": "momentum_block_score",
    "market_structure": "orderbook_block_score",
    "volatility_distribution": "volatility_block_score",
}


def _safe_float(v: Any, default: float = 0.0) -> float:
    try:
        f = float(v)
        return f if math.isfinite(f) else default
    except (TypeError, ValueError):
        return default


def parse_feature_health_sidecar(decision_data: dict[str, Any] | None) -> dict[str, Any] | None:
    if not decision_data:
        return None
    raw = decision_data.get("feature_health_json")
    if not raw:
        return None
    try:
        parsed = json.loads(raw) if isinstance(raw, str) else dict(raw)
        return parsed if isinstance(parsed, dict) else None
    except Exception:
        return None


def compute_block_scores_from_sidecar(sidecar: dict[str, Any] | None) -> dict[str, float]:
    """Aggregate per-feature trust into block scores in [0, 1]."""
    out: dict[str, float] = {k: 0.0 for k in BLOCK_SCORE_KEYS}
    if not sidecar:
        out["feature_health_score"] = 0.0
        return out

    rows = list(sidecar.get("features") or [])
    if not rows:
        hp = _safe_float(sidecar.get("health_pct"), 0.0)
        out["feature_health_score"] = max(0.0, min(1.0, hp / 100.0))
        return out

    accum: dict[str, list[float]] = {k: [] for k in BLOCK_SCORE_KEYS}
    trust_vals: list[float] = []

    for row in rows:
        block = str(row.get("block") or _block_for_index(int(row.get("index", 1)) - 1))
        score_key = _BLOCK_TO_SCORE_KEY.get(block)
        trust = _safe_float(row.get("trust_score"), 0.0)
        trust_vals.append(trust)
        if score_key:
            accum[score_key].append(trust)

    for key, vals in accum.items():
        if vals:
            out[key] = round(max(0.0, min(1.0, sum(vals) / len(vals))), 4)

    if trust_vals:
        out["feature_health_score"] = round(max(0.0, min(1.0, sum(trust_vals) / len(trust_vals))), 4)
    else:
        hp = _safe_float(sidecar.get("health_pct"), 0.0)
        out["feature_health_score"] = max(0.0, min(1.0, hp / 100.0))

    return out


def compute_block_scores_from_decision_data(decision_data: dict[str, Any] | None) -> dict[str, float]:
    sidecar = parse_feature_health_sidecar(decision_data)
    if sidecar:
        return compute_block_scores_from_sidecar(sidecar)
    hp = _safe_float((decision_data or {}).get("feature_health_pct"), 100.0)
    base = max(0.0, min(1.0, hp / 100.0))
    return {k: (base if k == "feature_health_score" else 0.5 * base) for k in BLOCK_SCORE_KEYS}


def block_scores_rank_delta(block_scores: dict[str, float]) -> float:
    """Bounded rank nudge from block health (not a gate)."""
    fh = _safe_float(block_scores.get("feature_health_score"), 0.5)
    trend = _safe_float(block_scores.get("trend_block_score"), 0.5)
    exec_proxy = _safe_float(block_scores.get("orderbook_block_score"), 0.5)
    composite = 0.45 * fh + 0.35 * trend + 0.20 * exec_proxy
    return round(max(-0.06, min(0.06, (composite - 0.55) * 0.20)), 4)


__all__ = [
    "BLOCK_SCORE_KEYS",
    "block_scores_rank_delta",
    "compute_block_scores_from_decision_data",
    "compute_block_scores_from_sidecar",
    "parse_feature_health_sidecar",
]
