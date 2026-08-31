"""Trust-weighted SCALP block scores (rank inputs only — not gates)."""

from __future__ import annotations

import json
import math
from typing import Any

from backend.services.scalp_feature_contract import _block_for_index

BLOCK_SCORE_KEYS: tuple[str, ...] = (
    "scalp_microstructure_score",
    "scalp_momentum_score",
    "scalp_volume_burst_score",
    "scalp_volatility_score",
    "scalp_spread_quality_score",
    "scalp_depth_quality_score",
    "scalp_execution_quality_score",
    "scalp_context_score",
    "scalp_feature_health_score",
)

_BLOCK_MAP: dict[str, str] = {
    "microstructure": "scalp_microstructure_score",
    "momentum": "scalp_momentum_score",
    "kline_1m": "scalp_volume_burst_score",
    "gross_estimate": "scalp_volatility_score",
    "micro_regime": "scalp_context_score",
    "execution": "scalp_execution_quality_score",
    "signal_meta": "scalp_momentum_score",
    "memory": "scalp_context_score",
}


def _sf(v: Any, d: float = 0.0) -> float:
    try:
        f = float(v)
        return f if math.isfinite(f) else d
    except (TypeError, ValueError):
        return d


def compute_block_scores_from_sidecar(sidecar: dict[str, Any] | None) -> dict[str, float]:
    out = dict.fromkeys(BLOCK_SCORE_KEYS, 0.0)
    if not sidecar:
        return out
    rows = list(sidecar.get("features") or [])
    accum: dict[str, list[float]] = {k: [] for k in BLOCK_SCORE_KEYS}
    trusts: list[float] = []
    for row in rows:
        block = str(row.get("block") or _block_for_index(int(row.get("index", 1)) - 1))
        key = _BLOCK_MAP.get(block)
        t = _sf(row.get("trust_score"))
        trusts.append(t)
        if key:
            accum[key].append(t)
    for key, vals in accum.items():
        if vals:
            out[key] = round(max(0.0, min(1.0, sum(vals) / len(vals))), 4)
    if trusts:
        out["scalp_feature_health_score"] = round(max(0.0, min(1.0, sum(trusts) / len(trusts))), 4)
    return out


def compute_block_scores_from_intelligence(data: dict[str, Any] | None) -> dict[str, float]:
    if not data:
        return dict.fromkeys(BLOCK_SCORE_KEYS, 0.0)
    raw = data.get("feature_health_json")
    if raw:
        try:
            side = json.loads(raw) if isinstance(raw, str) else dict(raw)
            return compute_block_scores_from_sidecar(side)
        except Exception:
            pass
    spread = _sf(data.get("spread_pct"), 0.001)
    depth = abs(_sf(data.get("order_book_imbalance")))
    mom = _sf(data.get("mid_change_30s"))
    out = {
        "scalp_microstructure_score": round(max(0.0, 1.0 - spread / 0.004), 4),
        "scalp_momentum_score": round(max(0.0, min(1.0, (mom + 0.002) / 0.004)), 4),
        "scalp_volume_burst_score": round(min(1.0, _sf(data.get("kline_volume_ratio")) / 2.0), 4),
        "scalp_volatility_score": round(min(1.0, _sf(data.get("realized_volatility_pct")) / 0.01), 4),
        "scalp_spread_quality_score": round(max(0.0, 1.0 - spread / 0.003), 4),
        "scalp_depth_quality_score": round(min(1.0, depth * 2.0), 4),
        "scalp_execution_quality_score": _sf(data.get("scalp_execution_quality_score"), 0.5),
        "scalp_context_score": _sf(data.get("micro_regime_score"), 0.5),
        "scalp_feature_health_score": _sf(data.get("scalp_feature_health_score"), 0.5),
    }
    return out


def block_scores_rank_delta(scores: dict[str, float]) -> float:
    health = _sf(scores.get("scalp_feature_health_score"), 0.5)
    return round(max(-0.05, min(0.05, (health - 0.55) * 0.12)), 4)


__all__ = [
    "BLOCK_SCORE_KEYS",
    "block_scores_rank_delta",
    "compute_block_scores_from_intelligence",
    "compute_block_scores_from_sidecar",
]
