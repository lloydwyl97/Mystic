"""
ai_feature_importance_diagnostics — surfaces which of the 145 live feature
dims are actually pulling weight, without touching the live feature contract.

The 145-dim vector has never been audited for dead weight. Removing dims would
be a breaking dimension-bump change (same blast radius day_chart_pattern_detector.py's
plan explicitly avoided by keeping its score out of the ML vector) — every
consumer of AI_FEATURE_DIM_V2/CONTEXT_DIMS_DAY_FULL would need coordinated
retraining. So this module is diagnostic-only: it computes and surfaces
importance from BOTH algorithm families in the blend (RandomForest's native
feature_importances_ and HistGradientBoostingClassifier's permutation
importance — HGB has no native importances_ attribute), logs the weakest
dims per symbol, and separately flags dims that are consistently weak across
ALL trained DAY symbols (a much stronger pruning signal than any single
symbol's read, since a dim could legitimately matter only for one coin).
Nothing here changes what a live model actually consumes.
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

_REDIS_KEY_PREFIX = "ai_feature_importance:"
_REDIS_TTL_SEC = 30 * 24 * 3600
_BOTTOM_N_LOG = 15
_WEAK_THRESHOLD_PERCENTILE = 20.0  # bottom 20% of combined importance = "weak" for that symbol


def feature_names_for(feature_dim: int, strategy_id: str) -> list[str]:
    """Best-effort human-readable names for diagnostic logging. Falls back to
    generic dim_i labels when no canonical name list exists for this contract —
    a missing name never blocks the importance computation itself."""
    try:
        if strategy_id == "day" and feature_dim == 145:
            from backend.services.ai_decision_contract import AI_FEATURE_DIM_V1, CONTEXT_DIMS_DAY_FULL

            names = [f"tech_{i}" for i in range(AI_FEATURE_DIM_V1)] + list(CONTEXT_DIMS_DAY_FULL)
            if len(names) == feature_dim:
                return names
    except Exception as exc:
        logger.debug("FEATURE_NAMES_LOOKUP_FAILED strategy=%s dim=%s: %s", strategy_id, feature_dim, exc)
    return [f"dim_{i}" for i in range(feature_dim)]


def _normalize(values: np.ndarray) -> np.ndarray:
    total = float(np.sum(values))
    if total <= 0 or not np.isfinite(total):
        return np.zeros_like(values)
    return values / total


def compute_and_record_feature_importance(
    *,
    strategy_id: str,
    symbol: str,
    rf_model: Any,
    gbm_model: Any | None,
    X_val_s: np.ndarray,
    y_val: np.ndarray,
    feature_dim: int,
) -> dict[str, Any]:
    """Compute RF + GBM-permutation importance for one just-trained symbol,
    log the weakest dims, persist a snapshot to Redis for cross-symbol
    aggregation, and return a small summary safe to embed in the model
    artifact for provenance. Never raises — any failure degrades to an empty
    summary so it can never break a training run."""
    names = feature_names_for(feature_dim, strategy_id)
    summary: dict[str, Any] = {"rf_importances": None, "gbm_perm_importances": None, "combined_top_weak": []}

    try:
        rf_imp = _normalize(np.asarray(rf_model.feature_importances_, dtype=np.float64))
    except Exception as exc:
        logger.debug("RF_FEATURE_IMPORTANCE_FAILED [%s] %s: %s", strategy_id, symbol, exc)
        rf_imp = np.zeros(feature_dim)

    gbm_imp = np.zeros(feature_dim)
    if gbm_model is not None and len(X_val_s) >= 10:
        try:
            from sklearn.inspection import permutation_importance

            result = permutation_importance(gbm_model, X_val_s, y_val, n_repeats=5, random_state=42, scoring="accuracy", n_jobs=1)
            gbm_imp = _normalize(np.clip(np.asarray(result.importances_mean, dtype=np.float64), 0.0, None))
        except Exception as exc:
            logger.debug("GBM_PERMUTATION_IMPORTANCE_FAILED [%s] %s: %s", strategy_id, symbol, exc)

    if len(rf_imp) != feature_dim or len(gbm_imp) != feature_dim:
        return summary

    combined = (rf_imp + gbm_imp) / 2.0 if np.sum(gbm_imp) > 0 else rf_imp
    order = np.argsort(combined)
    weakest_idx = order[:_BOTTOM_N_LOG]
    weakest = [{"index": int(i), "name": names[i], "combined_importance": round(float(combined[i]), 6)} for i in weakest_idx]

    summary = {
        "rf_importances": [round(float(v), 6) for v in rf_imp],
        "gbm_perm_importances": [round(float(v), 6) for v in gbm_imp],
        "combined_top_weak": weakest,
        "trained_at": time.time(),
    }

    logger.info(
        "FEATURE_IMPORTANCE_WEAKEST [%s] %s bottom_%d=%s",
        strategy_id,
        symbol,
        _BOTTOM_N_LOG,
        [w["name"] for w in weakest],
    )

    try:
        from backend.config.redis_config import get_shared_redis_sync

        r = get_shared_redis_sync()
        if r:
            r.hset(f"{_REDIS_KEY_PREFIX}{strategy_id}", symbol, json.dumps(summary, separators=(",", ":")))
            r.expire(f"{_REDIS_KEY_PREFIX}{strategy_id}", _REDIS_TTL_SEC)
    except Exception as exc:
        logger.debug("FEATURE_IMPORTANCE_REDIS_WRITE_FAILED [%s] %s: %s", strategy_id, symbol, exc)

    return summary


def log_cross_symbol_weak_features(strategy_id: str, *, feature_dim: int = 145) -> list[str]:
    """Once per full training pass (after all symbols for this strategy have
    trained this cycle): read every symbol's just-written importance snapshot
    and flag dims that land in the bottom _WEAK_THRESHOLD_PERCENTILE for EVERY
    symbol that has a snapshot — a dim only one coin barely uses is not a
    pruning candidate, a dim no coin uses is. Diagnostic only; returns the
    flagged names for the caller to log/expose, never mutates the feature
    contract itself."""
    names = feature_names_for(feature_dim, strategy_id)
    try:
        from backend.config.redis_config import get_shared_redis_sync

        r = get_shared_redis_sync()
        if not r:
            return []
        raw = r.hgetall(f"{_REDIS_KEY_PREFIX}{strategy_id}") or {}
    except Exception as exc:
        logger.debug("CROSS_SYMBOL_FEATURE_IMPORTANCE_READ_FAILED strategy=%s: %s", strategy_id, exc)
        return []

    weak_sets: list[set[int]] = []
    symbols_seen: list[str] = []
    for k, v in raw.items():
        sym = k.decode() if isinstance(k, bytes) else str(k)
        payload = v.decode() if isinstance(v, bytes) else str(v)
        try:
            data = json.loads(payload)
            rf_imp = data.get("rf_importances")
            gbm_imp = data.get("gbm_perm_importances")
            if not rf_imp or len(rf_imp) != feature_dim:
                continue
            combined = np.asarray(rf_imp, dtype=np.float64)
            if gbm_imp and len(gbm_imp) == feature_dim and sum(gbm_imp) > 0:
                combined = (combined + np.asarray(gbm_imp, dtype=np.float64)) / 2.0
            cutoff = np.percentile(combined, _WEAK_THRESHOLD_PERCENTILE)
            weak_sets.append({i for i, v2 in enumerate(combined) if v2 <= cutoff})
            symbols_seen.append(sym)
        except (ValueError, TypeError, json.JSONDecodeError) as exc:
            logger.debug("CROSS_SYMBOL_FEATURE_IMPORTANCE_PARSE_FAILED sym=%s: %s", sym, exc)
            continue

    if len(weak_sets) < 2:
        return []

    consistently_weak = set.intersection(*weak_sets)
    weak_names = sorted(names[i] for i in consistently_weak if i < len(names))
    if weak_names:
        logger.info(
            "FEATURE_PRUNING_CANDIDATES strategy=%s symbols=%s consistently_weak_across_all=%s",
            strategy_id,
            symbols_seen,
            weak_names,
        )
    return weak_names


__all__ = [
    "compute_and_record_feature_importance",
    "feature_names_for",
    "log_cross_symbol_weak_features",
]
