"""
ai_meta_labeling — a small secondary "trust" model over the primary per-coin
RF+GBM classifier's BUY decisions (Lopez de Prado meta-labeling pattern).

The primary model (ai_blended_classifier.py, trained in ai_training_pipeline.py)
answers "BUY/HOLD/SELL, how confident". This module answers a narrower
question: "given the primary model already said BUY, how much should this
specific instance actually be trusted?" — using the bounded-nudge and
diagnostic signals that were deliberately kept OUT of the live 145-dim feature
vector (chart_pattern_score, model_disagreement, cross_sectional_rank_delta,
setup_score, execution_rank_delta, spread/regime context). Meta-labeling only
ever filters/discounts an existing BUY; it can never turn a HOLD/SELL into a
BUY, and it can never boost confidence above what the primary model already
said — see score_meta_trust's [0.5, 1.0] multiplier floor/ceiling.

Trained pooled across all 4 DAY symbols (BTC/ETH/SOL/XRP) rather than
per-symbol — meta-labeling training sets are inherently smaller than the
primary model's (only BUY-decision rows count), so pooling is what makes this
viable with real historical volume today instead of waiting on a much larger
per-symbol sample. Symbol identity is itself a one-hot feature so the model
can still learn symbol-specific bias if/when data supports it.
"""

from __future__ import annotations

import logging
import pickle
import sqlite3
import time
from pathlib import Path
from typing import Any

import numpy as np

from backend.config.training_label_economics import required_edge_pct_for_strategy
from backend.database_schema import DATABASE_PATH

logger = logging.getLogger(__name__)

META_MODEL_DIR = Path("models/active/meta")
META_MODEL_FILENAME = "day_meta_label.pkl"
MIN_TRAIN_SAMPLES = 60
_CACHE_TTL_SEC = 300.0
_cached_artifact: tuple[float, dict[str, Any] | None] = (0.0, None)

_TOP4_BASES = ("BTC", "ETH", "SOL", "XRP")
_REGIME_BUCKETS = ("bull", "bear", "range", "chop", "neutral")

# Straight numeric reads from decision_data / ai_candidate_snapshots. The last
# four (model_disagreement..execution_rank_delta) are recent additions to the
# snapshot schema (see ai_learning_ingestion.py migration) — 0.0 on any row
# predating that migration, same graceful bootstrap as day_route_regime.
_NUMERIC_KEYS: tuple[str, ...] = (
    "confidence",
    "thesis_score",
    "relative_volume",
    "spread_pct",
    "cost_estimate_pct",
    "chart_pattern_score",
    "model_disagreement",
    "cross_sectional_rank_delta",
    "setup_score",
    "execution_rank_delta",
)


def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        v = float(x)
        return v if np.isfinite(v) else default
    except (TypeError, ValueError):
        return default


def _regime_one_hot(regime_raw: str) -> list[float]:
    r = (regime_raw or "").strip().lower()
    return [1.0 if r == bucket else 0.0 for bucket in _REGIME_BUCKETS]


def _symbol_one_hot(symbol: str) -> list[float]:
    base = (symbol or "").upper().replace("/", "")
    if base.endswith("USDT") and len(base) > 4:
        base = base[:-4]
    return [1.0 if base == b else 0.0 for b in _TOP4_BASES]


def meta_feature_names() -> list[str]:
    names = list(_NUMERIC_KEYS)
    names += [f"regime_{b}" for b in _REGIME_BUCKETS]
    names += [f"day_route_regime_{b}" for b in _REGIME_BUCKETS]
    names += [f"symbol_{b}" for b in _TOP4_BASES]
    return names


def build_meta_features(decision_data: dict[str, Any], symbol: str) -> list[float]:
    dd = decision_data or {}
    feats = [_safe_float(dd.get(k)) for k in _NUMERIC_KEYS]
    feats += _regime_one_hot(str(dd.get("regime") or dd.get("regime_label") or ""))
    feats += _regime_one_hot(str(dd.get("day_route_regime") or ""))
    feats += _symbol_one_hot(symbol)
    return feats


def _meta_model_path() -> Path:
    return META_MODEL_DIR / META_MODEL_FILENAME


def train_meta_label_model(*, db_path: str = DATABASE_PATH, strategy_id: str = "day") -> dict[str, Any]:
    """Pool BUY-decision snapshots across every symbol for this strategy, label
    each with the SAME self-supervised forward-return economics the primary
    model's own labels use (required_edge_pct_for_strategy — net of the
    row's own cost_estimate_pct), and fit a small LogisticRegression. Safe with
    thin data: returns a 'skipped_*' status and writes nothing on disk rather
    than risk shipping an unreliable meta-model."""
    req_edge = required_edge_pct_for_strategy(strategy_id)
    rows: list[tuple[str, dict[str, Any], float, float]] = []
    try:
        with sqlite3.connect(db_path, timeout=10) as conn:
            conn.row_factory = sqlite3.Row
            cur = conn.execute(
                """
                SELECT symbol, confidence, thesis_score, relative_volume, spread_pct,
                       cost_estimate_pct, regime, day_route_regime, chart_pattern_score,
                       model_disagreement, cross_sectional_rank_delta, setup_score,
                       execution_rank_delta, fwd_ret_1h
                FROM ai_candidate_snapshots
                WHERE strategy_id = ? AND decision = 'BUY' AND label_status = 'LABELED'
                      AND fwd_ret_1h IS NOT NULL
                ORDER BY epoch_ms DESC
                LIMIT 20000
                """,
                (strategy_id,),
            )
            for r in cur.fetchall():
                dd = {
                    "confidence": r["confidence"],
                    "thesis_score": r["thesis_score"],
                    "relative_volume": r["relative_volume"],
                    "spread_pct": r["spread_pct"],
                    "cost_estimate_pct": r["cost_estimate_pct"],
                    "regime": r["regime"],
                    "day_route_regime": r["day_route_regime"],
                    "chart_pattern_score": r["chart_pattern_score"],
                    "model_disagreement": r["model_disagreement"],
                    "cross_sectional_rank_delta": r["cross_sectional_rank_delta"],
                    "setup_score": r["setup_score"],
                    "execution_rank_delta": r["execution_rank_delta"],
                }
                cost = _safe_float(r["cost_estimate_pct"], 0.0006)
                rows.append((str(r["symbol"] or ""), dd, _safe_float(r["fwd_ret_1h"]), cost))
    except Exception as exc:
        logger.debug("META_LABEL_TRAIN_QUERY_FAILED strategy=%s: %s", strategy_id, exc)
        return {"status": "query_failed", "error": str(exc)[:200]}

    if len(rows) < MIN_TRAIN_SAMPLES:
        return {"status": "skipped_insufficient_data", "n": len(rows), "min_required": MIN_TRAIN_SAMPLES}

    X = np.asarray([build_meta_features(dd, sym) for sym, dd, _, _ in rows], dtype=np.float64)
    net_ret = np.asarray([fwd - cost for _, _, fwd, cost in rows], dtype=np.float64)
    y = (net_ret > req_edge).astype(int)

    if len(np.unique(y)) < 2:
        return {"status": "skipped_single_class", "n": len(rows), "positive_rate": float(np.mean(y))}

    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import train_test_split
        from sklearn.preprocessing import StandardScaler

        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)
        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_val_s = scaler.transform(X_val)

        model = LogisticRegression(max_iter=1000, class_weight="balanced", C=1.0)
        model.fit(X_train_s, y_train)
        val_acc = float(model.score(X_val_s, y_val))
    except Exception as exc:
        logger.debug("META_LABEL_FIT_FAILED strategy=%s: %s", strategy_id, exc)
        return {"status": "fit_failed", "error": str(exc)[:200]}

    artifact = {
        "model": model,
        "scaler": scaler,
        "feature_names": meta_feature_names(),
        "strategy_id": strategy_id,
        "n_train": len(X_train),
        "n_val": len(X_val),
        "val_accuracy": val_acc,
        "positive_rate": float(np.mean(y)),
        "required_edge_pct": req_edge,
        "trained_at": time.time(),
    }
    try:
        path = _meta_model_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as f:
            pickle.dump(artifact, f)
    except Exception as exc:
        logger.debug("META_LABEL_ARTIFACT_WRITE_FAILED strategy=%s: %s", strategy_id, exc)
        return {"status": "write_failed", "error": str(exc)[:200]}

    global _cached_artifact
    _cached_artifact = (0.0, None)  # force reload on next score_meta_trust call

    logger.info(
        "META_LABEL_MODEL_TRAINED: strategy=%s n=%d val_acc=%.4f positive_rate=%.4f",
        strategy_id,
        len(rows),
        val_acc,
        float(np.mean(y)),
    )
    return {"status": "trained", "n": len(rows), "val_accuracy": val_acc, "positive_rate": float(np.mean(y))}


def _load_meta_artifact() -> dict[str, Any] | None:
    global _cached_artifact
    now = time.time()
    ts, cached = _cached_artifact
    if cached is not None and (now - ts) < _CACHE_TTL_SEC:
        return cached
    path = _meta_model_path()
    if not path.exists():
        _cached_artifact = (now, None)
        return None
    try:
        with path.open("rb") as f:
            artifact = pickle.load(f)
        _cached_artifact = (now, artifact)
        return artifact
    except Exception as exc:
        logger.debug("META_LABEL_ARTIFACT_LOAD_FAILED: %s", exc)
        _cached_artifact = (now, None)
        return None


def score_meta_trust(decision_data: dict[str, Any], symbol: str) -> tuple[float, dict[str, Any]]:
    """Bounded [0.5, 1.0] confidence MULTIPLIER. 1.0 (no change) when no trained
    meta-model exists yet — same neutral-until-proven contract as every other
    validation signal added this session. Floor of 0.5 means meta-labeling can
    filter a marginal BUY down to half its confidence but can never veto it
    outright or boost it above what the primary model already said — sizing
    and thesis gates downstream keep their own independent authority."""
    artifact = _load_meta_artifact()
    if artifact is None:
        return 1.0, {"source": "untrained"}
    try:
        feats = np.asarray([build_meta_features(decision_data, symbol)], dtype=np.float64)
        feats_s = artifact["scaler"].transform(feats)
        model = artifact["model"]
        classes = list(model.classes_)
        if 1 not in classes:
            return 1.0, {"source": "no_positive_class"}
        col = classes.index(1)
        trust_prob = float(model.predict_proba(feats_s)[0][col])
    except Exception as exc:
        logger.debug("META_LABEL_SCORE_FAILED symbol=%s: %s", symbol, exc)
        return 1.0, {"source": "score_failed"}

    multiplier = 0.5 + 0.5 * max(0.0, min(1.0, trust_prob))
    detail = {
        "source": "meta_model",
        "trust_prob": round(trust_prob, 4),
        "multiplier": round(multiplier, 4),
        "trained_at": artifact.get("trained_at"),
        "val_accuracy": artifact.get("val_accuracy"),
    }
    return multiplier, detail


__all__ = [
    "MIN_TRAIN_SAMPLES",
    "build_meta_features",
    "meta_feature_names",
    "score_meta_trust",
    "train_meta_label_model",
]
