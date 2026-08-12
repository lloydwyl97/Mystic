"""
Multi-target ML regression heads (item p10).

Mystic's live RF/GBM models are binary classifiers only (BUY-worthy vs not) —
see ``ai_training_pipeline.py``. This module adds sibling regression heads,
trained on the exact same feature vectors already stored per closed trade in
``ai_outcome_training_rows.features_json``, predicting four continuous
targets from real historical outcomes:

  - expected_return   <- net_pnl_pct
  - expected_mfe      <- max_favorable_excursion
  - expected_mae      <- max_adverse_excursion
  - expected_time_to_target_sec <- hold_seconds (proxy: how long similar
    setups took to resolve; there is no explicit "time to hit target" label
    in the outcome schema, so hold_seconds of trades that closed as WINs is
    the closest honest proxy available and is labeled as such)

Each is an independent small RandomForestRegressor, trained per
(strategy_id, symbol) — mirroring the per-symbol classifier convention
already used in ``ai_training_pipeline.py`` (coin-specific separation, item
p24) — from a chronological 80/20 split (same convention as the classifier;
walk-forward purge/embargo is item p13's separate deliverable and is layered
on top for reporting, not required for this module to function).

These heads are diagnostic/ranking evidence, exactly like every other
p10-p23 addition — they are never a hard entry gate. `net_ev_estimate`
combines expected_return with the model's own historical hit-rate at
inference time, and is exposed for ranking/sizing consumption (e.g. as an
additional ai_context field), never as a new veto.
"""

from __future__ import annotations

import json
import logging
import os
import pickle
import sqlite3
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from backend.database_schema import DATABASE_PATH

logger = logging.getLogger(__name__)

MIN_TRAINING_ROWS = int(os.getenv("MULTI_TARGET_ML_MIN_ROWS", "40") or "40")
_TARGETS: tuple[str, ...] = ("expected_return", "expected_mfe", "expected_mae", "expected_time_to_target_sec")

_MODEL_DIR = Path(os.getenv("MULTI_TARGET_ML_MODEL_DIR", "models/multi_target"))


def _enabled() -> bool:
    return str(os.getenv("MULTI_TARGET_ML_ENABLED", "true")).strip().lower() in ("1", "true", "yes", "on")


def _artifact_path(strategy_id: str, symbol: str) -> Path:
    return _MODEL_DIR / f"{strategy_id.lower()}_{symbol.upper()}_multi_target.pkl"


@dataclass(frozen=True)
class MultiTargetTrainResult:
    strategy_id: str
    symbol: str
    trained: bool
    n_rows: int
    n_train: int
    n_val: int
    val_mae_by_target: dict[str, float] = field(default_factory=dict)
    reason: str | None = None


def _load_rows(strategy_id: str, symbol: str, db_path: str, limit: int) -> list[dict[str, Any]]:
    try:
        from backend.services.ai_canonical_storage import read_recent_outcome_training_rows

        return read_recent_outcome_training_rows(symbol=symbol, strategy_id=strategy_id, limit=limit, db_path=db_path)
    except Exception as exc:
        logger.debug("MULTI_TARGET_ML_ROWS_FAILED strategy=%s symbol=%s: %s", strategy_id, symbol, exc)
        return []


def _row_to_xy(row: dict[str, Any], feature_dim: int) -> tuple[list[float], dict[str, float]] | None:
    raw_features = row.get("features_json")
    if not raw_features:
        return None
    try:
        x = json.loads(raw_features)
    except (TypeError, ValueError):
        return None
    if not isinstance(x, list) or len(x) != feature_dim:
        return None
    try:
        x = [float(v) for v in x]
    except (TypeError, ValueError):
        return None

    def _f(key: str) -> float | None:
        v = row.get(key)
        if v is None:
            return None
        try:
            return float(v)
        except (TypeError, ValueError):
            return None

    net_pnl = _f("net_pnl_pct")
    mfe = _f("max_favorable_excursion")
    mae = _f("max_adverse_excursion")
    hold_s = _f("hold_seconds")
    if net_pnl is None or mfe is None or mae is None or hold_s is None:
        return None
    return x, {
        "expected_return": net_pnl,
        "expected_mfe": mfe,
        "expected_mae": abs(mae),
        "expected_time_to_target_sec": hold_s,
    }


def train_multi_target_regressors(
    strategy_id: str,
    symbol: str,
    *,
    db_path: str = DATABASE_PATH,
    feature_dim: int = 145,
    lookback_limit: int = 3000,
) -> MultiTargetTrainResult:
    """Fit 4 independent RandomForestRegressor heads for (strategy_id, symbol)
    from real closed-trade rows and persist the artifact. Chronological
    80/20 split — same convention as the live classifier."""
    if not _enabled():
        return MultiTargetTrainResult(strategy_id, symbol, trained=False, n_rows=0, n_train=0, n_val=0, reason="disabled")

    rows = _load_rows(strategy_id, symbol, db_path, lookback_limit)
    # read_recent_outcome_training_rows orders DESC by id; restore chronological order.
    rows = list(reversed(rows))

    pairs: list[tuple[list[float], dict[str, float]]] = []
    for row in rows:
        parsed = _row_to_xy(row, feature_dim)
        if parsed is not None:
            pairs.append(parsed)

    n = len(pairs)
    if n < MIN_TRAINING_ROWS:
        return MultiTargetTrainResult(strategy_id, symbol, trained=False, n_rows=n, n_train=0, n_val=0, reason="insufficient_rows")

    try:
        from sklearn.ensemble import RandomForestRegressor
    except Exception as exc:  # pragma: no cover - sklearn always present in this repo's venv
        logger.warning("MULTI_TARGET_ML_SKLEARN_MISSING: %s", exc)
        return MultiTargetTrainResult(strategy_id, symbol, trained=False, n_rows=n, n_train=0, n_val=0, reason="sklearn_unavailable")

    split_idx = int(n * 0.8)
    train_pairs, val_pairs = pairs[:split_idx], pairs[split_idx:]
    if len(train_pairs) < 10 or not val_pairs:
        return MultiTargetTrainResult(strategy_id, symbol, trained=False, n_rows=n, n_train=len(train_pairs), n_val=len(val_pairs), reason="insufficient_split")

    x_train = [p[0] for p in train_pairs]
    x_val = [p[0] for p in val_pairs]

    models: dict[str, Any] = {}
    val_mae: dict[str, float] = {}
    for target in _TARGETS:
        y_train = [p[1][target] for p in train_pairs]
        y_val = [p[1][target] for p in val_pairs]
        model = RandomForestRegressor(n_estimators=40, max_depth=8, min_samples_split=5, random_state=42, n_jobs=-1)
        model.fit(x_train, y_train)
        preds = model.predict(x_val)
        mae = sum(abs(p - y) for p, y in zip(preds, y_val, strict=False)) / max(1, len(y_val))
        models[target] = model
        val_mae[target] = float(mae)

    artifact = {
        "strategy_id": strategy_id.lower(),
        "symbol": symbol.upper(),
        "feature_dim": feature_dim,
        "models": models,
        "val_mae_by_target": val_mae,
        "n_rows": n,
        "trained_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    _MODEL_DIR.mkdir(parents=True, exist_ok=True)
    _artifact_path(strategy_id, symbol).write_bytes(pickle.dumps(artifact))

    return MultiTargetTrainResult(
        strategy_id,
        symbol,
        trained=True,
        n_rows=n,
        n_train=len(train_pairs),
        n_val=len(val_pairs),
        val_mae_by_target=val_mae,
    )


@dataclass(frozen=True)
class MultiTargetPrediction:
    strategy_id: str
    symbol: str
    available: bool
    expected_return: float | None = None
    expected_mfe: float | None = None
    expected_mae: float | None = None
    expected_time_to_target_sec: float | None = None
    net_ev_estimate: float | None = None
    trained_at: str | None = None
    n_rows: int = 0
    degraded_reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "strategy_id": self.strategy_id,
            "symbol": self.symbol,
            "available": self.available,
            "expected_return": round(self.expected_return, 6) if self.expected_return is not None else None,
            "expected_mfe": round(self.expected_mfe, 6) if self.expected_mfe is not None else None,
            "expected_mae": round(self.expected_mae, 6) if self.expected_mae is not None else None,
            "expected_time_to_target_sec": round(self.expected_time_to_target_sec, 1) if self.expected_time_to_target_sec is not None else None,
            "net_ev_estimate": round(self.net_ev_estimate, 6) if self.net_ev_estimate is not None else None,
            "trained_at": self.trained_at,
            "n_rows": self.n_rows,
            "degraded_reason": self.degraded_reason,
        }


_ARTIFACT_CACHE: dict[str, tuple[dict[str, Any], float]] = {}
_ARTIFACT_CACHE_TTL_SEC = 300.0


def _load_artifact(strategy_id: str, symbol: str) -> dict[str, Any] | None:
    key = f"{strategy_id.lower()}:{symbol.upper()}"
    now = time.time()
    cached = _ARTIFACT_CACHE.get(key)
    if cached is not None and cached[1] > now:
        return cached[0]
    path = _artifact_path(strategy_id, symbol)
    if not path.exists():
        return None
    try:
        artifact = pickle.loads(path.read_bytes())
    except Exception as exc:
        logger.debug("MULTI_TARGET_ML_ARTIFACT_LOAD_FAILED %s: %s", path, exc)
        return None
    _ARTIFACT_CACHE[key] = (artifact, now + _ARTIFACT_CACHE_TTL_SEC)
    return artifact


def predict_multi_target(
    strategy_id: str,
    symbol: str,
    features: list[float],
    *,
    cost_pct: float = 0.0015,
) -> MultiTargetPrediction:
    """Live inference — never raises; honest degraded state on any failure."""
    if not _enabled():
        return MultiTargetPrediction(strategy_id.lower(), symbol.upper(), available=False, degraded_reason="disabled")

    artifact = _load_artifact(strategy_id, symbol)
    if artifact is None:
        return MultiTargetPrediction(strategy_id.lower(), symbol.upper(), available=False, degraded_reason="no_trained_artifact")

    expected_dim = int(artifact.get("feature_dim") or 145)
    if not isinstance(features, list) or len(features) != expected_dim:
        return MultiTargetPrediction(strategy_id.lower(), symbol.upper(), available=False, degraded_reason="feature_dim_mismatch")

    try:
        models = artifact["models"]
        x = [features]
        expected_return = float(models["expected_return"].predict(x)[0])
        expected_mfe = float(models["expected_mfe"].predict(x)[0])
        expected_mae = float(models["expected_mae"].predict(x)[0])
        expected_time = float(models["expected_time_to_target_sec"].predict(x)[0])
    except Exception as exc:
        logger.debug("MULTI_TARGET_ML_PREDICT_FAILED strategy=%s symbol=%s: %s", strategy_id, symbol, exc)
        return MultiTargetPrediction(strategy_id.lower(), symbol.upper(), available=False, degraded_reason="predict_failed")

    net_ev_estimate = expected_return - cost_pct

    return MultiTargetPrediction(
        strategy_id=strategy_id.lower(),
        symbol=symbol.upper(),
        available=True,
        expected_return=expected_return,
        expected_mfe=expected_mfe,
        expected_mae=expected_mae,
        expected_time_to_target_sec=expected_time,
        net_ev_estimate=net_ev_estimate,
        trained_at=artifact.get("trained_at"),
        n_rows=int(artifact.get("n_rows") or 0),
    )


def predict_multi_target_from_latest_inference(
    strategy_id: str,
    symbol: str,
    *,
    db_path: str = DATABASE_PATH,
    cost_pct: float = 0.0015,
) -> MultiTargetPrediction:
    """Convenience live wiring: reuses the exact feature vector already
    logged for this symbol's most recent live decision
    (``ai_inference_log.features_json`` — "exact model input vector used")
    instead of re-deriving features, so this can be safely called from a
    read-only diagnostic loop (e.g. ai_market_context.py) without touching
    the buy-decision code path at all."""
    if not _enabled():
        return MultiTargetPrediction(strategy_id.lower(), symbol.upper(), available=False, degraded_reason="disabled")
    try:
        with sqlite3.connect(db_path, timeout=5.0) as conn:
            conn.row_factory = sqlite3.Row
            row = conn.execute(
                """
                SELECT features_json, feature_dim
                FROM ai_inference_log
                WHERE symbol = ? AND LOWER(COALESCE(strategy_id, 'day')) = LOWER(?)
                ORDER BY id DESC LIMIT 1
                """,
                (symbol, strategy_id),
            ).fetchone()
    except Exception as exc:
        logger.debug("MULTI_TARGET_ML_LATEST_INFERENCE_FETCH_FAILED symbol=%s: %s", symbol, exc)
        row = None

    if row is None or not row["features_json"]:
        return MultiTargetPrediction(strategy_id.lower(), symbol.upper(), available=False, degraded_reason="no_recent_inference_row")

    try:
        features = json.loads(row["features_json"])
    except (TypeError, ValueError):
        return MultiTargetPrediction(strategy_id.lower(), symbol.upper(), available=False, degraded_reason="unparseable_features")

    return predict_multi_target(strategy_id, symbol, features, cost_pct=cost_pct)


__all__ = [
    "MultiTargetPrediction",
    "MultiTargetTrainResult",
    "predict_multi_target",
    "predict_multi_target_from_latest_inference",
    "train_multi_target_regressors",
]
