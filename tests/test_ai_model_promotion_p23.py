"""Item p23: accuracy is diagnostic only for model promotion. The gate is the
paired net-after-cost comparison on the shared causal holdout, with an
uncertainty penalty, and a bad-trade-rate that may not worsen."""

from __future__ import annotations

import pickle
from pathlib import Path
from unittest.mock import patch

import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

from backend.services import ai_model_registry as registry
from backend.services.ai_canonical_storage import ensure_ai_canonical_tables
from backend.services.ai_model_promotion import register_candidate_and_maybe_promote
from backend.services.ai_model_promotion_decision import compare_paired
from backend.services.ai_model_promotion_holdout import artifact_predictions

DIM = 145


def _write_artifact(path: Path, threshold: float) -> Path:
    x0 = np.array([-5.0, -1.0, threshold - 0.01, threshold + 0.01, 50.0])
    X = np.zeros((len(x0), DIM))
    X[:, 0] = x0
    model = DecisionTreeClassifier(max_depth=1, random_state=0).fit(X, (x0 > threshold).astype(int))
    scaler = StandardScaler(with_mean=False, with_std=False).fit(np.zeros((2, DIM)))
    path.parent.mkdir(parents=True, exist_ok=True)
    art = {"model": model, "scaler": scaler, "feature_version": 5, "feature_dim": DIM, "live_strategy_id": "day", "accuracy": 0.6, "training_data_end": "2026-01-01T00:00:00+00:00"}
    path.write_bytes(pickle.dumps(art))
    return path


def _run(tmp_path: Path, nets: np.ndarray, labels: np.ndarray):
    active = _write_artifact(tmp_path / "models" / "active" / "day" / "BTCUSDT_direction.pkl", 1e6)
    cand = _write_artifact(tmp_path / "cand" / "c.pkl", 0.5)
    registry.set_promotion_enabled("day", True, "test", registry.registry_root(tmp_path / "models" / "active"))
    n = len(nets)
    X = np.zeros((n, DIM))
    X[:, 0] = 1.0
    X[n // 2 :, 0] = -1.0
    gbs = np.array([""] * n, dtype=object)
    cp, ap = artifact_predictions(cand, X), artifact_predictions(active, X)
    metrics = {
        "holdout_sample_count": n,
        "candidate_holdout": {"buy_signal_count": int(np.sum(cp == 1)), "accuracy": float(np.mean(cp == labels))},
        "active_holdout": {"accuracy": float(np.mean(ap == labels))},
        "holdout_window": {"first_opened_at": "2026-02-01T00:00:00+00:00", "n": n},
        "paired_decision": compare_paired(cp, ap, nets, gbs, labels),
    }
    db = tmp_path / "t.db"
    ensure_ai_canonical_tables(str(db))
    with patch("backend.services.ai_model_promotion.evaluate_signal_hash_artifact_contract", return_value=(True, None, {})):
        return register_candidate_and_maybe_promote(strategy_id="day", symbol="BTCUSDT", candidate_path=cand, active_path=active, validation_metrics=metrics, db_path=str(db)), metrics


def test_lower_accuracy_promotes_when_economics_are_better(tmp_path):
    nets = np.array([0.02] * 20 + [-0.001] * 20)
    labels = np.zeros(40, dtype=int)
    (promoted, reason), metrics = _run(tmp_path, nets, labels)
    assert metrics["candidate_holdout"]["accuracy"] < metrics["active_holdout"]["accuracy"]
    assert promoted is True, reason


def test_accuracy_reported_but_worse_economics_keeps_incumbent(tmp_path):
    nets = np.array([-0.02] * 20 + [0.01] * 20)
    labels = np.array([1] * 20 + [0] * 20)
    (promoted, reason), metrics = _run(tmp_path, nets, labels)
    assert promoted is False
    assert reason == "keep_incumbent:INFERIOR"
    assert metrics["paired_decision"]["candidate"]["accuracy"] > metrics["paired_decision"]["incumbent"]["accuracy"]
