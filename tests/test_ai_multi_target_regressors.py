"""Item p10: multi-target ML regression heads (expected return/MFE/MAE/time-to-target)."""

from __future__ import annotations

import json
import random
import sqlite3

import pytest

from backend.services import ai_multi_target_regressors as mtr
from backend.services.ai_canonical_storage import ensure_ai_canonical_tables

FEATURE_DIM = 8  # small dim for fast tests; module is dim-agnostic


def _insert_outcome_row(db_path, *, symbol, strategy_id, idx, features, net_pnl_pct, mfe, mae, hold_seconds):
    with sqlite3.connect(db_path) as conn:
        cols = {r[1] for r in conn.execute("PRAGMA table_info(ai_outcome_training_rows)")}
        row = {
            "symbol": symbol,
            "opened_at_utc": f"2026-08-01T00:{idx:02d}:00Z",
            "closed_at_utc": f"2026-08-01T00:{idx:02d}:30Z",
            "hold_seconds": hold_seconds,
            "outcome_label": 1 if net_pnl_pct > 0 else 0,
            "outcome_class": "WIN" if net_pnl_pct > 0 else "LOSS",
            "features_json": json.dumps(features),
            "strategy_id": strategy_id,
            "net_pnl_pct": net_pnl_pct,
            "max_favorable_excursion": mfe,
            "max_adverse_excursion": mae,
            "ingested_at_utc": f"2026-08-01T00:{idx:02d}:31Z",
        }
        use = {k: v for k, v in row.items() if k in cols}
        conn.execute(
            f"INSERT INTO ai_outcome_training_rows ({', '.join(use)}) VALUES ({', '.join('?' for _ in use)})",
            list(use.values()),
        )
        conn.commit()


@pytest.fixture()
def db_path(tmp_path):
    p = str(tmp_path / "mtr.db")
    ensure_ai_canonical_tables(p)
    return p


def _seed_rows(db_path, *, symbol="BTCUSDT", strategy_id="day", n=60):
    rng = random.Random(42)
    for i in range(n):
        features = [rng.random() for _ in range(FEATURE_DIM)]
        # Make target correlated with feature[0] so the regressor has a real signal to learn.
        net_pnl = (features[0] - 0.5) * 0.05
        mfe = abs(net_pnl) + 0.002
        mae = 0.01 - abs(net_pnl) * 0.3
        hold_seconds = 300 + features[1] * 3000
        _insert_outcome_row(db_path, symbol=symbol, strategy_id=strategy_id, idx=i, features=features, net_pnl_pct=net_pnl, mfe=mfe, mae=mae, hold_seconds=hold_seconds)


def test_train_reports_insufficient_rows_when_too_few(db_path):
    _seed_rows(db_path, n=5)
    result = mtr.train_multi_target_regressors("day", "BTCUSDT", db_path=db_path, feature_dim=FEATURE_DIM)
    assert result.trained is False
    assert result.reason == "insufficient_rows"


def test_train_succeeds_with_enough_rows_and_persists_artifact(db_path, monkeypatch, tmp_path):
    monkeypatch.setattr(mtr, "_MODEL_DIR", tmp_path / "multi_target_models")
    _seed_rows(db_path, n=60)
    result = mtr.train_multi_target_regressors("day", "BTCUSDT", db_path=db_path, feature_dim=FEATURE_DIM)
    assert result.trained is True
    assert result.n_rows == 60
    assert set(result.val_mae_by_target.keys()) == {"expected_return", "expected_mfe", "expected_mae", "expected_time_to_target_sec"}
    assert mtr._artifact_path("day", "BTCUSDT").exists()


def test_predict_unavailable_without_trained_artifact(tmp_path, monkeypatch):
    monkeypatch.setattr(mtr, "_MODEL_DIR", tmp_path / "empty_models")
    mtr._ARTIFACT_CACHE.clear()
    result = mtr.predict_multi_target("day", "ETHUSDT", [0.1] * FEATURE_DIM)
    assert result.available is False
    assert result.degraded_reason == "no_trained_artifact"


def test_predict_returns_all_targets_after_training(db_path, monkeypatch, tmp_path):
    monkeypatch.setattr(mtr, "_MODEL_DIR", tmp_path / "multi_target_models2")
    mtr._ARTIFACT_CACHE.clear()
    _seed_rows(db_path, n=60)
    train_result = mtr.train_multi_target_regressors("day", "BTCUSDT", db_path=db_path, feature_dim=FEATURE_DIM)
    assert train_result.trained is True

    pred = mtr.predict_multi_target("day", "BTCUSDT", [0.9] * FEATURE_DIM, cost_pct=0.001)
    assert pred.available is True
    assert pred.expected_return is not None
    assert pred.expected_mfe is not None
    assert pred.expected_mae is not None
    assert pred.expected_time_to_target_sec is not None
    assert pred.net_ev_estimate == pytest.approx(pred.expected_return - 0.001)


def test_predict_rejects_mismatched_feature_dim(db_path, monkeypatch, tmp_path):
    monkeypatch.setattr(mtr, "_MODEL_DIR", tmp_path / "multi_target_models3")
    mtr._ARTIFACT_CACHE.clear()
    _seed_rows(db_path, n=60)
    mtr.train_multi_target_regressors("day", "BTCUSDT", db_path=db_path, feature_dim=FEATURE_DIM)
    pred = mtr.predict_multi_target("day", "BTCUSDT", [0.1] * (FEATURE_DIM + 3))
    assert pred.available is False
    assert pred.degraded_reason == "feature_dim_mismatch"


def test_predict_from_latest_inference_uses_logged_feature_vector(db_path, monkeypatch, tmp_path):
    monkeypatch.setattr(mtr, "_MODEL_DIR", tmp_path / "multi_target_models4")
    mtr._ARTIFACT_CACHE.clear()
    _seed_rows(db_path, n=60)
    mtr.train_multi_target_regressors("day", "BTCUSDT", db_path=db_path, feature_dim=FEATURE_DIM)

    features = [0.4] * FEATURE_DIM
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            """
            INSERT INTO ai_inference_log (strategy_id, symbol, ts_utc, features_json, feature_dim)
            VALUES ('day', 'BTCUSDT', '2026-08-01T00:00:00Z', ?, ?)
            """,
            (json.dumps(features), FEATURE_DIM),
        )
        conn.commit()

    pred = mtr.predict_multi_target_from_latest_inference("day", "BTCUSDT", db_path=db_path)
    assert pred.available is True
    assert pred.expected_return is not None


def test_predict_from_latest_inference_degrades_without_inference_row(db_path, monkeypatch, tmp_path):
    monkeypatch.setattr(mtr, "_MODEL_DIR", tmp_path / "multi_target_models5")
    mtr._ARTIFACT_CACHE.clear()
    pred = mtr.predict_multi_target_from_latest_inference("day", "XRPUSDT", db_path=db_path)
    assert pred.available is False
    assert pred.degraded_reason == "no_recent_inference_row"


def test_disabled_via_env(db_path, monkeypatch):
    monkeypatch.setenv("MULTI_TARGET_ML_ENABLED", "false")
    result = mtr.train_multi_target_regressors("day", "BTCUSDT", db_path=db_path, feature_dim=FEATURE_DIM)
    assert result.trained is False
    assert result.reason == "disabled"
    pred = mtr.predict_multi_target("day", "BTCUSDT", [0.1] * FEATURE_DIM)
    assert pred.available is False
    assert pred.degraded_reason == "disabled"
