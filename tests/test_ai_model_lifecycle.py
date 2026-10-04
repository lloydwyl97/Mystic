"""Regression coverage for AI model promotion and rollback lifecycle.

Restored from test_p2_learning_lifecycle.py (deleted in commit 87ef12e).
These tests cover the live AI model promotion pipeline (ai_model_promotion.py,
ai_canonical_storage.py, ai_training_pipeline.py) which is NOT part of the
retired paper runner.  They were incorrectly classified as paper-only.

The test test_scalp_learning_resolves_nested_rank_score is NOT restored because
it tested binance_scalp/paper_engine.py which was deleted in the same commit.
"""

from __future__ import annotations

import json
import pickle
import sqlite3
import tempfile
from pathlib import Path
from unittest.mock import patch

from backend.services.ai_canonical_storage import ensure_ai_canonical_tables
from backend.services.ai_model_promotion import maybe_rollback_underperforming_model, register_candidate_and_maybe_promote

REPO = Path(__file__).resolve().parents[1]


def _write_artifact(path: Path, *, accuracy: float = 0.6, feature_version: int = 5) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "accuracy": accuracy,
        "feature_version": feature_version,
        "feature_dim": 145,
        "live_strategy_id": "day",
        "model": None,
    }
    path.write_bytes(pickle.dumps(payload))


def _servable(path: Path, threshold: float) -> Path:
    import numpy as np
    from sklearn.preprocessing import StandardScaler
    from sklearn.tree import DecisionTreeClassifier

    x0 = np.array([-5.0, -1.0, threshold - 0.01, threshold + 0.01, 50.0])
    X = np.zeros((len(x0), 145))
    X[:, 0] = x0
    art = {
        "model": DecisionTreeClassifier(max_depth=1, random_state=0).fit(X, (x0 > threshold).astype(int)),
        "scaler": StandardScaler(with_mean=False, with_std=False).fit(np.zeros((2, 145))),
        "feature_version": 5,
        "feature_dim": 145,
        "live_strategy_id": "day",
        "accuracy": threshold,
        "training_data_end": "2026-01-01T00:00:00+00:00",
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(pickle.dumps(art))
    return path


def test_promote_archives_prior_active_and_records_registry_path(tmp_path: Path):
    from backend.services import ai_model_registry as registry
    from backend.services.live_strategy_contracts import per_coin_artifact_file

    db = tmp_path / "lifecycle.db"
    ensure_ai_canonical_tables(str(db))
    active_dir = tmp_path / "models" / "active"
    active = _servable(per_coin_artifact_file(active_dir, "day", "BTCUSDT"), 1e6)
    cand1 = _servable(tmp_path / "versions" / "day_BTCUSDT_1.pkl", 0.5)
    cand2 = _servable(tmp_path / "versions" / "day_BTCUSDT_2.pkl", 0.7)
    registry.set_promotion_enabled("day", True, "test", registry.registry_root(active_dir))
    metrics = {
        "holdout_sample_count": 40,
        "candidate_holdout": {"buy_signal_count": 12},
        "holdout_window": {"first_opened_at": "2026-02-01T00:00:00+00:00", "n": 40},
        "paired_decision": {"verdict": "PROMOTE", "promote": True, "n": 40},
    }
    with patch("backend.services.ai_model_promotion.evaluate_signal_hash_artifact_contract", return_value=(True, None, {})):
        ok1, _ = register_candidate_and_maybe_promote(strategy_id="day", symbol="BTCUSDT", candidate_path=cand1, active_path=active, validation_metrics=metrics, db_path=str(db))
        ok2, _ = register_candidate_and_maybe_promote(strategy_id="day", symbol="BTCUSDT", candidate_path=cand2, active_path=active, validation_metrics=metrics, db_path=str(db))
    assert ok1 is True and ok2 is True
    sha1, sha2 = registry.sha256_file(cand1), registry.sha256_file(cand2)
    with sqlite3.connect(db) as conn:
        actives = conn.execute("SELECT path FROM ai_model_versions WHERE status='active'").fetchall()
        archived = conn.execute("SELECT path FROM ai_model_versions WHERE status='archived'").fetchall()
    assert len(actives) == 1 and sha2 in actives[0][0]
    assert any(sha1 in row[0] and Path(row[0]).exists() for row in archived)
    assert registry.read_pointer("day", "BTCUSDT", "PREVIOUS", registry.registry_root(active_dir))["sha256"] == sha1


def test_rollback_restores_registry_previous_artifact(tmp_path: Path):
    from backend.services import ai_model_registry as registry
    from backend.services.live_strategy_contracts import per_coin_artifact_file

    db = tmp_path / "rollback.db"
    ensure_ai_canonical_tables(str(db))
    active_dir = tmp_path / "models" / "active"
    root = registry.registry_root(active_dir)
    active = _servable(per_coin_artifact_file(active_dir, "day", "BTCUSDT"), 0.66)
    prev_bytes = active.read_bytes()
    registry.set_promotion_enabled("day", True, "test", root)
    assert registry.promote_atomic("day", "BTCUSDT", _servable(tmp_path / "versions" / "cur.pkl", 0.4), active, root=root)[0]
    _seed_losses(db, "2099-01-01", version=_active_version(active_dir))
    ok, reason = maybe_rollback_underperforming_model(strategy_id="day", symbol="BTCUSDT", db_path=str(db), active_dir=active_dir)
    assert (ok, reason) == (True, "rollback_executed")
    assert active.read_bytes() == prev_bytes
    assert maybe_rollback_underperforming_model(strategy_id="day", symbol="BTCUSDT", db_path=str(db), active_dir=active_dir)[1] in ("no_previous_model", "no_attributable_live_outcomes")


def test_fail_open_fallback_removed_from_pipeline():
    src = (REPO / "backend/ai_training_pipeline.py").read_text()
    assert "fallback_direct_write" not in src
    assert "MODEL_PROMOTION_ERROR" in src


def test_prune_retires_registry_candidates():
    src = (REPO / "backend/ai_training_pipeline.py").read_text()
    assert "status = 'retired'" in src
    assert "ai_model_versions" in src


def test_feature_version_no_longer_invents_five_on_missing():
    src = (REPO / "backend/services/portfolio_engine.py").read_text()
    assert 'ex_payload.get("feature_version") or 5' not in src
    assert 'signal.get("feature_version") or 5' not in src
    assert 'original_explain["feature_version"] = 5' not in src


def test_rollback_logger_bound():
    src = (REPO / "backend/services/ai_model_promotion.py").read_text()
    assert "logger = logging.getLogger(__name__)" in src


def _active_version(active_dir: Path) -> str:
    from backend.services import ai_model_registry as registry

    return str(registry.read_pointer("day", "BTCUSDT", "ACTIVE", registry.registry_root(active_dir)).get("version") or "")


def _seed_losses(db: Path, day: str, *, version: str = "", n: int = 25) -> None:
    attribution = {"ml_model_attribution": {"model_version": version, "attributable": True, "rank_contribution": 1.0}} if version else {}
    with sqlite3.connect(db) as conn:
        for i in range(n):
            conn.execute(
                "INSERT INTO ai_outcome_training_rows (symbol, opened_at_utc, closed_at_utc, strategy_id, net_pnl_pct, ingested_at_utc, score_components_json) "
                "VALUES ('BTC/USDT', ?, ?, 'day', -0.01, datetime('now'), ?)",
                (f"{day}T00:00:{i:02d}Z", f"{day}T01:00:{i:02d}Z", json.dumps(attribution)),
            )
        conn.commit()


def _promoted_pair(tmp_path: Path) -> tuple[Path, Path, Path]:
    from backend.services import ai_model_registry as registry
    from backend.services.live_strategy_contracts import per_coin_artifact_file

    db = tmp_path / "rb.db"
    ensure_ai_canonical_tables(str(db))
    active_dir = tmp_path / "models" / "active"
    root = registry.registry_root(active_dir)
    active = _servable(per_coin_artifact_file(active_dir, "day", "BTCUSDT"), 0.66)
    registry.set_promotion_enabled("day", True, "test", root)
    assert registry.promote_atomic("day", "BTCUSDT", _servable(tmp_path / "versions" / "cur.pkl", 0.4), active, root=root)[0]
    return db, active_dir, active


def test_rollback_ignores_outcomes_selected_by_a_previous_model(tmp_path: Path):
    db, active_dir, _active = _promoted_pair(tmp_path)
    _seed_losses(db, "2026-08-01", version=_active_version(active_dir))
    assert maybe_rollback_underperforming_model(strategy_id="day", symbol="BTCUSDT", db_path=str(db), active_dir=active_dir) == (False, "no_attributable_live_outcomes")


def test_rollback_never_reinstates_a_rolled_back_model(tmp_path: Path):
    from backend.services import ai_model_registry as registry

    db, active_dir, active = _promoted_pair(tmp_path)
    root = registry.registry_root(active_dir)
    assert registry.rollback_to_previous("day", "BTCUSDT", active, reason="test", root=root)[0]
    _seed_losses(db, "2099-01-01", version=_active_version(active_dir))
    assert maybe_rollback_underperforming_model(strategy_id="day", symbol="BTCUSDT", db_path=str(db), active_dir=active_dir) == (False, "no_previous_model")
