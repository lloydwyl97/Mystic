"""DAY model governance: versioned registry, disabled auto-promotion, atomic switch, causal holdout."""

from __future__ import annotations

import json
import pickle
import sqlite3
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

from backend.config.ai_day_htf_contract import FEATURE_VERSION_DAY_HTF
from backend.services import ai_model_registry as registry
from backend.services.ai_canonical_storage import ensure_ai_canonical_tables
from backend.services.ai_model_promotion import (
    holdout_is_causal_for,
    maybe_rollback_underperforming_model,
    register_candidate_and_maybe_promote,
)
from backend.services.ai_model_promotion_decision import compare_paired
from backend.services.ai_model_promotion_holdout import artifact_predictions, build_holdout_validation_metrics, holdout_window
from backend.services.day_feature_health import zero_learning_blocked_feature_dims
from backend.services.live_strategy_contracts import per_coin_artifact_file

REPO = Path(__file__).resolve().parents[1]
DIM = 145
SYM = "BTCUSDT"
EARLY = "2026-01-01T00:00:00+00:00"


def _artifact(threshold: float, *, training_data_end: str | None = EARLY, **extra) -> dict:
    """Model that predicts BUY iff feature 0 exceeds ``threshold``."""
    x0 = np.array([-5.0, -1.0, threshold - 0.01, threshold + 0.01, threshold + 1.0, 50.0])
    X = np.zeros((len(x0), DIM))
    X[:, 0] = x0
    y = (x0 > threshold).astype(int)
    model = DecisionTreeClassifier(max_depth=1, random_state=0).fit(X, y)
    scaler = StandardScaler(with_mean=False, with_std=False).fit(np.zeros((2, DIM)))
    art = {"model": model, "scaler": scaler, "feature_version": FEATURE_VERSION_DAY_HTF, "feature_dim": DIM, "live_strategy_id": "day", "accuracy": 0.5, **extra}
    if training_data_end is not None:
        art["training_data_end"] = training_data_end
    return art


def _write(path: Path, art: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(pickle.dumps(art))
    return path


@pytest.fixture
def models(tmp_path: Path):
    active_dir = tmp_path / "models" / "active"
    active = per_coin_artifact_file(active_dir, "day", SYM)
    _write(active, _artifact(1e6, trained_at=EARLY))  # incumbent: always HOLD
    db = tmp_path / "gov.db"
    ensure_ai_canonical_tables(str(db))
    return {"active": active, "active_dir": active_dir, "root": registry.registry_root(active_dir), "db": str(db), "tmp": tmp_path}


def _holdout(n_good: int, n_flat: int = 0) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    X = np.zeros((n_good + n_flat, DIM))
    X[:n_good, 0] = 1.0
    X[n_good:, 0] = -1.0
    nets = np.array([0.02] * n_good + [-0.001] * n_flat)
    gbs = np.array([""] * (n_good + n_flat), dtype=object)
    y = np.array([0] * (n_good + n_flat))
    return X, y, nets, gbs


def _metrics(cand: Path, active: Path, X, y, nets, gbs) -> dict:
    cp, ap = artifact_predictions(cand, X), artifact_predictions(active, X)
    return {
        "holdout_status": "OK",
        "holdout_sample_count": len(y),
        "candidate_holdout": {"buy_signal_count": int(np.sum(cp == 1))},
        "holdout_window": {"first_opened_at": "2026-02-01T00:00:00+00:00", "n": len(y)},
        "paired_decision": compare_paired(cp, ap, nets, gbs, y),
    }


def _register(models, cand: Path, metrics: dict):
    with patch("backend.services.ai_model_promotion.evaluate_signal_hash_artifact_contract", return_value=(True, None, {})):
        return register_candidate_and_maybe_promote(strategy_id="day", symbol=SYM, candidate_path=cand, active_path=models["active"], validation_metrics=metrics, db_path=models["db"])


def test_feature_zero_is_learnable():
    assert zero_learning_blocked_feature_dims([1.0] * DIM)[0] == 1.0


def test_retrain_without_promotion_keeps_active_unchanged(models):
    before = models["active"].read_bytes()
    cand = _write(models["tmp"] / "versions" / "c1.pkl", _artifact(0.5))
    X, y, nets, gbs = _holdout(30, 10)
    metrics = _metrics(cand, models["active"], X, y, nets, gbs)
    assert metrics["paired_decision"]["verdict"] == "PROMOTE"
    assert registry.promotion_enabled("day", models["root"]) == (False, "promotion_control_missing")
    ok, reason = _register(models, cand, metrics)
    assert (ok, reason) == (False, "would_promote_promotion_disabled")
    assert models["active"].read_bytes() == before
    assert registry.read_pointer("day", SYM, "ACTIVE", models["root"])["sha256"] == registry.sha256_file(models["active"])


def test_candidates_are_versioned_and_immutable(models):
    cand = _write(models["tmp"] / "versions" / "c1.pkl", _artifact(0.5))
    a = registry.register_artifact("day", SYM, cand, {"note": "first"}, models["root"])
    b = registry.register_artifact("day", SYM, cand, {"note": "second"}, models["root"])
    assert a["path"] == b["path"] and Path(a["path"]).exists()
    assert json.loads(Path(a["meta_path"]).read_text())["note"] == "first"
    other = registry.register_artifact("day", SYM, _write(models["tmp"] / "versions" / "c2.pkl", _artifact(0.7)), None, models["root"])
    assert other["path"] != a["path"]


def test_promotion_preserves_incumbent_and_rollback_artifact(models):
    registry.set_promotion_enabled("day", True, "test", models["root"])
    incumbent = registry.sha256_file(models["active"])
    cand = _write(models["tmp"] / "versions" / "c1.pkl", _artifact(0.5))
    X, y, nets, gbs = _holdout(30, 10)
    ok, reason = _register(models, cand, _metrics(cand, models["active"], X, y, nets, gbs))
    assert ok, reason
    assert registry.sha256_file(models["active"]) == registry.sha256_file(cand)
    prev = registry.read_pointer("day", SYM, "PREVIOUS", models["root"])
    assert prev["sha256"] == incumbent and Path(prev["path"]).exists()
    status = registry.symbol_status("day", SYM, models["active"], models["root"])
    assert status["rollback_available"] and status["active_matches_live_file"]
    with sqlite3.connect(models["db"]) as conn:
        path = conn.execute("SELECT path FROM ai_model_versions WHERE status='active'").fetchone()[0]
    assert str(models["root"]) in path


def test_tie_never_promotes(models):
    registry.set_promotion_enabled("day", True, "test", models["root"])
    before = models["active"].read_bytes()
    cand = _write(models["tmp"] / "versions" / "same.pkl", _artifact(1e6, trained_at="2026-03-01T00:00:00+00:00"))
    X, y, nets, gbs = _holdout(30, 10)
    metrics = _metrics(cand, models["active"], X, y, nets, gbs)
    assert metrics["paired_decision"]["verdict"] == "TIE"
    assert _register(models, cand, metrics) == (False, "keep_incumbent:TIE")
    assert models["active"].read_bytes() == before
    identical = _write(models["tmp"] / "versions" / "copy.pkl", pickle.loads(before))
    assert _register(models, identical, _metrics(identical, models["active"], X, y, nets, gbs))[0] is False


def test_ambiguous_comparison_keeps_incumbent(models):
    registry.set_promotion_enabled("day", True, "test", models["root"])
    before = models["active"].read_bytes()
    cand = _write(models["tmp"] / "versions" / "c1.pkl", _artifact(0.5))
    X, y, nets, gbs = _holdout(2, 0)
    nets = np.array([0.02, -0.018])
    metrics = _metrics(cand, models["active"], X, y, nets, gbs)
    assert metrics["paired_decision"]["verdict"] == "AMBIGUOUS"
    assert _register(models, cand, metrics) == (False, "keep_incumbent:AMBIGUOUS")
    assert models["active"].read_bytes() == before


def test_no_fixed_minimum_n_gate():
    X = np.zeros((3, DIM))
    d = compare_paired(np.ones(3, dtype=int), np.zeros(3, dtype=int), np.array([0.02] * 3), np.array([""] * 3, dtype=object))
    assert d["verdict"] == "PROMOTE" and d["n"] == 3
    one = compare_paired(np.ones(1, dtype=int), np.zeros(1, dtype=int), np.array([0.02]), np.array([""], dtype=object))
    assert one["verdict"] == "AMBIGUOUS"
    src = (REPO / "backend/services/ai_model_promotion.py").read_text()
    assert "holdout_count < 20" not in src and "min_samples_for_promotion" not in src
    assert X.shape[0] == 3


def test_atomic_rollback_is_deterministic(models):
    registry.set_promotion_enabled("day", True, "test", models["root"])
    incumbent = models["active"].read_bytes()
    cand = _write(models["tmp"] / "versions" / "c1.pkl", _artifact(0.5))
    assert registry.promote_atomic("day", SYM, cand, models["active"], root=models["root"])[0]
    assert registry.rollback_to_previous("day", SYM, models["active"], reason="test", root=models["root"]) == (True, "rollback_executed")
    assert models["active"].read_bytes() == incumbent
    assert registry.read_pointer("day", SYM, "ACTIVE", models["root"])["sha256"] == registry.sha256_file(models["active"])
    assert registry.rollback_to_previous("day", SYM, models["active"], reason="again", root=models["root"]) == (False, "previous_was_rolled_back")


def test_failed_candidate_load_cannot_replace_incumbent(models):
    before = models["active"].read_bytes()
    broken = models["tmp"] / "versions" / "broken.pkl"
    broken.parent.mkdir(parents=True, exist_ok=True)
    broken.write_bytes(pickle.dumps({"model": None, "scaler": None, "feature_dim": DIM}))
    ok, why, _ = registry.promote_atomic("day", SYM, broken, models["active"], root=models["root"])
    assert not ok and why.startswith("candidate_reload_failed")
    assert models["active"].read_bytes() == before
    good = _write(models["tmp"] / "versions" / "good.pkl", _artifact(0.5))
    real = registry.reload_test
    calls = {"n": 0}

    def _fail_after_switch(path, feature_dim=None):
        calls["n"] += 1
        return (False, "injected") if Path(path) == models["active"] and calls["n"] > 1 else real(path, feature_dim)

    with patch.object(registry, "reload_test", side_effect=_fail_after_switch):
        ok, why, _ = registry.promote_atomic("day", SYM, good, models["active"], root=models["root"])
    assert not ok and why.startswith("post_switch_verify_failed")
    assert models["active"].read_bytes() == before
    assert not list(models["active"].parent.glob(".*.tmp.*"))


def _seed_outcomes(db: str, n: int, *, start_id_close: str = "2026-03-01") -> None:
    with sqlite3.connect(db) as conn:
        for i in range(n):
            feats = [0.0] * DIM
            feats[0] = 1.0 if i % 2 == 0 else -1.0
            conn.execute(
                "INSERT INTO ai_outcome_training_rows (symbol, opened_at_utc, closed_at_utc, strategy_id, net_pnl_pct, features_json, context_json, outcome_label, ingested_at_utc) "
                "VALUES (?, ?, ?, 'day', ?, ?, ?, ?, datetime('now'))",
                (
                    SYM,
                    f"{start_id_close}T{i:02d}:00:00+00:00",
                    f"{start_id_close}T{i:02d}:30:00+00:00",
                    0.02 if i % 2 == 0 else -0.01,
                    json.dumps(feats),
                    json.dumps({"_feature_version": FEATURE_VERSION_DAY_HTF}),
                    1 if i % 2 == 0 else 0,
                ),
            )
        conn.commit()


def test_candidate_and_incumbent_share_one_causal_holdout(models):
    _seed_outcomes(models["db"], 20)
    _write(models["active"], _artifact(1e6, trained_at="2026-03-01T05:45:00+00:00", train_outcome_max_id=3))
    cand = _write(models["tmp"] / "versions" / "c1.pkl", _artifact(0.5, training_data_end="2026-03-01T00:00:00+00:00"))
    seen = []
    real = artifact_predictions

    def _spy(path, X):
        seen.append((Path(path), X))
        return real(path, X)

    with patch("backend.services.ai_model_promotion_holdout.artifact_predictions", side_effect=_spy):
        m = build_holdout_validation_metrics(strategy_id="day", symbol_bus=SYM, candidate_path=cand, active_path=models["active"], db_path=models["db"])
    assert len(seen) == 2 and seen[0][1] is seen[1][1]
    assert m["paired_decision"]["n"] == m["holdout_sample_count"] > 0
    w = holdout_window(strategy_id="day", symbol_bus=SYM, db_path=models["db"], active_path=models["active"])
    assert w["min_id"] > 3 and w["first_closed_at"] > "2026-03-01T05:45"


def test_no_training_row_leaks_into_holdout():
    from backend.ai_training_pipeline import _purge_after_holdout_start

    windows = {SYM: {"n": 4, "first_opened_at": "2026-03-01T10:00:00+00:00"}}
    h4 = 4 * 3600 * 1000
    t10 = 1772359200000  # 2026-03-01T10:00Z
    self_rows = [
        {"symbol": SYM, "label_anchor_4h_open_ms": t10 - 2 * h4},
        {"symbol": SYM, "label_anchor_4h_open_ms": t10 - h4 + 1},
        {"symbol": SYM, "label_anchor_4h_open_ms": t10 + h4},
    ]
    outcomes = [
        {"id": 1, "symbol": SYM, "closed_at_utc": "2026-03-01T09:00:00+00:00"},
        {"id": 2, "symbol": SYM, "closed_at_utc": "2026-03-01T11:00:00+00:00"},
    ]
    kept_self, kept_oc, info = _purge_after_holdout_start(self_rows, outcomes, windows)
    assert [r["label_anchor_4h_open_ms"] for r in kept_self] == [t10 - 2 * h4]
    assert [r["id"] for r in kept_oc] == [1]
    assert info[SYM]["training_data_end"].startswith("2026-03-01T10:00")


def test_candidate_trained_past_holdout_start_cannot_promote(models, tmp_path):
    registry.set_promotion_enabled("day", True, "test", models["root"])
    leaky = _write(tmp_path / "versions" / "leaky.pkl", _artifact(0.5, training_data_end="2026-03-01T00:00:00+00:00"))
    assert holdout_is_causal_for(leaky, {"first_opened_at": "2026-02-01T00:00:00+00:00"})[0] is False
    legacy = _write(tmp_path / "versions" / "legacy.pkl", _artifact(0.5, training_data_end=None))
    X, y, nets, gbs = _holdout(30, 10)
    before = models["active"].read_bytes()
    for cand in (leaky, legacy):
        metrics = _metrics(cand, models["active"], X, y, nets, gbs)
        assert _register(models, cand, metrics) == (False, "keep_incumbent:CAUSALITY_UNVERIFIED")
    assert models["active"].read_bytes() == before


def test_auto_rollback_suppressed_while_promotion_disabled(models):
    cand = _write(models["tmp"] / "versions" / "c1.pkl", _artifact(0.5))
    registry.set_promotion_enabled("day", True, "test", models["root"])
    assert registry.promote_atomic("day", SYM, cand, models["active"], root=models["root"])[0]
    registry.set_promotion_enabled("day", False, "freeze", models["root"])
    with sqlite3.connect(models["db"]) as conn:
        for i in range(25):
            conn.execute(
                "INSERT INTO ai_outcome_training_rows (symbol, opened_at_utc, closed_at_utc, strategy_id, net_pnl_pct, ingested_at_utc) VALUES (?, ?, ?, 'day', -0.01, datetime('now'))",
                (SYM, f"2099-01-01T00:00:{i:02d}Z", f"2099-01-01T01:00:{i:02d}Z"),
            )
        conn.commit()
    live = models["active"].read_bytes()
    res = maybe_rollback_underperforming_model(strategy_id="day", symbol=SYM, db_path=models["db"], active_dir=models["active_dir"])
    assert res == (False, "rollback_suppressed_promotion_disabled")
    assert models["active"].read_bytes() == live


def test_prune_never_deletes_served_artifacts(models):
    registry.set_promotion_enabled("day", True, "test", models["root"])
    cand = _write(models["tmp"] / "versions" / "c1.pkl", _artifact(0.5))
    assert registry.promote_atomic("day", SYM, cand, models["active"], root=models["root"])[0]
    for i in range(5):
        registry.register_artifact("day", SYM, _write(models["tmp"] / "versions" / f"x{i}.pkl", _artifact(0.1 * (i + 2))), None, models["root"])
    registry.prune_candidates("day", SYM, keep=0, root=models["root"])
    for which in ("ACTIVE", "PREVIOUS"):
        assert Path(registry.read_pointer("day", SYM, which, models["root"])["path"]).exists()


def test_legacy_layout_inference_rows_never_feed_tier_training(tmp_path):
    from backend.services.ai_learning_ingestion import _features_for_decision_ids

    db = tmp_path / "inf.db"
    ensure_ai_canonical_tables(str(db))
    feats = json.dumps([0.1] * DIM)
    with sqlite3.connect(db) as conn:
        for did, ctx in (("legacy", {"_htf_rebuild": {"legacy_4h_kept": True}}), ("clean", {"_htf_rebuild": {"legacy_4h_converted": True}})):
            conn.execute(
                "INSERT INTO ai_inference_log (decision_id, symbol, ts_utc, features_json, feature_version, ctx_json) VALUES (?, ?, '2026-09-18T01:00:00Z', ?, ?, ?)",
                (did, SYM, feats, FEATURE_VERSION_DAY_HTF, json.dumps(ctx)),
            )
        conn.commit()
        out = _features_for_decision_ids(conn, ["legacy", "clean"], DIM, FEATURE_VERSION_DAY_HTF)
    assert set(out) == {"clean"}


def test_day_live_and_scalp_paths_do_not_touch_model_governance():
    governance = ("ai_model_registry", "ai_model_promotion", "promote_atomic")
    paths = [*sorted((REPO / "backend/services/day_v2").glob("*.py")), *sorted((REPO / "backend").rglob("*scalp*.py"))]
    assert paths
    for path in paths:
        src = path.read_text()
        assert not any(g in src for g in governance), path
