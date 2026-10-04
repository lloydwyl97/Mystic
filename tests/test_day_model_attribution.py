"""DAY model rollback authority comes only from causally attributable live outcomes."""

from __future__ import annotations

import json
import pickle
import sqlite3
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

from backend.services import ai_model_registry as registry
from backend.services.ai_canonical_storage import ensure_ai_canonical_tables
from backend.services.ai_model_promotion import attributable_live_evidence, maybe_rollback_underperforming_model, register_candidate_and_maybe_promote
from backend.services.ai_model_promotion_decision import compare_paired
from backend.services.ai_model_promotion_holdout import artifact_predictions
from backend.services.day_model_attribution import ATTRIBUTION_KEY, attributable_to, day_v2_decision_attribution
from backend.services.day_v2.ranking import rank_day_candidates
from backend.services.live_strategy_contracts import per_coin_artifact_file

REPO = Path(__file__).resolve().parents[1]
DIM = 145
SYM = "BTCUSDT"


def _artifact(threshold: float) -> dict:
    x0 = np.array([-5.0, -1.0, threshold - 0.01, threshold + 0.01, threshold + 1.0, 50.0])
    X = np.zeros((len(x0), DIM))
    X[:, 0] = x0
    model = DecisionTreeClassifier(max_depth=1, random_state=0).fit(X, (x0 > threshold).astype(int))
    scaler = StandardScaler(with_mean=False, with_std=False).fit(np.zeros((2, DIM)))
    return {"model": model, "scaler": scaler, "feature_version": 5, "feature_dim": DIM, "live_strategy_id": "day", "accuracy": 0.5, "training_data_end": "2026-01-01T00:00:00+00:00"}


def _write(path: Path, art: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(pickle.dumps(art))
    return path


@pytest.fixture
def promoted(tmp_path: Path):
    """Enabled governance with a promoted active model and a PREVIOUS rollback target."""
    active_dir = tmp_path / "models" / "active"
    active = _write(per_coin_artifact_file(active_dir, "day", SYM), _artifact(1e6))
    root = registry.registry_root(active_dir)
    registry.set_promotion_enabled("day", True, "test", root)
    assert registry.promote_atomic("day", SYM, _write(tmp_path / "v" / "cur.pkl", _artifact(0.5)), active, root=root)[0]
    db = tmp_path / "attr.db"
    ensure_ai_canonical_tables(str(db))
    version = registry.read_pointer("day", SYM, "ACTIVE", root)["version"]
    return {"active": active, "active_dir": active_dir, "root": root, "db": str(db), "version": version, "tmp": tmp_path}


def _seed(db: str, comps: list[dict | None], net: float = -0.02) -> None:
    with sqlite3.connect(db) as conn:
        for i, c in enumerate(comps):
            conn.execute(
                "INSERT INTO ai_outcome_training_rows (symbol, opened_at_utc, closed_at_utc, strategy_id, net_pnl_pct, ingested_at_utc, score_components_json) "
                "VALUES (?, ?, ?, 'day', ?, datetime('now'), ?)",
                ("BTC/USDT", f"2099-01-01T00:{i // 60:02d}:{i % 60:02d}Z", f"2099-01-01T01:{i // 60:02d}:{i % 60:02d}Z", net, json.dumps(c) if c is not None else None),
            )
        conn.commit()


def _rollback(p):
    return maybe_rollback_underperforming_model(strategy_id="day", symbol=SYM, db_path=p["db"], active_dir=p["active_dir"])


def test_non_attributable_day_outcomes_cannot_roll_back(promoted):
    live = promoted["active"].read_bytes()
    stamp = day_v2_decision_attribution(SYM, model_version=promoted["version"], model_output={"prediction": "BUY"})
    _seed(promoted["db"], [None] * 40 + [{ATTRIBUTION_KEY: stamp}] * 40 + [{}] * 40, net=-0.05)
    assert _rollback(promoted) == (False, "no_attributable_live_outcomes")
    assert promoted["active"].read_bytes() == live


def test_attributable_harm_rolls_back_atomically(promoted):
    prev = registry.read_pointer("day", SYM, "PREVIOUS", promoted["root"])
    rec = {ATTRIBUTION_KEY: {"model_version": promoted["version"], "attributable": True, "size_contribution": 0.2}}
    _seed(promoted["db"], [rec] * 6)
    assert _rollback(promoted) == (True, "rollback_executed")
    assert registry.sha256_file(promoted["active"]) == prev["sha256"]
    events = [e for e in registry.read_events("day", SYM, promoted["root"]) if e.get("event") == "rollback"]
    assert events and "attributable_live_harm" in events[-1]["reason"]


def test_evidence_for_another_model_version_has_no_authority(promoted):
    rec = {ATTRIBUTION_KEY: {"model_version": "0000000000000000", "attributable": True, "rank_contribution": 1.0}}
    _seed(promoted["db"], [rec] * 30)
    assert _rollback(promoted) == (False, "no_attributable_live_outcomes")


def test_attributable_flag_without_any_contribution_is_not_evidence():
    assert not attributable_to({ATTRIBUTION_KEY: {"model_version": "v", "attributable": True}}, "v")
    assert attributable_to({ATTRIBUTION_KEY: {"model_version": "v", "attributable": True, "changed_selected_action": True}}, "v")
    assert attributable_to(json.dumps({ATTRIBUTION_KEY: {"model_version": "v", "attributable": True, "rank_contribution": -0.1}}), "v")


def test_no_fixed_minimum_count_uncertainty_decides():
    rec = {ATTRIBUTION_KEY: {"model_version": "v", "attributable": True, "rank_contribution": 1.0}}
    one = attributable_live_evidence([(-0.02, json.dumps(rec))], "v")
    assert one["attributable"] == 1 and one["verdict"] == "NOT_PROVEN_HARMFUL"
    three = attributable_live_evidence([(-0.02, json.dumps(rec))] * 3, "v")
    assert three["verdict"] == "HARMFUL"
    mixed = attributable_live_evidence([(-0.02, json.dumps(rec)), (0.03, json.dumps(rec)), (-0.01, json.dumps(rec))], "v")
    assert mixed["verdict"] == "NOT_PROVEN_HARMFUL"


def test_day_v2_decision_stamp_records_zero_model_contribution():
    stamp = day_v2_decision_attribution("BTC/USDT", model_version="abc", model_output={"prob_buy": "0.7"})
    assert stamp["model_version"] == "abc" and stamp["model_output"] == {"prob_buy": "0.7"}
    assert stamp["attributable"] is False and stamp["changed_selected_action"] is False
    assert stamp["score_contribution"] == stamp["rank_contribution"] == stamp["size_contribution"] == 0.0
    assert not attributable_to({ATTRIBUTION_KEY: stamp}, "abc")


def test_day_v2_ranking_ignores_model_outputs():
    def cand(sym: str, ml: dict) -> dict:
        sig = SimpleNamespace(setup="BREAKOUT", atr_1h=1.0, objective_structural=110.0)
        return {"symbol": sym, "signal": sig, "ask_price": 100.0, "adaptive": {}, **ml}

    plain = rank_day_candidates([cand("BTCUSDT", {}), cand("ETHUSDT", {})], ["BTCUSDT", "ETHUSDT"], 0.002)
    noisy = rank_day_candidates(
        [cand("BTCUSDT", {"prob_buy": 0.01, "confidence": 0.0}), cand("ETHUSDT", {"prob_buy": 0.99, "confidence": 1.0, "prediction": "BUY"})],
        ["BTCUSDT", "ETHUSDT"],
        0.002,
    )
    assert [c["symbol"] for c in plain] == [c["symbol"] for c in noisy]
    assert [c["rank"]["score"] for c in plain] == [c["rank"]["score"] for c in noisy]


def test_live_entry_stamps_attribution_and_outcome_persists_it(tmp_path):
    src = (REPO / "backend/services/day_v2/live_entry.py").read_text()
    assert '"ml_model_attribution": ml_attribution' in src
    from backend.services.ai_outcome_training_writer import record_outcome_training_row
    from backend.services.entry_decision_authority import copy_entry_provenance

    stamp = day_v2_decision_attribution(SYM, model_version="abc", model_output={})
    sell = copy_entry_provenance({ATTRIBUTION_KEY: stamp, "model_version": "day_deterministic_v1"})
    db = tmp_path / "o.db"
    rid = record_outcome_training_row(
        symbol="BTC/USDT",
        opened_at_utc="2026-10-01T00:00:00+00:00",
        closed_at_utc="2026-10-01T01:00:00+00:00",
        hold_seconds=3600,
        entry_price=100.0,
        exit_price=101.0,
        net_profit_usd=1.0,
        net_profit_pct=0.01,
        gross_pnl_pct=0.012,
        close_reason="NET_PROFIT",
        explainability=sell,
        db_path=str(db),
    )
    assert rid
    with sqlite3.connect(db) as conn:
        comps = json.loads(conn.execute("SELECT score_components_json FROM ai_outcome_training_rows WHERE id=?", (rid,)).fetchone()[0])
    assert comps[ATTRIBUTION_KEY]["attributable"] is False and comps[ATTRIBUTION_KEY]["model_version"] == "abc"


def test_auto_promotion_runs_safely_when_enabled(promoted):
    """Enabled: a causally better candidate promotes, keeps PREVIOUS; a tie then keeps it."""
    active_before = registry.sha256_file(promoted["active"])
    cand = _write(promoted["tmp"] / "v" / "better.pkl", _artifact(0.0))
    X = np.zeros((40, DIM))
    X[:30, 0] = 0.5
    X[30:, 0] = -1.0
    nets = np.array([0.02] * 30 + [-0.001] * 10)
    gbs = np.array([""] * 40, dtype=object)
    y = np.zeros(40, dtype=int)
    cp, ap = artifact_predictions(cand, X), artifact_predictions(promoted["active"], X)
    metrics = {
        "holdout_sample_count": 40,
        "candidate_holdout": {"buy_signal_count": int(np.sum(cp == 1))},
        "holdout_window": {"first_opened_at": "2026-02-01T00:00:00+00:00"},
        "paired_decision": compare_paired(cp, ap, nets, gbs, y),
    }
    with patch("backend.services.ai_model_promotion.evaluate_signal_hash_artifact_contract", return_value=(True, None, {})):
        ok, reason = register_candidate_and_maybe_promote(strategy_id="day", symbol=SYM, candidate_path=cand, active_path=promoted["active"], validation_metrics=metrics, db_path=promoted["db"])
        assert ok, reason
        assert registry.read_pointer("day", SYM, "PREVIOUS", promoted["root"])["sha256"] == active_before
        twin = _write(promoted["tmp"] / "v" / "twin.pkl", _artifact(0.0))
        tie_metrics = dict(metrics, paired_decision=compare_paired(cp, cp, nets, gbs, y))
        ok2, reason2 = register_candidate_and_maybe_promote(strategy_id="day", symbol=SYM, candidate_path=twin, active_path=promoted["active"], validation_metrics=tie_metrics, db_path=promoted["db"])
    assert not ok2 and reason2.startswith("keep_incumbent:")
    assert registry.sha256_file(promoted["active"]) == registry.sha256_file(cand)
