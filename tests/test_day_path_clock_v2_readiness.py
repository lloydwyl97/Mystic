"""CLOCK-V2 model-specific readiness. Offline only. Does not train."""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone

from backend.services.day_experiment_registry import TABLE as TABLE_REGISTRY
from backend.services.day_experiment_registry import seed_historical
from backend.services.day_forward_lock import challenger_export_schema
from backend.services.day_model_readiness import MIN_EVENTS_PER_FEATURE
from backend.services.day_path_clock_v2 import (
    CLOCK_V2_EFFECTIVE_FITTED_PARAMETERS,
    CLOCK_V2_LISTED_INPUT_COUNT,
    CLOCK_V2_NUMERIC_FEATURE_COUNT,
    GENERIC_SELECTED_TRADE_LABEL_REQUIREMENT,
    PLANNED_EXPERIMENT_ID,
    PLANNED_EXPERIMENT_ID_V2,
    PLANNED_EXPERIMENT_ID_V3,
    REQUIRED_CLOCK_V2_FIELDS,
    TARGET_HORIZON_STATUS,
    clock_v2_statistical_contract,
    clock_v2_v2_readiness_requirements,
    clock_v2_v3_parameter_contract,
    clock_v2_v3_readiness_requirements,
    planned_challenger_specification,
    planned_challenger_specification_v2,
    planned_challenger_specification_v3,
)
from backend.services.day_path_clock_v2_capture import capture_clock_v2_group
from backend.services.day_path_clock_v2_readiness import (
    evaluate_clock_v2_readiness,
    persist_clock_v2_readiness,
    record_planned_clock_v2_v2,
    record_planned_clock_v2_v3,
)
from tests.test_day_path_clock_v2_capture import _COINS, _book, _contract, _redis_all


def test_14_vs_15_contract_and_140_meaning():
    generic = challenger_export_schema()["inputs"]
    clock = list(REQUIRED_CLOCK_V2_FIELDS)
    assert len(generic) == 14
    assert len(clock) == 15
    assert CLOCK_V2_LISTED_INPUT_COUNT == 15
    assert CLOCK_V2_NUMERIC_FEATURE_COUNT == 14
    assert GENERIC_SELECTED_TRADE_LABEL_REQUIREMENT == 14 * MIN_EVENTS_PER_FEATURE == 140
    spec1 = planned_challenger_specification()
    spec2 = planned_challenger_specification_v2()
    assert spec1["experiment_id"] == PLANNED_EXPERIMENT_ID
    assert spec2["experiment_id"] == PLANNED_EXPERIMENT_ID_V2
    assert spec1["inputs"] != generic
    assert spec1["result"] == "PLANNED_NOT_RUN"
    assert spec2["result"] == "PLANNED_NOT_RUN"
    assert spec1["train"] is False and spec2["train"] is False
    contract = clock_v2_statistical_contract()
    assert contract["generic_required_authoritative_selected_trades"] == 140
    assert "complete decision groups" in contract["why_140_is_not_a_clock_v2_ranker_population"]
    req = clock_v2_v2_readiness_requirements()
    assert req["selected_trades_alone_insufficient"] is True
    assert req["min_feature_complete_groups"] == 140
    assert req["observation_unit"] == "complete_decision_group"


def test_selected_vs_counterfactual_and_readiness(tmp_path, monkeypatch):
    as_of = datetime(2026, 9, 2, 13, 29, tzinfo=timezone.utc)
    redis = _redis_all()
    monkeypatch.setattr("backend.services.decision_book_tape.snapshot_book", _book)
    db = str(tmp_path / "ready.db")
    contract = _contract(
        as_of=as_of,
        eligible={"BTCUSDT": True, "ETHUSDT": False, "SOLUSDT": False, "XRPUSDT": False},
        p_buys={"BTCUSDT": 0.5},
    )
    contract["selected_action"] = "BUY_BTCUSDT"
    contract["selected_symbol"] = "BTCUSDT"
    capture_clock_v2_group(db, contract, redis_client=redis)
    conn = sqlite3.connect(db)
    conn.execute("CREATE TABLE day_decision_group_records (decision_group_id TEXT, created_at TEXT, selected_action TEXT, selected_symbol TEXT)")
    conn.execute(
        "INSERT INTO day_decision_group_records VALUES (?,?,?,?)",
        (contract["decision_group_id"], as_of.isoformat(), "BUY_BTCUSDT", "BTCUSDT"),
    )
    conn.execute("CREATE TABLE day_decision_outcome_labels (decision_group_id TEXT, symbol TEXT, provenance TEXT, label_json TEXT, PRIMARY KEY (decision_group_id, symbol))")
    conn.execute(
        "INSERT INTO day_decision_outcome_labels VALUES (?,?,?,?)",
        (contract["decision_group_id"], "BTCUSDT", "authoritative", json.dumps({"counterfactual": False})),
    )
    conn.execute(
        "INSERT INTO day_decision_outcome_labels VALUES (?,?,?,?)",
        (contract["decision_group_id"], "ETHUSDT", "reconstructed", json.dumps({"counterfactual": True})),
    )
    conn.commit()
    conn.close()
    snap = evaluate_clock_v2_readiness(db, generic_state={"checks": {"F_accounting": {"pass": True}, "G_forward_span": {"pass": False}}})
    assert snap["selected_trade_only_sufficient_for_ranker"] is False
    assert snap["complete_feature_groups"] == 1
    assert snap["authoritative_selected_trade_labels"] == 1
    assert snap["counterfactual_candidate_label_coverage"] >= 1
    assert snap["DATA_READINESS"] == "FAIL"
    assert snap["train"] is False
    assert snap["promoted"] is False
    assert snap["replaces_generic_day_model_readiness"] is False


def test_planned_v2_does_not_mutate_original(tmp_path):
    db = str(tmp_path / "reg.db")
    seed_historical(db)
    record_planned_clock_v2_v2(db)
    persist_clock_v2_readiness(db, snapshot={"DATA_READINESS": "FAIL", "train": False, "promoted": False})
    conn = sqlite3.connect(db)
    v1 = conn.execute(
        f"SELECT result, promoted FROM {TABLE_REGISTRY} WHERE experiment_id=?",
        (PLANNED_EXPERIMENT_ID,),
    ).fetchone()
    v2 = conn.execute(
        f"SELECT result, promoted FROM {TABLE_REGISTRY} WHERE experiment_id=?",
        (PLANNED_EXPERIMENT_ID_V2,),
    ).fetchone()
    conn.close()
    assert v1 == ("PLANNED_NOT_RUN", 0)
    assert v2 == ("PLANNED_NOT_RUN", 0)


def test_v3_parameter_count_and_group_unit():
    params = clock_v2_v3_parameter_contract()
    req = clock_v2_v3_readiness_requirements()
    spec = planned_challenger_specification_v3()
    assert CLOCK_V2_EFFECTIVE_FITTED_PARAMETERS == 19
    assert params["effective_fitted_parameters"] == 19
    assert params["candidates_in_a_group_are_not_independent_events"] is True
    assert params["generic_140_selected_fills"]["justified_as_clock_v2_fitting_sample"] is False
    assert req["min_fully_comparable_independent_groups"] == 190
    assert req["min_authoritative_fills_execution_calibration"] == 50
    # V3 was frozen when horizon was NOT_FROZEN; that value is immutable in v3.
    # V4 (clock_v2_v4_readiness_requirements) carries the frozen status.
    assert req["target_horizon_status"] == "TARGET_HORIZON_NOT_FROZEN"
    assert req["train_blocked_until_horizon_frozen"] is True
    assert spec["experiment_id"] == PLANNED_EXPERIMENT_ID_V3
    assert spec["train"] is False
    assert spec["promoted"] is False
    assert spec["result"] == "PLANNED_NOT_RUN"
    assert spec["target_horizon_status"] == "TARGET_HORIZON_NOT_FROZEN"
    v2 = clock_v2_v2_readiness_requirements()
    assert v2["min_feature_complete_groups"] == 140
    assert v2["min_authoritative_selected_trade_labels"] == 140


def test_registry_v3_does_not_mutate_v1_or_v2(tmp_path):
    db = str(tmp_path / "reg3.db")
    seed_historical(db)
    record_planned_clock_v2_v2(db)
    record_planned_clock_v2_v3(db)
    persist_clock_v2_readiness(db, snapshot={"DATA_READINESS": "FAIL", "train": False, "promoted": False})
    conn = sqlite3.connect(db)
    rows = {
        eid: (result, promoted)
        for eid, result, promoted in conn.execute(
            f"SELECT experiment_id, result, promoted FROM {TABLE_REGISTRY} WHERE experiment_id IN (?,?,?)",
            (PLANNED_EXPERIMENT_ID, PLANNED_EXPERIMENT_ID_V2, PLANNED_EXPERIMENT_ID_V3),
        )
    }
    conn.close()
    assert rows[PLANNED_EXPERIMENT_ID] == ("PLANNED_NOT_RUN", 0)
    assert rows[PLANNED_EXPERIMENT_ID_V2] == ("PLANNED_NOT_RUN", 0)
    assert rows[PLANNED_EXPERIMENT_ID_V3] == ("PLANNED_NOT_RUN", 0)


def test_v3_horizon_blocks_training_even_if_counts_high():
    spec = planned_challenger_specification_v3()
    assert spec["training_procedure"]["executed"] is False
    assert spec["train"] is False
    req = clock_v2_v3_readiness_requirements()
    # v3 was written when horizon was not yet frozen; the constant it captured is preserved.
    # v4 (clock_v2_v4_readiness_requirements) carries the frozen status.
    assert req["target_horizon_status"] == "TARGET_HORIZON_NOT_FROZEN"


def test_v4_horizon_frozen_and_does_not_mutate_v3():
    from backend.services.day_path_clock_v2 import (
        clock_v2_v4_readiness_requirements,
    )

    req_v3 = clock_v2_v3_readiness_requirements()
    req_v4 = clock_v2_v4_readiness_requirements()
    # v4 carries the frozen horizon; v3 still shows NOT_FROZEN (immutable)
    assert req_v3["target_horizon_status"] == "TARGET_HORIZON_NOT_FROZEN"
    assert req_v4["target_horizon_status"] == "PRIMARY_TARGET_HORIZON_3H"
    # Sample-support numbers must be identical
    assert req_v4["min_fully_comparable_independent_groups"] == req_v3["min_fully_comparable_independent_groups"]
    assert req_v4["min_authoritative_fills_execution_calibration"] == req_v3["min_authoritative_fills_execution_calibration"]
