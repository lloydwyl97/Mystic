"""CLOCK-V2 corrected action semantics, partition authority and v5 readiness.

Research-contract tests only. Nothing here trains, promotes, inspects the sealed
4H lock, or asserts anything about live trading behaviour.
"""

from __future__ import annotations

import sqlite3

import pytest

from backend.services.day_clock_v2_action_contract import (
    AVAILABILITY_UNKNOWN,
    CONTRACT_VERSION,
    DUPLICATE_SAME_SYMBOL,
    HARD_UNAVAILABLE_REASONS,
    LEGACY_RANK_GENUINE,
    LEGACY_RANK_PATH_EV_SUBSTITUTED,
    MAX_OPEN_LIMIT,
    NO_SCORED_CANDIDATE,
    RECONSTRUCTED_PIT,
    VIOLATION_FILLED_UNAVAILABLE,
    VIOLATION_LEGACY_MEMBERSHIP_USED_AS_AVAILABILITY,
    VIOLATION_SELECTED_UNAVAILABLE,
    evaluate_action_availability,
    evaluate_action_row,
    evaluate_legacy_final_rank,
    evaluate_legacy_rank_candidate,
    reconstruct_group_action_state,
    selected_action_invariant,
)
from backend.services.day_clock_v2_partition import (
    CLOCK_V2_V5_DEVELOPMENT_START,
    DEVELOPMENT,
    FINAL_TEST_STATUS,
    GENERIC_4H_LOCK_ID,
    PRE_MODEL_QUARANTINE,
    declare_final_test_window,
    partition_contract,
    partition_for,
    register_partition_contract,
    stored_partition_contract,
)

# Real Ocean group daygrp_1788588900: path-EV selected ETH and filled it live,
# while capture-v1 recorded ETH as eligible=0 / NO_SCORED_CANDIDATE.
OCEAN_ETH_GROUP = {
    "decision_group_id": "daygrp_1788588900",
    "selected_symbol": "ETHUSDT",
    "lifecycle_state": "filled",
    "open_symbols": [],
    "slots_used": 0,
    "slot_count": 4,
    "candidates": [
        {
            "symbol": "BTCUSDT",
            "eligible": False,
            "exclusion_reason": "NO_SCORED_CANDIDATE",
            "path_input_valid": True,
            "path_ev": 0.0005243229492899691,
            "final_rank_score": 0.0005243229492899691,
        },
        {
            "symbol": "ETHUSDT",
            "eligible": False,
            "exclusion_reason": "NO_SCORED_CANDIDATE",
            "path_input_valid": True,
            "path_ev": 0.000787350605847948,
            "final_rank_score": 0.000787350605847948,
        },
        {
            "symbol": "SOLUSDT",
            "eligible": True,
            "exclusion_reason": None,
            "path_input_valid": True,
            "path_ev": 0.0004884052241086762,
            "final_rank_score": 0.465366,
        },
        {
            "symbol": "XRPUSDT",
            "eligible": True,
            "exclusion_reason": None,
            "path_input_valid": True,
            "path_ev": 9.922790316291389e-05,
            "final_rank_score": 0.480773,
        },
        {"symbol": "HOLD", "eligible": True, "exclusion_reason": None, "path_ev": 0.0, "final_rank_score": 0.0},
    ],
}


# --- selected-action availability invariant ---


def test_legacy_candidate_absence_does_not_make_action_unavailable():
    state = evaluate_action_availability(symbol="ETHUSDT", path_input_valid=True, open_symbols=[], slots_used=0, slot_count=4)
    assert state["action_available"] is True
    assert state["action_unavailable_reason"] is None
    assert NO_SCORED_CANDIDATE not in HARD_UNAVAILABLE_REASONS


def test_filled_action_cannot_be_unavailable_via_no_scored_candidate():
    """The exact audit defect: a filled ETH must never read action_available=false."""
    resolved = reconstruct_group_action_state(OCEAN_ETH_GROUP)
    eth = next(r for r in resolved["rows"] if r["symbol"] == "ETHUSDT")
    assert eth["action_available"] is True
    assert eth["legacy_rank_candidate_present"] is False
    assert eth["legacy_rank_candidate_reason"] == NO_SCORED_CANDIDATE
    assert resolved["selected_action_invariant"]["pass"] is True
    assert resolved["selected_action_invariant"]["violations"] == []


def test_selected_unavailable_is_reported_as_violation_with_named_defect():
    rows = [
        {"symbol": "ETHUSDT", "action_available": False, "action_unavailable_reason": DUPLICATE_SAME_SYMBOL},
        {"symbol": "HOLD", "action_available": True, "action_unavailable_reason": None},
    ]
    out = selected_action_invariant(rows=rows, selected_symbol="ETHUSDT", filled=True)
    assert out["pass"] is False
    assert VIOLATION_SELECTED_UNAVAILABLE in out["violations"]
    assert VIOLATION_FILLED_UNAVAILABLE in out["violations"]
    assert out["proven_production_defect"] == f"PRODUCTION_SELECTED_HARD_BLOCKED_ACTION:{DUPLICATE_SAME_SYMBOL}"


def test_legacy_membership_used_as_availability_is_flagged():
    rows = [
        {"symbol": "ETHUSDT", "action_available": False, "action_unavailable_reason": NO_SCORED_CANDIDATE},
        {"symbol": "HOLD", "action_available": True, "action_unavailable_reason": None},
    ]
    out = selected_action_invariant(rows=rows, selected_symbol="ETHUSDT")
    assert out["pass"] is False
    assert VIOLATION_LEGACY_MEMBERSHIP_USED_AS_AVAILABILITY in out["violations"]


def test_hold_selection_always_satisfies_invariant():
    out = selected_action_invariant(rows=[], selected_symbol="HOLD")
    assert out["pass"] is True


# --- legacy candidate membership is a separate concept ---


def test_legacy_membership_separate_from_availability():
    membership = evaluate_legacy_rank_candidate(symbol="BTCUSDT", candidate_present=False)
    availability = evaluate_action_availability(symbol="BTCUSDT", path_input_valid=True, open_symbols=[], slots_used=0, slot_count=4)
    assert membership["legacy_rank_candidate_present"] is False
    assert availability["action_available"] is True


def test_hold_is_always_available_and_always_a_member():
    assert evaluate_action_availability(symbol="HOLD")["action_available"] is True
    assert evaluate_legacy_rank_candidate(symbol="HOLD", candidate_present=False)["legacy_rank_candidate_present"] is True


# --- genuine hard-ineligible actions ---


@pytest.mark.parametrize(
    ("kwargs", "expected"),
    [
        ({"path_input_valid": False, "path_invalid_reason": "PATH_INPUT_INVALID_GAP"}, "PATH_INPUT_INVALID_GAP"),
        ({"path_input_valid": True, "open_symbols": ["ETH/USDT"]}, DUPLICATE_SAME_SYMBOL),
        ({"path_input_valid": True, "slots_used": 4, "slot_count": 4}, MAX_OPEN_LIMIT),
    ],
)
def test_genuine_hard_gates_make_action_unavailable(kwargs, expected):
    state = evaluate_action_availability(symbol="ETHUSDT", **kwargs)
    assert state["action_available"] is False
    assert state["action_unavailable_reason"] == expected


def test_unknown_availability_is_none_not_false():
    state = evaluate_action_availability(symbol="ETHUSDT", path_input_valid=None)
    assert state["action_available"] is None
    assert state["action_unavailable_reason"] == AVAILABILITY_UNKNOWN


def test_symbol_outside_day_universe_is_unavailable():
    assert evaluate_action_availability(symbol="DOGEUSDT", path_input_valid=True)["action_available"] is False


# --- legacy final-rank provenance / no fabrication ---


def test_genuine_legacy_rank_is_preserved():
    out = evaluate_legacy_final_rank(symbol="XRPUSDT", candidate_present=True, final_selection_score=0.480773)
    assert out["legacy_final_rank_score"] == pytest.approx(0.480773)
    assert out["legacy_final_rank_score_valid"] is True
    assert out["legacy_final_rank_reason"] == LEGACY_RANK_GENUINE


def test_path_ev_substitution_is_detected_and_stored_null():
    out = evaluate_legacy_final_rank(
        symbol="ETHUSDT",
        candidate_present=False,
        recorded_final_rank_score=0.000787350605847948,
        path_ev=0.000787350605847948,
    )
    assert out["legacy_final_rank_score"] is None
    assert out["legacy_final_rank_score_valid"] is False
    assert out["legacy_final_rank_reason"] == LEGACY_RANK_PATH_EV_SUBSTITUTED


def test_no_fake_final_rank_for_any_absent_candidate_in_real_group():
    resolved = reconstruct_group_action_state(OCEAN_ETH_GROUP)
    for row in resolved["rows"]:
        if row["legacy_rank_candidate_present"] or row["symbol"] == "HOLD":
            continue
        assert row["legacy_final_rank_score"] is None
        assert row["legacy_final_rank_score_valid"] is False


def test_hold_reference_rank_is_zero():
    out = evaluate_legacy_final_rank(symbol="HOLD", candidate_present=True)
    assert out["legacy_final_rank_score"] == 0.0
    assert out["legacy_final_rank_score_valid"] is True


# --- point-in-time reconstruction ---


def test_reconstruction_is_pit_when_path_telemetry_present():
    resolved = reconstruct_group_action_state(OCEAN_ETH_GROUP)
    assert resolved["reconstruction_status"] == RECONSTRUCTED_PIT
    assert resolved["trainable_support_eligible"] is True
    assert resolved["contract_version"] == CONTRACT_VERSION


def test_reconstruction_refuses_to_guess_without_path_telemetry():
    payload = {
        **OCEAN_ETH_GROUP,
        "candidates": [{**row, "path_input_valid": None} if row["symbol"] != "HOLD" else row for row in OCEAN_ETH_GROUP["candidates"]],
    }
    resolved = reconstruct_group_action_state(payload)
    assert resolved["reconstruction_status"] != RECONSTRUCTED_PIT
    assert resolved["trainable_support_eligible"] is False


def test_reconstruction_does_not_mutate_source_contract():
    before = OCEAN_ETH_GROUP["candidates"][1]["eligible"]
    reconstruct_group_action_state(OCEAN_ETH_GROUP)
    assert OCEAN_ETH_GROUP["candidates"][1]["eligible"] is before


def test_evaluate_action_row_reports_all_five_concepts():
    row = evaluate_action_row(
        symbol="ETHUSDT",
        candidate_present=False,
        exclusion_reason=NO_SCORED_CANDIDATE,
        path_input_valid=True,
        open_symbols=[],
        slots_used=0,
        slot_count=4,
        production_selected=True,
        execute_authorized=True,
        filled=True,
    )
    assert row["action_available"] is True
    assert row["legacy_rank_candidate_present"] is False
    assert row["production_selected"] is True
    assert row["execute_authorized"] is True
    assert row["filled"] is True


# --- partition authority ---


def test_partition_boundaries():
    assert partition_for("2026-09-05T22:15:00+00:00") == PRE_MODEL_QUARANTINE
    assert partition_for(CLOCK_V2_V5_DEVELOPMENT_START) == DEVELOPMENT
    assert partition_for("2026-09-06T00:00:01+00:00") == DEVELOPMENT


def test_development_start_is_declared_and_not_backdated():
    contract = partition_contract()
    assert contract["clock_v2_v5_development_start"] == CLOCK_V2_V5_DEVELOPMENT_START
    assert contract["development_start_backdated"] is False
    assert CLOCK_V2_V5_DEVELOPMENT_START > "2026-09-05"


def test_quarantine_never_counts_toward_trainability():
    assert partition_contract()["pre_model_quarantine"]["counts_toward_v5_trainability"] is False
    assert partition_contract()["pre_model_quarantine"]["used_for_model_fitting"] is False


def test_final_test_is_not_yet_created_and_cannot_be_declared_here():
    contract = partition_contract()
    assert contract["final_test"]["status"] == FINAL_TEST_STATUS == "NOT_YET_CREATED"
    assert contract["final_test"]["start"] is None
    assert contract["final_test"]["must_be_future_relative_to_training"] is True
    assert contract["final_test"]["earlier_period_as_final_test_forbidden"] is True
    with pytest.raises(RuntimeError):
        declare_final_test_window()


def test_generic_4h_lock_is_not_the_clock_v2_partition_and_is_unchanged():
    lock = partition_contract()["generic_4h_lock"]
    assert lock["experiment_id"] == GENERIC_4H_LOCK_ID == "forward_4h_entry_lock_20260903"
    assert lock["is_clock_v2_partition"] is False
    assert lock["mutated"] is False
    assert lock["inspected"] is False


def test_partition_registration_is_idempotent(tmp_path):
    db = tmp_path / "p.db"
    register_partition_contract(db)
    first = stored_partition_contract(db)
    register_partition_contract(db)
    second = stored_partition_contract(db)
    assert first is not None
    assert first["created_at"] == second["created_at"]
    assert first["development_start"] == CLOCK_V2_V5_DEVELOPMENT_START


def test_partition_registration_does_not_touch_forward_lock_table(tmp_path):
    db = tmp_path / "p.db"
    register_partition_contract(db)
    conn = sqlite3.connect(str(db))
    try:
        tables = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    finally:
        conn.close()
    assert "day_forward_lock_registry" not in tables
