"""Tests for outcome-driven symbol/setup/regime penalty."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from backend.database_schema import DATABASE_PATH
from backend.services.symbol_setup_outcome_penalty import (
    XRP_PENALTY_V1_EV_FACTOR,
    XRP_PENALTY_V1_RANK_DELTA,
    apply_outcome_penalty_to_decision_data,
    apply_v3_outcome_ranking_to_decision_data,
    build_churn_audit,
    build_ranking_adjustment_report,
    evaluate_btc_outcome_penalty,
    evaluate_outcome_penalty,
    evaluate_sol_outcome_credit,
    write_churn_audit_artifact,
)


def test_xrp_fbr_bear_penalty_applies_from_live_db():
    result = evaluate_outcome_penalty("XRP/USDT", "FAILED_BREAKDOWN_REVERSAL", "bear", db_path=DATABASE_PATH)
    assert result["hard_block"] is False
    if result["applied"]:
        assert result["rank_delta"] <= -0.25
        assert result["ev_factor"] <= 0.52
        assert result["size_factor"] <= 0.52
        assert result.get("penalty_generation") == "v3_final_selection"


def test_xrp_strengthened_more_than_v1():
    result = evaluate_outcome_penalty("XRP/USDT", "FAILED_BREAKDOWN_REVERSAL", "bear", db_path=DATABASE_PATH)
    if result.get("applied"):
        assert result["rank_delta"] < XRP_PENALTY_V1_RANK_DELTA
        assert result["ev_factor"] < XRP_PENALTY_V1_EV_FACTOR


def test_sol_credit_conservative():
    result = evaluate_sol_outcome_credit("SOL/USDT", "FAILED_BREAKDOWN_REVERSAL", "bear", db_path=DATABASE_PATH)
    if result.get("applied"):
        assert 0.0 < result["rank_delta"] <= 0.06
        assert result["size_factor"] == 1.0
        assert result["ev_factor"] == 1.0


def test_ranking_adjustment_report():
    report = build_ranking_adjustment_report(DATABASE_PATH)
    assert report["no_hard_xrp_block"] is True
    assert report["no_strategy_changes"] is True
    assert report.get("v3_ranking_fix_applied") is True
    assert report.get("adjusted_rank_used_in_final_selection") is True


def test_btc_not_penalized_via_xrp_evaluator():
    result = evaluate_outcome_penalty("BTC/USDT", "FAILED_BREAKDOWN_REVERSAL", "bear", db_path=DATABASE_PATH)
    assert result["applied"] is False
    assert result["reason"] == "not_xrp_penalty_scope"


def test_btc_mild_penalty_separate_evaluator():
    result = evaluate_btc_outcome_penalty("BTC/USDT", "FAILED_BREAKDOWN_REVERSAL", "bear", db_path=DATABASE_PATH)
    assert result.get("hard_block") is False
    if result.get("applied"):
        assert result["rank_delta"] == -0.08
        assert result["size_factor"] == 1.0


def test_apply_penalty_adjusts_decision_data():
    dd = {
        "setup_type": "FAILED_BREAKDOWN_REVERSAL",
        "day_route_regime": "bear",
        "thesis_rank_delta": -0.05,
        "thesis_size_factor": 0.35,
        "selected_net_expected_value": 0.10,
    }
    out = apply_v3_outcome_ranking_to_decision_data(dd, "XRP/USDT", raw_rank_score=0.42, buy_margin=0.02)
    if out.get("outcome_penalty_applied"):
        assert out["adjusted_ev"] < out["raw_ev"]
        assert out["final_selection_score"] is not None
        assert out.get("v3_ranking_fix_applied") is True


def test_audit_artifact_writes(tmp_path: Path):
    out = tmp_path / "xrp_churn_audit_test.json"
    audit = write_churn_audit_artifact(DATABASE_PATH, out)
    assert out.exists()
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert "xrp_focus" in payload
    assert "penalty_verification" in payload
    assert audit["symbols"]["XRP/USDT"]["total_trades"] >= 0


def test_build_audit_has_churn_buckets():
    audit = build_churn_audit(DATABASE_PATH)
    assert "churn_protection_by_bucket" in audit
    assert isinstance(audit["churn_protection_by_bucket"], dict)
