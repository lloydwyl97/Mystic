"""v3 final-selection outcome ranking tests."""

from __future__ import annotations

import os
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from backend.database_schema import DATABASE_PATH
from backend.services.symbol_setup_outcome_penalty import (
    apply_v3_outcome_ranking_to_decision_data,
    assign_v3_selection_ranks,
    compute_final_selection_score,
    evaluate_btc_outcome_penalty,
    evaluate_eth_outcome_credit,
    evaluate_outcome_penalty,
    evaluate_sol_outcome_credit,
)


def _base_dd(**overrides):
    dd = {
        "setup_type": "FAILED_BREAKDOWN_REVERSAL",
        "day_route_regime": "bear",
        "selected_net_expected_value": 0.012,
        "buy_margin": 0.02,
    }
    dd.update(overrides)
    return dd


def test_compute_final_selection_score_prefers_adjusted_ev():
    low = compute_final_selection_score(
        adjusted_ev=0.001,
        outcome_adjusted_rank=0.55,
        raw_rank_score=0.50,
        buy_margin=0.40,
        final_score_adjustment=-0.10,
    )
    high = compute_final_selection_score(
        adjusted_ev=0.015,
        outcome_adjusted_rank=0.42,
        raw_rank_score=0.38,
        buy_margin=0.02,
        final_score_adjustment=0.04,
    )
    assert high > low


@patch("backend.services.symbol_setup_outcome_penalty.evaluate_low_mfe_stall_penalty")
@patch("backend.services.symbol_setup_outcome_penalty.evaluate_eth_outcome_credit")
@patch("backend.services.symbol_setup_outcome_penalty.evaluate_sol_outcome_credit")
@patch("backend.services.symbol_setup_outcome_penalty.evaluate_btc_outcome_penalty")
@patch("backend.services.symbol_setup_outcome_penalty.evaluate_outcome_penalty")
def test_xrp_loses_final_rank_to_sol_when_comparable(mock_xrp, mock_btc, mock_sol, mock_eth, mock_low_mfe):
    mock_low_mfe.return_value = {"applied": False, "rank_delta": 0.0, "ev_factor": 1.0, "size_factor": 1.0, "final_score_adjustment": 0.0, "reason": "no_low_mfe_stall"}
    mock_xrp.side_effect = lambda sym, setup, regime, **kw: (
        {
            "applied": True,
            "rank_delta": -0.38,
            "ev_factor": 0.35,
            "size_factor": 0.45,
            "final_score_adjustment": -0.10,
            "peer_ev_ceiling": 0.012,
            "peer_ev_cap_multiplier": 0.88,
            "reason": "negative_expectancy_time_stop_churn",
        }
        if sym == "XRP/USDT"
        else {"applied": False, "rank_delta": 0.0, "ev_factor": 1.0, "size_factor": 1.0, "final_score_adjustment": 0.0, "reason": "not_xrp"}
    )
    mock_btc.return_value = {"applied": False, "rank_delta": 0.0, "ev_factor": 1.0, "size_factor": 1.0, "final_score_adjustment": 0.0, "reason": "not_btc"}
    mock_sol.side_effect = lambda sym, setup, regime, **kw: (
        {
            "applied": True,
            "rank_delta": 0.10,
            "ev_factor": 1.0,
            "size_factor": 1.0,
            "final_score_adjustment": 0.04,
            "credit_amount": 0.10,
            "reason": "sol_fbr_bear_positive_outcomes",
        }
        if sym == "SOL/USDT"
        else {"applied": False, "rank_delta": 0.0, "ev_factor": 1.0, "size_factor": 1.0, "final_score_adjustment": 0.0, "reason": "not_sol"}
    )
    mock_eth.return_value = {"applied": False, "rank_delta": 0.0, "ev_factor": 1.0, "size_factor": 1.0, "final_score_adjustment": 0.0, "reason": "not_eth"}

    xrp_dd = apply_v3_outcome_ranking_to_decision_data(_base_dd(), "XRP/USDT", raw_rank_score=0.44, buy_margin=0.35)
    sol_dd = apply_v3_outcome_ranking_to_decision_data(_base_dd(selected_net_expected_value=0.011), "SOL/USDT", raw_rank_score=0.40, buy_margin=0.02)

    # Default: coin identity ranking is off. Opportunity scores are not
    # rewritten because the symbol is XRP or SOL.
    assert xrp_dd["outcome_penalty_applied"] is False
    assert sol_dd["outcome_credit_applied"] is False
    assert xrp_dd.get("outcome_churn_penalty_eval", {}).get("reason") == "coin_identity_ranking_disabled"
    mock_xrp.assert_not_called()
    mock_sol.assert_not_called()

    with patch.dict(os.environ, {"DAY_COIN_IDENTITY_RANKING": "true"}):
        xrp_on = apply_v3_outcome_ranking_to_decision_data(_base_dd(), "XRP/USDT", raw_rank_score=0.44, buy_margin=0.35)
        sol_on = apply_v3_outcome_ranking_to_decision_data(_base_dd(selected_net_expected_value=0.011), "SOL/USDT", raw_rank_score=0.40, buy_margin=0.02)
    assert xrp_on["outcome_penalty_applied"] is True
    assert sol_on["outcome_credit_applied"] is True
    assert sol_on["final_selection_score"] > xrp_on["final_selection_score"]


@patch("backend.services.symbol_setup_outcome_penalty.evaluate_low_mfe_stall_penalty")
@patch("backend.services.symbol_setup_outcome_penalty.evaluate_eth_outcome_credit")
@patch("backend.services.symbol_setup_outcome_penalty.evaluate_sol_outcome_credit")
@patch("backend.services.symbol_setup_outcome_penalty.evaluate_btc_outcome_penalty")
@patch("backend.services.symbol_setup_outcome_penalty.evaluate_outcome_penalty")
def test_sol_credit_raises_final_selection_score(mock_xrp, mock_btc, mock_sol, mock_eth, mock_low_mfe):
    # Neutralize the low-MFE stall penalty path so the test isolates the SOL-credit contribution.
    mock_low_mfe.return_value = {"applied": False, "rank_delta": 0.0, "ev_factor": 1.0, "size_factor": 1.0, "final_score_adjustment": 0.0, "reason": "no_low_mfe_stall"}
    mock_xrp.return_value = {"applied": False, "rank_delta": 0.0, "ev_factor": 1.0, "size_factor": 1.0, "final_score_adjustment": 0.0, "reason": "not_xrp"}
    mock_btc.return_value = {"applied": False, "rank_delta": 0.0, "ev_factor": 1.0, "size_factor": 1.0, "final_score_adjustment": 0.0, "reason": "not_btc"}
    mock_sol.side_effect = lambda sym, setup, regime, **kw: (
        {
            "applied": True,
            "rank_delta": 0.08,
            "ev_factor": 1.0,
            "size_factor": 1.0,
            "final_score_adjustment": 0.04,
            "credit_amount": 0.08,
            "reason": "sol_fbr_bear_positive_outcomes",
        }
        if sym == "SOL/USDT"
        else {"applied": False, "rank_delta": 0.0, "ev_factor": 1.0, "size_factor": 1.0, "final_score_adjustment": 0.0, "reason": "not_sol"}
    )
    mock_eth.return_value = {"applied": False, "rank_delta": 0.0, "ev_factor": 1.0, "size_factor": 1.0, "final_score_adjustment": 0.0, "reason": "not_eth"}

    base = apply_v3_outcome_ranking_to_decision_data(_base_dd(), "SOL/USDT", raw_rank_score=0.41, buy_margin=0.02)
    neutral = apply_v3_outcome_ranking_to_decision_data(_base_dd(), "ETH/USDT", raw_rank_score=0.41, buy_margin=0.02)

    assert base["outcome_credit_applied"] is False
    assert abs(base["final_selection_score"] - neutral["final_selection_score"]) < 1e-9

    with patch.dict(os.environ, {"DAY_COIN_IDENTITY_RANKING": "true"}):
        base_on = apply_v3_outcome_ranking_to_decision_data(_base_dd(), "SOL/USDT", raw_rank_score=0.41, buy_margin=0.02)
        neutral_on = apply_v3_outcome_ranking_to_decision_data(_base_dd(), "ETH/USDT", raw_rank_score=0.41, buy_margin=0.02)
    assert base_on["outcome_credit_applied"] is True
    assert base_on["final_selection_score"] > neutral_on["final_selection_score"]


def test_solo_candidate_still_has_positive_final_score():
    dd = apply_v3_outcome_ranking_to_decision_data(_base_dd(), "BTC/USDT", raw_rank_score=0.42, buy_margin=0.02)
    solo = [SimpleNamespace(symbol="BTC/USDT", decision_data=dd)]
    assign_v3_selection_ranks(solo)
    assert solo[0].decision_data["final_selected_rank"] == 1
    assert solo[0].decision_data["why_selected"] == "solo_candidate_no_peer"
    assert solo[0].decision_data["final_selection_score"] > 0


def test_no_hard_symbol_block_from_evaluators():
    for sym in ("XRP/USDT", "BTC/USDT", "SOL/USDT", "ETH/USDT"):
        xrp = evaluate_outcome_penalty(sym, "FAILED_BREAKDOWN_REVERSAL", "bear", db_path=DATABASE_PATH)
        assert xrp.get("hard_block") is False
        btc = evaluate_btc_outcome_penalty(sym, "FAILED_BREAKDOWN_REVERSAL", "bear", db_path=DATABASE_PATH)
        assert btc.get("hard_block") is False


def test_v3_fields_persisted_on_apply():
    out = apply_v3_outcome_ranking_to_decision_data(_base_dd(), "XRP/USDT", raw_rank_score=0.40, buy_margin=0.02)
    required = (
        "raw_ev",
        "adjusted_ev",
        "raw_rank_score",
        "outcome_adjusted_rank_score",
        "buy_margin_at_rank",
        "outcome_penalty_or_credit",
        "final_selection_score",
        "outcome_penalty_applied",
        "outcome_credit_applied",
        "penalty_reason",
        "raw_score",
        "adjusted_score",
        "v3_ranking_fix_applied",
        "adjusted_rank_used_in_final_selection",
    )
    for key in required:
        assert key in out, f"missing {key}"


def test_btc_scope_separate_from_xrp_evaluator():
    xrp_scope = evaluate_outcome_penalty("BTC/USDT", "FAILED_BREAKDOWN_REVERSAL", "bear", db_path=DATABASE_PATH)
    assert xrp_scope["applied"] is False
    assert xrp_scope["reason"] == "not_xrp_penalty_scope"
    btc = evaluate_btc_outcome_penalty("BTC/USDT", "FAILED_BREAKDOWN_REVERSAL", "bear", db_path=DATABASE_PATH)
    assert btc.get("hard_block") is False


def test_xrp_v3_penalty_generation_when_applied():
    result = evaluate_outcome_penalty("XRP/USDT", "FAILED_BREAKDOWN_REVERSAL", "bear", db_path=DATABASE_PATH)
    if result.get("applied"):
        assert result.get("penalty_generation") == "v3_final_selection"
        assert result["rank_delta"] <= -0.32
        assert result.get("final_score_adjustment", 0) < 0


def test_eth_credit_scope_only():
    result = evaluate_eth_outcome_credit("BTC/USDT", "FAILED_BREAKDOWN_REVERSAL", "bear", db_path=DATABASE_PATH)
    assert result["applied"] is False
    assert result["reason"] == "not_eth_credit_scope"
