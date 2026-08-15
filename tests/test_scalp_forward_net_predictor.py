"""Forward-net predictor: leakage, features, HOLD-as-action, DAY persist aliases."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from backend.services.binance_scalp.forward_net_predictor import (
    FEATURE_KEYS,
    GAP_BARS,
    chronological_folds,
    flatten_measurements,
    path_labels,
    reset_artifact_cache,
)
from backend.services.binance_scalp.scalp_candidate_ranking import (
    HOLD_ACTION_EV,
    attach_action_predictions,
    candidate_expected_net_ev,
    pick_best_global_candidate,
    rank_actions_with_hold,
)
from backend.services.binance_scalp.scalp_opportunity_dataset import compact_signals_json
from backend.services.portfolio_engine import TradeExplainability


def test_feature_set_has_no_coin_or_label_targets():
    banned = {
        "symbol",
        "passed",
        "rank_score",
        "signal_score",
        "signal_confidence",
        "evidence_rank_delta",
        "buy",
        "hold",
    }
    assert banned.isdisjoint(FEATURE_KEYS)
    assert "projected_move" in FEATURE_KEYS
    assert "orderbook_imbalance" not in FEATURE_KEYS


def test_flatten_ignores_passed_and_broken_signals():
    meas = {
        "vwap_ema_reclaim": {"projected_move": 0.003, "reclaim_strength": 1.2, "passed": True},
        "range_bounce_scalp": {"projected_move": 0.001, "reversal_strength": 0.4},
    }
    feats = flatten_measurements(meas, live_book=False)
    assert feats["projected_move"] == 0.003
    assert "passed" not in feats
    assert compact_signals_json("not-json{{{") == "[]"


def test_replay_imbalance_not_used_without_live_book():
    meas = {"vwap_ema_reclaim": {"orderbook_imbalance": 0.14, "projected_move": 0.002}}
    feats = flatten_measurements(meas, live_book=False)
    assert "orderbook_imbalance" not in feats
    live = flatten_measurements(meas, live_book=True)
    assert live["orderbook_imbalance"] == 0.14


def test_chronological_folds_have_horizon_gap():
    folds = chronological_folds(200, gap=GAP_BARS)
    assert folds
    train, valid, test = folds[0]
    assert train.stop <= valid.start
    assert valid.stop <= test.start
    assert test.start - valid.stop >= 0
    assert valid.start - train.stop >= GAP_BARS


def test_path_labels_subtract_costs():
    future = [{"high": 101.0, "low": 99.8, "close": 100.5} for _ in range(5)]
    labels = path_labels(100.0, future, cost_pct=0.0006)
    assert abs(labels["gross_5m"] - 0.005) < 1e-9
    assert abs(labels["net_5m"] - (0.005 - 0.0006)) < 1e-9
    assert labels["mfe_5m"] > 0


def test_hold_still_wins_when_all_buy_ev_negative():
    reset_artifact_cache()
    rows = [
        {
            "symbol": sym,
            "rank_score": 0.9,
            "entry_eligible": True,
            "signal": SimpleNamespace(passed=False, spread_pct=0.0002, expected_move_pct=0.0, impact_pct=0.0, confidence=0.0),
        }
        for sym in ("BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT")
    ]
    assert pick_best_global_candidate(rows) is None
    actions = rank_actions_with_hold(rows)
    assert actions[0]["action_name"] == "HOLD"
    assert actions[0]["expected_net_ev"] == HOLD_ACTION_EV


def test_fallback_ev_is_gross_minus_cost_without_artifact():
    reset_artifact_cache()
    row = {
        "symbol": "BTCUSDT",
        "signal": SimpleNamespace(passed=True, spread_pct=0.0002, expected_move_pct=0.0035, impact_pct=0.0, confidence=0.7),
    }
    ev = candidate_expected_net_ev(row)
    stamped = attach_action_predictions(dict(row))
    assert ev < 0.0035
    assert stamped["expected_net_ev"] == ev
    assert stamped["forward_net_model_version"] == ""


def test_day_explainability_persists_buy_probability_aliases():
    ex = TradeExplainability(
        trade_id="t1",
        symbol="BTC/USDT",
        side="BUY",
        timestamp="ts",
        ai_confidence=0.61,
        prob_buy=0.61,
        selected_net_expected_value=0.004,
    )
    payload = ex.to_dict()
    assert payload["buy_probability"] == 0.61
    assert payload["model_probability"] == 0.61
    assert payload["predicted_net_return"] == 0.004
    assert payload["ai_confidence"] == 0.61
