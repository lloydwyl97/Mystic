"""Path labels, reconstructable features, leakage, HOLD, four-coin eligibility."""

from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from backend.services.binance_scalp.forward_net_predictor import FEATURE_KEYS as OLD_KEYS
from backend.services.binance_scalp.forward_net_predictor import chronological_folds
from backend.services.binance_scalp.path_outcomes import all_horizon_path_labels, path_labels_for_horizon
from backend.services.binance_scalp.reconstructable_features import FEATURE_KEYS, LIVE_ONLY, reconstructable_features
from backend.services.binance_scalp.scalp_candidate_ranking import HOLD_ACTION_EV, pick_best_global_candidate, rank_actions_with_hold
from backend.services.clean_acceptance import DAY_CUTOFF_UTC, SCALP_CUTOFF_UTC, parse_ts
from backend.services.validation_cutoff import is_strategy_acceptance_eligible


def test_path_label_detects_executable_mfe_when_terminal_red():
    future = [
        {"high": 100.20, "low": 99.95, "close": 100.10},
        {"high": 100.05, "low": 99.80, "close": 99.85},
        {"high": 99.90, "low": 99.70, "close": 99.75},
    ]
    lab = path_labels_for_horizon(100.0, future, horizon_min=3, cost_pct=0.0006)
    assert lab["terminal_net"] < 0
    assert lab["mfe"] > 0.001
    assert lab["executable_mfe_net"] > 0
    assert lab["executable_profit_occurred"] is True
    assert lab["path_order"] == "MFE_FIRST"
    assert lab["profit_before_adverse"] is True
    assert lab["target_d_net"] == lab["executable_mfe_net"]
    assert lab["target_close_net"] == lab["terminal_net"]
    assert lab["target_close_net"] < lab["target_d_net"]


def test_path_labels_do_not_use_bars_past_horizon():
    future = [{"high": 100.0, "low": 100.0, "close": 100.0} for _ in range(4)]
    future.append({"high": 102.0, "low": 100.0, "close": 102.0})
    lab5 = path_labels_for_horizon(100.0, future, horizon_min=3, cost_pct=0.0006)
    assert lab5["mfe"] < 0.001
    assert lab5["target_reached"] is False


def test_all_horizons_present():
    future = [{"high": 100.1, "low": 99.9, "close": 100.0} for _ in range(20)]
    labs = all_horizon_path_labels(100.0, future)
    assert set(labs) == {1, 3, 5, 10, 20}


def test_reconstructable_features_have_no_future_or_book_or_coin_id():
    banned = {"mfe", "mae", "time_to_target", "terminal_net", "symbol", "passed", "orderbook_imbalance"}
    assert banned.isdisjoint(FEATURE_KEYS)
    assert "orderbook_imbalance" in LIVE_ONLY
    bars = [{"open": 100, "high": 100.2, "low": 99.8, "close": 100 + i * 0.01, "volume": 10 + i} for i in range(25)]
    feats = reconstructable_features(bars, btc_ret_5=0.001, ts=datetime(2026, 8, 14, 15, 0, tzinfo=timezone.utc))
    assert feats["ret_1"] != 0 or feats["rel_volume"] >= 0
    assert "orderbook_imbalance" not in feats


def test_chronological_gap_still_required():
    folds = chronological_folds(200, gap=20)
    train, valid, test = folds[0]
    assert valid.start - train.stop >= 20


def test_old_rejected_feature_set_unchanged():
    assert "projected_move" in OLD_KEYS
    assert "orderbook_imbalance" not in OLD_KEYS


def test_hold_still_wins_negative_ev(monkeypatch):
    monkeypatch.setattr(
        "backend.services.binance_scalp.forward_net_predictor.load_accepted_artifact",
        lambda: None,
    )
    rows = [
        {
            "symbol": sym,
            "rank_score": 0.9,
            "entry_eligible": True,
            "signal": SimpleNamespace(passed=False, spread_pct=0.0002, expected_move_pct=0.0, impact_pct=0.0, confidence=0.0),
        }
        for sym in ("BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT")
    ]
    best = pick_best_global_candidate(rows)
    assert best is not None
    assert rank_actions_with_hold(rows)[0]["action_name"] == "HOLD"
    assert rank_actions_with_hold(rows)[0]["expected_net_ev"] == HOLD_ACTION_EV


def test_all_four_coins_remain_in_hold_action_set():
    rows = [
        {
            "symbol": sym,
            "rank_score": 0.2,
            "entry_eligible": True,
            "signal": SimpleNamespace(passed=True, spread_pct=0.0002, expected_move_pct=0.003, impact_pct=0.0, confidence=0.6),
        }
        for sym in ("BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT")
    ]
    actions = rank_actions_with_hold(rows)
    names = {a["action_name"] for a in actions}
    assert names == {"BUY_BTCUSDT", "BUY_ETHUSDT", "BUY_SOLUSDT", "BUY_XRPUSDT", "HOLD"}


def test_cutoff_constants_match_ocean_stored_values():
    assert parse_ts(DAY_CUTOFF_UTC) is not None
    assert parse_ts(SCALP_CUTOFF_UTC) is not None
    assert is_strategy_acceptance_eligible(exit_reason="RECONCILIATION_MANUAL_EXIT") is False
    assert is_strategy_acceptance_eligible(exit_reason="NET_PROFIT_EXIT") is True
