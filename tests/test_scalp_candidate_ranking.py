"""Unit tests for scalp candidate ranking repairs."""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from backend.services.binance_scalp.config import get_scalp_config
from backend.services.binance_scalp.economics import ScalpEconomics
from backend.services.binance_scalp.scalp_candidate_ranking import (
    HOLD_ACTION_EV,
    SOFT_REJECT_SCORE,
    attach_action_predictions,
    pick_best_global_candidate,
    rank_actions_with_hold,
    rank_setup_signal,
)
from backend.services.binance_scalp.strategies.base import ScalpSetupSignal, StrategyMarketContext


@pytest.fixture(autouse=True)
def _cost_based_ev_fallback(monkeypatch):
    """Pin these to the cost-based EV fallback rather than a trained artifact.

    candidate_expected_net_ev prefers a forward-net prediction when one loads,
    so whichever model file happens to sit in models/ decides the outcome —
    these rows carry no real features and score 0.0, which is neither the
    fallback arithmetic under test nor a stable result across machines.
    """
    import backend.services.binance_scalp.forward_net_predictor as fnp

    monkeypatch.setattr(fnp, "predict_row_expected_net", lambda _row: None)


def _ctx(*, spread: float = 0.0002, mid_change_15s: float = 0.0, confirmed: bool = False) -> StrategyMarketContext:
    econ = ScalpEconomics.from_env()
    config = get_scalp_config()
    snap = SimpleNamespace(
        symbol="BTCUSDT",
        spread_pct=spread,
        best_ask=100.0,
        best_bid=99.98,
        mid=100.0,
        asks=[[100.0, 1000.0]],
    )
    mom = SimpleNamespace(
        mid_change_15s=mid_change_15s,
        mid_change_30s=mid_change_15s,
        bid_change_15s=mid_change_15s,
        momentum_confirmed=confirmed,
    )
    return StrategyMarketContext(
        symbol="BTCUSDT",
        snap=snap,
        mom=mom,
        bars_1m=[{"low": 99.5, "high": 100.5, "close": 100.0}] * 15,
        econ=econ,
        config=config,
        notional_usd=25.0,
    )


def _soft_sig(reason: str, *, expected: float = 0.0035) -> ScalpSetupSignal:
    return ScalpSetupSignal(
        symbol="BTCUSDT",
        side="BUY",
        score=0.0,
        setup_name="range_bounce_scalp",
        confidence=0.4,
        entry_reason="",
        invalidation_reason=None,
        required_target_pct=0.0025,
        expected_move_pct=expected,
        spread_pct=0.0002,
        impact_pct=0.0,
        depth_sufficient=True,
        limit_buy_price=100.0,
        passed=False,
        reject_reason=reason,
    )


def test_no_rejection_wick_below_min_tradeable():
    ranked = rank_setup_signal(_soft_sig("NO_REJECTION_WICK"), regime="range", ctx=_ctx())
    assert ranked.rank_score < 1.45
    if ranked.hard_block is None:
        assert ranked.entry_eligible is True
    else:
        assert ranked.entry_eligible is False


def test_not_near_support_below_wick_when_flat():
    wick = rank_setup_signal(_soft_sig("NO_REJECTION_WICK"), regime="range", ctx=_ctx())
    support = rank_setup_signal(_soft_sig("NOT_NEAR_SUPPORT"), regime="range", ctx=_ctx())
    assert support.rank_score > wick.rank_score


def test_passed_signal_still_eligible():
    ctx = _ctx()
    passed = ScalpSetupSignal(
        symbol="BTCUSDT",
        side="BUY",
        score=2.6,
        setup_name="range_bounce_scalp",
        confidence=0.7,
        entry_reason="support_bounce",
        invalidation_reason="support_break",
        required_target_pct=0.0025,
        expected_move_pct=0.0035,
        spread_pct=0.0002,
        impact_pct=0.0,
        depth_sufficient=True,
        limit_buy_price=100.0,
        passed=True,
        reject_reason=None,
    )
    ranked = rank_setup_signal(passed, regime="range", ctx=ctx)
    assert ranked.entry_eligible is True
    assert ranked.rank_components.get("primary") == "EV_10s"
    assert ranked.rank_components.get("static_rank", 0) >= 2.0


def test_weak_global_tie_still_selects_highest_available_rank():
    rows = [
        {
            "symbol": sym,
            "rank_score": 1.35,
            "entry_eligible": True,
            "rank_meta": {"soft_reason": "NO_REJECTION_WICK", "regime_native": True},
            "signal": SimpleNamespace(passed=False, spread_pct=0.00005 if sym == "BTCUSDT" else 0.0002),
            "mom": SimpleNamespace(mid_change_15s=0.0, mid_change_30s=0.0),
        }
        for sym in ("BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT")
    ]
    best = pick_best_global_candidate(rows)
    assert best is not None
    assert best["symbol"] in {"BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT"}
    assert best["entry_eligible"] is True


def test_global_tie_prefers_non_btc_when_scores_equal():
    rows = [
        {
            "symbol": "BTCUSDT",
            "rank_score": 1.62,
            "entry_eligible": True,
            "rank_meta": {"soft_reason": "NOT_NEAR_SUPPORT", "regime_native": True, "reachability_surplus": 0.001},
            "signal": SimpleNamespace(passed=False, spread_pct=0.0002, expected_move_pct=0.0035, impact_pct=0.0, confidence=0.6),
            "mom": SimpleNamespace(mid_change_15s=0.0001, mid_change_30s=0.0001),
            "intelligence": {"memory_rank_delta": 0.0},
        },
        {
            "symbol": "ETHUSDT",
            "rank_score": 1.62,
            "entry_eligible": True,
            "rank_meta": {"soft_reason": "NOT_NEAR_SUPPORT", "regime_native": True, "reachability_surplus": 0.0012},
            "signal": SimpleNamespace(passed=False, spread_pct=0.0002, expected_move_pct=0.0035, impact_pct=0.0, confidence=0.6),
            "mom": SimpleNamespace(mid_change_15s=0.0002, mid_change_30s=0.0002),
            "intelligence": {"memory_rank_delta": 0.01},
        },
    ]
    best = pick_best_global_candidate(rows)
    assert best is not None
    assert best["symbol"] == "ETHUSDT"


def test_hold_wins_when_all_buy_expected_net_is_negative():
    rows = [
        {
            "symbol": sym,
            "rank_score": score,
            "entry_eligible": True,
            "rank_meta": {"soft_reason": "NO_VWAP_EMA_RECLAIM", "regime_native": True},
            "signal": SimpleNamespace(
                passed=False,
                spread_pct=0.0002,
                expected_move_pct=0.0,
                impact_pct=0.0,
                confidence=0.0,
            ),
        }
        for sym, score in (("XRPUSDT", 0.9518), ("ETHUSDT", 0.7916), ("BTCUSDT", 0.34), ("SOLUSDT", 0.28))
    ]
    best = pick_best_global_candidate(rows)
    assert best is not None
    assert best["symbol"] == "XRPUSDT"
    actions = rank_actions_with_hold(rows)
    assert actions[0]["action_name"] == "HOLD"
    assert actions[0]["expected_net_ev"] == HOLD_ACTION_EV
    assert all(float(r["expected_net_ev"]) < HOLD_ACTION_EV for r in actions if r["action_name"] != "HOLD")


def test_positive_expected_net_buy_beats_hold():
    rows = [
        {
            "symbol": "BTCUSDT",
            "rank_score": 0.4,
            "entry_eligible": True,
            "signal": SimpleNamespace(passed=True, spread_pct=0.0002, expected_move_pct=0.0035, impact_pct=0.0, confidence=0.7),
        },
        {
            "symbol": "ETHUSDT",
            "rank_score": 0.9,
            "entry_eligible": True,
            "signal": SimpleNamespace(passed=False, spread_pct=0.0002, expected_move_pct=0.0, impact_pct=0.0, confidence=0.0),
        },
    ]
    best = pick_best_global_candidate(rows)
    assert best is not None
    assert best["symbol"] == "ETHUSDT"
    assert float(best["rank_score"]) >= float(rows[1]["rank_score"])


def test_ranking_report_includes_absolute_predicted_outcomes():
    row = {
        "symbol": "SOLUSDT",
        "rank_score": 0.8,
        "entry_eligible": True,
        "signal": SimpleNamespace(passed=False, spread_pct=0.0002, expected_move_pct=0.0, impact_pct=0.0, confidence=0.0),
    }
    stamped = attach_action_predictions(row)
    assert "predicted_net_return" in stamped
    assert "predicted_prob_positive_net" in stamped
    assert "expected_mfe" in stamped
    assert "expected_mae" in stamped
    assert stamped["hold_action_ev"] == HOLD_ACTION_EV
    assert stamped["predicted_net_return"] < 0


def test_clear_rank_leader_survives_an_expected_net_tie():
    """Equal EV is not a tie. The tie-break is for clustered rank scores, so a
    clear leader must not be overridden by it."""
    rows = [
        {
            "symbol": "BTCUSDT",
            "rank_score": 1.8,
            "entry_eligible": True,
            "signal": SimpleNamespace(passed=True, spread_pct=0.0002, expected_move_pct=0.004, impact_pct=0.0, confidence=0.7),
        },
        {
            "symbol": "SOLUSDT",
            "rank_score": 2.0,
            "entry_eligible": True,
            "signal": SimpleNamespace(passed=True, spread_pct=0.0002, expected_move_pct=0.004, impact_pct=0.0, confidence=0.7),
        },
    ]
    best = pick_best_global_candidate(rows)
    assert best is not None
    assert float(rows[0]["expected_net_ev"]) == float(rows[1]["expected_net_ev"])
    assert best["symbol"] == "SOLUSDT"


def test_soft_scores_materially_lower_than_before():
    assert SOFT_REJECT_SCORE["NO_REJECTION_WICK"] < 1.0
    assert SOFT_REJECT_SCORE["NOT_NEAR_SUPPORT"] < 1.45
    assert SOFT_REJECT_SCORE["NOT_NEAR_SUPPORT"] > SOFT_REJECT_SCORE["NO_REJECTION_WICK"]
