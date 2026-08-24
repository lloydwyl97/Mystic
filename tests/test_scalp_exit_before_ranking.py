"""Exit-first fidelity only — no SCALP strategy retune."""

from __future__ import annotations

import inspect
from types import SimpleNamespace

from backend.services.binance_scalp.exit_manager import (
    DECISION_SELL,
    EXIT_PATH_MAX_ADVERSE_STOP,
    PositionTrack,
    _path_max_adverse_net_pct,
    evaluate_exit,
)
from backend.services.binance_scalp.scalp_candidate_ranking import (
    HOLD_ACTION_EV,
    pick_best_global_candidate,
)
from backend.services.binance_scalp.scalp_dynamic_sizing import compute_scalp_position_size
import pytest

from backend.services.scalp_ai_rank_enrichment import INTELLIGENCE_DELTA_CAP


@pytest.fixture(autouse=True)
def _cost_based_ev_fallback(monkeypatch):
    import backend.services.binance_scalp.forward_net_predictor as fnp

    monkeypatch.setattr(fnp, "predict_row_expected_net", lambda row: None)


def test_exit_runs_before_ranking_in_tick():
    from backend.services.binance_scalp import paper_engine as pe

    src = open(pe.__file__, encoding="utf-8").read()
    tick_at = src.find("def tick(")
    early = src.find("self._exit_open_positions_now()", tick_at)
    ranking = src.find("self._router.evaluate_all(", tick_at)
    klines = src.find("self._klines.get(", tick_at)
    assert tick_at != -1 and early != -1 and ranking != -1
    assert early < klines
    assert early < ranking


def test_adverse_stop_threshold_unchanged():
    assert _path_max_adverse_net_pct() == 0.0015


def test_adverse_stop_fires_without_review_phase():
    from backend.services.binance_scalp.config import get_scalp_config
    from backend.services.binance_scalp.economics import ScalpEconomics
    track = PositionTrack(
        entry_price=100.0,
        state="OPEN",
        max_favorable_pct=0.0,
        max_adverse_pct=0.0,
        session_low_bid=100.0,
        stale_review_count=0,
        review_lows=(),
        setup_name="range_bounce_scalp",
        setup_context={},
    )
    snap = SimpleNamespace(symbol="BTCUSDT", best_bid=99.80, best_ask=99.81, mid=99.805, spread_pct=0.0001)
    mom = SimpleNamespace(
        mid_change_15s=0.0,
        mid_change_30s=0.0,
        mid_change_60s=0.0,
        bid_change_15s=0.0,
        bid_change_30s=0.0,
        bid_change_60s=0.0,
        momentum_confirmed=False,
        realized_volatility_pct=None,
    )
    review = evaluate_exit(
        track=track,
        snap=snap,
        mom=mom,
        econ=ScalpEconomics.from_env(),
        config=get_scalp_config(),
        trade_id="t1",
        hold_sec=5.0,
        executable_net_pct=-0.0020,
        profit_hit=False,
        exit_spread_ok=True,
        perform_review=False,
    )
    assert review.decision == DECISION_SELL
    assert review.exit_reason == EXIT_PATH_MAX_ADVERSE_STOP


def test_production_pick_still_holds_when_all_ev_negative():
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


def test_production_pick_still_uses_positive_ev_not_soft_rank_penalty():
    rows = [
        {
            "symbol": "BTCUSDT",
            "rank_score": 0.4,
            "entry_eligible": True,
            "selection_confidence": "genuine_pass",
            "signal": SimpleNamespace(passed=True, spread_pct=0.0002, expected_move_pct=0.0035, impact_pct=0.0, confidence=0.7),
        },
        {
            "symbol": "XRPUSDT",
            "rank_score": 0.9,
            "entry_eligible": True,
            "selection_confidence": "soft_rank_ranked_below_min_score",
            "signal": SimpleNamespace(passed=False, spread_pct=0.0002, expected_move_pct=0.0, impact_pct=0.0, confidence=0.0),
        },
    ]
    best = pick_best_global_candidate(rows)
    assert best is not None
    assert best["symbol"] == "BTCUSDT"
    assert float(best["expected_net_ev"]) > HOLD_ACTION_EV


def test_learned_size_factor_not_in_production_sizing():
    params = inspect.signature(compute_scalp_position_size).parameters
    assert "learned_size_factor" not in params
    a = compute_scalp_position_size(base_cap=100.0, free_cash=1000.0, strategy_passed=True)
    b = compute_scalp_position_size(base_cap=100.0, free_cash=1000.0, strategy_passed=True)
    assert a.notional == b.notional == 100.0


def test_intelligence_cap_and_four_coins_restored():
    assert INTELLIGENCE_DELTA_CAP == 0.16
    rows = [
        {
            "symbol": sym,
            "rank_score": 0.4,
            "entry_eligible": True,
            "signal": SimpleNamespace(passed=True, spread_pct=0.0002, expected_move_pct=0.003, impact_pct=0.0, confidence=0.6),
        }
        for sym in ("BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT")
    ]
    best = pick_best_global_candidate(rows)
    assert best is not None
    assert best["symbol"] in {"BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT"}
