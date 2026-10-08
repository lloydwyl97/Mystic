"""Queue-supported fills and past-only factors stay out of live accounting."""

from __future__ import annotations

import inspect

import numpy as np

from backend.services import adaptive_learning as al
from backend.services import scalp_execution_research as sx
from backend.services.portfolio_engine_integration import PortfolioEngineIntegration


def test_a_passive_bid_does_not_fill_without_seller_flow():
    quiet = sx.passive_fill(5.0, 1.0, [(100.0, 3.0, False)], 100.0, hit_bid=True)
    assert quiet["filled_qty"] == 0.0
    short = sx.passive_fill(5.0, 1.0, [(100.0, 4.0, True)], 100.0, hit_bid=True)
    assert short["fraction"] == 0.0


def test_queue_ahead_must_be_consumed_before_a_partial_fill():
    prints = [(100.0, 10.0, True), (100.0, 0.4, True)]
    got = sx.passive_fill(10.0, 1.0, prints, 100.0, hit_bid=True)
    assert got["filled_qty"] == 0.4
    assert got["fraction"] == 0.4
    assert sx.candidate_net(got["fraction"], 0.01) == 0.004
    assert sx.candidate_net(0.0, -0.02) == 0.0


def test_a_trade_through_fills_because_the_level_was_cleared():
    got = sx.passive_fill(50.0, 1.0, [(99.5, 0.01, True)], 100.0, hit_bid=True)
    assert got["full"] is True and got["through"] is True


def test_maker_and_taker_nets_stay_separate():
    maker = sx.maker_entry_taker_exit(100.0, 100.1, maker_fee=0.0, taker_fee=0.0002, slippage=0.0001)
    assert maker is not None
    taker = (100.1 - 100.2) / 100.2 - 0.0004 - 0.0002
    assert maker != taker


def test_the_common_factor_is_fit_without_the_row_being_scored():
    past = np.array([[1.0, 0.0], [2.0, 0.0], [3.0, 0.0], [4.0, 0.0]])
    fitted = sx.fit_first_component(past)
    future = np.array([100.0, 50.0])
    refit = sx.fit_first_component(np.vstack([past, future]))
    assert fitted is not None and refit is not None
    assert abs(float(fitted[2][0]) - float(refit[2][0])) > 1e-6
    assert sx.project_component(future, fitted) != sx.project_component(future, refit)


def test_perfect_group_knowledge_is_not_a_live_gate_and_does_not_touch_accounting():
    groups = [
        {"nets": {"BTCUSDT": 0.002, "ETHUSDT": -0.001}, "live": {"BTCUSDT": -0.01, "ETHUSDT": 0.001}},
        {"nets": {"BTCUSDT": -0.003, "ETHUSDT": -0.002}, "live": {"BTCUSDT": 0.01, "ETHUSDT": 0.002}},
    ]
    value = sx.information_value(groups)
    assert value["perfect_both"] > value["always_best"]
    assert value["current_group_perfect_rank"] == 0.0
    fund = inspect.getsource(PortfolioEngineIntegration._fund_day_v2_candidate)
    entry = inspect.getsource(al.day_net_expectancy)
    assert "scalp_execution_research" not in fund
    assert "scalp_execution_research" not in entry
    assert "portfolio_engine_positions" not in inspect.getsource(sx)
