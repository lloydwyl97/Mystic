"""DAY mid-hold stall exit — cut dead/worsening holds before hard time-stop."""

import pytest

from backend.services.day_controlled_exits import (
    EXIT_STALL_DEAD,
    EXIT_TIME_STOP,
    STALL_HOLD_MFE_TOO_HIGH,
    STALL_HOLD_NET_PROFIT_ELIGIBLE,
    STALL_HOLD_NOT_RED,
    STALL_HOLD_TOO_YOUNG,
    evaluate_engine_managed_exit,
    evaluate_stall_exit,
)


@pytest.fixture(autouse=True)
def _day_trade_stall_defaults(monkeypatch):
    monkeypatch.setenv("DAY_STALL_MIN_HOLD_MIN", "120")
    monkeypatch.setenv("DAY_STALL_MAX_MFE_PCT", "0.0050")
    monkeypatch.setenv("DAY_STALL_MIN_ADVERSE_PCT", "0.0025")
    monkeypatch.setenv("DAY_STALL_RECOVERY_PCT", "0.0010")
    monkeypatch.setenv("DAY_GIVEBACK_EXIT_ENABLED", "false")
    monkeypatch.setenv("DAY_PATH_AWARE_EXIT", "false")


class _Pos:
    def __init__(self, **kw):
        self.entry_price = kw.get("entry_price", 100.0)
        self.highest_price = kw.get("highest_price", 100.0)
        self.lowest_price = kw.get("lowest_price", 100.0)
        self.stop_price = kw.get("stop_price", 0.0)
        self.trailing_stop_price = kw.get("trailing_stop_price", 0.0)
        self.trail_pct = kw.get("trail_pct", 0.005)
        self.take_profit_1_price = kw.get("take_profit_1_price", 0.0)
        self.entry_thesis = kw.get("entry_thesis", "HTF_TREND_PULLBACK")
        self.entry_vwap = kw.get("entry_vwap", 100.0)
        self.thesis_invalid_level = kw.get("thesis_invalid_level", 0.0)
        self.thesis_target_level = kw.get("thesis_target_level", 0.0)
        self.thesis_score = kw.get("thesis_score", 0.7)
        self.max_hold_min = kw.get("max_hold_min", 360)
        self.day_route_regime_at_entry = kw.get("day_route_regime_at_entry", "")


def test_stall_cuts_dead_hold_at_120m():
    out = evaluate_stall_exit(
        entry_price=100.0,
        highest_price=100.05,  # MFE 0.05% < 0.50%
        lowest_price=99.60,  # MAE 0.40%
        current_price=99.65,
        net_pnl_pct=-0.0045,
        hold_minutes=120.0,
        max_hold_min=360,
    )
    assert out is not None
    assert out["action"] == "sell"
    assert out["reason"] == EXIT_STALL_DEAD


def test_stall_skips_when_mfe_progressed():
    out = evaluate_stall_exit(
        entry_price=100.0,
        highest_price=100.60,  # MFE 0.60% > 0.50%
        lowest_price=99.50,
        current_price=99.70,
        net_pnl_pct=-0.004,
        hold_minutes=130.0,
        max_hold_min=360,
    )
    assert out is not None
    assert out["action"] == "hold"
    assert out["reason"] == STALL_HOLD_MFE_TOO_HIGH


def test_stall_skips_before_min_hold():
    out = evaluate_stall_exit(
        entry_price=100.0,
        highest_price=100.0,
        lowest_price=99.50,
        current_price=99.50,
        net_pnl_pct=-0.006,
        hold_minutes=90.0,
        max_hold_min=360,
    )
    assert out is not None
    assert out["action"] == "hold"
    assert out["reason"] == STALL_HOLD_TOO_YOUNG


def test_stall_skips_when_above_profit_floor():
    out = evaluate_stall_exit(
        entry_price=100.0,
        highest_price=100.0,
        current_price=100.50,
        net_pnl_pct=0.005,  # above MIN_NET_PROFIT_TO_SELL default 0.004
        hold_minutes=140.0,
        max_hold_min=360,
    )
    assert out is not None
    assert out["action"] == "hold"
    assert out["reason"] == STALL_HOLD_NET_PROFIT_ELIGIBLE


def test_stall_skips_tiny_green():
    """Never scratch small greens — stall only when net is negative."""
    out = evaluate_stall_exit(
        entry_price=100.0,
        highest_price=100.05,
        current_price=100.05,
        net_pnl_pct=0.0005,
        hold_minutes=130.0,
        max_hold_min=360,
    )
    assert out is not None
    assert out["action"] == "hold"
    assert out["reason"] == STALL_HOLD_NOT_RED


def test_engine_managed_exit_fires_stall_before_time_stop():
    pos = _Pos(
        entry_price=100.0,
        highest_price=100.05,
        lowest_price=99.60,
        max_hold_min=360,
    )
    out = evaluate_engine_managed_exit(
        position=pos,
        current_price=99.65,
        net_pnl_pct=-0.0045,
        hold_minutes=130.0,
        coin_profile={"max_hold_min": 360, "trail": 0.005, "sl": 0.01},
        bundle=None,
    )
    assert out["action"] == "sell"
    assert out["reason"] == EXIT_STALL_DEAD


def test_engine_managed_exit_time_stop_still_owns_ceiling():
    pos = _Pos(entry_price=100.0, highest_price=100.0, lowest_price=99.50, max_hold_min=300)
    out = evaluate_engine_managed_exit(
        position=pos,
        current_price=99.50,
        net_pnl_pct=-0.006,
        hold_minutes=300.0,
        coin_profile={"max_hold_min": 300, "trail": 0.005, "sl": 0.01},
        bundle=None,
    )
    assert out["action"] == "sell"
    assert out["reason"] == EXIT_TIME_STOP


def test_open_position_max_hold_upgrades_to_profile():
    """Stamped short holds (legacy 50/60m) adopt current day-trade ceiling."""
    pos = _Pos(entry_price=100.0, highest_price=100.0, lowest_price=99.50, max_hold_min=60)
    out = evaluate_engine_managed_exit(
        position=pos,
        current_price=99.50,
        net_pnl_pct=-0.006,
        hold_minutes=60.0,
        coin_profile={"max_hold_min": 360, "trail": 0.005, "sl": 0.01},
        bundle=None,
    )
    assert pos.max_hold_min == 360
    assert out["action"] == "hold"
    assert out.get("reason") != EXIT_TIME_STOP
