"""DAY mid-hold stall exit — cut dead holds before hard time-stop."""

from backend.services.day_controlled_exits import (
    EXIT_STALL,
    EXIT_TIME_STOP,
    evaluate_engine_managed_exit,
    evaluate_stall_exit,
)


class _Pos:
    def __init__(self, **kw):
        self.entry_price = kw.get("entry_price", 100.0)
        self.highest_price = kw.get("highest_price", 100.0)
        self.stop_price = kw.get("stop_price", 0.0)
        self.trailing_stop_price = kw.get("trailing_stop_price", 0.0)
        self.trail_pct = kw.get("trail_pct", 0.005)
        self.take_profit_1_price = kw.get("take_profit_1_price", 0.0)
        self.entry_thesis = kw.get("entry_thesis", "HTF_TREND_PULLBACK")
        self.entry_vwap = kw.get("entry_vwap", 100.0)
        self.thesis_invalid_level = kw.get("thesis_invalid_level", 0.0)
        self.thesis_target_level = kw.get("thesis_target_level", 0.0)
        self.thesis_score = kw.get("thesis_score", 0.7)
        self.max_hold_min = kw.get("max_hold_min", 60)


def test_stall_cuts_dead_hold_at_30m():
    out = evaluate_stall_exit(
        entry_price=100.0,
        highest_price=100.05,  # MFE 0.05% < 0.15%
        net_pnl_pct=-0.003,
        hold_minutes=30.0,
        max_hold_min=60,
    )
    assert out is not None
    assert out["action"] == "sell"
    assert out["reason"] == EXIT_STALL


def test_stall_skips_when_mfe_progressed():
    out = evaluate_stall_exit(
        entry_price=100.0,
        highest_price=100.30,  # MFE 0.30% > 0.15%
        net_pnl_pct=-0.001,
        hold_minutes=35.0,
        max_hold_min=60,
    )
    assert out is None


def test_stall_skips_before_min_hold():
    out = evaluate_stall_exit(
        entry_price=100.0,
        highest_price=100.0,
        net_pnl_pct=-0.004,
        hold_minutes=20.0,
        max_hold_min=60,
    )
    assert out is None


def test_stall_skips_when_above_profit_floor():
    out = evaluate_stall_exit(
        entry_price=100.0,
        highest_price=100.0,
        net_pnl_pct=0.005,  # above MIN_NET_PROFIT_TO_SELL default 0.004
        hold_minutes=40.0,
        max_hold_min=60,
    )
    assert out is None


def test_stall_skips_tiny_green():
    """Never scratch small greens — stall only when net is negative."""
    out = evaluate_stall_exit(
        entry_price=100.0,
        highest_price=100.05,
        net_pnl_pct=0.0005,
        hold_minutes=35.0,
        max_hold_min=60,
    )
    assert out is None


def test_engine_managed_exit_fires_stall_before_time_stop():
    pos = _Pos(entry_price=100.0, highest_price=100.05, max_hold_min=60)
    out = evaluate_engine_managed_exit(
        position=pos,
        current_price=99.70,
        net_pnl_pct=-0.004,
        hold_minutes=35.0,
        coin_profile={"max_hold_min": 60, "trail": 0.005, "sl": 0.01},
        bundle=None,
    )
    assert out["action"] == "sell"
    assert out["reason"] == EXIT_STALL


def test_engine_managed_exit_time_stop_still_owns_ceiling():
    pos = _Pos(entry_price=100.0, highest_price=100.0, max_hold_min=50)
    out = evaluate_engine_managed_exit(
        position=pos,
        current_price=99.50,
        net_pnl_pct=-0.006,
        hold_minutes=50.0,
        coin_profile={"max_hold_min": 50, "trail": 0.005, "sl": 0.01},
        bundle=None,
    )
    assert out["action"] == "sell"
    assert out["reason"] == EXIT_TIME_STOP
