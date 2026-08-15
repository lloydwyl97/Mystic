"""P1A: STALL exits only dead/worsening DAY holds — not flat low-MFE timers."""

from __future__ import annotations

from pathlib import Path

import pytest

from backend.config.trading_economics import MIN_NET_PROFIT_TO_SELL
from backend.services.day_controlled_exits import (
    EXIT_NET_PROFIT,
    EXIT_STALL_DEAD,
    EXIT_STOP_LOSS,
    STALL_HOLD_FLAT_NOT_DEAD,
    STALL_HOLD_NOT_RED,
    STALL_HOLD_RECOVERY_PRESENT,
    STALL_HOLD_TOO_YOUNG,
    evaluate_engine_managed_exit,
    evaluate_stall_exit,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
CORE_ONLY_LOCAL_ENV = REPO_ROOT / "deploy" / "core_only_local.env"


def test_core_only_local_env_stall_adverse_floor_not_below_p1a():
    """deploy/core_only_local.env is sourced after .env and must not disable the MAE floor."""
    text = CORE_ONLY_LOCAL_ENV.read_text(encoding="utf-8")
    value = None
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("DAY_STALL_MIN_ADVERSE_PCT="):
            value = float(line.split("=", 1)[1].strip().strip('"').strip("'"))
            break
    assert value is not None, "DAY_STALL_MIN_ADVERSE_PCT must be set explicitly in core_only_local.env"
    assert value >= 0.0025, f"DAY_STALL_MIN_ADVERSE_PCT={value} disables P1A MAE floor (need >= 0.0025)"


@pytest.fixture(autouse=True)
def _stall_quality_defaults(monkeypatch):
    monkeypatch.setenv("DAY_STALL_EXIT_ENABLED", "true")
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


def test_1_red_younger_than_stall_min_no_exit():
    out = evaluate_stall_exit(
        entry_price=100.0,
        highest_price=100.05,
        lowest_price=99.50,
        current_price=99.50,
        net_pnl_pct=-0.006,
        hold_minutes=90.0,
        max_hold_min=360,
    )
    assert out is not None
    assert out["action"] == "hold"
    assert out["reason"] == STALL_HOLD_TOO_YOUNG


def test_2_red_low_mfe_adverse_confirmed_allows_stall():
    out = evaluate_stall_exit(
        entry_price=100.0,
        highest_price=100.10,  # MFE 0.10% < 0.50%
        lowest_price=99.60,  # MAE 0.40% >= 0.25%
        current_price=99.65,  # still adverse
        net_pnl_pct=-0.005,
        hold_minutes=125.0,
        max_hold_min=360,
    )
    assert out is not None
    assert out["action"] == "sell"
    assert out["reason"] == EXIT_STALL_DEAD


def test_3_red_low_mfe_but_recovery_present_holds():
    out = evaluate_stall_exit(
        entry_price=100.0,
        highest_price=100.10,
        lowest_price=99.50,
        current_price=99.95,  # reclaimed near entry
        net_pnl_pct=-0.002,
        hold_minutes=125.0,
        max_hold_min=360,
    )
    assert out is not None
    assert out["action"] == "hold"
    assert out["reason"] == STALL_HOLD_RECOVERY_PRESENT


def test_4_flat_slightly_red_not_worsening_holds():
    out = evaluate_stall_exit(
        entry_price=100.0,
        highest_price=100.05,
        lowest_price=99.90,  # MAE 0.10% < 0.25%
        current_price=99.92,
        net_pnl_pct=-0.0015,
        hold_minutes=130.0,
        max_hold_min=360,
    )
    assert out is not None
    assert out["action"] == "hold"
    assert out["reason"] == STALL_HOLD_FLAT_NOT_DEAD


def test_5_hard_stop_still_exits_immediately():
    pos = _Pos(
        entry_price=100.0,
        highest_price=100.10,
        lowest_price=98.90,
        stop_price=99.00,
        max_hold_min=360,
    )
    out = evaluate_engine_managed_exit(
        position=pos,
        current_price=98.90,
        net_pnl_pct=-0.012,
        hold_minutes=10.0,
        coin_profile={"max_hold_min": 360, "trail": 0.005, "sl": 0.01},
        bundle=None,
    )
    assert out["action"] == "sell"
    assert out["reason"] == EXIT_STOP_LOSS


def test_6_net_profit_still_exits_when_floor_met():
    pos = _Pos(
        entry_price=100.0,
        highest_price=100.80,
        lowest_price=99.90,
        take_profit_1_price=100.50,
        max_hold_min=360,
    )
    net = float(MIN_NET_PROFIT_TO_SELL) + 0.001
    out = evaluate_engine_managed_exit(
        position=pos,
        current_price=100.60,
        net_pnl_pct=net,
        hold_minutes=40.0,
        coin_profile={"max_hold_min": 360, "trail": 0.005, "sl": 0.01},
        bundle=None,
    )
    assert out["action"] == "sell"
    assert out["reason"] == EXIT_NET_PROFIT


def test_7_stall_does_not_force_green_before_profit_path():
    out = evaluate_stall_exit(
        entry_price=100.0,
        highest_price=100.40,
        lowest_price=99.80,
        current_price=100.30,
        net_pnl_pct=0.002,
        hold_minutes=130.0,
        max_hold_min=360,
    )
    assert out is not None
    assert out["action"] == "hold"
    assert out["reason"] == STALL_HOLD_NOT_RED

    pos = _Pos(
        entry_price=100.0,
        highest_price=100.40,
        lowest_price=99.80,
        max_hold_min=360,
    )
    engine = evaluate_engine_managed_exit(
        position=pos,
        current_price=100.30,
        net_pnl_pct=0.002,
        hold_minutes=130.0,
        coin_profile={"max_hold_min": 360, "trail": 0.005, "sl": 0.01},
        bundle=None,
    )
    assert engine["action"] != "sell" or engine.get("reason") != EXIT_STALL_DEAD
    assert engine.get("reason") != EXIT_STALL_DEAD
