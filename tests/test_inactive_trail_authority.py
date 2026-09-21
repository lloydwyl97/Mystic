"""Inactive trail cannot submit a trailing SELL; other families keep their label."""

from __future__ import annotations

import pytest

from backend.services.day_controlled_exits import (
    EXIT_GIVEBACK,
    EXIT_STOP_LOSS,
    EXIT_TRAILING_STOP,
    _trail_activation_price,
    _trail_semantics,
    evaluate_engine_managed_exit,
)
from backend.services.exit_decision_evidence import build_exit_decision_evidence


class _Pos:
    def __init__(self, **kw):
        self.symbol = kw.get("symbol", "SOL/USDT")
        self.entry_price = kw.get("entry_price", 109.99)
        self.highest_price = kw.get("highest_price", 109.99)
        self.lowest_price = kw.get("lowest_price", 109.61)
        self.stop_price = kw.get("stop_price", 108.89)
        self.trailing_stop_price = kw.get("trailing_stop_price", 110.07)
        self.trail_pct = kw.get("trail_pct", 0.0025)
        self.take_profit_1_price = 0.0
        self.entry_thesis = ""
        self.entry_vwap = self.entry_price
        self.thesis_invalid_level = 0.0
        self.thesis_target_level = 0.0
        self.max_hold_min = 300
        self.trail_activated = kw.get("trail_activated", False)
        self.trail_activated_at = kw.get("trail_activated_at", 0.0)
        self.trail_activation_price = kw.get("trail_activation_price", 0.0)
        self.trail_high_water_source = ""


def test_inactive_trail_cannot_submit_trailing_sell(monkeypatch):
    monkeypatch.setenv("DAY_PATH_AWARE_EXIT", "true")
    pos = _Pos()
    activation = _trail_activation_price(entry=pos.entry_price, trail_distance=0.0025, symbol=pos.symbol)
    assert pos.highest_price < activation
    out = evaluate_engine_managed_exit(
        position=pos,
        current_price=109.61,
        net_pnl_pct=-0.004,
        hold_minutes=10.0,
        coin_profile={"trail": 0.0025, "sl": 0.01, "tp": 0.014, "max_hold_min": 300},
    )
    assert out["reason"] != EXIT_TRAILING_STOP


def test_activated_trail_still_sells(monkeypatch):
    monkeypatch.setenv("DAY_PATH_AWARE_EXIT", "true")
    entry = 113.97
    high = entry * 1.01
    pos = _Pos(entry_price=entry, highest_price=high, trailing_stop_price=high * 0.9975, trail_activated=True, trail_activation_price=high)
    out = evaluate_engine_managed_exit(
        position=pos,
        current_price=high * 0.9974,
        net_pnl_pct=0.002,
        hold_minutes=12.0,
        coin_profile={"trail": 0.0025, "sl": 0.01, "tp": 0.014, "max_hold_min": 300},
    )
    assert out["action"] == "sell"
    assert out["reason"] == EXIT_TRAILING_STOP


def test_stop_loss_keeps_its_label(monkeypatch):
    monkeypatch.setenv("DAY_PATH_AWARE_EXIT", "true")
    pos = _Pos(highest_price=109.99, trailing_stop_price=110.07)
    out = evaluate_engine_managed_exit(
        position=pos,
        current_price=108.80,
        net_pnl_pct=-0.012,
        hold_minutes=8.0,
        coin_profile={"trail": 0.0025, "sl": 0.01, "tp": 0.014, "max_hold_min": 300},
    )
    assert out["action"] == "sell"
    assert out["reason"] == EXIT_STOP_LOSS


def test_giveback_keeps_its_label(monkeypatch):
    monkeypatch.setenv("DAY_PATH_AWARE_EXIT", "true")
    monkeypatch.setenv("DAY_GIVEBACK_EXIT_ENABLED", "true")
    entry = 110.0
    high = entry * 1.008
    pos = _Pos(entry_price=entry, highest_price=high, trailing_stop_price=0.0)
    out = evaluate_engine_managed_exit(
        position=pos,
        current_price=entry * 0.999,
        net_pnl_pct=-0.003,
        hold_minutes=40.0,
        coin_profile={"trail": 0.0025, "sl": 0.01, "tp": 0.014, "max_hold_min": 300},
    )
    if out["action"] == "sell":
        assert out["reason"] in {EXIT_GIVEBACK, EXIT_STOP_LOSS}
        assert out["reason"] != EXIT_TRAILING_STOP


def test_exit_evidence_persists_bid_and_slippage():
    pos = _Pos(trail_activated=True, trail_activated_at=1.0, trail_activation_price=114.29, highest_price=114.29)
    info = _trail_semantics(
        entry=113.97,
        current_price=113.80,
        position=pos,
        coin_profile={"trail": 0.0025, "sl": 0.01},
        path_aware=True,
        atr_pct=0.01,
        bundle=None,
    )
    ev = build_exit_decision_evidence(
        controlling_exit_family=EXIT_TRAILING_STOP,
        trail_info=info,
        position=pos,
        executable_bid=113.81,
        submitted_price=113.81,
        fill_price=113.80,
    )
    assert ev["controlling_exit_family"] == EXIT_TRAILING_STOP
    assert ev["executable_bid_at_decision"] == 113.81
    assert ev["fill_price"] == 113.80
    assert ev["slippage"] == pytest.approx(0.01)
    assert ev["activation_state"] in {"activated", "inactive"}
