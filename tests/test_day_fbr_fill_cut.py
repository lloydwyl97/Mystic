"""DAY FBR fill cut — stop minting the bleed bucket."""

from __future__ import annotations

import os

from backend.services.day_trade_thesis import (
    SETUP_BREAKOUT_CONTINUATION,
    SETUP_FAILED_BREAKDOWN_REVERSAL,
    SETUP_HTF_TREND_PULLBACK,
    SETUP_RANGE_BOUNCE,
    apply_ml_locked_setup_override,
    remap_setup_for_day_regime,
)
from backend.services.symbol_setup_outcome_penalty import (
    day_fbr_fills_enabled,
    day_htf_fills_enabled,
    should_defer_day_fbr_fill,
    should_defer_day_htf_fill,
)


def test_bear_no_longer_mints_fbr_from_htf():
    assert remap_setup_for_day_regime(SETUP_HTF_TREND_PULLBACK, "bear") == SETUP_RANGE_BOUNCE
    assert remap_setup_for_day_regime(SETUP_BREAKOUT_CONTINUATION, "bear") == SETUP_RANGE_BOUNCE


def test_bear_default_lock_is_range_bounce():
    dd = apply_ml_locked_setup_override(
        {"day_route_regime": "bear", "setup_type": "", "entry_thesis": ""},
        current_price=100.0,
        atr=1.0,
    )
    assert dd["setup_type"] == SETUP_RANGE_BOUNCE
    assert dd["setup_type"] != SETUP_FAILED_BREAKDOWN_REVERSAL


def test_fbr_fills_disabled_by_default(monkeypatch):
    monkeypatch.delenv("DAY_FBR_FILLS_ENABLED", raising=False)
    assert day_fbr_fills_enabled() is False
    assert should_defer_day_fbr_fill({"setup_type": SETUP_FAILED_BREAKDOWN_REVERSAL}) is True
    assert should_defer_day_fbr_fill({"setup_type": SETUP_RANGE_BOUNCE}) is False


def test_fbr_fills_can_reenable(monkeypatch):
    monkeypatch.setenv("DAY_FBR_FILLS_ENABLED", "true")
    assert day_fbr_fills_enabled() is True
    assert should_defer_day_fbr_fill({"setup_type": SETUP_FAILED_BREAKDOWN_REVERSAL}) is False


def test_htf_fills_disabled_by_default(monkeypatch):
    monkeypatch.delenv("DAY_HTF_FILLS_ENABLED", raising=False)
    assert day_htf_fills_enabled() is False
    assert should_defer_day_htf_fill({"setup_type": SETUP_HTF_TREND_PULLBACK}) is True
    assert should_defer_day_htf_fill({"setup_type": "TREND_PULLBACK"}) is True
    assert should_defer_day_htf_fill({"setup_type": SETUP_RANGE_BOUNCE}) is False
    assert should_defer_day_htf_fill({"setup_type": SETUP_BREAKOUT_CONTINUATION}) is False


def test_htf_fills_can_reenable(monkeypatch):
    monkeypatch.setenv("DAY_HTF_FILLS_ENABLED", "true")
    assert day_htf_fills_enabled() is True
    assert should_defer_day_htf_fill({"setup_type": SETUP_HTF_TREND_PULLBACK}) is False


def test_bull_default_is_breakout_not_htf():
    dd = apply_ml_locked_setup_override(
        {"day_route_regime": "bull", "setup_type": "", "entry_thesis": ""},
        current_price=100.0,
        atr=1.0,
    )
    assert dd["setup_type"] == SETUP_BREAKOUT_CONTINUATION


def test_bull_keeps_range_bounce():
    assert remap_setup_for_day_regime(SETUP_RANGE_BOUNCE, "bull") == SETUP_RANGE_BOUNCE
