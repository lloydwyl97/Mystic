"""DAY setup identity + ranking eligibility (hard fill gates removed)."""

from __future__ import annotations

from backend.services.day_trade_thesis import (
    SETUP_BREAKOUT_CONTINUATION,
    SETUP_FAILED_BREAKDOWN_REVERSAL,
    SETUP_HTF_TREND_PULLBACK,
    SETUP_RANGE_BOUNCE,
    apply_ml_locked_setup_override,
    remap_setup_for_day_regime,
)


def test_htf_and_fbr_identity_preserved_across_regimes():
    assert remap_setup_for_day_regime(SETUP_HTF_TREND_PULLBACK, "bear") == SETUP_HTF_TREND_PULLBACK
    assert remap_setup_for_day_regime(SETUP_HTF_TREND_PULLBACK, "range") == SETUP_HTF_TREND_PULLBACK
    assert remap_setup_for_day_regime(SETUP_FAILED_BREAKDOWN_REVERSAL, "range") == SETUP_FAILED_BREAKDOWN_REVERSAL
    assert remap_setup_for_day_regime(SETUP_BREAKOUT_CONTINUATION, "bear") == SETUP_BREAKOUT_CONTINUATION


def test_bear_default_lock_is_range_bounce():
    dd = apply_ml_locked_setup_override(
        {"day_route_regime": "bear", "setup_type": "", "entry_thesis": ""},
        current_price=100.0,
        atr=1.0,
    )
    assert dd["setup_type"] == SETUP_RANGE_BOUNCE
    assert dd["setup_type"] != SETUP_FAILED_BREAKDOWN_REVERSAL


def test_bull_default_is_htf():
    dd = apply_ml_locked_setup_override(
        {"day_route_regime": "bull", "setup_type": "", "entry_thesis": ""},
        current_price=100.0,
        atr=1.0,
    )
    assert dd["setup_type"] == SETUP_HTF_TREND_PULLBACK


def test_bull_keeps_range_bounce():
    assert remap_setup_for_day_regime(SETUP_RANGE_BOUNCE, "bull") == SETUP_RANGE_BOUNCE
