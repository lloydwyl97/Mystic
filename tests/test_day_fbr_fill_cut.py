"""DAY hard fill gates retired — FBR/HTF remain soft-demoted but fill-eligible."""

from __future__ import annotations

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
    should_defer_low_mfe_stall_fill,
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


def test_fbr_htf_hard_defer_retired(monkeypatch):
    monkeypatch.delenv("DAY_FBR_FILLS_ENABLED", raising=False)
    monkeypatch.delenv("DAY_HTF_FILLS_ENABLED", raising=False)
    assert day_fbr_fills_enabled() is True
    assert day_htf_fills_enabled() is True
    assert should_defer_day_fbr_fill({"setup_type": SETUP_FAILED_BREAKDOWN_REVERSAL}) is False
    assert should_defer_day_htf_fill({"setup_type": SETUP_HTF_TREND_PULLBACK}) is False
    assert should_defer_day_htf_fill({"setup_type": "TREND_PULLBACK"}) is False


def test_bull_default_is_htf():
    dd = apply_ml_locked_setup_override(
        {"day_route_regime": "bull", "setup_type": "", "entry_thesis": ""},
        current_price=100.0,
        atr=1.0,
    )
    assert dd["setup_type"] == SETUP_HTF_TREND_PULLBACK


def test_bull_keeps_range_bounce():
    assert remap_setup_for_day_regime(SETUP_RANGE_BOUNCE, "bull") == SETUP_RANGE_BOUNCE


def test_low_mfe_never_hard_fill_deferred():
    for setup in (
        SETUP_RANGE_BOUNCE,
        SETUP_BREAKOUT_CONTINUATION,
        SETUP_HTF_TREND_PULLBACK,
        SETUP_FAILED_BREAKDOWN_REVERSAL,
        "VWAP_REVERSION",
    ):
        dd = {
            "setup_type": setup,
            "outcome_low_mfe_stall_penalty_applied": True,
            "penalty_reason": "repeated_low_mfe_stall_losses",
            "final_selection_score": -0.20,
            "low_mfe_stall_count": 5,
        }
        assert should_defer_low_mfe_stall_fill(dd) is False, setup
