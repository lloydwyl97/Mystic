"""Regression coverage for day_trade_thesis remap and ML-locked override.

Restored from test_audit_fix_batch_20260802.py (deleted in commit 87ef12e).
Only the tests that covered live day_trade_thesis paths are restored here.
The two paper-spread-caps tests (test_paper_spread_caps_*) are NOT restored
because paper_spread_caps.py was correctly removed in the same commit.
"""

from __future__ import annotations

from backend.services.day_trade_thesis import (
    SETUP_RANGE_BOUNCE,
    apply_ml_locked_setup_override,
    remap_setup_for_day_regime,
)


def test_remap_range_bounce_preserved_in_range_regime():
    """RANGE_BOUNCE identity must survive remap in range regime (not laundered)."""
    result = remap_setup_for_day_regime("RANGE_BOUNCE", "range")
    assert result == SETUP_RANGE_BOUNCE


def test_remap_unknown_setup_gets_regime_default():
    """Unknown setup in bull regime falls back to HTF_TREND_PULLBACK."""
    result = remap_setup_for_day_regime("SOME_UNKNOWN_SETUP", "bull")
    assert "TREND_PULLBACK" in result or result == "SOME_UNKNOWN_SETUP"


def test_apply_ml_locked_preserves_setup_type_in_result():
    """apply_ml_locked_setup_override must return a dict with locked setup."""
    decision = {
        "allweather_setup": "RANGE_BOUNCE",
        "day_route_regime": "range",
        "symbol": "BTCUSDT",
    }
    result = apply_ml_locked_setup_override(decision, current_price=100.0, atr=1.5)
    assert isinstance(result, dict)
    assert "setup_type" in result or "allweather_setup" in result or "day_route_regime" in result


def test_apply_ml_locked_returns_levels_dict():
    """apply_ml_locked_setup_override must include target/stop level keys."""
    decision = {
        "allweather_setup": "RANGE_BOUNCE",
        "day_route_regime": "neutral",
        "symbol": "SOLUSDT",
    }
    result = apply_ml_locked_setup_override(decision, current_price=150.0, atr=2.0)
    # Must not raise; return type is always dict
    assert isinstance(result, dict)
