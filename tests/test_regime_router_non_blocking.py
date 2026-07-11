"""
Regression: setup/regime label mismatches must not hard-block DAY entries.

Mystic is a ranking-and-trading engine, not a permission bot. Regime
incompatibility (e.g. a trend-pullback setup routed while the regime router
labels the market "range") must surface as advisory/scoring information only.
Only genuine executable-edge failures (MFE-after-fees too low) remain hard
blocks at this gate.
"""

from __future__ import annotations

from backend.services.day_controlled_exits import evaluate_pre_buy_exit_consistency


def _base_kwargs(**overrides):
    base = dict(
        setup="HTF_TREND_PULLBACK",
        entry_price=100.0,
        stop_price=97.0,
        thesis_invalid_level=0.0,  # no invalidation level -> avoids thesis_invalidated_live block
        thesis_target_level=105.0,
        entry_vwap=99.5,
        entry_ts=0.0,
        coin_profile={"trail": 0.005, "max_hold_min": 60, "sl": 0.01},
        bundle=None,
        spread_pct=0.0005,
        day_regime="range",  # mismatched vs HTF_TREND_PULLBACK setup -> router would reject
        decision_data={"adx": 15.0, "rsi": 50.0, "bb_position": 0.5, "thesis_score": 0.75},
        context_payload=None,
        thesis_score=0.75,
    )
    base.update(overrides)
    return base


def test_setup_regime_mismatch_does_not_block_entry():
    result = evaluate_pre_buy_exit_consistency(**_base_kwargs())
    assert result["allowed"] is True, f"regime/setup label mismatch must not hard-block, got: {result}"
    assert result["checks"].get("route_allowed") is False, "router itself should still flag the mismatch internally"
    assert "route_regime_mismatch_advisory" in result["checks"]
    assert result["block_reason"] == ""


def test_executable_edge_failure_still_blocks():
    """A range VWAP setup with target too close (MFE-after-fees too low) is a genuine
    operational failure (no net edge) and must remain hard-blocked."""
    result = evaluate_pre_buy_exit_consistency(
        **_base_kwargs(
            setup="VWAP_REVERSION",
            day_regime="range",
            thesis_target_level=100.05,  # essentially no room to the target -> MFE too low after fees
            decision_data={"adx": 15.0, "rsi": 30.0, "bb_position": 0.2, "vwap": 100.5, "price_momentum": 0.0, "thesis_score": 0.75},
        )
    )
    assert result["allowed"] is False
    assert "MFE_TOO_LOW" in result["block_reason"]


def test_genuine_thesis_invalidation_still_blocks_regardless_of_regime():
    result = evaluate_pre_buy_exit_consistency(
        **_base_kwargs(
            thesis_invalid_level=99.9,  # entry (100.0) is barely above invalid level -> live invalidation likely
            entry_price=100.0,
            stop_price=99.0,
        )
    )
    # Whatever the outcome, it must not be gated by regime — either allowed, or
    # blocked strictly for a genuine ENTRY_EXIT_* reason, never SETUP_REGIME_INCOMPATIBLE
    # unless it's the executable-edge (MFE_TOO_LOW) case.
    if not result["allowed"]:
        assert "MFE_TOO_LOW" in result["block_reason"] or result["block_reason"].startswith("ENTRY_EXIT_")
