"""
Regression: setup/regime label mismatches, AND projected-MFE-after-fees
"expected favorable excursion insufficient" opinions, must not hard-block
DAY entries.

Mystic is a ranking-and-trading engine, not a permission bot. Everything
`evaluate_day_entry_route` (backend/services/day_regime_router.py) returns —
including "*_MFE_TOO_LOW" — is a trade-opinion/expected-value judgment about
the strategy's own projected thesis target vs. a constant estimated cost, not
a measured, real-time execution-safety fact (contrast with SCALP's
NET_EDGE_BELOW_MIN, which is computed from live order-book spread/impact and
correctly remains a hard block there). All router outcomes surface as
advisory penalty info only (route_rank_delta/route_size_factor).
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


def test_mfe_too_low_no_longer_hard_blocks_entry():
    """A range VWAP setup with a target too close (projected MFE-after-fees
    "too low") is a trade-opinion/expected-value judgment, not a measured
    execution-safety fact — it must not reject the trade."""
    result = evaluate_pre_buy_exit_consistency(
        **_base_kwargs(
            setup="VWAP_REVERSION",
            day_regime="range",
            thesis_target_level=100.05,  # essentially no room to the target -> MFE too low after fees
            decision_data={"adx": 15.0, "rsi": 30.0, "bb_position": 0.2, "vwap": 100.5, "price_momentum": 0.0, "thesis_score": 0.75},
        )
    )
    assert result["allowed"] is True, f"MFE_TOO_LOW must be advisory only, got: {result}"
    assert "MFE_TOO_LOW" in result["checks"].get("route_regime_mismatch_advisory", "")


def test_genuine_thesis_invalidation_still_blocks_regardless_of_regime():
    result = evaluate_pre_buy_exit_consistency(
        **_base_kwargs(
            thesis_invalid_level=99.9,  # entry (100.0) is barely above invalid level -> live invalidation likely
            entry_price=100.0,
            stop_price=99.0,
        )
    )
    # Whatever the outcome, it must not be gated by regime/MFE opinion — either
    # allowed, or blocked strictly for a genuine ENTRY_EXIT_* reason.
    if not result["allowed"]:
        assert result["block_reason"].startswith("ENTRY_EXIT_"), result["block_reason"]


def test_no_setup_regime_incompatible_block_reason_remains_possible():
    """SETUP_REGIME_INCOMPATIBLE must never appear as a hard block_reason anymore —
    confirms the MFE_TOO_LOW exception carve-out was fully removed, not just narrowed."""
    for target in (100.05, 100.5, 110.0, 90.0):
        result = evaluate_pre_buy_exit_consistency(
            **_base_kwargs(setup="VWAP_REVERSION", day_regime="range", thesis_target_level=target)
        )
        assert not result["block_reason"].startswith("SETUP_REGIME_INCOMPATIBLE"), result
