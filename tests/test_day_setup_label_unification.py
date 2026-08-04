"""P1D: setup identity unification for learning/ranking (label consistency only)."""

from __future__ import annotations

from backend.services.day_trade_thesis import (
    SETUP_FAILED_BREAKDOWN_REVERSAL,
    SETUP_HTF_TREND_PULLBACK,
    SETUP_RANGE_BOUNCE,
    apply_ml_locked_setup_override,
    remap_setup_for_day_regime,
    resolve_setup_identity,
)


def test_resolve_setup_identity_prefers_canonical_over_narrative_htf():
    dd = {
        "setup_type": SETUP_RANGE_BOUNCE,
        "entry_thesis": SETUP_RANGE_BOUNCE,
        "setup_regime_remapped_from": SETUP_HTF_TREND_PULLBACK,
        "day_route_regime": "range",
        "adaptive_regime": "trending_up::HTF_TREND_PULLBACK",
        "candidate_explanation_narrative": "setup=HTF_TREND_PULLBACK in bull regime",
    }
    ident = resolve_setup_identity(dd)
    assert ident["setup_type_canonical"] == SETUP_RANGE_BOUNCE
    assert ident["setup_type_raw"] == SETUP_HTF_TREND_PULLBACK
    assert ident["entry_thesis"] == SETUP_RANGE_BOUNCE
    assert ident["day_route_regime"] == "range"
    # Hybrid adaptive string must not become the setup or pollute regime.
    assert "::" not in ident["adaptive_regime"]
    assert ident["setup_type"] == SETUP_RANGE_BOUNCE


def test_resolve_setup_identity_fbr_raw_preserved():
    dd = {
        "setup_type": SETUP_FAILED_BREAKDOWN_REVERSAL,
        "setup_type_raw": SETUP_HTF_TREND_PULLBACK,
        "day_route_regime": "bear",
    }
    ident = resolve_setup_identity(dd)
    assert ident["setup_type_canonical"] == SETUP_FAILED_BREAKDOWN_REVERSAL
    assert ident["setup_type_raw"] == SETUP_HTF_TREND_PULLBACK


def test_locked_override_stamps_canonical_and_raw():
    dd = {
        "setup_type": SETUP_HTF_TREND_PULLBACK,
        "entry_thesis": SETUP_HTF_TREND_PULLBACK,
        "day_route_regime": "range",
        "regime": "range",
    }
    out = apply_ml_locked_setup_override(dd, current_price=100.0, atr=1.0)
    assert out["setup_type"] == SETUP_RANGE_BOUNCE
    assert out["setup_type_canonical"] == SETUP_RANGE_BOUNCE
    assert out["setup_type_raw"] == SETUP_HTF_TREND_PULLBACK
    assert out["entry_thesis"] == SETUP_RANGE_BOUNCE
    assert out["day_route_regime"] == "range"
    # Ranking and learning share the same key.
    ident = resolve_setup_identity(out)
    assert ident["setup_type_canonical"] == out["setup_type"]
    assert ident["setup_type_canonical"] == SETUP_RANGE_BOUNCE


def test_remap_deterministic_for_range_and_bear():
    assert remap_setup_for_day_regime(SETUP_HTF_TREND_PULLBACK, "range") == SETUP_RANGE_BOUNCE
    assert remap_setup_for_day_regime(SETUP_HTF_TREND_PULLBACK, "bear") == SETUP_FAILED_BREAKDOWN_REVERSAL
    assert remap_setup_for_day_regime(SETUP_FAILED_BREAKDOWN_REVERSAL, "range") == SETUP_FAILED_BREAKDOWN_REVERSAL
