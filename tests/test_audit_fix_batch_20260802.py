"""Regression tests for the 2026-08-02 full-app audit fix batch."""

from __future__ import annotations

from backend.services.binance_scalp.paper_spread_caps import (
    DEFAULT_PAPER_SPREAD_CAPS,
    _repair_bash_stripped_json_object,
    parse_paper_spread_caps_json,
)
from backend.services.day_trade_thesis import (
    SETUP_FAILED_BREAKDOWN_REVERSAL,
    SETUP_HTF_TREND_PULLBACK,
    SETUP_RANGE_BOUNCE,
    apply_ml_locked_setup_override,
    remap_setup_for_day_regime,
)


def test_repair_bash_stripped_spread_json():
    stripped = "{BTCUSDT:0.0008,ETHUSDT:0.0006,SOLUSDT:0.0005,XRPUSDT:0.0009}"
    repaired = _repair_bash_stripped_json_object(stripped)
    assert '"BTCUSDT"' in repaired
    caps = parse_paper_spread_caps_json(stripped)
    assert caps["BTCUSDT"] == 0.0008
    assert caps["XRPUSDT"] == 0.0009


def test_parse_spread_caps_falls_back_on_garbage():
    caps = parse_paper_spread_caps_json("not-json{{{{")
    assert caps == DEFAULT_PAPER_SPREAD_CAPS


def test_fbr_preserved_in_range_remap():
    assert remap_setup_for_day_regime(SETUP_FAILED_BREAKDOWN_REVERSAL, "range") == SETUP_FAILED_BREAKDOWN_REVERSAL
    assert remap_setup_for_day_regime(SETUP_FAILED_BREAKDOWN_REVERSAL, "neutral") == SETUP_FAILED_BREAKDOWN_REVERSAL
    assert remap_setup_for_day_regime(SETUP_HTF_TREND_PULLBACK, "range") == SETUP_RANGE_BOUNCE


def test_apply_ml_locked_keeps_fbr_and_unifies_narrative():
    dd = apply_ml_locked_setup_override(
        {
            "setup_type": SETUP_FAILED_BREAKDOWN_REVERSAL,
            "allweather_setup": SETUP_FAILED_BREAKDOWN_REVERSAL,
            "entry_thesis": SETUP_FAILED_BREAKDOWN_REVERSAL,
            "regime": "range",
            "candidate_explanation_narrative": "prior note",
        },
        current_price=100.0,
        atr=1.0,
    )
    assert dd["setup_type"] == SETUP_FAILED_BREAKDOWN_REVERSAL
    assert dd["allweather_setup"] == SETUP_FAILED_BREAKDOWN_REVERSAL
    narr = str(dd.get("candidate_explanation_narrative") or "")
    assert f"setup={SETUP_FAILED_BREAKDOWN_REVERSAL}" in narr
