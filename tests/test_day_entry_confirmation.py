"""Batch 4: entry-confirmation soft demotion.

Guards:
1. All checks green → score 1.0, no rank penalty, size 1.0.
2. All checks red → score ≈ 0.0, negative rank delta, size at floor (0.20).
3. Missing data yields neutral (0.5) credit — never punishes for a data gap.
4. Never hard-blocks: `candidate_eligible` stays True in all cases.
5. `thesis_size_factor` is multiplied down, not overwritten.
6. `thesis_rank_delta` accumulates the confirmation delta.
7. Disable via env DAY_ENTRY_CONFIRMATION_ENABLED=false → no-op.
"""

from __future__ import annotations

import pytest

from backend.services.day_entry_confirmation import (
    SIZE_FACTOR_AT_ZERO,
    apply_entry_confirmation_to_decision_data,
    compute_entry_confirmation,
    entry_confirmation_enabled,
)


def test_all_green_scores_one():
    dd = {
        "entry_vwap": 100.0,
        "candle_body_pct": 0.70,
        "candle_upper_wick_pct": 0.10,
        "candle_lower_wick_pct": 0.20,
        "signal_ema_alignment": 0.85,
        "signal_adx": 24.0,
        "setup_type_canonical": "HTF_TREND_PULLBACK",
    }
    out = compute_entry_confirmation(dd, current_price=100.25)
    assert out["entry_confirmation_score"] == pytest.approx(1.0, abs=1e-6)
    assert out["entry_confirmation_state"] == "confirmed"
    assert out["entry_confirmation_rank_delta"] == pytest.approx(0.0, abs=1e-6)
    assert out["entry_confirmation_size_factor"] == pytest.approx(1.0, abs=1e-6)


def test_all_red_scores_low_and_size_at_floor():
    dd = {
        "entry_vwap": 100.0,
        "candle_body_pct": 0.15,
        "candle_upper_wick_pct": 0.70,
        "candle_lower_wick_pct": 0.15,
        "signal_ema_alignment": 0.10,
        "signal_adx": 8.0,
        "setup_type_canonical": "HTF_TREND_PULLBACK",
    }
    out = compute_entry_confirmation(dd, current_price=99.80)  # below VWAP
    assert out["entry_confirmation_score"] < 0.25
    assert out["entry_confirmation_state"] == "contradicted"
    assert out["entry_confirmation_rank_delta"] < 0.0
    assert out["entry_confirmation_size_factor"] == pytest.approx(
        SIZE_FACTOR_AT_ZERO, abs=0.05
    )


def test_missing_data_yields_neutral():
    dd = {"setup_type_canonical": "RANGE_BOUNCE"}
    out = compute_entry_confirmation(dd, current_price=100.0)
    # No entry_vwap, no candle, no ema, no adx → each check returns neutral 0.5
    assert 0.40 <= out["entry_confirmation_score"] <= 0.60


def test_never_hard_blocks():
    dd = {
        "entry_vwap": 100.0,
        "candle_body_pct": 0.10,
        "candle_upper_wick_pct": 0.80,
        "signal_ema_alignment": 0.05,
        "signal_adx": 6.0,
        "setup_type_canonical": "HTF_TREND_PULLBACK",
    }
    out = apply_entry_confirmation_to_decision_data(dd, current_price=99.5)
    assert out["candidate_eligible"] is True
    assert out.get("hard_block") is False


def test_applies_multiplicative_size_and_additive_rank():
    dd = {
        "entry_vwap": 100.0,
        "candle_body_pct": 0.35,
        "candle_upper_wick_pct": 0.35,
        "signal_ema_alignment": 0.30,
        "signal_adx": 14.0,
        "setup_type_canonical": "HTF_TREND_PULLBACK",
        "thesis_size_factor": 0.80,
        "thesis_rank_delta": -0.05,
    }
    out = apply_entry_confirmation_to_decision_data(dd, current_price=99.90)
    # Confirmation score should be middling (~0.4), size factor ~0.5
    conf_size = float(out["entry_confirmation_size_factor"])
    assert 0.30 <= conf_size <= 0.75
    assert out["thesis_size_factor"] == pytest.approx(
        max(SIZE_FACTOR_AT_ZERO, 0.80 * conf_size), rel=1e-4
    )
    assert out["thesis_rank_delta"] == pytest.approx(
        -0.05 + float(out["entry_confirmation_rank_delta"]), rel=1e-4
    )


def test_disable_flag_produces_noop(monkeypatch):
    monkeypatch.setenv("DAY_ENTRY_CONFIRMATION_ENABLED", "false")
    assert entry_confirmation_enabled() is False
    dd = {
        "entry_vwap": 100.0,
        "candle_body_pct": 0.10,
        "candle_upper_wick_pct": 0.80,
        "signal_ema_alignment": 0.05,
        "signal_adx": 6.0,
        "setup_type_canonical": "HTF_TREND_PULLBACK",
        "thesis_size_factor": 0.80,
        "thesis_rank_delta": -0.02,
    }
    out = apply_entry_confirmation_to_decision_data(dd, current_price=99.5)
    assert out["entry_confirmation_enabled"] is False
    assert out["entry_confirmation_score"] == 1.0
    # No mutation of thesis_size_factor or thesis_rank_delta
    assert out["thesis_size_factor"] == 0.80
    assert out["thesis_rank_delta"] == -0.02


def test_range_setup_prefers_low_adx():
    dd_range_ok = {
        "entry_vwap": 100.0,
        "candle_body_pct": 0.55,
        "candle_upper_wick_pct": 0.20,
        "signal_ema_alignment": 0.60,
        "signal_adx": 15.0,
        "setup_type_canonical": "RANGE_BOUNCE",
    }
    dd_range_choppy_bad = dict(dd_range_ok)
    dd_range_choppy_bad["signal_adx"] = 40.0  # too trendy for a range setup
    ok = compute_entry_confirmation(dd_range_ok, current_price=100.05)
    bad = compute_entry_confirmation(dd_range_choppy_bad, current_price=100.05)
    assert ok["entry_confirmation_score"] > bad["entry_confirmation_score"]


def test_trend_setup_requires_higher_adx():
    dd_trend_ok = {
        "entry_vwap": 100.0,
        "candle_body_pct": 0.55,
        "candle_upper_wick_pct": 0.20,
        "signal_ema_alignment": 0.60,
        "signal_adx": 26.0,
        "setup_type_canonical": "HTF_TREND_PULLBACK",
    }
    dd_trend_dead = dict(dd_trend_ok)
    dd_trend_dead["signal_adx"] = 10.0
    ok = compute_entry_confirmation(dd_trend_ok, current_price=100.05)
    dead = compute_entry_confirmation(dd_trend_dead, current_price=100.05)
    assert ok["entry_confirmation_score"] > dead["entry_confirmation_score"]
