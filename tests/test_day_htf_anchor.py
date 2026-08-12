"""Batch 5: HTF anchor soft demotion.

Guards:
1. Trend-long setup with strong 1h/4h bull alignment → high score, positive rank_delta, size >= 1.0.
2. Trend-long setup with 1h/4h bearish → low score, negative rank_delta, size < 1.0.
3. Range setup with strong trend HTF → penalized.
4. Reversal setup with bearish HTF → rewarded.
5. Missing MTF → neutral 0.5 credit per component (no punishment for data gap).
6. Never hard-blocks: candidate_eligible stays True in all cases.
7. thesis_size_factor / thesis_rank_delta compound multiplicatively / additively.
8. Env DAY_HTF_ANCHOR_ENABLED=false → no-op with score=0.5.
"""

from __future__ import annotations

import json

import pytest

from backend.services.day_htf_anchor import (
    SIZE_FACTOR_AT_ZERO,
    apply_htf_anchor_to_decision_data,
    compute_htf_anchor,
    htf_anchor_enabled,
)


def _mtf(h1: float, h4: float, m15: float | None = None) -> str:
    d = {"1h": {"ema_align": h1}, "4h": {"ema_align": h4}}
    if m15 is not None:
        d["15m"] = {"ema_align": m15}
    return json.dumps(d)


def test_trend_long_with_bull_htf_gets_bonus():
    dd = {
        "setup_type_canonical": "HTF_TREND_PULLBACK",
        "mtf_json": _mtf(0.72, 0.68),
        "price_momentum": 0.025,
        "ema_alignment": 0.80,
    }
    out = compute_htf_anchor(dd)
    assert out["htf_anchor_score"] >= 0.85
    assert out["htf_anchor_state"] == "htf_aligned"
    assert out["htf_anchor_rank_delta"] > 0.0
    assert out["htf_anchor_size_factor"] > 1.0


def test_trend_long_with_bear_htf_gets_penalty():
    dd = {
        "setup_type_canonical": "HTF_TREND_PULLBACK",
        "mtf_json": _mtf(0.28, 0.30),
        "price_momentum": -0.03,
        "ema_alignment": 0.25,
    }
    out = compute_htf_anchor(dd)
    assert out["htf_anchor_score"] <= 0.30
    assert out["htf_anchor_state"] == "htf_counter_setup"
    assert out["htf_anchor_rank_delta"] < 0.0
    assert out["htf_anchor_size_factor"] < 0.75


def test_range_setup_with_strong_trend_htf_penalized():
    dd = {
        "setup_type_canonical": "RANGE_BOUNCE",
        "mtf_json": _mtf(0.75, 0.72),
        "price_momentum": 0.05,
        "ema_alignment": 0.85,
    }
    out = compute_htf_anchor(dd)
    assert out["htf_anchor_score"] < 0.55


def test_range_setup_with_flat_htf_gets_bonus():
    dd = {
        "setup_type_canonical": "RANGE_BOUNCE",
        "mtf_json": _mtf(0.50, 0.51),
        "price_momentum": 0.001,
        "ema_alignment": 0.50,
    }
    out = compute_htf_anchor(dd)
    assert out["htf_anchor_score"] >= 0.80


def test_reversal_setup_with_bearish_htf_gets_bonus():
    dd = {
        "setup_type_canonical": "FAILED_BREAKDOWN_REVERSAL",
        "mtf_json": _mtf(0.30, 0.28),
        "price_momentum": -0.01,
        "ema_alignment": 0.30,
    }
    out = compute_htf_anchor(dd)
    assert out["htf_anchor_score"] >= 0.75
    assert out["htf_anchor_family"] == "reversal"


def test_missing_mtf_yields_neutral():
    dd = {
        "setup_type_canonical": "HTF_TREND_PULLBACK",
        # no mtf_json
        "price_momentum": 0.0,
        "ema_alignment": 0.5,
    }
    out = compute_htf_anchor(dd)
    # h1/h4 unknown → 0.5 credit; momentum flat → 0.5; ema neutral → ~0.5
    assert 0.45 <= out["htf_anchor_score"] <= 0.65


def test_never_hard_blocks():
    dd = {
        "setup_type_canonical": "HTF_TREND_PULLBACK",
        "mtf_json": _mtf(0.20, 0.20),
        "price_momentum": -0.05,
        "ema_alignment": 0.10,
    }
    out = apply_htf_anchor_to_decision_data(dd)
    assert out["candidate_eligible"] is True
    assert out.get("hard_block") is False


def test_multiplicative_size_and_additive_rank():
    dd = {
        "setup_type_canonical": "HTF_TREND_PULLBACK",
        "mtf_json": _mtf(0.72, 0.68),
        "price_momentum": 0.025,
        "ema_alignment": 0.80,
        "thesis_size_factor": 0.80,
        "thesis_rank_delta": -0.05,
    }
    out = apply_htf_anchor_to_decision_data(dd)
    anchor_size = float(out["htf_anchor_size_factor"])
    assert out["thesis_size_factor"] == pytest.approx(max(SIZE_FACTOR_AT_ZERO, 0.80 * anchor_size), rel=1e-4)
    assert out["thesis_rank_delta"] == pytest.approx(-0.05 + float(out["htf_anchor_rank_delta"]), rel=1e-4)


def test_disable_flag(monkeypatch):
    monkeypatch.setenv("DAY_HTF_ANCHOR_ENABLED", "false")
    assert htf_anchor_enabled() is False
    dd = {
        "setup_type_canonical": "HTF_TREND_PULLBACK",
        "mtf_json": _mtf(0.20, 0.20),
        "thesis_size_factor": 0.80,
        "thesis_rank_delta": -0.02,
    }
    out = apply_htf_anchor_to_decision_data(dd)
    assert out["htf_anchor_enabled"] is False
    assert out["htf_anchor_score"] == 0.5
    assert out["thesis_size_factor"] == 0.80
    assert out["thesis_rank_delta"] == -0.02
