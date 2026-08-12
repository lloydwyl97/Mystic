"""Batch 6: liquidity / spread quality gate.

Guards:
1. Tight spread + balanced depth → score 1.0, no rank penalty, no hard block.
2. Wide spread → soft demotion + non-zero rank_delta / size_factor < 1.
3. Catastrophic spread (>= 4x typical, floor 30 bps) → hard block set.
4. Depth imbalance heavy skew → soft demotion.
5. Missing spread / depth data → neutral credit (never punishes for missing).
6. Feature can be disabled via env DAY_LIQUIDITY_GATE_ENABLED=false.
7. Hard-block half can be independently disabled via DAY_LIQUIDITY_HARD_BLOCK_ENABLED=false.
8. Symbol-specific typical spread thresholds honored via env override.
"""

from __future__ import annotations

import pytest

from backend.services.day_liquidity_gate import (
    SIZE_FACTOR_AT_ZERO,
    apply_liquidity_gate_to_decision_data,
    compute_liquidity_quality,
    liquidity_gate_enabled,
    liquidity_hard_block_enabled,
)


def test_tight_spread_balanced_depth_full_score():
    dd = {"spread_pct": 0.00025, "signal_ctx_depth_imbalance": 0.51}
    out = compute_liquidity_quality(dd, "BTC/USDT")
    assert out["liquidity_quality_score"] == pytest.approx(1.0, abs=1e-4)
    assert out["liquidity_hard_blocked"] is False


def test_wide_spread_soft_demote():
    # ETH typical 3 bps → 9 bps = 3x → "spread_wide" credit 0.35
    dd = {"spread_pct": 0.0009, "signal_ctx_depth_imbalance": 0.50}
    out = compute_liquidity_quality(dd, "ETH/USDT")
    assert 0.30 <= out["liquidity_quality_score"] <= 0.65
    assert out["liquidity_quality_rank_delta"] < 0.0
    assert out["liquidity_quality_size_factor"] < 1.0
    assert out["liquidity_hard_blocked"] is False


def test_catastrophic_spread_hard_blocks():
    # BTC typical 3 bps, catastrophic = max(30, 3*4=12) = 30 bps → set 40 bps
    dd = {"spread_pct": 0.0040, "signal_ctx_depth_imbalance": 0.5}
    out = compute_liquidity_quality(dd, "BTC/USDT")
    assert out["liquidity_hard_blocked"] is True
    assert "SPREAD_CATASTROPHIC" in out["liquidity_hard_block_reason"]


def test_depth_extreme_skew_penalized():
    dd = {"spread_pct": 0.00025, "signal_ctx_depth_imbalance": 0.02}
    out = compute_liquidity_quality(dd, "BTC/USDT")
    # spread credit ≈ 1.0, depth extreme ≈ 0.1 → weighted ≈ 0.73
    assert 0.60 <= out["liquidity_quality_score"] <= 0.90
    assert out["liquidity_quality_size_factor"] < 1.0


def test_unknown_spread_and_depth_neutral():
    dd = {}
    out = compute_liquidity_quality(dd, "BTC/USDT")
    # Both components fall to their unknown-neutral credits
    assert 0.5 <= out["liquidity_quality_score"] <= 0.75
    assert out["liquidity_hard_blocked"] is False


def test_disable_full_gate(monkeypatch):
    monkeypatch.setenv("DAY_LIQUIDITY_GATE_ENABLED", "false")
    assert liquidity_gate_enabled() is False
    dd = {"spread_pct": 0.005, "signal_ctx_depth_imbalance": 0.01}
    out = compute_liquidity_quality(dd, "BTC/USDT")
    assert out["liquidity_gate_enabled"] is False
    assert out["liquidity_quality_score"] == 1.0
    assert out["liquidity_hard_blocked"] is False


def test_disable_hard_block_only(monkeypatch):
    monkeypatch.setenv("DAY_LIQUIDITY_HARD_BLOCK_ENABLED", "false")
    assert liquidity_hard_block_enabled() is False
    dd = {"spread_pct": 0.010, "signal_ctx_depth_imbalance": 0.5}
    out = compute_liquidity_quality(dd, "BTC/USDT")
    assert out["liquidity_hard_blocked"] is False
    # Soft demotion still fires (spread credit is 0.0 at 33x typical, but
    # balanced depth 0.5 pulls weighted score up to ~0.30).
    assert out["liquidity_quality_score"] <= 0.40
    assert out["liquidity_quality_rank_delta"] < 0.0


def test_symbol_specific_env_override(monkeypatch):
    monkeypatch.setenv("DAY_LIQUIDITY_TYPICAL_SPREAD_BPS_BTC_USDT", "10.0")
    # 6 bps < 10 → tight (would have been "elevated" at default 3 bps typical)
    dd = {"spread_pct": 0.0006, "signal_ctx_depth_imbalance": 0.5}
    out = compute_liquidity_quality(dd, "BTC/USDT")
    assert out["liquidity_quality_score"] >= 0.85


def test_apply_liquidity_gate_hard_block_sets_hard_block_flag():
    dd = {"spread_pct": 0.005, "signal_ctx_depth_imbalance": 0.5, "thesis_size_factor": 1.0}
    out = apply_liquidity_gate_to_decision_data(dd, "BTC/USDT")
    assert out["hard_block"] is True
    assert out["candidate_eligible"] is False
    assert out["liquidity_hard_blocked"] is True


def test_apply_liquidity_gate_soft_case_never_hard_blocks():
    dd = {"spread_pct": 0.0006, "signal_ctx_depth_imbalance": 0.30, "thesis_size_factor": 1.0}
    out = apply_liquidity_gate_to_decision_data(dd, "BTC/USDT")
    assert out.get("hard_block", False) is False
    assert out.get("candidate_eligible", True) is True
    assert out["liquidity_hard_blocked"] is False


def test_apply_liquidity_gate_size_and_rank_compound():
    dd = {
        "spread_pct": 0.0009,
        "signal_ctx_depth_imbalance": 0.50,
        "thesis_size_factor": 0.80,
        "thesis_rank_delta": -0.03,
    }
    out = apply_liquidity_gate_to_decision_data(dd, "ETH/USDT")
    liq_size = float(out["liquidity_quality_size_factor"])
    assert out["thesis_size_factor"] == pytest.approx(max(SIZE_FACTOR_AT_ZERO, 0.80 * liq_size), rel=1e-4)
    assert out["thesis_rank_delta"] == pytest.approx(-0.03 + float(out["liquidity_quality_rank_delta"]), rel=1e-4)
