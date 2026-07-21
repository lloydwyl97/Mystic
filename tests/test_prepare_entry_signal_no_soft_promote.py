"""Soft-rank promotion must never resurrect rejected SCALP setups."""

from __future__ import annotations

from types import SimpleNamespace

from backend.services.binance_scalp.scalp_candidate_ranking import RankedCandidate, prepare_entry_signal
from backend.services.binance_scalp.strategies.base import ScalpSetupSignal


def _sig(*, passed: bool) -> ScalpSetupSignal:
    return ScalpSetupSignal(
        symbol="BTCUSDT",
        side="BUY",
        score=1.2,
        setup_name="range_bounce_scalp",
        confidence=0.5,
        entry_reason="test",
        invalidation_reason=None,
        required_target_pct=0.0025,
        expected_move_pct=0.003,
        spread_pct=0.0002,
        impact_pct=0.0001,
        depth_sufficient=True,
        limit_buy_price=100.0,
        passed=passed,
        reject_reason=None if passed else "NOT_NEAR_SUPPORT",
        setup_context={},
    )


def test_prepare_entry_signal_passes_through_genuine_pass():
    sig = _sig(passed=True)
    ranked = RankedCandidate(
        signal=sig,
        rank_score=2.0,
        entry_eligible=True,
        hard_block=None,
        regime="range",
        regime_native=True,
        soft_reason=None,
        selection_confidence="ok",
        reachability_surplus=0.001,
    )
    out = prepare_entry_signal(ranked, ctx=SimpleNamespace())
    assert out is sig
    assert out.passed is True


def test_prepare_entry_signal_refuses_soft_promotion():
    sig = _sig(passed=False)
    ranked = RankedCandidate(
        signal=sig,
        rank_score=1.8,
        entry_eligible=False,
        hard_block=None,
        regime="range",
        regime_native=True,
        soft_reason="NOT_NEAR_SUPPORT",
        selection_confidence="soft",
        reachability_surplus=0.0,
    )
    out = prepare_entry_signal(ranked, ctx=SimpleNamespace())
    assert out.passed is False
    assert out.reject_reason == "NOT_NEAR_SUPPORT"
    assert (out.setup_context or {}).get("soft_rank_entry") is not True
