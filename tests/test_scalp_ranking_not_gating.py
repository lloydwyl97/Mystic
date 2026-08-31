"""SCALP ranking vs execution.

Rejected setups may still rank and fill when mechanical safety is clear.
Mechanical safety still owns hard_block.
Regime / stall / arm-EV remain rank/size only.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from backend.services.binance_scalp.config import get_scalp_config
from backend.services.binance_scalp.economics import ScalpEconomics
from backend.services.binance_scalp.scalp_candidate_ranking import (
    pick_best_global_candidate,
    prepare_entry_signal,
    rank_setup_signal,
)
from backend.services.binance_scalp.scalp_dynamic_sizing import compute_scalp_position_size
from backend.services.binance_scalp.strategies.base import ScalpSetupSignal, StrategyMarketContext


def _ctx(*, spread: float = 0.0002, mid_change_15s: float = 0.0) -> StrategyMarketContext:
    econ = ScalpEconomics.from_env()
    config = get_scalp_config()
    snap = SimpleNamespace(
        symbol="ETHUSDT",
        spread_pct=spread,
        best_ask=100.0,
        best_bid=99.98,
        mid=100.0,
        asks=[[100.0, 1000.0]],
    )
    mom = SimpleNamespace(
        mid_change_15s=mid_change_15s,
        mid_change_30s=mid_change_15s,
        bid_change_15s=mid_change_15s,
        momentum_confirmed=False,
    )
    return StrategyMarketContext(
        symbol="ETHUSDT",
        snap=snap,
        mom=mom,
        bars_1m=[{"low": 99.5, "high": 100.5, "close": 100.0}] * 15,
        econ=econ,
        config=config,
        notional_usd=25.0,
    )


def _reachable_soft_sig(reason: str = "NOT_NEAR_SUPPORT", symbol: str = "ETHUSDT") -> ScalpSetupSignal:
    """A strategy-rejected setup whose expected move comfortably clears net
    edge — i.e. the only reason it isn't executable under the old regime is
    the strategy's own opinion, not economics."""
    return ScalpSetupSignal(
        symbol=symbol,
        side="BUY",
        score=0.0,
        setup_name="range_bounce_scalp",
        confidence=0.4,
        entry_reason="",
        invalidation_reason=None,
        required_target_pct=0.0025,
        expected_move_pct=0.006,
        spread_pct=0.0002,
        impact_pct=0.0,
        depth_sufficient=True,
        limit_buy_price=100.0,
        passed=False,
        reject_reason=reason,
    )


def test_soft_reject_ranks_and_is_eligible():
    """NOT_NEAR_SUPPORT with a real net edge is not a safety hard_block."""
    ranked = rank_setup_signal(_reachable_soft_sig(), regime="range", ctx=_ctx())
    assert ranked.hard_block is None
    assert ranked.entry_eligible is True
    assert ranked.rank_score is not None
    assert ranked.soft_reason == "NOT_NEAR_SUPPORT"
    assert ranked.signal.passed is False
    assert ranked.selection_confidence.startswith("soft_rank_ranked")


def test_case_a_no_pullback_recovery_eligible():
    ranked = rank_setup_signal(_reachable_soft_sig("NO_PULLBACK_RECOVERY"), regime="range", ctx=_ctx())
    assert ranked.hard_block is None
    assert ranked.entry_eligible is True
    assert ranked.signal.passed is False
    best = pick_best_global_candidate(
        [
            {
                "symbol": "ETHUSDT",
                "rank_score": ranked.rank_score,
                "entry_eligible": ranked.entry_eligible,
                "hard_block": ranked.hard_block,
                "signal": ranked.signal,
            }
        ]
    )
    assert best is not None
    assert best["entry_eligible"] is True


def test_regime_mismatch_never_sets_hard_block_on_genuine_pass():
    passed = ScalpSetupSignal(
        symbol="ETHUSDT",
        side="BUY",
        score=2.6,
        setup_name="range_bounce_scalp",
        confidence=0.7,
        entry_reason="support_bounce",
        invalidation_reason="support_break",
        required_target_pct=0.0025,
        expected_move_pct=0.006,
        spread_pct=0.0002,
        impact_pct=0.0,
        depth_sufficient=True,
        limit_buy_price=100.0,
        passed=True,
        reject_reason=None,
    )
    with mock.patch.dict(os.environ, {"SCALP_REQUIRE_REGIME_NATIVE": "true"}):
        ranked = rank_setup_signal(passed, regime="trend_up", ctx=_ctx())
    assert ranked.hard_block is None
    assert ranked.entry_eligible is True
    assert ranked.regime_mismatch is True
    assert "regime_mismatch" in ranked.selection_confidence


def test_symbol_stall_risk_never_blocks_genuine_pass_but_penalizes_score():
    def _pass(symbol: str) -> ScalpSetupSignal:
        return ScalpSetupSignal(
            symbol=symbol,
            side="BUY",
            score=2.6,
            setup_name="range_bounce_scalp",
            confidence=0.7,
            entry_reason="support_bounce",
            invalidation_reason="support_break",
            required_target_pct=0.0025,
            expected_move_pct=0.006,
            spread_pct=0.0002,
            impact_pct=0.0,
            depth_sufficient=True,
            limit_buy_price=100.0,
            passed=True,
            reject_reason=None,
        )

    ctx = _ctx()
    with mock.patch.dict(os.environ, {"SCALP_STALL_RISK_SYMBOL_GATE_ENABLED": "true", "SCALP_STALL_RISK_SYMBOL_BLOCKLIST": "ETHUSDT"}):
        blocked_symbol = rank_setup_signal(_pass("ETHUSDT"), regime="range", ctx=ctx)
        clear_symbol = rank_setup_signal(_pass("BTCUSDT"), regime="range", ctx=_ctx())
    assert blocked_symbol.entry_eligible is True
    assert blocked_symbol.symbol_stall_risk is True
    assert blocked_symbol.rank_components.get("static_rank", 0) < clear_symbol.rank_components.get("static_rank", 1)


def test_arm_negative_ev_never_blocks_genuine_pass():
    passed = ScalpSetupSignal(
        symbol="ETHUSDT",
        side="BUY",
        score=2.6,
        setup_name="range_bounce_scalp",
        confidence=0.7,
        entry_reason="support_bounce",
        invalidation_reason="support_break",
        required_target_pct=0.0025,
        expected_move_pct=0.006,
        spread_pct=0.0002,
        impact_pct=0.0,
        depth_sufficient=True,
        limit_buy_price=100.0,
        passed=True,
        reject_reason=None,
    )
    with mock.patch(
        "backend.services.binance_scalp.scalp_arm_blocker.arm_blocked",
        return_value=(True, "NEGATIVE_EV_ARM", {"sample_count": 40, "win_rate": 0.1}),
    ):
        ranked = rank_setup_signal(passed, regime="range", ctx=_ctx())
    assert ranked.hard_block is None
    assert ranked.entry_eligible is True
    assert ranked.arm_penalty_mult < 1.0


def test_prepare_entry_signal_stamps_soft_rank_without_forging_pass():
    ranked = rank_setup_signal(_reachable_soft_sig(), regime="range", ctx=_ctx())
    assert ranked.entry_eligible is True
    sig = prepare_entry_signal(ranked, _ctx())
    assert sig.passed is False
    assert (sig.setup_context or {}).get("entry_owner") == "ranking_ev"
    assert (sig.setup_context or {}).get("soft_rank_entry") is True


def test_prepare_entry_signal_stamps_strategy_owner_for_genuine_pass():
    passed_sig = ScalpSetupSignal(
        symbol="BTCUSDT",
        side="BUY",
        score=2.6,
        setup_name="range_bounce_scalp",
        confidence=0.7,
        entry_reason="support_bounce",
        invalidation_reason="support_break",
        required_target_pct=0.0025,
        expected_move_pct=0.0035,
        spread_pct=0.0002,
        impact_pct=0.0,
        depth_sufficient=True,
        limit_buy_price=100.0,
        passed=True,
        reject_reason=None,
    )
    ranked = rank_setup_signal(passed_sig, regime="range", ctx=_ctx())
    sig = prepare_entry_signal(ranked, _ctx())
    assert sig.setup_context["entry_owner"] == "strategy"
    assert sig.setup_context["soft_rank_entry"] is False


def test_hard_safety_reasons_still_block_entry_eligible():
    """Mechanical safety must still block — this is the one form of gate the
    architecture explicitly keeps."""
    sig = ScalpSetupSignal(
        symbol="ETHUSDT",
        side="BUY",
        score=0.0,
        setup_name="range_bounce_scalp",
        confidence=0.0,
        entry_reason="",
        invalidation_reason=None,
        required_target_pct=0.0025,
        expected_move_pct=0.0,
        spread_pct=0.02,
        impact_pct=0.0,
        depth_sufficient=True,
        limit_buy_price=100.0,
        passed=False,
        reject_reason="NOT_NEAR_SUPPORT",
    )
    ranked = rank_setup_signal(sig, regime="range", ctx=_ctx(spread=0.02))
    assert ranked.entry_eligible is False
    assert ranked.hard_block == "SPREAD_TOO_WIDE"


def test_dynamic_sizing_genuine_pass_sizes_larger_than_soft_rank():
    passed = compute_scalp_position_size(
        base_cap=100.0,
        free_cash=1000.0,
        strategy_passed=True,
    )
    soft = compute_scalp_position_size(
        base_cap=100.0,
        free_cash=1000.0,
        strategy_passed=False,
    )
    assert passed.notional > soft.notional
    assert soft.notional > 0.0  # never fully refused


def test_dynamic_sizing_never_exceeds_base_cap_or_free_cash():
    result = compute_scalp_position_size(
        base_cap=100.0,
        free_cash=40.0,
        strategy_passed=True,
    )
    assert result.notional <= 40.0
    assert result.notional <= 100.0


def test_dynamic_sizing_stall_risk_and_regime_mismatch_compound_discount():
    clean = compute_scalp_position_size(base_cap=100.0, free_cash=1000.0, strategy_passed=True)
    conflicted = compute_scalp_position_size(
        base_cap=100.0,
        free_cash=1000.0,
        strategy_passed=True,
        regime_mismatch=True,
        symbol_stall_risk=True,
        arm_penalty_mult=0.20,
        mtf_penalty_mult=0.40,
    )
    assert conflicted.notional < clean.notional
    assert conflicted.notional > 0.0


def test_dynamic_sizing_respects_min_floor():
    result = compute_scalp_position_size(
        base_cap=100.0,
        free_cash=1000.0,
        min_notional=5.0,
        strategy_passed=False,
        regime_mismatch=True,
        symbol_stall_risk=True,
        arm_penalty_mult=0.2,
        mtf_penalty_mult=0.2,
    )
    assert result.notional >= 5.0


def test_case_d_spread_blocks_soft_rank():
    sig = _reachable_soft_sig("NOT_NEAR_SUPPORT")
    ranked = rank_setup_signal(sig, regime="range", ctx=_ctx(spread=0.02))
    assert ranked.entry_eligible is False
    assert ranked.hard_block == "SPREAD_TOO_WIDE"


def test_case_e_stale_data_is_hard_block():
    stale = _reachable_soft_sig("STALE_DATA")
    ranked = rank_setup_signal(stale, regime="range", ctx=_ctx())
    assert ranked.entry_eligible is False
    assert ranked.hard_block == "STALE_DATA"


def test_case_f_no_executable_net_edge_blocks():
    unreachable = _reachable_soft_sig("TARGET_NOT_REACHABLE")
    ranked = rank_setup_signal(unreachable, regime="range", ctx=_ctx())
    assert ranked.entry_eligible is False
    assert ranked.hard_block in {"TARGET_NOT_REACHABLE", "NO_EXECUTABLE_NET_EDGE"}


def test_case_g_breaker_blocks_regardless_of_passed():
    from backend.services.binance_scalp.paper_engine import BinanceScalpPaperEngine

    engine = object.__new__(BinanceScalpPaperEngine)
    engine._last_circuit_breaker_open = False

    def _open(_conn=None):
        return True

    engine._check_scalp_circuit_breaker = _open
    assert engine._check_scalp_circuit_breaker() is True


def test_paper_engine_no_sig_passed_buy_locks():
    import inspect

    from backend.services.binance_scalp import paper_engine as pe

    cand_src = inspect.getsource(pe.BinanceScalpPaperEngine._entry_candidates)
    try_src = inspect.getsource(pe.BinanceScalpPaperEngine._try_entry)
    full = cand_src + try_src
    assert "STRATEGY_NOT_PASSED" not in full
    assert "if not strategy_passed:" not in inspect.getsource(pe.BinanceScalpPaperEngine._try_entry)
    assert 'or not bool(getattr(sig, "passed", False))' not in cand_src


def test_soft_rank_reservation_not_popped_for_opinion():
    import inspect

    from backend.services.binance_scalp.paper_engine import BinanceScalpPaperEngine

    src = inspect.getsource(BinanceScalpPaperEngine._try_entry)
    assert "if not strategy_passed:" not in src
    assert "self._entry_reservations.pop" not in src or "INSUFFICIENT_CASH" in src
