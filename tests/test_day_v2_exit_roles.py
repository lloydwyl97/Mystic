"""Tests for DAY V2 exit roles (shadow only)."""

import pytest

from backend.services.day_v2.engine_identity import EngineId
from backend.services.day_v2.exit_roles import (
    ExitRole,
    evaluate_all_roles,
    evaluate_catastrophic_protection,
    evaluate_objective_complete,
    evaluate_structural_invalidation,
    evaluate_time_expiration,
    evaluate_winner_protection,
)

# ---------------------------------------------------------------------------
# Catastrophic protection
# ---------------------------------------------------------------------------


def test_catastrophic_fires_on_large_adverse_move():
    # atr_pct=0.01 (1%), multiplier=3.0 -> threshold=3%
    result = evaluate_catastrophic_protection(
        entry_price=100.0,
        current_price=96.5,  # 3.5% adverse — above 3% threshold
        atr_pct=0.01,
        hold_minutes=30.0,
    )
    assert result.should_exit is True
    assert result.role == ExitRole.DAY_V2_CATASTROPHIC_PROTECTION
    assert result.is_shadow_only is True


def test_catastrophic_does_not_fire_on_favorable_move():
    result = evaluate_catastrophic_protection(
        entry_price=100.0,
        current_price=103.0,  # favorable
        atr_pct=0.01,
        hold_minutes=10.0,
    )
    assert result.should_exit is False


def test_catastrophic_does_not_check_hold_time():
    # Should fire regardless of hold time
    result_short = evaluate_catastrophic_protection(
        entry_price=100.0,
        current_price=96.5,
        atr_pct=0.01,
        hold_minutes=1.0,  # very short hold
    )
    result_long = evaluate_catastrophic_protection(
        entry_price=100.0,
        current_price=96.5,
        atr_pct=0.01,
        hold_minutes=400.0,  # very long hold
    )
    assert result_short.should_exit is True
    assert result_long.should_exit is True


def test_catastrophic_does_not_fire_below_threshold():
    # 1.5% adverse, threshold=3% — should NOT fire
    result = evaluate_catastrophic_protection(
        entry_price=100.0,
        current_price=98.5,
        atr_pct=0.01,
        hold_minutes=30.0,
    )
    assert result.should_exit is False


# ---------------------------------------------------------------------------
# Structural invalidation
# ---------------------------------------------------------------------------


def test_structural_invalidation_requires_closed_bars():
    # Only 2 closed bars, need 3
    result = evaluate_structural_invalidation(
        entry_price=100.0,
        current_price=98.0,
        structural_anchor_price=99.0,
        regime="bull",
        n_closed_15m_bars=2,
    )
    assert result.should_exit is False
    assert "insufficient" in result.reason


def test_structural_invalidation_fires_below_anchor():
    result = evaluate_structural_invalidation(
        entry_price=100.0,
        current_price=98.0,  # below anchor at 99
        structural_anchor_price=99.0,
        regime="bull",
        n_closed_15m_bars=4,
    )
    assert result.should_exit is True
    assert "anchor" in result.reason or "anchor" in result.detail.lower()


def test_structural_invalidation_does_not_fire_above_anchor():
    result = evaluate_structural_invalidation(
        entry_price=100.0,
        current_price=101.0,  # above anchor
        structural_anchor_price=99.0,
        regime="bull",
        n_closed_15m_bars=5,
    )
    assert result.should_exit is False


# ---------------------------------------------------------------------------
# Time expiration
# ---------------------------------------------------------------------------


def test_time_expiration_fires_at_max_hold():
    result = evaluate_time_expiration(
        hold_minutes=310.0,  # exceeds 300m default
        max_hold_minutes=300,
        net_pnl_pct=-0.002,  # losing
    )
    assert result.should_exit is True
    assert result.role == ExitRole.DAY_V2_TIME_EXPIRATION


def test_time_expiration_does_not_fire_on_profitable_position():
    result = evaluate_time_expiration(
        hold_minutes=310.0,
        max_hold_minutes=300,
        net_pnl_pct=0.005,  # profitable — winner protection should handle it
    )
    assert result.should_exit is False
    assert "profitable" in result.reason


def test_time_expiration_does_not_fire_within_window():
    result = evaluate_time_expiration(
        hold_minutes=180.0,
        max_hold_minutes=300,
        net_pnl_pct=-0.001,
    )
    assert result.should_exit is False


# ---------------------------------------------------------------------------
# Winner protection
# ---------------------------------------------------------------------------


def test_winner_protection_requires_mfe_gate():
    # MFE = 0.3% — below the 0.8% gate
    result = evaluate_winner_protection(
        entry_price=100.0,
        highest_price=100.3,  # 0.3% MFE
        current_price=100.2,
        atr_pct=0.005,
        hold_minutes=60.0,
        net_pnl_pct=0.002,
    )
    assert result.should_exit is False
    assert "mfe_gate" in result.reason


def test_winner_protection_fires_on_large_atr_pullback():
    # MFE=2%, trail=max(0.5%, 1.5*1%)=max(0.5%, 1.5%)=1.5%
    # highest=102, trail_trigger=102*(1-0.015)=100.47
    # current=100.0 < 100.47 -> fires
    result = evaluate_winner_protection(
        entry_price=100.0,
        highest_price=102.0,  # 2% MFE
        current_price=100.0,  # pulled back 1.96% from high
        atr_pct=0.01,
        hold_minutes=120.0,
        net_pnl_pct=0.0,
    )
    assert result.should_exit is True
    assert result.role == ExitRole.DAY_V2_WINNER_PROTECTION


def test_winner_protection_does_not_fire_on_small_move():
    """Specifically: 0.20% MFE with 0.25% trail does NOT fire.

    Legacy trail is ~0.20-0.25%. DAY V2 uses max(0.5%, 1.5*ATR).
    With atr=0.002 (0.2%), trail=max(0.5%, 0.3%)=0.5%.
    0.20% MFE is below the 0.8% gate, so it should never fire.
    """
    result = evaluate_winner_protection(
        entry_price=100.0,
        highest_price=100.20,  # 0.20% MFE — below gate
        current_price=99.95,  # pulled back 0.25% from high
        atr_pct=0.002,  # 0.2% ATR
        hold_minutes=15.0,
        net_pnl_pct=-0.0005,
    )
    assert result.should_exit is False
    assert "mfe_gate" in result.reason


def test_winner_protection_trail_is_atr_based_not_fixed():
    # With large ATR of 2%, trail = max(0.5%, 3%) = 3%
    # MFE = 5%, highest=105, trail_trigger=105*(1-0.03)=101.85
    # current=103 > 101.85 -> should NOT fire
    result = evaluate_winner_protection(
        entry_price=100.0,
        highest_price=105.0,  # 5% MFE
        current_price=103.0,
        atr_pct=0.02,  # 2% ATR -> trail = 3%
        hold_minutes=180.0,
        net_pnl_pct=0.03,
    )
    assert result.should_exit is False


# ---------------------------------------------------------------------------
# Objective complete
# ---------------------------------------------------------------------------


def test_objective_complete_fires_at_target_on_closed_bar():
    result = evaluate_objective_complete(
        entry_price=100.0,
        current_price=112.0,  # at or above target
        thesis_target_price=110.0,
        n_closed_15m_bars=3,
        net_pnl_pct=0.12,
    )
    assert result.should_exit is True
    assert result.role == ExitRole.DAY_V2_OBJECTIVE_COMPLETE


def test_objective_complete_does_not_require_04pct_net_floor():
    # Even with very small net_pnl_pct, if target is hit, it fires
    result = evaluate_objective_complete(
        entry_price=100.0,
        current_price=110.0,
        thesis_target_price=108.0,
        n_closed_15m_bars=2,
        net_pnl_pct=0.001,  # tiny net — below legacy 0.4% floor
    )
    assert result.should_exit is True


def test_objective_complete_does_not_fire_below_target():
    result = evaluate_objective_complete(
        entry_price=100.0,
        current_price=107.0,  # below target
        thesis_target_price=110.0,
        n_closed_15m_bars=4,
        net_pnl_pct=0.07,
    )
    assert result.should_exit is False


def test_objective_complete_requires_at_least_one_closed_bar():
    result = evaluate_objective_complete(
        entry_price=100.0,
        current_price=115.0,
        thesis_target_price=110.0,
        n_closed_15m_bars=0,  # no closed bar yet
        net_pnl_pct=0.15,
    )
    assert result.should_exit is False
    assert "no_closed_bar" in result.reason


def test_objective_complete_with_pnl_above_04pct_not_exited_if_below_target():
    """Position with 0.4%+ net_pnl that hasn't hit structural target should NOT exit."""
    result = evaluate_objective_complete(
        entry_price=100.0,
        current_price=100.5,  # 0.5% up, but target is 110
        thesis_target_price=110.0,
        n_closed_15m_bars=2,
        net_pnl_pct=0.005,  # above 0.4% floor
    )
    assert result.should_exit is False


# ---------------------------------------------------------------------------
# evaluate_all_roles
# ---------------------------------------------------------------------------


def test_evaluate_all_roles_calls_assert_no_live_authority():
    """evaluate_all_roles must reject LEGACY_DAY_LIVE engine."""
    with pytest.raises(PermissionError):
        evaluate_all_roles(
            entry_price=100.0,
            highest_price=102.0,
            current_price=101.0,
            structural_anchor_price=98.0,
            thesis_target_price=110.0,
            atr_pct=0.01,
            hold_minutes=60.0,
            max_hold_minutes=300,
            net_pnl_pct=0.01,
            regime="bull",
            n_closed_15m_bars=4,
            engine_id=EngineId.LEGACY_DAY_LIVE,
        )


def test_evaluate_all_roles_raises_for_live_engine():
    with pytest.raises(PermissionError):
        evaluate_all_roles(
            entry_price=100.0,
            highest_price=100.0,
            current_price=100.0,
            structural_anchor_price=99.0,
            thesis_target_price=105.0,
            atr_pct=0.01,
            hold_minutes=30.0,
            max_hold_minutes=300,
            net_pnl_pct=0.0,
            regime="neutral",
            n_closed_15m_bars=1,
            engine_id=EngineId.LEGACY_DAY_LIVE,
        )


def test_all_evaluations_are_shadow_only():
    results = evaluate_all_roles(
        entry_price=100.0,
        highest_price=101.0,
        current_price=100.5,
        structural_anchor_price=98.0,
        thesis_target_price=110.0,
        atr_pct=0.01,
        hold_minutes=60.0,
        max_hold_minutes=300,
        net_pnl_pct=0.005,
        regime="bull",
        n_closed_15m_bars=3,
        engine_id=EngineId.DAY_V2_SHADOW,
    )
    assert len(results) == 5
    for r in results:
        assert r.is_shadow_only is True, f"{r.role} returned is_shadow_only=False"


def test_evaluate_all_roles_returns_all_five_roles():
    results = evaluate_all_roles(
        entry_price=100.0,
        highest_price=100.5,
        current_price=100.4,
        structural_anchor_price=98.0,
        thesis_target_price=110.0,
        atr_pct=0.01,
        hold_minutes=30.0,
        max_hold_minutes=300,
        net_pnl_pct=0.004,
        regime="bull",
        n_closed_15m_bars=5,
        engine_id=EngineId.SCALP_V2_CANDIDATE,
    )
    roles_returned = {r.role for r in results}
    assert ExitRole.DAY_V2_CATASTROPHIC_PROTECTION in roles_returned
    assert ExitRole.DAY_V2_STRUCTURAL_INVALIDATION in roles_returned
    assert ExitRole.DAY_V2_TIME_EXPIRATION in roles_returned
    assert ExitRole.DAY_V2_WINNER_PROTECTION in roles_returned
    assert ExitRole.DAY_V2_OBJECTIVE_COMPLETE in roles_returned


def test_exit_evaluation_is_shadow_only_enforced():
    """ExitEvaluation raises if is_shadow_only is set to False."""
    from backend.services.day_v2.exit_roles import ExitEvaluation

    with pytest.raises(ValueError):
        ExitEvaluation(
            role=ExitRole.DAY_V2_CATASTROPHIC_PROTECTION,
            should_exit=False,
            confidence=0.0,
            reason="test",
            is_shadow_only=False,  # must raise
        )
