"""
Regression: STALL_EXIT must gate on true estimated NET P&L after fees
(gross mark movement minus ESTIMATED_ROUNDTRIP_COST), never on gross mark
movement alone, and must never close a genuinely net-positive or net-flat
position merely because time elapsed.

P1A: flat/tiny-red low-MFE holds are no longer force-sold; only dead/worsening
trades with confirmed adverse movement may STALL_EXIT_DEAD_NO_MFE.
"""

from __future__ import annotations

from backend.config.trading_economics import ESTIMATED_ROUNDTRIP_COST, MIN_NET_PROFIT_TO_SELL
from backend.services.day_controlled_exits import (
    EXIT_STALL_DEAD,
    STALL_HOLD_FLAT_NOT_DEAD,
    STALL_HOLD_MFE_TOO_HIGH,
    STALL_HOLD_NOT_RED,
    evaluate_stall_exit,
)

ENTRY = 100.0
STALL_ELIGIBLE_HOLD_MIN = 125.0  # > default 120m min hold, < default 360m max hold
LOW_MFE_HIGH = ENTRY * 1.0005  # 0.05% MFE, below the 0.50% stall MFE ceiling


def _net_pnl_pct(current_price: float) -> float:
    """Exact production formula: gross mark movement minus estimated roundtrip cost."""
    gross = (current_price - ENTRY) / ENTRY
    return gross - ESTIMATED_ROUNDTRIP_COST


def test_tiny_gross_green_but_net_red_after_fees_not_force_stalled_when_flat():
    """Gross +0.03% is net red after fees, but MAE is tiny — P1A holds as flat-not-dead."""
    current_price = ENTRY * 1.0003  # +0.03% gross
    net = _net_pnl_pct(current_price)
    assert net < 0.0, "fixture must be net-negative after fees to test this branch"
    out = evaluate_stall_exit(
        entry_price=ENTRY,
        highest_price=LOW_MFE_HIGH,
        lowest_price=current_price,
        current_price=current_price,
        net_pnl_pct=net,
        hold_minutes=STALL_ELIGIBLE_HOLD_MIN,
        max_hold_min=360,
    )
    assert out is not None
    assert out["action"] == "hold"
    assert out["reason"] == STALL_HOLD_FLAT_NOT_DEAD


def test_tiny_net_green_never_stalled():
    """Even a razor-thin net-positive P&L (net_pnl_pct > 0) must never be stalled."""
    current_price = ENTRY * (1.0 + ESTIMATED_ROUNDTRIP_COST + 0.00005)
    net = _net_pnl_pct(current_price)
    assert 0.0 < net < 0.0001, f"fixture must be a razor-thin net green, got {net}"
    out = evaluate_stall_exit(
        entry_price=ENTRY,
        highest_price=current_price,
        current_price=current_price,
        net_pnl_pct=net,
        hold_minutes=STALL_ELIGIBLE_HOLD_MIN,
        max_hold_min=360,
    )
    assert out is not None
    assert out["action"] == "hold"
    assert out["reason"] == STALL_HOLD_NOT_RED


def test_flat_at_exactly_zero_net_after_fees_never_stalled():
    """net_pnl_pct == 0.0 exactly must hit the >= 0 guard and hold, not stall."""
    current_price = ENTRY * (1.0 + ESTIMATED_ROUNDTRIP_COST)
    net = _net_pnl_pct(current_price)
    assert abs(net) < 1e-9
    out = evaluate_stall_exit(
        entry_price=ENTRY,
        highest_price=current_price,
        current_price=current_price,
        net_pnl_pct=net,
        hold_minutes=STALL_ELIGIBLE_HOLD_MIN,
        max_hold_min=360,
    )
    assert out is not None
    assert out["action"] == "hold"
    assert out["reason"] == STALL_HOLD_NOT_RED


def test_tiny_net_negative_flat_loss_not_force_stalled():
    out = evaluate_stall_exit(
        entry_price=ENTRY,
        highest_price=ENTRY,
        lowest_price=ENTRY * 0.9999,
        current_price=ENTRY * 0.9999,
        net_pnl_pct=-1e-13,
        hold_minutes=STALL_ELIGIBLE_HOLD_MIN,
        max_hold_min=360,
    )
    assert out is not None
    assert out["action"] == "hold"
    assert out["reason"] == STALL_HOLD_FLAT_NOT_DEAD


def test_clearly_profitable_position_never_stalled():
    current_price = ENTRY * 1.01  # +1% gross, well net-positive after 0.072% cost
    net = _net_pnl_pct(current_price)
    assert net > MIN_NET_PROFIT_TO_SELL
    out = evaluate_stall_exit(
        entry_price=ENTRY,
        highest_price=current_price,
        current_price=current_price,
        net_pnl_pct=net,
        hold_minutes=STALL_ELIGIBLE_HOLD_MIN,
        max_hold_min=360,
    )
    assert out is not None
    assert out["action"] == "hold"


def test_deteriorating_red_position_with_no_progress_can_stall():
    """Never had meaningful favorable excursion and is meaningfully adverse -> cut."""
    current_price = ENTRY * 0.996  # -0.4% gross -> clearly net red after fees
    net = _net_pnl_pct(current_price)
    assert net < 0.0
    out = evaluate_stall_exit(
        entry_price=ENTRY,
        highest_price=LOW_MFE_HIGH,
        lowest_price=current_price,
        current_price=current_price,
        net_pnl_pct=net,
        hold_minutes=STALL_ELIGIBLE_HOLD_MIN,
        max_hold_min=360,
    )
    assert out is not None
    assert out["action"] == "sell"
    assert out["reason"] == EXIT_STALL_DEAD


def test_improving_red_position_with_real_mfe_progress_is_not_stalled():
    """Currently net red, but the position DID show real favorable excursion."""
    current_price = ENTRY * 0.999  # currently slightly net red after fees
    net = _net_pnl_pct(current_price)
    assert net < 0.0
    strong_mfe_high = ENTRY * 1.006  # 0.6% MFE, above 0.50% stall ceiling
    out = evaluate_stall_exit(
        entry_price=ENTRY,
        highest_price=strong_mfe_high,
        lowest_price=current_price,
        current_price=current_price,
        net_pnl_pct=net,
        hold_minutes=STALL_ELIGIBLE_HOLD_MIN,
        max_hold_min=360,
    )
    assert out is not None
    assert out["action"] == "hold"
    assert out["reason"] == STALL_HOLD_MFE_TOO_HIGH


def test_no_entry_gate_side_effect_from_stall_exit_module():
    """STALL_EXIT is exit-only: importing/calling it must not touch any
    entry/candidate-ranking gate. Sanity check that the function has no
    required-artifact or entry-context side effects."""
    import inspect

    sig = inspect.signature(evaluate_stall_exit)
    assert {
        "entry_price",
        "highest_price",
        "net_pnl_pct",
        "hold_minutes",
        "max_hold_min",
        "current_price",
        "lowest_price",
    } == set(sig.parameters)
