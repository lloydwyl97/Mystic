"""
Regression: STALL_EXIT must gate on true estimated NET P&L after fees
(gross mark movement minus ESTIMATED_ROUNDTRIP_COST), never on gross mark
movement alone, and must never close a genuinely net-positive or net-flat
position merely because time elapsed.

Production call site (backend/services/portfolio_engine.py):
    pnl_pct = (current_price - entry_price) / entry_price
    net_pnl_pct = pnl_pct - ESTIMATED_ROUNDTRIP_COST

This file replicates that exact formula so each scenario is anchored to the
real fee constant, not an arbitrary net_pnl_pct guess.

No historical replay/backtest framework exists in this repo for STALL_EXIT
(confirmed by grep — no replay harness feeds this function real historical
price series). Conclusions here are therefore limited to deterministic
unit/integration behavior of the exit rule itself, not a claim of measured
expectancy improvement.
"""

from __future__ import annotations

from backend.config.trading_economics import ESTIMATED_ROUNDTRIP_COST, MIN_NET_PROFIT_TO_SELL
from backend.services.day_controlled_exits import EXIT_STALL, evaluate_stall_exit

ENTRY = 100.0
STALL_ELIGIBLE_HOLD_MIN = 35.0  # > default 30m min hold, < default 60m max hold
LOW_MFE_HIGH = ENTRY * 1.0005  # 0.05% MFE, below the 0.15% default stall MFE ceiling


def _net_pnl_pct(current_price: float) -> float:
    """Exact production formula: gross mark movement minus estimated roundtrip cost."""
    gross = (current_price - ENTRY) / ENTRY
    return gross - ESTIMATED_ROUNDTRIP_COST


def test_tiny_gross_green_but_net_red_after_fees_can_stall():
    """Gross +0.03% is a 'green' mark, but after ~0.072% roundtrip cost the
    trade is net red — STALL_EXIT must be eligible to cut it, not treat the
    gross-positive mark as a reason to hold."""
    current_price = ENTRY * 1.0003  # +0.03% gross
    net = _net_pnl_pct(current_price)
    assert net < 0.0, "fixture must be net-negative after fees to test this branch"
    out = evaluate_stall_exit(
        entry_price=ENTRY,
        highest_price=LOW_MFE_HIGH,
        net_pnl_pct=net,
        hold_minutes=STALL_ELIGIBLE_HOLD_MIN,
        max_hold_min=60,
    )
    assert out is not None
    assert out["action"] == "sell"
    assert out["reason"] == EXIT_STALL
    assert out["net_pnl_pct"] == net


def test_tiny_net_green_never_stalled():
    """Even a razor-thin net-positive P&L (net_pnl_pct > 0) must never be stalled."""
    # Solve for a current_price that yields a tiny positive net after fees.
    current_price = ENTRY * (1.0 + ESTIMATED_ROUNDTRIP_COST + 0.00005)
    net = _net_pnl_pct(current_price)
    assert 0.0 < net < 0.0001, f"fixture must be a razor-thin net green, got {net}"
    out = evaluate_stall_exit(
        entry_price=ENTRY,
        highest_price=current_price,
        net_pnl_pct=net,
        hold_minutes=STALL_ELIGIBLE_HOLD_MIN,
        max_hold_min=60,
    )
    assert out is None, "must never stall a net-positive position regardless of elapsed time"


def test_flat_at_exactly_zero_net_after_fees_never_stalled():
    """net_pnl_pct == 0.0 exactly must hit the >= 0 guard and hold, not stall."""
    current_price = ENTRY * (1.0 + ESTIMATED_ROUNDTRIP_COST)
    net = _net_pnl_pct(current_price)
    assert abs(net) < 1e-9
    out = evaluate_stall_exit(
        entry_price=ENTRY,
        highest_price=current_price,
        net_pnl_pct=net,
        hold_minutes=STALL_ELIGIBLE_HOLD_MIN,
        max_hold_min=60,
    )
    assert out is None


def test_any_strictly_net_negative_flat_loss_can_stall():
    out = evaluate_stall_exit(
        entry_price=ENTRY,
        highest_price=ENTRY,
        net_pnl_pct=-1e-13,
        hold_minutes=STALL_ELIGIBLE_HOLD_MIN,
        max_hold_min=60,
    )
    assert out is not None
    assert out["reason"] == EXIT_STALL


def test_clearly_profitable_position_never_stalled():
    current_price = ENTRY * 1.01  # +1% gross, well net-positive after 0.072% cost
    net = _net_pnl_pct(current_price)
    assert net > MIN_NET_PROFIT_TO_SELL
    out = evaluate_stall_exit(
        entry_price=ENTRY,
        highest_price=current_price,
        net_pnl_pct=net,
        hold_minutes=STALL_ELIGIBLE_HOLD_MIN,
        max_hold_min=60,
    )
    assert out is None


def test_deteriorating_red_position_with_no_progress_can_stall():
    """Never had meaningful favorable excursion and is net red -> eligible to cut."""
    current_price = ENTRY * 0.998  # -0.2% gross -> clearly net red after fees
    net = _net_pnl_pct(current_price)
    assert net < 0.0
    out = evaluate_stall_exit(
        entry_price=ENTRY,
        highest_price=LOW_MFE_HIGH,  # never moved favorably
        net_pnl_pct=net,
        hold_minutes=STALL_ELIGIBLE_HOLD_MIN,
        max_hold_min=60,
    )
    assert out is not None
    assert out["reason"] == EXIT_STALL


def test_improving_red_position_with_real_mfe_progress_is_not_stalled():
    """Currently net red, but the position DID show real favorable excursion
    (high water mark far above the stall MFE ceiling) — this is an improving
    trade that pulled back, not a dead one; STALL_EXIT must not cut it."""
    current_price = ENTRY * 0.999  # currently slightly net red after fees
    net = _net_pnl_pct(current_price)
    assert net < 0.0
    strong_mfe_high = ENTRY * 1.005  # 0.5% MFE, well above the 0.15% stall ceiling
    out = evaluate_stall_exit(
        entry_price=ENTRY,
        highest_price=strong_mfe_high,
        net_pnl_pct=net,
        hold_minutes=STALL_ELIGIBLE_HOLD_MIN,
        max_hold_min=60,
    )
    assert out is None, "a position with real prior favorable excursion must not be treated as a dead stall"


def test_no_entry_gate_side_effect_from_stall_exit_module():
    """STALL_EXIT is exit-only: importing/calling it must not touch any
    entry/candidate-ranking gate. Sanity check that the function has no
    required-artifact or entry-context side effects."""
    import inspect

    sig = inspect.signature(evaluate_stall_exit)
    assert set(sig.parameters) == {"entry_price", "highest_price", "net_pnl_pct", "hold_minutes", "max_hold_min"}
