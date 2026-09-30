"""DAY V2 live exit evaluation.

Evaluates real open positions with engine_id='DAY_V2'. Returns an action
dict or None. Called from portfolio_engine._check_exit_conditions.

This is NOT shadow-only. It routes to real order execution. Do not add
assert_no_live_authority() calls here.

Exit priority (highest to lowest):
  1. CATASTROPHIC_PROTECTION  — adverse move from entry >= 3x 15m ATR
  2. STRUCTURAL_INVALIDATION  — price below the setup's structural anchor (after 3+ bars)
  3. WINNER_PROTECTION        — structure-runner ratchet (day_v2.winner_contract):
                                armed only after 1x 1h-ATR favourable excursion,
                                never below break-even after round-trip costs,
                                only moves up. Reported as OBJECTIVE_COMPLETE when
                                the setup objective was reached before the stop.
  4. TIME_EXPIRATION          — hold >= 300 min, never armed, still net-negative

There is no fixed profit target: reaching the objective tightens the ratchet,
it does not sell.
"""

from __future__ import annotations

import logging
import time

from backend.services.day_v2.config import (
    DAY_V2_CATASTROPHIC_ATR_MULTIPLIER,
    DAY_V2_MAX_HOLD_MINUTES,
    DAY_V2_STRUCTURAL_INVALIDATION_BARS_CLOSED,
)
from backend.services.day_v2.winner_contract import (
    LEGACY_ATR_1H_PER_ATR_15M,
    objective_level,
    runner_stop,
)

logger = logging.getLogger(__name__)

DAY_V2_ENGINE_ID: str = "DAY_V2"

_DAY_V2_RECORDED_EXIT_REASONS: dict[str, str] = {
    "DAY_V2_CATASTROPHIC_PROTECTION": "STOP_LOSS_EXIT",
    "DAY_V2_STRUCTURAL_INVALIDATION": "THESIS_INVALIDATION_EXIT",
    "DAY_V2_WINNER_PROTECTION": "TRAILING_STOP_EXIT",
    "DAY_V2_OBJECTIVE_COMPLETE": "NET_PROFIT_EXIT",
    "DAY_V2_TIME_EXPIRATION": "TIME_STOP_EXIT",
}


def day_v2_recorded_exit_reason(exit_trigger: str) -> str:
    """Reporting label for a known DAY V2 exit trigger, else '' (caller keeps its existing label)."""
    return _DAY_V2_RECORDED_EXIT_REASONS.get(str(exit_trigger or "").strip().upper(), "")


def evaluate_day_v2_exit(
    *,
    engine_id: str,
    entry_price: float,
    current_price: float,
    bar_low: float,
    highest_price: float,
    atr_at_entry: float,
    structural_anchor: float,
    target_price: float,
    entry_time: float,
    estimated_roundtrip_cost: float,
    setup: str = "",
    atr_1h_at_entry: float = 0.0,
    objective_structural: float = 0.0,
) -> dict | None:
    """Evaluate all DAY V2 exit roles.

    Args:
        engine_id: Must be "DAY_V2" or this function returns None.
        entry_price: Average fill price.
        current_price: Current executable mark.
        bar_low: Lowest price since entry (catastrophic check).
        highest_price: High-water mark since entry (runner ratchet).
        atr_at_entry: 15m ATR at entry (absolute price units).
        structural_anchor: Price below which the thesis is invalid.
        target_price: Entry-time setup target; objective fallback for
            positions opened before the structural objective was stamped.
        entry_time: Unix epoch of position open.
        estimated_roundtrip_cost: Total cost fraction (e.g. 0.0006).
        setup: Entry setup name (selects the objective ATR floor).
        atr_1h_at_entry: 1h ATR at entry; 0 means a pre-runner position,
            which uses LEGACY_ATR_1H_PER_ATR_15M x atr_at_entry.
        objective_structural: Setup structural objective stamped at entry.

    Returns:
        {"action": "sell", "reason": str, "exit_price_estimate": float,
         "detail": str}
        or None if no exit condition is met.
    """
    if engine_id != DAY_V2_ENGINE_ID:
        return None
    if entry_price <= 0 or current_price <= 0:
        return None

    now = time.time()
    hold_minutes = max(0.0, (now - entry_time) / 60.0) if entry_time > 0 else 0.0
    # Approximate closed-15m-bar count from wall-clock hold time.
    # Used only for the structural-invalidation guard (requires N closed bars).
    bars_held_approx = int(hold_minutes / 15.0)

    # Role 1: Catastrophic protection — uses the lowest price since entry
    if atr_at_entry > 0:
        adverse_move = (entry_price - max(bar_low, 0.0)) / entry_price
        catastro_pct = DAY_V2_CATASTROPHIC_ATR_MULTIPLIER * atr_at_entry / entry_price
        if adverse_move >= catastro_pct:
            exit_px = entry_price * (1.0 - catastro_pct)
            logger.warning(
                "DAY_V2_CATASTROPHIC symbol=? adverse=%.4f%% threshold=%.4f%% exit_est=%.6f",
                adverse_move * 100,
                catastro_pct * 100,
                exit_px,
            )
            return {
                "action": "sell",
                "reason": "DAY_V2_CATASTROPHIC_PROTECTION",
                "exit_price_estimate": exit_px,
                "detail": (f"adverse={adverse_move * 100:.2f}% >= {catastro_pct * 100:.2f}% ({DAY_V2_CATASTROPHIC_ATR_MULTIPLIER}x ATR)"),
            }

    # Role 2: Structural invalidation — only fires after N closed bars
    if structural_anchor > 0 and bars_held_approx >= DAY_V2_STRUCTURAL_INVALIDATION_BARS_CLOSED and current_price < structural_anchor:
        logger.warning(
            "DAY_V2_STRUCTURAL price=%.6f < anchor=%.6f bars_approx=%d",
            current_price,
            structural_anchor,
            bars_held_approx,
        )
        return {
            "action": "sell",
            "reason": "DAY_V2_STRUCTURAL_INVALIDATION",
            "exit_price_estimate": current_price,
            "detail": (f"price={current_price:.6f} < anchor={structural_anchor:.6f} bars_approx={bars_held_approx}"),
        }

    # Role 3: Structure-runner ratchet
    atr_1h = float(atr_1h_at_entry or 0.0)
    if atr_1h <= 0 and atr_at_entry > 0:
        atr_1h = LEGACY_ATR_1H_PER_ATR_15M * atr_at_entry
    structural = float(objective_structural or 0.0) or float(target_price or 0.0)
    objective = objective_level(setup, entry_price, atr_1h, structural) if atr_1h > 0 else 0.0
    runner = runner_stop(
        entry_price=entry_price,
        highest_price=highest_price,
        atr_1h=atr_1h,
        objective=objective,
        estimated_roundtrip_cost=estimated_roundtrip_cost,
    )
    if runner["activated"] and current_price <= runner["stop"]:
        reason = "DAY_V2_OBJECTIVE_COMPLETE" if runner["objective_reached"] else "DAY_V2_WINNER_PROTECTION"
        mfe_pct = (max(highest_price, entry_price) - entry_price) / entry_price
        logger.warning(
            "DAY_V2_RUNNER_STOP reason=%s mfe=%.3f%% stop=%.6f objective=%.6f trail_atr_1h=%.2f price=%.6f",
            reason,
            mfe_pct * 100,
            runner["stop"],
            objective,
            runner["trail_atr_1h"],
            current_price,
        )
        return {
            "action": "sell",
            "reason": reason,
            "exit_price_estimate": current_price,
            "detail": (f"mfe={mfe_pct * 100:.2f}% stop={runner['stop']:.6f} objective={objective:.6f} atr_1h={atr_1h:.6f} trail={runner['trail_atr_1h']}x"),
        }

    # Role 4: Time expiration — only a trade that never proved itself and is
    # still net-negative after costs. An armed runner is never timed out.
    pnl_pct = (current_price - entry_price) / entry_price
    net_pnl = pnl_pct - estimated_roundtrip_cost
    if hold_minutes >= DAY_V2_MAX_HOLD_MINUTES and net_pnl <= 0 and not runner["activated"]:
        logger.warning(
            "DAY_V2_TIME_EXPIRATION hold=%.1fmin >= %.0fmin net_pnl=%.4f%%",
            hold_minutes,
            DAY_V2_MAX_HOLD_MINUTES,
            net_pnl * 100,
        )
        return {
            "action": "sell",
            "reason": "DAY_V2_TIME_EXPIRATION",
            "exit_price_estimate": current_price,
            "detail": (f"hold={hold_minutes:.0f}min net_pnl={net_pnl * 100:.2f}%"),
        }

    return None  # no exit condition met
