"""DAY V2 live exit evaluation.

Evaluates real open positions with engine_id='DAY_V2'. Returns an action
dict or None. Called from portfolio_engine._check_exit_conditions.

This is NOT shadow-only. It routes to real order execution. Do not add
assert_no_live_authority() calls here.

Exit priority (highest to lowest):
  1. CATASTROPHIC_PROTECTION  — intra-bar adverse move >= 3x ATR
  2. STRUCTURAL_INVALIDATION  — closed price below structural anchor (after 3+ bars)
  3. WINNER_PROTECTION        — trail from highest price once MFE >= 0.8%,
                                never below break-even after round-trip costs
  4. OBJECTIVE_COMPLETE       — price reaches or exceeds target
  5. TIME_EXPIRATION          — hold >= 300 min and still net-negative

Calibrated from qualifying replay (day_v2_replay.py @ a88479a):
  MAX_HOLD = 300 min, CATASTRO_ATR_MULT = 3.0, MIN_MFE_FOR_WINNER = 0.8%
  WINNER_TRAIL_ATR_MULT = 1.5, WINNER_TRAIL_FLOOR = 0.5%
  STRUCTURAL_BARS = 3
"""

from __future__ import annotations

import logging
import time

from backend.services.day_v2.config import (
    DAY_V2_CATASTROPHIC_ATR_MULTIPLIER,
    DAY_V2_MAX_HOLD_MINUTES,
    DAY_V2_STRUCTURAL_INVALIDATION_BARS_CLOSED,
    DAY_V2_WINNER_PROTECTION_MIN_MFE_PCT,
)

logger = logging.getLogger(__name__)

# Winner-trail calibration — from qualifying replay
WINNER_TRAIL_ATR_MULT: float = 1.5  # trail = max(floor, 1.5 x ATR)
WINNER_TRAIL_FLOOR_PCT: float = 0.005  # 0.5% minimum trail distance

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
) -> dict | None:
    """Evaluate all DAY V2 exit roles.

    Args:
        engine_id: Must be "DAY_V2" or this function returns None.
        entry_price: Average fill price.
        current_price: Current mark (closed 15m bar close or last tick).
        bar_low: Low of the current monitoring period (for catastrophic check).
        highest_price: Session high-watermark (for winner trail).
        atr_at_entry: ATR measured at entry (absolute price units).
        structural_anchor: Price below which thesis is invalid.
        target_price: Objective completion price.
        entry_time: Unix epoch of position open.
        estimated_roundtrip_cost: Total cost fraction (e.g. 0.0006).

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

    # Role 1: Catastrophic protection — uses bar_low (intra-bar)
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

    # Role 4: Winner protection — activated after meaningful MFE
    if highest_price > entry_price:
        mfe_pct = (highest_price - entry_price) / entry_price
        if mfe_pct >= DAY_V2_WINNER_PROTECTION_MIN_MFE_PCT and atr_at_entry > 0:
            atr_pct = atr_at_entry / entry_price
            trail_distance = max(WINNER_TRAIL_FLOOR_PCT, WINNER_TRAIL_ATR_MULT * atr_pct)
            break_even = entry_price * (1.0 + max(0.0, estimated_roundtrip_cost))
            trail_trigger = max(highest_price * (1.0 - trail_distance), break_even)
            if current_price <= trail_trigger:
                logger.warning(
                    "DAY_V2_WINNER_TRAIL mfe=%.3f%% trail=%.3f%% trigger=%.6f break_even=%.6f price=%.6f",
                    mfe_pct * 100,
                    trail_distance * 100,
                    trail_trigger,
                    break_even,
                    current_price,
                )
                return {
                    "action": "sell",
                    "reason": "DAY_V2_WINNER_PROTECTION",
                    "exit_price_estimate": current_price,
                    "detail": (f"mfe={mfe_pct * 100:.2f}% trail_dist={trail_distance * 100:.2f}% trigger={trail_trigger:.6f} break_even={break_even:.6f}"),
                }

    # Role 5: Objective complete
    if target_price > 0 and current_price >= target_price:
        logger.info(
            "DAY_V2_OBJECTIVE price=%.6f >= target=%.6f",
            current_price,
            target_price,
        )
        return {
            "action": "sell",
            "reason": "DAY_V2_OBJECTIVE_COMPLETE",
            "exit_price_estimate": current_price,
            "detail": f"price={current_price:.6f} >= target={target_price:.6f}",
        }

    # Role 3: Time expiration — only if still net-negative after costs
    pnl_pct = (current_price - entry_price) / entry_price
    net_pnl = pnl_pct - estimated_roundtrip_cost
    if hold_minutes >= DAY_V2_MAX_HOLD_MINUTES and net_pnl <= 0:
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
