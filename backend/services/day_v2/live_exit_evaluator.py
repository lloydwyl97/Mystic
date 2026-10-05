"""DAY V2 live exit evaluation.

Evaluates real open positions with engine_id='DAY_V2'. Returns an action
dict or None. Called from portfolio_engine._check_exit_conditions.

This is NOT shadow-only. It routes to real order execution. Do not add
assert_no_live_authority() calls here.

Exit priority (highest to lowest):
  1. CATASTROPHIC_PROTECTION  — adverse move beyond max(3x 15m ATR,
                                distance from entry to the structural anchor
                                plus one 15m ATR). Quiet ATR cannot fire inside
                                the structural region.
  2. STRUCTURAL_INVALIDATION  — price below the setup's structural anchor (after 3+ bars)
  3. WINNER_PROTECTION        — structure-runner ratchet (day_v2.winner_contract):
                                live only when the computed ATR trail itself is
                                above break-even after round-trip cost. A trail
                                still at or below that level is not replaced
                                with break-even. Reported as OBJECTIVE_COMPLETE
                                when the setup objective was reached before the stop.

Elapsed hold is telemetry only. It is not a sell authority.
There is no fixed profit target: reaching the objective tightens the ratchet,
it does not sell.
"""

from __future__ import annotations

import logging
import time

from backend.services.day_v2.config import (
    DAY_V2_CATASTROPHIC_ANCHOR_BUFFER_ATR,
    DAY_V2_CATASTROPHIC_ATR_MULTIPLIER,
    DAY_V2_MAX_HOLD_MINUTES,
    DAY_V2_STRUCTURAL_INVALIDATION_BARS_CLOSED,
)
from backend.services.day_v2.winner_contract import (
    DAY_EXIT_CONTRACT_RUNNER,
    LEGACY_ATR_1H_PER_ATR_15M,
    RUNNER_ACTIVATION_ATR_1H,
    RUNNER_TIGHT_TRAIL_ATR_1H,
    RUNNER_TRAIL_ATR_1H,
    objective_level,
    runner_stop,
)

logger = logging.getLogger(__name__)

DAY_V2_ENGINE_ID: str = "DAY_V2"


def catastrophic_threshold_price(entry_price: float, atr_at_entry: float, structural_anchor: float) -> float:
    """Price at which catastrophic protection fires.

    The distance is the larger of 3x 15m ATR and (entry-to-anchor plus one
    15m ATR). Missing or non-positive anchors keep the 3x ATR distance so
    hard safety is not removed.
    """
    atr_distance = DAY_V2_CATASTROPHIC_ATR_MULTIPLIER * max(0.0, atr_at_entry)
    if entry_price > 0 and atr_at_entry > 0 and 0.0 < structural_anchor < entry_price:
        outside_structure = (entry_price - structural_anchor) + DAY_V2_CATASTROPHIC_ANCHOR_BUFFER_ATR * atr_at_entry
        atr_distance = max(atr_distance, outside_structure)
    return entry_price - atr_distance


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


def _runner_state(
    *,
    entry_price: float,
    highest_price: float,
    atr_at_entry: float,
    target_price: float,
    estimated_roundtrip_cost: float,
    setup: str,
    atr_1h_at_entry: float,
    objective_structural: float,
    objective_atr_mult: float,
    structural_emphasis: float,
    runner_activation_mult: float,
    runner_trail_mult: float,
    runner_tighten_mult: float,
) -> tuple[float, float, dict]:
    """(1h ATR, objective, runner_stop state) exactly as the live exit evaluates them."""
    atr_1h = float(atr_1h_at_entry or 0.0)
    if atr_1h <= 0 and atr_at_entry > 0:
        atr_1h = LEGACY_ATR_1H_PER_ATR_15M * atr_at_entry
    structural = float(objective_structural or 0.0) or float(target_price or 0.0)
    objective = objective_level(setup, entry_price, atr_1h, structural, atr_mult=objective_atr_mult, structural_emphasis=structural_emphasis) if atr_1h > 0 else 0.0
    runner = runner_stop(
        entry_price=entry_price,
        highest_price=highest_price,
        atr_1h=atr_1h,
        objective=objective,
        estimated_roundtrip_cost=estimated_roundtrip_cost,
        activation_mult=runner_activation_mult,
        trail_mult=runner_trail_mult,
        tighten_mult=runner_tighten_mult,
    )
    return atr_1h, objective, runner


def day_v2_exit_policy() -> dict:
    """Status description of the DAY_V2 live exit contract."""
    return {
        "exit_contract": DAY_EXIT_CONTRACT_RUNNER,
        "automated_sells_triggered_by": "day_v2_live_exit_contract",
        "exit_paths": [
            "DAY_V2_CATASTROPHIC_PROTECTION",
            "DAY_V2_STRUCTURAL_INVALIDATION",
            "DAY_V2_WINNER_PROTECTION",
            "DAY_V2_OBJECTIVE_COMPLETE",
        ],
        "catastrophic_basis": (f"low <= entry - max({DAY_V2_CATASTROPHIC_ATR_MULTIPLIER}x 15m ATR, entry-to-anchor + {DAY_V2_CATASTROPHIC_ANCHOR_BUFFER_ATR}x 15m ATR)"),
        "structural_invalidation_bars_required": DAY_V2_STRUCTURAL_INVALIDATION_BARS_CLOSED,
        "runner_activation_atr_1h": RUNNER_ACTIVATION_ATR_1H,
        "runner_trail_atr_1h": RUNNER_TRAIL_ATR_1H,
        "runner_tight_trail_atr_1h": RUNNER_TIGHT_TRAIL_ATR_1H,
        "runner_requires_trail_above_breakeven_after_cost": True,
        "hold_time_is_exit_authority": False,
        "time_exit_sell_path_active": False,
        "fixed_take_profit_active": False,
        "stop_tp_fields_drive_engine_exits": False,
        "code_path": "monitor_all_positions -> _check_exit_conditions -> evaluate_day_v2_exit",
    }


def preview_day_v2_exit(
    *,
    entry_price: float,
    current_price: float,
    highest_price: float,
    atr_at_entry: float,
    structural_anchor: float,
    target_price: float,
    entry_time: float,
    estimated_roundtrip_cost: float,
    setup: str = "",
    atr_1h_at_entry: float = 0.0,
    objective_structural: float = 0.0,
    objective_atr_mult: float = 1.0,
    structural_emphasis: float = 1.0,
    runner_activation_mult: float = 1.0,
    runner_trail_mult: float = 1.0,
    runner_tighten_mult: float = 1.0,
    now: float | None = None,
) -> dict:
    """Read-only status view of the DAY_V2 exit contract. Never an exit decision.

    Levels come from the same functions ``evaluate_day_v2_exit`` uses. There is
    no time exit, fixed take-profit, or percentage stop in this contract.
    """
    ts = time.time() if now is None else float(now)
    hold_minutes = max(0.0, (ts - entry_time) / 60.0) if entry_time > 0 else 0.0
    bars_held_approx = int(hold_minutes / 15.0)
    catastrophic = catastrophic_threshold_price(entry_price, atr_at_entry, structural_anchor) if entry_price > 0 and atr_at_entry > 0 else None
    atr_1h, objective, runner = _runner_state(
        entry_price=entry_price,
        highest_price=highest_price,
        atr_at_entry=atr_at_entry,
        target_price=target_price,
        estimated_roundtrip_cost=estimated_roundtrip_cost,
        setup=setup,
        atr_1h_at_entry=atr_1h_at_entry,
        objective_structural=objective_structural,
        objective_atr_mult=objective_atr_mult,
        structural_emphasis=structural_emphasis,
        runner_activation_mult=runner_activation_mult,
        runner_trail_mult=runner_trail_mult,
        runner_tighten_mult=runner_tighten_mult,
    )
    activation_mult = RUNNER_ACTIVATION_ATR_1H * max(0.80, min(1.25, float(runner_activation_mult or 1.0)))
    arming_price = entry_price + activation_mult * atr_1h if atr_1h > 0 and entry_price > 0 else None
    structural_active = structural_anchor > 0 and bars_held_approx >= DAY_V2_STRUCTURAL_INVALIDATION_BARS_CLOSED
    if runner["activated"]:
        authority = "DAY_V2_OBJECTIVE_COMPLETE" if runner["objective_reached"] else "DAY_V2_WINNER_PROTECTION"
        next_exit = f"price <= runner stop {runner['stop']:.6f}"
    elif structural_active:
        authority = "DAY_V2_STRUCTURAL_INVALIDATION"
        next_exit = f"price < structural anchor {structural_anchor:.6f}"
    else:
        authority = "DAY_V2_CATASTROPHIC_PROTECTION"
        next_exit = f"low <= catastrophic {catastrophic:.6f}" if catastrophic else "no executable exit level (missing entry ATR)"
    return {
        "exit_contract": DAY_EXIT_CONTRACT_RUNNER,
        "catastrophic_price": catastrophic,
        "structural_anchor": structural_anchor if structural_anchor > 0 else None,
        "structural_invalidation_active": bool(structural_active),
        "structural_invalidation_bars_required": DAY_V2_STRUCTURAL_INVALIDATION_BARS_CLOSED,
        "bars_held_approx": bars_held_approx,
        "atr_1h": atr_1h or None,
        "objective_price": objective or None,
        "objective_reached": bool(runner["objective_reached"]),
        "runner_arming_price": arming_price,
        "runner_activated": bool(runner["activated"]),
        "runner_stop": runner["stop"] or None,
        "runner_trail_atr_1h": runner["trail_atr_1h"] or None,
        "high_water": max(float(highest_price or 0.0), float(entry_price or 0.0)) or None,
        "current_exit_authority": authority,
        "next_executable_exit_condition": next_exit,
        "hold_minutes": round(hold_minutes, 2),
        "hold_time_is_exit_authority": False,
        "time_exit": None,
        "fixed_take_profit": None,
    }


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
    objective_atr_mult: float = 1.0,
    structural_emphasis: float = 1.0,
    runner_activation_mult: float = 1.0,
    runner_trail_mult: float = 1.0,
    runner_tighten_mult: float = 1.0,
    now: float | None = None,
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
        now: Evaluation clock (defaults to wall clock). The lifecycle
            simulator passes each bar's close time.

    Returns:
        {"action": "sell", "reason": str, "exit_price_estimate": float,
         "detail": str}
        or None if no exit condition is met.
    """
    if engine_id != DAY_V2_ENGINE_ID:
        return None
    if entry_price <= 0 or current_price <= 0:
        return None

    now = time.time() if now is None else float(now)
    hold_minutes = max(0.0, (now - entry_time) / 60.0) if entry_time > 0 else 0.0
    # Approximate closed-15m-bar count from wall-clock hold time.
    # Used only for the structural-invalidation guard (requires N closed bars).
    bars_held_approx = int(hold_minutes / 15.0)

    # Role 1: Catastrophic protection — uses the lowest price since entry.
    # Sits outside the structural anchor when that anchor is known, so a quiet
    # 15m ATR cannot front-run normal thesis invalidation.
    if atr_at_entry > 0:
        exit_px = catastrophic_threshold_price(entry_price, atr_at_entry, structural_anchor)
        if max(bar_low, 0.0) <= exit_px:
            adverse_move = (entry_price - max(bar_low, 0.0)) / entry_price
            catastro_pct = (entry_price - exit_px) / entry_price
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
                "detail": (f"adverse={adverse_move * 100:.2f}% >= {catastro_pct * 100:.2f}% (outside anchor or {DAY_V2_CATASTROPHIC_ATR_MULTIPLIER}x ATR)"),
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
    atr_1h, objective, runner = _runner_state(
        entry_price=entry_price,
        highest_price=highest_price,
        atr_at_entry=atr_at_entry,
        target_price=target_price,
        estimated_roundtrip_cost=estimated_roundtrip_cost,
        setup=setup,
        atr_1h_at_entry=atr_1h_at_entry,
        objective_structural=objective_structural,
        objective_atr_mult=objective_atr_mult,
        structural_emphasis=structural_emphasis,
        runner_activation_mult=runner_activation_mult,
        runner_trail_mult=runner_trail_mult,
        runner_tighten_mult=runner_tighten_mult,
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

    if hold_minutes >= DAY_V2_MAX_HOLD_MINUTES:
        logger.debug(
            "DAY_V2_HOLD_TELEMETRY hold=%.1fmin ceiling=%.0fmin price=%.6f anchor=%.6f",
            hold_minutes,
            DAY_V2_MAX_HOLD_MINUTES,
            current_price,
            structural_anchor,
        )

    return None  # no exit condition met
