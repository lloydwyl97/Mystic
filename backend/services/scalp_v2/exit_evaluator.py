"""SCALP V2 dedicated exit evaluator.

Distinct from DAY V2 and legacy exits. Uses the calibrated policy from
exit_calibration.py: preserve_giveback_stall_and_hard_stop.

Exit ladder (priority order):
  1. Catastrophic stop  — intra-bar adverse move >= SCALP_V2_CATASTROPHIC_PCT
  2. Net profit take    — net P&L >= scalp_v2_min_net_profit_pct (default 0.4%)
  3. Giveback           — reached MFE, then reversed to net-negative
  4. Stall              — flat/dead hold with confirmed adverse drift
  5. Time stop          — hold >= SCALP_V2_TIME_STOP_MIN (default 120 min) and
                          still net-negative

SCALP V2 does NOT use:
  - DAY structural invalidation (no 4H/1H thesis anchor on a scalp)
  - DAY objective complete (no multi-ATR structural target)
  - DAY 300-min time ceiling (too long for a scalp setup)
  - allweather bracket exits
  - trailing buy min_dip / rebound parameters

Every call must receive a position object with engine_id == 'SCALP_V2'.
Any other engine_id returns an empty dict (fail-closed, no spurious exits).
"""

from __future__ import annotations

import logging
import os
import time
from typing import Any

from backend.services.scalp_v2.exit_calibration import (
    scalp_v2_giveback_exit_enabled,
    scalp_v2_min_net_profit_pct,
    scalp_v2_stall_exit_enabled,
)

logger = logging.getLogger(__name__)

SCALP_V2_ENGINE_ID = "SCALP_V2"

# SCALP time ceiling (minutes). Shorter than DAY's 300 min because stale
# scalp positions bleed opportunity cost. Exit only when net-negative.
SCALP_V2_TIME_STOP_MIN: float = float(os.getenv("SCALP_V2_TIME_STOP_MIN", "120"))

# Hard catastrophic stop: fraction adverse from entry.
# 1.5% chosen to be wide enough to absorb a real scalp dip (min_dip = 20 bps)
# but tight enough to prevent catastrophic loss on adverse spikes.
SCALP_V2_CATASTROPHIC_PCT: float = float(os.getenv("SCALP_V2_CATASTROPHIC_PCT", "0.015"))

# Exit reason labels — SCALP-specific so scorecards attribute correctly.
SCALP_V2_EXIT_CATASTROPHIC = "SCALP_V2_CATASTROPHIC_STOP"
SCALP_V2_EXIT_NET_PROFIT = "SCALP_V2_NET_PROFIT"
SCALP_V2_EXIT_GIVEBACK = "SCALP_V2_GIVEBACK"
SCALP_V2_EXIT_STALL = "SCALP_V2_STALL"
SCALP_V2_EXIT_TIME_STOP = "SCALP_V2_TIME_STOP"


def evaluate_scalp_v2_exit(
    *,
    position: Any,
    current_price: float,
    net_pnl_pct: float,
    hold_minutes: float,
    bar_low: float,
    coin_profile: dict | None = None,
    symbol: str = "",
) -> dict[str, Any]:
    """Evaluate all SCALP V2 exit roles for an open SCALP position.

    Args:
        position:       Position object. engine_id must equal 'SCALP_V2'.
        current_price:  Current mark price (ask or last).
        net_pnl_pct:    Net P&L fraction after estimated round-trip cost.
        hold_minutes:   Minutes since position entry.
        bar_low:        Bar low of the current monitoring period (intra-bar).
        coin_profile:   Optional coin-profile dict (unused today; reserved for
                        per-coin calibration overrides).
        symbol:         Symbol string for log messages.

    Returns:
        dict with {"action": "sell", "reason": ..., "detail": ...}
        or {"action": "hold", "reason": ...}
        or {} (empty) if engine_id is not SCALP_V2 or inputs invalid.

    Raises:
        Never. All exceptions are swallowed and logged at DEBUG level.
    """
    try:
        engine_id = str(getattr(position, "engine_id", "") or "")
        if engine_id != SCALP_V2_ENGINE_ID:
            return {}

        entry_price = float(getattr(position, "cost_basis", 0.0) or getattr(position, "entry_price", 0.0) or 0.0)
        if entry_price <= 0 or current_price <= 0:
            return {}

        highest_price = float(getattr(position, "highest_price", 0.0) or entry_price)
        lowest_price = float(getattr(position, "lowest_price", 0.0) or current_price)
        highest_price = max(highest_price, entry_price)
        if lowest_price <= 0:
            lowest_price = min(current_price, entry_price)

        sym = symbol or str(getattr(position, "symbol", "") or "")

        # ──────────────────────────────────────────────────────────────────
        # Role 1: Catastrophic stop — intra-bar adverse move
        # ──────────────────────────────────────────────────────────────────
        _effective_low = min(bar_low, current_price) if bar_low > 0 else current_price
        if _effective_low > 0 and entry_price > 0:
            adverse_move = (entry_price - _effective_low) / entry_price
            if adverse_move >= SCALP_V2_CATASTROPHIC_PCT:
                logger.warning(
                    "SCALP_V2_CATASTROPHIC symbol=%s adverse=%.4f%% threshold=%.4f%%",
                    sym,
                    adverse_move * 100,
                    SCALP_V2_CATASTROPHIC_PCT * 100,
                )
                return {
                    "action": "sell",
                    "reason": SCALP_V2_EXIT_CATASTROPHIC,
                    "detail": (f"adverse={adverse_move * 100:.2f}% >= {SCALP_V2_CATASTROPHIC_PCT * 100:.2f}% bar_low={_effective_low:.6f} entry={entry_price:.6f}"),
                }

        # ──────────────────────────────────────────────────────────────────
        # Role 2: Net profit take
        # ──────────────────────────────────────────────────────────────────
        min_net = scalp_v2_min_net_profit_pct(sym)
        if net_pnl_pct >= min_net:
            logger.info(
                "SCALP_V2_PROFIT_TAKE symbol=%s net_pnl=%.4f%% >= min=%.4f%%",
                sym,
                net_pnl_pct * 100,
                min_net * 100,
            )
            return {
                "action": "sell",
                "reason": SCALP_V2_EXIT_NET_PROFIT,
                "detail": f"net_pnl={net_pnl_pct * 100:.3f}% >= min={min_net * 100:.3f}%",
            }

        # ──────────────────────────────────────────────────────────────────
        # Role 3: Giveback — reuse DAY function with SCALP engine_id context
        # ──────────────────────────────────────────────────────────────────
        if scalp_v2_giveback_exit_enabled():
            try:
                from backend.services.day_controlled_exits import evaluate_giveback_exit

                gb = evaluate_giveback_exit(
                    entry_price=entry_price,
                    highest_price=highest_price,
                    net_pnl_pct=net_pnl_pct,
                    hold_minutes=hold_minutes,
                    position=position,
                )
                if gb and str(gb.get("action") or "") == "sell":
                    # Relabel so scorecards attribute to SCALP, not legacy DAY.
                    reason = str(gb.get("reason") or SCALP_V2_EXIT_GIVEBACK)
                    if "GIVEBACK" in reason.upper():
                        reason = SCALP_V2_EXIT_GIVEBACK
                    logger.info(
                        "SCALP_V2_GIVEBACK symbol=%s net_pnl=%.4f%%",
                        sym,
                        net_pnl_pct * 100,
                    )
                    return {
                        "action": "sell",
                        "reason": reason,
                        "detail": str(gb.get("detail", "")),
                    }
            except Exception:
                logger.debug("SCALP_V2_GIVEBACK_EVAL_ERROR symbol=%s", sym, exc_info=True)

        # ──────────────────────────────────────────────────────────────────
        # Role 4: Stall — flat/dead hold with confirmed adverse drift
        # ──────────────────────────────────────────────────────────────────
        if scalp_v2_stall_exit_enabled():
            try:
                from backend.services.day_controlled_exits import evaluate_stall_exit

                stall = evaluate_stall_exit(
                    entry_price=entry_price,
                    highest_price=highest_price,
                    net_pnl_pct=net_pnl_pct,
                    hold_minutes=hold_minutes,
                    max_hold_min=int(SCALP_V2_TIME_STOP_MIN),
                    current_price=current_price,
                    lowest_price=lowest_price,
                )
                if stall and str(stall.get("action") or "") == "sell":
                    reason = str(stall.get("reason") or SCALP_V2_EXIT_STALL)
                    if "STALL" in reason.upper():
                        reason = SCALP_V2_EXIT_STALL
                    logger.info(
                        "SCALP_V2_STALL symbol=%s net_pnl=%.4f%%",
                        sym,
                        net_pnl_pct * 100,
                    )
                    return {
                        "action": "sell",
                        "reason": reason,
                        "detail": str(stall.get("detail", "")),
                    }
            except Exception:
                logger.debug("SCALP_V2_STALL_EVAL_ERROR symbol=%s", sym, exc_info=True)

        # ──────────────────────────────────────────────────────────────────
        # Role 5: Time stop — exit only when net-negative after SCALP ceiling
        # ──────────────────────────────────────────────────────────────────
        if hold_minutes >= SCALP_V2_TIME_STOP_MIN and net_pnl_pct <= 0:
            logger.warning(
                "SCALP_V2_TIME_STOP symbol=%s hold=%.1fmin >= %.0fmin net_pnl=%.4f%%",
                sym,
                hold_minutes,
                SCALP_V2_TIME_STOP_MIN,
                net_pnl_pct * 100,
            )
            return {
                "action": "sell",
                "reason": SCALP_V2_EXIT_TIME_STOP,
                "detail": (f"hold={hold_minutes:.0f}min >= {SCALP_V2_TIME_STOP_MIN:.0f}min net_pnl={net_pnl_pct * 100:.2f}%"),
            }

        return {"action": "hold", "reason": "SCALP_V2_HOLD"}

    except Exception:
        logger.debug("SCALP_V2_EXIT_EVALUATOR_INTERNAL_ERROR", exc_info=True)
        return {}
