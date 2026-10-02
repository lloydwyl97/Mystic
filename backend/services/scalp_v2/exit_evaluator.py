"""SCALP V2 dedicated exit evaluator.

Distinct from DAY V2 and legacy exits. Active policy: target_stop_horizon
(contract SCALP_V2_TARGET_STOP_HORIZON_V1).

A scalp is admitted only when its canonical executable net edge is positive
(scalp_v2.executable_edge). The exit takes the adaptive target, cuts a scalp
that moves against it by the learned adverse distance (never wider than the
SCALP adverse bound), and ends every scalp at the SCALP horizon. It never
holds for hours.

Exit ladder (priority order):
  1. Catastrophic stop  — intra-bar adverse move >= SCALP_V2_CATASTROPHIC_PCT (1.5%)
  2. Net profit take    — net P&L >= scalp_v2_min_net_profit_pct
                          (SCALP_NET_PROFIT_TARGET_PCT, default 0.25%)
  3. Adverse stop       — net P&L <= -min(SCALP_PATH_MAX_ADVERSE_NET_PCT,
                          learned gross MAE + round-trip cost)
  4. Giveback           — off unless SCALP_V2_GIVEBACK_EXIT_ENABLED=true
  5. Stall              — off unless SCALP_V2_STALL_EXIT_ENABLED=true
  6. Horizon            — hold >= SCALP_V2_TIME_STOP_MIN (SCALP_HOLD_MAX_MINUTES,
                          default 20 min), whatever the sign

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
    scalp_v2_max_adverse_net_pct,
    scalp_v2_min_net_profit_pct,
    scalp_v2_stall_exit_enabled,
)

logger = logging.getLogger(__name__)

SCALP_V2_ENGINE_ID = "SCALP_V2"

# SCALP horizon (minutes): the canonical SCALP hold (SCALP_HOLD_MAX_MINUTES).
SCALP_V2_TIME_STOP_MIN: float = float(os.getenv("SCALP_V2_TIME_STOP_MIN") or os.getenv("SCALP_HOLD_MAX_MINUTES") or "20")

# Hard catastrophic stop: fraction adverse from entry.
# 1.5% chosen to be wide enough to absorb a real scalp dip (min_dip = 20 bps)
# but tight enough to prevent catastrophic loss on adverse spikes.
SCALP_V2_CATASTROPHIC_PCT: float = float(os.getenv("SCALP_V2_CATASTROPHIC_PCT", "0.015"))

# Exit reason labels — SCALP-specific so scorecards attribute correctly.
SCALP_V2_EXIT_CATASTROPHIC = "SCALP_V2_CATASTROPHIC_STOP"
SCALP_V2_EXIT_NET_PROFIT = "SCALP_V2_NET_PROFIT"
SCALP_V2_EXIT_ADVERSE = "SCALP_V2_ADVERSE_STOP"
SCALP_V2_EXIT_GIVEBACK = "SCALP_V2_GIVEBACK"
SCALP_V2_EXIT_STALL = "SCALP_V2_STALL"
SCALP_V2_EXIT_TIME_STOP = "SCALP_V2_TIME_STOP"

# Reporting labels for paper_trades.exit_reason / exit_type. The raw SCALP_V2_*
# trigger stays in explainability raw_exit_reason.
_SCALP_V2_RECORDED_EXIT_REASONS: dict[str, str] = {
    SCALP_V2_EXIT_CATASTROPHIC: "STOP_LOSS_EXIT",
    SCALP_V2_EXIT_NET_PROFIT: "NET_PROFIT_EXIT",
    SCALP_V2_EXIT_ADVERSE: "STOP_LOSS_EXIT",
    SCALP_V2_EXIT_GIVEBACK: "GIVEBACK_EXIT",
    SCALP_V2_EXIT_STALL: "STALL_EXIT",
    SCALP_V2_EXIT_TIME_STOP: "TIME_STOP_EXIT",
}

# Exit cadence for open SCALP lots inside the existing exit-monitor loop. The
# shared 45s cadence let an adverse move run 6-13 bp past the stop before it
# was seen on ~100s holds.
SCALP_V2_EXIT_MONITOR_INTERVAL_SEC: float = float(os.getenv("SCALP_V2_EXIT_MONITOR_INTERVAL_SEC", "2.0"))


def scalp_v2_net_pnl_at_bid_pct(entry_price: float, executable_bid: float) -> float:
    """Net P&L if the lot is sold at the executable best bid now.

    Same units as realized net (sell price vs entry fill, net of costs): the
    entry ask and the exit half-spread are already in the two prices, so only
    the taker fees and slippage allowance of the canonical round trip remain.
    """
    from backend.config.trading_economics import ESTIMATED_ROUNDTRIP_COST, ORDERBOOK_HALF_SPREAD_ESTIMATE

    entry = float(entry_price)
    return (float(executable_bid) - entry) / entry - (float(ESTIMATED_ROUNDTRIP_COST) - float(ORDERBOOK_HALF_SPREAD_ESTIMATE))


def scalp_v2_adverse_net_threshold_pct(symbol: str, adaptive_decision: dict | None) -> float:
    """Net-P&L adverse stop distance for an open SCALP position.

    ``risk_estimate`` is the learned gross MAE (price excursion from entry). The
    evaluator compares against net P&L, which already carries the canonical
    round-trip cost, so the learned distance is that excursion plus the cost.
    The contract bound (SCALP_PATH_MAX_ADVERSE_NET_PCT) is the ceiling: learned
    risk can tighten the stop, never widen it.
    """
    from backend.config.trading_economics import ESTIMATED_ROUNDTRIP_COST

    contract_adverse = scalp_v2_max_adverse_net_pct(symbol)
    adapt = adaptive_decision if isinstance(adaptive_decision, dict) else {}
    try:
        learned_mae = float(adapt.get("risk_estimate") or 0.0)
    except (TypeError, ValueError):
        learned_mae = 0.0
    if learned_mae <= 0:
        return contract_adverse
    return min(contract_adverse, learned_mae + float(ESTIMATED_ROUNDTRIP_COST))


def scalp_v2_recorded_exit_reason(exit_trigger: str) -> str:
    """Reporting label for a known SCALP V2 exit trigger, else '' (caller keeps its existing label)."""
    return _SCALP_V2_RECORDED_EXIT_REASONS.get(str(exit_trigger or "").strip().upper(), "")


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
        adapt = getattr(position, "adaptive_decision", None)
        adapt = adapt if isinstance(adapt, dict) else {}

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
        min_net = float(adapt.get("target_pct") or scalp_v2_min_net_profit_pct(sym))
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
        # Role 3: Adverse stop — the scalp moved against the thesis; fail fast
        # ──────────────────────────────────────────────────────────────────
        max_adverse = scalp_v2_adverse_net_threshold_pct(sym, adapt)
        if max_adverse > 0 and net_pnl_pct <= -max_adverse:
            logger.warning(
                "SCALP_V2_ADVERSE_STOP symbol=%s net_pnl=%.4f%% <= -%.4f%%",
                sym,
                net_pnl_pct * 100,
                max_adverse * 100,
            )
            return {
                "action": "sell",
                "reason": SCALP_V2_EXIT_ADVERSE,
                "detail": f"net_pnl={net_pnl_pct * 100:.3f}% <= -{max_adverse * 100:.3f}%",
            }

        # ──────────────────────────────────────────────────────────────────
        # Role 4: Giveback — reuse DAY function with SCALP engine_id context
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
        # Role 5: Stall — flat/dead hold with confirmed adverse drift
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
        # Role 6: Horizon — a scalp ends at the SCALP horizon, win or lose
        # ──────────────────────────────────────────────────────────────────
        hold_limit = float(adapt.get("hold_min") or SCALP_V2_TIME_STOP_MIN)
        hold_limit = min(float(SCALP_V2_TIME_STOP_MIN), max(0.0, hold_limit))
        if hold_minutes >= hold_limit:
            logger.warning(
                "SCALP_V2_TIME_STOP symbol=%s hold=%.1fmin >= %.0fmin net_pnl=%.4f%%",
                sym,
                hold_minutes,
                hold_limit,
                net_pnl_pct * 100,
            )
            return {
                "action": "sell",
                "reason": SCALP_V2_EXIT_TIME_STOP,
                "detail": (f"hold={hold_minutes:.0f}min >= {hold_limit:.0f}min net_pnl={net_pnl_pct * 100:.2f}%"),
            }

        return {"action": "hold", "reason": "SCALP_V2_HOLD"}

    except Exception:
        logger.debug("SCALP_V2_EXIT_EVALUATOR_INTERNAL_ERROR", exc_info=True)
        return {}
