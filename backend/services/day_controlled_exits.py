"""
Controlled-risk DAY bracket exits — profitability-first, replay-proven before live.

Allowed exits: NET_PROFIT, VOLATILITY_STOP, TIME_STOP, FAILED_RECLAIM, EXTREME_PROTECTION.
Thesis invalidation remains warn-only (no stale THESIS_INVALIDATION sells).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from backend.services.day_trade_thesis import (
    EXIT_EXTREME_PROTECTION,
    EXIT_NET_PROFIT,
    EXIT_THESIS_WARNING,
    SETUP_VWAP_REVERSION,
    evaluate_extreme_protection,
    evaluate_thesis_exit,
)

EXIT_VOLATILITY_STOP = "VOLATILITY_STOP_EXIT"
EXIT_TIME_STOP = "TIME_STOP_EXIT"
EXIT_FAILED_RECLAIM = "FAILED_RECLAIM_EXIT"

ALLOWED_DAY_EXIT_REASONS = frozenset({
    EXIT_NET_PROFIT,
    EXIT_VOLATILITY_STOP,
    EXIT_TIME_STOP,
    EXIT_FAILED_RECLAIM,
    EXIT_EXTREME_PROTECTION,
    "MANUAL_EXIT",
    "LEGACY_CLEANUP_EXIT",
    "LEGACY_INVENTORY_CLEANUP_EXIT",
    "ADMIN_CLEAR",
})


@dataclass(frozen=True)
class ControlledExitConfig:
    """Replay/live bracket parameters."""

    enabled: bool = False
    profit_floor_pct: float = 0.004
    atr_stop_mult: float = 1.0
    time_stop_hours: float = 48.0
    max_loss_pct: float = 0.015
    failed_reclaim_hours: float = 6.0
    failed_reclaim_buffer_pct: float = 0.0015
    use_fill_based_gate: bool = True


def evaluate_controlled_bracket_exit(
    *,
    entry_price: float,
    mark: float,
    bar_low: float,
    entry_ts: int,
    bar_ts: int,
    setup: str,
    invalid_level: float,
    atr_pct: float,
    net_pct_fill: float,
    net_pct_mid: float,
    bundle: dict[str, Any] | None,
    cfg: ControlledExitConfig,
    entry_vwap: float = 0.0,
) -> dict[str, Any]:
    """
    Bracket exit: profit target + volatility stop + time stop + failed reclaim.
    Never sells on thesis invalidation alone (warn-only path preserved).
    """
    if entry_price <= 0 or mark <= 0:
        return {"action": "hold", "reason": "missing_price"}

    atr = max(0.003, float(atr_pct or 0.01))
    hold_h = max(0.0, (bar_ts - entry_ts) / 3600.0)
    gate = net_pct_fill if cfg.use_fill_based_gate else net_pct_mid

    extreme = evaluate_extreme_protection(
        entry_price=entry_price,
        mark=mark,
        net_pnl_pct=net_pct_mid,
        atr_pct=atr,
        bundle=bundle,
    )
    if str(extreme.get("action")) == "sell":
        return {
            "action": "sell",
            "reason": EXIT_EXTREME_PROTECTION,
            "net_pnl_pct": gate,
            "hold_hours": hold_h,
        }

    if not cfg.enabled:
        te = evaluate_thesis_exit(
            entry_thesis=setup,
            thesis_score=0.5,
            thesis_invalid_level=invalid_level,
            thesis_target_level=0.0,
            entry_vwap=entry_vwap,
            entry_price=entry_price,
            mark=mark,
            bundle=bundle,
        )
        if str(te.get("action")) == "warn":
            return {"action": "hold", "reason": EXIT_THESIS_WARNING, "net_pnl_pct": gate}
        if str(te.get("action")) == "sell" or gate >= cfg.profit_floor_pct:
            if gate >= cfg.profit_floor_pct:
                return {"action": "sell", "reason": EXIT_NET_PROFIT, "net_pnl_pct": gate, "hold_hours": hold_h}
        return {"action": "hold", "reason": "profit_only_hold", "net_pnl_pct": gate}

    stop_dist = cfg.atr_stop_mult * atr
    stop_px = entry_price * (1.0 - stop_dist)
    eff_loss_pct = min(stop_dist, cfg.max_loss_pct)
    if bar_low <= stop_px or gate <= -eff_loss_pct:
        return {
            "action": "sell",
            "reason": EXIT_VOLATILITY_STOP,
            "net_pnl_pct": gate,
            "hold_hours": hold_h,
            "stop_dist_pct": stop_dist,
        }

    if (
        setup == SETUP_VWAP_REVERSION
        and hold_h >= cfg.failed_reclaim_hours
        and gate < cfg.profit_floor_pct * 0.5
    ):
        buf = cfg.failed_reclaim_buffer_pct
        ref = entry_vwap if entry_vwap > 0 else entry_price
        if mark < ref * (1.0 - buf):
            return {
                "action": "sell",
                "reason": EXIT_FAILED_RECLAIM,
                "net_pnl_pct": gate,
                "hold_hours": hold_h,
            }

    if hold_h >= cfg.time_stop_hours and gate < cfg.profit_floor_pct:
        return {
            "action": "sell",
            "reason": EXIT_TIME_STOP,
            "net_pnl_pct": gate,
            "hold_hours": hold_h,
        }

    if gate + 1e-12 >= cfg.profit_floor_pct:
        return {
            "action": "sell",
            "reason": EXIT_NET_PROFIT,
            "net_pnl_pct": gate,
            "hold_hours": hold_h,
        }

    te = evaluate_thesis_exit(
        entry_thesis=setup,
        thesis_score=0.5,
        thesis_invalid_level=invalid_level,
        thesis_target_level=0.0,
        entry_vwap=entry_vwap,
        entry_price=entry_price,
        mark=mark,
        bundle=bundle,
    )
    if str(te.get("action")) == "warn":
        return {"action": "hold", "reason": EXIT_THESIS_WARNING, "net_pnl_pct": gate}

    return {"action": "hold", "reason": "bracket_hold", "net_pnl_pct": gate, "hold_hours": hold_h}
