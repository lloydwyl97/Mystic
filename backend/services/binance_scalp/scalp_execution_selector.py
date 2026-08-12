"""
Dynamic execution-style selection for SCALP live order placement (item p21).

Chooses LIMIT_IOC vs MARKET per order based on measured spread, exit urgency,
and adverse-selection risk (from the microstructure engine's aggressor-flow
signal) — replacing a single static env-configured order type used for every
order regardless of context.

This module only decides HOW an already-decided BUY/SELL is placed. It never
delays, blocks, or reverses an entry/exit decision — consistent with the
"ranking, not gating" architecture rule. An urgent exit (catastrophic stop,
circuit breaker, max-hold) always prefers MARKET (guaranteed fill over a
possibly-better but uncertain price); a patient entry with a wide spread and
low adverse-selection risk prefers LIMIT_IOC for real cost savings, since a
missed entry is not itself a bad outcome — there will be another candidate.

KNOWN LIMITATION (documented honestly, not hidden): the only consumer of
this module, ``scalp_order_bridge.ScalpOrderBridge``, is itself only called
from ``ScalpLiveEngine``, which has zero callers anywhere in the live
signal/ranking loop as of this writing (its own docstring says
"NOT YET CONNECTED TO paper_engine signals — Phase 2 wiring"). SCALP runs
100% in paper mode today. This module is IMPLEMENTED and WIRED into the one
real live order-placement code path that exists, and is fully unit-tested,
but has no live runtime caller until SCALP live trading itself is armed and
connected — that gap predates this change (see item p26 dead-code audit).
"""

from __future__ import annotations

import os
from dataclasses import dataclass

WIDE_SPREAD_LIMIT_THRESHOLD_PCT = float(os.getenv("SCALP_EXEC_WIDE_SPREAD_PCT", "0.0008"))  # 8 bps
HIGH_ADVERSE_SELECTION_THRESHOLD = float(os.getenv("SCALP_EXEC_ADVERSE_SELECTION_THRESHOLD", "0.6"))

VALID_ORDER_TYPES = ("MARKET", "LIMIT_IOC")


@dataclass(frozen=True)
class ExecutionChoice:
    order_type: str  # "MARKET" | "LIMIT_IOC"
    reason: str


def choose_execution_style(
    *,
    is_urgent_exit: bool,
    spread_pct: float,
    adverse_selection_risk: float = 0.0,
) -> ExecutionChoice:
    """Pure decision function. `adverse_selection_risk` is expected in
    [0, 1] (e.g. normalized aggressor-sell-flow-against-a-passive-buy
    strength from microstructure_engine); values outside that range are
    clamped defensively rather than raising."""
    risk = max(0.0, min(1.0, adverse_selection_risk))
    spread = max(0.0, spread_pct)

    if is_urgent_exit:
        return ExecutionChoice("MARKET", "urgent_exit_guaranteed_fill")

    if risk >= HIGH_ADVERSE_SELECTION_THRESHOLD:
        return ExecutionChoice("MARKET", "high_adverse_selection_prefer_certainty")

    if spread >= WIDE_SPREAD_LIMIT_THRESHOLD_PCT:
        return ExecutionChoice("LIMIT_IOC", "wide_spread_low_adverse_selection")

    return ExecutionChoice("MARKET", "tight_spread_default_market")


def resolve_order_type(
    *,
    is_urgent_exit: bool,
    spread_pct: float,
    adverse_selection_risk: float = 0.0,
) -> str:
    """Entry point used by scalp_order_bridge.py. Honors SCALP_ORDER_TYPE_OVERRIDE
    as an explicit operator kill-switch (set to MARKET or LIMIT_IOC to force
    that style for every order, bypassing dynamic selection entirely).
    Unset/invalid values fall through to the dynamic chooser."""
    override = os.getenv("SCALP_ORDER_TYPE_OVERRIDE", "").strip().upper()
    if override in VALID_ORDER_TYPES:
        return override
    return choose_execution_style(
        is_urgent_exit=is_urgent_exit,
        spread_pct=spread_pct,
        adverse_selection_risk=adverse_selection_risk,
    ).order_type


__all__ = [
    "HIGH_ADVERSE_SELECTION_THRESHOLD",
    "WIDE_SPREAD_LIMIT_THRESHOLD_PCT",
    "ExecutionChoice",
    "choose_execution_style",
    "resolve_order_type",
]
