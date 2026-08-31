"""Mandatory DAY flatten residual execution.

Once trail / 4H-break / risk-floor (or another full-flatten) has fired, a
partial IOC must not return the meaningful residual to discretionary ACTIVE
where entry-style spread/impact preflight can refuse liquidation for a full
monitor cycle.

Entry safety is unchanged. This module only applies to already-triggered
full-flatten exits. No market-order fallback. No unlimited slippage.
"""

from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any

from backend.config.protected_execution import (
    MANDATORY_EXIT_MAX_IMPACT_PCT,
    MANDATORY_EXIT_SAME_CALL_ATTEMPTS,
    MAX_ORDERBOOK_PRICE_IMPACT_PCT,
)
from backend.services.day_controlled_exits import ENGINE_RISK_EXIT_PREFIXES
from backend.services.day_trade_thesis import (
    EXIT_DAY_4H_STRUCTURE_BREAK,
    EXIT_DAY_RISK_FLOOR,
    EXIT_TRAILING_STOP,
)

logger = logging.getLogger(__name__)

STATUS_EXIT_RESIDUAL_PENDING = "EXIT_RESIDUAL_PENDING"

MANDATORY_FLATTEN_PREFIXES: tuple[str, ...] = (
    *ENGINE_RISK_EXIT_PREFIXES,
    EXIT_TRAILING_STOP,
    EXIT_DAY_4H_STRUCTURE_BREAK,
    EXIT_DAY_RISK_FLOOR,
    "TRAILING_STOP_EXIT",
    "DAY_4H_STRUCTURE_BREAK",
    "DAY_RISK_FLOOR",
    "FORCE_FLATTEN",
)

# First attempts stay at entry impact; later same-call attempts escalate to the
# absolute mandatory ceiling. Never above MANDATORY_EXIT_MAX_IMPACT_PCT.
MANDATORY_IMPACT_LADDER: tuple[float, ...] = (
    MAX_ORDERBOOK_PRICE_IMPACT_PCT,
    MAX_ORDERBOOK_PRICE_IMPACT_PCT,
    0.0025,
    0.005,
    MANDATORY_EXIT_MAX_IMPACT_PCT,
)

PreflightFn = Callable[[float, float], Awaitable[Any]]
PlaceIocFn = Callable[[float, float], Awaitable[dict[str, Any] | None]]
MeaningfulFn = Callable[[float], bool]


def is_mandatory_day_flatten(
    exit_trigger: str | None,
    *,
    force_sell: bool = False,
    exit_type_name: str | None = None,
) -> bool:
    """True for already-triggered full-flatten DAY exits. TP1 is not flatten."""
    trig = str(exit_trigger or "").strip().upper()
    et = str(exit_type_name or "").strip().upper()
    if "TP1" in trig or et == "TAKE_PROFIT_1":
        return False
    if "NET_PROFIT" in trig and "TRAILING" not in trig and "4H" not in trig:
        return False
    if any(trig.startswith(str(p).upper()) for p in MANDATORY_FLATTEN_PREFIXES):
        return True
    return bool(force_sell and et == "MANUAL")


def is_exit_residual_pending(position: Any) -> bool:
    return str(getattr(position, "status", "") or "") == STATUS_EXIT_RESIDUAL_PENDING


def is_meaningful_residual(
    qty: float,
    price: float,
    *,
    min_qty: float = 0.0,
    min_notional: float = 0.0,
    qty_step: float = 0.0,
) -> bool:
    """Executable leftover vs true dust. Uses exchange LOT_SIZE / MIN_NOTIONAL."""
    q = float(qty or 0.0)
    px = float(price or 0.0)
    if q <= 0 or px <= 0:
        return False
    stepped = q
    if qty_step > 0:
        stepped = int(q / qty_step + 1e-12) * qty_step
    if qty_step > 0 and stepped + 1e-15 < qty_step:
        return False
    if min_qty > 0 and stepped + 1e-15 < min_qty:
        return False
    return not (min_notional > 0 and stepped * px + 1e-15 < min_notional)


def mark_exit_residual_pending(position: Any, reason: str) -> None:
    position.status = STATUS_EXIT_RESIDUAL_PENDING
    position.exit_residual_reason = str(reason or "")
    import time

    position.exit_residual_since = float(time.time())


def clear_exit_residual_pending(position: Any) -> None:
    if str(getattr(position, "status", "") or "") == STATUS_EXIT_RESIDUAL_PENDING:
        position.status = "ACTIVE"
    position.exit_residual_reason = ""
    position.exit_residual_since = 0.0


def impact_for_attempt(attempt_index: int) -> float:
    attempt_index = max(attempt_index, 0)
    if attempt_index >= len(MANDATORY_IMPACT_LADDER):
        return float(MANDATORY_IMPACT_LADDER[-1])
    return float(MANDATORY_IMPACT_LADDER[attempt_index])


@dataclass
class MandatoryFlattenResult:
    filled_qty: float = 0.0
    average_price: float = 0.0
    remaining_qty: float = 0.0
    attempts: int = 0
    orders: list[dict[str, Any]] = field(default_factory=list)
    combined_order: dict[str, Any] | None = None
    abandoned_reason: str = ""

    @property
    def any_fill(self) -> bool:
        return self.filled_qty > 0


def _combine_orders(orders: list[dict[str, Any]], requested: float) -> dict[str, Any] | None:
    fills: list[tuple[float, float]] = []
    for o in orders:
        fq = float(o.get("filled") or 0.0)
        px = float(o.get("average") or o.get("price") or 0.0)
        if fq > 0 and px > 0:
            fills.append((fq, px))
    if not fills:
        return None
    tot = sum(f[0] for f in fills)
    vwap = sum(f[0] * f[1] for f in fills) / tot
    last = dict(orders[-1])
    last["filled"] = tot
    last["average"] = vwap
    last["amount"] = float(requested)
    last["_mystic_mandatory_flatten_fills"] = len(fills)
    last["_mystic_partial_fill"] = tot + 1e-12 < float(requested)
    last["_mystic_ioc_incomplete"] = bool(last.get("_mystic_partial_fill"))
    return last


async def run_mandatory_exit_ioc_loop(
    *,
    quantity: float,
    preflight: PreflightFn,
    place_ioc: PlaceIocFn,
    is_meaningful: MeaningfulFn,
    max_attempts: int = MANDATORY_EXIT_SAME_CALL_ATTEMPTS,
) -> MandatoryFlattenResult:
    """Same-call bounded IOC flatten. Fresh book each attempt. No sleep."""
    remaining = float(quantity or 0.0)
    requested = remaining
    out = MandatoryFlattenResult(remaining_qty=remaining)
    if remaining <= 0:
        out.abandoned_reason = "zero_qty"
        return out

    for i in range(max(1, int(max_attempts))):
        if not is_meaningful(remaining):
            break
        impact = impact_for_attempt(i)
        out.attempts = i + 1
        pf = await preflight(remaining, impact)
        if pf is None or not bool(getattr(pf, "passed", False)):
            out.abandoned_reason = str(getattr(pf, "reject_reason", "") or "preflight_failed")
            logger.warning(
                "MANDATORY_EXIT_PREFLIGHT_HOLD attempt=%s reason=%s remaining=%.8f",
                i + 1,
                out.abandoned_reason,
                remaining,
            )
            break
        chunk = float(getattr(pf, "executable_qty", 0.0) or getattr(pf, "quantity", 0.0) or 0.0)
        chunk = min(chunk, remaining)
        limit = float(getattr(pf, "protected_limit_price", 0.0) or 0.0)
        if chunk <= 0 or limit <= 0:
            out.abandoned_reason = "no_executable_chunk"
            continue
        order = await place_ioc(chunk, limit)
        filled = float((order or {}).get("filled") or 0.0)
        if order is None or filled <= 0:
            logger.warning(
                "MANDATORY_EXIT_IOC_ZERO attempt=%s chunk=%.8f limit=%.8f remaining=%.8f",
                i + 1,
                chunk,
                limit,
                remaining,
            )
            continue
        if filled > remaining + 1e-12:
            filled = remaining
            order = dict(order)
            order["filled"] = filled
        out.orders.append(order)
        remaining -= filled
        out.filled_qty += filled
        out.remaining_qty = max(0.0, remaining)
        logger.warning(
            "MANDATORY_EXIT_IOC_FILL attempt=%s filled=%.8f remaining=%.8f limit=%.8f",
            i + 1,
            filled,
            remaining,
            limit,
        )

    out.combined_order = _combine_orders(out.orders, requested)
    if out.combined_order:
        out.average_price = float(out.combined_order.get("average") or 0.0)
    if out.remaining_qty > 0 and is_meaningful(out.remaining_qty) and not out.abandoned_reason:
        out.abandoned_reason = "residual_after_same_call"
    return out
