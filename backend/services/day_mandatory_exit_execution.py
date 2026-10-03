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
from backend.services.protected_limit_execution import PREFLIGHT_AUDIT_KEY, PREFLIGHT_CHUNKS_KEY, stamp_preflight

logger = logging.getLogger(__name__)

STATUS_EXIT_RESIDUAL_PENDING = "EXIT_RESIDUAL_PENDING"

MANDATORY_FLATTEN_PREFIXES: tuple[str, ...] = (
    *ENGINE_RISK_EXIT_PREFIXES,
    EXIT_TRAILING_STOP,
    # EXIT_DAY_4H_STRUCTURE_BREAK removed — 4H has no trading authority (2026-09-17)
    EXIT_DAY_RISK_FLOOR,
    "TRAILING_STOP_EXIT",
    # "DAY_4H_STRUCTURE_BREAK" removed — 4H has no trading authority (2026-09-17)
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
    # DAY_V2/SCALP_V2 exits keep their own trigger (e.g. SCALP_V2_NET_PROFIT);
    # a forced strategy exit is a full flatten whatever the trigger says.
    if force_sell and et == "STRATEGY":
        return True
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
    # The resume sell must carry the strategy trigger (DAY_V2_* / SCALP_V2_*),
    # not the display label it collapses to (MANUAL_EXIT).
    raw = str(getattr(position, "_learning_raw_exit_reason", "") or "")
    if raw.upper().startswith(("DAY_V2_", "SCALP_V2_")):
        reason = raw
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


def _f(raw: Any) -> float:
    try:
        return float(raw or 0.0)
    except (TypeError, ValueError):
        return 0.0


def _order_fee_pairs(order: dict[str, Any]) -> list[tuple[float, str]]:
    fee = order.get("fee")
    if isinstance(fee, dict) and _f(fee.get("cost")):
        return [(_f(fee.get("cost")), str(fee.get("currency") or "").upper())]
    out: list[tuple[float, str]] = []
    for item in order.get("fees") or []:
        if isinstance(item, dict) and _f(item.get("cost")):
            out.append((_f(item.get("cost")), str(item.get("currency") or "").upper()))
    return out


def _combine_orders(orders: list[dict[str, Any]], requested: float) -> dict[str, Any] | None:
    """One venue view of a multi-order close: every chunk's fills, cost and fees.

    Economics downstream read info.fills first, so each chunk's fills must be
    present there; a chunk whose reply omits fills contributes one synthetic
    fill built from its own order-level filled/average/fee.
    """
    used: list[dict[str, Any]] = []
    for o in orders:
        if _f(o.get("filled")) > 0 and _f(o.get("average") or o.get("price")) > 0:
            used.append(o)
    if not used:
        return None
    tot = sum(_f(o.get("filled")) for o in used)
    notional = sum(_f(o.get("filled")) * _f(o.get("average") or o.get("price")) for o in used)
    vwap = notional / tot
    cost = 0.0
    order_ids: list[str] = []
    fills: list[dict[str, Any]] = []
    trades: list[dict[str, Any]] = []
    fee_by_ccy: dict[str, float] = {}
    for o in used:
        fq = _f(o.get("filled"))
        px = _f(o.get("average") or o.get("price"))
        cost += _f(o.get("cost")) or fq * px
        oid = str(o.get("id") or (o.get("info") or {}).get("orderId") or "").strip()
        if oid and oid not in order_ids:
            order_ids.append(oid)
        info = o.get("info") if isinstance(o.get("info"), dict) else {}
        chunk_fills = [dict(f) for f in (info.get("fills") or []) if isinstance(f, dict)]
        if chunk_fills:
            for f in chunk_fills:
                f.setdefault("orderId", oid)
                fills.append(f)
                ccy = str(f.get("commissionAsset") or "").upper()
                fee_by_ccy[ccy] = fee_by_ccy.get(ccy, 0.0) + _f(f.get("commission"))
        else:
            pairs = _order_fee_pairs(o)
            for amt, ccy in pairs:
                fee_by_ccy[ccy] = fee_by_ccy.get(ccy, 0.0) + amt
            fills.append(
                {
                    "price": str(px),
                    "qty": str(fq),
                    "commission": str(sum(a for a, _c in pairs)),
                    "commissionAsset": pairs[0][1] if pairs else "",
                    "orderId": oid,
                    "_mystic_synthetic": True,
                }
            )
        for t in o.get("trades") or []:
            if isinstance(t, dict):
                trades.append(dict(t))
    last = dict(used[-1])
    info = dict(last.get("info") or {}) if isinstance(last.get("info"), dict) else {}
    info["fills"] = fills
    info["executedQty"] = str(tot)
    info["cummulativeQuoteQty"] = str(cost)
    last["info"] = info
    last["trades"] = trades
    last["filled"] = tot
    last["average"] = vwap
    last["cost"] = cost
    fees = [{"cost": amt, "currency": ccy} for ccy, amt in fee_by_ccy.items() if amt]
    last["fees"] = fees
    last["fee"] = dict(fees[0]) if len(fees) == 1 else None
    last["amount"] = float(requested)
    chunk_preflights = [dict(o[PREFLIGHT_AUDIT_KEY]) for o in used if isinstance(o.get(PREFLIGHT_AUDIT_KEY), dict)]
    if chunk_preflights:
        last[PREFLIGHT_CHUNKS_KEY] = chunk_preflights
    last["_mystic_order_ids"] = order_ids
    last["_mystic_mandatory_flatten_fills"] = len(used)
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
        order = stamp_preflight(await place_ioc(chunk, limit), pf)
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
