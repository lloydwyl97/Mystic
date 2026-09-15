"""Exact-money spendable cash and downward quantity sizing for DAY buys.

Reservations belonging to other intents are subtracted once. The submitting
intent's own reservation is spendable and must not be counted as both
unavailable cash and required cash. Quantity is floored to the exchange step
after commission (and any already-applied execution allowance) are included.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import ROUND_FLOOR, Decimal
from typing import Any

INSUFFICIENT_EXECUTABLE_CASH_AFTER_QUANTIZATION = "INSUFFICIENT_EXECUTABLE_CASH_AFTER_QUANTIZATION"
INSUFFICIENT_CASH = "INSUFFICIENT_CASH"


def money(value: object) -> Decimal:
    if isinstance(value, Decimal):
        return value
    if value is None:
        return Decimal("0")
    if isinstance(value, bool):
        return Decimal(int(value))
    if isinstance(value, int):
        return Decimal(value)
    if isinstance(value, float):
        return Decimal(str(value))
    text = str(value).strip()
    if not text:
        return Decimal("0")
    return Decimal(text)


def floor_to_step(qty: Decimal, step: Decimal) -> Decimal:
    quantity = money(qty)
    increment = money(step)
    if increment <= 0:
        return quantity
    if quantity <= 0:
        return Decimal("0")
    return (quantity / increment).to_integral_value(rounding=ROUND_FLOOR) * increment


def cash_covers(required: Decimal, spendable: Decimal) -> bool:
    return money(required) <= money(spendable)


def is_terminal_buy_cash_reason(reason: str) -> bool:
    text = str(reason or "").strip().upper()
    if text.startswith(INSUFFICIENT_EXECUTABLE_CASH_AFTER_QUANTIZATION):
        return True
    return text.startswith(INSUFFICIENT_CASH)


def spendable_quote(
    *,
    account_cash: object,
    other_reservations: object = 0,
    open_order_commitment: object = 0,
    own_reservation: object = 0,
    include_own_reservation: bool = True,
) -> Decimal:
    """Cash this intent may spend.

    ``other_reservations`` and ``open_order_commitment`` are subtracted once.
    ``own_reservation`` is available when ``include_own_reservation`` is true
    (the reservation converts into the order; it is not a second hold).
    """
    available = money(account_cash) - money(other_reservations) - money(open_order_commitment)
    if include_own_reservation:
        return available
    return available - money(own_reservation)


def unit_gross_cost(price: object, commission_rate: object, slippage_rate: object = 0) -> Decimal:
    fill = money(price) * (Decimal("1") + money(slippage_rate))
    return fill * (Decimal("1") + money(commission_rate))


def gross_cost(
    quantity: object,
    price: object,
    commission_rate: object,
    slippage_rate: object = 0,
) -> tuple[Decimal, Decimal, Decimal]:
    qty = money(quantity)
    fill = money(price) * (Decimal("1") + money(slippage_rate))
    notional = qty * fill
    fee = notional * money(commission_rate)
    return notional, fee, notional + fee


@dataclass(frozen=True)
class BuyCashPlan:
    ok: bool
    reason: str
    quantity: Decimal
    fill_price: Decimal
    commission: Decimal
    total_cost: Decimal
    notional: Decimal
    spendable: Decimal
    requested_quantity: Decimal
    shrunk: bool

    def as_floats(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "reason": self.reason,
            "quantity": float(self.quantity),
            "fill_price": float(self.fill_price),
            "commission": float(self.commission),
            "total_cost": float(self.total_cost),
            "notional": float(self.notional),
            "spendable": float(self.spendable),
            "requested_quantity": float(self.requested_quantity),
            "shrunk": self.shrunk,
        }


def _terminal(reason: str, *, requested: Decimal, spendable: Decimal, fill_price: Decimal) -> BuyCashPlan:
    return BuyCashPlan(
        ok=False,
        reason=reason,
        quantity=Decimal("0"),
        fill_price=fill_price,
        commission=Decimal("0"),
        total_cost=Decimal("0"),
        notional=Decimal("0"),
        spendable=spendable,
        requested_quantity=requested,
        shrunk=False,
    )


def plan_executable_buy(
    *,
    requested_qty: object,
    price: object,
    commission_rate: object,
    spendable: object,
    qty_step: object,
    min_qty: object,
    min_notional: object,
    slippage_rate: object = 0,
    allocation: object | None = None,
) -> BuyCashPlan:
    """Largest valid quantity that fits exact spendable cash and allocation."""
    requested = money(requested_qty)
    px = money(price)
    budget = money(spendable)
    if allocation is not None:
        allocated = money(allocation)
        if allocated > 0:
            budget = min(budget, allocated)
    step = money(qty_step)
    minimum_qty = money(min_qty)
    minimum_notional = money(min_notional)
    if px <= 0:
        return _terminal("INVALID_PRICE", requested=requested, spendable=budget, fill_price=px)
    if budget <= 0:
        return _terminal(
            INSUFFICIENT_EXECUTABLE_CASH_AFTER_QUANTIZATION,
            requested=requested,
            spendable=budget,
            fill_price=px,
        )
    unit = unit_gross_cost(px, commission_rate, slippage_rate)
    if unit <= 0:
        return _terminal("INVALID_PRICE", requested=requested, spendable=budget, fill_price=px)
    max_qty = floor_to_step(budget / unit, step if step > 0 else Decimal("0"))
    qty = floor_to_step(requested, step if step > 0 else Decimal("0"))
    if qty <= 0:
        qty = max_qty
    qty = min(qty, max_qty)
    while qty > 0:
        notional, fee, total = gross_cost(qty, px, commission_rate, slippage_rate)
        if total > budget:
            if step <= 0:
                break
            qty = floor_to_step(qty - step, step)
            continue
        if minimum_qty > 0 and qty < minimum_qty:
            return _terminal(
                INSUFFICIENT_EXECUTABLE_CASH_AFTER_QUANTIZATION,
                requested=requested,
                spendable=budget,
                fill_price=px * (Decimal("1") + money(slippage_rate)),
            )
        if minimum_notional > 0 and notional < minimum_notional:
            return _terminal(
                INSUFFICIENT_EXECUTABLE_CASH_AFTER_QUANTIZATION,
                requested=requested,
                spendable=budget,
                fill_price=px * (Decimal("1") + money(slippage_rate)),
            )
        fill = px * (Decimal("1") + money(slippage_rate))
        return BuyCashPlan(
            ok=True,
            reason="OK",
            quantity=qty,
            fill_price=fill,
            commission=fee,
            total_cost=total,
            notional=notional,
            spendable=budget,
            requested_quantity=requested,
            shrunk=qty < requested,
        )
    return _terminal(
        INSUFFICIENT_EXECUTABLE_CASH_AFTER_QUANTIZATION,
        requested=requested,
        spendable=budget,
        fill_price=px * (Decimal("1") + money(slippage_rate)),
    )
