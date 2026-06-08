"""Order book walk helpers for Binance.US scalp preflight."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class BookWalkResult:
    depth_sufficient: bool
    expected_avg_fill: float
    last_fill_price: float
    impact_pct: float
    levels_consumed: int
    filled_notional_usd: float
    filled_qty: float


def walk_book(
    levels: list[list[float]], qty_needed: float
) -> tuple[float, float, bool, int]:
    remaining = float(qty_needed)
    cost = 0.0
    filled = 0.0
    levels_used = 0
    last_price = 0.0
    for level in levels:
        if remaining <= 1e-15:
            break
        if not level or len(level) < 2:
            continue
        px = float(level[0])
        q = float(level[1])
        if px <= 0 or q <= 0:
            continue
        take = min(remaining, q)
        cost += take * px
        filled += take
        remaining -= take
        last_price = px
        levels_used += 1
    if filled <= 0:
        return 0.0, 0.0, False, 0
    avg = cost / filled
    fully = remaining <= max(1e-12, qty_needed * 1e-9)
    return avg, last_price, fully, levels_used


def walk_buy_notional(
    asks: list[list[float]], notional_usd: float, best_ask: float
) -> BookWalkResult:
    if notional_usd <= 0 or not asks or best_ask <= 0:
        return BookWalkResult(False, 0.0, 0.0, 1.0, 0, 0.0, 0.0)
    remaining = notional_usd
    total_cost = 0.0
    total_qty = 0.0
    last_price = best_ask
    levels = 0
    for price, size in asks:
        if remaining <= 0:
            break
        level_usd = price * size
        take_usd = min(remaining, level_usd)
        take_qty = take_usd / price
        total_cost += take_usd
        total_qty += take_qty
        remaining -= take_usd
        last_price = price
        levels += 1
        if remaining <= 1e-12:
            break
    if remaining > 1e-8 or total_qty <= 0:
        return BookWalkResult(
            False, 0.0, last_price, 1.0, levels, total_cost, total_qty
        )
    avg = total_cost / total_qty
    impact = abs(avg - best_ask) / best_ask if best_ask > 0 else 1.0
    return BookWalkResult(
        True, avg, last_price, impact, levels, total_cost, total_qty
    )


def walk_sell_qty(
    bids: list[list[float]], qty: float, best_bid: float
) -> BookWalkResult:
    if qty <= 0 or not bids or best_bid <= 0:
        return BookWalkResult(False, 0.0, 0.0, 1.0, 0, 0.0, 0.0)
    avg, last_price, fully, levels = walk_book(bids, qty)
    if not fully or avg <= 0:
        return BookWalkResult(
            False, avg, last_price, 1.0, levels, avg * qty if avg else 0.0, qty
        )
    impact = abs(best_bid - avg) / best_bid if best_bid > 0 else 1.0
    return BookWalkResult(
        True, avg, last_price, impact, levels, avg * qty, qty
    )
