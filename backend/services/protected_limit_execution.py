"""
Protected limit execution: order-book preflight + protected limit pricing.

Shared by paper simulation and live-capable paths. No market-order fallback.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import time
from dataclasses import asdict, dataclass, field
from typing import Any

from backend.config.protected_execution import (
    DEPTH_INSUFFICIENT,
    EXECUTABLE_NET_PROFIT_BELOW_FLOOR,
    MAX_ORDERBOOK_PRICE_IMPACT_PCT,
    MAX_ORDERBOOK_SPREAD_PCT,
    ORDERBOOK_DEPTH_LIMIT,
    ORDERBOOK_MAX_AGE_SEC,
    ORDERBOOK_MISSING,
    ORDERBOOK_STALE,
    PRICE_IMPACT_TOO_HIGH,
    PROTECTED_FILL_NOT_PROFITABLE,
    PROTECTED_LIMIT_ALLOW_PARTIAL,
    PROTECTED_LIMIT_ORDER_TIMEOUT_SEC,
    SPREAD_TOO_WIDE,
    USE_PROTECTED_LIMIT_EXECUTION,
    effective_max_orderbook_spread_pct,
)
from backend.config.trading_economics import ESTIMATED_ROUNDTRIP_COST, MIN_NET_PROFIT_TO_SELL, TAKER_FEE
from backend.utils.symbols import normalize_symbol

logger = logging.getLogger(__name__)

# Last preflight telemetry (engine/API reads via get_last_execution_protection_state)
_last_state: dict[str, Any] = {
    "last_preflight_passed": None,
    "last_preflight_reject_reason": "",
    "last_symbol": "",
    "last_side": "",
    "orderbook_best_bid": None,
    "orderbook_best_ask": None,
    "spread_pct": None,
    "last_expected_avg_fill": None,
    "last_protected_limit_price": None,
    "last_price_impact_pct": None,
    "last_execution_mode": "",
    "updated_at": None,
}


@dataclass
class ProtectedPreflightResult:
    passed: bool
    reject_reason: str = ""
    symbol: str = ""
    side: str = ""
    best_bid: float = 0.0
    best_ask: float = 0.0
    spread_pct: float = 0.0
    expected_avg_fill: float = 0.0
    protected_limit_price: float = 0.0
    price_impact_pct: float = 0.0
    reference_price: float = 0.0
    quantity: float = 0.0
    execution_mode: str = ""
    book_age_sec: float = 0.0
    diagnostics: dict[str, Any] = field(default_factory=dict)

    def to_audit_dict(self) -> dict[str, Any]:
        return {
            "orderbook_best_bid": self.best_bid,
            "orderbook_best_ask": self.best_ask,
            "spread_pct": self.spread_pct,
            "expected_avg_fill": self.expected_avg_fill,
            "protected_limit_price": self.protected_limit_price,
            "price_impact_pct": self.price_impact_pct,
            "execution_mode": self.execution_mode,
            "book_age_sec": self.book_age_sec,
            "reject_reason": self.reject_reason,
        }


def get_last_execution_protection_state(*, taker_fee: float | None = None) -> dict[str, Any]:
    from backend.config.protected_execution import get_protected_execution_snapshot

    tf = float(taker_fee if taker_fee is not None else TAKER_FEE)
    snap = get_protected_execution_snapshot(taker_fee=tf)
    out = dict(_last_state)
    out.update(
        {
            "maker_fee": snap.maker_fee,
            "taker_fee": snap.taker_fee,
            "use_protected_limit_execution": snap.use_protected_limit_execution,
            "max_orderbook_spread_pct": snap.max_orderbook_spread_pct,
            "max_orderbook_price_impact_pct": snap.max_orderbook_price_impact_pct,
            "protected_limit_order_timeout_sec": snap.protected_limit_order_timeout_sec,
            "protected_limit_allow_partial": snap.protected_limit_allow_partial,
        }
    )
    return out


def _update_last_state(result: ProtectedPreflightResult) -> None:
    _last_state.update(
        {
            "last_preflight_passed": bool(result.passed),
            "last_preflight_reject_reason": result.reject_reason or "",
            "last_symbol": result.symbol,
            "last_side": result.side,
            "orderbook_best_bid": result.best_bid,
            "orderbook_best_ask": result.best_ask,
            "spread_pct": result.spread_pct,
            "last_expected_avg_fill": result.expected_avg_fill,
            "last_protected_limit_price": result.protected_limit_price,
            "last_price_impact_pct": result.price_impact_pct,
            "last_execution_mode": result.execution_mode,
            "updated_at": time.time(),
        }
    )


def _walk_book(levels: list[list[float]], qty_needed: float) -> tuple[float, float, bool]:
    remaining = float(qty_needed)
    cost = 0.0
    filled = 0.0
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
    if filled <= 0:
        return 0.0, 0.0, False
    avg = cost / filled
    fully = remaining <= max(1e-12, qty_needed * 1e-9)
    return avg, filled, fully


@dataclass
class ExecutableSellProfitCheck:
    passed: bool
    reject_reason: str = ""
    executable_sell_price: float = 0.0
    executable_gross_pct: float = 0.0
    executable_net_pct: float = 0.0
    executable_net_profit_usd: float = 0.0
    entry_price: float = 0.0
    quantity: float = 0.0
    mark_price: float = 0.0

    def to_audit_dict(self) -> dict[str, Any]:
        return {
            "executable_sell_price": self.executable_sell_price,
            "executable_gross_pct": self.executable_gross_pct,
            "executable_net_pct": self.executable_net_pct,
            "executable_net_profit_usd": self.executable_net_profit_usd,
            "reject_reason": self.reject_reason,
        }


def evaluate_executable_sell_profit(
    *,
    entry_price: float,
    quantity: float,
    executable_sell_price: float,
    entry_fee: float = 0.0,
    sell_fee_rate: float = 0.0,
    position_qty: float | None = None,
    mark_price: float | None = None,
) -> ExecutableSellProfitCheck:
    """
    Final gate before SELL commit: profit must clear using executable fill price
    from protected preflight (or live fill), not monitor mark alone.
    """
    base = ExecutableSellProfitCheck(
        passed=False,
        executable_sell_price=float(executable_sell_price or 0.0),
        entry_price=float(entry_price or 0.0),
        quantity=float(quantity or 0.0),
        mark_price=float(mark_price if mark_price is not None else 0.0),
    )
    if entry_price <= 0 or quantity <= 0 or executable_sell_price <= 0:
        base.reject_reason = PROTECTED_FILL_NOT_PROFITABLE
        return base

    gross_pct = (executable_sell_price - entry_price) / entry_price
    net_pct = gross_pct - ESTIMATED_ROUNDTRIP_COST
    pos_qty = float(position_qty if position_qty is not None else quantity)
    entry_fee_pro_rata = float(entry_fee or 0.0) * (quantity / pos_qty) if pos_qty > 0 else 0.0
    entry_cost = (quantity * entry_price) + entry_fee_pro_rata
    fee = quantity * executable_sell_price * float(sell_fee_rate or 0.0)
    proceeds = (quantity * executable_sell_price) - fee
    net_profit_usd = proceeds - entry_cost

    base.executable_gross_pct = gross_pct
    base.executable_net_pct = net_pct
    base.executable_net_profit_usd = net_profit_usd

    if net_pct + 1e-12 < MIN_NET_PROFIT_TO_SELL:
        base.reject_reason = EXECUTABLE_NET_PROFIT_BELOW_FLOOR
        return base
    if net_profit_usd <= 0:
        base.reject_reason = PROTECTED_FILL_NOT_PROFITABLE
        return base

    base.passed = True
    return base


async def _fetch_order_book(ccxt_symbol: str) -> tuple[list[list[float]], list[list[float]], float] | None:
    """Fetch L2 via live_market_data service (existing path, depth limit capped)."""
    try:
        from backend.services.live_market_data import live_market_data_service

        if live_market_data_service is None:
            return None
        ob = await live_market_data_service.get_order_book(ccxt_symbol, limit=ORDERBOOK_DEPTH_LIMIT)
        bids = ob.get("bids") or []
        asks = ob.get("asks") or []
        if not bids or not asks:
            return None
        return bids, asks, 0.0
    except Exception as ex:
        logger.warning("PROTECTED_EXEC order book fetch failed %s: %s", ccxt_symbol, ex)
        return None


def _fail(
    symbol: str,
    side: str,
    reason: str,
    *,
    execution_mode: str,
    reference_price: float,
    quantity: float,
    **extra: Any,
) -> ProtectedPreflightResult:
    res = ProtectedPreflightResult(
        passed=False,
        reject_reason=reason,
        symbol=symbol,
        side=side,
        reference_price=reference_price,
        quantity=quantity,
        execution_mode=execution_mode,
        diagnostics=extra,
    )
    _update_last_state(res)
    logger.info(
        "PROTECTED_PREFLIGHT_REJECT %s %s reason=%s ref=%.8f qty=%.8f extra=%s",
        side,
        symbol,
        reason,
        reference_price,
        quantity,
        extra,
    )
    return res


async def run_protected_preflight(
    *,
    symbol: str,
    side: str,
    quantity: float,
    reference_price: float,
    live_capable: bool = False,
) -> ProtectedPreflightResult:
    """
    Order-book preflight for BUY or SELL. Returns passed=False on any reject — no fallback.
    """
    ns = normalize_symbol(symbol)
    side_u = str(side or "").strip().upper()
    exec_mode = "PROTECTED_LIMIT_LIVE" if live_capable else "PROTECTED_LIMIT_SIM"

    if not USE_PROTECTED_LIMIT_EXECUTION:
        res = ProtectedPreflightResult(
            passed=True,
            symbol=ns,
            side=side_u,
            expected_avg_fill=float(reference_price),
            protected_limit_price=float(reference_price),
            reference_price=float(reference_price),
            quantity=float(quantity),
            execution_mode="LEGACY_BUFFER",
        )
        _update_last_state(res)
        return res

    if quantity <= 0 or reference_price <= 0:
        return _fail(
            ns,
            side_u,
            ORDERBOOK_MISSING,
            execution_mode=exec_mode,
            reference_price=reference_price,
            quantity=quantity,
            detail="invalid_qty_or_price",
        )

    fetched = await _fetch_order_book(ns)
    if fetched is None:
        return _fail(
            ns,
            side_u,
            ORDERBOOK_MISSING,
            execution_mode=exec_mode,
            reference_price=reference_price,
            quantity=quantity,
        )

    bids, asks, book_age = fetched
    if book_age > ORDERBOOK_MAX_AGE_SEC:
        return _fail(
            ns,
            side_u,
            ORDERBOOK_STALE,
            execution_mode=exec_mode,
            reference_price=reference_price,
            quantity=quantity,
            book_age_sec=book_age,
        )

    best_bid = float(bids[0][0])
    best_ask = float(asks[0][0])
    if best_bid <= 0 or best_ask <= 0 or best_ask < best_bid:
        return _fail(
            ns,
            side_u,
            ORDERBOOK_MISSING,
            execution_mode=exec_mode,
            reference_price=reference_price,
            quantity=quantity,
            detail="invalid_top_of_book",
        )

    mid = (best_bid + best_ask) / 2.0
    spread_pct = (best_ask - best_bid) / mid if mid > 0 else 1.0
    max_spread = effective_max_orderbook_spread_pct(live_capable=live_capable)
    if spread_pct > max_spread + 1e-15:
        return _fail(
            ns,
            side_u,
            SPREAD_TOO_WIDE,
            execution_mode=exec_mode,
            reference_price=reference_price,
            quantity=quantity,
            spread_pct=spread_pct,
            max_spread=max_spread,
            best_bid=best_bid,
            best_ask=best_ask,
        )

    if side_u == "BUY":
        avg_fill, filled_qty, fully = _walk_book(asks, quantity)
        if avg_fill <= 0:
            return _fail(
                ns,
                side_u,
                DEPTH_INSUFFICIENT,
                execution_mode=exec_mode,
                reference_price=reference_price,
                quantity=quantity,
            )
        if not PROTECTED_LIMIT_ALLOW_PARTIAL and not fully:
            return _fail(
                ns,
                side_u,
                DEPTH_INSUFFICIENT,
                execution_mode=exec_mode,
                reference_price=reference_price,
                quantity=quantity,
                filled_qty=filled_qty,
                requested_qty=quantity,
            )
        price_impact = (avg_fill - best_ask) / best_ask if best_ask > 0 else 0.0
        if price_impact > MAX_ORDERBOOK_PRICE_IMPACT_PCT + 1e-15:
            return _fail(
                ns,
                side_u,
                PRICE_IMPACT_TOO_HIGH,
                execution_mode=exec_mode,
                reference_price=reference_price,
                quantity=quantity,
                price_impact_pct=price_impact,
                max_impact=MAX_ORDERBOOK_PRICE_IMPACT_PCT,
            )
        protected_limit = min(avg_fill, best_ask * (1.0 + MAX_ORDERBOOK_PRICE_IMPACT_PCT))
    elif side_u == "SELL":
        avg_fill, filled_qty, fully = _walk_book(bids, quantity)
        if avg_fill <= 0:
            return _fail(
                ns,
                side_u,
                DEPTH_INSUFFICIENT,
                execution_mode=exec_mode,
                reference_price=reference_price,
                quantity=quantity,
            )
        if not PROTECTED_LIMIT_ALLOW_PARTIAL and not fully:
            return _fail(
                ns,
                side_u,
                DEPTH_INSUFFICIENT,
                execution_mode=exec_mode,
                reference_price=reference_price,
                quantity=quantity,
                filled_qty=filled_qty,
                requested_qty=quantity,
            )
        price_impact = (best_bid - avg_fill) / best_bid if best_bid > 0 else 0.0
        if price_impact > MAX_ORDERBOOK_PRICE_IMPACT_PCT + 1e-15:
            return _fail(
                ns,
                side_u,
                PRICE_IMPACT_TOO_HIGH,
                execution_mode=exec_mode,
                reference_price=reference_price,
                quantity=quantity,
                price_impact_pct=price_impact,
                max_impact=MAX_ORDERBOOK_PRICE_IMPACT_PCT,
            )
        protected_limit = max(avg_fill, best_bid * (1.0 - MAX_ORDERBOOK_PRICE_IMPACT_PCT))
    else:
        return _fail(
            ns,
            side_u,
            ORDERBOOK_MISSING,
            execution_mode=exec_mode,
            reference_price=reference_price,
            quantity=quantity,
            detail=f"invalid_side={side_u}",
        )

    res = ProtectedPreflightResult(
        passed=True,
        symbol=ns,
        side=side_u,
        best_bid=best_bid,
        best_ask=best_ask,
        spread_pct=spread_pct,
        expected_avg_fill=avg_fill,
        protected_limit_price=protected_limit,
        price_impact_pct=price_impact,
        reference_price=float(reference_price),
        quantity=float(quantity),
        execution_mode=exec_mode,
        book_age_sec=book_age,
    )
    _update_last_state(res)
    logger.info(
        "PROTECTED_PREFLIGHT_PASS %s %s qty=%.8f avg=%.8f limit=%.8f impact=%.6f spread=%.6f mode=%s",
        side_u,
        ns,
        quantity,
        avg_fill,
        protected_limit,
        price_impact,
        spread_pct,
        exec_mode,
    )
    return res


async def _enrich_live_order_fills(live_service: Any, order: dict[str, Any], exchange_symbol: str) -> dict[str, Any]:
    """Re-fetch so commission/fills survive IOC expire. Never invent quantity."""
    order_id = str(order.get("id") or "")
    if not order_id:
        return order
    try:
        st = await live_service.fetch_order("binanceus", order_id, exchange_symbol)
    except Exception:
        return order
    if st.get("status") != "success":
        return order
    fetched = st.get("order") or {}
    merged = dict(order)
    for key in (
        "filled",
        "average",
        "cost",
        "status",
        "fee",
        "fees",
        "trades",
        "commission",
        "commissionAsset",
        "info",
        "amount",
        "price",
    ):
        if fetched.get(key) is not None:
            merged[key] = fetched[key]
    return merged


async def execute_protected_limit_live(
    live_service: Any,
    *,
    symbol: str,
    side: str,
    quantity: float,
    limit_price: float,
) -> dict[str, Any] | None:
    """
    Place protected limit on Binance.US with strict timeout; cancel if not fully filled.
    Returns order dict on full fill, None on failure. Never falls back to market.
    """
    from backend.utils.symbols import to_exchange_symbol

    exchange_symbol = to_exchange_symbol(symbol).replace("/", "")
    side_l = side.lower()
    timeout = float(PROTECTED_LIMIT_ORDER_TIMEOUT_SEC)
    poll_interval = 0.5

    # Try IOC first (full fill or cancel); fall back to GTC+timeout if unsupported.
    for tif in ("IOC", "GTC"):
        try:
            await live_service._ensure_initialized()
            params: dict[str, Any] = {}
            if tif == "IOC":
                params["timeInForce"] = "IOC"
            result = await live_service.place_order(
                exchange="binanceus",
                symbol=exchange_symbol,
                order_type="limit",
                side=side_l,
                amount=quantity,
                price=limit_price,
                time_in_force=tif if tif == "IOC" else None,
            )
        except TypeError:
            result = await live_service.place_order(
                exchange="binanceus",
                symbol=exchange_symbol,
                order_type="limit",
                side=side_l,
                amount=quantity,
                price=limit_price,
            )
        except Exception as ex:
            logger.warning("PROTECTED_LIMIT_LIVE place failed tif=%s %s: %s", tif, exchange_symbol, ex)
            continue

        if not result or result.get("status") != "success":
            if tif == "IOC":
                continue
            return None

        order = result.get("order") or {}
        order_id = str(order.get("id") or "")
        if order_id:
            order = await _enrich_live_order_fills(live_service, order, exchange_symbol)
        filled = float(order.get("filled") or 0.0)
        amount = float(order.get("amount") or quantity)

        if tif == "IOC":
            if filled + 1e-12 >= amount and filled > 0:
                return order
            logger.warning(
                "PROTECTED_LIMIT_IOC_INCOMPLETE %s %s filled=%.8f amount=%.8f — no market fallback",
                side_l,
                exchange_symbol,
                filled,
                amount,
            )
            if filled > 0:
                order["_mystic_partial_fill"] = True
                order["_mystic_ioc_incomplete"] = True
                return order
            return None

        deadline = time.time() + timeout
        last = order
        while time.time() < deadline:
            await asyncio.sleep(poll_interval)
            st = await live_service.fetch_order("binanceus", order_id, exchange_symbol)
            if st.get("status") != "success":
                continue
            o = st.get("order") or {}
            last = o
            filled = float(o.get("filled") or 0.0)
            amount = float(o.get("amount") or quantity)
            status = str(o.get("status") or "").lower()
            if filled + 1e-12 >= amount and filled > 0:
                return o
            if status in ("closed", "filled") and filled > 0:
                if PROTECTED_LIMIT_ALLOW_PARTIAL or filled + 1e-12 >= amount:
                    return o
                break
            if status in ("canceled", "cancelled", "expired", "rejected"):
                break

        if order_id:
            with contextlib.suppress(Exception):
                await live_service.cancel_order("binanceus", order_id, exchange_symbol)
            last = await _enrich_live_order_fills(live_service, last, exchange_symbol)
            filled = float(last.get("filled") or filled or 0.0)
        logger.warning(
            "PROTECTED_LIMIT_TIMEOUT_CANCEL %s %s order_id=%s filled=%.8f — no market fallback",
            side_l,
            exchange_symbol,
            order_id,
            filled,
        )
        if filled > 0:
            last["_mystic_partial_fill"] = True
            last["_mystic_ioc_incomplete"] = True
            return last
        return None

    return None
