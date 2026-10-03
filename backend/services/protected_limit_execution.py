"""
Protected limit execution: order-book preflight + protected limit pricing.

Shared by paper simulation and live-capable paths. No market-order fallback.
"""

from __future__ import annotations

import asyncio
import contextlib
import contextvars
import logging
import time
from dataclasses import asdict, dataclass, field
from typing import Any

from backend.config.execution_cost_model import honest_all_in_rt_pct
from backend.config.protected_execution import (
    DEPTH_INSUFFICIENT,
    EXECUTABLE_NET_PROFIT_BELOW_FLOOR,
    MANDATORY_EXIT_MAX_IMPACT_PCT,
    MANDATORY_EXIT_MAX_SPREAD_PCT,
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
from backend.config.trading_economics import (
    ESTIMATED_ROUNDTRIP_COST,
    MIN_NET_PROFIT_TO_SELL,
    TAKER_FEE,
    min_net_profit_for_symbol,
)
from backend.utils.symbols import normalize_symbol

logger = logging.getLogger(__name__)

_PREFLIGHT_FRESHNESS: contextvars.ContextVar[dict[str, Any] | None] = contextvars.ContextVar("_PREFLIGHT_FRESHNESS", default=None)

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
    book_age_sec: float | None = None
    executable_qty: float = 0.0
    book_freshness: dict[str, Any] = field(default_factory=dict)
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
            "book_freshness": dict(self.book_freshness),
        }


@dataclass
class ExecutableBook:
    """One REST depth snapshot plus the identity and timing that date it."""

    bids: list[list[float]]
    asks: list[list[float]]
    last_update_id: int | None
    source_ts: float | None
    receive_ts: float | None
    exchange_ts: float | None = None


@dataclass
class _AcceptedBook:
    update_id: int
    source_ts: float
    receive_ts: float


# Newest depth identity accepted per symbol in this process. An older id is
# rejected; a repeated id keeps the age of its first observation.
_ACCEPTED_BOOKS: dict[str, _AcceptedBook] = {}

FRESHNESS_FRESH = "FRESH"
FRESHNESS_STALE = "STALE"
FRESHNESS_OUT_OF_ORDER = "OUT_OF_ORDER"
FRESHNESS_MISSING_UPDATE_ID = "MISSING_UPDATE_ID"
FRESHNESS_MISSING_SOURCE_TS = "MISSING_SOURCE_TS"


def reset_book_identity() -> None:
    _ACCEPTED_BOOKS.clear()


def assess_book_freshness(
    symbol: str,
    book: ExecutableBook,
    *,
    now: float | None = None,
    max_age_sec: float | None = None,
) -> dict[str, Any]:
    """Age of the book update used for execution, not of local processing.

    Age runs from the exchange event time when present, else from the send
    time of the request that first returned this update id. A late response,
    a repeated id, or an older id cannot look fresher than the exchange book.
    """
    processing_ts = time.time() if now is None else float(now)
    cap = float(ORDERBOOK_MAX_AGE_SEC if max_age_sec is None else max_age_sec)
    sym = normalize_symbol(symbol)
    out: dict[str, Any] = {
        "source_book_timestamp": None,
        "exchange_book_timestamp": book.exchange_ts,
        "request_sent_timestamp": book.source_ts,
        "local_receive_timestamp": book.receive_ts,
        "processing_timestamp": processing_ts,
        "book_age_ms": None,
        "last_update_id": book.last_update_id,
        "accepted_update_id": None,
        "duplicate_update_id": False,
        "max_age_ms": cap * 1000.0,
        "freshness_result": FRESHNESS_STALE,
    }
    if book.last_update_id is None:
        out["freshness_result"] = FRESHNESS_MISSING_UPDATE_ID
        return out
    source_ts = book.exchange_ts if book.exchange_ts and book.exchange_ts > 0 else book.source_ts
    if source_ts is None or source_ts <= 0:
        out["freshness_result"] = FRESHNESS_MISSING_SOURCE_TS
        return out
    update_id = int(book.last_update_id)
    prev = _ACCEPTED_BOOKS.get(sym)
    if prev is not None and update_id < prev.update_id:
        out["accepted_update_id"] = prev.update_id
        out["source_book_timestamp"] = float(source_ts)
        out["book_age_ms"] = (processing_ts - float(source_ts)) * 1000.0
        out["freshness_result"] = FRESHNESS_OUT_OF_ORDER
        return out
    if prev is not None and update_id == prev.update_id:
        out["duplicate_update_id"] = True
        source_ts = min(float(source_ts), prev.source_ts)
    else:
        _ACCEPTED_BOOKS[sym] = _AcceptedBook(
            update_id=update_id,
            source_ts=float(source_ts),
            receive_ts=float(book.receive_ts or processing_ts),
        )
    out["accepted_update_id"] = update_id
    out["source_book_timestamp"] = float(source_ts)
    age_sec = processing_ts - float(source_ts)
    out["book_age_ms"] = age_sec * 1000.0
    if age_sec < 0 or age_sec > cap:
        out["freshness_result"] = FRESHNESS_STALE
        return out
    out["freshness_result"] = FRESHNESS_FRESH
    return out


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


def walk_book_within_impact(
    levels: list[list[float]],
    qty_needed: float,
    *,
    best_px: float,
    max_impact_pct: float,
    sell: bool,
) -> tuple[float, float, bool]:
    """Walk book but stop before avg fill exceeds max_impact vs top of book."""
    remaining = float(qty_needed)
    cost = 0.0
    filled = 0.0
    cap = float(max_impact_pct)
    top = float(best_px)
    if remaining <= 0 or top <= 0:
        return 0.0, 0.0, False
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
        new_filled = filled + take
        new_avg = (cost + take * px) / new_filled
        if sell:
            impact = (top - new_avg) / top
        else:
            impact = (new_avg - top) / top
        if impact > cap + 1e-15:
            if filled <= 0:
                # First level is always 0 impact vs itself; take it even if later
                # levels would breach. Never skip a real top-of-book bid/ask.
                take = min(remaining, q)
                cost += take * px
                filled += take
                remaining -= take
            break
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
    symbol: str = "",
) -> ExecutableSellProfitCheck:
    """
    Final gate before SELL commit: profit must clear using executable fill price
    from protected preflight (or live fill), not monitor mark alone.

    ``symbol`` selects the same per-coin floor and honest round-trip cost the
    exit decision used. Without it this re-tested the exit against the global
    MIN_NET_PROFIT_TO_SELL, so an exit authorized by a lower per-coin floor was
    rejected here by an unrelated threshold it never had to clear.
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

    floor = min_net_profit_for_symbol(symbol) if symbol else MIN_NET_PROFIT_TO_SELL
    rt_cost = honest_all_in_rt_pct(symbol) if symbol else ESTIMATED_ROUNDTRIP_COST
    gross_pct = (executable_sell_price - entry_price) / entry_price
    net_pct = gross_pct - rt_cost
    pos_qty = float(position_qty if position_qty is not None else quantity)
    entry_fee_pro_rata = float(entry_fee or 0.0) * (quantity / pos_qty) if pos_qty > 0 else 0.0
    entry_cost = (quantity * entry_price) + entry_fee_pro_rata
    fee = quantity * executable_sell_price * float(sell_fee_rate or 0.0)
    proceeds = (quantity * executable_sell_price) - fee
    net_profit_usd = proceeds - entry_cost

    base.executable_gross_pct = gross_pct
    base.executable_net_pct = net_pct
    base.executable_net_profit_usd = net_profit_usd

    if net_pct + 1e-12 < floor:
        base.reject_reason = EXECUTABLE_NET_PROFIT_BELOW_FLOOR
        return base
    if net_profit_usd <= 0:
        base.reject_reason = PROTECTED_FILL_NOT_PROFITABLE
        return base

    base.passed = True
    return base


async def _fetch_order_book(ccxt_symbol: str) -> ExecutableBook | None:
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
        return ExecutableBook(
            bids=bids,
            asks=asks,
            last_update_id=ob.get("last_update_id"),
            source_ts=ob.get("request_sent_ts"),
            receive_ts=ob.get("received_ts"),
            exchange_ts=ob.get("exchange_ts"),
        )
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
    freshness = dict(_PREFLIGHT_FRESHNESS.get() or {})
    res = ProtectedPreflightResult(
        passed=False,
        reject_reason=reason,
        symbol=symbol,
        side=side,
        reference_price=reference_price,
        quantity=quantity,
        execution_mode=execution_mode,
        book_age_sec=(freshness["book_age_ms"] / 1000.0) if freshness.get("book_age_ms") is not None else None,
        book_freshness=freshness,
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
    mandatory_exit: bool = False,
    allow_chunk: bool = False,
    max_impact_pct: float | None = None,
) -> ProtectedPreflightResult:
    """
    Order-book preflight for BUY or SELL.

    Entry / discretionary SELL: reject on spread, impact, or incomplete depth.

    Mandatory DAY flatten (``mandatory_exit=True``): spread/impact size the
    executable chunk and limit. They must not refuse liquidation unless the
    book is missing/stale or the absolute catastrophic bound is breached.
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

    _PREFLIGHT_FRESHNESS.set({})
    fetched = await _fetch_order_book(ns)
    if fetched is None or not fetched.bids or not fetched.asks:
        return _fail(
            ns,
            side_u,
            ORDERBOOK_MISSING,
            execution_mode=exec_mode,
            reference_price=reference_price,
            quantity=quantity,
        )

    bids, asks = fetched.bids, fetched.asks
    freshness = assess_book_freshness(ns, fetched)
    try:
        best_bid = float(bids[0][0])
        best_ask = float(asks[0][0])
    except (TypeError, ValueError, IndexError):
        best_bid = best_ask = 0.0
    freshness["best_bid"] = best_bid
    freshness["best_ask"] = best_ask
    freshness["spread"] = best_ask - best_bid
    _PREFLIGHT_FRESHNESS.set(freshness)
    book_age = freshness["book_age_ms"] / 1000.0 if freshness.get("book_age_ms") is not None else None
    if freshness["freshness_result"] != FRESHNESS_FRESH:
        return _fail(
            ns,
            side_u,
            ORDERBOOK_STALE,
            execution_mode=exec_mode,
            reference_price=reference_price,
            quantity=quantity,
            book_age_sec=book_age,
            freshness_result=freshness["freshness_result"],
        )

    if best_bid <= 0 or best_ask <= 0 or best_ask <= best_bid:
        freshness["freshness_result"] = "INVALID_TOP_OF_BOOK"
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
    flatten = bool(mandatory_exit) and side_u == "SELL"
    max_spread = float(MANDATORY_EXIT_MAX_SPREAD_PCT) if flatten else effective_max_orderbook_spread_pct(live_capable=live_capable)
    impact_cap = float(max_impact_pct) if max_impact_pct is not None else float(MAX_ORDERBOOK_PRICE_IMPACT_PCT)
    if flatten:
        impact_cap = min(impact_cap, float(MANDATORY_EXIT_MAX_IMPACT_PCT))
    chunk_ok = bool(flatten and allow_chunk)

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
            mandatory_exit=flatten,
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
        executable_qty = float(quantity)
    elif side_u == "SELL":
        if chunk_ok:
            avg_fill, filled_qty, fully = walk_book_within_impact(
                bids,
                quantity,
                best_px=best_bid,
                max_impact_pct=impact_cap,
                sell=True,
            )
        else:
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
        if not flatten and not PROTECTED_LIMIT_ALLOW_PARTIAL and not fully:
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
        if not flatten and price_impact > MAX_ORDERBOOK_PRICE_IMPACT_PCT + 1e-15:
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
        if flatten and not chunk_ok and price_impact > impact_cap + 1e-15:
            # Flatten without chunk: still send at best bid for whatever the
            # impact cap allows via a forced chunk of top-of-book.
            avg_fill, filled_qty, fully = walk_book_within_impact(
                bids,
                quantity,
                best_px=best_bid,
                max_impact_pct=impact_cap,
                sell=True,
            )
            if avg_fill <= 0 or filled_qty <= 0:
                return _fail(
                    ns,
                    side_u,
                    PRICE_IMPACT_TOO_HIGH,
                    execution_mode=exec_mode,
                    reference_price=reference_price,
                    quantity=quantity,
                    price_impact_pct=price_impact,
                    max_impact=impact_cap,
                    mandatory_exit=True,
                )
            price_impact = (best_bid - avg_fill) / best_bid if best_bid > 0 else 0.0
        executable_qty = float(filled_qty if (flatten or chunk_ok) else quantity)
        if flatten and executable_qty <= 0:
            return _fail(
                ns,
                side_u,
                DEPTH_INSUFFICIENT,
                execution_mode=exec_mode,
                reference_price=reference_price,
                quantity=quantity,
                mandatory_exit=True,
            )
        protected_limit = max(avg_fill, best_bid * (1.0 - impact_cap))
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
        quantity=float(executable_qty if side_u == "SELL" and flatten else quantity),
        execution_mode=exec_mode,
        book_age_sec=book_age,
        book_freshness=dict(freshness),
        executable_qty=float(executable_qty if side_u == "SELL" else quantity),
        diagnostics={"mandatory_exit": flatten, "allow_chunk": chunk_ok, "impact_cap": impact_cap},
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
        "commission",
        "commissionAsset",
        "amount",
        "price",
    ):
        if fetched.get(key) is not None:
            merged[key] = fetched[key]
    # "trades" and info["fills"] are the only carriers of the per-fill venue trade
    # ids, and only the create-order response has them: Binance's GET /order reply
    # omits fills entirely and CCXT leaves trades empty for it. Overwriting them
    # with the fetched (empty) values is why every stored live fill had
    # fill_ids_json=[] and venue_trade_ids_json=[]. Take the fetched value only
    # when it actually carries something.
    if fetched.get("trades"):
        merged["trades"] = fetched["trades"]
    merged["info"] = _merge_info_preserving_fills(order.get("info"), fetched.get("info"))
    return merged


def _merge_info_preserving_fills(original: Any, fetched: Any) -> Any:
    """Overlay a fetched ``info`` without dropping the create response's fills."""
    if not isinstance(fetched, dict):
        return original
    if not isinstance(original, dict):
        return fetched
    merged = dict(original)
    merged.update({k: v for k, v in fetched.items() if v is not None})
    if original.get("fills") and not fetched.get("fills"):
        merged["fills"] = original["fills"]
    return merged


def _ms_to_sec(raw: Any) -> float | None:
    try:
        v = float(raw)
    except (TypeError, ValueError):
        return None
    return v / 1000.0 if v > 0 else None


PREFLIGHT_AUDIT_KEY = "_mystic_preflight"
PREFLIGHT_CHUNKS_KEY = "_mystic_preflight_chunks"
ORDER_AUDIT_KEYS = ("_mystic_latency", PREFLIGHT_AUDIT_KEY, PREFLIGHT_CHUNKS_KEY)


def stamp_preflight(order: dict[str, Any] | None, preflight: Any) -> dict[str, Any] | None:
    """Attach the preflight audit (book freshness included) that priced this order."""
    if isinstance(order, dict) and hasattr(preflight, "to_audit_dict"):
        order[PREFLIGHT_AUDIT_KEY] = preflight.to_audit_dict()
    return order


def preflight_audit_fields(order: dict[str, Any] | None) -> dict[str, Any]:
    """Sell-row audit fields from the preflight(s) stamped on a live order."""
    if not isinstance(order, dict):
        return {}
    out: dict[str, Any] = dict(order.get(PREFLIGHT_AUDIT_KEY) or {})
    chunks = [dict(c) for c in (order.get(PREFLIGHT_CHUNKS_KEY) or []) if isinstance(c, dict)]
    if chunks:
        out["preflight_chunks"] = chunks
    return out


def _stamp_latency(order: dict[str, Any], submit_ts: float, response_ts: float) -> dict[str, Any]:
    if isinstance(order, dict):
        order["_mystic_latency"] = {"order_submit_timestamp": submit_ts, "order_response_timestamp": response_ts}
    return order


def execution_latency_fields(order: dict[str, Any] | None) -> dict[str, Any]:
    """Submit / ack / fill times for one live order. Telemetry only."""
    if not isinstance(order, dict):
        return {}
    lat = order.get("_mystic_latency") if isinstance(order.get("_mystic_latency"), dict) else {}
    info = order.get("info") if isinstance(order.get("info"), dict) else {}
    ack_ts = _ms_to_sec(info.get("transactTime")) or _ms_to_sec(order.get("timestamp"))
    fill_times = [t for t in (_ms_to_sec(tr.get("timestamp")) for tr in (order.get("trades") or []) if isinstance(tr, dict)) if t]
    fill_source = "venue_trades" if fill_times else ""
    fill_ts = max(fill_times) if fill_times else _ms_to_sec(order.get("lastTradeTimestamp"))
    if fill_ts and not fill_source:
        fill_source = "last_trade_timestamp"
    if not fill_ts and info.get("fills") and _ms_to_sec(info.get("transactTime")):
        # A FULL response's fills matched at transactTime; they carry no own time.
        fill_ts = _ms_to_sec(info.get("transactTime"))
        fill_source = "full_response_transact_time"
    submit_ts = lat.get("order_submit_timestamp")
    out: dict[str, Any] = {
        "order_submit_timestamp": submit_ts,
        "order_response_timestamp": lat.get("order_response_timestamp"),
        "exchange_ack_timestamp": ack_ts,
        "fill_timestamp": fill_ts,
        "fill_timestamp_source": fill_source or None,
    }
    if submit_ts and ack_ts:
        out["submit_to_ack_ms"] = (ack_ts - float(submit_ts)) * 1000.0
    if submit_ts and fill_ts:
        out["submit_to_fill_ms"] = (fill_ts - float(submit_ts)) * 1000.0
    return out


async def execute_protected_limit_live(
    live_service: Any,
    *,
    symbol: str,
    side: str,
    quantity: float,
    limit_price: float,
    client_order_id: str | None = None,
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
            submit_ts = time.time()
            result = await live_service.place_order(
                exchange="binanceus",
                symbol=exchange_symbol,
                order_type="limit",
                side=side_l,
                amount=quantity,
                price=limit_price,
                client_order_id=client_order_id,
                time_in_force=tif if tif == "IOC" else None,
            )
        except TypeError:
            submit_ts = time.time()
            result = await live_service.place_order(
                exchange="binanceus",
                symbol=exchange_symbol,
                order_type="limit",
                side=side_l,
                amount=quantity,
                price=limit_price,
                client_order_id=client_order_id,
            )
        except Exception as ex:
            logger.warning("PROTECTED_LIMIT_LIVE place failed tif=%s %s: %s", tif, exchange_symbol, ex)
            continue

        if not result or result.get("status") != "success":
            if tif == "IOC":
                continue
            return None

        response_ts = time.time()
        order = result.get("order") or {}
        order_id = str(order.get("id") or "")
        if order_id:
            order = await _enrich_live_order_fills(live_service, order, exchange_symbol)
        _stamp_latency(order, submit_ts, response_ts)
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
                # PROTECTED_LIMIT_ALLOW_PARTIAL is enforced pre-trade, in
                # preflight: a quantity the visible book cannot fill completely
                # is rejected as DEPTH_INSUFFICIENT and no order is sent. Once
                # an IOC has partially filled the asset is already in the
                # account, so the fill must be adopted and tracked. Returning
                # None here would leave inventory we own invisible to the
                # engine, which is what the dust reconciler then has to chase.
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
                return _stamp_latency(o, submit_ts, response_ts)
            if status in ("closed", "filled") and filled > 0:
                if PROTECTED_LIMIT_ALLOW_PARTIAL or filled + 1e-12 >= amount:
                    return _stamp_latency(o, submit_ts, response_ts)
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
            return _stamp_latency(last, submit_ts, response_ts)
        return None

    return None
