"""Exchange-complete live equity: cash, active marks, and retained dust.

``228.06746265`` is the adopted forward cash baseline at SHA 9039923, not
contributed principal. Lifetime owner deposits remain UNKNOWN.
"""

from __future__ import annotations

import json
import sqlite3
from decimal import Decimal
from typing import Any

from backend.config.execution_cost_model import TAKER_COMMISSION_PCT
from backend.services.day_entry_spendable import money
from backend.services.live_account_basis import (
    TRAILING_BUY_ANCHOR_EQUITY,
    TRAILING_BUY_ANCHOR_SHA,
    load_operational_json,
    persist_operational_json,
)

FORWARD_BASELINE_CASH = TRAILING_BUY_ANCHOR_EQUITY
FORWARD_BASELINE_SHA = TRAILING_BUY_ANCHOR_SHA
FORWARD_BASELINE_LABEL = "forward baseline equity (adopted cash at SHA 9039923; not contributed principal)"
LIFETIME_CONTRIBUTED_CAPITAL = "UNKNOWN"
DUST_TRADE_PREFIX = "dust_exchange:"
BASELINE_DUST_KEY = "forward_baseline_dust"
CURRENT_DUST_SNAPSHOT_KEY = "exchange_dust_snapshot"
QUOTE_ASSETS = frozenset({"USDT", "USD", "BUSD", "USDC"})


def dust_trade_id(symbol: str) -> str:
    raw = str(symbol or "").replace("/", "").replace("-", "").replace("_", "").upper()
    return f"{DUST_TRADE_PREFIX}{raw}"


def is_dust_import_id(trade_id: str) -> bool:
    return str(trade_id or "").startswith(DUST_TRADE_PREFIX)


def sell_fee_pct() -> Decimal:
    return money(TAKER_COMMISSION_PCT)


def exact_dust_quantity(exchange_qty: object) -> Decimal:
    """Lot-size floor must never write a real leftover off as zero."""
    qty = money(exchange_qty)
    return qty if qty > 0 else Decimal("0")


def mark_dust_asset(
    *,
    symbol: str,
    asset: str,
    quantity: object,
    executable_bid: object,
    fee_pct: object | None = None,
) -> dict[str, Any]:
    qty = money(quantity)
    bid = money(executable_bid)
    rate = money(fee_pct) if fee_pct is not None else sell_fee_pct()
    if qty < 0:
        qty = Decimal("0")
    if bid < 0:
        bid = Decimal("0")
    gross = qty * bid
    fee = gross * rate
    return {
        "symbol": str(symbol or ""),
        "asset": str(asset or ""),
        "quantity": str(qty),
        "executable_bid": str(bid),
        "dust_market_value": str(gross),
        "estimated_liquidation_cost": str(fee),
        "net_liquidatable_value": str(gross - fee),
        "status": "DUST_PENDING",
        "trade_id": dust_trade_id(symbol),
    }


def should_import_exchange_dust(
    *,
    existing_status: str = "",
    existing_trade_id: str = "",
    existing_qty: object = 0,
    exchange_qty: object,
) -> str:
    """Return import | update | skip. Never duplicates an already-held ACTIVE lot."""
    qty = money(exchange_qty)
    if qty <= 0:
        return "skip"
    status = str(existing_status or "").upper()
    if status == "ACTIVE":
        return "skip"
    if status == "DUST_PENDING" or is_dust_import_id(existing_trade_id):
        if money(existing_qty) == qty:
            return "skip"
        return "update"
    if status:
        return "skip"
    return "import"


def build_exchange_equity(
    *,
    cash_usdt: object,
    active_marks: list[dict[str, Any]] | None = None,
    dust_marks: list[dict[str, Any]] | None = None,
    realized_strategy_pnl: object = 0,
    unrealized_strategy_pnl: object = 0,
    accounting_corrections: object = 0,
    forward_baseline_cash: object | None = None,
    forward_baseline_dust_net: object | None = None,
    baseline_dust_known: bool = False,
) -> dict[str, Any]:
    cash = money(cash_usdt)
    active = Decimal("0")
    for row in active_marks or []:
        active += money(row.get("market_value") or (money(row.get("quantity")) * money(row.get("mark") or row.get("bid"))))
    dust_gross = Decimal("0")
    dust_fee = Decimal("0")
    dust_by_coin: list[dict[str, Any]] = []
    for row in dust_marks or []:
        marked = mark_dust_asset(
            symbol=str(row.get("symbol") or ""),
            asset=str(row.get("asset") or ""),
            quantity=row.get("quantity") or 0,
            executable_bid=row.get("executable_bid") or row.get("bid") or 0,
            fee_pct=row.get("fee_pct"),
        )
        dust_gross += money(marked["dust_market_value"])
        dust_fee += money(marked["estimated_liquidation_cost"])
        dust_by_coin.append(marked)
    active_fee = Decimal("0")
    for row in active_marks or []:
        val = money(row.get("market_value") or 0)
        if val <= 0:
            val = money(row.get("quantity")) * money(row.get("mark") or row.get("bid") or 0)
        active_fee += val * sell_fee_pct()
    liq_cost = dust_fee + active_fee
    gross = cash + active + dust_gross
    net = cash + active + dust_gross - liq_cost
    baseline_cash = money(forward_baseline_cash) if forward_baseline_cash is not None else FORWARD_BASELINE_CASH
    comparable = None
    forward_change = None
    uncertainty = ""
    if baseline_dust_known and forward_baseline_dust_net is not None:
        comparable = baseline_cash + money(forward_baseline_dust_net)
        forward_change = net - comparable
    else:
        uncertainty = "Baseline dust at SHA 9039923 cannot be reconstructed exactly. forward_cash_change is cash-only and is not total-account P&L."
    return {
        "cash_usdt": str(cash),
        "active_position_market_value": str(active),
        "dust_market_value": str(dust_gross),
        "dust_by_coin": dust_by_coin,
        "gross_exchange_equity": str(gross),
        "estimated_liquidation_cost": str(liq_cost),
        "net_liquidatable_equity": str(net),
        "forward_baseline_equity": str(baseline_cash),
        "forward_baseline_label": FORWARD_BASELINE_LABEL,
        "forward_baseline_sha": FORWARD_BASELINE_SHA,
        "forward_baseline_comparable": str(comparable) if comparable is not None else None,
        "forward_net_equity_change": str(forward_change) if forward_change is not None else None,
        "forward_cash_change": str(cash - baseline_cash),
        "realized_strategy_pnl": str(money(realized_strategy_pnl)),
        "unrealized_strategy_pnl": str(money(unrealized_strategy_pnl)),
        "accounting_corrections": str(money(accounting_corrections)),
        "lifetime_contributed_capital": LIFETIME_CONTRIBUTED_CAPITAL,
        "baseline_dust_known": bool(baseline_dust_known),
        "uncertainty": uncertainty,
    }


def reconstruct_forward_baseline_dust(db_path: str) -> dict[str, Any]:
    """Best-effort dust at the 9039923 cash baseline. Exact only if a snapshot exists."""
    for key in (BASELINE_DUST_KEY, "exchange_balance_snapshot_9039923", "position_dust_snapshot_9039923"):
        stored = load_operational_json(db_path, key)
        coins = stored.get("coins") if isinstance(stored, dict) else None
        sha = str(stored.get("sha") or stored.get("baseline_sha") or "")
        if isinstance(coins, list) and coins and (not sha or sha.startswith("9039923")):
            net = Decimal("0")
            for row in coins:
                net += money(row.get("net_liquidatable_value") or 0)
            return {
                "known": True,
                "source": str(stored.get("source") or key),
                "net_liquidatable": str(net),
                "coins": coins,
            }
    return {
        "known": False,
        "source": "unavailable",
        "net_liquidatable": None,
        "coins": [],
        "uncertainty": (
            "No exchange-balance, position, or dust snapshot exists at SHA 9039923. "
            "Current leftovers plus later Binance.US fills cannot prove the baseline inventory, "
            "so cash-only change is not total-account P&L."
        ),
    }


def equity_basis_from_db(db_path: str, *, cash_usdt: object, principal: object) -> dict[str, Any]:
    """Presentation inputs from the current dust snapshot and the 9039923 baseline."""
    baseline = reconstruct_forward_baseline_dust(db_path)
    snap = load_operational_json(db_path, CURRENT_DUST_SNAPSHOT_KEY)
    coins = snap.get("coins") if isinstance(snap, dict) else []
    eq = build_exchange_equity(
        cash_usdt=cash_usdt,
        dust_marks=coins if isinstance(coins, list) else [],
        forward_baseline_cash=principal,
        forward_baseline_dust_net=baseline.get("net_liquidatable"),
        baseline_dust_known=bool(baseline.get("known")),
    )
    return {
        "current_equity": float(eq["cash_usdt"]),
        "forward_baseline_equity": float(eq["forward_baseline_equity"]),
        "net_liquidatable_equity": float(eq["net_liquidatable_equity"]),
        "baseline_dust_known": bool(eq["baseline_dust_known"]),
    }


def persist_current_dust_snapshot(db_path: str, coins: list[dict[str, Any]]) -> None:
    existing = load_operational_json(db_path, CURRENT_DUST_SNAPSHOT_KEY)
    by_symbol: dict[str, dict[str, Any]] = {}
    for row in existing.get("coins") or []:
        if isinstance(row, dict) and row.get("symbol"):
            by_symbol[str(row["symbol"])] = row
    for row in coins or []:
        if isinstance(row, dict) and row.get("symbol"):
            by_symbol[str(row["symbol"])] = row
    persist_operational_json(
        db_path,
        CURRENT_DUST_SNAPSHOT_KEY,
        {"coins": list(by_symbol.values()), "source": "live_exchange_retain"},
    )


def cap_qty_coverage_pct(*, matched_qty: float, venue_qty: float) -> float:
    if venue_qty <= 0:
        return 0.0
    return min(100.0, 100.0 * float(matched_qty) / float(venue_qty))


def backfill_provable_fill_identities(
    db_path: str,
    *,
    recorded: list[dict[str, Any]],
    venue_fills: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Write live_exchange_fills only when symbol+side+order id already agree."""
    from backend.services.live_order_identity import OrderIdentity, record_fill

    written: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str]] = set()
    rec_by_order: dict[tuple[str, str, str], dict[str, Any]] = {}
    for row in recorded:
        oid = str(row.get("order_id") or "").strip()
        if not oid:
            continue
        key = (str(row.get("symbol") or ""), str(row.get("side") or "").upper(), oid)
        rec_by_order[key] = row
    for fill in venue_fills:
        oid = str(fill.get("order") or fill.get("exchange_order_id") or "").strip()
        if not oid:
            continue
        key = (str(fill.get("symbol") or ""), str(fill.get("side") or "").upper(), oid)
        row = rec_by_order.get(key)
        if row is None:
            continue
        if key in seen:
            continue
        seen.add(key)
        trade_id = str(fill.get("id") or fill.get("venue_trade_id") or "").strip()
        if not trade_id:
            continue
        qty = float(fill.get("qty") or fill.get("amount") or 0.0)
        cost = float(fill.get("cost") or 0.0)
        px = float(fill.get("price") or 0.0)
        if px <= 0 and qty > 0 and cost > 0:
            px = cost / qty
        identity = OrderIdentity(
            symbol=key[0],
            side=key[1],
            exchange_order_id=oid,
            client_order_id=str(row.get("client_order_id") or fill.get("client_order_id") or ""),
            fill_ids=[trade_id],
            venue_trade_ids=[trade_id],
            executed_qty=qty,
            avg_fill_price=px,
            cost_quote=cost,
            fee_amount=float(fill.get("fee_cost") or 0.0),
            fee_asset=str(fill.get("fee_ccy") or ""),
            fee_from_exchange=True,
            mystic_trade_id=str(row.get("trade_id") or row.get("local_id") or ""),
            event_ts_exchange=str(fill.get("timestamp") or ""),
            raw={"source": "provable_order_id_backfill", "venue_trade_id": trade_id},
        )
        if record_fill(db_path, identity):
            written.append({"symbol": key[0], "side": key[1], "exchange_order_id": oid, "venue_trade_id": trade_id})
    return written


def backfill_exchange_reconciled_orders(
    db_path: str,
    *,
    unmatched: list[dict[str, Any]],
    known_order_ids: set[str] | None = None,
) -> list[dict[str, Any]]:
    """Write EXCHANGE_RECONCILED identity for full unmatched venue orders only."""
    from backend.services.live_order_identity import record_exchange_reconciled
    from backend.services.live_pnl_reconciliation import classify_unmatched_venue_fills

    known = {str(x) for x in (known_order_ids or set()) if str(x).strip()}
    groups = classify_unmatched_venue_fills(unmatched, known_order_ids=known)
    written: list[dict[str, Any]] = []
    for row in groups["full_buys_lacking_local"] + groups["full_sells_lacking_local"]:
        oid = str(row.get("exchange_order_id") or "").strip()
        if not oid or oid in known:
            continue
        ok = record_exchange_reconciled(
            db_path,
            symbol=str(row.get("symbol") or ""),
            side=str(row.get("side") or ""),
            exchange_order_id=oid,
            client_order_id=str(row.get("client_order_id") or ""),
            venue_trade_ids=[str(row.get("venue_trade_id") or "")] if row.get("venue_trade_id") else [],
            executed_qty=float(row.get("quantity") or row.get("unmatched_quantity") or 0.0),
            avg_fill_price=float(row.get("price") or 0.0),
            cost_quote=float(row.get("quantity") or 0.0) * float(row.get("price") or 0.0),
            classification=str(row.get("group") or "full_unmatched"),
            event_ts_exchange=str(row.get("timestamp") or ""),
            matching_sell_order_id=row.get("matching_sell_order_id"),
            remaining_asset=row.get("remaining_asset"),
            raw={"unmatched_reason": row.get("unmatched_reason") or ""},
        )
        if ok:
            written.append({"symbol": row.get("symbol"), "side": row.get("side"), "exchange_order_id": oid, "source": "EXCHANGE_RECONCILED"})
            known.add(oid)
    return written


def load_json_state(db_path: str, key: str) -> dict[str, Any]:
    return load_operational_json(db_path, key)


def dump_json(payload: dict[str, Any]) -> str:
    return json.dumps(payload, separators=(",", ":"), default=str)


def ledger_has_dust_row(db_path: str, symbol: str) -> bool:
    if not db_path:
        return False
    try:
        with sqlite3.connect(db_path, timeout=5) as conn:
            row = conn.execute(
                "SELECT trade_id, status FROM portfolio_engine_positions WHERE replace(replace(upper(symbol),'/',''),'-','') = ?",
                (str(symbol or "").replace("/", "").replace("-", "").upper(),),
            ).fetchone()
        if not row:
            return False
        return str(row[1] or "").upper() == "DUST_PENDING" or is_dust_import_id(str(row[0] or ""))
    except sqlite3.Error:
        return False
