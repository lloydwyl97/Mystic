"""Live fill economics — Binance commissions and live-mode PnL isolation.

Paper cost models are unchanged. Historical paper rows are not rewritten.
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from decimal import Decimal
from typing import Any

from backend.services.day_entry_spendable import floor_to_step, money
from backend.services.live_account_basis import load_operational_json, persist_operational_json
from backend.utils.symbols import normalize_symbol

QUOTE_FEE_ASSETS = frozenset({"USDT", "USD", "BUSD", "USDC"})
FIRST_XRP_RT_CORRECTION_ID = "xrp_rt_488230379_488239569"
FIRST_XRP_RT_CORRECTION_KEY = f"live_accounting_correction:{FIRST_XRP_RT_CORRECTION_ID}"
ETH_LOT_CORRECTION_ID = "eth_integrity_1587754176_1587893573"
ETH_LOT_CORRECTION_KEY = f"live_accounting_correction:{ETH_LOT_CORRECTION_ID}"
_CORRECTIONS_TABLE = "live_accounting_corrections"

# coin_performance field classification (audit 2026-08-24).
# A = shared learning signal (intentionally mixed paper/live history)
# B = live-only safety / account-state authority when DAY is LIVE
# C = currently contaminated if a B field is computed from mixed history
COIN_PERFORMANCE_FIELD_CLASS: dict[str, str] = {
    "win_rate_20": "A",
    "sizing_multiplier": "A",
    "expectancy": "A",
    "avg_win": "A",
    "avg_loss": "A",
    "profit_factor": "A",  # learning; live PF *pause* is B
    "trades_last_30d": "A",
    "stop_loss_hits_10": "B",  # C until live rolling last-N is used
    "pause_until": "B",  # C if inherited from paper loss_heavy / 24h / PF pause
    "trades_24h": "B",  # pause Rule 1 authority when LIVE
    "pnl_24h": "B",  # pause Rule 1 authority when LIVE
}


@dataclass(frozen=True)
class LiveCommission:
    """Asset-aware exchange commission for one live fill."""

    usd: float
    items: tuple[dict[str, Any], ...] = ()
    fee_from_exchange: bool = False
    base_qty_reduction: float = 0.0
    quote_commission_usd: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "exchange_commission_usd": round(float(self.usd), 8),
            "quote_commission_usd": round(float(self.quote_commission_usd), 8),
            "fee_from_exchange": bool(self.fee_from_exchange),
            "base_qty_reduction": float(self.base_qty_reduction),
            "items": list(self.items),
        }


def _base_asset(symbol: str) -> str:
    ns = normalize_symbol(symbol).replace("/", "")
    if ns.endswith("USDT"):
        return ns[:-4]
    if ns.endswith("USD"):
        return ns[:-3]
    return ns


def _as_float(raw: Any) -> float | None:
    if raw is None:
        return None
    try:
        return float(raw)
    except (TypeError, ValueError):
        return None


def _fee_pairs_from_fills(fills: Any) -> list[tuple[float, str]]:
    out: list[tuple[float, str]] = []
    if not isinstance(fills, list):
        return out
    for fill in fills:
        if not isinstance(fill, dict):
            continue
        qty = _as_float(fill.get("commission") if fill.get("commission") is not None else fill.get("amount"))
        asset = str(fill.get("commissionAsset") or fill.get("asset") or fill.get("currency") or "").upper()
        if qty and asset:
            out.append((abs(qty), asset))
    return out


def _fee_pairs_from_trades(trades: Any) -> list[tuple[float, str]]:
    out: list[tuple[float, str]] = []
    if not isinstance(trades, list):
        return out
    for trade in trades:
        if not isinstance(trade, dict):
            continue
        tfee = trade.get("fee")
        if isinstance(tfee, dict):
            qty = _as_float(tfee.get("cost") if tfee.get("cost") is not None else tfee.get("amount"))
            asset = str(tfee.get("currency") or tfee.get("asset") or "").upper()
            if qty and asset:
                out.append((abs(qty), asset))
                continue
        qty = _as_float(trade.get("commission"))
        asset = str(trade.get("commissionAsset") or "").upper()
        if qty and asset:
            out.append((abs(qty), asset))
    return out


def _iter_fee_items(order: dict[str, Any] | None) -> list[tuple[float, str]]:
    """Read each venue commission once. fills, trades, and fee objects are aliases."""
    if not order or not isinstance(order, dict):
        return []
    info = order.get("info") if isinstance(order.get("info"), dict) else {}
    from_fills = _fee_pairs_from_fills(info.get("fills"))
    if from_fills:
        return from_fills
    from_trades = _fee_pairs_from_trades(order.get("trades"))
    if from_trades:
        return from_trades
    out: list[tuple[float, str]] = []

    def _add(amount: Any, asset: Any) -> None:
        qty = _as_float(amount)
        if qty is None or qty == 0:
            return
        out.append((abs(qty), str(asset or "").upper()))

    raw_fee = order.get("fee")
    if isinstance(raw_fee, dict):
        _add(raw_fee.get("cost") if raw_fee.get("cost") is not None else raw_fee.get("amount"), raw_fee.get("currency") or raw_fee.get("asset"))
    elif isinstance(raw_fee, (int, float)) and float(raw_fee) != 0:
        _add(raw_fee, order.get("feeAsset") or order.get("commissionAsset") or "USDT")
    raw_fees = order.get("fees")
    if isinstance(raw_fees, list):
        for item in raw_fees:
            if isinstance(item, dict):
                _add(item.get("cost") if item.get("cost") is not None else item.get("amount"), item.get("currency") or item.get("asset"))
    if out:
        return out
    _add(order.get("commission"), order.get("commissionAsset"))
    if isinstance(info, dict):
        _add(info.get("commission"), info.get("commissionAsset"))
    return out


def extract_live_commission(
    order: dict[str, Any] | None,
    *,
    symbol: str,
    fill_price: float,
) -> LiveCommission:
    """Convert Binance/CCXT fee fields into quote-USD and optional base-qty reduction."""
    items = _iter_fee_items(order)
    if not items:
        return LiveCommission(usd=0.0, fee_from_exchange=False)

    base = _base_asset(symbol)
    usd = 0.0
    quote_usd = 0.0
    base_qty = 0.0
    detail: list[dict[str, Any]] = []
    px = float(fill_price or 0.0)
    for amount, asset in items:
        if not asset or asset in {"USDT", "USD", "BUSD"}:
            usd += amount
            quote_usd += amount
            conv = amount
        elif asset == base:
            base_qty += amount
            conv = amount * px if px > 0 else 0.0
            usd += conv
        else:
            # Unknown asset (e.g. BNB). Keep the raw item; do not invent a FX rate.
            conv = 0.0
        detail.append({"amount": amount, "asset": asset, "usd": round(conv, 8)})
    return LiveCommission(
        usd=float(usd),
        items=tuple(detail),
        fee_from_exchange=True,
        base_qty_reduction=float(base_qty),
        quote_commission_usd=float(quote_usd),
    )


def apply_live_buy_economics(
    *,
    filled_qty: float,
    fill_price: float,
    modeled_fee: float,
    commission: LiveCommission,
) -> tuple[float, float, float]:
    """Return (received_qty, entry_fee_usd, cash_debit). Exchange fees win when present."""
    filled = money(filled_qty)
    if filled < 0:
        filled = Decimal("0")
    px = money(fill_price)
    if commission.fee_from_exchange:
        received = filled - money(commission.base_qty_reduction or 0)
        if received < 0:
            received = Decimal("0")
        fee = money(commission.usd)
        cash = filled * px + money(commission.quote_commission_usd or 0)
        return float(received), float(fee), float(cash)
    modeled = money(modeled_fee)
    return float(filled), float(modeled), float(filled * px + modeled)


def sellable_and_residual_qty(
    *,
    credited_qty: object,
    qty_step: object,
) -> tuple[Decimal, Decimal]:
    """Maximum exchange-valid sell qty and leftover real inventory."""
    credited = money(credited_qty)
    if credited < 0:
        credited = Decimal("0")
    sellable = floor_to_step(credited, money(qty_step))
    residual = credited - sellable
    if residual < 0:
        residual = Decimal("0")
    return sellable, residual


@dataclass(frozen=True)
class PlannedSell:
    sellable: Decimal
    residual: Decimal
    protected_dust: Decimal
    exchange_available: Decimal
    borrowed_from_dust: Decimal


def plan_sell_quantity(
    *,
    net_active_qty: object,
    exchange_free_qty: object,
    protected_dust_qty: object,
    qty_step: object,
) -> PlannedSell:
    """Sell only net active inventory. Never borrow pre-existing DUST_PENDING."""
    active = money(net_active_qty)
    free = money(exchange_free_qty)
    protected = money(protected_dust_qty)
    if active < 0:
        active = Decimal("0")
    if free < 0:
        free = Decimal("0")
    if protected < 0:
        protected = Decimal("0")
    available = free - protected
    if available < 0:
        available = Decimal("0")
    capped = active if active <= available else available
    sellable, _floored_residual = sellable_and_residual_qty(credited_qty=capped, qty_step=qty_step)
    residual = active - sellable
    if residual < 0:
        residual = Decimal("0")
    return PlannedSell(
        sellable=sellable,
        residual=residual,
        protected_dust=protected,
        exchange_available=available,
        borrowed_from_dust=Decimal("0"),
    )


def quantity_identity(
    *,
    starting_dust: object,
    gross_bought: object,
    buy_base_commission: object,
    sold: object,
) -> Decimal:
    """starting + gross - base BUY commission - sold = ending base balance."""
    ending = money(starting_dust) + money(gross_bought) - money(buy_base_commission) - money(sold)
    return ending


def merge_venue_trades_into_order(order: dict[str, Any] | None, trades: list[dict[str, Any]] | None) -> dict[str, Any]:
    """Attach myTrades onto a GET /order payload that omitted fills."""
    raw = dict(order) if isinstance(order, dict) else {}
    rows = [t for t in (trades or []) if isinstance(t, dict)]
    if not rows:
        return raw
    if raw.get("trades") or (isinstance(raw.get("info"), dict) and raw["info"].get("fills")):
        return raw
    fills: list[dict[str, Any]] = []
    normalized: list[dict[str, Any]] = []
    for t in rows:
        info = t.get("info") if isinstance(t.get("info"), dict) else {}
        fee = t.get("fee") if isinstance(t.get("fee"), dict) else {}
        tid = str(t.get("trade_id") or t.get("id") or info.get("id") or "")
        oid = str(t.get("order_id") or t.get("order") or info.get("orderId") or "")
        qty = t.get("qty") if t.get("qty") is not None else t.get("amount") or info.get("qty")
        px = t.get("price") if t.get("price") is not None else info.get("price")
        comm = t.get("commission") if t.get("commission") is not None else fee.get("cost") or info.get("commission")
        asset = t.get("commission_asset") or fee.get("currency") or info.get("commissionAsset")
        fills.append(
            {
                "tradeId": tid,
                "orderId": oid,
                "price": px,
                "qty": qty,
                "commission": comm,
                "commissionAsset": asset,
                "quoteQty": t.get("quote_qty") or t.get("cost") or info.get("quoteQty"),
                "isBuyerMaker": info.get("isBuyerMaker"),
            }
        )
        normalized.append(
            {
                "id": tid,
                "order": oid,
                "amount": qty,
                "price": px,
                "cost": t.get("quote_qty") or t.get("cost") or info.get("quoteQty"),
                "takerOrMaker": t.get("taker_or_maker") or t.get("takerOrMaker"),
                "fee": {"cost": comm, "currency": asset},
                "commission": comm,
                "commissionAsset": asset,
                "timestamp": t.get("timestamp") or info.get("time"),
            }
        )
    raw["trades"] = normalized
    info = dict(raw.get("info") or {}) if isinstance(raw.get("info"), dict) else {}
    info["fills"] = fills
    raw["info"] = info
    return raw


def first_xrp_rt_venue_facts() -> dict[str, Any]:
    """Authoritative Binance.US fills for BUY 488230379 and SELL 488239569."""
    buy_px = money("1.3863")
    sell_px = money("1.3958")
    buy_qty = money("28.4")
    sell_qty = money("28.4")
    buy_fee_base = money("0.00568")
    sell_fee_quote = money("0.00792814")
    buy_quote = money("39.37092")
    sell_quote = money("39.64072")
    starting_dust = money("0.0938")
    ending_dust = money("0.08812")
    cash_before = money("223.92463088")
    cash_after = money("224.18650274")
    estimated_buy_quote = buy_quote * money("0.0006")
    buy_fee_quote_value = buy_fee_base * buy_px
    net_credited = buy_qty - buy_fee_base
    dust_consumed = starting_dust - ending_dust
    cash_increase = cash_after - cash_before
    consumed_dust_value = dust_consumed * sell_px
    economic = cash_increase - consumed_dust_value
    trade_sold = net_credited
    sell_fee_trade = sell_fee_quote * (trade_sold / sell_qty) if sell_qty else Decimal("0")
    realized_on_sold_trade_qty = trade_sold * sell_px - sell_fee_trade - trade_sold * buy_px - buy_fee_quote_value
    return {
        "correction_id": FIRST_XRP_RT_CORRECTION_ID,
        "is_new_trade": False,
        "buy": {
            "trade_id": "2627001",
            "order_id": "488230379",
            "quantity": str(buy_qty),
            "price": str(buy_px),
            "quote_quantity": str(buy_quote),
            "commission_amount": str(buy_fee_base),
            "commission_asset": "XRP",
            "maker_taker": "taker",
            "fill_timestamp_ms": 1789757821812,
        },
        "sell": {
            "trade_id": "2627040",
            "order_id": "488239569",
            "quantity": str(sell_qty),
            "price": str(sell_px),
            "quote_quantity": str(sell_quote),
            "commission_amount": str(sell_fee_quote),
            "commission_asset": "USDT",
            "maker_taker": "taker",
            "fill_timestamp_ms": 1789758635098,
        },
        "actual_exchange_commission": {
            "buy_amount": str(buy_fee_base),
            "buy_asset": "XRP",
            "buy_quote_value": str(buy_fee_quote_value),
            "sell_amount": str(sell_fee_quote),
            "sell_asset": "USDT",
        },
        "estimated_pre_trade_cost": {
            "buy_quote_6bps": str(estimated_buy_quote),
            "source": "ESTIMATED_ROUNDTRIP_COST 6 bps of buy notional; not venue commission",
        },
        "spread": None,
        "slippage": None,
        "base_commission_accounting_value": str(buy_fee_quote_value),
        "quantity_identity": {
            "starting_dust": str(starting_dust),
            "gross_bought": str(buy_qty),
            "buy_base_commission": str(buy_fee_base),
            "sold": str(sell_qty),
            "ending": str(
                quantity_identity(
                    starting_dust=starting_dust,
                    gross_bought=buy_qty,
                    buy_base_commission=buy_fee_base,
                    sold=sell_qty,
                )
            ),
            "expected_ending": str(ending_dust),
        },
        "old_dust_consumed": True,
        "dust_consumed_qty": str(dust_consumed),
        "net_credited_qty": str(net_credited),
        "remaining_new_residual": "0",
        "gross_price_pnl_sold_qty": str(sell_qty * (sell_px - buy_px)),
        "realized_pnl_on_trade_qty": str(realized_on_sold_trade_qty),
        "preexisting_dust_change_qty": str(-dust_consumed),
        "cash_increase": str(cash_increase),
        "consumed_dust_value_at_sell": str(consumed_dust_value),
        "total_economic_change": str(economic),
        "mystic_sell_trade_id": "mystic_sell_XRP/USDT_1789758633629",
        "mystic_buy_trade_id": "mystic_XRP/USDT_1789757821721",
        "intent_id": "tb8d49b4f6d2f3433d",
    }


def _ensure_corrections_table(conn: sqlite3.Connection) -> None:
    conn.execute(
        f"""
        CREATE TABLE IF NOT EXISTS {_CORRECTIONS_TABLE} (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            correction_id TEXT NOT NULL UNIQUE,
            buy_order_id TEXT,
            sell_order_id TEXT,
            mystic_sell_trade_id TEXT,
            payload_json TEXT NOT NULL,
            created_ts TEXT NOT NULL
        )
        """
    )


def active_lot_keeps_booked_qty(*, booked_qty: object, exchange_qty: object) -> bool:
    """ACTIVE booked qty stays if exchange still covers it. Surplus is dust."""
    booked = money(booked_qty)
    exchange = money(exchange_qty)
    if booked <= 0:
        return False
    return exchange + Decimal("0.000000000001") >= booked


def eth_captured_buy_1587754176() -> dict[str, Any]:
    """Authoritative Binance.US fill for BUY 1587754176."""
    gross = money("0.0227")
    fee = money("0.00000454")
    px = money("2638.42")
    return {
        "trade_id": "18722272",
        "order_id": "1587754176",
        "quantity": str(gross),
        "price": str(px),
        "quote_quantity": "59.892134",
        "commission_amount": str(fee),
        "commission_asset": "ETH",
        "maker_taker": "taker",
        "fill_timestamp_ms": 1789761330207,
        "net_credited": str(gross - fee),
        "quote_commission": "0",
        "buy_fee_quote_value": str(fee * px),
    }


def eth_current_buy_1587893573() -> dict[str, Any]:
    """Authoritative Binance.US fill for the live ETH lot after 1587754176 exited."""
    gross = money("0.0166")
    fee = money("0.00000332")
    px = money("2629.51")
    return {
        "trade_id": "18722674",
        "order_id": "1587893573",
        "quantity": str(gross),
        "price": str(px),
        "quote_quantity": "43.649866",
        "commission_amount": str(fee),
        "commission_asset": "ETH",
        "maker_taker": "taker",
        "fill_timestamp_ms": 1789770820700,
        "net_credited": str(gross - fee),
        "quote_commission": "0",
        "buy_fee_quote_value": str(fee * px),
    }


def _eth_integrity_reservation_audit(db_path: str) -> dict[str, Any]:
    """Canonicalize filled ETH reservations. Safe to call on every replay."""
    from backend.services.day_entry_reservations import correct_filled_reservation_to_consumed

    return {
        "named_reservation": correct_filled_reservation_to_consumed(db_path, reservation_id="res_b1271f7a936e4a82"),
        "live_reservation": correct_filled_reservation_to_consumed(db_path, reservation_id="res_9e7ebc24ecf34ffe"),
    }


def _eth_integrity_stamp_dust(db_path: str, payload: dict[str, Any], exchange_eth: object | None) -> None:
    from backend.services.live_exchange_equity import stamp_protected_preexisting_dust

    live_net = money(eth_current_buy_1587893573()["net_credited"])
    if exchange_eth is None:
        return
    surplus = money(exchange_eth) - live_net
    if surplus < 0:
        surplus = Decimal("0")
    payload["protected_dust"] = str(surplus)
    payload["exchange_eth"] = str(money(exchange_eth))
    stamp_protected_preexisting_dust(db_path, "ETH/USDT", surplus)


def apply_eth_lot_integrity_correction(db_path: str, *, exchange_eth: object | None = None) -> dict[str, Any]:
    """Reconcile ETH qty/reservations. Idempotent. Never inserts a second trade."""
    from datetime import datetime, timezone

    existing = load_operational_json(db_path, ETH_LOT_CORRECTION_KEY)
    if existing.get("applied"):
        existing.update(_eth_integrity_reservation_audit(db_path))
        _eth_integrity_stamp_dust(db_path, existing, exchange_eth)
        persist_operational_json(db_path, ETH_LOT_CORRECTION_KEY, existing)
        return existing
    named = eth_captured_buy_1587754176()
    live = eth_current_buy_1587893573()
    live_net = money(live["net_credited"])
    payload: dict[str, Any] = {
        "correction_id": ETH_LOT_CORRECTION_ID,
        "is_new_trade": False,
        "named_buy": named,
        "live_buy": live,
        "named_reservation_id": "res_b1271f7a936e4a82",
        "live_reservation_id": "res_9e7ebc24ecf34ffe",
        "named_intent_id": "tb52cbcd4ef74e4e1b",
        "named_decision_id": "day_ETHUSDT_1789760671103",
        "live_intent_id": "tba3fa216c3f724628",
        "live_decision_id": "day_ETHUSDT_1789770008654",
        "exchange_order_ids": [named["order_id"], live["order_id"]],
        "venue_trade_ids": [named["trade_id"], live["trade_id"]],
        "previous_reservation_status": "RELEASED",
        "corrected_canonical_status": "CONSUMED",
        "correction_reason": "triple-counted base fee and RELEASED-after-fill reservation",
        "reason": "triple-counted base fee and RELEASED-after-fill reservation",
        "correction_timestamp": datetime.now(timezone.utc).isoformat(),
    }
    try:
        with sqlite3.connect(db_path, timeout=15) as conn:
            conn.execute("BEGIN IMMEDIATE")
            _ensure_corrections_table(conn)
            already = conn.execute(
                f"SELECT 1 FROM {_CORRECTIONS_TABLE} WHERE correction_id=?",
                (ETH_LOT_CORRECTION_ID,),
            ).fetchone()
            if already:
                conn.commit()
                payload["applied"] = True
                payload.update(_eth_integrity_reservation_audit(db_path))
                _eth_integrity_stamp_dust(db_path, payload, exchange_eth)
                persist_operational_json(db_path, ETH_LOT_CORRECTION_KEY, payload)
                return payload
            pos_row = conn.execute(
                "SELECT symbol, quantity, status, trade_id, entry_price, entry_order_id, stop_price, "
                "take_profit_1_price, trailing_stop_price, highest_price, thesis_json "
                "FROM portfolio_engine_positions WHERE symbol IN ('ETH/USDT','ETHUSDT')"
            ).fetchone()
            prior_pos = None
            if pos_row:
                prior_pos = {
                    "symbol": pos_row[0],
                    "quantity": pos_row[1],
                    "status": pos_row[2],
                    "trade_id": pos_row[3],
                    "entry_price": pos_row[4],
                    "entry_order_id": pos_row[5],
                    "stop_price": pos_row[6],
                    "take_profit_1_price": pos_row[7],
                    "trailing_stop_price": pos_row[8],
                    "highest_price": pos_row[9],
                    "thesis_json": pos_row[10],
                }
            payload["prior_position"] = prior_pos
            buy_row = conn.execute("SELECT quantity, fees_paid FROM paper_trades WHERE order_id='1587754176' AND UPPER(side)='BUY'").fetchone()
            payload["prior_named_buy"] = {"quantity": buy_row[0], "fees_paid": buy_row[1]} if buy_row else None
            sell_row = conn.execute("SELECT trade_id, quantity, price, fees_paid, pnl, pnl_usd_net FROM paper_trades WHERE order_id='1587872124' AND UPPER(side)='SELL'").fetchone()
            payload["prior_named_sell"] = (
                dict(
                    zip(
                        ("trade_id", "quantity", "price", "fees_paid", "pnl", "pnl_usd_net"),
                        sell_row,
                        strict=False,
                    )
                )
                if sell_row
                else None
            )
            conn.execute(
                "UPDATE paper_trades SET quantity=?, fees_paid=? WHERE order_id='1587754176' AND UPPER(side)='BUY'",
                (float(money(named["net_credited"])), float(money(named["buy_fee_quote_value"]))),
            )
            if sell_row:
                sold = money(sell_row[1])
                exit_px = money(sell_row[2])
                entry_px = money(named["price"])
                sell_fee = money(sell_row[3] if sell_row[3] is not None else "0.01187381")
                buy_fee_q = money(named["buy_fee_quote_value"])
                net_all = sold * (exit_px - entry_px) - buy_fee_q - sell_fee
                payload["named_sell_net_pnl"] = str(net_all)
                cols = {str(r[1]) for r in conn.execute("PRAGMA table_info(paper_trades)")}
                sets = ["fees_paid=?", "pnl=?"]
                params: list[Any] = [float(sell_fee), float(net_all)]
                if "pnl_usd_net" in cols:
                    sets.append("pnl_usd_net=?")
                    params.append(float(net_all))
                params.append("1587872124")
                conn.execute(
                    f"UPDATE paper_trades SET {', '.join(sets)} WHERE order_id=? AND UPPER(side)='SELL'",
                    params,
                )
            if prior_pos and str(prior_pos.get("entry_order_id") or "") == "1587893573":
                conn.execute(
                    "UPDATE portfolio_engine_positions SET quantity=? WHERE symbol=? AND trade_id=?",
                    (float(live_net), prior_pos["symbol"], prior_pos["trade_id"]),
                )
                payload["corrected_active_qty"] = str(live_net)
                payload["preserved_entry_price"] = prior_pos.get("entry_price")
                payload["preserved_trade_id"] = prior_pos.get("trade_id")
            conn.execute(
                f"""
                INSERT INTO {_CORRECTIONS_TABLE} (
                    correction_id, buy_order_id, sell_order_id, mystic_sell_trade_id,
                    payload_json, created_ts
                ) VALUES (?, ?, ?, ?, ?, datetime('now'))
                """,
                (
                    ETH_LOT_CORRECTION_ID,
                    named["order_id"],
                    "1587872124",
                    (sell_row[0] if sell_row else ""),
                    json.dumps(payload, default=str),
                ),
            )
            conn.commit()
    except sqlite3.Error:
        return {**payload, "applied": False, "error": "sqlite"}
    payload.update(_eth_integrity_reservation_audit(db_path))
    payload["applied"] = True
    _eth_integrity_stamp_dust(db_path, payload, exchange_eth)
    persist_operational_json(db_path, ETH_LOT_CORRECTION_KEY, payload)
    return payload


def apply_first_xrp_rt_correction(db_path: str) -> dict[str, Any]:
    """Correct derived P&L for the first XRP RT. Never inserts a second trade."""
    facts = first_xrp_rt_venue_facts()
    existing = load_operational_json(db_path, FIRST_XRP_RT_CORRECTION_KEY)
    if existing.get("applied"):
        return existing
    economic = float(money(facts["total_economic_change"]))
    sell_fee = float(money(facts["sell"]["commission_amount"]))
    sell_id = str(facts["mystic_sell_trade_id"])
    prior_pnl = None
    prior_fees = None
    updated = False
    try:
        with sqlite3.connect(db_path, timeout=10) as conn:
            _ensure_corrections_table(conn)
            already = conn.execute(
                f"SELECT 1 FROM {_CORRECTIONS_TABLE} WHERE correction_id=?",
                (FIRST_XRP_RT_CORRECTION_ID,),
            ).fetchone()
            if already:
                persist_operational_json(db_path, FIRST_XRP_RT_CORRECTION_KEY, {**facts, "applied": True})
                return {**facts, "applied": True}
            cols = {str(r[1]) for r in conn.execute("PRAGMA table_info(paper_trades)")}
            if {"trade_id", "pnl"} <= cols:
                row = (
                    conn.execute(
                        "SELECT pnl, fees_paid FROM paper_trades WHERE trade_id=? AND UPPER(side)='SELL'",
                        (sell_id,),
                    ).fetchone()
                    if "fees_paid" in cols
                    else conn.execute(
                        "SELECT pnl FROM paper_trades WHERE trade_id=? AND UPPER(side)='SELL'",
                        (sell_id,),
                    ).fetchone()
                )
                if row:
                    prior_pnl = row[0]
                    prior_fees = row[1] if len(row) > 1 else None
                    sets = ["pnl=?"]
                    params: list[Any] = [economic]
                    if "pnl_usd_net" in cols:
                        sets.append("pnl_usd_net=?")
                        params.append(economic)
                    if "fees_paid" in cols:
                        sets.append("fees_paid=?")
                        params.append(sell_fee)
                    params.append(sell_id)
                    conn.execute(
                        f"UPDATE paper_trades SET {', '.join(sets)} WHERE trade_id=? AND UPPER(side)='SELL'",
                        params,
                    )
                    updated = True
            conn.execute(
                f"""
                INSERT INTO {_CORRECTIONS_TABLE} (
                    correction_id, buy_order_id, sell_order_id, mystic_sell_trade_id,
                    payload_json, created_ts
                ) VALUES (?, ?, ?, ?, ?, datetime('now'))
                """,
                (
                    FIRST_XRP_RT_CORRECTION_ID,
                    facts["buy"]["order_id"],
                    facts["sell"]["order_id"],
                    sell_id,
                    json.dumps({**facts, "prior_derived_pnl": prior_pnl, "prior_fees_paid": prior_fees}),
                ),
            )
            conn.commit()
    except sqlite3.Error:
        return {**facts, "applied": False, "error": "sqlite"}
    payload = {
        **facts,
        "applied": True,
        "derived_row_updated": updated,
        "prior_derived_pnl": prior_pnl,
        "prior_fees_paid": prior_fees,
    }
    persist_operational_json(db_path, FIRST_XRP_RT_CORRECTION_KEY, payload)
    return payload


def apply_live_sell_economics(
    *,
    quantity: float,
    fill_price: float,
    modeled_fee: float,
    commission: LiveCommission,
) -> tuple[float, float]:
    """Return (fee_usd, proceeds). Do not also subtract modeled slippage."""
    qty = float(quantity)
    px = float(fill_price or 0.0)
    if commission.fee_from_exchange:
        fee = float(commission.usd)
    else:
        fee = float(modeled_fee)
    return fee, (qty * px) - fee


def live_round_trip_net(
    *,
    quantity: float,
    entry_price: float,
    exit_price: float,
    entry_commission_usd: float,
    exit_commission_usd: float,
) -> dict[str, float]:
    """Gross is fill-to-fill. Net subtracts actual exchange commissions only."""
    qty = float(quantity)
    gross = qty * (float(exit_price) - float(entry_price))
    entry_fee = max(0.0, float(entry_commission_usd or 0.0))
    exit_fee = max(0.0, float(exit_commission_usd or 0.0))
    return {
        "gross_price_pnl": round(gross, 8),
        "exchange_commission_usd": round(entry_fee + exit_fee, 8),
        "net_realized_pnl": round(gross - entry_fee - exit_fee, 8),
    }


def sum_realized_pnl_by_mode(
    db_path: str,
    *,
    mode: str,
    day: str | None = None,
) -> float:
    """Sum SELL pnl for one paper_trades.mode. Does not rewrite rows."""
    wanted = str(mode or "").strip().lower()
    if not wanted:
        return 0.0
    try:
        with sqlite3.connect(db_path, timeout=5) as conn:
            expr = _sell_pnl_expr(conn)
            cols = {str(r[1]) for r in conn.execute("PRAGMA table_info(paper_trades)")}
            ghost_filter = ""
            if "order_id" in cols:
                ghost_filter += " AND COALESCE(order_id, '') != ''"
            if "counts_toward_realized" in cols:
                ghost_filter += " AND COALESCE(counts_toward_realized, 1) = 1"
            sql = f"""
                SELECT COALESCE(SUM({expr}), 0)
                FROM paper_trades
                WHERE UPPER(side)='SELL'
                  AND pnl IS NOT NULL
                  AND LOWER(COALESCE(mode, '')) = ?
                  AND COALESCE(is_synthetic, 0) = 0
                  AND COALESCE(exit_type, '') NOT IN (
                    'ADMIN_POSITION_CLEAR', 'STALE_PRE_CORRECTION_POSITION_CLEAR', 'RESEARCH_RESET_EXIT',
                    'DUST_WRITEOFF'
                  )
                  {ghost_filter}
            """
            params: list[Any] = [wanted]
            if day:
                sql += " AND date(timestamp) = ?"
                params.append(str(day))
            row = conn.execute(sql, params).fetchone()
        return float((row or (0.0,))[0] or 0.0)
    except sqlite3.Error:
        return 0.0


def sum_dust_adjustment_pnl(db_path: str) -> float:
    """Sum DUST_WRITEOFF SELL pnl. Account adjustment only — not strategy expectancy."""
    try:
        with sqlite3.connect(db_path, timeout=5) as conn:
            expr = _sell_pnl_expr(conn)
            row = conn.execute(
                f"""
                SELECT COALESCE(SUM({expr}), 0)
                FROM paper_trades
                WHERE UPPER(side)='SELL'
                  AND COALESCE(exit_type, '') = 'DUST_WRITEOFF'
                """,
            ).fetchone()
        return float((row or (0.0,))[0] or 0.0)
    except sqlite3.Error:
        return 0.0


def mode_scoped_equity_views(
    *,
    account_equity: float,
    principal: float,
    live_realized: float,
    paper_realized: float,
    unrealized: float,
    dust_adjustment: float,
    is_live: bool,
) -> dict[str, Any]:
    """Live consumers must not see paper-inflated performance_equity as live equity.

    cash_plus_positions_equity is the live account mark. performance_equity equals
    that mark in live mode. Paper realized stays in paper_realized_pnl only.
    """
    cash_plus = float(account_equity)
    if is_live:
        performance = cash_plus
    else:
        performance = float(principal) + float(paper_realized) + float(unrealized)
    return {
        "cash_plus_positions_equity": cash_plus,
        "performance_equity": performance,
        "live_realized_pnl": float(live_realized),
        "paper_realized_pnl": float(paper_realized),
        "dust_adjustment_pnl": float(dust_adjustment),
        "performance_equity_uses_live_account": bool(is_live),
    }


def recent_sell_pnls(
    db_path: str,
    symbol: str,
    *,
    limit: int = 10,
    mode: str | None = None,
) -> list[float]:
    """Most recent completed SELL pnls for a symbol, optionally mode-filtered."""
    alt = normalize_symbol(symbol).replace("/", "").upper()
    try:
        with sqlite3.connect(db_path, timeout=5) as conn:
            expr = _sell_pnl_expr(conn)
            sql_params: list[Any] = [alt]
            mode_sql = ""
            if mode:
                mode_sql = " AND LOWER(COALESCE(mode, '')) = ?"
                sql_params.append(str(mode).strip().lower())
            sql_params.append(int(limit))
            rows = conn.execute(
                f"""
                SELECT {expr}
                FROM paper_trades
                WHERE UPPER(side)='SELL'
                  AND pnl IS NOT NULL
                  AND COALESCE(is_synthetic, 0) = 0
                  AND REPLACE(REPLACE(UPPER(symbol), '/', ''), '_', '') = ?
                  {mode_sql}
                ORDER BY timestamp DESC
                LIMIT ?
                """,
                sql_params,
            ).fetchall()
        return [float(r[0]) for r in rows if r and r[0] is not None]
    except sqlite3.Error:
        return []


def _sell_pnl_expr(conn: sqlite3.Connection) -> str:
    try:
        cols = {str(r[1]) for r in conn.execute("PRAGMA table_info(paper_trades)")}
    except sqlite3.Error:
        return "pnl"
    if "pnl_usd_net" in cols:
        return "COALESCE(pnl_usd_net, pnl)"
    return "pnl"


def live_risk_loss_hits(*, is_live_day: bool, sticky_hits: int, live_pnls: list[float], window: int = 10) -> int:
    """LIVE uses true rolling last-N. Paper keeps existing sticky counter."""
    if is_live_day:
        return rolling_loss_count(live_pnls, window)
    return int(sticky_hits)


def live_closes_24h(db_path: str, symbol: str) -> tuple[int, float]:
    """Live-mode SELL count and pnl sum in the last 24h."""
    alt = normalize_symbol(symbol).replace("/", "").upper()
    try:
        with sqlite3.connect(db_path, timeout=5) as conn:
            expr = _sell_pnl_expr(conn)
            row = conn.execute(
                f"""
                SELECT COUNT(*), COALESCE(SUM({expr}), 0)
                FROM paper_trades
                WHERE UPPER(side)='SELL'
                  AND pnl IS NOT NULL
                  AND COALESCE(is_synthetic, 0) = 0
                  AND LOWER(COALESCE(mode, '')) = 'live'
                  AND REPLACE(REPLACE(UPPER(symbol), '/', ''), '_', '') = ?
                  AND timestamp >= datetime('now', '-24 hours')
                """,
                (alt,),
            ).fetchone()
        return int((row or (0, 0.0))[0] or 0), float((row or (0, 0.0))[1] or 0.0)
    except sqlite3.Error:
        return 0, 0.0


def rolling_loss_count(pnls: list[float], window: int = 10) -> int:
    """True last-N losses. Wins in the window reduce the count."""
    slice_ = list(pnls)[: max(1, int(window))]
    return sum(1 for p in slice_ if float(p) < 0)
