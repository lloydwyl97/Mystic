"""Durable strategy identity for a live order, written before the venue can fill.

Binance.US client order ids are at most 36 characters. The id stored here is 32
hex characters so the execution layer can append its ``-i`` / ``-g`` attempt
suffix. The same engine, trade id, and side always mint the same id.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
import time
from typing import Any

_SCHEMA = """
CREATE TABLE IF NOT EXISTS strategy_order_intents (
    client_order_id TEXT PRIMARY KEY,
    engine_id TEXT NOT NULL,
    trade_id TEXT NOT NULL,
    decision_id TEXT NOT NULL DEFAULT '',
    symbol TEXT NOT NULL,
    side TEXT NOT NULL,
    quantity REAL NOT NULL,
    limit_price REAL NOT NULL DEFAULT 0,
    setup TEXT NOT NULL DEFAULT '',
    regime TEXT NOT NULL DEFAULT '',
    feature_version TEXT NOT NULL DEFAULT '',
    exchange_order_id TEXT NOT NULL DEFAULT '',
    status TEXT NOT NULL DEFAULT 'PENDING',
    filled_qty REAL NOT NULL DEFAULT 0,
    filled_cost REAL NOT NULL DEFAULT 0,
    filled_fee REAL NOT NULL DEFAULT 0,
    fill_ids_json TEXT NOT NULL DEFAULT '[]',
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL
)
"""


def _connect(db_path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path, timeout=15)
    conn.row_factory = sqlite3.Row
    conn.execute(_SCHEMA)
    # One trade can have a later remainder order. The client id is the idempotency key.
    conn.execute("DROP INDEX IF EXISTS ux_strategy_intent_trade")
    cols = {str(row[1]) for row in conn.execute("PRAGMA table_info(strategy_order_intents)")}
    if "filled_gross" not in cols:
        conn.execute("ALTER TABLE strategy_order_intents ADD COLUMN filled_gross REAL NOT NULL DEFAULT 0")
    if "filled_cash" not in cols:
        conn.execute("ALTER TABLE strategy_order_intents ADD COLUMN filled_cash REAL NOT NULL DEFAULT 0")
    for name in ("pending_high", "pending_low", "booked_cash", "booked_high", "booked_low", "booked_fee", "venue_executed_qty"):
        if name not in cols:
            conn.execute(f"ALTER TABLE strategy_order_intents ADD COLUMN {name} REAL NOT NULL DEFAULT 0")
    if "deferred_close_json" not in cols:
        conn.execute("ALTER TABLE strategy_order_intents ADD COLUMN deferred_close_json TEXT NOT NULL DEFAULT '[]'")
    return conn


def client_order_id_for(*, engine_id: str, trade_id: str, side: str, attempt: int = 0) -> str:
    """32-char hex id. Stable for one strategy order attempt. Fits under the 36-char venue limit."""
    raw = f"{engine_id}|{trade_id}|{side}|{int(attempt)}".encode()
    return hashlib.sha256(raw).hexdigest()[:32]


def allocate_intent(
    db_path: str,
    *,
    engine_id: str,
    trade_id: str,
    decision_id: str = "",
    symbol: str,
    side: str,
    quantity: float,
    limit_price: float = 0.0,
    setup: str = "",
    regime: str = "",
    feature_version: str = "",
) -> str:
    """Persist strategy identity before submit. An in-flight order keeps its client id.

    A later order for the same trade, after the first one has a fill, gets a new id.
    Repeating the call before any fill returns the original id.
    """
    now = time.time()
    side_u = str(side).upper()
    conn = _connect(db_path)
    try:
        existing = conn.execute(
            """
            SELECT client_order_id, status, filled_qty
            FROM strategy_order_intents
            WHERE engine_id=? AND trade_id=? AND side=?
            ORDER BY created_at
            """,
            (str(engine_id), str(trade_id), side_u),
        ).fetchall()
        for prior in existing:
            if str(prior["status"] or "") in _WORKING_SELL_STATUSES:
                return str(prior["client_order_id"])
        cid = client_order_id_for(engine_id=engine_id, trade_id=trade_id, side=side_u, attempt=len(existing))
        conn.execute(
            """
            INSERT INTO strategy_order_intents (
                client_order_id, engine_id, trade_id, decision_id, symbol, side,
                quantity, limit_price, setup, regime, feature_version, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(client_order_id) DO NOTHING
            """,
            (
                cid,
                str(engine_id),
                str(trade_id),
                str(decision_id or ""),
                str(symbol),
                side_u,
                float(quantity),
                float(limit_price or 0.0),
                str(setup or ""),
                str(regime or ""),
                str(feature_version or ""),
                now,
                now,
            ),
        )
        conn.commit()
    finally:
        conn.close()
    return cid


def mark_exchange_order(db_path: str, client_order_id: str, exchange_order_id: str, *, status: str) -> None:
    conn = _connect(db_path)
    try:
        conn.execute(
            """
            UPDATE strategy_order_intents
            SET exchange_order_id=?, status=?, updated_at=?
            WHERE client_order_id=?
            """,
            (str(exchange_order_id or ""), str(status), time.time(), normalize_client_order_id(client_order_id)),
        )
        conn.commit()
    finally:
        conn.close()


def normalize_client_order_id(raw: str) -> str:
    """Strip the execution layer's ``-i`` / ``-g`` attempt suffix."""
    text = str(raw or "")
    if text.endswith("-i") or text.endswith("-g"):
        return text[:-2]
    return text


def intent_by_client(db_path: str, client_order_id: str) -> dict[str, Any] | None:
    conn = _connect(db_path)
    try:
        row = conn.execute(
            "SELECT * FROM strategy_order_intents WHERE client_order_id=?",
            (normalize_client_order_id(client_order_id),),
        ).fetchone()
    finally:
        conn.close()
    return dict(row) if row else None


def _fill_ids(raw: str) -> set[str]:
    try:
        return {str(x) for x in json.loads(raw or "[]")}
    except (TypeError, ValueError):
        return set()


def adopt_venue_fill(
    db_path: str,
    *,
    client_order_id: str,
    exchange_order_id: str,
    fill_id: str,
    quantity: float,
    price: float,
    fee: float = 0.0,
    fee_asset: str = "",
    quote_fee: float = 0.0,
) -> dict[str, Any]:
    """Apply one venue fill to its pre-submit strategy intent. The same fill id is a no-op.

    A BUY position quantity is gross base minus base-asset commission. The price
    stays the execution price of the gross fill. A SELL quantity is the base sold.
    """
    from backend.services.filled_asset_custody import ensure_canonical_buy_row

    cid = normalize_client_order_id(client_order_id)
    conn = _connect(db_path)
    try:
        row = conn.execute("SELECT * FROM strategy_order_intents WHERE client_order_id=?", (cid,)).fetchone()
        if row is None:
            return {"adopted": False, "reason": "NO_INTENT"}
        seen = _fill_ids(str(row["fill_ids_json"] or "[]"))
        fid = str(fill_id or "")
        aggregate = fid.startswith("order:")
        per_trade_seen = any(item and not item.startswith("order:") for item in seen)
        aggregate_seen = any(item.startswith("order:") for item in seen)
        # An order-level snapshot and the venue's per-trade legs count the same coins once.
        if (aggregate and per_trade_seen) or (fid and not aggregate and aggregate_seen) or (fid and fid in seen):
            return {"adopted": False, "reason": "DUPLICATE_FILL", "trade_id": row["trade_id"], "filled_qty": float(row["filled_qty"])}
        qty = float(quantity or 0.0)
        px = float(price or 0.0)
        if qty <= 0 or px <= 0:
            return {"adopted": False, "reason": "EMPTY_FILL"}
        from backend.services.live_fill_economics import QUOTE_FEE_ASSETS, _base_asset, canonical_buy_economics

        side = str(row["side"] or "").upper()
        asset = str(fee_asset or "").upper()
        base_fee = 0.0
        qfee = float(quote_fee or 0.0)
        legacy_usd = 0.0
        if side == "BUY":
            base_name = _base_asset(str(row["symbol"] or ""))
            if asset == base_name:
                base_fee = float(fee or 0.0)
            elif asset in QUOTE_FEE_ASSETS:
                qfee += float(fee or 0.0)
            elif not asset:
                legacy_usd = float(fee or 0.0)
            leg = canonical_buy_economics(gross_qty=qty, price=px, base_fee=base_fee, quote_fee=qfee)
            add_net = leg["net_qty"]
            add_gross = leg["gross_qty"]
            add_quote = leg["quote_notional"]
            add_fee = leg["fee_usd"] + legacy_usd
            add_cash = leg["cash_debit"]
        else:
            add_net = qty
            add_gross = qty
            add_quote = qty * px
            add_fee = float(fee or 0.0) + qfee
            add_cash = add_quote
        if aggregate:
            filled_qty = add_net
            filled_gross = add_gross
            filled_cost = add_quote
            filled_fee = add_fee
            filled_cash = add_cash
        else:
            filled_qty = float(row["filled_qty"] or 0.0) + add_net
            filled_gross = float(row["filled_gross"] or 0.0) + add_gross
            filled_cost = float(row["filled_cost"] or 0.0) + add_quote
            filled_fee = float(row["filled_fee"] or 0.0) + add_fee
            filled_cash = float(row["filled_cash"] or 0.0) + add_cash
        seen.add(fid or f"{exchange_order_id}:{filled_qty}")
        vwap = filled_cost / filled_gross if filled_gross else 0.0
        prior_high = float(row["pending_high"] or 0.0)
        prior_low = float(row["pending_low"] or 0.0)
        pending_high = px if prior_high <= 0 else max(prior_high, px)
        pending_low = px if prior_low <= 0 else min(prior_low, px)
        conn.execute(
            """
            UPDATE strategy_order_intents
            SET exchange_order_id=?, status='FILLED', filled_qty=?, filled_gross=?, filled_cost=?,
                filled_fee=?, filled_cash=?, fill_ids_json=?, pending_high=?, pending_low=?, updated_at=?
            WHERE client_order_id=?
            """,
            (
                str(exchange_order_id or row["exchange_order_id"] or ""),
                filled_qty,
                filled_gross,
                filled_cost,
                filled_fee,
                filled_cash,
                json.dumps(sorted(seen)),
                pending_high,
                pending_low,
                time.time(),
                cid,
            ),
        )
        conn.commit()
        intent = dict(row)
    finally:
        conn.close()
    if str(intent["side"]).upper() == "BUY":
        ensure_canonical_buy_row(
            db_path,
            {
                "trade_id": intent["trade_id"],
                "exchange_order_id": str(exchange_order_id or ""),
                "symbol": intent["symbol"],
                "quantity": filled_qty,
                "fill_price": vwap,
                "fee": filled_fee,
                "engine_id": intent["engine_id"],
                "decision_id": intent["decision_id"],
            },
        )
        # Keep the canonical row aligned if a later partial arrives.
        fix = sqlite3.connect(db_path, timeout=15)
        try:
            cols = {str(r[1]) for r in fix.execute("PRAGMA table_info(paper_trades)")}
            held = None
            try:
                held = fix.execute(
                    "SELECT 1 FROM portfolio_engine_positions WHERE trade_id=? AND IFNULL(quantity, 0) > 0",
                    (intent["trade_id"],),
                ).fetchone()
            except sqlite3.OperationalError:
                held = None
            closed = False
            if "remaining_position" in cols and "quantity" in cols:
                paper = fix.execute(
                    "SELECT quantity, remaining_position FROM paper_trades WHERE trade_id=? AND UPPER(side)='BUY'",
                    (intent["trade_id"],),
                ).fetchone()
                if paper is not None and abs(float(paper[1] or 0.0)) <= 1e-12 and float(paper[0] or 0.0) > 0:
                    closed = True
            # A live lot, or a lot already sold down to zero, owns the booked
            # quantity. Updating that baseline first hides the next fill's delta.
            if held is None and not closed:
                sets = ["quantity=?", "price=?"]
                params: list[Any] = [filled_qty, vwap]
                if "remaining_position" in cols:
                    sets.append("remaining_position=CASE WHEN ABS(COALESCE(remaining_position, quantity) - quantity) < 1e-8 THEN ? ELSE remaining_position END")
                    params.append(filled_qty)
            elif not closed:
                sets = ["price=?"]
                params = [vwap]
            else:
                sets = []
                params = []
            if "fees_paid" in cols:
                sets.append("fees_paid=?")
                params.append(filled_fee)
            if "order_id" in cols:
                sets.append("order_id=?")
                params.append(str(exchange_order_id or ""))
            if "decision_id" in cols and intent.get("decision_id"):
                sets.append("decision_id=COALESCE(NULLIF(decision_id, ''), ?)")
                params.append(str(intent["decision_id"]))
            if "explainability_json" in cols:
                sets.append("explainability_json=CASE WHEN COALESCE(explainability_json, '') = '' THEN ? ELSE explainability_json END")
                params.append(
                    json.dumps(
                        {
                            "trade_id": intent["trade_id"],
                            "setup_type": intent.get("setup") or "",
                            "regime": intent.get("regime") or "",
                            "feature_version": intent.get("feature_version") or "",
                            "decision_id": intent.get("decision_id") or "",
                            "engine_id": intent.get("engine_id") or "",
                            "source": "strategy_order_intent",
                        },
                        separators=(",", ":"),
                    )
                )
            if sets:
                params.append(intent["trade_id"])
                fix.execute(f"UPDATE paper_trades SET {', '.join(sets)} WHERE trade_id=? AND UPPER(side)='BUY'", params)
            fix.commit()
        finally:
            fix.close()
    return {
        "adopted": True,
        "trade_id": intent["trade_id"],
        "engine_id": intent["engine_id"],
        "decision_id": intent["decision_id"],
        "setup": intent["setup"],
        "regime": intent["regime"],
        "feature_version": intent["feature_version"],
        "filled_qty": filled_qty,
        "gross_qty": filled_gross,
        "vwap": vwap,
        "fee": filled_fee,
        "cash_debit": filled_cash,
    }


def intent_by_trade(db_path: str, trade_id: str, side: str) -> dict[str, Any] | None:
    conn = _connect(db_path)
    try:
        row = conn.execute(
            """
            SELECT * FROM strategy_order_intents
            WHERE trade_id=? AND side=?
            ORDER BY created_at DESC LIMIT 1
            """,
            (str(trade_id), str(side).upper()),
        ).fetchone()
    finally:
        conn.close()
    return dict(row) if row else None


def intent_by_exchange_order(db_path: str, exchange_order_id: str, side: str = "") -> dict[str, Any] | None:
    oid = str(exchange_order_id or "").strip()
    if not oid:
        return None
    conn = _connect(db_path)
    try:
        if side:
            row = conn.execute(
                "SELECT * FROM strategy_order_intents WHERE exchange_order_id=? AND side=? ORDER BY created_at DESC LIMIT 1",
                (oid, str(side).upper()),
            ).fetchone()
        else:
            row = conn.execute(
                "SELECT * FROM strategy_order_intents WHERE exchange_order_id=? ORDER BY created_at DESC LIMIT 1",
                (oid,),
            ).fetchone()
    finally:
        conn.close()
    return dict(row) if row else None


def complete_intent(db_path: str, *, trade_id: str, side: str, exchange_order_id: str = "", status: str = "COMMITTED") -> None:
    conn = _connect(db_path)
    try:
        if exchange_order_id:
            conn.execute(
                """
                UPDATE strategy_order_intents
                SET status=?, updated_at=?
                WHERE trade_id=? AND side=? AND exchange_order_id=?
                """,
                (status, time.time(), str(trade_id), str(side).upper(), str(exchange_order_id)),
            )
        else:
            conn.execute(
                """
                UPDATE strategy_order_intents
                SET status=?, updated_at=?
                WHERE trade_id=? AND side=? AND status NOT IN ('NO_VENUE_ORDER', 'CLOSED')
                """,
                (status, time.time(), str(trade_id), str(side).upper()),
            )
        conn.commit()
    finally:
        conn.close()


def claim_strategy_fill(
    db_path: str,
    *,
    exchange_order_id: str,
    client_order_id: str = "",
    fill_id: str,
    quantity: float,
    price: float,
    fee: float = 0.0,
    fee_asset: str = "",
    side: str = "",
) -> dict[str, Any]:
    """Adopt a venue fill when a pre-submit intent owns it. Otherwise leave it unclassified."""
    intent = intent_by_client(db_path, client_order_id) if client_order_id else None
    if intent is None:
        intent = intent_by_exchange_order(db_path, exchange_order_id, side)
    if intent is None:
        return {"adopted": False, "reason": "NO_INTENT"}
    if side and str(intent.get("side") or "").upper() != str(side).upper():
        return {"adopted": False, "reason": "NO_INTENT"}
    return adopt_venue_fill(
        db_path,
        client_order_id=str(intent["client_order_id"]),
        exchange_order_id=str(exchange_order_id or intent.get("exchange_order_id") or ""),
        fill_id=fill_id,
        quantity=quantity,
        price=price,
        fee=fee,
        fee_asset=fee_asset,
    )


def _trade_fee(trade: dict[str, Any]) -> tuple[float, str]:
    fee_raw = trade.get("fee") if isinstance(trade.get("fee"), dict) else {}
    amount = trade.get("commission")
    if amount is None:
        amount = fee_raw.get("cost")
    asset = str(trade.get("commissionAsset") or trade.get("commission_asset") or fee_raw.get("currency") or "").upper()
    try:
        qty = float(amount or 0.0)
    except (TypeError, ValueError):
        qty = 0.0
    return qty, asset


def apply_venue_order(db_path: str, order: dict[str, Any]) -> dict[str, Any]:
    """Apply a fetched venue order to the intent named by its client id.

    A summary that has executed quantity but no commission asset is not booked.
    The caller fetches myTrades and calls again. Gross executedQty is never
    treated as the position size when the fee asset is unknown.
    """
    cid = str(order.get("clientOrderId") or order.get("client_order_id") or "")
    oid = str(order.get("id") or "")
    info = order.get("info") if isinstance(order.get("info"), dict) else {}
    trades = order.get("trades") or info.get("fills") or []
    last: dict[str, Any] = {"adopted": False, "reason": "NO_FILL"}
    if trades:
        for index, trade in enumerate(trades):
            if not isinstance(trade, dict):
                continue
            try:
                qty = float(trade.get("qty") or trade.get("amount") or 0.0)
                px = float(trade.get("price") or 0.0)
            except (TypeError, ValueError):
                continue
            fee, asset = _trade_fee(trade)
            fid = str(trade.get("tradeId") or trade.get("trade_id") or trade.get("id") or f"{oid}:{index}")
            last = adopt_venue_fill(
                db_path,
                client_order_id=cid,
                exchange_order_id=oid,
                fill_id=fid,
                quantity=qty,
                price=px,
                fee=fee,
                fee_asset=asset,
            )
        return _finish_venue_lifecycle(db_path, order, last)
    filled = float(order.get("filled") or 0.0)
    average = float(order.get("average") or order.get("price") or 0.0)
    fee_raw = order.get("fee") if isinstance(order.get("fee"), dict) else {}
    asset = str(fee_raw.get("currency") or order.get("commissionAsset") or info.get("commissionAsset") or "").upper()
    if filled > 0 and average > 0 and oid and asset:
        try:
            fee = float(fee_raw.get("cost") if fee_raw.get("cost") is not None else order.get("commission") or info.get("commission") or 0.0)
        except (TypeError, ValueError):
            fee = 0.0
        return _finish_venue_lifecycle(
            db_path,
            order,
            adopt_venue_fill(
                db_path,
                client_order_id=cid,
                exchange_order_id=oid,
                fill_id=f"order:{oid}",
                quantity=filled,
                price=average,
                fee=fee,
                fee_asset=asset,
            ),
        )
    if filled > 0 and not asset:
        return _finish_venue_lifecycle(
            db_path,
            order,
            {"adopted": False, "reason": "NEEDS_TRADES", "exchange_order_id": oid, "client_order_id": normalize_client_order_id(cid)},
        )
    if oid and cid and order_is_working(order):
        mark_exchange_order(db_path, normalize_client_order_id(cid), oid, status="OPEN")
        return {"adopted": False, "reason": "NO_FILL", "exchange_order_id": oid, "client_order_id": normalize_client_order_id(cid)}
    return _finish_venue_lifecycle(db_path, order, last)


def intents_to_probe(db_path: str, *, max_age_sec: float = 172800, limit: int = 16) -> list[dict[str, Any]]:
    """Recent orders with no fill yet. The venue is the only place their result can be."""
    cutoff = time.time() - float(max_age_sec)
    conn = _connect(db_path)
    try:
        rows = conn.execute(
            """
            SELECT * FROM strategy_order_intents
            WHERE status IN ('PENDING', 'ACKED') AND filled_qty <= 0 AND created_at >= ?
            ORDER BY created_at DESC
            LIMIT ?
            """,
            (cutoff, int(limit)),
        ).fetchall()
    finally:
        conn.close()
    return [dict(row) for row in rows]


_WORKING_SELL_STATUSES = ("PENDING", "ACKED", "OPEN", "PARTIAL", "WORKING")
# EXPIRED_IN_MATCH is Binance self-trade prevention. CCXT maps it to expired;
# a raw payload keeps the original status and is just as terminal.
_TERMINAL_VENUE_STATUSES = {"canceled", "cancelled", "expired", "expired_in_match", "rejected", "closed", "filled"}


def venue_order_status(order: dict[str, Any] | None) -> str:
    if not isinstance(order, dict):
        return ""
    return str(order.get("status") or "").strip().lower()


def order_is_working(order: dict[str, Any] | None) -> bool:
    """True while the venue can still fill this order. An unknown status stays working."""
    return venue_order_status(order) not in _TERMINAL_VENUE_STATUSES


def terminal_unfilled_label(order: dict[str, Any] | None, filled: float) -> str | None:
    """A venue order that can accept no more size and has no fill."""
    if float(filled or 0.0) > 0.0 or order_is_working(order):
        return None
    status = venue_order_status(order)
    if status in {"expired", "expired_in_match"}:
        return "EXPIRED"
    if status == "rejected":
        return "REJECTED"
    if status in {"canceled", "cancelled", "closed", "filled"}:
        return "CANCELED"
    return None


def release_unfilled_intent(db_path: str, client_order_id: str, status: str, exchange_order_id: str = "") -> None:
    """A confirmed dead order must not keep its client id or block the next protective sell."""
    conn = _connect(db_path)
    try:
        conn.execute(
            f"""
            UPDATE strategy_order_intents
            SET status=?, exchange_order_id=CASE WHEN ? != '' THEN ? ELSE exchange_order_id END, updated_at=?
            WHERE client_order_id=? AND filled_qty <= 0 AND status IN ({", ".join("?" for _ in _WORKING_SELL_STATUSES)})
            """,
            (
                str(status),
                str(exchange_order_id or ""),
                str(exchange_order_id or ""),
                time.time(),
                normalize_client_order_id(client_order_id),
                *_WORKING_SELL_STATUSES,
            ),
        )
        conn.commit()
    finally:
        conn.close()


def mark_working_sell(db_path: str, client_order_id: str, status: str, exchange_order_id: str = "") -> None:
    """The venue order can still fill. A just-adopted FILLED row is put back to PARTIAL."""
    conn = _connect(db_path)
    try:
        conn.execute(
            """
            UPDATE strategy_order_intents
            SET status=?, exchange_order_id=CASE WHEN ? != '' THEN ? ELSE exchange_order_id END, updated_at=?
            WHERE client_order_id=? AND status NOT IN ('CLOSED', 'COMMITTED', 'CANCELED', 'EXPIRED', 'REJECTED', 'NO_VENUE_ORDER')
            """,
            (
                str(status),
                str(exchange_order_id or ""),
                str(exchange_order_id or ""),
                time.time(),
                normalize_client_order_id(client_order_id),
            ),
        )
        conn.commit()
    finally:
        conn.close()


def _note_venue_executed_qty(db_path: str, client_order_id: str, order: dict[str, Any]) -> None:
    """Remember the venue's gross executedQty without treating it as canonical net base."""
    if not client_order_id:
        return
    try:
        venue_qty = float(order.get("filled") or 0.0)
    except (TypeError, ValueError):
        venue_qty = 0.0
    if venue_qty <= 0:
        return
    conn = _connect(db_path)
    try:
        conn.execute(
            """
            UPDATE strategy_order_intents
            SET venue_executed_qty=CASE WHEN ? > venue_executed_qty THEN ? ELSE venue_executed_qty END
            WHERE client_order_id=? AND side='BUY'
            """,
            (venue_qty, venue_qty, normalize_client_order_id(client_order_id)),
        )
        conn.commit()
    finally:
        conn.close()


def _finish_venue_lifecycle(db_path: str, order: dict[str, Any], result: dict[str, Any]) -> dict[str, Any]:
    """Record whether the venue order is still able to fill. A dead zero-fill releases the intent."""
    cid = str(order.get("clientOrderId") or order.get("client_order_id") or "")
    oid = str(order.get("id") or "")
    try:
        filled = float(order.get("filled") or 0.0)
    except (TypeError, ValueError):
        filled = 0.0
    if isinstance(result, dict):
        try:
            filled = max(filled, float(result.get("filled_qty") or 0.0))
        except (TypeError, ValueError):
            pass
    label = terminal_unfilled_label(order, filled)
    if label:
        release_unfilled_intent(db_path, cid, label, oid)
        out = dict(result or {})
        out["released"] = True
        out["reason"] = label
        return out
    _note_venue_executed_qty(db_path, cid, order)
    if cid and filled > 0.0 and not order_is_working(order):
        # The resting remainder is dead. The executed base stays reserved
        # until its own close, and it no longer blocks the unsold base.
        # An executedQty that myTrades does not yet cover is not copied in.
        from backend.services.live_fill_economics import proven_sell_quote, sell_commission_known

        known = sell_commission_known(order)
        status = "FILLED" if known else FEE_PENDING_STATUS
        conn = _connect(db_path)
        try:
            if known:
                conn.execute(
                    """
                    UPDATE strategy_order_intents
                    SET exchange_order_id=CASE WHEN ? != '' THEN ? ELSE exchange_order_id END,
                        status=?, filled_qty=?, filled_cost=CASE WHEN ? > 0 THEN ? ELSE filled_cost END, updated_at=?
                    WHERE client_order_id=? AND status NOT IN ('CLOSED', 'COMMITTED', 'CANCELED', 'EXPIRED', 'REJECTED', 'NO_VENUE_ORDER')
                    """,
                    (oid, oid, status, filled, proven_sell_quote(order), proven_sell_quote(order), time.time(), normalize_client_order_id(cid)),
                )
            else:
                conn.execute(
                    """
                    UPDATE strategy_order_intents
                    SET exchange_order_id=CASE WHEN ? != '' THEN ? ELSE exchange_order_id END,
                        status=?, updated_at=?
                    WHERE client_order_id=? AND status NOT IN ('CLOSED', 'COMMITTED', 'CANCELED', 'EXPIRED', 'REJECTED', 'NO_VENUE_ORDER')
                    """,
                    (oid, oid, status, time.time(), normalize_client_order_id(cid)),
                )
            conn.commit()
        finally:
            conn.close()
        return result
    if order_is_working(order) and cid:
        mark_working_sell(db_path, cid, "PARTIAL" if filled > 0.0 else "OPEN", oid)
    return result


def unresolved_buy_on_symbol(db_path: str, symbol: str) -> bool:
    """A BUY on this symbol can still explain venue base. It is not another engine's lot."""
    if working_buy_intents(db_path, symbol):
        return True
    from backend.utils.symbols import normalize_symbol

    want = normalize_symbol(symbol)
    conn = _connect(db_path)
    try:
        row = conn.execute(
            """
            SELECT symbol FROM strategy_order_intents
            WHERE side='BUY' AND status='FEE_PENDING' AND filled_qty > 0
            """
        ).fetchall()
    except sqlite3.OperationalError:
        return False
    finally:
        conn.close()
    return any(normalize_symbol(str(item["symbol"] or "")) == want for item in row)


def note_open_buy_basis(
    db_path: str,
    trade_id: str,
    *,
    cash: float = 0.0,
    high: float = 0.0,
    low: float = 0.0,
    fee: float = 0.0,
) -> None:
    """Remember the basis already booked, so a later fill can debit only its own delta."""
    if not trade_id:
        return
    conn = _connect(db_path)
    try:
        conn.execute(
            """
            UPDATE strategy_order_intents
            SET booked_cash=CASE WHEN ? > booked_cash THEN ? ELSE booked_cash END,
                booked_high=CASE WHEN ? > booked_high THEN ? ELSE booked_high END,
                booked_low=CASE WHEN booked_low <= 0 THEN ? WHEN ? > 0 THEN MIN(booked_low, ?) ELSE booked_low END,
                booked_fee=CASE WHEN ? > booked_fee THEN ? ELSE booked_fee END
            WHERE trade_id=? AND side='BUY'
            """,
            (
                float(cash or 0.0),
                float(cash or 0.0),
                float(high or 0.0),
                float(high or 0.0),
                float(low or 0.0),
                float(low or 0.0),
                float(low or 0.0),
                float(fee or 0.0),
                float(fee or 0.0),
                str(trade_id),
            ),
        )
        conn.commit()
    finally:
        conn.close()


def _paper_buy_quantity(db_path: str, trade_id: str) -> float:
    conn = sqlite3.connect(db_path, timeout=15)
    try:
        tables = {str(row[0]) for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        if "paper_trades" not in tables:
            return 0.0
        row = conn.execute(
            "SELECT quantity FROM paper_trades WHERE trade_id=? AND UPPER(side)='BUY' LIMIT 1",
            (str(trade_id),),
        ).fetchone()
    except sqlite3.OperationalError:
        return 0.0
    finally:
        conn.close()
    return float(row[0] or 0.0) if row is not None else 0.0


def buy_intents_to_reread(db_path: str, symbol: str | None = None) -> list[dict[str, Any]]:
    """Working buys, plus a terminal buy whose commission is not proven yet.

    A zero-fill OPEN order stays on the ordinary working list. FEE_PENDING is
    not a live order; it is reread so a missing trade leg can still be booked.
    """
    rows = working_buy_intents(db_path, symbol)
    seen = {str(row.get("client_order_id") or "") for row in rows}
    from backend.utils.symbols import normalize_symbol

    conn = _connect(db_path)
    try:
        extra = conn.execute(
            """
            SELECT * FROM strategy_order_intents
            WHERE side='BUY' AND status='FEE_PENDING' AND filled_qty > 0
            ORDER BY created_at
            """
        ).fetchall()
    finally:
        conn.close()
    want = normalize_symbol(symbol) if symbol else ""
    for row in extra:
        item = dict(row)
        cid = str(item.get("client_order_id") or "")
        if cid in seen:
            continue
        if want and normalize_symbol(str(item.get("symbol") or "")) != want:
            continue
        rows.append(item)
    return rows


def working_buy_intents(db_path: str, symbol: str | None = None) -> list[dict[str, Any]]:
    """BUY intents that can still fill. No age cutoff."""
    from backend.utils.symbols import normalize_symbol

    conn = _connect(db_path)
    try:
        rows = conn.execute(
            f"""
            SELECT * FROM strategy_order_intents
            WHERE side='BUY' AND status IN ({", ".join("?" for _ in _WORKING_SELL_STATUSES)})
            ORDER BY created_at
            """,
            _WORKING_SELL_STATUSES,
        ).fetchall()
    finally:
        conn.close()
    out = [dict(row) for row in rows]
    if not symbol:
        return out
    want = normalize_symbol(symbol)
    return [row for row in out if normalize_symbol(str(row.get("symbol") or "")) == want]


def working_sell_intents(db_path: str, symbol: str | None = None) -> list[dict[str, Any]]:
    """SELL intents that can still fill or have not been classified. No age cutoff."""
    from backend.utils.symbols import normalize_symbol

    conn = _connect(db_path)
    try:
        rows = conn.execute(
            f"""
            SELECT * FROM strategy_order_intents
            WHERE side='SELL' AND status IN ({", ".join("?" for _ in _WORKING_SELL_STATUSES)})
            ORDER BY created_at
            """,
            _WORKING_SELL_STATUSES,
        ).fetchall()
    finally:
        conn.close()
    out = [dict(row) for row in rows]
    if not symbol:
        return out
    want = normalize_symbol(symbol)
    return [row for row in out if normalize_symbol(str(row.get("symbol") or "")) == want]


def sell_intent_blocks(db_path: str, symbol: str) -> bool:
    """True while a SELL for this symbol is submitted and its venue result is not final."""
    return bool(working_sell_intents(db_path, symbol))


def sell_protection_snapshot(db_path: str) -> dict[str, Any]:
    """Operational view of working protective sells. No strategy decision."""
    try:
        rows = working_sell_intents(db_path)
    except Exception:
        return {
            "active_sell_intents": 0,
            "oldest_sell_intent_age_sec": None,
            "partial_sell_count": 0,
            "remaining_sell_qty": 0.0,
            "venue_order_status": [],
            "fee_pending_qty": 0.0,
            "executed_unbooked_sell_qty": 0.0,
            "working_sell_qty": 0.0,
            "fee_pending_count": 0,
            "working_sells": [],
        }
    now = time.time()
    ages = [max(0.0, now - float(row.get("created_at") or now)) for row in rows]
    partial = 0
    remaining = 0.0
    for row in rows:
        filled = float(row.get("filled_qty") or 0.0)
        if str(row.get("status") or "") == "PARTIAL" or filled > 0.0:
            partial += 1
        remaining += max(0.0, float(row.get("quantity") or 0.0) - filled)
    executed = 0.0
    working = 0.0
    fee_pending_qty = 0.0
    fee_pending_count = 0
    oldest_fee = None
    try:
        pending_rows = _unbooked_sell_rows(db_path)
    except Exception:
        pending_rows = []
    for row in pending_rows:
        sold, resting = _row_sell_parts(db_path, row)
        executed += sold
        working += resting
        if str(row["status"] or "") == FEE_PENDING_STATUS and sold > 0.0:
            fee_pending_qty += sold
            fee_pending_count += 1
            age = max(0.0, now - float(row["created_at"] or now))
            oldest_fee = age if oldest_fee is None else max(oldest_fee, age)
    return {
        "active_sell_intents": len(rows),
        "oldest_sell_intent_age_sec": max(ages) if ages else None,
        "partial_sell_count": partial,
        "remaining_sell_qty": remaining,
        "venue_order_status": [str(row.get("status") or "") for row in rows],
        "limit_prices": [float(row.get("limit_price") or 0.0) for row in rows],
        "working_sells": [
            {
                "engine_id": str(row.get("engine_id") or ""),
                "trade_id": str(row.get("trade_id") or ""),
                "symbol": str(row.get("symbol") or ""),
                "status": str(row.get("status") or ""),
                "age_sec": max(0.0, now - float(row.get("created_at") or now)),
                "quantity": float(row.get("quantity") or 0.0),
                "filled_qty": float(row.get("filled_qty") or 0.0),
                "remaining_qty": max(0.0, float(row.get("quantity") or 0.0) - float(row.get("filled_qty") or 0.0)),
                "limit_price": float(row.get("limit_price") or 0.0),
                "exit_reason": None,
            }
            for row in rows
        ],
        "fee_pending_qty": fee_pending_qty,
        "fee_pending_count": fee_pending_count,
        "oldest_fee_pending_age_sec": oldest_fee,
        "executed_unbooked_sell_qty": executed,
        "working_sell_qty": working,
        "cash_sync_held": executed > 0.0,
    }


def mark_intent_absent(db_path: str, client_order_id: str) -> None:
    conn = _connect(db_path)
    try:
        conn.execute(
            "UPDATE strategy_order_intents SET status='NO_VENUE_ORDER', updated_at=? WHERE client_order_id=? AND filled_qty <= 0",
            (time.time(), normalize_client_order_id(client_order_id)),
        )
        conn.commit()
    finally:
        conn.close()


def _buy_was_closed(conn: sqlite3.Connection, trade_id: str) -> bool:
    tables = {str(row[0]) for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    if "paper_trades" not in tables:
        return False
    cols = {str(row[1]) for row in conn.execute("PRAGMA table_info(paper_trades)")}
    if "remaining_position" not in cols:
        return False
    row = conn.execute(
        "SELECT quantity, remaining_position FROM paper_trades WHERE trade_id=? AND UPPER(side)='BUY' LIMIT 1",
        (str(trade_id),),
    ).fetchone()
    if row is None:
        return False
    return abs(float(row[1] or 0.0)) <= 1e-12 and float(row[0] or 0.0) > 0


def recover_recorded_fills(db_path: str) -> int:
    """Adopt fills already written to the fill ledger when their intent never reached custody.

    Only intents that are still open are considered, so a reconciliation pass does
    not walk the historical fill tape.
    """
    open_conn = _connect(db_path)
    try:
        pending = open_conn.execute(
            """
            SELECT trade_id, exchange_order_id, client_order_id, side
            FROM strategy_order_intents
            WHERE filled_qty <= 0 AND status IN ('PENDING', 'ACKED')
            """
        ).fetchall()
    finally:
        open_conn.close()
    if not pending:
        return 0
    order_ids = [str(row["exchange_order_id"]) for row in pending if str(row["exchange_order_id"] or "")]
    trade_ids = [str(row["trade_id"]) for row in pending if str(row["trade_id"] or "")]
    conn = sqlite3.connect(db_path, timeout=15)
    conn.row_factory = sqlite3.Row
    try:
        tables = {str(row[0]) for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        if "live_exchange_fills" not in tables:
            return 0
        clauses = []
        params: list[Any] = []
        if order_ids:
            clauses.append(f"exchange_order_id IN ({', '.join('?' for _ in order_ids)})")
            params.extend(order_ids)
        if trade_ids:
            clauses.append(f"mystic_trade_id IN ({', '.join('?' for _ in trade_ids)})")
            params.extend(trade_ids)
        if not clauses:
            return 0
        rows = conn.execute(
            f"""
            SELECT exchange_order_id, client_order_id, side, symbol, executed_qty, avg_fill_price,
                   fee_amount, fee_asset, fee_items_json, venue_trade_ids_json, mystic_trade_id
            FROM live_exchange_fills
            WHERE {" OR ".join(clauses)}
            """,
            params,
        ).fetchall()
    finally:
        conn.close()
    adopted = 0
    for raw in rows:
        row = dict(raw)
        side = str(row.get("side") or "").upper()
        intent = intent_by_client(db_path, str(row.get("client_order_id") or "")) if row.get("client_order_id") else None
        if intent is None:
            intent = intent_by_exchange_order(db_path, str(row.get("exchange_order_id") or ""), side)
        if intent is None and row.get("mystic_trade_id"):
            intent = intent_by_trade(db_path, str(row["mystic_trade_id"]), side)
        if intent is None or str(intent.get("side") or "").upper() != side:
            continue
        executed = float(row.get("executed_qty") or 0.0)
        already = float(intent.get("filled_gross") or intent.get("filled_qty") or 0.0)
        if executed <= 0 or already + 1e-12 >= executed:
            continue
        try:
            venue_ids = [str(item) for item in json.loads(row.get("venue_trade_ids_json") or "[]") if str(item)]
        except (TypeError, ValueError):
            venue_ids = []
        try:
            fee_items = json.loads(row.get("fee_items_json") or "[]")
        except (TypeError, ValueError):
            fee_items = []
        from backend.services.live_fill_economics import commission_parts

        parts = commission_parts(
            str(row.get("symbol") or intent.get("symbol") or ""),
            fee_items if isinstance(fee_items, list) else [],
            gross_qty=executed,
            fee_amount=float(row.get("fee_amount") or 0.0),
            fee_asset=str(row.get("fee_asset") or ""),
        )
        fill_id = venue_ids[0] if len(venue_ids) == 1 else f"order:{row.get('exchange_order_id')}"
        asset = ""
        fee = 0.0
        extra_quote = float(parts["quote_fee"] or 0.0)
        if float(parts["base_fee"] or 0.0) > 0 and side == "BUY":
            from backend.services.live_fill_economics import _base_asset

            asset = _base_asset(str(row.get("symbol") or intent.get("symbol") or ""))
            fee = float(parts["base_fee"])
        elif extra_quote > 0:
            asset = "USDT"
            fee = extra_quote
            extra_quote = 0.0
        result = adopt_venue_fill(
            db_path,
            client_order_id=str(intent["client_order_id"]),
            exchange_order_id=str(row.get("exchange_order_id") or ""),
            fill_id=fill_id,
            quantity=executed,
            price=float(row.get("avg_fill_price") or 0.0),
            fee=fee,
            fee_asset=asset,
            quote_fee=extra_quote,
        )
        if result.get("adopted"):
            adopted += 1
    return adopted


def materialize_buy_custody(db_path: str) -> int:
    """Write custody for an adopted BUY that never reached the open transaction.

    A lot whose paper row was already reduced to zero is closed and is not reopened.
    """
    from backend.services.filled_asset_custody import remember

    recover_recorded_fills(db_path)

    conn = _connect(db_path)
    try:
        rows = conn.execute(
            """
            SELECT * FROM strategy_order_intents
            WHERE side='BUY' AND filled_qty > 0
              AND status NOT IN ('COMMITTED', 'CLOSED', 'NO_VENUE_ORDER', 'FEE_PENDING')
            """
        ).fetchall()
        staged = [dict(row) for row in rows]
    finally:
        conn.close()
    written = 0
    for intent in staged:
        check = sqlite3.connect(db_path, timeout=15)
        try:
            tables = {str(row[0]) for row in check.execute("SELECT name FROM sqlite_master WHERE type='table'")}
            owned = False
            if "portfolio_engine_positions" in tables:
                owned = (
                    check.execute(
                        "SELECT 1 FROM portfolio_engine_positions WHERE trade_id=? AND IFNULL(quantity, 0) > 0",
                        (intent["trade_id"],),
                    ).fetchone()
                    is not None
                )
            closed = _buy_was_closed(check, str(intent["trade_id"]))
        finally:
            check.close()
        if owned:
            if str(intent.get("status") or "") in {"PENDING", "ACKED", "OPEN", "PARTIAL", "WORKING"}:
                continue
            complete_intent(db_path, trade_id=str(intent["trade_id"]), side="BUY", status="COMMITTED")
            continue
        if closed:
            booked = _paper_buy_quantity(db_path, str(intent["trade_id"]))
            if float(intent["filled_qty"] or 0.0) <= booked + 1e-8:
                complete_intent(db_path, trade_id=str(intent["trade_id"]), side="BUY", status="CLOSED")
            continue
        qty = float(intent["filled_qty"] or 0.0)
        gross = float(intent.get("filled_gross") or 0.0)
        quote = float(intent["filled_cost"] or 0.0)
        price = (quote / gross) if gross else ((quote / qty) if qty else 0.0)
        cash = float(intent.get("filled_cash") or 0.0) or quote
        remember(
            db_path,
            engine_id=str(intent["engine_id"]),
            symbol=str(intent["symbol"]),
            trade_id=str(intent["trade_id"]),
            quantity=qty,
            fill_price=price,
            fee=float(intent["filled_fee"] or 0.0),
            cash_debit=cash,
            client_order_id=str(intent["client_order_id"]),
            exchange_order_id=str(intent["exchange_order_id"] or ""),
        )
        written += 1
    return written


FEE_PENDING_STATUS = "FEE_PENDING"
_UNBOOKED_SELL_STATUSES = ("FEE_PENDING", "FILLED")


def _intent_quote_and_qty(order: dict[str, Any]) -> tuple[float, float]:
    from backend.services.live_fill_economics import proven_sell_base_qty, proven_sell_quote

    return proven_sell_base_qty(order), proven_sell_quote(order)


def record_sell_attempt_result(
    db_path: str,
    *,
    client_order_id: str,
    order: dict[str, Any] | None,
    outcome: str,
) -> str:
    """Keep a working id only while the venue result is still unknown.

    A zero-fill that the venue already closed releases the id. A fill whose
    commission asset is missing stays recoverable and does not block the next
    attempt's id. It is not a booked close.
    """
    from backend.services.live_fill_economics import sell_attempt_disposition

    cid = normalize_client_order_id(client_order_id)
    disposition = sell_attempt_disposition(order if isinstance(order, dict) else None, outcome)
    if not cid:
        return disposition
    if disposition == "release":
        label = "REJECTED" if str(outcome or "") == "EXCHANGE_REJECTED" else "EXPIRED" if str(outcome or "") == "EXPIRED_UNFILLED" else "CANCELED"
        release_unfilled_intent(db_path, cid, label, str((order or {}).get("id") or "") if isinstance(order, dict) else "")
        return disposition
    if disposition == "keep":
        return disposition
    qty, quote = _intent_quote_and_qty(order or {})
    status = FEE_PENDING_STATUS if disposition == "fee_pending" else "FILLED"
    conn = _connect(db_path)
    try:
        conn.execute(
            """
            UPDATE strategy_order_intents
            SET exchange_order_id=CASE WHEN ? != '' THEN ? ELSE exchange_order_id END,
                status=?, filled_qty=?, filled_cost=?, updated_at=?
            WHERE client_order_id=? AND status NOT IN ('CLOSED', 'COMMITTED', 'CANCELED', 'EXPIRED', 'REJECTED', 'NO_VENUE_ORDER')
            """,
            (
                str((order or {}).get("id") or ""),
                str((order or {}).get("id") or ""),
                status,
                qty,
                quote,
                time.time(),
                cid,
            ),
        )
        conn.commit()
    finally:
        conn.close()
    return disposition


def _unbooked_sell_rows(db_path: str, trade_id: str = "") -> list[sqlite3.Row]:
    conn = _connect(db_path)
    try:
        if trade_id:
            rows = conn.execute(
                """
                SELECT status, quantity, filled_qty, filled_cost, exchange_order_id, created_at
                FROM strategy_order_intents
                WHERE trade_id=? AND side='SELL'
                  AND status IN ('FEE_PENDING', 'FILLED', 'PENDING', 'ACKED', 'OPEN', 'PARTIAL', 'WORKING')
                """,
                (str(trade_id),),
            ).fetchall()
        else:
            rows = conn.execute(
                """
                SELECT status, quantity, filled_qty, filled_cost, exchange_order_id, created_at
                FROM strategy_order_intents
                WHERE side='SELL'
                  AND status IN ('FEE_PENDING', 'FILLED', 'PENDING', 'ACKED', 'OPEN', 'PARTIAL', 'WORKING')
                """
            ).fetchall()
    finally:
        conn.close()
    return list(rows)


def _row_sell_parts(db_path: str, row: sqlite3.Row) -> tuple[float, float]:
    """Executed-but-unbooked base, and base still resting on a live order."""
    status = str(row["status"] or "")
    oid = str(row["exchange_order_id"] or "")
    filled = max(0.0, float(row["filled_qty"] or 0.0))
    ordered = max(0.0, float(row["quantity"] or 0.0))
    if oid and sell_order_already_booked(db_path, oid):
        return 0.0, 0.0
    if status in _UNBOOKED_SELL_STATUSES:
        return filled, 0.0
    if status in _WORKING_SELL_STATUSES:
        # The resting remainder can still fill. The executed part is already sold.
        return filled, max(0.0, ordered - filled)
    return 0.0, 0.0


def sell_quantity_state(db_path: str, trade_id: str, ledger_qty: float) -> dict[str, float]:
    """Ledger quantity stays intact. Reservation is what the venue no longer offers."""
    executed = 0.0
    working = 0.0
    for row in _unbooked_sell_rows(db_path, trade_id):
        sold, resting = _row_sell_parts(db_path, row)
        executed += sold
        working += resting
    ledger = max(0.0, float(ledger_qty or 0.0))
    reserved = executed + working
    return {
        "ledger_qty": ledger,
        "executed_unbooked_qty": executed,
        "working_sell_qty": working,
        "reserved_qty": reserved,
        "available_to_sell_qty": max(0.0, ledger - reserved),
        "economic_open_qty": max(0.0, ledger - executed),
    }


def reserved_unbooked_sell_qty(db_path: str, trade_id: str) -> float:
    """Base already sent or sold, and not yet written as a close."""
    if not str(trade_id or ""):
        return 0.0
    return float(sell_quantity_state(db_path, trade_id, 0.0)["reserved_qty"])


def has_unbooked_executed_sell(db_path: str) -> bool:
    """True while a venue fill is still waiting to credit cash and cut the lot."""
    for row in _unbooked_sell_rows(db_path):
        sold, _resting = _row_sell_parts(db_path, row)
        if sold > 0.0:
            return True
    return False


def unbooked_protective_sells(db_path: str, trade_id: str) -> list[dict[str, Any]]:
    """Fills waiting for a commission asset, oldest first."""
    if not str(trade_id or ""):
        return []
    conn = _connect(db_path)
    try:
        rows = conn.execute(
            """
            SELECT * FROM strategy_order_intents
            WHERE trade_id=? AND side='SELL' AND status IN ('FEE_PENDING', 'FILLED') AND filled_qty > 0
            ORDER BY created_at
            """,
            (str(trade_id),),
        ).fetchall()
    finally:
        conn.close()
    out = []
    for row in rows:
        intent = dict(row)
        if sell_order_already_booked(db_path, str(intent.get("exchange_order_id") or "")):
            continue
        out.append(intent)
    return out


def close_sell_order_ids(db_path: str, trade_id: str, order_ids: list[str]) -> None:
    """A booked close owns every chunk id, so none of them can be booked again."""
    ids = [str(oid or "").strip() for oid in order_ids if str(oid or "").strip()]
    if not ids or not str(trade_id or ""):
        return
    conn = _connect(db_path)
    try:
        conn.execute(
            f"""
            UPDATE strategy_order_intents
            SET status='CLOSED', updated_at=?
            WHERE trade_id=? AND side='SELL' AND exchange_order_id IN ({", ".join("?" for _ in ids)})
            """,
            (time.time(), str(trade_id), *ids),
        )
        conn.commit()
    finally:
        conn.close()


def sell_order_already_booked(db_path: str, exchange_order_id: str) -> bool:
    oid = str(exchange_order_id or "").strip()
    if not oid:
        return False
    conn = sqlite3.connect(db_path, timeout=15)
    try:
        tables = {str(row[0]) for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        if "paper_trades" not in tables:
            return False
        cols = {str(row[1]) for row in conn.execute("PRAGMA table_info(paper_trades)")}
        if "order_id" not in cols:
            return False
        row = conn.execute(
            "SELECT 1 FROM paper_trades WHERE order_id=? AND UPPER(side)='SELL' LIMIT 1",
            (oid,),
        ).fetchone()
    finally:
        conn.close()
    return row is not None


def recovered_sell_order(db_path: str, trade_id: str, *, requested_qty: float) -> dict[str, Any] | None:
    """The venue order that already sold this quantity, if the close was not booked."""
    conn = _connect(db_path)
    try:
        rows = conn.execute(
            """
            SELECT * FROM strategy_order_intents
            WHERE trade_id=? AND side='SELL' AND filled_qty > 0 AND exchange_order_id != ''
            ORDER BY created_at
            """,
            (str(trade_id),),
        ).fetchall()
    finally:
        conn.close()
    for row in rows:
        intent = dict(row)
        qty = float(intent["filled_qty"] or 0.0)
        if qty <= 0.0:
            continue
        if sell_order_already_booked(db_path, str(intent["exchange_order_id"])):
            continue
        cost = float(intent["filled_cost"] or 0.0)
        return {
            "id": str(intent["exchange_order_id"]),
            "filled": qty,
            "average": (cost / qty) if qty else 0.0,
            "status": "closed",
            "clientOrderId": str(intent["client_order_id"]),
            "_mystic_recovered_fill": True,
        }
    return None
