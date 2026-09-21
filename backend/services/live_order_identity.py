"""Persist the exchange identity of every live fill.

A confirmed Binance.US fill used to leave almost no durable trace. The order
id, client order id, per-fill trade ids, fee asset and venue timestamps were
read into memory, logged, then dropped: ``paper_trades.order_id`` was never
written on any of 502 live rows, and a live SELL carried no identifier at all.
Without them a recorded row cannot be tied back to the venue, so reconciliation
has to guess by quantity and time, and restart recovery cannot tell which order
a position came from.

This module writes one append-only ``live_exchange_fills`` row per confirmed
live fill. It is the join table between the trailing-buy intent, the FIFO trade
rows, the position, recovery state and the reconciliation output. It records
what the exchange reported and never overwrites or deletes a prior row.
"""

from __future__ import annotations

import json
import logging
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

logger = logging.getLogger(__name__)

TABLE = "live_exchange_fills"

# Every identifier the venue gives us for a fill. A live fill missing any of
# these is logged loudly: it means the venue response shape changed, and
# silently storing a partial identity is what produced the unreconcilable
# history in the first place.
REQUIRED_FIELDS: tuple[str, ...] = (
    "exchange_order_id",
    "symbol",
    "side",
    "executed_qty",
    "avg_fill_price",
    "event_ts_exchange",
)

_SCHEMA = f"""
CREATE TABLE IF NOT EXISTS {TABLE} (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    venue TEXT NOT NULL DEFAULT 'binanceus',
    symbol TEXT NOT NULL,
    side TEXT NOT NULL,
    exchange_order_id TEXT NOT NULL,
    client_order_id TEXT,
    fill_ids_json TEXT,
    venue_trade_ids_json TEXT,
    fill_count INTEGER NOT NULL DEFAULT 0,
    executed_qty REAL NOT NULL,
    avg_fill_price REAL NOT NULL,
    cost_quote REAL,
    fee_amount REAL,
    fee_asset TEXT,
    fee_items_json TEXT,
    fee_from_exchange INTEGER NOT NULL DEFAULT 0,
    order_status TEXT,
    mystic_trade_id TEXT,
    intent_id TEXT,
    decision_id TEXT,
    position_entry_ts TEXT,
    event_ts_exchange TEXT,
    event_ts_submitted TEXT,
    event_ts_recorded TEXT NOT NULL,
    missing_fields_json TEXT,
    raw_json TEXT
)
"""

_INDEXES = (
    f"CREATE INDEX IF NOT EXISTS idx_{TABLE}_order ON {TABLE}(exchange_order_id)",
    f"CREATE INDEX IF NOT EXISTS idx_{TABLE}_trade ON {TABLE}(mystic_trade_id)",
    f"CREATE INDEX IF NOT EXISTS idx_{TABLE}_intent ON {TABLE}(intent_id)",
    f"CREATE INDEX IF NOT EXISTS idx_{TABLE}_symbol_side ON {TABLE}(symbol, side)",
    # One row per (order, mystic trade row). A retry of the same commit must not
    # double-book the same venue fill.
    f"CREATE UNIQUE INDEX IF NOT EXISTS uq_{TABLE}_order_trade ON {TABLE}(venue, exchange_order_id, side, COALESCE(mystic_trade_id,''))",
)


@dataclass
class OrderIdentity:
    """Exchange identity of one confirmed live fill."""

    venue: str = "binanceus"
    symbol: str = ""
    side: str = ""
    exchange_order_id: str = ""
    client_order_id: str = ""
    fill_ids: list[str] = field(default_factory=list)
    venue_trade_ids: list[str] = field(default_factory=list)
    executed_qty: float = 0.0
    avg_fill_price: float = 0.0
    cost_quote: float = 0.0
    fee_amount: float = 0.0
    fee_asset: str = ""
    fee_items: list[dict[str, Any]] = field(default_factory=list)
    fee_from_exchange: bool = False
    order_status: str = ""
    mystic_trade_id: str = ""
    intent_id: str = ""
    decision_id: str = ""
    position_entry_ts: str = ""
    event_ts_exchange: str = ""
    event_ts_submitted: str = ""
    raw: dict[str, Any] = field(default_factory=dict)

    def missing(self) -> list[str]:
        out: list[str] = []
        for name in REQUIRED_FIELDS:
            val = getattr(self, name, None)
            blank = val is None or (isinstance(val, str) and not val.strip()) or (isinstance(val, (int, float)) and not val)
            if blank:
                out.append(name)
        # Fee asset is only required once a fee was actually charged.
        if self.fee_amount and not str(self.fee_asset or "").strip():
            out.append("fee_asset")
        return out


def _iso_from_ms(ms: Any) -> str:
    try:
        v = float(ms)
    except (TypeError, ValueError):
        return ""
    if v <= 0:
        return ""
    if v > 1e12:  # milliseconds
        v = v / 1000.0
    try:
        return datetime.fromtimestamp(v, tz=timezone.utc).isoformat()
    except (OSError, OverflowError, ValueError):
        return ""


def iso_or_blank(epoch_seconds: Any) -> str:
    """Render an epoch-seconds position timestamp as ISO, or blank."""
    try:
        v = float(epoch_seconds or 0.0)
    except (TypeError, ValueError):
        return ""
    if v <= 0:
        return ""
    try:
        return datetime.fromtimestamp(v, tz=timezone.utc).isoformat()
    except (OSError, OverflowError, ValueError):
        return ""


def _as_float(raw: Any) -> float:
    try:
        if raw is None or raw == "":
            return 0.0
        return float(raw)
    except (TypeError, ValueError):
        return 0.0


def extract_identity(
    order: dict[str, Any] | None,
    *,
    symbol: str,
    side: str,
    mystic_trade_id: str = "",
    intent_id: str = "",
    decision_id: str = "",
    client_order_id: str = "",
    position_entry_ts: str = "",
    submitted_at: str = "",
    fallback_qty: float = 0.0,
    fallback_price: float = 0.0,
    fee_amount: float = 0.0,
    fee_asset: str = "",
    fee_items: list[dict[str, Any]] | None = None,
    fee_from_exchange: bool = False,
    venue: str = "binanceus",
) -> OrderIdentity:
    """Pull every identifier out of a CCXT order response.

    ``fallback_qty`` / ``fallback_price`` are the engine's own post-economics
    values, used only when the venue response omits them; the venue's figures
    take priority because they are what actually settled.
    """
    raw = order if isinstance(order, dict) else {}
    info = raw.get("info") if isinstance(raw.get("info"), dict) else {}

    executed = _as_float(raw.get("filled")) or _as_float(info.get("executedQty")) or _as_float(fallback_qty)
    avg = _as_float(raw.get("average")) or _as_float(raw.get("price")) or _as_float(fallback_price)
    cost = _as_float(raw.get("cost")) or _as_float(info.get("cummulativeQuoteQty"))
    if not cost and executed and avg:
        cost = executed * avg

    fill_ids: list[str] = []
    venue_trade_ids: list[str] = []
    trades = raw.get("trades")
    if isinstance(trades, list):
        for t in trades:
            if not isinstance(t, dict):
                continue
            tid = str(t.get("id") or "").strip()
            if tid:
                fill_ids.append(tid)
            otid = str(t.get("order") or "").strip()
            if otid:
                venue_trade_ids.append(otid)
    # Binance.US returns the per-fill trade ids in info["fills"] on the
    # create-order reply (newOrderRespType FULL), as
    # [{price, qty, commission, commissionAsset, tradeId}]. CCXT only mirrors
    # them into the normalized "trades" list for some order types, so reading
    # "trades" alone left fill_ids empty on every recorded live fill. GET /order
    # omits fills altogether, which is why a re-fetch cannot substitute here.
    for f in info.get("fills") or []:
        if not isinstance(f, dict):
            continue
        tid = str(f.get("tradeId") or f.get("trade_id") or f.get("id") or "").strip()
        if tid and tid not in fill_ids:
            fill_ids.append(tid)
        oid = str(f.get("orderId") or f.get("order_id") or "").strip()
        if oid and oid not in venue_trade_ids:
            venue_trade_ids.append(oid)

    # Binance also reports a single aggregate fill id on some responses.
    for key in ("fill_id", "tradeId", "trade_id"):
        v = str(raw.get(key) or info.get(key) or "").strip()
        if v and v not in fill_ids:
            fill_ids.append(v)

    items = list(fee_items or [])
    resolved_fee = _as_float(fee_amount)
    resolved_asset = str(fee_asset or "").strip().upper()
    native_pairs: list[tuple[float, str]] = []
    for it in items:
        if not isinstance(it, dict):
            continue
        amt = _as_float(it.get("amount") if it.get("amount") is not None else it.get("cost"))
        asset = str(it.get("asset") or it.get("currency") or "").strip().upper()
        if amt and asset:
            native_pairs.append((amt, asset))
    if not resolved_asset:
        # The engine's own commission breakdown carries the settled asset per
        # item ({"amount", "asset", "usd"}); a fill can be charged in more than
        # one asset, so keep all of them rather than picking one.
        assets = []
        for _amt, a in native_pairs:
            if a and a not in assets:
                assets.append(a)
        resolved_asset = ",".join(assets)
    quote_assets = {"USDT", "USD", "BUSD", "USDC"}
    non_quote = [(amt, a) for amt, a in native_pairs if a not in quote_assets]
    # A quote-denominated estimate must never be stored as a base-asset commission.
    if len({a for _amt, a in non_quote}) == 1:
        native_asset = non_quote[0][1]
        native_sum = sum(amt for amt, a in non_quote if a == native_asset)
        if native_sum > 0:
            resolved_fee = native_sum
            if "," not in resolved_asset:
                resolved_asset = native_asset
    if not resolved_asset:
        fee = raw.get("fee") if isinstance(raw.get("fee"), dict) else {}
        resolved_asset = str(fee.get("currency") or info.get("commissionAsset") or raw.get("commissionAsset") or "").strip().upper()
    if not resolved_fee:
        fee = raw.get("fee") if isinstance(raw.get("fee"), dict) else {}
        resolved_fee = _as_float(fee.get("cost")) or _as_float(info.get("commission")) or _as_float(raw.get("commission"))
    if not items:
        for f in raw.get("fees") or []:
            if isinstance(f, dict):
                items.append({"cost": _as_float(f.get("cost")), "currency": str(f.get("currency") or "").upper()})

    return OrderIdentity(
        venue=str(venue or "binanceus"),
        symbol=str(symbol or raw.get("symbol") or ""),
        side=str(side or raw.get("side") or "").upper(),
        exchange_order_id=str(raw.get("id") or info.get("orderId") or "").strip(),
        client_order_id=str(raw.get("clientOrderId") or info.get("clientOrderId") or client_order_id or "").strip(),
        fill_ids=fill_ids,
        venue_trade_ids=venue_trade_ids,
        executed_qty=executed,
        avg_fill_price=avg,
        cost_quote=cost,
        fee_amount=resolved_fee,
        fee_asset=resolved_asset,
        fee_items=items,
        fee_from_exchange=bool(fee_from_exchange),
        order_status=str(raw.get("status") or info.get("status") or "").strip(),
        mystic_trade_id=str(mystic_trade_id or ""),
        intent_id=str(intent_id or ""),
        decision_id=str(decision_id or ""),
        position_entry_ts=str(position_entry_ts or ""),
        event_ts_exchange=_iso_from_ms(raw.get("timestamp")) or _iso_from_ms(info.get("transactTime")),
        event_ts_submitted=str(submitted_at or ""),
        raw=raw,
    )


def ensure_schema(db_path: str) -> None:
    with sqlite3.connect(db_path, timeout=15) as conn:
        conn.execute(_SCHEMA)
        for stmt in _INDEXES:
            try:
                conn.execute(stmt)
            except sqlite3.OperationalError:
                # A pre-existing row set may violate the unique index; the
                # table still works without it.
                logger.debug("LIVE_FILL_INDEX_SKIPPED: %s", stmt)
        conn.commit()


def record_fill(db_path: str, identity: OrderIdentity) -> bool:
    """Append one confirmed live fill. Never overwrites an existing row."""
    if not identity.exchange_order_id:
        logger.error(
            "LIVE_FILL_IDENTITY_MISSING_ORDER_ID symbol=%s side=%s trade=%s — venue confirmed a fill with no order id",
            identity.symbol,
            identity.side,
            identity.mystic_trade_id,
        )
        return False

    missing = identity.missing()
    if missing:
        logger.error(
            "LIVE_FILL_IDENTITY_INCOMPLETE order=%s symbol=%s side=%s trade=%s missing=%s",
            identity.exchange_order_id,
            identity.symbol,
            identity.side,
            identity.mystic_trade_id,
            ",".join(missing),
        )

    try:
        ensure_schema(db_path)
        with sqlite3.connect(db_path, timeout=15) as conn:
            conn.execute(
                f"""
                INSERT OR IGNORE INTO {TABLE} (
                    venue, symbol, side, exchange_order_id, client_order_id,
                    fill_ids_json, venue_trade_ids_json, fill_count,
                    executed_qty, avg_fill_price, cost_quote,
                    fee_amount, fee_asset, fee_items_json, fee_from_exchange,
                    order_status, mystic_trade_id, intent_id, decision_id,
                    position_entry_ts, event_ts_exchange, event_ts_submitted,
                    event_ts_recorded, missing_fields_json, raw_json
                ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                """,
                (
                    identity.venue,
                    identity.symbol,
                    identity.side,
                    identity.exchange_order_id,
                    identity.client_order_id or None,
                    json.dumps(identity.fill_ids),
                    json.dumps(identity.venue_trade_ids),
                    len(identity.fill_ids),
                    float(identity.executed_qty),
                    float(identity.avg_fill_price),
                    float(identity.cost_quote),
                    float(identity.fee_amount),
                    identity.fee_asset or None,
                    json.dumps(identity.fee_items),
                    1 if identity.fee_from_exchange else 0,
                    identity.order_status or None,
                    identity.mystic_trade_id or None,
                    identity.intent_id or None,
                    identity.decision_id or None,
                    identity.position_entry_ts or None,
                    identity.event_ts_exchange or None,
                    identity.event_ts_submitted or None,
                    datetime.now(timezone.utc).isoformat(),
                    json.dumps(missing),
                    json.dumps(identity.raw, default=str)[:20000],
                ),
            )
            conn.commit()
        return True
    except sqlite3.Error as exc:
        # Never let bookkeeping abort a settled trade; the loud log is the
        # signal that a fill needs manual reconciliation.
        logger.error(
            "LIVE_FILL_IDENTITY_WRITE_FAILED order=%s symbol=%s side=%s: %s",
            identity.exchange_order_id,
            identity.symbol,
            identity.side,
            exc,
        )
        return False


def rows_missing_venue_trade_ids(db_path: str, limit: int = 500) -> list[dict[str, Any]]:
    """Recorded live fills whose per-fill venue trade ids were never captured."""
    return _query(
        db_path,
        "WHERE COALESCE(fill_ids_json,'[]') IN ('[]','','null') AND LENGTH(COALESCE(exchange_order_id,''))>0 ORDER BY id DESC LIMIT ?",
        (int(limit),),
    )


def backfill_venue_trade_ids(db_path: str, row_id: int, trade_ids: list[str], order_ids: list[str] | None = None) -> bool:
    """Write venue-proven per-fill trade ids onto one existing identity row.

    Only fills in ids that the venue actually reported for that exact order.
    Never invents, never overwrites a non-empty value, and never touches any
    other column, so the recorded fill economics stay exactly as settled.
    """
    ids = [str(x).strip() for x in (trade_ids or []) if str(x).strip()]
    if not ids:
        return False
    oids = [str(x).strip() for x in (order_ids or []) if str(x).strip()]
    try:
        with sqlite3.connect(db_path, timeout=15) as conn:
            cols = {str(r[1]) for r in conn.execute(f"PRAGMA table_info({TABLE})")}
            if "venue_order_ids_json" not in cols:
                conn.execute(f"ALTER TABLE {TABLE} ADD COLUMN venue_order_ids_json TEXT DEFAULT '[]'")
            cur = conn.execute(
                f"""
                UPDATE {TABLE}
                SET fill_ids_json = ?,
                    venue_trade_ids_json = CASE
                        WHEN COALESCE(venue_trade_ids_json,'[]') IN ('[]','','null') THEN ?
                        ELSE venue_trade_ids_json END,
                    venue_order_ids_json = CASE
                        WHEN COALESCE(venue_order_ids_json,'[]') IN ('[]','','null') THEN ?
                        ELSE venue_order_ids_json END,
                    fill_count = ?
                WHERE id = ? AND COALESCE(fill_ids_json,'[]') IN ('[]','','null')
                """,
                (json.dumps(ids), json.dumps(ids), json.dumps(oids), len(ids), int(row_id)),
            )
            conn.commit()
            return bool(cur.rowcount)
    except sqlite3.Error as exc:
        logger.error("LIVE_FILL_BACKFILL_FAILED row=%s: %s", row_id, exc)
        return False


async def reconcile_venue_trade_ids(
    db_path: str,
    live_service: Any,
    *,
    limit: int = 500,
    exchange: str = "binanceus",
) -> dict[str, Any]:
    """Capture missing per-fill venue trade ids from Binance.US.

    The venue response is the only proof of a trade id, so a row the venue
    cannot account for is reported as unproven and left untouched rather than
    filled with a guess.
    """
    ensure_schema(db_path)
    pending = rows_missing_venue_trade_ids(db_path, limit=limit)
    out = {"examined": len(pending), "reconciled": 0, "unproven": 0, "errors": 0, "unproven_orders": []}
    for row in pending:
        order_id = str(row.get("exchange_order_id") or "")
        symbol = str(row.get("symbol") or "")
        if not order_id or not symbol:
            out["unproven"] += 1
            continue
        try:
            res = await live_service.fetch_order_trades(exchange, symbol, order_id)
        except Exception as exc:
            logger.warning("LIVE_FILL_RECONCILE_FETCH_FAILED order=%s: %s", order_id, exc)
            out["errors"] += 1
            continue
        if res.get("status") != "success":
            out["errors"] += 1
            continue
        trades = res.get("trades") or []
        ids = [str(t.get("trade_id") or "") for t in trades if str(t.get("trade_id") or "").strip()]
        oids = [str(t.get("order_id") or "") for t in trades if str(t.get("order_id") or "").strip()]
        if not ids:
            out["unproven"] += 1
            out["unproven_orders"].append(order_id)
            continue
        if backfill_venue_trade_ids(db_path, int(row["id"]), ids, oids):
            out["reconciled"] += 1
    logger.info(
        "LIVE_FILL_RECONCILE examined=%d reconciled=%d unproven=%d errors=%d",
        out["examined"],
        out["reconciled"],
        out["unproven"],
        out["errors"],
    )
    return out


def fills_for_trade(db_path: str, mystic_trade_id: str) -> list[dict[str, Any]]:
    return _query(db_path, "WHERE mystic_trade_id = ?", (str(mystic_trade_id),))


def fills_for_order(db_path: str, exchange_order_id: str) -> list[dict[str, Any]]:
    return _query(db_path, "WHERE exchange_order_id = ?", (str(exchange_order_id),))


def fills_for_intent(db_path: str, intent_id: str) -> list[dict[str, Any]]:
    return _query(db_path, "WHERE intent_id = ?", (str(intent_id),))


def all_order_ids(db_path: str) -> set[str]:
    rows = _query(db_path, "", ())
    return {str(r["exchange_order_id"]) for r in rows if r.get("exchange_order_id")}


def record_exchange_reconciled(
    db_path: str,
    *,
    symbol: str,
    side: str,
    exchange_order_id: str,
    client_order_id: str = "",
    venue_trade_ids: list[str] | None = None,
    executed_qty: float = 0.0,
    avg_fill_price: float = 0.0,
    cost_quote: float = 0.0,
    fee_amount: float = 0.0,
    fee_asset: str = "",
    order_status: str = "",
    event_ts_exchange: str = "",
    matching_sell_order_id: str | None = None,
    remaining_asset: str | None = None,
    classification: str = "",
    raw: dict[str, Any] | None = None,
) -> bool:
    """Persist an exchange-authoritative fill that has no original local row.

    Decision, intent and mystic trade ids stay empty. This is not a paper_trades
    row and invents no strategy association.
    """
    oid = str(exchange_order_id or "").strip()
    if not oid:
        return False
    if fills_for_order(db_path, oid):
        return False
    payload = dict(raw or {})
    payload["source"] = "EXCHANGE_RECONCILED"
    payload["classification"] = str(classification or "")
    payload["matching_sell_order_id"] = matching_sell_order_id
    payload["remaining_asset"] = remaining_asset
    payload["decision_id"] = None
    payload["intent_id"] = None
    payload["mystic_trade_id"] = None
    identity = OrderIdentity(
        symbol=str(symbol or ""),
        side=str(side or "").upper(),
        exchange_order_id=oid,
        client_order_id=str(client_order_id or ""),
        fill_ids=list(venue_trade_ids or []),
        venue_trade_ids=list(venue_trade_ids or []),
        executed_qty=float(executed_qty or 0.0),
        avg_fill_price=float(avg_fill_price or 0.0),
        cost_quote=float(cost_quote or 0.0),
        fee_amount=float(fee_amount or 0.0),
        fee_asset=str(fee_asset or ""),
        fee_from_exchange=True,
        order_status=str(order_status or "FILLED"),
        mystic_trade_id="",
        intent_id="",
        decision_id="",
        event_ts_exchange=str(event_ts_exchange or ""),
        raw=payload,
    )
    return record_fill(db_path, identity)


def _query(db_path: str, where: str, params: tuple[Any, ...]) -> list[dict[str, Any]]:
    try:
        with sqlite3.connect(f"file:{db_path}?mode=ro", uri=True, timeout=10) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(f"SELECT * FROM {TABLE} {where} ORDER BY id", params).fetchall()
        return [dict(r) for r in rows]
    except sqlite3.Error:
        return []


def coverage(db_path: str) -> dict[str, Any]:
    """How much of the recorded live history carries exchange identity."""
    out = {
        "live_rows": 0,
        "live_rows_with_order_id": 0,
        "identity_rows": 0,
        "identity_rows_incomplete": 0,
        "identity_order_ids": 0,
    }
    try:
        with sqlite3.connect(f"file:{db_path}?mode=ro", uri=True, timeout=10) as conn:
            live = "LOWER(COALESCE(mode,''))='live' AND COALESCE(is_synthetic,0)=0"
            out["live_rows"] = int(conn.execute(f"SELECT COUNT(*) FROM paper_trades WHERE {live}").fetchone()[0])
            out["live_rows_with_order_id"] = int(conn.execute(f"SELECT COUNT(*) FROM paper_trades WHERE {live} AND LENGTH(COALESCE(order_id,''))>0").fetchone()[0])
            try:
                out["identity_rows"] = int(conn.execute(f"SELECT COUNT(*) FROM {TABLE}").fetchone()[0])
                out["identity_rows_incomplete"] = int(conn.execute(f"SELECT COUNT(*) FROM {TABLE} WHERE COALESCE(missing_fields_json,'[]') NOT IN ('[]','')").fetchone()[0])
                out["identity_order_ids"] = int(conn.execute(f"SELECT COUNT(DISTINCT exchange_order_id) FROM {TABLE}").fetchone()[0])
            except sqlite3.Error:
                pass
    except sqlite3.Error:
        pass
    return out
