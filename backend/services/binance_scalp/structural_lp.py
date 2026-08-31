"""Event-ordered paper LP. Snapshot movement is never a fill.

A resting bid fills only from post-placement sell-side aggTrades at or below the bid
after queue-ahead is consumed. Asks are symmetric. Quotes persist until cancel,
requote, fill, or timeout. Touch/mid/L1 through is not execution proof.
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import asdict, dataclass, field, replace
from typing import Any

from backend.services.binance_scalp.structural_mode import FILL_MODEL_VERSION
from backend.services.binance_scalp.structural_tape import TradeEvent, apply_queue, event_eligible_for_quote

SETUP_NAME = "structural_lp"
BOOK = "structural_lp"
FILL_BID = "STRUCTURAL_BID_TRADE"
FILL_ASK = "STRUCTURAL_ASK_TRADE"
TIMEOUT = "STRUCTURAL_INVENTORY_TIMEOUT"
NO_BOOK = "STRUCTURAL_NO_BOOK"
STALE_TAPE = "STRUCTURAL_TAPE_STALE"
TOXIC = "STRUCTURAL_TOXIC_BOOK"
EDGE_SHORT = "STRUCTURAL_MIN_NET_EDGE"
CANCEL_REQUOTE = "STRUCTURAL_REQUOTE"


@dataclass
class RestingQuote:
    symbol: str
    side: str
    price: float
    qty: float
    posted_epoch: float
    posted_ts: float
    queue_ahead: float
    queue_consumed: float = 0.0
    filled_qty: float = 0.0
    remaining_qty: float = 0.0
    last_agg_id: int = 0
    last_event_ts: float = 0.0
    data_source: str = "binance_us_ws_depth20"
    quote_id: str = ""

    def __post_init__(self) -> None:
        if self.remaining_qty <= 0 and self.qty > 0 and self.filled_qty <= 0:
            self.remaining_qty = float(self.qty)


@dataclass(frozen=True)
class BookPrint:
    symbol: str
    best_bid: float
    best_ask: float
    epoch: float
    bids: tuple[tuple[float, float], ...] = ()
    asks: tuple[tuple[float, float], ...] = ()
    source: str = "unknown"
    age_sec: float = 0.0

    @property
    def mid(self) -> float:
        if self.best_bid > 0 and self.best_ask > 0:
            return (self.best_bid + self.best_ask) / 2.0
        return 0.0

    @property
    def spread_pct(self) -> float:
        mid = self.mid
        if mid <= 0:
            return 1.0
        return (self.best_ask - self.best_bid) / mid

    @property
    def spread_bps(self) -> float:
        return self.spread_pct * 10_000.0


@dataclass(frozen=True)
class LpAction:
    kind: str
    symbol: str
    reason: str
    price: float = 0.0
    qty: float = 0.0
    quote: RestingQuote | None = None
    audit: dict[str, Any] = field(default_factory=dict)


def depth_ahead(levels: list[list[float]] | tuple, price: float) -> float:
    """Displayed size already resting at the exact price. Conservative: exact tick only."""
    total = 0.0
    target = float(price)
    for row in levels or []:
        if len(row) < 2:
            continue
        px, sz = float(row[0]), float(row[1])
        if abs(px - target) <= 1e-12:
            total += max(0.0, sz)
    return total


def book_from_snap(snap: Any, *, epoch: float) -> BookPrint | None:
    bid = float(getattr(snap, "best_bid", 0.0) or 0.0)
    ask = float(getattr(snap, "best_ask", 0.0) or 0.0)
    if bid <= 0 or ask <= 0 or ask < bid:
        return None
    bids = tuple((float(r[0]), float(r[1])) for r in (getattr(snap, "bids", None) or [])[:20] if r)
    asks = tuple((float(r[0]), float(r[1])) for r in (getattr(snap, "asks", None) or [])[:20] if r)
    return BookPrint(
        symbol=str(getattr(snap, "symbol", "") or ""),
        best_bid=bid,
        best_ask=ask,
        epoch=float(epoch),
        bids=bids,
        asks=asks,
        source=str(getattr(snap, "book_source", "") or "unknown"),
        age_sec=float(getattr(snap, "orderbook_age_sec", 0.0) or 0.0),
    )


def place_quote(
    book: BookPrint,
    *,
    side: str,
    qty: float,
    now_ts: float,
    quote_id: str,
    last_agg_id: int = 0,
) -> RestingQuote | None:
    if qty <= 0:
        return None
    side_u = str(side).upper()
    if side_u == "BID":
        price = float(book.best_bid)
        ahead = depth_ahead(list(book.bids), price)
    elif side_u == "ASK":
        price = float(book.best_ask)
        ahead = depth_ahead(list(book.asks), price)
    else:
        return None
    if price <= 0:
        return None
    return RestingQuote(
        symbol=book.symbol,
        side=side_u,
        price=price,
        qty=float(qty),
        posted_epoch=book.epoch,
        posted_ts=float(now_ts),
        queue_ahead=float(ahead),
        remaining_qty=float(qty),
        last_agg_id=int(last_agg_id),
        data_source=book.source,
        quote_id=quote_id,
    )


def quote_still_current(quote: RestingQuote | None, book: BookPrint, *, side: str) -> bool:
    if quote is None or quote.remaining_qty <= 0:
        return False
    if str(quote.side).upper() != str(side).upper():
        return False
    want = book.best_bid if str(side).upper() == "BID" else book.best_ask
    return abs(float(quote.price) - float(want)) <= 1e-12


def consume_events(quote: RestingQuote, events: list[TradeEvent]) -> tuple[RestingQuote, list[LpAction]]:
    actions: list[LpAction] = []
    cur = quote
    for ev in events:
        if not event_eligible_for_quote(
            ev,
            side=cur.side,
            price=cur.price,
            posted_ts=cur.posted_ts,
            last_agg_id=cur.last_agg_id,
        ):
            continue
        consumed, fill, remain = apply_queue(
            queue_ahead=cur.queue_ahead,
            queue_consumed=cur.queue_consumed,
            remaining_qty=cur.remaining_qty,
            event_qty=ev.qty,
        )
        cur = replace(
            cur,
            queue_consumed=consumed,
            filled_qty=cur.filled_qty + fill,
            remaining_qty=remain,
            last_agg_id=ev.agg_id,
            last_event_ts=ev.trade_ts,
        )
        if fill > 0:
            reason = FILL_BID if cur.side == "BID" else FILL_ASK
            actions.append(
                LpAction(
                    kind="partial_fill" if remain > 1e-12 else "fill",
                    symbol=cur.symbol,
                    reason=reason,
                    price=float(cur.price),
                    qty=float(fill),
                    quote=cur,
                    audit={
                        "fill_model": FILL_MODEL_VERSION,
                        "agg_id": ev.agg_id,
                        "event_price": ev.price,
                        "event_qty": ev.qty,
                        "queue_ahead": cur.queue_ahead,
                        "queue_consumed": cur.queue_consumed,
                        "filled_qty": cur.filled_qty,
                        "remaining_qty": cur.remaining_qty,
                        "data_source": ev.source,
                        "assumption": "cancellations_do_not_consume_queue",
                    },
                )
            )
        if remain <= 1e-12:
            break
    return cur, actions


def legacy_snapshot_through_fill(quote: RestingQuote | None, book: BookPrint) -> bool:
    """Retired dc93d31 approximation. Replay comparison only. Not used for execution."""
    if quote is None or quote.price <= 0:
        return False
    if str(quote.side).upper() == "BID":
        return book.best_ask <= quote.price
    if str(quote.side).upper() == "ASK":
        return book.best_bid >= quote.price
    return False


def is_structural_position(row: Any) -> bool:
    try:
        sid = str(row["strategy_id"] or "")
    except Exception:
        sid = str(getattr(row, "strategy_id", "") or "")
    if sid == SETUP_NAME:
        return True
    raw = ""
    try:
        raw = str(row["diagnostics_json"] or "")
    except Exception:
        raw = str(getattr(row, "diagnostics_json", "") or "")
    return f'"{BOOK}"' in raw or '"setup_name": "structural_lp"' in raw or '"setup_name":"structural_lp"' in raw


def quote_to_dict(quote: RestingQuote) -> dict[str, Any]:
    return asdict(quote)


def quote_from_dict(raw: dict[str, Any] | None) -> RestingQuote | None:
    if not raw:
        return None
    try:
        return RestingQuote(
            symbol=str(raw["symbol"]),
            side=str(raw["side"]),
            price=float(raw["price"]),
            qty=float(raw["qty"]),
            posted_epoch=float(raw.get("posted_epoch") or 0.0),
            posted_ts=float(raw.get("posted_ts") or 0.0),
            queue_ahead=float(raw.get("queue_ahead") or 0.0),
            queue_consumed=float(raw.get("queue_consumed") or 0.0),
            filled_qty=float(raw.get("filled_qty") or 0.0),
            remaining_qty=float(raw.get("remaining_qty") or 0.0),
            last_agg_id=int(raw.get("last_agg_id") or 0),
            last_event_ts=float(raw.get("last_event_ts") or 0.0),
            data_source=str(raw.get("data_source") or ""),
            quote_id=str(raw.get("quote_id") or ""),
        )
    except (KeyError, TypeError, ValueError):
        return None


def ensure_working_quote_table(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS scalp_structural_working (
            symbol TEXT PRIMARY KEY,
            quote_json TEXT,
            last_stream_id TEXT,
            updated_at TEXT NOT NULL DEFAULT (datetime('now'))
        )
        """
    )


def load_working_quotes(conn: sqlite3.Connection) -> tuple[dict[str, RestingQuote], dict[str, str]]:
    ensure_working_quote_table(conn)
    quotes: dict[str, RestingQuote] = {}
    tape_ids: dict[str, str] = {}
    for row in conn.execute("SELECT symbol, quote_json, last_stream_id FROM scalp_structural_working"):
        q = quote_from_dict(json.loads(row[1]) if row[1] else None)
        if q is not None:
            quotes[str(row[0])] = q
        if row[2]:
            tape_ids[str(row[0])] = str(row[2])
    return quotes, tape_ids


def save_working_quote(
    conn: sqlite3.Connection,
    symbol: str,
    quote: RestingQuote | None,
    last_stream_id: str,
) -> None:
    ensure_working_quote_table(conn)
    conn.execute(
        """
        INSERT INTO scalp_structural_working (symbol, quote_json, last_stream_id, updated_at)
        VALUES (?, ?, ?, datetime('now'))
        ON CONFLICT(symbol) DO UPDATE SET
          quote_json=excluded.quote_json,
          last_stream_id=excluded.last_stream_id,
          updated_at=datetime('now')
        """,
        (symbol, json.dumps(quote_to_dict(quote)) if quote is not None else None, last_stream_id or "0-0"),
    )
