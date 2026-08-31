"""Paper liquidity provision: rest at touch, fill only on through-price.

Does not predict direction. Does not treat mid or cross-venue dislocation as arb.
Touch is not a fill. Queue priority is not modeled. Live is not armed here.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

SETUP_NAME = "structural_lp"
BOOK = "structural_lp"
FILL_BID = "STRUCTURAL_BID_THROUGH"
FILL_ASK = "STRUCTURAL_ASK_THROUGH"
TIMEOUT = "STRUCTURAL_INVENTORY_TIMEOUT"
NO_BOOK = "STRUCTURAL_NO_BOOK"


@dataclass(frozen=True)
class RestingQuote:
    symbol: str
    side: str
    price: float
    qty: float
    posted_epoch: float


@dataclass(frozen=True)
class BookPrint:
    symbol: str
    best_bid: float
    best_ask: float
    epoch: float

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


@dataclass(frozen=True)
class LpAction:
    kind: str
    symbol: str
    reason: str
    price: float = 0.0
    qty: float = 0.0
    quote: RestingQuote | None = None


def book_from_snap(snap: Any, *, epoch: float) -> BookPrint | None:
    bid = float(getattr(snap, "best_bid", 0.0) or 0.0)
    ask = float(getattr(snap, "best_ask", 0.0) or 0.0)
    if bid <= 0 or ask <= 0 or ask < bid:
        return None
    return BookPrint(
        symbol=str(getattr(snap, "symbol", "") or ""),
        best_bid=bid,
        best_ask=ask,
        epoch=float(epoch),
    )


def through_fill(quote: RestingQuote | None, book: BookPrint) -> bool:
    """Conservative: the opposite side walked through our rest. Touch is not a fill."""
    if quote is None or quote.symbol != book.symbol or quote.price <= 0:
        return False
    side = str(quote.side or "").upper()
    if side == "BID":
        return book.best_ask <= quote.price
    if side == "ASK":
        return book.best_bid >= quote.price
    return False


def next_rest(
    book: BookPrint,
    *,
    inventory_qty: float,
    lot_qty: float,
) -> RestingQuote | None:
    """Long-only: bid when flat, ask when long. Always at current touch."""
    if lot_qty <= 0:
        return None
    if inventory_qty > 0:
        return RestingQuote(
            symbol=book.symbol,
            side="ASK",
            price=float(book.best_ask),
            qty=float(inventory_qty),
            posted_epoch=book.epoch,
        )
    return RestingQuote(
        symbol=book.symbol,
        side="BID",
        price=float(book.best_bid),
        qty=float(lot_qty),
        posted_epoch=book.epoch,
    )


def step(
    quote: RestingQuote | None,
    book: BookPrint | None,
    *,
    inventory_qty: float,
    lot_qty: float,
    hold_sec: float,
    max_hold_sec: float,
) -> tuple[RestingQuote | None, list[LpAction]]:
    """Advance one symbol. Inventory timeout is a taker flatten, labeled as such."""
    actions: list[LpAction] = []
    if book is None:
        return None, [LpAction(kind="cancel", symbol="", reason=NO_BOOK)]

    if inventory_qty > 0 and max_hold_sec > 0 and hold_sec >= max_hold_sec:
        actions.append(
            LpAction(
                kind="timeout_sell",
                symbol=book.symbol,
                reason=TIMEOUT,
                price=float(book.best_bid),
                qty=float(inventory_qty),
            )
        )
        nxt = next_rest(book, inventory_qty=0.0, lot_qty=lot_qty)
        return nxt, actions

    if through_fill(quote, book):
        assert quote is not None
        if str(quote.side).upper() == "BID" and inventory_qty <= 0:
            actions.append(
                LpAction(
                    kind="buy",
                    symbol=book.symbol,
                    reason=FILL_BID,
                    price=float(quote.price),
                    qty=float(quote.qty),
                    quote=quote,
                )
            )
            inventory_qty = float(quote.qty)
        elif str(quote.side).upper() == "ASK" and inventory_qty > 0:
            actions.append(
                LpAction(
                    kind="sell",
                    symbol=book.symbol,
                    reason=FILL_ASK,
                    price=float(quote.price),
                    qty=float(inventory_qty),
                    quote=quote,
                )
            )
            inventory_qty = 0.0

    nxt = next_rest(book, inventory_qty=inventory_qty, lot_qty=lot_qty)
    if nxt is not None:
        actions.append(
            LpAction(
                kind="quote",
                symbol=book.symbol,
                reason="REST_TOUCH",
                price=float(nxt.price),
                qty=float(nxt.qty),
                quote=nxt,
            )
        )
    return nxt, actions


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
