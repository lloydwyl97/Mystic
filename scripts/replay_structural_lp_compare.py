#!/usr/bin/env python3
"""Compare retired snapshot through-fill vs event-queue LP on the same window.

Uses tests/fixtures/scalp_l2_sample.jsonl plus a deterministic synthetic tape.
Does not wait for live days of paper data.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.services.binance_scalp.structural_lp import (
    BookPrint,
    consume_events,
    legacy_snapshot_through_fill,
    place_quote,
)
from backend.services.binance_scalp.structural_tape import TradeEvent

FIXTURE = ROOT / "tests" / "fixtures" / "scalp_l2_sample.jsonl"


def _books() -> list[BookPrint]:
    out: list[BookPrint] = []
    if not FIXTURE.exists():
        return out
    for line in FIXTURE.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        data = row.get("data") or {}
        stream = str(row.get("stream") or "")
        if "btcusdt" not in stream:
            continue
        bids = [(float(a), float(b)) for a, b in (data.get("bids") or [])[:5]]
        asks = [(float(a), float(b)) for a, b in (data.get("asks") or [])[:5]]
        if not bids or not asks:
            continue
        out.append(
            BookPrint(
                symbol="BTCUSDT",
                best_bid=bids[0][0],
                best_ask=asks[0][0],
                epoch=float(row.get("ts") or 0.0),
                bids=tuple(bids),
                asks=tuple(asks),
                source="fixture_l2",
                age_sec=0.0,
            )
        )
    return out


def _synthetic_tape(books: list[BookPrint]) -> list[TradeEvent]:
    events: list[TradeEvent] = []
    if not books:
        return events
    first = books[0]
    # Pre-placement prints — must not fill the repaired model.
    events.append(TradeEvent("BTCUSDT", 1, first.best_bid, 0.05, True, first.epoch - 1.0, first.epoch - 1.0))
    # After placement: sell-side at/below bid, but smaller than queue-ahead unless we size it.
    events.append(TradeEvent("BTCUSDT", 2, first.best_bid, first.bids[0][1] * 0.25, True, first.epoch + 0.05, first.epoch + 0.05))
    # Walk that would trigger the old through-fill if later ask <= bid.
    for i, book in enumerate(books[1:6], start=3):
        events.append(TradeEvent("BTCUSDT", i, book.best_bid, 0.00001, True, book.epoch + 0.01, book.epoch + 0.01))
    # Duplicate / out-of-order
    events.append(TradeEvent("BTCUSDT", 2, first.best_bid, 1.0, True, first.epoch + 2.0, first.epoch + 2.0))
    events.append(TradeEvent("BTCUSDT", 1, first.best_bid, 1.0, True, first.epoch + 3.0, first.epoch + 3.0))
    return events


def main() -> int:
    books = _books()
    if len(books) < 2:
        print("REPLAY_FAIL: not enough BTC L2 rows")
        return 1
    first = books[0]
    quote = place_quote(first, side="BID", qty=0.01, now_ts=first.epoch, quote_id="replay")
    if quote is None:
        print("REPLAY_FAIL: could not place quote")
        return 1
    legacy_fills = 0
    for book in books[1:]:
        if legacy_snapshot_through_fill(quote, book):
            legacy_fills += 1
    tape = _synthetic_tape(books)
    nxt, actions = consume_events(quote, tape)
    event_fill_qty = sum(a.qty for a in actions)
    print(
        json.dumps(
            {
                "window_books": len(books),
                "tape_events": len(tape),
                "queue_ahead": quote.queue_ahead,
                "legacy_through_fill_hits": legacy_fills,
                "event_queue_fill_actions": len(actions),
                "event_queue_filled_qty": event_fill_qty,
                "event_remaining_qty": nxt.remaining_qty,
                "assumption": "cancellations_do_not_consume_queue",
                "note": "legacy hits are L1 through approximations, not proven trades",
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
