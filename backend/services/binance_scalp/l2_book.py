"""Canonical local L2 book for SCALP microstructure.

Supports:
- full top-N snapshots (Binance ``depth20@100ms`` + ``lastUpdateId``)
- incremental diffs (Binance ``@depth`` U/u sync) for tests and optional use

A stale, gapped, crossed, or empty book is never authoritative.
Nothing here is a strategy hard-block — callers treat ``healthy=False`` as
missing microstructure evidence, not a trade refusal.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

STALE_SEC = 2.0
MAX_ID_JUMP = 50_000


def _now() -> float:
    return time.time()


@dataclass
class BookLevel:
    price: float
    size: float


@dataclass
class LocalL2Book:
    symbol: str
    bids: dict[float, float] = field(default_factory=dict)
    asks: dict[float, float] = field(default_factory=dict)
    last_update_id: int = 0
    last_ts: float = 0.0
    healthy: bool = False
    stale_reason: str = "uninitialized"
    rebuilds: int = 0
    gaps: int = 0
    duplicates: int = 0
    snapshot_ready: bool = False

    def apply_snapshot(
        self,
        bids: list[list[float]] | list[tuple[float, float]],
        asks: list[list[float]] | list[tuple[float, float]],
        last_update_id: int | None = None,
        ts: float | None = None,
    ) -> bool:
        """Replace the book with a full top-N snapshot."""
        t = float(ts if ts is not None else _now())
        uid = int(last_update_id or 0)
        if uid and self.last_update_id and uid < self.last_update_id:
            self.duplicates += 1
            return False
        if uid and self.last_update_id and uid == self.last_update_id:
            self.duplicates += 1
            return False
        if uid and self.last_update_id and uid > self.last_update_id + MAX_ID_JUMP:
            self.gaps += 1
            self.rebuilds += 1
        self.bids = {float(p): float(q) for p, q in bids if float(q) > 0 and float(p) > 0}
        self.asks = {float(p): float(q) for p, q in asks if float(q) > 0 and float(p) > 0}
        if uid:
            self.last_update_id = uid
        self.last_ts = t
        self.snapshot_ready = True
        return self._validate()

    def apply_diff(
        self,
        bids: list[list[float]] | list[tuple[float, float]],
        asks: list[list[float]] | list[tuple[float, float]],
        first_id: int,
        final_id: int,
        ts: float | None = None,
    ) -> bool:
        """Apply one incremental depth event (Binance U/u)."""
        t = float(ts if ts is not None else _now())
        u = int(first_id)
        v = int(final_id)
        if not self.snapshot_ready:
            self.stale_reason = "awaiting_snapshot"
            self.healthy = False
            return False
        if v < self.last_update_id:
            self.duplicates += 1
            return False
        if v == self.last_update_id:
            self.duplicates += 1
            return False
        if u > self.last_update_id + 1:
            self.gaps += 1
            self.healthy = False
            self.stale_reason = "sequence_gap"
            self.snapshot_ready = False
            return False
        for p, q in bids:
            self._upsert(self.bids, float(p), float(q))
        for p, q in asks:
            self._upsert(self.asks, float(p), float(q))
        self.last_update_id = v
        self.last_ts = t
        return self._validate()

    def force_rebuild(
        self,
        bids: list[list[float]] | list[tuple[float, float]],
        asks: list[list[float]] | list[tuple[float, float]],
        last_update_id: int,
        ts: float | None = None,
    ) -> bool:
        self.rebuilds += 1
        self.snapshot_ready = False
        self.last_update_id = 0
        return self.apply_snapshot(bids, asks, last_update_id, ts)

    def mark_reconnect(self) -> None:
        self.snapshot_ready = False
        self.healthy = False
        self.stale_reason = "reconnect"

    @staticmethod
    def _upsert(side: dict[float, float], price: float, qty: float) -> None:
        if qty <= 0:
            side.pop(price, None)
        else:
            side[price] = qty

    def _validate(self) -> bool:
        if not self.bids or not self.asks:
            self.healthy = False
            self.stale_reason = "empty_side"
            return False
        best_bid = max(self.bids)
        best_ask = min(self.asks)
        if best_bid >= best_ask:
            self.healthy = False
            self.stale_reason = "crossed_book"
            return False
        self.healthy = True
        self.stale_reason = ""
        return True

    def age_sec(self, now: float | None = None) -> float:
        t = float(now if now is not None else _now())
        if self.last_ts <= 0:
            return 1e9
        return max(0.0, t - self.last_ts)

    def is_authoritative(self, now: float | None = None) -> bool:
        if not self.healthy or not self.snapshot_ready:
            return False
        if self.age_sec(now) > STALE_SEC:
            self.healthy = False
            self.stale_reason = "stale"
            return False
        return True

    def best_bid(self) -> tuple[float, float]:
        if not self.bids:
            return 0.0, 0.0
        px = max(self.bids)
        return px, self.bids[px]

    def best_ask(self) -> tuple[float, float]:
        if not self.asks:
            return 0.0, 0.0
        px = min(self.asks)
        return px, self.asks[px]

    def levels(self, n: int = 20) -> tuple[list[tuple[float, float]], list[tuple[float, float]]]:
        bids = sorted(self.bids.items(), key=lambda x: -x[0])[:n]
        asks = sorted(self.asks.items(), key=lambda x: x[0])[:n]
        return bids, asks

    def as_dict(self) -> dict[str, Any]:
        bb, bq = self.best_bid()
        ba, aq = self.best_ask()
        return {
            "symbol": self.symbol,
            "healthy": self.is_authoritative(),
            "stale_reason": self.stale_reason,
            "last_update_id": self.last_update_id,
            "age_sec": round(self.age_sec(), 4),
            "best_bid": bb,
            "best_bid_sz": bq,
            "best_ask": ba,
            "best_ask_sz": aq,
            "rebuilds": self.rebuilds,
            "gaps": self.gaps,
            "duplicates": self.duplicates,
        }


_BOOKS: dict[str, LocalL2Book] = {}


def _key(symbol: str) -> str:
    return symbol.upper().replace("/", "").replace("USDT", "") or symbol.upper()


def book_for(symbol: str) -> LocalL2Book:
    k = _key(symbol)
    b = _BOOKS.get(k)
    if b is None:
        b = LocalL2Book(symbol=k)
        _BOOKS[k] = b
    return b


def apply_partial_snapshot(symbol: str, bids, asks, last_update_id: int | None = None, ts: float | None = None) -> LocalL2Book:
    book = book_for(symbol)
    book.apply_snapshot(bids, asks, last_update_id, ts)
    return book


def reset_books() -> None:
    _BOOKS.clear()


__all__ = [
    "STALE_SEC",
    "LocalL2Book",
    "apply_partial_snapshot",
    "book_for",
    "reset_books",
]
