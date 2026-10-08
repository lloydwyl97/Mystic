"""Causal tape of an outside venue. A feature at time t uses only prints at or before t."""

from __future__ import annotations

import math
from collections import deque

STALE_SEC = 5.0
_WINDOWS = (5.0, 15.0)


class VenueTape:
    """Best bid/ask and signed trades for one symbol on one venue."""

    def __init__(self) -> None:
        self.book_ts = 0.0
        self.bid = 0.0
        self.ask = 0.0
        self.mids: deque[tuple[float, float]] = deque()
        self.trades: deque[tuple[float, float]] = deque()

    def add_book(self, ts: float, bid: float, ask: float) -> None:
        if bid <= 0.0 or ask <= 0.0 or ask < bid or not math.isfinite(ts):
            return
        self.book_ts = float(ts)
        self.bid = float(bid)
        self.ask = float(ask)
        self.mids.append((self.book_ts, (self.bid + self.ask) / 2.0))
        self._trim(self.book_ts)

    def add_trade(self, ts: float, qty: float, buy_aggressor: bool) -> None:
        if qty <= 0.0 or not math.isfinite(ts):
            return
        signed = float(qty) if buy_aggressor else -float(qty)
        self.trades.append((float(ts), signed))
        self._trim(float(ts))

    def _trim(self, now: float) -> None:
        cutoff = now - 60.0
        while self.mids and self.mids[0][0] < cutoff:
            self.mids.popleft()
        while self.trades and self.trades[0][0] < cutoff:
            self.trades.popleft()

    def _return(self, now: float, window: float) -> float | None:
        if not self.mids:
            return None
        mid_now = self.mids[-1][1]
        if mid_now <= 0.0 or self.mids[-1][0] > now:
            return None
        target = now - window
        prior = None
        for ts, mid in self.mids:
            if ts <= target:
                prior = mid
            elif ts <= now:
                break
        if prior is None or prior <= 0.0:
            return None
        return mid_now / prior - 1.0

    def _flow(self, now: float, window: float) -> float:
        start = now - window
        return sum(qty for ts, qty in self.trades if start <= ts <= now)

    def features(self, now: float, binance_mid: float | None, binance_ret_5s: float | None) -> dict[str, float] | None:
        """None when the book is stale. A later print is not included."""
        if self.book_ts <= 0.0 or now - self.book_ts > STALE_SEC or self.bid <= 0.0:
            return None
        mid = (self.bid + self.ask) / 2.0
        ret5 = self._return(now, 5.0)
        ret15 = self._return(now, 15.0)
        if ret5 is None or ret15 is None:
            return None
        dislocation = None
        if binance_mid is not None and binance_mid > 0.0 and mid > 0.0:
            dislocation = math.log(mid / float(binance_mid))
        lead = None if binance_ret_5s is None else ret5 - float(binance_ret_5s)
        if dislocation is None or lead is None:
            return None
        return {
            "ret_5s": ret5,
            "ret_15s": ret15,
            "flow_5s": self._flow(now, 5.0),
            "spread": (self.ask - self.bid) / mid,
            "dislocation": dislocation,
            "lead_5s": lead,
        }
