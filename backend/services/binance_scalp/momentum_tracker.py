"""Short-term bid/mid momentum samples for scalp entry gate."""

from __future__ import annotations

import math
import os
from collections import deque
from dataclasses import dataclass


def _float_env(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None or not str(raw).strip():
        return default
    return float(raw)


def _int_env(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or not str(raw).strip():
        return default
    return int(raw)


@dataclass(frozen=True)
class MomentumDiagnostics:
    mid_change_15s: float
    mid_change_30s: float
    mid_change_60s: float
    bid_change_15s: float
    bid_change_30s: float
    bid_change_60s: float
    last_n_ticks_up_count: int
    sample_count: int
    history_sec: float
    recent_range_pct: float
    realized_volatility_pct: float
    momentum_confirmed: bool
    flat_regime: bool

    def as_dict(self) -> dict:
        return {
            "mid_change_15s": self.mid_change_15s,
            "mid_change_30s": self.mid_change_30s,
            "mid_change_60s": self.mid_change_60s,
            "bid_change_15s": self.bid_change_15s,
            "bid_change_30s": self.bid_change_30s,
            "bid_change_60s": self.bid_change_60s,
            "last_n_ticks_up_count": self.last_n_ticks_up_count,
            "sample_count": self.sample_count,
            "history_sec": self.history_sec,
            "recent_range_pct": self.recent_range_pct,
            "realized_volatility_pct": self.realized_volatility_pct,
            "momentum_confirmed": self.momentum_confirmed,
            "flat_regime": self.flat_regime,
        }


class MomentumTracker:
    """In-memory bid/mid history per symbol — scalp package only."""

    def __init__(self, *, max_age_sec: float = 90.0, compare_ticks: int = 6) -> None:
        self._max_age_sec = max_age_sec
        self._compare_ticks = compare_ticks
        self._history: dict[str, deque[tuple[float, float, float]]] = {}

    def record(self, symbol: str, ts: float, bid: float, mid: float) -> None:
        sym = symbol.strip().upper()
        if sym not in self._history:
            self._history[sym] = deque(maxlen=256)
        self._history[sym].append((ts, bid, mid))
        self._prune(sym, ts)

    def _prune(self, symbol: str, now: float) -> None:
        hist = self._history.get(symbol)
        if not hist:
            return
        cutoff = now - self._max_age_sec
        while hist and hist[0][0] < cutoff:
            hist.popleft()

    def _sample_at(self, symbol: str, now: float, age_sec: float) -> tuple[float, float] | None:
        hist = self._history.get(symbol)
        if not hist:
            return None
        target = now - age_sec
        best: tuple[float, float, float] | None = None
        best_delta = 1e9
        for ts, bid, mid in hist:
            delta = abs(ts - target)
            if delta < best_delta:
                best_delta = delta
                best = (ts, bid, mid)
        if best is None or best_delta > age_sec * 0.5:
            return None
        return best[1], best[2]

    def diagnostics(self, symbol: str, now: float, bid: float, mid: float) -> MomentumDiagnostics:
        sym = symbol.strip().upper()
        hist = self._history.get(sym, deque())
        sample_count = len(hist)
        history_sec = (now - hist[0][0]) if hist else 0.0

        def _chg(age: float, cur: float, field: str) -> float:
            old = self._sample_at(sym, now, age)
            if old is None:
                return 0.0
            old_val = old[0] if field == "bid" else old[1]
            if old_val <= 0:
                return 0.0
            return (cur - old_val) / old_val

        mid15 = _chg(15.0, mid, "mid")
        mid30 = _chg(30.0, mid, "mid")
        mid60 = _chg(60.0, mid, "mid")
        bid15 = _chg(15.0, bid, "bid")
        bid30 = _chg(30.0, bid, "bid")
        bid60 = _chg(60.0, bid, "bid")

        mids = [h[2] for h in hist]
        recent_range = 0.0
        if mids and mid > 0:
            recent_range = (max(mids) - min(mids)) / mid
        rets: list[float] = []
        if len(hist) >= 2:
            for i in range(1, len(hist)):
                p0, p1 = hist[i - 1][2], hist[i][2]
                if p0 > 0:
                    rets.append((p1 - p0) / p0)
        realized_vol = math.sqrt(sum(r * r for r in rets) / len(rets)) if rets else 0.0

        up_count = 0
        if len(hist) >= 2:
            n = min(self._compare_ticks, len(mids) - 1)
            for i in range(len(mids) - n, len(mids)):
                if mids[i] > mids[i - 1]:
                    up_count += 1

        min_change = _float_env("SCALP_MOMENTUM_MIN_CHANGE_PCT", 0.00003)
        min_up = _int_env("SCALP_MOMENTUM_MIN_UP_TICKS", 3)
        flat_threshold = _float_env("SCALP_MOMENTUM_FLAT_THRESHOLD_PCT", 0.00002)
        min_history = _float_env("SCALP_MOMENTUM_MIN_HISTORY_SEC", 30.0)

        flat = abs(mid15) <= flat_threshold and abs(bid15) <= flat_threshold
        has_history = history_sec >= min_history and sample_count >= 4
        rising_15 = bid15 >= min_change and mid15 >= min_change
        rising_30 = bid30 > 0.0 and mid30 > 0.0
        rising_60 = bid60 > 0.0 and mid60 > 0.0
        confirmed = has_history and rising_15 and rising_30 and rising_60 and up_count >= min_up and not flat

        return MomentumDiagnostics(
            mid_change_15s=mid15,
            mid_change_30s=mid30,
            mid_change_60s=mid60,
            bid_change_15s=bid15,
            bid_change_30s=bid30,
            bid_change_60s=bid60,
            last_n_ticks_up_count=up_count,
            sample_count=sample_count,
            history_sec=history_sec,
            recent_range_pct=recent_range,
            realized_volatility_pct=realized_vol,
            momentum_confirmed=confirmed,
            flat_regime=flat,
        )
