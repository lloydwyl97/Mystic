"""Streaming as-of 4H state for production-conformant DAY replay.

O(bars + evaluations): each 1-minute bar is applied once. Forming OHLC uses
only bars with timestamp <= evaluation time. Completed bars are those whose
close timestamp (open + 14400) is <= evaluation time.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

FOURH_SEC = 14400
FOURH_KEEP_MIN = 80
FOURH_KEEP_MAX = 500
EMA_ALIGN_MIN_CLOSES = 50


def _open_sec(epoch: int) -> int:
    return (int(epoch) // FOURH_SEC) * FOURH_SEC


@dataclass
class FourHAsOfTracker:
    completed: list[list[float]] = field(default_factory=list)
    bars_1m: list[tuple[int, float, float, float, float] | tuple[int, float, float, float, float, float]] = field(default_factory=list)
    keep: int = FOURH_KEEP_MAX
    ptr: int = 0
    last_now: float = -1.0
    forming_open: int | None = None
    fo: float = 0.0
    fh: float = 0.0
    fl: float = 0.0
    fc: float = 0.0
    fv: float = 0.0
    forming_last_1m_ts: int = -1
    cache: dict[int, dict[str, Any]] = field(default_factory=dict)
    stats: dict[str, int] = field(
        default_factory=lambda: {
            "advance_calls": 0,
            "cache_hits": 0,
            "bars_consumed": 0,
            "finalized": 0,
        }
    )

    def seed_completed(self, rows: list[list[float]]) -> None:
        """Seed finalized 4H bars. Each row is [open_sec, o, h, l, c, v]."""
        cleaned: list[list[float]] = []
        seen: set[int] = set()
        for r in rows:
            ot = int(float(r[0]))
            if ot in seen:
                continue
            if float(r[1]) <= 0 or float(r[4]) <= 0:
                continue
            seen.add(ot)
            cleaned.append([float(ot), float(r[1]), float(r[2]), float(r[3]), float(r[4]), float(r[5] if len(r) > 5 else 0.0)])
        cleaned.sort(key=lambda x: x[0])
        self.completed = cleaned[-self.keep :]

    def _start_forming(self, open_sec: int, o: float, h: float, low: float, c: float, v: float, ts: int) -> None:
        self.forming_open = int(open_sec)
        self.fo, self.fh, self.fl, self.fc, self.fv = float(o), float(h), float(low), float(c), float(v)
        self.forming_last_1m_ts = int(ts)

    def _finalize_current(self) -> None:
        if self.forming_open is None or self.fo <= 0:
            self.forming_open = None
            return
        row = [float(self.forming_open), self.fo, self.fh, self.fl, self.fc, self.fv]
        if not self.completed or int(self.completed[-1][0]) < self.forming_open:
            self.completed.append(row)
            if len(self.completed) > self.keep:
                self.completed = self.completed[-self.keep :]
            self.stats["finalized"] += 1
        self.forming_open = None
        self.fo = self.fh = self.fl = self.fc = self.fv = 0.0
        self.forming_last_1m_ts = -1

    def _apply_bar(self, bar: tuple[int, float, ...]) -> None:
        ep, o, h, low, c = int(bar[0]), float(bar[1]), float(bar[2]), float(bar[3]), float(bar[4])
        v = float(bar[5]) if len(bar) > 5 else 0.0
        open_sec = _open_sec(ep)
        if self.forming_open is None:
            self._start_forming(open_sec, o, h, low, c, v, ep)
            return
        if open_sec > self.forming_open:
            self._finalize_current()
            self._start_forming(open_sec, o, h, low, c, v, ep)
            return
        self.fh = max(self.fh, h)
        self.fl = min(self.fl, low)
        self.fc = c
        self.fv += v
        self.forming_last_1m_ts = int(ep)

    def _maybe_finalize(self, now: float) -> None:
        if self.forming_open is not None and now + 1e-9 >= float(self.forming_open) + FOURH_SEC:
            self._finalize_current()

    def advance(self, now: float) -> dict[str, Any]:
        now_f = float(now)
        if now_f + 1e-9 < self.last_now:
            msg = f"FourHAsOfTracker time moved backward {self.last_now} -> {now_f}"
            raise ValueError(msg)
        self.stats["advance_calls"] += 1
        cache_key = int(now_f)
        if cache_key == int(self.last_now) and cache_key in self.cache:
            self.stats["cache_hits"] += 1
            return self.cache[cache_key]
        while self.ptr < len(self.bars_1m) and self.bars_1m[self.ptr][0] <= now_f + 1e-9:
            self._apply_bar(self.bars_1m[self.ptr])
            self.ptr += 1
            self.stats["bars_consumed"] += 1
        self._maybe_finalize(now_f)
        self.last_now = now_f
        bundle = self.bundle(now_f)
        self.cache[cache_key] = bundle
        if len(self.cache) > 8:
            oldest = min(self.cache)
            del self.cache[oldest]
        return bundle

    def bundle(self, now: float) -> dict[str, Any]:
        now_f = float(now)
        completed = [r for r in self.completed if float(r[0]) + FOURH_SEC <= now_f + 1e-9]
        forming: list[float] | None = None
        if self.forming_open is not None and self.fo > 0 and float(self.forming_open) <= now_f + 1e-9 and float(self.forming_open) + FOURH_SEC > now_f + 1e-9:
            if self.forming_last_1m_ts > now_f + 1e-9:
                msg = "forming bar used a 1m timestamp after evaluation time"
                raise AssertionError(msg)
            forming = [float(self.forming_open), self.fo, self.fh, self.fl, self.fc, self.fv]
        for r in completed:
            if float(r[0]) + FOURH_SEC > now_f + 1e-9:
                msg = "completed 4H close is after evaluation time"
                raise AssertionError(msg)
        rows = completed[-self.keep :]
        if forming is not None:
            rows = [*rows, forming]
        return {
            "4h": rows,
            "_asof": {
                "n_completed": len(completed),
                "forming": forming is not None,
                "forming_last_1m_ts": self.forming_last_1m_ts if forming is not None else None,
                "align_ready": len(completed) >= EMA_ALIGN_MIN_CLOSES,
            },
        }

    def assert_as_of(self, now: float, bundle: dict[str, Any]) -> None:
        now_f = float(now)
        rows = bundle.get("4h") or []
        for r in rows:
            ot = float(r[0])
            close_ts = ot + FOURH_SEC
            if close_ts <= now_f + 1e-9:
                continue
            if self.forming_open is not None and int(ot) == int(self.forming_open):
                if self.forming_last_1m_ts > now_f + 1e-9:
                    raise AssertionError("forming 1m after now")
                continue
            raise AssertionError(f"unclosed 4H open={ot} close={close_ts} now={now_f}")


def merge_seed_and_1m_completed(
    seed_4h: list[list[float]],
    bars_1m: list[tuple[int, float, ...]],
) -> list[list[float]]:
    """Completed 4H from exchange seed; drop any seed bar that is still open at first 1m."""
    if not bars_1m:
        return seed_4h
    first_1m = int(bars_1m[0][0])
    return [r for r in seed_4h if float(r[0]) + FOURH_SEC <= first_1m + 1e-9]
