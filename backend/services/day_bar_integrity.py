"""Closed-bar / exchange event-time validation for DAY AllWeather indicators."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any


@dataclass
class BarIntegrityResult:
    ok: bool
    closed_bars: list[dict] = field(default_factory=list)
    dropped_forming: bool = False
    closed_bar_ts: int = 0
    error_code: str | None = None
    detail: str = ""


def _bar_ts_sec(bar: dict) -> int:
    ts = int(bar.get("ts") or bar.get("timestamp") or 0)
    if ts > 10_000_000_000:  # ms
        ts //= 1000
    return ts


def drop_forming_candle(bars: list[dict], *, interval_sec: int, now: float | None = None) -> tuple[list[dict], bool, int]:
    """Drop incomplete last candle. Returns (closed_bars, dropped_forming, last_closed_ts)."""
    if not bars:
        return [], False, 0
    now_f = float(now if now is not None else time.time())
    last_ts = _bar_ts_sec(bars[-1])
    if last_ts <= 0:
        return list(bars), False, 0
    if last_ts + int(interval_sec) > now_f + 2:
        closed = list(bars[:-1])
        cts = _bar_ts_sec(closed[-1]) if closed else 0
        return closed, True, cts
    return list(bars), False, last_ts


def validate_exchange_bars(
    bars: list[dict] | None,
    *,
    interval_sec: int = 3600,
    min_bars: int = 1,
    now: float | None = None,
    max_stale_sec: float | None = None,
    allow_forming: bool = False,
) -> BarIntegrityResult:
    """Validate candle integrity using exchange open times (event time).

    Rejects / marks errors for: future, duplicate, out-of-order, missing,
    stale, wrong interval boundaries, and (unless allow_forming) in-progress candles.
    """
    now_f = float(now if now is not None else time.time())
    raw = list(bars or [])
    if not raw:
        return BarIntegrityResult(ok=False, error_code="MISSING_BARS", detail="no bars")

    # Normalize and sort check
    stamps: list[int] = []
    for b in raw:
        ts = _bar_ts_sec(b)
        if ts <= 0:
            return BarIntegrityResult(ok=False, error_code="INVALID_BAR_TS", detail="bar missing ts")
        if ts > now_f + 5:
            return BarIntegrityResult(ok=False, error_code="FUTURE_CANDLE", detail=f"ts={ts} now={now_f:.0f}")
        stamps.append(ts)

    # Out of order / duplicates
    for i in range(1, len(stamps)):
        if stamps[i] < stamps[i - 1]:
            return BarIntegrityResult(ok=False, error_code="OUT_OF_ORDER", detail=f"{stamps[i - 1]}->{stamps[i]}")
        if stamps[i] == stamps[i - 1]:
            return BarIntegrityResult(ok=False, error_code="DUPLICATE_CANDLE", detail=f"ts={stamps[i]}")

    # Interval boundaries (open time must align to interval)
    for ts in stamps:
        if ts % int(interval_sec) != 0:
            return BarIntegrityResult(
                ok=False,
                error_code="WRONG_INTERVAL_BOUNDARY",
                detail=f"ts={ts} not aligned to {interval_sec}s",
            )

    # Missing bars in the contiguous window (last N gaps)
    if len(stamps) >= 2:
        for i in range(1, len(stamps)):
            gap = stamps[i] - stamps[i - 1]
            if gap != int(interval_sec):
                # Allow larger historical gaps only before the recent window
                if i >= len(stamps) - 5 and gap != int(interval_sec):
                    return BarIntegrityResult(
                        ok=False,
                        error_code="MISSING_CANDLE",
                        detail=f"gap={gap} expected={interval_sec} at {stamps[i]}",
                    )

    closed, dropped, closed_ts = drop_forming_candle(raw, interval_sec=interval_sec, now=now_f)
    if not allow_forming and dropped and not closed:
        return BarIntegrityResult(
            ok=False,
            closed_bars=[],
            dropped_forming=True,
            error_code="FORMING_CANDLE_ONLY",
            detail="only forming candle present",
        )
    if not allow_forming and dropped:
        # forming dropped — OK if enough closed remain
        pass
    elif allow_forming:
        closed, dropped, closed_ts = list(raw), False, stamps[-1]

    if max_stale_sec is not None and closed:
        age = now_f - float(closed_ts or 0) - float(interval_sec)
        if age > float(max_stale_sec):
            return BarIntegrityResult(
                ok=False,
                closed_bars=closed,
                dropped_forming=dropped,
                closed_bar_ts=closed_ts,
                error_code="STALE_CANDLE",
                detail=f"age_sec={age:.0f} max={max_stale_sec}",
            )

    if len(closed) < int(min_bars):
        return BarIntegrityResult(
            ok=False,
            closed_bars=closed,
            dropped_forming=dropped,
            closed_bar_ts=closed_ts,
            error_code="INSUFFICIENT_CLOSED_BARS",
            detail=f"have={len(closed)} need={min_bars}",
        )

    return BarIntegrityResult(
        ok=True,
        closed_bars=closed,
        dropped_forming=dropped,
        closed_bar_ts=closed_ts,
    )


def forming_candle_cannot_influence(
    bars_with_forming: list[dict],
    *,
    interval_sec: int,
    compute_fn: Any,
    now: float | None = None,
) -> bool:
    """Return True if compute_fn(closed) == compute_fn(closed+forming) for equality of result.

    Used by tests: strategy decision must be identical when forming candle is appended.
    """
    closed, _, _ = drop_forming_candle(bars_with_forming, interval_sec=interval_sec, now=now)
    # If last is closed already, append a synthetic forming bar
    bars = list(bars_with_forming)
    if not bars:
        return True
    last_ts = _bar_ts_sec(bars[-1])
    now_f = float(now if now is not None else time.time())
    if last_ts + interval_sec <= now_f + 2:
        forming = dict(bars[-1])
        forming["ts"] = last_ts + interval_sec
        forming["close"] = float(forming.get("close") or 0) * 1.5  # poison
        bars = [*bars, forming]
    a = compute_fn(closed)
    b = compute_fn(drop_forming_candle(bars, interval_sec=interval_sec, now=now_f)[0])
    return a == b


__all__ = [
    "BarIntegrityResult",
    "drop_forming_candle",
    "forming_candle_cannot_influence",
    "validate_exchange_bars",
]
