"""DAY V2 cycle inputs: one as-of, completed candle first, fresh executable price.

A missing price is a hard data rejection. Zero is not a price.
"""

from __future__ import annotations

from typing import Any

HARD_MISSING_PRICE = "MISSING_EXECUTABLE_PRICE"
HARD_MISSING_CANDLE = "MISSING_COMPLETED_CANDLE"
_FIFTEEN_MIN = 900


def required_15m_open(as_of: float) -> float:
    """Open time of the last 15m bar that has closed at as_of.

    An unaligned clock must not demand a candle that has not closed yet.
    """
    aligned = (int(as_of) // _FIFTEEN_MIN) * _FIFTEEN_MIN
    return float(aligned - _FIFTEEN_MIN)


def cycle_decision(
    *,
    completed_bar_count: int,
    minimum_bars: int,
    executable_price: float,
    book_age_sec: float | None,
    book_stale_sec: float,
    already_evaluated: bool,
    retried: bool,
    latest_bar_epoch: float | None = None,
    required_open_epoch: float | None = None,
) -> dict[str, Any]:
    """Return proceed, retry, or one hard rejection for this causal cycle."""
    if already_evaluated:
        return {"action": "skip", "reason": "DUPLICATE_EVALUATION"}
    candle_fresh = required_open_epoch is None or (latest_bar_epoch is not None and float(latest_bar_epoch) + 1.0 >= float(required_open_epoch))
    candle_ready = completed_bar_count >= minimum_bars and candle_fresh
    price_ready = executable_price > 0 and (book_age_sec is None or book_age_sec <= book_stale_sec)
    if candle_ready and price_ready:
        return {"action": "proceed", "reason": "READY", "price": float(executable_price)}
    if not retried and (not candle_ready or not price_ready):
        return {"action": "retry", "reason": "WAITING_FOR_FRESH_DATA"}
    if not price_ready:
        return {"action": "reject", "reason": HARD_MISSING_PRICE}
    return {"action": "reject", "reason": HARD_MISSING_CANDLE}
