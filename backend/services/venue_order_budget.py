"""Venue new-order budget.

Binance.US publishes an ORDERS rate limit that is separate from request weight.
CCXT's ``enableRateLimit`` only spaces HTTP calls. This gate counts new orders
against the limits the venue returned, and leaves the exit reserve free for
protective sells when entries are also submitting.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
import urllib.request
from typing import Any

logger = logging.getLogger(__name__)

_INTERVAL_SEC = {"SECOND": 1.0, "MINUTE": 60.0, "HOUR": 3600.0, "DAY": 86400.0}


def is_venue_rate_limit(error: Any) -> bool:
    """True for a venue throttle, ban, or order-count rejection."""
    text = str(error or "")
    lowered = text.lower()
    markers = ("429", "418", "-1015", "-1003", "too many", "rate limit", "retry-after", "way too much request weight")
    return any(marker in lowered for marker in markers)


def parse_order_limits(rate_limits: list[dict[str, Any]] | None) -> list[tuple[float, int]]:
    """ORDERS windows as (seconds, max new orders). Request weight is ignored."""
    parsed: list[tuple[float, int]] = []
    for row in rate_limits or []:
        if str(row.get("rateLimitType") or "").upper() != "ORDERS":
            continue
        unit = _INTERVAL_SEC.get(str(row.get("interval") or "").upper())
        if unit is None:
            continue
        try:
            window = float(row.get("intervalNum") or 1) * unit
            limit = int(row.get("limit") or 0)
        except (TypeError, ValueError):
            continue
        if window > 0 and limit > 0:
            parsed.append((window, limit))
    return parsed


def exit_reserve_slots() -> int:
    """One held slot per allowed DAY lot and per allowed SCALP lot."""
    from backend.services.portfolio_engine import DAY_MAX_OPEN_POSITIONS, SCALP_MAX_OPEN_POSITIONS

    return max(1, int(DAY_MAX_OPEN_POSITIONS) + int(SCALP_MAX_OPEN_POSITIONS))


class VenueOrderBudget:
    def __init__(self, clock: Any = None) -> None:
        self._clock = clock or time.monotonic
        self._limits: list[tuple[float, int]] = []
        self._stamps: list[float] = []
        self._loaded = False
        self._load_failed = False
        self.last_rate_limit_error = ""
        self.last_rate_limit_at: float | None = None
        self.protective_sell_attempts = 0
        self.protective_zero_fill_count = 0
        self.last_zero_fill_at: float | None = None

    def set_limits(self, rate_limits: list[dict[str, Any]] | None) -> None:
        parsed = parse_order_limits(rate_limits)
        if parsed:
            self._limits = parsed
            self._loaded = True
            self._load_failed = False

    async def ensure_limits(self) -> None:
        if self._loaded:
            return
        try:
            payload = await asyncio.to_thread(_fetch_exchange_limits)
        except Exception:
            self._load_failed = True
            logger.warning("VENUE_ORDER_LIMITS_UNAVAILABLE", exc_info=True)
            return
        self.set_limits(payload)
        if not self._loaded:
            self._load_failed = True
            logger.warning("VENUE_ORDER_LIMITS_MISSING")

    def _prune(self, now: float) -> None:
        if not self._limits:
            return
        horizon = max(window for window, _limit in self._limits)
        self._stamps = [stamp for stamp in self._stamps if now - stamp < horizon]

    def seconds_until_slot(self, priority: str) -> float:
        """How long a new order must wait. Exits may use the full venue limit."""
        if not self._limits:
            return 0.0
        now = float(self._clock())
        self._prune(now)
        reserve = 0 if str(priority) == "exit" else exit_reserve_slots()
        wait = 0.0
        for window, limit in self._limits:
            cap = limit if str(priority) == "exit" else max(1, limit - reserve)
            in_window = [stamp for stamp in self._stamps if now - stamp < window]
            if len(in_window) < cap:
                continue
            oldest = min(in_window)
            wait = max(wait, window - (now - oldest))
        return wait

    def try_acquire(self, priority: str) -> bool:
        if self.seconds_until_slot(priority) > 0:
            return False
        self._stamps.append(float(self._clock()))
        return True

    async def acquire(self, priority: str) -> None:
        if not self._loaded:
            # The venue document was unreachable. Do not invent a limit and do not block a sell.
            logger.warning("VENUE_ORDER_BUDGET_UNENFORCED priority=%s", priority)
            return
        while True:
            wait = self.seconds_until_slot(priority)
            if wait <= 0:
                self._stamps.append(float(self._clock()))
                return
            await asyncio.sleep(min(wait, 1.0))

    def note_rate_limit(self, error: Any) -> None:
        self.last_rate_limit_error = str(error or "")[:300]
        self.last_rate_limit_at = time.time()

    def note_protective_attempt(self) -> None:
        self.protective_sell_attempts += 1

    def note_protective_zero_fill(self) -> None:
        self.protective_zero_fill_count += 1
        self.last_zero_fill_at = time.time()

    def snapshot(self) -> dict[str, Any]:
        now = float(self._clock())
        self._prune(now)
        windows = []
        for window, limit in self._limits:
            used = sum(1 for stamp in self._stamps if now - stamp < window)
            windows.append({"window_sec": window, "limit": limit, "used": used})
        return {
            "order_limits_loaded": self._loaded,
            "order_count_windows": windows,
            "exit_reserve_slots": exit_reserve_slots() if self._loaded else None,
            "protective_sell_attempts": self.protective_sell_attempts,
            "protective_zero_fill_count": self.protective_zero_fill_count,
            "last_zero_fill_at": self.last_zero_fill_at,
            "last_rate_limit_error": self.last_rate_limit_error,
            "last_rate_limit_at": self.last_rate_limit_at,
        }


def _fetch_exchange_limits() -> list[dict[str, Any]]:
    url = os.getenv("BINANCEUS_EXCHANGE_INFO_URL", "https://api.binance.us/api/v3/exchangeInfo?symbol=BTCUSDT")
    with urllib.request.urlopen(url, timeout=10) as resp:
        payload = json.loads(resp.read().decode())
    limits = payload.get("rateLimits")
    return limits if isinstance(limits, list) else []


venue_order_budget = VenueOrderBudget()
