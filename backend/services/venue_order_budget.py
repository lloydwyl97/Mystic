"""Venue new-order budget.

Binance.US publishes an ORDERS rate limit that is separate from request weight.
CCXT's ``enableRateLimit`` only spaces HTTP calls. This gate counts new orders
against the limits the venue returned, and leaves the exit reserve free for
protective sells when entries are also submitting.

The 10-second window is aged with a monotonic clock. A restart cannot see the
previous process's stamps, so the first short window after start is not treated
as empty. The daily count is the UTC day named by exchangeInfo's DAY interval
and is stored on disk so a restart does not begin that day at zero.
"""

from __future__ import annotations

import asyncio
import fcntl
import json
import logging
import os
import threading
import time
import urllib.request
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_INTERVAL_SEC = {"SECOND": 1.0, "MINUTE": 60.0, "HOUR": 3600.0, "DAY": 86400.0}
_HELD_WRITER_LOCKS: list[Any] = []


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


def hold_writer_lock(path: str) -> Any | None:
    """Hold an exclusive lock for this process. A second writer gets None."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    handle = open(path, "a+")
    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        handle.close()
        return None
    handle.seek(0)
    handle.truncate()
    handle.write(str(os.getpid()))
    handle.flush()
    _HELD_WRITER_LOCKS.append(handle)
    return handle


class VenueOrderBudget:
    def __init__(self, clock: Any = None, state_path: str | None = None, day_fn: Any = None) -> None:
        self._clock = clock or time.monotonic
        self._state_path = state_path
        self._day_fn = day_fn or (lambda: time.strftime("%Y-%m-%d", time.gmtime()))
        self._lock = threading.Lock()
        self._limits: list[tuple[float, int]] = []
        self._stamps: list[float] = []
        self._loaded = False
        self._load_failed = False
        self._blackout_until = 0.0
        self._day_count = 0
        self._day_key = ""
        self._day_from_disk = False
        self._day_from_venue = False
        self._short_from_venue = False
        self._venue_short_read_at = -1.0
        self._external_detected = False
        self.last_rate_limit_error = ""
        self.last_rate_limit_at: float | None = None
        self.protective_sell_attempts = 0
        self.protective_zero_fill_count = 0
        self.last_zero_fill_at: float | None = None
        if state_path:
            self._load_day()

    def set_limits(self, rate_limits: list[dict[str, Any]] | None) -> None:
        parsed = parse_order_limits(rate_limits)
        if parsed:
            with self._lock:
                self._limits = parsed
                self._loaded = True
                self._load_failed = False

    def arm_restart_blackout(self) -> None:
        """The previous process's 10-second orders are invisible. Do not start at zero."""
        with self._lock:
            self._blackout_until = float(self._clock()) + self._short_window_unlocked()

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

    def _short_window_unlocked(self) -> float:
        shorts = [window for window, _limit in self._limits if window <= 120]
        return min(shorts) if shorts else 10.0

    def _utc_day(self) -> str:
        return str(self._day_fn())

    def _load_day(self) -> None:
        try:
            payload = json.loads(Path(self._state_path or "").read_text())
        except (OSError, ValueError, TypeError):
            return
        if not isinstance(payload, dict):
            return
        today = self._utc_day()
        self._day_key = today
        if str(payload.get("utc_day") or "") == today:
            self._day_count = max(0, int(payload.get("count") or 0))
        else:
            self._day_count = 0
        self._day_from_disk = True

    def _save_day(self) -> None:
        if not self._state_path:
            return
        path = Path(self._state_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(json.dumps({"utc_day": self._day_key, "count": self._day_count}))
        tmp.replace(path)

    def _seconds_to_utc_midnight(self) -> float:
        now = time.time()
        return max(0.0, ((int(now // 86400) + 1) * 86400) - now)

    def _wait_unlocked(self, priority: str) -> float:
        """Seconds until a slot exists. Caller holds the lock. Negative means granted."""
        now = float(self._clock())
        if now < self._blackout_until:
            return self._blackout_until - now
        if not self._limits:
            return -1.0
        self._stamps = [stamp for stamp in self._stamps if now - stamp < self._short_window_unlocked()]
        today = self._utc_day()
        if self._day_key != today:
            self._day_key = today
            self._day_count = 0
            self._day_from_disk = True
        reserve = 0 if str(priority) == "exit" else exit_reserve_slots()
        wait = 0.0
        for window, limit in self._limits:
            cap = limit if str(priority) == "exit" else max(1, limit - reserve)
            if window >= 86400:
                if self._day_count >= cap:
                    wait = max(wait, self._seconds_to_utc_midnight())
                continue
            in_window = [stamp for stamp in self._stamps if now - stamp < window]
            if len(in_window) < cap:
                continue
            wait = max(wait, window - (now - min(in_window)))
        return wait

    def _charge_unlocked(self) -> None:
        now = float(self._clock())
        self._stamps.append(now)
        today = self._utc_day()
        if self._day_key != today:
            self._day_key = today
            self._day_count = 0
            self._day_from_disk = True
        self._day_count += 1
        self._save_day()

    def apply_venue_usage(self, rows: list[dict[str, Any]] | None) -> bool:
        """Raise local usage to the venue's reported count. Never lower it."""
        with self._lock:
            now = float(self._clock())
            applied = False
            for row in rows or []:
                if str(row.get("rateLimitType") or "").upper() != "ORDERS":
                    continue
                unit = _INTERVAL_SEC.get(str(row.get("interval") or "").upper())
                if unit is None:
                    continue
                try:
                    window = float(row.get("intervalNum") or 1) * unit
                    count = int(row.get("count") or 0)
                except (TypeError, ValueError):
                    continue
                if window <= 0 or count < 0:
                    continue
                if window >= 86400:
                    if count > self._day_count:
                        self._external_detected = True
                    self._day_count = max(self._day_count, count)
                    self._day_key = self._utc_day()
                    self._day_from_disk = True
                    self._day_from_venue = True
                    self._save_day()
                else:
                    local = sum(1 for stamp in self._stamps if now - stamp < window)
                    if count > local:
                        self._external_detected = True
                        self._stamps.extend([now] * (count - local))
                    self._short_from_venue = True
                    self._venue_short_read_at = now
                applied = True
            if applied:
                self._blackout_until = 0.0
            return applied

    async def refresh_order_usage(self) -> bool:
        """Read account order-count usage. Does not place an order."""
        try:
            rows = await asyncio.to_thread(_fetch_order_usage)
        except Exception:
            logger.warning("VENUE_ORDER_USAGE_UNAVAILABLE", exc_info=True)
            return False
        return self.apply_venue_usage(rows)

    def try_acquire(self, priority: str) -> bool:
        with self._lock:
            if not self._loaded:
                return True
            wait = self._wait_unlocked(priority)
            if wait > 0:
                return False
            self._charge_unlocked()
            return True

    async def acquire(self, priority: str) -> None:
        if self._loaded:
            now = float(self._clock())
            stale = self._venue_short_read_at < 0 or now - self._venue_short_read_at >= self._short_window_unlocked()
            if stale:
                await self.refresh_order_usage()
        while True:
            with self._lock:
                if not self._loaded:
                    logger.warning("VENUE_ORDER_BUDGET_UNENFORCED priority=%s", priority)
                    return
                wait = self._wait_unlocked(priority)
                if wait <= 0:
                    self._charge_unlocked()
                    return
            await asyncio.sleep(min(wait, 1.0))

    def note_rate_limit(self, error: Any) -> None:
        """A venue rejection means the local window was too optimistic. Fill it."""
        text = str(error or "")
        lowered = text.lower()
        with self._lock:
            self.last_rate_limit_error = text[:300]
            self.last_rate_limit_at = time.time()
            now = float(self._clock())
            day_hit = "day" in lowered
            short_hit = "second" in lowered or not day_hit
            for window, limit in self._limits:
                if window >= 86400 and day_hit:
                    self._day_key = self._utc_day()
                    self._day_count = max(self._day_count, limit)
                    self._day_from_disk = True
                    self._save_day()
                elif window < 86400 and short_hit:
                    in_window = [stamp for stamp in self._stamps if now - stamp < window]
                    missing = limit - len(in_window)
                    if missing > 0:
                        self._stamps.extend([now] * missing)

    def note_protective_attempt(self) -> None:
        self.protective_sell_attempts += 1

    def note_protective_zero_fill(self) -> None:
        self.protective_zero_fill_count += 1
        self.last_zero_fill_at = time.time()

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            now = float(self._clock())
            blackout = now < self._blackout_until
            wait = 0.0 if not self._limits else max(0.0, self._wait_unlocked("exit"))
            short = self._short_window_unlocked() if self._limits else 10.0
            used_10 = sum(1 for stamp in self._stamps if now - stamp < short)
            limit_10 = next((limit for window, limit in self._limits if abs(window - short) < 1e-9), None)
            limit_day = next((limit for window, limit in self._limits if window >= 86400), None)
            return {
                "order_limits_loaded": self._loaded,
                "orders_used_10s_local": None if blackout or not self._loaded else used_10,
                "orders_used_10s": None if blackout or not self._loaded else used_10,
                "orders_limit_10s": limit_10,
                "orders_10s_authoritative": bool(self._short_from_venue and not blackout),
                "orders_used_day_local": self._day_count if (self._day_from_disk or self._day_from_venue or self._day_count > 0) else None,
                "orders_used_day": self._day_count if (self._day_from_disk or self._day_from_venue or self._day_count > 0) else None,
                "orders_limit_day": limit_day,
                "orders_day_authoritative": bool(self._day_from_venue),
                "orders_day_scope": "venue_account" if self._day_from_venue else "mystic_recorded",
                "external_order_activity_detected": self._external_detected,
                "account_exclusive_assumption": False,
                "exit_reserved": exit_reserve_slots() if self._loaded else None,
                "next_slot_available_at": None if wait <= 0 else time.time() + wait,
                "protective_sell_attempts": self.protective_sell_attempts,
                "protective_zero_fill_count": self.protective_zero_fill_count,
                "last_zero_fill_at": self.last_zero_fill_at,
                "last_rate_limit_error": self.last_rate_limit_error,
                "last_rate_limit_at": self.last_rate_limit_at,
            }


def _fetch_order_usage() -> list[dict[str, Any]]:
    import hashlib
    import hmac
    import urllib.parse

    from backend.utils.binance_credentials import get_binance_us_credentials

    api_key, secret = get_binance_us_credentials()
    if not api_key or not secret:
        raise RuntimeError("missing_binance_credentials")
    params = {"timestamp": int(time.time() * 1000), "recvWindow": 5000}
    query = urllib.parse.urlencode(params)
    signature = hmac.new(secret.encode(), query.encode(), hashlib.sha256).hexdigest()
    request = urllib.request.Request(
        "https://api.binance.us/api/v3/rateLimit/order?" + query + "&signature=" + signature,
        headers={"X-MBX-APIKEY": api_key},
    )
    with urllib.request.urlopen(request, timeout=10) as response:
        payload = json.loads(response.read().decode())
    return payload if isinstance(payload, list) else []


def _fetch_exchange_limits() -> list[dict[str, Any]]:
    url = os.getenv("BINANCEUS_EXCHANGE_INFO_URL", "https://api.binance.us/api/v3/exchangeInfo?symbol=BTCUSDT")
    with urllib.request.urlopen(url, timeout=10) as resp:
        payload = json.loads(resp.read().decode())
    limits = payload.get("rateLimits")
    return limits if isinstance(limits, list) else []


venue_order_budget = VenueOrderBudget(state_path=os.getenv("MYSTIC_VENUE_ORDER_BUDGET_STATE", "/home/mystic/mystic/logs/venue_order_budget.json"))
