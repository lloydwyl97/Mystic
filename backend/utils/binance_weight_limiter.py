from __future__ import annotations

import asyncio
import contextlib
import logging
import os
import time
from collections import defaultdict
from typing import Any

from backend.config.mystic_api_schedule import BINANCEUS_WEIGHT_PER_MIN
from backend.config.protected_execution import MANDATORY_EXIT_SAME_CALL_ATTEMPTS
from backend.metrics import (
    limiter_circuit_open,
    limiter_consume_wait_seconds,
    limiter_consumes_total,
    limiter_denied_total,
    limiter_tokens,
)
from backend.services.redis_service import get_redis_service, get_sync_redis_service

logger = logging.getLogger(__name__)

WEIGHT_BUDGET_PER_MIN = BINANCEUS_WEIGHT_PER_MIN
CIRCUIT_TTL_SEC = int(os.getenv("BINANCEUS_CIRCUIT_TTL", "30"))  # Reduced for faster recovery
# Consume wait timeouts (seconds) — critical OHLCV paths wait longer under bucket contention.
LIMITER_CONSUME_TIMEOUT_CRITICAL = float(os.getenv("BINANCE_LIMITER_CONSUME_TIMEOUT_CRITICAL_SEC", "12"))
LIMITER_CONSUME_TIMEOUT_DEFAULT = float(os.getenv("BINANCE_LIMITER_CONSUME_TIMEOUT_SEC", "8"))
LIMITER_CONSUME_TIMEOUT_LOOP = float(os.getenv("BINANCE_LIMITER_CONSUME_TIMEOUT_LOOP_SEC", "5"))
OHLCV_STALE_FALLBACK_MAX_AGE_SEC = float(os.getenv("BINANCE_OHLCV_STALE_FALLBACK_MAX_AGE_SEC", "150"))
# All Live Data, No Fallback/Hardcoded Data
# REDIS_URL validation deferred to create() method to allow module import without env vars

# Weight logging counters (in-memory, 60s rolling window)
_weight_counters: dict[str, dict[str, int]] = defaultdict(lambda: {"calls": 0, "weight": 0})
_weight_log_last_time: float = 0.0
_weight_log_interval: float = 60.0

# Rate limiter timing constants
TOKEN_CHECK_INITIAL_BACKOFF = 0.001  # 1ms initial backoff for token checks
TOKEN_CHECK_MAX_BACKOFF = 0.1  # 100ms maximum backoff to prevent excessive waiting
TOKEN_CHECK_BACKOFF_MULTIPLIER = 1.5  # Exponential growth factor for backoff

# Weights below were read from x-mbx-used-weight-1m deltas on Binance.US.
# ticker/price stayed at 1: a single-symbol pair was contaminated by other
# traffic, so it is not claimed as a measured cost. POST and DELETE /order
# were not sent. myTrades is the weight header on the first request of a new
# venue minute (20), which can include other traffic in that same instant.
ENDPOINT_WEIGHTS: dict[str, int] = {
    "/api/v3/ticker/price": 1,
    "/api/v3/ticker/bookTicker": 2,
    "/api/v3/ticker/24hr": 2,
    "/api/v3/klines": 2,
    "/api/v3/depth": 5,
    "/api/v3/trades": 1,
    "/api/v3/order": 4,
    "/api/v3/order/test": 1,
    "/api/v3/openOrders": 8,
    "/api/v3/allOrders": 10,
    "/api/v3/myTrades": 20,
    "/api/v3/account": 20,
    "/api/v3/exchangeInfo": 20,
    "/api/v3/rateLimit/order": 40,
    "/api/v3/time": 1,
    "/api/v3/ping": 1,
    "/api/v3/avgPrice": 1,
    "/sapi/v1/capital/withdraw/apply": 10,
    "/sapi/v1/capital/config/getall": 10,
}

# POST /api/v3/order was not placed. Binance.com publishes 1; the venue header
# raises the shared bucket after the response if the IP total is higher.
ORDER_SUBMIT_WEIGHT = 1
ORDER_CANCEL_WEIGHT = 1
OPEN_ORDERS_ALL_SYMBOLS_WEIGHT = 85
ACCOUNT_WEIGHT = 20
EXCHANGE_INFO_WEIGHT = 20
ORDER_USAGE_WEIGHT = 40
BOOK_TICKER_WEIGHT = 2
TICKER_24H_WEIGHT = 2
KLINE_WEIGHT = 2
KLINE_1000_WEIGHT = 10
DEPTH_WEIGHT = 5
MY_TRADES_WEIGHT = 20
ORDER_STATUS_WEIGHT = 4

CHARGED_KEY = "bwl:charged"
VENUE_USED_KEY = "bwl:venue_used"
VENUE_USED_AT_KEY = "bwl:venue_used_at"
VENUE_HW_KEY = "bwl:venue_hw"
VENUE_HW_AT_KEY = "bwl:venue_hw_at"
VENUE_HW_WINDOW_KEY = "bwl:venue_hw_window"
WINDOW_KEY = "bwl:window_id"
PROTECTIVE_KEY = "bwl:protective_pending"
EXIT_USED_KEY = "bwl:exit_weight_used"
DEFERRED_KEY = "bwl:deferred_weight"
EXIT_REFUSAL_KEY = "bwl:last_exit_refusal"
# 4 DAY + 4 SCALP lots. One clean attempt is a fresh book plus the submit.
# The mandatory loop can repeat that pair in the same minute. One in-flight
# order may also need a status read, myTrades, and a cancel.
PROTECTIVE_POSITION_SLOTS = 8
PROTECTIVE_ATTEMPT_WEIGHT = DEPTH_WEIGHT + ORDER_SUBMIT_WEIGHT
PROTECTIVE_RECOVERY_WEIGHT = ORDER_STATUS_WEIGHT + MY_TRADES_WEIGHT + ORDER_CANCEL_WEIGHT
PROTECTIVE_WEIGHT_RESERVE = PROTECTIVE_POSITION_SLOTS * MANDATORY_EXIT_SAME_CALL_ATTEMPTS * PROTECTIVE_ATTEMPT_WEIGHT + PROTECTIVE_RECOVERY_WEIGHT
_PRIORITY_CODE = {"background": 0, "entry": 1, "exit": 2, "exit_decision": 2}
# A venue minute is 60 seconds. A sample older than that cannot still be the open minute.
VENUE_SAMPLE_MAX_AGE_SEC = 60

# Roll and charge are one script so two processes cannot both refill.
_CONSUME_LUA = """
local now = tonumber(ARGV[1])
local budget = tonumber(ARGV[2])
local weight = tonumber(ARGV[3])
local max_age = tonumber(ARGV[4])
if redis.call('GET', KEYS[7]) == '1' then
  return 0
end
local window = math.floor(now / 60)
local stored = tonumber(redis.call('GET', KEYS[4]))
local tokens = tonumber(redis.call('GET', KEYS[1]))
if stored == nil or stored ~= window then
  local hw = tonumber(redis.call('GET', KEYS[5]))
  local hw_at = tonumber(redis.call('GET', KEYS[6]))
  if hw == nil then
    hw = tonumber(redis.call('GET', KEYS[10]))
    hw_at = tonumber(redis.call('GET', KEYS[11]))
    if hw ~= nil then
      redis.call('SET', KEYS[5], hw)
      if hw_at ~= nil then
        redis.call('SET', KEYS[6], hw_at)
      end
    end
  end
  if hw == nil or hw_at == nil or (now - hw_at) >= max_age then
    tokens = budget
  else
    tokens = budget - hw
    if tokens < 0 then tokens = 0 end
  end
  redis.call('SET', KEYS[1], tokens)
  redis.call('SET', KEYS[4], window)
  redis.call('SET', KEYS[2], (window + 1) * 60)
  redis.call('SET', KEYS[3], 0)
  redis.call('SET', KEYS[13], 0)
  redis.call('SET', KEYS[14], 0)
end
local priority = tonumber(ARGV[5])
local reserve = tonumber(ARGV[6])
if tokens == nil or tokens < weight then
  if priority >= 2 then
    redis.call('SET', KEYS[15], ARGV[1])
  end
  return 0
end
if priority < 2 then
  local pending = tonumber(redis.call('GET', KEYS[12])) or 0
  if pending > 0 or (tokens - weight) < reserve then
    redis.call('INCRBY', KEYS[14], weight)
    return 0
  end
end
redis.call('DECRBY', KEYS[1], weight)
redis.call('INCRBY', KEYS[3], weight)
redis.call('INCRBY', KEYS[8], weight)
redis.call('INCR', KEYS[9])
if priority >= 2 then
  redis.call('INCRBY', KEYS[13], weight)
end
return 1
"""

_ROLL_LUA = """
local now = tonumber(ARGV[1])
local budget = tonumber(ARGV[2])
local max_age = tonumber(ARGV[3])
local window = math.floor(now / 60)
local stored = tonumber(redis.call('GET', KEYS[4]))
if stored ~= nil and stored == window then
  return tonumber(redis.call('GET', KEYS[1])) or budget
end
local hw = tonumber(redis.call('GET', KEYS[5]))
local hw_at = tonumber(redis.call('GET', KEYS[6]))
if hw == nil then
  hw = tonumber(redis.call('GET', KEYS[7]))
  hw_at = tonumber(redis.call('GET', KEYS[8]))
  if hw ~= nil then
    redis.call('SET', KEYS[5], hw)
    if hw_at ~= nil then
      redis.call('SET', KEYS[6], hw_at)
    end
  end
end
local tokens = budget
if hw ~= nil and hw_at ~= nil and (now - hw_at) < max_age then
  tokens = budget - hw
  if tokens < 0 then tokens = 0 end
end
redis.call('SET', KEYS[1], tokens)
redis.call('SET', KEYS[4], window)
redis.call('SET', KEYS[2], (window + 1) * 60)
redis.call('SET', KEYS[3], 0)
return tokens
"""

_NOTE_LUA = """
local used = tonumber(ARGV[1])
local budget = tonumber(ARGV[2])
local now = tonumber(ARGV[4])
local started = tonumber(ARGV[5])
local max_age = tonumber(ARGV[6])
local window = math.floor(now / 60)
local stored = tonumber(redis.call('GET', KEYS[4]))
local tokens = tonumber(redis.call('GET', KEYS[1]))
if stored == nil or stored ~= window then
  local hw = tonumber(redis.call('GET', KEYS[5]))
  local hw_at = tonumber(redis.call('GET', KEYS[6]))
  if hw == nil then
    hw = tonumber(redis.call('GET', KEYS[2]))
    hw_at = tonumber(redis.call('GET', KEYS[3]))
  end
  if hw == nil or hw_at == nil or (now - hw_at) >= max_age then
    tokens = budget
  else
    tokens = budget - hw
    if tokens < 0 then tokens = 0 end
  end
  redis.call('SET', KEYS[1], tokens)
  redis.call('SET', KEYS[4], window)
  redis.call('SET', KEYS[9], (window + 1) * 60)
  redis.call('SET', KEYS[8], 0)
  stored = window
end
if tokens == nil then
  tokens = budget
end
local hw = tonumber(redis.call('GET', KEYS[5]))
local hw_at = tonumber(redis.call('GET', KEYS[6]))
local hw_window = tonumber(redis.call('GET', KEYS[7]))
if hw == nil then
  hw = tonumber(redis.call('GET', KEYS[2]))
  hw_at = tonumber(redis.call('GET', KEYS[3]))
  if hw ~= nil then
    redis.call('SET', KEYS[5], hw)
    if hw_at ~= nil then
      redis.call('SET', KEYS[6], hw_at)
    end
    if hw_window == nil then
      if hw_at ~= nil and hw_at >= window * 60 then
        hw_window = window
      else
        hw_window = window - 1
      end
      redis.call('SET', KEYS[7], hw_window)
    end
  end
end
redis.call('SET', KEYS[2], ARGV[1])
redis.call('SET', KEYS[3], ARGV[3])
if hw == nil or used >= hw then
  local cap = budget - used
  if cap < 0 then cap = 0 end
  if cap < tokens then
    tokens = cap
  end
  redis.call('SET', KEYS[1], tokens)
  redis.call('SET', KEYS[5], used)
  if started >= 0 then
    redis.call('SET', KEYS[6], ARGV[5])
  elseif hw_at == nil then
    redis.call('SET', KEYS[6], ARGV[4])
  end
  if hw_window == nil then
    redis.call('SET', KEYS[7], window)
  end
  return tokens
end
if started < 0 or hw_at == nil or started < hw_at then
  return tokens
end
local released = hw - used
local cap = budget - used
if cap < 0 then cap = 0 end
tokens = tokens + released
if tokens > cap then
  tokens = cap
end
if tokens < 0 then
  tokens = 0
end
redis.call('SET', KEYS[1], tokens)
redis.call('SET', KEYS[5], used)
redis.call('SET', KEYS[6], ARGV[5])
redis.call('SET', KEYS[7], window)
return tokens
"""


class RateLimitedErrorError(Exception):
    pass


class CircuitOpenErrorError(Exception):
    pass


# Aliases for backward compatibility and expected imports
RateLimited = RateLimitedErrorError
CircuitOpen = CircuitOpenErrorError
RateLimitedError = RateLimitedErrorError
CircuitOpenError = CircuitOpenErrorError


def kline_request_weight(limit: int) -> int:
    """limit 1..500 measured at 2. limit 1000's one sample was +8, so charge 10."""
    return KLINE_1000_WEIGHT if int(limit) >= 1000 else KLINE_WEIGHT


def depth_request_weight(limit: int) -> int:
    """limit 100 measured at 5. Larger limits use the published Binance brackets."""
    n = int(limit)
    if n <= 100:
        return DEPTH_WEIGHT
    if n <= 500:
        return 25
    if n <= 1000:
        return 50
    return 250


def header_used_weight(headers: Any) -> int | None:
    if headers is None or not hasattr(headers, "get"):
        return None
    raw = headers.get("x-mbx-used-weight-1m")
    if raw is None:
        raw = headers.get("X-MBX-USED-WEIGHT-1M")
    if raw is None:
        return None
    try:
        return max(0, int(raw))
    except (TypeError, ValueError):
        return None


def window_id(epoch: float) -> int:
    """Binance.US request-weight minute, observed to open on the Unix minute."""
    return int(float(epoch) // 60)


def remaining_after_venue(local_remaining: int, venue_used: int, budget: int) -> int:
    """Raise usage to the venue header. Never restore capacity the header does not show."""
    cap = max(0, int(budget) - max(0, int(venue_used)))
    return min(max(0, int(local_remaining)), cap)


def carried_remaining(
    budget: int,
    venue_used: int | None,
    venue_observed_at: float | None,
    now: float,
) -> int:
    """Capacity opened when the local minute rolls.

    A sample younger than one minute may still belong to the venue minute that
    has not reset. A full budget is opened only when there is no such sample.
    """
    if venue_used is None or venue_observed_at is None:
        return int(budget)
    if float(now) - float(venue_observed_at) >= VENUE_SAMPLE_MAX_AGE_SEC:
        return int(budget)
    return max(0, int(budget) - max(0, int(venue_used)))


def apply_venue_header(
    *,
    remaining: int,
    budget: int,
    high_water: int | None,
    high_water_at: float | None,
    high_water_window: int | None,
    used: int,
    request_started_at: float | None,
    now_window: int,
) -> tuple[int, int, float | None, int | None, str]:
    """Apply one venue header.

    A lower reading reopens capacity only when the request started at or after
    the high-water observation. That request cannot be a stale earlier sample,
    so the counter itself reset. Returns remaining, high water, its time, its
    window, and a decision name.
    """
    used = max(0, int(used))
    current = max(0, int(remaining))
    if high_water is None or used >= int(high_water):
        cap = max(0, int(budget) - used)
        observed = request_started_at if request_started_at is not None else high_water_at
        window = high_water_window if high_water_window is not None else int(now_window)
        decision = "raised" if high_water is None or used > int(high_water) else "same"
        return min(current, cap), used, observed, window, decision
    if request_started_at is None or high_water_at is None or float(request_started_at) < float(high_water_at):
        return current, int(high_water), high_water_at, high_water_window, "stale_ignored"
    cap = max(0, int(budget) - used)
    opened = min(cap, current + (int(high_water) - used))
    return opened, used, float(request_started_at), int(now_window), "reset_confirmed"


def alignment_state(current_window: int | None, high_water_window: int | None, high_water: int | None) -> str:
    if high_water is None or current_window is None or high_water_window is None:
        return "UNKNOWN"
    if int(high_water_window) == int(current_window):
        return "ALIGNED"
    if int(high_water_window) < int(current_window):
        return "VENUE_AHEAD"
    return "LOCAL_AHEAD"


def _consume_keys(path: str) -> tuple[str, ...]:
    return (
        "bwl:tokens",
        "bwl:reset_ts",
        CHARGED_KEY,
        WINDOW_KEY,
        VENUE_HW_KEY,
        VENUE_HW_AT_KEY,
        "bwl:circuit_open",
        f"bwl:usage:{path}",
        f"bwl:req:{path}",
        VENUE_USED_KEY,
        VENUE_USED_AT_KEY,
        PROTECTIVE_KEY,
        EXIT_USED_KEY,
        DEFERRED_KEY,
        EXIT_REFUSAL_KEY,
    )


def weight_grant_allowed(tokens: int, weight: int, *, priority: str, reserve: int, pending: bool) -> bool:
    """Who may spend the shared minute. Exit may use the reserve. A pending sell blocks the rest."""
    if weight <= 0 or int(tokens) < int(weight):
        return False
    if priority in ("exit", "exit_decision"):
        return True
    if pending:
        return False
    return int(tokens) - int(weight) >= int(reserve)


def note_venue_used_sync(used: int, started_at: float | None = None) -> None:
    client = get_sync_redis_service()
    now = time.time()
    client.eval(
        _NOTE_LUA,
        9,
        "bwl:tokens",
        VENUE_USED_KEY,
        VENUE_USED_AT_KEY,
        WINDOW_KEY,
        VENUE_HW_KEY,
        VENUE_HW_AT_KEY,
        VENUE_HW_WINDOW_KEY,
        CHARGED_KEY,
        "bwl:reset_ts",
        int(used),
        int(WEIGHT_BUDGET_PER_MIN),
        str(now),
        str(now),
        "-1" if started_at is None else str(float(started_at)),
        str(VENUE_SAMPLE_MAX_AGE_SEC),
    )


def observe_headers_sync(headers: Any, started_at: float | None = None) -> None:
    used = header_used_weight(headers)
    if used is None:
        return
    try:
        note_venue_used_sync(used, started_at)
    except Exception:
        logger.debug("venue weight header not applied", exc_info=True)


def try_consume_sync(path: str, weight: int, priority: str = "background") -> bool:
    """Atomic shared-bucket charge for synchronous callers. False sends nothing."""
    try:
        client = get_sync_redis_service()
        now = time.time()
        granted = client.eval(
            _CONSUME_LUA,
            15,
            *_consume_keys(path),
            str(now),
            int(WEIGHT_BUDGET_PER_MIN),
            max(1, int(weight)),
            str(VENUE_SAMPLE_MAX_AGE_SEC),
            str(_PRIORITY_CODE.get(priority, 0)),
            str(PROTECTIVE_WEIGHT_RESERVE),
        )
        return int(granted or 0) == 1
    except Exception:
        logger.warning("REQUEST_WEIGHT_SYNC_UNAVAILABLE path=%s", path)
        return False


def request_weight_status() -> dict[str, Any]:
    """Shared Redis view. Not authoritative: the header is a past IP sample."""
    empty: dict[str, Any] = {
        "request_weight_charged": None,
        "request_weight_venue": None,
        "request_weight_effective": None,
        "request_weight_remaining": None,
        "request_weight_limit": WEIGHT_BUDGET_PER_MIN,
        "venue_weight_observed_at": None,
        "venue_weight_age_ms": None,
        "request_weight_authoritative": False,
        "rate_limited_until": None,
        "protective_request_pending": None,
        "request_weight_shared": True,
    }
    try:
        client = get_sync_redis_service()
        tokens_raw = client.get("bwl:tokens")
        venue_raw = client.get(VENUE_USED_KEY)
        observed_raw = client.get(VENUE_USED_AT_KEY)
        charged_raw = client.get(CHARGED_KEY)
        pending_raw = client.get(PROTECTIVE_KEY)
        exit_used_raw = client.get(EXIT_USED_KEY)
        deferred_raw = client.get(DEFERRED_KEY)
        refusal_raw = client.get(EXIT_REFUSAL_KEY)
        window_raw = client.get(WINDOW_KEY)
        hw_raw = client.get(VENUE_HW_KEY)
        hw_window_raw = client.get(VENUE_HW_WINDOW_KEY)
        reset_raw = client.get("bwl:reset_ts")
        ttl = int(client.ttl("bwl:circuit_open") or -1)
    except Exception:
        logger.debug("request weight status unavailable", exc_info=True)
        return empty
    remaining = int(tokens_raw) if tokens_raw is not None else None
    venue = int(venue_raw) if venue_raw is not None else None
    observed = float(observed_raw) if observed_raw is not None else None
    now = time.time()
    age = None if observed is None else max(0, int((now - observed) * 1000))
    effective = None if remaining is None else max(0, WEIGHT_BUDGET_PER_MIN - remaining)
    current = window_id(now)
    stored_window = int(window_raw) if window_raw is not None else None
    hw = int(hw_raw) if hw_raw is not None else None
    hw_window = int(hw_window_raw) if hw_window_raw is not None else None
    return {
        "request_weight_charged": int(charged_raw) if charged_raw is not None else 0,
        "request_weight_venue": venue,
        "request_weight_effective": effective,
        "request_weight_remaining": remaining,
        "request_weight_limit": WEIGHT_BUDGET_PER_MIN,
        "venue_weight_observed_at": observed,
        "venue_weight_age_ms": age,
        "request_weight_authoritative": False,
        "rate_limited_until": (now + ttl) if ttl > 0 else None,
        "protective_request_pending": int(pending_raw or 0) > 0,
        "protective_pending": int(pending_raw or 0) > 0,
        "protective_weight_reserved": PROTECTIVE_WEIGHT_RESERVE,
        "protective_weight_used": int(exit_used_raw or 0),
        "background_weight_deferred": int(deferred_raw or 0),
        "last_protective_weight_refusal": float(refusal_raw) if refusal_raw is not None else None,
        "critical_capacity_remaining": remaining,
        "noncritical_capacity_remaining": None if remaining is None else (0 if int(pending_raw or 0) > 0 else max(0, remaining - PROTECTIVE_WEIGHT_RESERVE)),
        "request_weight_shared": True,
        "request_weight_window_id": stored_window if stored_window is not None else current,
        "request_weight_window_started_at": (stored_window if stored_window is not None else current) * 60,
        "request_weight_window_resets_at": int(reset_raw) if reset_raw is not None else (current + 1) * 60,
        "venue_weight_window_id": hw_window,
        "local_effective_usage": effective,
        "venue_reported_usage": venue,
        "venue_high_water": hw,
        "window_alignment_state": alignment_state(stored_window if stored_window is not None else current, hw_window, hw),
    }


async def observe_headers(headers: Any, started_at: float | None = None) -> None:
    used = header_used_weight(headers)
    if used is None:
        return
    try:
        limiter = await BinanceWeightLimiter.create()
        await limiter.note_venue_used_weight(used, started_at)
    except Exception:
        logger.debug("venue weight header not applied", exc_info=True)


class BinanceWeightLimiter:
    def __init__(self, r: Any) -> None:
        self.r = r
        self.bucket_key = "bwl:tokens"
        self.reset_key = "bwl:reset_ts"
        self.breaker_key = "bwl:circuit_open"
        self.last_1003_key = "bwl:last_1003"
        self._init_lock = asyncio.Lock()

    @classmethod
    async def create(cls) -> BinanceWeightLimiter:
        # All Live Data, No Fallback/Hardcoded Data - Use shared Redis service (same as AI)
        r = get_redis_service()  # Returns the Redis client directly, not a wrapper
        inst = cls(r)
        await inst._ensure_window()
        return inst

    async def _roll_window(self) -> None:
        now = time.time()
        await self.r.eval(
            _ROLL_LUA,
            8,
            self.bucket_key,
            self.reset_key,
            CHARGED_KEY,
            WINDOW_KEY,
            VENUE_HW_KEY,
            VENUE_HW_AT_KEY,
            VENUE_USED_KEY,
            VENUE_USED_AT_KEY,
            str(now),
            int(WEIGHT_BUDGET_PER_MIN),
            str(VENUE_SAMPLE_MAX_AGE_SEC),
        )

    async def _ensure_window(self) -> None:
        async with self._init_lock:
            await self._roll_window()

    async def _tick_window(self) -> None:
        await self._roll_window()

    async def note_venue_used_weight(self, used: int, started_at: float | None = None) -> None:
        now = time.time()
        await self.r.eval(
            _NOTE_LUA,
            9,
            self.bucket_key,
            VENUE_USED_KEY,
            VENUE_USED_AT_KEY,
            WINDOW_KEY,
            VENUE_HW_KEY,
            VENUE_HW_AT_KEY,
            VENUE_HW_WINDOW_KEY,
            CHARGED_KEY,
            self.reset_key,
            int(used),
            int(WEIGHT_BUDGET_PER_MIN),
            str(now),
            str(now),
            "-1" if started_at is None else str(float(started_at)),
            str(VENUE_SAMPLE_MAX_AGE_SEC),
        )

    async def mark_protective_pending(self, delta: int) -> None:
        if int(delta) > 0:
            await self.r.incr(PROTECTIVE_KEY)
            await self.r.expire(PROTECTIVE_KEY, 120)
            return
        value = int(await self.r.decr(PROTECTIVE_KEY) or 0)
        if value < 0:
            await self.r.set(PROTECTIVE_KEY, 0)

    async def open_circuit(self, ttl: int | None = None) -> None:
        ttl = ttl or CIRCUIT_TTL_SEC
        await self.r.set(self.breaker_key, "1", ex=ttl)
        await self.r.set(self.last_1003_key, str(int(time.time())))
        with contextlib.suppress(ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError):
            limiter_circuit_open.set(1)

    async def is_circuit_open(self) -> bool:
        return (await self.r.get(self.breaker_key)) == "1"

    async def consume(
        self,
        path: str,
        weight: int | None = None,
        wait: bool = True,
        timeout: float = 5.0,
        priority: str = "background",
    ) -> None:
        global _weight_log_last_time

        if weight is None:
            try:
                ow = await self.r.get(f"bwl:weight_override:{path}")
                weight = max(1, int(ow)) if ow is not None else ENDPOINT_WEIGHTS.get(path, 1)
            except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError):
                weight = ENDPOINT_WEIGHTS.get(path, 1)

        if await self.is_circuit_open():
            with contextlib.suppress(ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError):
                limiter_denied_total.labels(path=path, reason="circuit").inc()
            msg = "Circuit is open due to recent rate-limit ban."
            raise CircuitOpenErrorError(msg)

        start = time.time()
        backoff = TOKEN_CHECK_INITIAL_BACKOFF

        while True:
            granted = await self.r.eval(
                _CONSUME_LUA,
                15,
                *_consume_keys(path),
                str(time.time()),
                int(WEIGHT_BUDGET_PER_MIN),
                int(weight or 1),
                str(VENUE_SAMPLE_MAX_AGE_SEC),
                str(_PRIORITY_CODE.get(priority, 0)),
                str(PROTECTIVE_WEIGHT_RESERVE),
            )
            if int(granted or 0) == 1:
                with contextlib.suppress(ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError):
                    limiter_consume_wait_seconds.labels(path=path).observe(max(0.0, time.time() - start))
                    limiter_consumes_total.labels(path=path).inc()
                now = time.time()
                _weight_counters[path]["calls"] += 1
                _weight_counters[path]["weight"] += weight or 1
                if now - _weight_log_last_time >= _weight_log_interval:
                    _log_weight_summary()
                    _weight_log_last_time = now
                return

            if not wait:
                with contextlib.suppress(ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError):
                    limiter_denied_total.labels(path=path, reason="empty").inc()
                with contextlib.suppress(ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError):
                    await self.r.incr(f"bwl:denied:{path}:empty")
                msg = f"Not enough tokens for {path}, need={weight}"
                raise RateLimitedErrorError(msg)

            if time.time() - start > timeout:
                with contextlib.suppress(ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError):
                    limiter_denied_total.labels(path=path, reason="timeout").inc()
                with contextlib.suppress(ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError):
                    await self.r.incr(f"bwl:denied:{path}:timeout")
                msg = f"Timeout waiting for tokens for {path}, need={weight}"
                raise RateLimitedErrorError(msg)

            # Exponential backoff instead of fixed sleep - reduces latency in hot paths
            await asyncio.sleep(min(backoff, TOKEN_CHECK_MAX_BACKOFF))
            backoff *= TOKEN_CHECK_BACKOFF_MULTIPLIER

    async def metrics(self) -> dict[str, object]:
        await self._tick_window()
        tokens = await self.r.get(self.bucket_key)
        reset_ts = await self.r.get(self.reset_key)
        last_1003 = await self.r.get(self.last_1003_key)
        circuit = await self.r.get(self.breaker_key)
        try:
            limiter_tokens.set(int(tokens) if tokens else 0)
            limiter_circuit_open.set(1 if circuit == "1" else 0)
        except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError):
            pass
        venue = await self.r.get(VENUE_USED_KEY)
        observed = await self.r.get(VENUE_USED_AT_KEY)
        charged = await self.r.get(CHARGED_KEY)
        pending = await self.r.get(PROTECTIVE_KEY)
        remaining = int(tokens) if tokens else 0
        return {
            "tokens": remaining,
            "window_reset_epoch": int(reset_ts) if reset_ts else None,
            "circuit_open": circuit == "1",
            "last_1003_epoch": int(last_1003) if last_1003 else None,
            "budget_per_min": WEIGHT_BUDGET_PER_MIN,
            "endpoint_weights": ENDPOINT_WEIGHTS,
            "request_weight_charged": int(charged) if charged else 0,
            "request_weight_venue": int(venue) if venue is not None else None,
            "request_weight_effective": max(0, WEIGHT_BUDGET_PER_MIN - remaining),
            "request_weight_remaining": remaining,
            "request_weight_limit": WEIGHT_BUDGET_PER_MIN,
            "venue_weight_observed_at": float(observed) if observed is not None else None,
            "request_weight_authoritative": False,
            "protective_request_pending": int(pending or 0) > 0,
        }


def _log_weight_summary() -> None:
    """Log weight consumption summary for the last 60s window."""
    global _weight_counters
    if not _weight_counters:
        return

    parts: list[str] = []
    total_weight = 0
    for path, counts in _weight_counters.items():
        calls = counts["calls"]
        weight = counts["weight"]
        total_weight += weight
        parts.append(f"{path} calls={calls} weight={weight}")

    if parts:
        summary = "; ".join(parts)
        logger.info(f"BINANCE_WEIGHT_60S: {summary}; total={total_weight}")
        warn_threshold = int(os.getenv("BINANCE_WEIGHT_WARN_TOTAL", str(int(WEIGHT_BUDGET_PER_MIN * 0.82))))
        if total_weight >= warn_threshold:
            logger.warning(
                "BINANCE_WEIGHT_HIGH total=%d threshold=%d budget=%d",
                total_weight,
                warn_threshold,
                WEIGHT_BUDGET_PER_MIN,
            )

    # Reset counters for next window
    _weight_counters.clear()
