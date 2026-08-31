from __future__ import annotations

import asyncio
import contextlib
import logging
import os
import time
from collections import defaultdict
from typing import Any

import redis

from backend.config.mystic_api_schedule import BINANCEUS_WEIGHT_PER_MIN
from backend.metrics import (
    limiter_circuit_open,
    limiter_consume_wait_seconds,
    limiter_consumes_total,
    limiter_denied_total,
    limiter_tokens,
)
from backend.services.redis_service import get_redis_service

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

ENDPOINT_WEIGHTS: dict[str, int] = {
    "/api/v3/ticker/price": 1,
    "/api/v3/ticker/24hr": 1,
    "/api/v3/klines": 1,
    # weight=5 for limit<=100 (Binance.US); accounted for separately in
    # backend/services/binance_scalp/market_reader.py since that path is a
    # synchronous cross-process caller and doesn't go through this async limiter.
    "/api/v3/depth": 5,
    "/api/v3/trades": 1,
    "/api/v3/order": 1,
    "/api/v3/order/test": 1,
    "/api/v3/openOrders": 5,
    "/api/v3/allOrders": 10,
    "/api/v3/myTrades": 10,
    "/api/v3/account": 10,
    "/api/v3/exchangeInfo": 10,
    "/api/v3/time": 1,
    "/api/v3/ping": 1,
    "/api/v3/avgPrice": 1,
    "/sapi/v1/capital/withdraw/apply": 10,
    "/sapi/v1/capital/config/getall": 10,
}


class RateLimitedErrorError(Exception):
    pass


class CircuitOpenErrorError(Exception):
    pass


# Aliases for backward compatibility and expected imports
RateLimited = RateLimitedErrorError
CircuitOpen = CircuitOpenErrorError
RateLimitedError = RateLimitedErrorError
CircuitOpenError = CircuitOpenErrorError


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

    async def _ensure_window(self) -> None:
        async with self._init_lock:
            now = int(time.time())
            reset_ts = await self.r.get(self.reset_key)
            if reset_ts is None or int(reset_ts) <= now:
                pipe = self.r.pipeline(transaction=True)
                pipe.set(self.bucket_key, WEIGHT_BUDGET_PER_MIN)
                pipe.set(self.reset_key, now + 60)
                await pipe.execute()

    async def _tick_window(self) -> None:
        now = int(time.time())
        reset_ts = await self.r.get(self.reset_key)
        if reset_ts is None or int(reset_ts) <= now:
            pipe = self.r.pipeline(transaction=True)
            pipe.set(self.bucket_key, WEIGHT_BUDGET_PER_MIN)
            pipe.set(self.reset_key, now + 60)
            await pipe.execute()

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
            await self._tick_window()
            async with self.r.pipeline(transaction=True) as pipe:
                try:
                    await pipe.watch(self.bucket_key)
                    tokens_raw = await pipe.get(self.bucket_key)
                    tokens = int(tokens_raw) if tokens_raw is not None else 0
                    with contextlib.suppress(ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError):
                        limiter_tokens.set(tokens)
                    if tokens >= (weight or 1):
                        pipe.multi()
                        pipe.decrby(self.bucket_key, weight or 1)
                        await pipe.execute()
                        try:
                            await self.r.incrby(f"bwl:usage:{path}", weight or 1)
                            await self.r.incr(f"bwl:req:{path}")
                        except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError):
                            pass
                        try:
                            limiter_consume_wait_seconds.labels(path=path).observe(max(0.0, time.time() - start))
                            limiter_consumes_total.labels(path=path).inc()
                        except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError):
                            pass

                        # Update weight logging counters
                        now = time.time()
                        _weight_counters[path]["calls"] += 1
                        _weight_counters[path]["weight"] += weight or 1

                        # Log summary every 60 seconds
                        if now - _weight_log_last_time >= _weight_log_interval:
                            _log_weight_summary()
                            _weight_log_last_time = now

                        return
                    await pipe.unwatch()
                except redis.WatchError:
                    pass

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
        return {
            "tokens": int(tokens) if tokens else 0,
            "window_reset_epoch": int(reset_ts) if reset_ts else None,
            "circuit_open": circuit == "1",
            "last_1003_epoch": int(last_1003) if last_1003 else None,
            "budget_per_min": WEIGHT_BUDGET_PER_MIN,
            "endpoint_weights": ENDPOINT_WEIGHTS,
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
