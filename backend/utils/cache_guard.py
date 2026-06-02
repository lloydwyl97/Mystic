from __future__ import annotations

import contextlib
import json
import logging
import os
import time
from typing import Any

from backend.config.redis_config import get_shared_redis_async

logger = logging.getLogger(__name__)

# Rate limiting for mark_market_update error logging (prevents log flood during Redis outages)
_last_mark_error_log_time: float = 0.0
_MARK_ERROR_LOG_INTERVAL = 60.0  # Log at most once per 60 seconds


# All Live Data, No Fallback/Hardcoded Data
# Defer REDIS_URL check until actually used to allow app to start
def _get_redis_url() -> str:
    """Get REDIS_URL, raising error only when actually needed."""
    redis_url = os.getenv("REDIS_URL")
    if not redis_url:
        msg = "REDIS_URL environment variable is required - no fallback/hardcoded Redis URL"
        raise RuntimeError(msg)
    return redis_url


PRICE_KEY = "price:{symbol}"
FRESHNESS_SEC = int(os.getenv("CACHE_FRESH_SEC", "30"))


class CacheGuard:
    def __init__(self, r: Any) -> None:
        self.r = r

    @classmethod
    async def create(cls) -> CacheGuard:
        # Use shared async Redis client (returns async client, but function itself is sync)
        r = get_shared_redis_async()
        return cls(r)

    async def get_price_cached(self, symbol: str, freshness_sec: int | None = None) -> float | None:
        key = PRICE_KEY.format(symbol=symbol.strip().upper())
        try:
            raw = await self.r.hget(key, "v")
            ts = await self.r.hget(key, "ts")
            if raw is None or ts is None:
                return None
            try:
                limit = FRESHNESS_SEC if freshness_sec is None else int(max(0, freshness_sec))
                if int(time.time()) - int(ts) > limit:
                    return None
                return float(raw)
            except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError):
                return None
        except Exception as e:
            # Handle WRONGTYPE errors by deleting the key and returning None
            # Catch all exceptions to handle redis.asyncio specific error types
            error_str = str(e)
            if "WRONGTYPE" in error_str or "wrong kind of value" in error_str.lower():
                with contextlib.suppress(Exception):
                    await self.r.delete(key)
            return None

    async def set_price(self, symbol: str, value: float) -> None:
        key = PRICE_KEY.format(symbol=symbol.strip().upper())
        try:
            # Delete the key first to ensure it's the right type
            await self.r.delete(key)
            await self.r.hset(key, "v", str(value))
            await self.r.hset(key, "ts", str(int(time.time())))
        except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
            # If there's still a WRONGTYPE error, try to force the correct type
            if "WRONGTYPE" in str(e):
                try:
                    await self.r.delete(key)
                    await self.r.hset(key, "v", str(value))
                    await self.r.hset(key, "ts", str(int(time.time())))
                except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError):
                    pass

    async def mark_market_update(self, source: str) -> None:
        """Mark that market data was updated from a specific source.

        Updates both market_data:last_update timestamp and market_data:last_update_source.
        Used by both WebSocket and REST data sources to keep stale data detection current.
        """
        global _last_mark_error_log_time
        try:
            now = int(time.time())
            # Increased expiry to 600s (10 minutes) to prevent key from disappearing
            # Heartbeat updates every 2s, so key will be refreshed well before expiry
            await self.r.set("market_data:last_update", str(now), ex=600)
            await self.r.set("market_data:last_update_source", source, ex=600)
        except Exception as e:
            # Rate-limited error logging (prevents log flood during Redis outages)
            current_time = time.time()
            if current_time - _last_mark_error_log_time >= _MARK_ERROR_LOG_INTERVAL:
                logger.exception(f"Failed to mark market data update from {source}: {e}")
                _last_mark_error_log_time = current_time

    async def get_klines_cached(self, symbol: str, interval: str, n: int = 100) -> list[Any] | None:
        k = f"klines:{symbol.strip().upper()}:{interval}"
        try:
            raw = await self.r.get(k)
            if not raw:
                return None
            try:
                data = json.loads(raw)
                return data[-n:] if len(data) > n else data
            except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError):
                return None
        except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
            # Handle WRONGTYPE errors by deleting the key and returning None
            if "WRONGTYPE" in str(e):
                with contextlib.suppress(ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError):
                    await self.r.delete(k)
            return None

    async def set_klines(self, symbol: str, interval: str, klines: list[Any]) -> None:
        k = f"klines:{symbol.strip().upper()}:{interval}"
        try:
            # Delete the key first to ensure it's the right type
            await self.r.delete(k)
            await self.r.set(k, json.dumps(klines), ex=300)
        except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
            # If there's still a WRONGTYPE error, try to force the correct type
            if "WRONGTYPE" in str(e):
                try:
                    await self.r.delete(k)
                    await self.r.set(k, json.dumps(klines), ex=300)
                except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError):
                    pass
