"""
Global Redis Connection Registry and Cleanup
Tracks ALL Redis connections and ensures they close before event loop closes
"""

import asyncio
import logging
from typing import Any

try:
    import redis.asyncio as redis
except ImportError:
    redis = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)

# Global registry of all Redis connections (using weakref to not prevent garbage collection)
_redis_connections: set[Any] = set()


def _get_redis_lock() -> asyncio.Lock:
    """Get or create redis lock lazily."""
    if not hasattr(_get_redis_lock, "_lock"):
        _get_redis_lock._lock = asyncio.Lock()
    return _get_redis_lock._lock


def register_redis_connection(redis_client: Any) -> None:
    """Register a Redis connection for cleanup during shutdown."""
    if redis_client is not None:
        try:
            _redis_connections.add(redis_client)
            logger.debug(f"Registered Redis connection: {id(redis_client)}")
        except Exception as e:
            logger.debug(f"Failed to register Redis connection: {e}")


async def close_all_redis_connections() -> None:
    """Close ALL registered Redis connections gracefully."""
    async with _get_redis_lock():
        closed_count = 0
        failed_count = 0

        # Make a copy to avoid modification during iteration
        connections = list(_redis_connections)

        for redis_client in connections:
            try:
                if redis_client is not None:
                    # Try multiple close methods
                    if hasattr(redis_client, "close"):
                        # Always await close() - redis.asyncio.Redis.close() returns a coroutine
                        result = redis_client.close()
                        if asyncio.iscoroutine(result):
                            await result
                        closed_count += 1
                    elif hasattr(redis_client, "aclose"):
                        await redis_client.aclose()
                        closed_count += 1
                    elif hasattr(redis_client, "disconnect"):
                        result = redis_client.disconnect()
                        if asyncio.iscoroutine(result):
                            await result
                        closed_count += 1
            except Exception as e:
                logger.debug(f"Error closing Redis connection {id(redis_client)}: {e}")
                failed_count += 1

        # Clear the registry
        _redis_connections.clear()

        if closed_count > 0:
            logger.info(f"[OK] Closed {closed_count} Redis connections")
        if failed_count > 0:
            logger.debug(f"Failed to close {failed_count} Redis connections")


# Monkey-patch redis.from_url to auto-register connections
def patch_redis_from_url():
    """Patch redis.from_url to automatically register connections."""
    if redis is None:
        logger.debug("Redis not available, skipping patch")
        return False

    try:
        original_from_url = redis.from_url

        def patched_from_url(*args, **kwargs):
            client = original_from_url(*args, **kwargs)
            register_redis_connection(client)
            return client

        redis.from_url = patched_from_url
        logger.info("[OK] Patched redis.from_url to auto-register connections")
    except Exception as e:
        logger.debug(f"Failed to patch redis.from_url: {e}")
        return False
    else:
        return True


# Monkey-patch redis.Redis to auto-register connections
def patch_redis_redis():
    """Patch redis.Redis to automatically register connections."""
    if redis is None:
        logger.debug("Redis not available, skipping patch")
        return False

    try:
        original_init = redis.Redis.__init__

        def patched_init(self, *args, **kwargs):
            original_init(self, *args, **kwargs)
            register_redis_connection(self)

        redis.Redis.__init__ = patched_init
        logger.info("[OK] Patched redis.Redis.__init__ to auto-register connections")
    except Exception as e:
        logger.debug(f"Failed to patch redis.Redis: {e}")
        return False
    else:
        return True


def initialize_redis_cleanup():
    """Initialize Redis cleanup system with monkey patches."""
    patch_redis_from_url()
    patch_redis_redis()
    logger.info("[OK] Redis cleanup system initialized")
