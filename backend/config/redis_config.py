#!/usr/bin/env python3
"""
Redis Configuration - Centralized Connection Pool Management

This module provides a SINGLE SHARED Redis connection pool for the entire application.
All Redis operations MUST use this shared pool to prevent connection exhaustion.

CRITICAL: Do NOT create new Redis connections directly. Use:
- get_shared_redis_sync() for synchronous operations
- get_shared_redis_async() for asynchronous operations

Connection Pool Architecture:
- Single shared sync pool: 50 connections max
- Single shared async pool: 50 connections max
- All services share these pools to prevent connection exhaustion
"""

import asyncio
import logging
import os
import sys
from typing import Any

import redis

# CRITICAL FIX: Windows ProactorEventLoop has bugs with async Redis connections
# Switch to SelectorEventLoop for reliable Redis async connections on Windows
if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

logger = logging.getLogger(__name__)


def _parse_timeout_value(env_var: str, default: float) -> float:
    """Parse timeout environment variables while keeping defaults safe."""
    raw_value = os.getenv(env_var)
    if raw_value is None:
        return default

    try:
        parsed = float(raw_value)
    except ValueError:
        logger.warning(
            "Invalid value for %s (%s); falling back to default %s",
            env_var,
            raw_value,
            default,
        )
        return default

    if parsed <= 0:
        logger.warning(
            "Invalid value for %s (%s); falling back to default %s",
            env_var,
            raw_value,
            default,
        )
        return default

    return parsed


# CRITICAL FIX: Windows + WSL2 + asyncio requires much longer timeouts for initial connection
# Sync connections work instantly, but async connections need 60+ seconds on first connect
# After first connection, subsequent operations are fast
DEFAULT_REDIS_SOCKET_TIMEOUT = 90.0  # Increased from 30 for Windows + WSL2
DEFAULT_REDIS_SOCKET_CONNECT_TIMEOUT = 90.0  # Increased from 30 for Windows + WSL2
REDIS_SOCKET_TIMEOUT_SECONDS = _parse_timeout_value("REDIS_SOCKET_TIMEOUT", DEFAULT_REDIS_SOCKET_TIMEOUT)
REDIS_SOCKET_CONNECT_TIMEOUT_SECONDS = _parse_timeout_value("REDIS_SOCKET_CONNECT_TIMEOUT", DEFAULT_REDIS_SOCKET_CONNECT_TIMEOUT)

# Standardized Redis pool configuration for all Redis connections
# CRITICAL Fix: Added protocol=2 to disable CLIENT SETINFO which causes
# "Error UNKNOWN while writing to socket" on Windows Redis installations
# FIXED: Restored to 50 connections with proper connection management
# Connection leaks were due to null client handling, not pool size
REDIS_POOL_CONFIG = {
    "max_connections": 50,  # Restored from 10 to handle multiple agents
    "socket_timeout": REDIS_SOCKET_TIMEOUT_SECONDS,
    "socket_connect_timeout": REDIS_SOCKET_CONNECT_TIMEOUT_SECONDS,
    "socket_keepalive": True,
    "socket_keepalive_options": {},
    "health_check_interval": 0,  # Disabled to prevent connection issues
    "retry_on_timeout": True,
    "decode_responses": True,
    "protocol": 2,  # CRITICAL: Use RESP2 to avoid CLIENT SETINFO issues on Windows Redis
}

# CRITICAL FIX: Use asyncio.Lock() instead of threading.Lock() for async context
_async_lock = asyncio.Lock()
_sync_lock_created = False


class SharedRedisState:
    """
    Singleton state for shared Redis connections.

    IMPORTANT: All Redis connections MUST go through this class to prevent
    connection pool exhaustion. Do NOT create Redis connections directly.
    """

    _sync_pool: redis.ConnectionPool | None = None
    _sync_client: redis.Redis | None = None
    _async_pool: Any = None  # redis.asyncio.ConnectionPool
    _async_client: Any = None  # redis.asyncio.Redis

    @classmethod
    def get_redis_url(cls) -> str:
        """Get Redis URL from environment variables."""
        redis_url = (os.getenv("REDIS_URL") or "").strip()
        if redis_url:
            return redis_url
        # Local dev / fresh VM: match REDIS_HOST/REDIS_PORT/DB used by sync pool and .env.example
        host = os.getenv("REDIS_HOST", "127.0.0.1")
        port = int(os.getenv("REDIS_PORT", "6379"))
        db = int(os.getenv("REDIS_DB", "0"))
        fallback = f"redis://{host}:{port}/{db}"
        logger.info(
            "REDIS_URL not set; using default %s (local). Set REDIS_URL in .env for non-local Redis.",
            fallback,
        )
        return fallback

    @classmethod
    def get_sync_pool(cls) -> redis.ConnectionPool:
        """Get or create the shared sync connection pool."""
        if cls._sync_pool is None:
            # CRITICAL FIX: Graceful degradation instead of raising exception
            redis_host = os.getenv("REDIS_HOST", "127.0.0.1")
            redis_port = int(os.getenv("REDIS_PORT", "6379"))
            redis_db = int(os.getenv("REDIS_DB", "0"))

            try:
                cls._sync_pool = redis.ConnectionPool(
                    host=redis_host,
                    port=redis_port,
                    db=redis_db,
                    **REDIS_POOL_CONFIG,
                )
                logger.info(
                    "Shared sync Redis pool created: %s:%s/%s (max=%s connections)",
                    redis_host,
                    redis_port,
                    redis_db,
                    REDIS_POOL_CONFIG["max_connections"],
                )
            except Exception as e:
                logger.exception(f"CRITICAL: Failed to create Redis pool: {e}")
                logger.warning("This will cause agents to fail. Check Redis server is running and accessible.")
                return None
        return cls._sync_pool

    @classmethod
    def get_sync_client(cls) -> redis.Redis:
        """Get the shared sync Redis client."""
        if cls._sync_client is None:
            pool = cls.get_sync_pool()
            if pool is None:
                logger.error("Cannot create Redis client: pool is None")
                return None
            try:
                cls._sync_client = redis.Redis(connection_pool=pool)
                logger.info("Shared sync Redis client created")
            except Exception as e:
                logger.exception(f"Failed to create Redis client: {e}")
                return None
        return cls._sync_client

    @classmethod
    def get_async_client(cls) -> Any:
        """Get the shared async Redis client. Recreates if closed."""
        # Check if existing client is closed and reset if needed
        if cls._async_client is not None:
            try:
                # Check if connection is still valid by accessing internal state
                conn_pool = getattr(cls._async_client, "connection_pool", None)
                if conn_pool and hasattr(conn_pool, "_available_connections"):
                    # If pool exists but no available connections, may need reset
                    pass
            except Exception:
                # If any error checking, reset the client
                cls._async_client = None

        if cls._async_client is None:
            try:
                import redis.asyncio as aioredis
            except ImportError as e:
                logger.exception(f"redis.asyncio not available: {e}")
                return None

            redis_url = cls.get_redis_url()

            try:
                # Create shared async client with connection pool
                cls._async_client = aioredis.from_url(
                    redis_url,
                    encoding="utf-8",
                    decode_responses=True,
                    socket_timeout=REDIS_SOCKET_TIMEOUT_SECONDS,
                    socket_connect_timeout=REDIS_SOCKET_CONNECT_TIMEOUT_SECONDS,
                    retry_on_timeout=True,
                    protocol=2,
                    health_check_interval=0,
                    max_connections=50,  # Restored from 10 to handle multiple agents
                )
                logger.info("Shared async Redis client created: %s", redis_url)
            except Exception as e:
                logger.exception(f"Failed to create async Redis client: {e}")
                return None
        return cls._async_client

    @classmethod
    async def close_all(cls) -> None:
        """Close all shared Redis connections."""
        if cls._async_client is not None:
            try:
                if hasattr(cls._async_client, "aclose"):
                    await cls._async_client.aclose()
                elif hasattr(cls._async_client, "close"):
                    await cls._async_client.close()
            except Exception as e:
                logger.debug(f"Error closing async client: {e}")
            cls._async_client = None
            cls._async_pool = None

        if cls._sync_client is not None:
            try:
                cls._sync_client.close()
            except Exception as e:
                logger.debug(f"Error closing sync client: {e}")
            cls._sync_client = None

        if cls._sync_pool is not None:
            try:
                cls._sync_pool.disconnect()
            except Exception as e:
                logger.debug(f"Error disconnecting sync pool: {e}")
            cls._sync_pool = None

        logger.info("All shared Redis connections closed")


# ============================================================================
# PUBLIC API - Use these functions for Redis access
# ============================================================================


def get_shared_redis_sync() -> redis.Redis:
    """
    Get the shared synchronous Redis client.

    IMPORTANT: Use this instead of creating new Redis connections!
    This returns a client that uses the shared connection pool.

    Returns:
        Shared Redis client connected to the application pool
    """
    return SharedRedisState.get_sync_client()


def get_shared_redis_async() -> Any:
    """
    Get the shared asynchronous Redis client.

    IMPORTANT: Use this instead of creating new Redis connections!
    This returns a client that uses the shared connection pool.

    Returns:
        Shared async Redis client connected to the application pool
    """
    return SharedRedisState.get_async_client()


# ============================================================================
# LEGACY API - For backward compatibility (prefer shared clients above)
# ============================================================================


class RedisConfigState:
    """Legacy state class - delegates to SharedRedisState."""

    @classmethod
    def get_pool(cls) -> redis.ConnectionPool | None:
        """Get Redis pool - delegates to SharedRedisState."""
        return SharedRedisState._sync_pool

    @classmethod
    def set_pool(cls, pool: redis.ConnectionPool) -> None:
        """Set Redis pool - delegates to SharedRedisState."""
        SharedRedisState._sync_pool = pool

    @classmethod
    def get_client(cls) -> redis.Redis | None:
        """Get Redis client - delegates to SharedRedisState."""
        return SharedRedisState._sync_client

    @classmethod
    def set_client(cls, client: redis.Redis) -> None:
        """Set Redis client - delegates to SharedRedisState."""
        SharedRedisState._sync_client = client

    @classmethod
    def clear_all(cls) -> None:
        """Clear all Redis connections."""
        SharedRedisState._sync_pool = None
        SharedRedisState._sync_client = None


def get_redis_pool() -> redis.ConnectionPool:
    """Get the shared Redis connection pool."""
    return SharedRedisState.get_sync_pool()


def get_redis_client() -> redis.Redis:
    """Get the shared Redis client."""
    return SharedRedisState.get_sync_client()


async def close_redis_connections() -> None:
    """Close all Redis connections."""
    await SharedRedisState.close_all()


def get_redis_socket_timeout_seconds() -> float:
    """Return the socket timeout configured for Redis connections."""
    return REDIS_SOCKET_TIMEOUT_SECONDS


def get_redis_socket_connect_timeout_seconds() -> float:
    """Return the socket connect timeout configured for Redis connections."""
    return REDIS_SOCKET_CONNECT_TIMEOUT_SECONDS


def get_redis_url() -> str:
    """Get Redis URL from environment variables."""
    return SharedRedisState.get_redis_url()


def create_redis_client_sync(_decode_responses: bool = True) -> redis.Redis:
    """
    Get the shared synchronous Redis client.

    NOTE: This now returns the SHARED client instead of creating a new one.
    This prevents connection pool exhaustion.

    Args:
        _decode_responses: Unused parameter (kept for backward compatibility)
    """
    # Return shared client instead of creating new connections
    return get_shared_redis_sync()


async def create_redis_client_async(_decode_responses: bool = True) -> Any:
    """
    Get the shared asynchronous Redis client.

    NOTE: This now returns the SHARED client instead of creating a new one.
    This prevents connection pool exhaustion.

    Args:
        _decode_responses: Unused parameter (kept for backward compatibility)
    """
    # Return shared client instead of creating new connections
    return get_shared_redis_async()
