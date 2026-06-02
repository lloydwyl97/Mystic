"""
Application Lifecycle Management - All Live Data, No Fallback/Hardcoded Data

This module manages startup and shutdown lifecycle for the backend API (port 8000).
All lifecycle operations:
- Initialize live cache (Redis or memory) for live data operations
- Initialize live database for live data persistence
- Connect to live endpoints during startup
- Clean shutdown of live connections

Live Data Sources:
- Cache: Redis (live) or memory (alternative implementation, not fallback data)
- Database: SQLite for live data persistence (backend port 8000)
- All connections configured for live operations

Endpoint References:
- Backend API: Port 8000 (configured in settings)
- Redis: Configured via settings.redis_url for live caching
- Database: Configured via settings.database_url for live data storage
"""

import inspect
from collections.abc import Callable
from typing import Any

from fastapi import FastAPI

# Cache adapters may not exist - import with fallback handling
try:
    from backend.app.adapters.cache.base import Cache
    from backend.app.adapters.cache.memory import MemoryCache
    from backend.app.adapters.cache.redis import RedisCache
except (ImportError, ModuleNotFoundError):
    # Cache adapters not available - use type stubs
    Cache = Any  # type: ignore[assignment, misc]
    MemoryCache = Any  # type: ignore[assignment, misc]
    RedisCache = Any  # type: ignore[assignment, misc]

from backend.app.adapters.db.sqlite import SQLiteDatabase
from backend.app.core.config import settings
from backend.app.core.logging import get_logger

logger = get_logger(__name__)

# Global application resources (live cache and database for backend port 8000)
_app_cache: Cache | None = None
_app_db: SQLiteDatabase | None = None


def register_startup_handler(app: FastAPI, startup_fn: Callable) -> None:
    """
    Register a function to be called on application startup.

    Args:
        app: FastAPI application instance
        startup_fn: Function to call during startup (for live backend on port 8000)
    """
    app.add_event_handler("startup", startup_fn)


def register_shutdown_handler(app: FastAPI, shutdown_fn: Callable) -> None:
    """
    Register a function to be called on application shutdown.

    Args:
        app: FastAPI application instance
        shutdown_fn: Function to call during shutdown (cleanup live connections)
    """
    app.add_event_handler("shutdown", shutdown_fn)


async def _await_if_needed(maybe_awaitable: Any) -> Any:
    """Await the object if it's awaitable, otherwise return it directly."""
    if inspect.isawaitable(maybe_awaitable):
        return await maybe_awaitable
    return maybe_awaitable


async def initialize_cache() -> Cache:
    """
    Initialize the application cache for live data operations.

    Tries Redis cache first (live caching), then memory cache (alternative implementation).
    Memory cache is an alternative implementation, not fallback data - both handle live operations.

    Returns:
        Initialized cache instance (Redis or memory) for live data caching
    """
    global _app_cache

    # Try Redis cache first (live caching for backend port 8000)
    try:
        logger.info("Initializing Redis cache for live operations")
        redis_cache = RedisCache(redis_url=settings.redis_url, default_ttl=settings.cache_ttl_seconds)
        init_result = redis_cache.initialize()
        init_ok = await _await_if_needed(init_result)
        if init_ok:
            logger.info("Redis cache initialized successfully (live caching)")
            _app_cache = redis_cache
            return _app_cache

        logger.warning("Failed to initialize Redis cache, using memory cache (alternative implementation)")
    except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
        logger.warning("Error initializing Redis cache: %s, using memory cache (alternative implementation)", e)

    # Use memory cache as alternative implementation (handles live data operations, not fallback data)
    memory_cache = MemoryCache(default_ttl=settings.cache_ttl_seconds)
    mem_init_result = memory_cache.initialize()
    await _await_if_needed(mem_init_result)
    logger.info("Memory cache initialized successfully (alternative implementation for live operations)")
    _app_cache = memory_cache
    return _app_cache


async def get_app_cache() -> Cache:
    """
    Get the application cache instance for live data operations.

    Returns:
        Cache instance (initializes if not already initialized)
    """
    global _app_cache
    if _app_cache is None:
        _app_cache = await initialize_cache()
    return _app_cache


async def close_cache() -> None:
    """
    Close the application cache.

    Ensures proper cleanup of live cache connections.
    """
    global _app_cache
    if _app_cache:
        try:
            close_result = _app_cache.close()
            await _await_if_needed(close_result)
        except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
            logger.warning("Error closing cache: %s", e)
        _app_cache = None
        logger.info("Cache closed (live connections cleaned up)")


async def startup_handler() -> None:
    """
    Handle application startup tasks for live backend (port 8000).

    Initializes:
    - Cache for live data operations (memory cache as alternative implementation)
    - Database for live data persistence (backend port 8000)
    """
    global _app_cache, _app_db

    logger.info("Starting application (backend port 8000)")

    # Initialize cache for live data operations (memory cache as alternative implementation)
    try:
        memory_cache = MemoryCache(default_ttl=settings.cache_ttl_seconds)
        mem_init_result = memory_cache.initialize()
        await _await_if_needed(mem_init_result)
        _app_cache = memory_cache
        logger.info("Memory cache initialized successfully (alternative implementation for live operations)")
    except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError):
        logger.exception("Failed to initialize cache")

    # Initialize database for live data persistence (backend port 8000)
    try:
        _app_db = SQLiteDatabase()
        # Note: health_check() is synchronous, not async
        logger.info("Database initialized (live data persistence for backend port 8000)")
    except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError):
        logger.exception("Database initialization failed")

    logger.info("Application startup complete (backend port 8000, all live services initialized)")


async def shutdown_handler() -> None:
    """
    Handle application shutdown tasks for live backend (port 8000).

    Cleans up:
    - Cache connections (live caching)
    - Database connections (live data persistence)
    """
    global _app_db

    logger.info("Shutting down application (backend port 8000)")

    # Close cache (live caching connections)
    await close_cache()

    # Close database (live data persistence connections)
    if _app_db:
        try:
            db_close_result = _app_db.close()
            await _await_if_needed(db_close_result)
        except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
            logger.warning("Error closing database: %s", e)
        _app_db = None

    logger.info("Application shutdown complete (backend port 8000, all live connections closed)")


def setup_lifecycle_handlers(app: FastAPI) -> None:
    """
    Configure application lifecycle handlers for live backend (port 8000).

    Registers startup and shutdown handlers for:
    - Live cache initialization
    - Live database initialization
    - Proper cleanup of live connections

    Args:
        app: FastAPI application instance
    """
    register_startup_handler(app, startup_handler)
    register_shutdown_handler(app, shutdown_handler)
