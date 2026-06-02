"""
Optimized database connection pool settings for Mystic Trading Platform.
Connected to live configuration and integrated with database services.
"""

import asyncio
import contextvars
import os

from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine

# Import live configuration
try:
    from backend.config_bridge import get_mystic_config

    _mystic_config = get_mystic_config()
except (ImportError, AttributeError, ValueError, TypeError, RuntimeError):
    _mystic_config = None


def _get_pool_size() -> int:
    """Get pool size from live configuration."""
    try:
        value = int(os.getenv("DB_POOL_SIZE", "5"))  # Personal laptop: 5 is plenty
        return max(1, value)
    except (ValueError, TypeError):
        return 20


def _get_max_overflow() -> int:
    """Get max overflow from live configuration."""
    try:
        value = int(os.getenv("DB_MAX_OVERFLOW", "2"))  # Personal laptop: 2 overflow max
        return max(0, value)
    except (ValueError, TypeError):
        return 10


def _get_pool_recycle() -> int:
    """Get pool recycle time from live configuration."""
    try:
        value = int(os.getenv("DB_POOL_RECYCLE", "3600"))
        return max(1, value)
    except (ValueError, TypeError):
        return 3600


def _get_pool_timeout() -> float:
    """Get pool timeout from live configuration."""
    try:
        value = float(os.getenv("DB_POOL_TIMEOUT", "15"))
        return max(1.0, value)
    except (ValueError, TypeError):
        return 15.0


def _get_db_semaphore_limit() -> int:
    """Get database semaphore limit from live configuration."""
    try:
        value = int(os.getenv("DB_SEMAPHORE_LIMIT", "3"))  # Personal laptop: 3 concurrent ops max
        return max(1, value)
    except (ValueError, TypeError):
        return 10


def _get_max_active_connections() -> int:
    """Get max active connections from live configuration."""
    try:
        value = int(os.getenv("DB_MAX_ACTIVE_CONNECTIONS", "5"))
        return max(1, value)
    except (ValueError, TypeError):
        return 5


def _get_db_pool_settings() -> dict[str, int | float | bool]:
    """Get database pool settings from live configuration."""
    return {
        "pool_size": _get_pool_size(),
        "max_overflow": _get_max_overflow(),
        "pool_recycle": _get_pool_recycle(),
        "pool_timeout": _get_pool_timeout(),
        "pool_pre_ping": True,
    }


# Database pool settings (dynamically loaded from config)
DB_POOL_SETTINGS = _get_db_pool_settings()

# Track active connections per coroutine
_active_connections = contextvars.ContextVar("db_connections", default=0)

# Semaphore to limit concurrent database operations (initialized from live config)
_db_semaphore: asyncio.Semaphore | None = None


def _get_db_semaphore() -> asyncio.Semaphore:
    """Get or create database semaphore from live configuration."""
    global _db_semaphore
    if _db_semaphore is None:
        _db_semaphore = asyncio.Semaphore(_get_db_semaphore_limit())
    return _db_semaphore


# Shared engines per connection string
_engines: dict[str, AsyncEngine] = {}


def get_async_engine(connection_string: str) -> AsyncEngine:
    """Get or create a shared SQLAlchemy async engine"""
    if connection_string not in _engines:
        # Reload settings to ensure live config is used
        settings = _get_db_pool_settings()
        _engines[connection_string] = create_async_engine(
            connection_string,
            echo=False,
            pool_size=settings["pool_size"],
            max_overflow=settings["max_overflow"],
            pool_recycle=settings["pool_recycle"],
            pool_timeout=settings["pool_timeout"],
            pool_pre_ping=settings["pool_pre_ping"],
        )
    return _engines[connection_string]


async def close_engines() -> None:
    """Close all engines gracefully"""
    for engine in _engines.values():
        await engine.dispose()
    _engines.clear()


class DatabaseConnectionGuard:
    """Context manager for controlled database access"""

    def __init__(self, max_active: int | None = None):
        self.max_active = max_active if max_active is not None else _get_max_active_connections()

    async def __aenter__(self):
        active = _active_connections.get()
        if active >= self.max_active:
            msg = f"Too many active DB connections: {active}"
            raise RuntimeError(msg)

        semaphore = _get_db_semaphore()
        await semaphore.acquire()
        _active_connections.set(active + 1)
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        semaphore = _get_db_semaphore()
        semaphore.release()
        _active_connections.set(_active_connections.get() - 1)
