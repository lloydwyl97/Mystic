"""
Service Manager

Handles initialization and management of all application services.

Quick Test Checklist:
- ASCII-only logging; UTC-safe timestamps handled by dependencies.
- No exchange strings; this module only wires services.
- Idempotent initialize(); safe shutdown even if partially initialized.
- Tolerates missing optional services (market data, unified signal manager).
- Python 3.12 compatible.
"""

from __future__ import annotations

import asyncio
import inspect
import logging
import sqlite3
from typing import Any

from backend.database_schema import DATABASE_PATH

# Optional imports - try at top level
try:
    from connection_manager import get_connection_manager as _gcm_project  # type: ignore[import-not-found]
except (ImportError, ModuleNotFoundError, AttributeError, ValueError, TypeError, RuntimeError):
    _gcm_project = None

try:
    from backend.connection_manager import (
        get_connection_manager as _gcm_backend,
    )  # type: ignore[import-not-found]
except (ImportError, ModuleNotFoundError, AttributeError, ValueError, TypeError, RuntimeError):
    _gcm_backend = None

try:
    from backend.services.market_data import MarketDataService as _MDS_backend  # type: ignore[import-not-found]
except (ImportError, ModuleNotFoundError, AttributeError, ValueError, TypeError, RuntimeError):
    _MDS_backend = None

try:
    from services.market_data import MarketDataService as _MDS_services  # type: ignore[import-not-found]
except (ImportError, ModuleNotFoundError, AttributeError, ValueError, TypeError, RuntimeError):
    _MDS_services = None

try:
    import requests
except (ImportError, ModuleNotFoundError, AttributeError, ValueError, TypeError, RuntimeError):
    requests = None

logger = logging.getLogger(__name__)


async def _maybe_await(x: Any) -> Any:
    """Await if awaitable; otherwise return value."""
    if inspect.isawaitable(x):
        return await x
    return x


class ServiceManager:
    """Manages all application services."""

    def __init__(self) -> None:
        self.market_data_service: Any | None = None
        self.connection_manager: Any | None = None
        self.redis_client: Any | None = None
        self._initialized = False

    async def initialize_services(self) -> None:
        """Initialize all services (idempotent)."""
        if self._initialized:
            logger.info("ServiceManager already initialized")
            return

        try:
            # Initialize connection manager
            try:
                get_conn_mgr = None
                # Project-root import
                if _gcm_project is not None:
                    get_conn_mgr = _gcm_project
                # Backend namespace fallback
                elif _gcm_backend is not None:
                    get_conn_mgr = _gcm_backend

                if get_conn_mgr is not None:
                    logger.info("Initializing connection manager...")
                    self.connection_manager = get_conn_mgr()

                    # Prefer explicit async initializer if provided
                    init_fn = None
                    if hasattr(self.connection_manager, "initialize_connections"):
                        init_fn = self.connection_manager.initialize_connections
                    elif hasattr(self.connection_manager, "initialize"):
                        init_fn = self.connection_manager.initialize

                    if callable(init_fn):
                        await _maybe_await(init_fn())
                    else:
                        logger.info("Connection manager has no explicit initialize routine")

                    # Capture redis client if exposed by connection manager (after init)
                    self.redis_client = getattr(self.connection_manager, "redis_client", None)
                else:
                    logger.info("ConnectionManager not available - using RedisService directly")
                    # Use shared Redis service directly
                    try:
                        from backend.services.redis_service import get_redis_service

                        redis_svc = get_redis_service()
                        self.redis_client = redis_svc
                        logger.info("Using shared RedisService as redis_client")
                    except (ImportError, ModuleNotFoundError) as re:
                        logger.warning("RedisService not available: %s", re)

                # Health check (best-effort)
                redis_ok = False
                if self.connection_manager is not None and hasattr(self.connection_manager, "check_redis_health"):
                    try:
                        health = self.connection_manager.check_redis_health()  # type: ignore[attr-defined]
                        redis_ok = bool(health.get("connected"))
                    except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError):
                        redis_ok = False
                elif self.redis_client is not None:
                    try:
                        # Support both sync and async redis clients
                        pong = getattr(self.redis_client, "ping", None)
                        if callable(pong):
                            res = pong()
                            redis_ok = bool(res if not inspect.isawaitable(res) else await res)
                    except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError):
                        redis_ok = False

                if redis_ok:
                    logger.info("Redis connection established")
                else:
                    logger.warning("Redis not available - limited functionality")
            except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
                logger.warning("ConnectionManager not available: %s", e)

            # Initialize market data service (optional)
            try:
                MarketDataService = None  # type: ignore[assignment]
                if _MDS_backend is not None:
                    MarketDataService = _MDS_backend
                elif _MDS_services is not None:
                    MarketDataService = _MDS_services

                if MarketDataService is not None:
                    logger.info("Initializing market data service...")
                    # Use singleton pattern to prevent duplicate instances/background tasks
                    self.market_data_service = MarketDataService.shared() if hasattr(MarketDataService, "shared") else MarketDataService()
                    if hasattr(self.market_data_service, "initialize"):
                        await _maybe_await(self.market_data_service.initialize())
                    logger.info("Market data service initialized")
                else:
                    logger.info("MarketDataService not available")
            except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
                logger.warning("MarketDataService initialization failed: %s", e)

            self._initialized = True
            logger.info("Service initialization completed")
        except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
            logger.exception("Service initialization error: %s", e)
            raise

    async def shutdown_services(self) -> None:
        """Shutdown all services gracefully."""
        try:
            logger.info("Shutting down services...")
            # Close market data service
            try:
                if self.market_data_service and hasattr(self.market_data_service, "close"):
                    await _maybe_await(self.market_data_service.close())
            except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
                logger.warning("Market data service shutdown issue: %s", e)

            # Close connections
            try:
                if self.connection_manager and hasattr(self.connection_manager, "close_connections"):
                    await _maybe_await(self.connection_manager.close_connections())
            except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
                logger.warning("Connection manager shutdown issue: %s", e)

            logger.info("Services shut down successfully")
        except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
            logger.exception("Shutdown error: %s", e)

    def get_health_status(self) -> dict[str, Any]:
        """Get health status of all services."""
        return {
            "backend": "online",
            "database": "online" if self._check_database_health() else "offline",
            "api": "online" if self._check_api_health() else "offline",
            "market_data": "online" if self.market_data_service else "offline",
            "connection_manager": "online" if self.connection_manager else "offline",
            "redis": "online" if self.redis_client else "offline",
            "initialized": self._initialized,
        }

    def _check_database_health(self) -> bool:
        """Check database health status"""
        if sqlite3 is None:
            return False
        conn = None
        try:
            conn = sqlite3.connect(DATABASE_PATH)
            return True
        except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError):
            return False
        finally:
            if conn is not None:
                try:
                    conn.close()
                except Exception:
                    logger.debug("sqlite conn.close() failed", exc_info=True)
                    pass

    def _check_api_health(self) -> bool:
        """Check API health status"""
        try:
            # Since we're running inside the API server, just check if we're initialized
            # Making HTTP requests to ourselves creates circular dependencies
            return self._initialized and self.redis_client is not None
        except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError):
            return False

    @property
    def is_initialized(self) -> bool:
        """Check if services are initialized."""
        return self._initialized


# Global service manager instance
service_manager = ServiceManager()

__all__ = ["ServiceManager", "service_manager"]
