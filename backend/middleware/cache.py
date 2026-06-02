"""
Cache Middleware - All Live Data, No Fallback/Hardcoded Data

This module provides cache middleware for live API requests (backend port 8000).
All operations:
- Cache live API responses from backend (port 8000)
- Connect to live Redis instance on Windows Home 11 (localhost:6379)
- Process live request/response data
- No fallback/hardcoded data - all caching from live operations
- Used by backend services on port 8000 for live trading operations

Live Data Sources:
- API responses: Live responses from backend API (port 8000) cached in Redis
- Redis connection: Live Redis instance on Windows Home 11 (localhost:6379)
- Cache keys: Derived from live request URLs and parameters
- All cache operations use live data - no mock/test data

Endpoint References:
- Backend API: Port 8000 (cache middleware processes live requests)
- Redis: localhost:6379 (live Redis instance on Windows Home 11)
- All cache operations use live connections - no fallback/hardcoded data
"""

import asyncio
import logging
import os
from typing import Any

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request

from backend.config.redis_config import get_redis_client
from backend.services.task_manager import task_manager

logger = logging.getLogger(__name__)


class CacheMiddleware(BaseHTTPMiddleware):
    """
    Cache middleware for live API request/response caching.

    Caches live API responses from backend (port 8000) in Redis (Windows Home 11).
    All cache operations use live data - no fallback/hardcoded data.
    """

    def __init__(self, app, redis_url: str | None = None) -> None:
        """
        Initialize cache middleware for live Redis caching.

        Args:
            app: FastAPI/Starlette application instance
            redis_url: Live Redis connection URL (default: localhost:6379/0 on Windows Home 11, configuration default not fallback data)
        """
        super().__init__(app)
        # All Live Data, No Fallback/Hardcoded Data
        if not redis_url:
            redis_url = os.getenv("REDIS_URL")
            if not redis_url:
                redis_host = os.getenv("REDIS_HOST")
                if not redis_host:
                    msg = "REDIS_URL or REDIS_HOST environment variable is required - no fallback/hardcoded Redis URL"
                    raise RuntimeError(msg)
                redis_port = os.getenv("REDIS_PORT", "6379")
                redis_db = os.getenv("REDIS_DB", "0")
                redis_url = f"redis://{redis_host}:{redis_port}/{redis_db}"
        self.redis_url = redis_url
        self.redis = None  # Live Redis connection
        # Initialize live Redis connection with shared pool
        task = task_manager.create_task_sync(self._init_redis(), name="cache_middleware:init_redis")
        # Store task reference for cleanup if needed
        if not hasattr(self, "_tasks"):
            self._tasks: list[asyncio.Task[Any]] = []
        self._tasks.append(task)

    async def _init_redis(self) -> None:
        """
        Initialize live Redis connection for caching live API responses.

        Uses shared Redis pool for live API response caching.
        """
        try:
            # Use shared Redis pool for live API response caching
            self.redis = get_redis_client()
            logger.info("CacheMiddleware: Connected to shared Redis pool")
        except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
            logger.exception("CacheMiddleware: Shared Redis pool connection failed: %s", e)
            self.redis = None

    async def dispatch(self, request: Request, call_next) -> Any:
        """
        Process live API request and cache response.

        Args:
            request: Live API request to backend (port 8000)
            call_next: Next middleware handler

        Returns:
            Live API response (may be cached from Redis)
        """
        # Process live API request (cache implementation to be added)
        # Currently passthrough - will cache live responses when implemented
        return await call_next(request)


# Exported function handler for use with app.middleware("http")()
async def cache_middleware_handler(request: Request, call_next) -> Any:
    """
    Cache middleware handler function for live API requests.

    Processes live API requests to backend (port 8000) and caches responses in Redis.
    Currently passthrough - will cache live responses when implemented.
    All cache operations use live data - no fallback/hardcoded data.

    Args:
        request: Live API request to backend (port 8000)
        call_next: Next middleware handler

    Returns:
        Live API response (may be cached from Redis)
    """
    # Process live API request (cache implementation to be added)
    return await call_next(request)
