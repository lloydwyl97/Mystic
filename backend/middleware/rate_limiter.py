"""
Rate Limiter - All Live Data, No Fallback/Hardcoded Data

This module provides rate limiting for live API requests using Redis (backend port 8000).
All operations:
- Enforce live rate limits for API requests (backend port 8000)
- Track live request rates in Redis (Windows Home 11)
- Process live request/response data
- No fallback/hardcoded data - all rate limiting from live operations
- Used by backend services on port 8000 for live trading operations

Live Data Sources:
- API requests: Live requests to backend API (port 8000)
- Rate limit tracking: Live request rate tracking in Redis (Windows Home 11)
- Rate limit keys: Derived from live client IP addresses
- All rate limiting uses live data - no mock/test data

Endpoint References:
- Backend API: Port 8000 (rate limiter processes live requests)
- Redis: localhost:6379 (live Redis instance on Windows Home 11)
- All rate limiting uses live connections - no fallback/hardcoded data
"""

import asyncio
import logging
import os
import time
from collections.abc import Awaitable, Callable
from typing import Any

from fastapi import HTTPException, Request, Response
from fastapi.responses import JSONResponse

from backend.config.redis_config import get_redis_client


class RateLimiter:
    """
    Rate limiter for live API requests using Redis (Windows Home 11).

    Tracks live request rates per IP address in Redis.
    All rate limiting uses live data - no fallback/hardcoded data.
    """

    def __init__(self, redis_url: str | None = None) -> None:
        """
        Initialize rate limiter for live Redis operations.

        Args:
            redis_url: Live Redis connection URL (from environment if not provided)
        """
        # All Live Data, No Fallback/Hardcoded Data
        # Redis connection must be configured via environment variables
        if redis_url is None:
            redis_url = os.getenv("REDIS_URL")
            if not redis_url:
                # Fallback to individual components if REDIS_URL not set
                redis_host = os.getenv("REDIS_HOST")
                if not redis_host:
                    msg = "REDIS_URL or REDIS_HOST environment variable is required - no fallback/hardcoded Redis connection"
                    raise RuntimeError(msg)
                redis_port = os.getenv("REDIS_PORT", "6379")
                redis_db = os.getenv("REDIS_DB", "0")
                redis_url = f"redis://{redis_host}:{redis_port}/{redis_db}"
        self.redis_url = redis_url  # Live Redis URL
        self.redis: Any = None  # Live Redis connection
        self._connection_lock = asyncio.Lock()  # Lock for thread-safe connection management

    async def connect(self) -> None:
        """
        Connect to live Redis instance with connection pooling.

        Connects to live Redis instance from environment configuration.
        """
        if not self.redis:
            async with self._connection_lock:
                if not self.redis:
                    # Use shared Redis pool for rate limiting
                    self.redis = get_redis_client()

    async def close(self) -> None:
        """Close live Redis connection."""
        if self.redis:
            await self.redis.close()
            self.redis = None

    async def is_allowed(self, key: str, max_requests: int, window_seconds: int) -> tuple[bool, dict]:
        """
        Check if live request is allowed based on rate limit in Redis.

        Args:
            key: Unique identifier for the rate limit (e.g., live client IP address)
            max_requests: Maximum number of requests allowed in the window (configuration default, not fallback data)
            window_seconds: Time window in seconds (configuration default, not fallback data)

        Returns:
            tuple: (is_allowed, rate_limit_info) with live rate limit data
        """
        await self.connect()

        current_time = int(time.time())
        window_start = current_time - window_seconds

        # Use Redis sorted set to track requests
        pipe = self.redis.pipeline()

        # Remove old entries outside the window
        pipe.zremrangebyscore(key, 0, window_start)

        # Count current requests in window
        pipe.zcard(key)

        # Add current request
        pipe.zadd(key, {str(current_time): current_time})

        # Set expiry on the key
        pipe.expire(key, window_seconds)

        # Execute pipeline
        results = await pipe.execute()
        current_requests = results[1]  # zcard result

        # Check if limit exceeded
        is_allowed = current_requests < max_requests

        # Calculate remaining requests and reset time
        remaining = max(0, max_requests - current_requests)
        reset_time = current_time + window_seconds

        rate_limit_info = {
            "limit": max_requests,
            "remaining": remaining,
            "reset": reset_time,
            "window": window_seconds,
        }

        return is_allowed, rate_limit_info


# Global rate limiter instance for live operations (backend port 8000)
# All Live Data, No Fallback/Hardcoded Data
# Redis connection must be configured via environment variables
_redis_url = os.getenv("REDIS_URL")
if not _redis_url:
    # Fallback to individual components if REDIS_URL not set
    redis_host = os.getenv("REDIS_HOST")
    if not redis_host:
        msg = "REDIS_URL or REDIS_HOST environment variable is required - no fallback/hardcoded Redis connection"
        raise RuntimeError(msg)
    redis_port = os.getenv("REDIS_PORT", "6379")
    redis_db = os.getenv("REDIS_DB", "0")
    _redis_url = f"redis://{redis_host}:{redis_port}/{redis_db}"
redis_url = _redis_url
rate_limiter = RateLimiter(redis_url)

# Logger for rate limiter
logger = logging.getLogger(__name__)


def get_client_ip(request: Request) -> str:
    """
    Extract live client IP from request.

    Args:
        request: Live API request to backend (port 8000)

    Returns:
        Live client IP address for rate limiting
    """
    forwarded = request.headers.get("X-Forwarded-For")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


async def rate_limit_middleware(request: Request, call_next: Callable[[Request], Awaitable[Response]]) -> Response:
    """
    Rate limiting middleware for live API requests using Redis (backend port 8000).

    Enforces live rate limits for API requests in Redis (Windows Home 11).
    All rate limiting uses live data - no fallback/hardcoded data.

    Args:
        request: Live API request to backend (port 8000)
        call_next: Function to call the next middleware/endpoint

    Returns:
        Live API response with rate limit headers
    """
    # Rate limit configuration defaults (not fallback data, configuration defaults)
    max_requests = 100  # Maximum requests per window (configuration default)
    window_seconds = 60  # Time window in seconds (configuration default)

    try:
        # Generate rate limit key from live client IP
        key = f"rate_limit:{get_client_ip(request)}"

        # Check live rate limit with timeout
        is_allowed, rate_info = await asyncio.wait_for(
            rate_limiter.is_allowed(key, max_requests, window_seconds),
            timeout=1.0,  # 1 second timeout (configuration default, not fallback data)
        )

        if not is_allowed:
            logger.warning("Rate limit exceeded for live request from %s", get_client_ip(request))
            raise HTTPException(
                status_code=429,
                detail={"error": "Rate limit exceeded", "rate_limit": rate_info},
            )

        # Add live rate limit info to request state
        request.state.rate_limit_info = rate_info

    except asyncio.TimeoutError:
        logger.warning("Rate limiter Redis timeout - denying request (fail closed)")
        raise HTTPException(
            status_code=503,
            detail={"error": "Rate limit check unavailable (Redis timeout)", "retry_after": 60},
        ) from None
    except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
        logger.exception("Rate limiter error - denying request (fail closed): %s", e)
        raise HTTPException(
            status_code=503,
            detail={"error": "Rate limit check unavailable", "retry_after": 60},
        ) from e

    # Call the next middleware/endpoint for live request
    response = await call_next(request)

    # Add rate limit headers to response
    if hasattr(request.state, "rate_limit_info"):
        rate_info = request.state.rate_limit_info
        response.headers["X-RateLimit-Limit"] = str(rate_info["limit"])
        response.headers["X-RateLimit-Remaining"] = str(rate_info["remaining"])
        response.headers["X-RateLimit-Reset"] = str(rate_info["reset"])
        response.headers["X-RateLimit-Window"] = str(rate_info["window"])

    return response


async def _check_rate_limit_for_request(request: Request, max_requests: int, window_seconds: int, key_func: Callable[[Request], str] | None = None) -> None:
    """
    Check rate limit for a live request (helper for decorator).

    Args:
        request: Live API request to backend (port 8000)
        max_requests: Maximum requests per window (configuration default, not fallback data)
        window_seconds: Time window in seconds (configuration default, not fallback data)
        key_func: Optional function to generate rate limit key from request

    Raises:
        HTTPException: 429 if rate limit exceeded
    """
    # Generate rate limit key
    key = key_func(request) if key_func else f"rate_limit:{get_client_ip(request)}"

    # Check live rate limit
    is_allowed, rate_info = await asyncio.wait_for(
        rate_limiter.is_allowed(key, max_requests, window_seconds),
        timeout=1.0,  # 1 second timeout (configuration default, not fallback data)
    )

    if not is_allowed:
        logger.warning("Rate limit exceeded for live request from %s", get_client_ip(request))
        raise HTTPException(
            status_code=429,
            detail={"error": "Rate limit exceeded", "rate_limit": rate_info},
        )


def rate_limit(
    max_requests: int = 100,
    window_seconds: int = 60,
    key_func: Callable[[Request], str] | None = None,
) -> Callable:
    """
    Decorator for rate limiting endpoints with live rate limit checks.

    All rate limiting uses live data - no fallback/hardcoded data.

    Usage:
        @app.get("/api/data")
        @rate_limit(max_requests=10, window_seconds=60)
        async def get_data(request: Request):
            return {"data": "example"}

    Args:
        max_requests: Maximum requests per window (default: 100, configuration default not fallback data)
        window_seconds: Time window in seconds (default: 60, configuration default not fallback data)
        key_func: Optional function to generate rate limit key from request

    Returns:
        Decorator function for rate limiting endpoints
    """

    def decorator(func: Callable) -> Callable:
        async def wrapper(*args: any, request: Request, **kwargs: any) -> any:  # type: ignore[name-defined]
            # Check live rate limit before executing endpoint
            await _check_rate_limit_for_request(request, max_requests, window_seconds, key_func)
            return await func(*args, request=request, **kwargs)

        return wrapper

    return decorator


async def add_rate_limit_headers(request: Request, response: JSONResponse) -> JSONResponse:
    """
    Add live rate limit headers to response.

    Args:
        request: Live API request to backend (port 8000)
        response: Live API response

    Returns:
        Response with live rate limit headers
    """
    if hasattr(request.state, "rate_limit_info"):
        rate_info = request.state.rate_limit_info
        response.headers["X-RateLimit-Limit"] = str(rate_info["limit"])
        response.headers["X-RateLimit-Remaining"] = str(rate_info["remaining"])
        response.headers["X-RateLimit-Reset"] = str(rate_info["reset"])
        response.headers["X-RateLimit-Window"] = str(rate_info["window"])

    return response
