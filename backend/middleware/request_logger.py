"""
Request Logger Middleware - All Live Data, No Fallback/Hardcoded Data

This module provides request logging middleware for live API requests (backend port 8000).
All operations:
- Log live API requests to backend (port 8000)
- Track live request/response times and status codes
- Record live client IP addresses and request methods
- No fallback/hardcoded data - all logging from live operations
- Used by backend services on port 8000 for live trading operations

Live Data Sources:
- API requests: Live requests to backend API (port 8000)
- Client IPs: Live client IP addresses from requests
- Response times: Live request/response handling durations
- Status codes: Live HTTP status codes from responses
- All logging uses live data - no mock/test data

Endpoint References:
- Backend API: Port 8000 (request logger processes live requests)
- All request logging uses live connections - no fallback/hardcoded data
"""

import logging
import time
from collections.abc import Awaitable, Callable

from fastapi import Request, Response
from fastapi.responses import JSONResponse

from backend.utils.enhanced_logging import performance_logger

logger = logging.getLogger(__name__)


@performance_logger("request_logger")
async def request_logger_middleware(request: Request, call_next: Callable[[Request], Awaitable[Response]]) -> Response | JSONResponse:
    """
    Request logging middleware for live API requests (backend port 8000).

    Logs live request/response data for observability.
    All logging uses live data - no fallback/hardcoded data.

    Args:
        request: Live API request to backend (port 8000)
        call_next: Next middleware handler

    Returns:
        Live API response
    """
    start_time = time.time()

    # Log live request
    client_ip = request.client.host if request.client else "unknown"
    logger.info("Request: %s %s from %s", request.method, request.url.path, client_ip)

    # Process live API request
    try:
        response = await call_next(request)

        # Calculate live request duration
        duration = time.time() - start_time

        # Log live response
        logger.info("Response: %s for %s %s (%.3fs)", response.status_code, request.method, request.url.path, duration)
    except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError):
        # Calculate live request duration
        duration = time.time() - start_time

        # Log live exception (not fallback data, error handling)
        logger.exception("Request logger caught exception for live request %s %s: %.3fs", request.method, request.url.path, duration)

        # Return error response (not fallback data, error handling)
        return JSONResponse(status_code=500, content={"detail": "Internal server error"})
    else:
        return response
