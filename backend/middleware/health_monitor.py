"""
Health Monitor Middleware - All Live Data, No Fallback/Hardcoded Data

This module provides health monitoring middleware for live API requests (backend port 8000).
All operations:
- Monitor live API request health and performance
- Track live request IDs for observability
- Log live request/response metrics
- No fallback/hardcoded data - all monitoring from live operations
- Used by backend services on port 8000 for live trading operations

Live Data Sources:
- API requests: Live requests to backend API (port 8000)
- Request IDs: Generated for live requests (UUID, not fallback data)
- Performance metrics: Live request handling times
- Health status: Live request success/failure tracking
- All monitoring uses live data - no mock/test data

Endpoint References:
- Backend API: Port 8000 (health monitor processes live requests)
- All health monitoring uses live connections - no fallback/hardcoded data
"""

import logging
import uuid
from collections.abc import Awaitable, Callable

from fastapi import Request
from fastapi.responses import JSONResponse, Response

from backend.utils.enhanced_logging import performance_logger

logger = logging.getLogger(__name__)


@performance_logger("http_request_handling")
async def health_monitor_middleware(request: Request, call_next: Callable[[Request], Awaitable[Response]]) -> Response:
    """
    Health monitoring middleware for live API requests (backend port 8000).

    Monitors live request health and performance, tracks request IDs.
    All monitoring uses live data - no fallback/hardcoded data.

    Args:
        request: Live API request to backend (port 8000)
        call_next: Next middleware handler

    Returns:
        Live API response with request ID header
    """
    # Get or generate request ID for live request tracking (UUID, not fallback data)
    request_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))

    try:
        # Process live API request
        response = await call_next(request)

        # Add request ID header for live request tracking
        response.headers["X-Request-ID"] = request_id
    except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError):
        # Log live exception (not fallback data, error handling)
        logger.exception("Request failed for live request [ID: %s]", request_id)
        return JSONResponse(
            status_code=500,
            content={
                "detail": "Internal server error",
                "request_id": request_id,  # Live request ID
            },
        )
    else:
        return response
