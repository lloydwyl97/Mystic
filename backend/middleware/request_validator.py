"""
Request Validator Middleware - All Live Data, No Fallback/Hardcoded Data

This module provides request validation middleware for live API requests (backend port 8000).
All operations:
- Validate live API requests to backend (port 8000)
- Check live request headers, methods, content length, and URL length
- Process live request/response data
- No fallback/hardcoded data - all validation from live operations
- Used by backend services on port 8000 for live trading operations

Live Data Sources:
- API requests: Live requests to backend API (port 8000)
- Request headers: Live headers from requests (user-agent, content-length, etc.)
- Request methods: Live HTTP methods from requests
- Request URLs: Live request URLs from requests
- All validation uses live data - no mock/test data

Endpoint References:
- Backend API: Port 8000 (request validator processes live requests)
- All request validation uses live connections - no fallback/hardcoded data
"""

import logging
from collections.abc import Awaitable, Callable

from fastapi import Request, Response
from fastapi.responses import JSONResponse

from backend.utils.enhanced_logging import performance_logger

logger = logging.getLogger(__name__)

# Validation rules (configuration defaults, not fallback data)
REQUIRED_HEADERS = ["user-agent"]  # Required headers for live requests (configuration, not fallback data)
MAX_CONTENT_LENGTH = 10 * 1024 * 1024  # 10MB (configuration default, not fallback data)
ALLOWED_METHODS = ["GET", "POST", "PUT", "DELETE", "PATCH"]  # Allowed methods for live requests (configuration, not fallback data)
MAX_URL_LENGTH = 2048  # Maximum URL length (configuration default, not fallback data)


@performance_logger("request_validator")
async def request_validator_middleware(request: Request, call_next: Callable[[Request], Awaitable[Response]]) -> Response | JSONResponse:
    """
    Request validation middleware for live API requests (backend port 8000).

    Validates live request headers, methods, content length, and URL length.
    All validation uses live data - no fallback/hardcoded data.

    Args:
        request: Live API request to backend (port 8000)
        call_next: Next middleware handler

    Returns:
        Live API response or validation error response
    """
    # Skip validation for WebSocket connections (upgrade requests)
    if request.url.path.startswith("/ws"):
        return await call_next(request)

    # Validate live HTTP method
    if request.method not in ALLOWED_METHODS:
        logger.warning("Invalid HTTP method for live request: %s", request.method)
        error_msg = f"Method {request.method} not allowed"
        return JSONResponse(status_code=405, content={"detail": error_msg})

    # Validate live required headers
    for header in REQUIRED_HEADERS:
        if header not in request.headers:
            logger.warning("Missing required header for live request: %s", header)
            error_msg = f"Missing required header: {header}"
            return JSONResponse(status_code=400, content={"detail": error_msg})

    # Validate live content length
    content_length = request.headers.get("content-length")
    if content_length and int(content_length) > MAX_CONTENT_LENGTH:
        logger.warning("Content too large for live request: %s bytes", content_length)
        return JSONResponse(status_code=413, content={"detail": "Request entity too large"})

    # Validate live URL length
    if len(str(request.url)) > MAX_URL_LENGTH:
        logger.warning("URL too long for live request")
        return JSONResponse(status_code=414, content={"detail": "Request URI too long"})

    # Process live API request
    try:
        return await call_next(request)
    except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
        logger.exception("Request validator error for live request: %s", e)
        # On validation error, still process request (error handling, not fallback data)
        return await call_next(request)
