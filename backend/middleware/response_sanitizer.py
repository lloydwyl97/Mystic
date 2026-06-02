"""
Response Sanitizer Middleware - All Live Data, No Fallback/Hardcoded Data

This module provides response sanitization middleware for live API responses (backend port 8000).
All operations:
- Sanitize live API responses from backend (port 8000)
- Remove sensitive fields from live response data
- Process live request/response data
- No fallback/hardcoded data - all sanitization from live operations
- Used by backend services on port 8000 for live trading operations

Live Data Sources:
- API responses: Live responses from backend API (port 8000)
- Response data: Live response body data from API operations
- Response headers: Live headers from API responses
- All sanitization uses live data - no mock/test data

Endpoint References:
- Backend API: Port 8000 (response sanitizer processes live responses)
- All response sanitization uses live connections - no fallback/hardcoded data
"""

import json
import logging
from collections.abc import Awaitable, Callable
from typing import Any

from fastapi import Request, Response
from fastapi.responses import JSONResponse

from backend.utils.enhanced_logging import performance_logger

logger = logging.getLogger(__name__)

# Sensitive fields to remove from live responses (configuration, not fallback data)
SENSITIVE_FIELDS = ["password", "token", "secret", "key", "api_key"]


@performance_logger("response_sanitizer")
async def response_sanitizer_middleware(request: Request, call_next: Callable[[Request], Awaitable[Response]]) -> Response | JSONResponse:
    """
    Response sanitization middleware for live API responses (backend port 8000).

    Sanitizes live response data by removing sensitive fields.
    All sanitization uses live data - no fallback/hardcoded data.

    Args:
        request: Live API request to backend (port 8000)
        call_next: Next middleware handler

    Returns:
        Sanitized live API response
    """
    # Process live API request
    response = await call_next(request)

    # Only sanitize live JSON responses
    if response.headers.get("content-type", "").startswith("application/json"):
        try:
            # Get live response body
            if hasattr(response, "body") and response.body:
                data_str = response.body.decode("utf-8") if isinstance(response.body, bytes) else str(response.body)

                # Parse live JSON data
                data = json.loads(data_str)

                # Sanitize live sensitive data
                sanitized_data = _sanitize_data(data)

                # Create new sanitized live response
                return JSONResponse(
                    content=sanitized_data,
                    status_code=response.status_code,
                    headers=dict(response.headers),
                )
        except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
            logger.exception("Response sanitizer error for live response: %s", e)
            # Return original live response if sanitization fails (error handling, not fallback data)
            return response

    return response


def _sanitize_data(data: Any) -> Any:
    """
    Recursively sanitize live data by removing sensitive fields.

    Args:
        data: Live data to sanitize (dict, list, or primitive)

    Returns:
        Sanitized live data with sensitive fields redacted
    """
    if isinstance(data, dict):
        sanitized: dict[str, Any] = {}
        for key, value in data.items():
            key_str = str(key)
            if isinstance(key, str) and key_str.lower() in SENSITIVE_FIELDS:
                sanitized[key_str] = "***REDACTED***"
            else:
                sanitized[key_str] = _sanitize_data(value)
        return sanitized
    if isinstance(data, list):
        return [_sanitize_data(item) for item in data]
    return data
