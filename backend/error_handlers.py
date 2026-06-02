#!/usr/bin/env python3
"""
Error Handlers for Mystic Trading Platform

Centralized error handling for the application using standardized exceptions.
Windows/Python 3.12+ compatible with proper error handling, lightweight responses,
and dashboard integration. Fixed for reliable page loading and responsive UI.
"""

import logging
import os
import traceback
import uuid
from datetime import datetime, timezone
from typing import Any, cast

from fastapi import FastAPI, HTTPException, Request, status
from fastapi.responses import JSONResponse

# Direct imports for production
try:
    from slowapi.errors import RateLimitExceeded

    from backend.utils.exceptions import (
        ErrorCode,
        MysticException,
        _get_status_code_for_error_code,
    )

    SLOWAPI_AVAILABLE = True
    BACKEND_UTILS_AVAILABLE = True
except ImportError:
    SLOWAPI_AVAILABLE = False
    BACKEND_UTILS_AVAILABLE = False


# MysticException and _get_status_code_for_error_code are imported from backend.utils.exceptions


logger = logging.getLogger(__name__)

# Global state for handler registration
# Error handlers registration state - using dict to avoid global keyword
_error_handlers_registered_state: dict[str, bool] = {"registered": False}


def _now_iso() -> str:
    """Get current timestamp in ISO format with Z suffix for consistency"""
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _coerce_retry_after(value: Any, default: int = 60) -> int:
    """Coerce retry-after value to integer with fallback"""
    try:
        if value is None:
            return default
        if isinstance(value, (int, float)):
            result = int(value)
        elif isinstance(value, str) and value.strip().isdigit():
            result = int(value.strip())
        else:
            result = default
    except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError):
        return default
    else:
        return result


def _get_request_id(request: Request) -> str:
    """Get request ID from headers or generate one"""
    # Check for common request ID headers
    request_id = request.headers.get("x-request-id") or request.headers.get("x-correlation-id")
    if not request_id:
        request_id = str(uuid.uuid4())
    return request_id


def _normalize_message(detail: Any) -> str:
    """Normalize error detail to string message"""
    if isinstance(detail, str):
        return detail
    if isinstance(detail, dict):
        # Extract message from structured error
        return detail.get("message", str(detail))
    return str(detail)


def _create_error_response(
    error_code: str,
    error_type: str,
    message: str,
    _status_code: int,
    details: dict | None = None,
    include_trace: bool = False,
    request: Request | None = None,
) -> dict[str, Any]:
    """Create standardized error response"""
    response = {
        "error": True,
        "error_code": error_code,
        "error_type": error_type,
        "message": message,
        "timestamp": _now_iso(),
    }

    if details:
        response["details"] = details

    # Add request context if available
    if request:
        request_context = {
            "path": str(request.url.path),
            "method": request.method,
            "request_id": _get_request_id(request),
        }
        if details:
            response["details"].update(request_context)
        else:
            response["details"] = request_context

    # Only include trace in debug mode for 5xx errors
    if include_trace and os.getenv("DEBUG", "false").lower() == "true":
        response["trace"] = traceback.format_exc()

    return response


async def rate_limit_handler(request: Request, exc: RateLimitExceeded):
    """Handle rate limit exceeded errors with lightweight response"""
    default_retry_after = 60
    retry_after: int = default_retry_after

    # Extract retry_after from exception
    if hasattr(exc, "detail"):
        if isinstance(exc.detail, dict):
            detail_dict: dict[str, Any] = cast(dict[str, Any], exc.detail)
            retry_after = _coerce_retry_after(detail_dict.get("retry_after"), default_retry_after)
        elif hasattr(exc.detail, "retry_after"):
            retry_after = _coerce_retry_after(getattr(exc.detail, "retry_after", None), default_retry_after)

    if hasattr(exc, "retry_after"):
        retry_after = _coerce_retry_after(getattr(exc, "retry_after", None), retry_after)

    # Create lightweight response (no trace for 4xx errors)
    response_data = _create_error_response(
        error_code=ErrorCode.RATE_LIMIT_EXCEEDED,
        error_type="RateLimitExceeded",
        message="Rate limit exceeded",
        _status_code=status.HTTP_429_TOO_MANY_REQUESTS,
        details={"retry_after": retry_after},
        include_trace=False,  # No trace for 4xx errors
        request=request,
    )

    return JSONResponse(
        status_code=status.HTTP_429_TOO_MANY_REQUESTS,
        content=response_data,
        headers={"Retry-After": str(retry_after)},
    )


async def custom_rate_limit_handler(request: Request, exc: HTTPException):
    """Handle HTTP 429 rate limit errors"""
    if exc.status_code == status.HTTP_429_TOO_MANY_REQUESTS:
        retry_after = 60
        if hasattr(exc, "detail") and isinstance(exc.detail, dict):
            retry_after = _coerce_retry_after(exc.detail.get("retry_after"), retry_after)

        # Create lightweight response (no trace for 4xx errors)
        response_data = _create_error_response(
            error_code=ErrorCode.RATE_LIMIT_EXCEEDED,
            error_type="RateLimitExceeded",
            message="Rate limit exceeded",
            _status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            details={"retry_after": retry_after},
            include_trace=False,  # No trace for 4xx errors
            request=request,
        )

        return JSONResponse(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            content=response_data,
            headers={"Retry-After": str(retry_after)},
        )

    return await generic_exception_handler(request, exc)


async def generic_exception_handler(request: Request, exc: Exception):
    """Handle all exceptions with appropriate response"""
    if isinstance(exc, MysticException):
        status_code = _get_status_code_for_error_code(exc.error_code)
        return JSONResponse(status_code=status_code, content=exc.to_dict())

    if isinstance(exc, HTTPException):
        # Normalize message from detail
        message = _normalize_message(exc.detail)

        # Create lightweight response (no trace for 4xx errors)
        response_data = _create_error_response(
            error_code=str(ErrorCode.UNKNOWN_ERROR.value) if hasattr(ErrorCode.UNKNOWN_ERROR, "value") else str(ErrorCode.UNKNOWN_ERROR),
            error_type="HTTPException",
            message=message,
            _status_code=exc.status_code,
            details={},
            include_trace=False,  # No trace for 4xx errors
            request=request,
        )

        return JSONResponse(
            status_code=exc.status_code,
            content=response_data,
        )

    msg = str(exc)
    if "our state is ERROR" in msg or "BrokenPipeError" in msg or "ConnectionResetError" in msg:
        logger.debug("Client disconnect (suppressed): %s", exc)
    else:
        logger.error("Unhandled exception: %s\n%s", exc, traceback.format_exc())

    # Create response for 5xx errors
    debug = os.getenv("DEBUG", "false").lower() == "true"
    response_data = _create_error_response(
        error_code=str(ErrorCode.UNKNOWN_ERROR.value) if hasattr(ErrorCode.UNKNOWN_ERROR, "value") else str(ErrorCode.UNKNOWN_ERROR),
        error_type="InternalServerError",
        message=str(exc) if debug else "An unexpected error occurred",
        _status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        details={},
        include_trace=True,  # Include trace for 5xx errors in debug mode
        request=request,
    )

    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content=response_data,
    )


# Error handlers registration state - using dict to avoid global keyword
_error_handlers_registered_state: dict[str, bool] = {"registered": False}


def register_error_handlers(app: FastAPI, force: bool = False) -> None:
    """Register error handlers with idempotent behavior and graceful fallbacks"""
    if _error_handlers_registered_state["registered"] and not force:
        logger.info("Error handlers already registered, skipping")
        return

    logger.info("Registering error handlers...")

    if SLOWAPI_AVAILABLE and RateLimitExceeded:
        app.add_exception_handler(RateLimitExceeded, rate_limit_handler)
        logger.info("[OK] RateLimitExceeded handler registered")
    else:
        logger.warning("[WARN] slowapi not available, skipping RateLimitExceeded handler")

    # Register HTTP exception handler
    app.add_exception_handler(HTTPException, custom_rate_limit_handler)
    logger.info("[OK] HTTPException handler registered")

    # Register generic exception handler
    app.add_exception_handler(Exception, generic_exception_handler)
    logger.info("[OK] Generic exception handler registered")

    _error_handlers_registered_state["registered"] = True
    logger.info("Error handlers registration completed")


def get_error_handler_status() -> dict[str, Any]:
    """Get status of error handler dependencies"""
    return {
        "handlers_registered": _error_handlers_registered_state["registered"],
        "slowapi_available": SLOWAPI_AVAILABLE,
        "backend_utils_available": BACKEND_UTILS_AVAILABLE,
        "error_codes_available": list(ErrorCode.__dict__.keys()) if hasattr(ErrorCode, "__dict__") else [],
    }
