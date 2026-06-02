"""
Exception Handling - Live Configuration Only

Comprehensive exception classes and error handling utilities for the Mystic Trading System.
All configuration values come from live config - no hardcoded values.
"""

import logging
import os
from collections.abc import Awaitable, Callable
from enum import Enum
from typing import Any

# Import live configuration
try:
    from backend.config_bridge import get_mystic_config

    _mystic_config = get_mystic_config()
except (ImportError, AttributeError, ValueError, TypeError, RuntimeError):
    _mystic_config = None

logger = logging.getLogger(__name__)

# --- Live Configuration Helpers -------------------------------------------------------------------


def _get_default_http_status_code() -> int:
    """Get default HTTP status code from live config."""
    if _mystic_config:
        try:
            value = _mystic_config
            if hasattr(value, "exceptions") and hasattr(value.exceptions, "default_http_status_code"):
                status_code = value.exceptions.default_http_status_code
                if isinstance(status_code, int) and 400 <= status_code < 600:
                    return status_code
        except (AttributeError, ValueError, TypeError):
            pass

    status_code = os.getenv("EXCEPTIONS_DEFAULT_HTTP_STATUS_CODE", "").strip()
    if status_code:
        try:
            code = int(status_code)
            if 400 <= code < 600:
                return code
        except (ValueError, TypeError):
            pass

    return 500


class ErrorCode(Enum):
    UNKNOWN_ERROR = 1000
    AI_MODEL_ERROR = 6000


class MysticError(Exception):
    def __init__(
        self,
        message: str,
        error_code: ErrorCode = ErrorCode.UNKNOWN_ERROR,
        details: dict[str, Any] | None = None,
        original_exception: Exception | None = None,
    ):
        self.message = message
        self.error_code = error_code
        self.details = details or {}
        self.original_exception = original_exception
        super().__init__(self.message)

    def to_dict(self) -> dict[str, Any]:
        return {
            "error": True,
            "error_code": self.error_code.value,
            "error_type": self.error_code.name,
            "message": self.message,
            "details": self.details,
        }


class ModelError(Exception):
    pass


class NotificationError(Exception):
    pass


class StrategyError(Exception):
    pass


class TradingError(Exception):
    pass


class AIError(Exception):
    pass


class MarketDataError(Exception):
    pass


class DatabaseConnectionError(Exception):
    pass


class DatabaseError(Exception):
    """General database-related exceptions"""


class AnalyticsError(MysticError):
    """Analytics-related exceptions"""

    def __init__(
        self,
        message: str,
        details: dict[str, Any] | None = None,
        original_exception: Exception | None = None,
    ):
        super().__init__(message, ErrorCode.AI_MODEL_ERROR, details, original_exception)


class MetricsError(MysticError):
    """Metrics-related exceptions"""

    def __init__(
        self,
        message: str,
        details: dict[str, Any] | None = None,
        original_exception: Exception | None = None,
    ):
        super().__init__(message, ErrorCode.AI_MODEL_ERROR, details, original_exception)


class RateLimitError(Exception):
    """Exception raised when API rate limit is exceeded."""


def handle_async_exception(
    error_message: str,
    exception_class: type[Exception] = Exception,
    reraise: bool = True,
    default_return: Any = None,
) -> Callable[[Callable[..., Awaitable[Any]]], Callable[..., Awaitable[Any]]]:
    """Decorator to handle exceptions in async functions."""

    def decorator(
        func: Callable[..., Awaitable[Any]],
    ) -> Callable[..., Awaitable[Any]]:
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            try:
                return await func(*args, **kwargs)
            except exception_class:
                logger.exception(f"{error_message}")
                if reraise:
                    raise
                return default_return

        return wrapper

    return decorator


def handle_exception(
    error_message: str,
    exception_class: type[Exception] = Exception,
    reraise: bool = True,
    default_return: Any = None,
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Decorator to handle exceptions in synchronous functions."""

    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            try:
                return func(*args, **kwargs)
            except exception_class:
                logger.exception(f"{error_message}")
                if reraise:
                    raise
                return default_return

        return wrapper

    return decorator


def _get_status_code_for_error_code(error_code: int | ErrorCode) -> int:
    """Map error codes to HTTP status codes for error handling."""
    default_status = _get_default_http_status_code()
    # Default mapping, can be extended as needed
    code_map = {
        1000: default_status,  # UNKNOWN_ERROR -> Internal Server Error
        6000: default_status,  # AI_MODEL_ERROR -> Internal Server Error
        400: 400,  # Bad Request
        401: 401,  # Unauthorized
        403: 403,  # Forbidden
        404: 404,  # Not Found
        409: 409,  # Conflict
        422: 422,  # Unprocessable Entity
        429: 429,  # Too Many Requests
        500: default_status,  # Internal Server Error
    }
    if isinstance(error_code, ErrorCode):
        return code_map.get(error_code.value, default_status)
    return code_map.get(int(error_code), default_status)
