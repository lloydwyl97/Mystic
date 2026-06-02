"""
Standardized Exception Handling for Mystic Trading Platform

Provides consistent exception handling across the entire application.
"""

import logging
import traceback
from collections.abc import Callable
from datetime import datetime, timezone
from enum import Enum
from functools import wraps
from typing import Any

from fastapi import HTTPException, Request
from fastapi.responses import JSONResponse

logger = logging.getLogger(__name__)


class ErrorCode(Enum):
    """Standardized error codes for the application"""

    # General errors (1000-1999)
    UNKNOWN_ERROR = 1000
    VALIDATION_ERROR = 1001
    CONFIGURATION_ERROR = 1002
    TIMEOUT_ERROR = 1003

    # Database errors (2000-2999)
    DATABASE_CONNECTION_ERROR = 2000
    DATABASE_QUERY_ERROR = 2001
    DATABASE_TRANSACTION_ERROR = 2002

    # External API errors (3000-3999)
    API_CONNECTION_ERROR = 3000
    API_RATE_LIMIT_ERROR = 3001
    API_AUTHENTICATION_ERROR = 3002
    API_TIMEOUT_ERROR = 3003
    API_RESPONSE_ERROR = 3004

    # Trading errors (4000-4999)
    TRADING_ORDER_ERROR = 4000
    TRADING_BALANCE_ERROR = 4001
    TRADING_SYMBOL_ERROR = 4002
    TRADING_EXCHANGE_ERROR = 4003

    # Market data errors (5000-5999)
    MARKET_DATA_ERROR = 5000
    MARKET_DATA_FETCH_ERROR = 5001
    MARKET_DATA_PARSING_ERROR = 5002

    # AI/ML errors (6000-6999)
    AI_MODEL_ERROR = 6000
    AI_PREDICTION_ERROR = 6001
    AI_TRAINING_ERROR = 6002

    # Authentication/Authorization errors (7000-7999)
    AUTHENTICATION_ERROR = 7000
    AUTHORIZATION_ERROR = 7001
    TOKEN_ERROR = 7002

    # Rate limiting errors (8000-8999)
    RATE_LIMIT_ERROR = 8000
    RATE_LIMIT_EXCEEDED = 429  # Standard HTTP rate limit code


class MysticError(Exception):
    """Base exception class for Mystic Trading Platform"""

    def __init__(
        self,
        message: str,
        error_code: ErrorCode = ErrorCode.UNKNOWN_ERROR,
        details: dict[str, Any] | None = None,
        original_exception: Exception | None = None,
    ) -> None:
        self.message = message
        self.error_code = error_code
        self.details = details or {}
        self.original_exception = original_exception
        self.timestamp = datetime.now(timezone.utc)
        self._log_exception()
        super().__init__(self.message)

    def _log_exception(self) -> None:
        log_data = {
            "error_code": self.error_code.value,
            "error_type": self.error_code.name,
            "message": self.message,
            "details": self.details,
            "timestamp": self.timestamp.isoformat(),
        }
        if self.original_exception:
            log_data["original_exception"] = {
                "type": type(self.original_exception).__name__,
                "message": str(self.original_exception),
            }
        logger.error(f"MysticError: {log_data}")

    def to_dict(self) -> dict[str, Any]:
        return {
            "error": True,
            "error_code": self.error_code.value,
            "error_type": self.error_code.name,
            "message": self.message,
            "details": self.details,
            "timestamp": self.timestamp.isoformat(),
        }


# Alias for backward compatibility (production-critical)
# This ensures existing code using MysticException continues to work
MysticException = MysticError


class DatabaseError(MysticError):
    """Database-related exceptions"""

    def __init__(
        self,
        message: str,
        details: dict[str, Any] | None = None,
        original_exception: Exception | None = None,
    ) -> None:
        super().__init__(message, ErrorCode.DATABASE_QUERY_ERROR, details, original_exception)


class DatabaseConnectionError(MysticError):
    """Database connection-related exceptions"""

    def __init__(
        self,
        message: str,
        details: dict[str, Any] | None = None,
        original_exception: Exception | None = None,
    ) -> None:
        super().__init__(message, ErrorCode.DATABASE_CONNECTION_ERROR, details, original_exception)


class APIError(MysticError):
    """External API-related exceptions"""

    def __init__(
        self,
        message: str,
        details: dict[str, Any] | None = None,
        original_exception: Exception | None = None,
    ) -> None:
        super().__init__(message, ErrorCode.API_CONNECTION_ERROR, details, original_exception)


class TradingError(MysticError):
    """Trading-related exceptions"""

    def __init__(
        self,
        message: str,
        details: dict[str, Any] | None = None,
        original_exception: Exception | None = None,
    ) -> None:
        super().__init__(message, ErrorCode.TRADING_ORDER_ERROR, details, original_exception)


class MarketDataError(MysticError):
    """Market data-related exceptions"""

    def __init__(
        self,
        message: str,
        details: dict[str, Any] | None = None,
        original_exception: Exception | None = None,
    ) -> None:
        super().__init__(message, ErrorCode.MARKET_DATA_ERROR, details, original_exception)


class AIError(MysticError):
    """AI/ML-related exceptions"""

    def __init__(
        self,
        message: str,
        details: dict[str, Any] | None = None,
        original_exception: Exception | None = None,
    ) -> None:
        super().__init__(message, ErrorCode.AI_MODEL_ERROR, details, original_exception)


class AnalyticsError(MysticError):
    """Analytics-related exceptions"""

    def __init__(
        self,
        message: str,
        details: dict[str, Any] | None = None,
        original_exception: Exception | None = None,
    ) -> None:
        super().__init__(message, ErrorCode.AI_MODEL_ERROR, details, original_exception)


class MetricsError(MysticError):
    """Metrics-related exceptions"""

    def __init__(
        self,
        message: str,
        details: dict[str, Any] | None = None,
        original_exception: Exception | None = None,
    ) -> None:
        super().__init__(message, ErrorCode.AI_MODEL_ERROR, details, original_exception)


class AuthenticationError(MysticError):
    """Authentication-related exceptions"""

    def __init__(
        self,
        message: str,
        details: dict[str, Any] | None = None,
        original_exception: Exception | None = None,
    ) -> None:
        super().__init__(message, ErrorCode.AUTHENTICATION_ERROR, details, original_exception)


class RateLimitError(MysticError):
    """Rate limiting exceptions"""

    def __init__(
        self,
        message: str,
        details: dict[str, Any] | None = None,
        original_exception: Exception | None = None,
    ) -> None:
        super().__init__(message, ErrorCode.RATE_LIMIT_ERROR, details, original_exception)


class ModelError(MysticError):
    """Model-related exceptions"""

    def __init__(
        self,
        message: str,
        details: dict[str, Any] | None = None,
        original_exception: Exception | None = None,
    ) -> None:
        super().__init__(message, ErrorCode.AI_MODEL_ERROR, details, original_exception)


class NotificationError(MysticError):
    """Notification-related exceptions"""

    def __init__(
        self,
        message: str,
        details: dict[str, Any] | None = None,
        original_exception: Exception | None = None,
    ) -> None:
        super().__init__(message, ErrorCode.UNKNOWN_ERROR, details, original_exception)


class StrategyError(MysticError):
    """Strategy-related exceptions"""

    def __init__(
        self,
        message: str,
        details: dict[str, Any] | None = None,
        original_exception: Exception | None = None,
    ) -> None:
        super().__init__(message, ErrorCode.TRADING_ORDER_ERROR, details, original_exception)


# Backward compatibility aliases for renamed exceptions
# These must be defined after all exception classes are defined
DatabaseException = DatabaseError
DatabaseConnectionException = DatabaseConnectionError
APIException = APIError
TradingException = TradingError
MarketDataException = MarketDataError
AIException = AIError
AnalyticsException = AnalyticsError
MetricsException = MetricsError
AuthenticationException = AuthenticationError
RateLimitException = RateLimitError
ModelException = ModelError
NotificationException = NotificationError
StrategyException = StrategyError


def handle_exception(
    error_message: str,
    exception_class: type[MysticError] = MysticError,
    error_code: ErrorCode = ErrorCode.UNKNOWN_ERROR,
    reraise: bool = True,
    default_return: Any = None,
) -> Callable:
    """Decorator for standardized exception handling"""

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except MysticError:
                raise
            except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
                if exception_class == MysticError:
                    mystic_exception = exception_class(
                        message=error_message,
                        error_code=error_code,
                        details={"function": func.__name__},
                        original_exception=e,
                    )
                else:
                    mystic_exception = exception_class(
                        message=error_message,
                        details={"function": func.__name__},
                        original_exception=e,
                    )
                if reraise:
                    raise mystic_exception from e
                logger.exception(f"Exception in {func.__name__}: {mystic_exception}")
                return default_return

        return wrapper

    return decorator


def handle_async_exception(
    error_message: str,
    exception_class: type[MysticError] = MysticError,
    error_code: ErrorCode = ErrorCode.UNKNOWN_ERROR,
    reraise: bool = True,
    default_return: Any = None,
) -> Callable:
    """Decorator for standardized async exception handling"""

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args, **kwargs):
            try:
                return await func(*args, **kwargs)
            except MysticError:
                raise
            except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
                if exception_class == MysticError:
                    mystic_exception = exception_class(
                        message=error_message,
                        error_code=error_code,
                        details={"function": func.__name__},
                        original_exception=e,
                    )
                else:
                    mystic_exception = exception_class(
                        message=error_message,
                        details={"function": func.__name__},
                        original_exception=e,
                    )
                if reraise:
                    raise mystic_exception from e
                logger.exception(f"Exception in {func.__name__}: {mystic_exception}")
                return default_return

        return wrapper

    return decorator


def create_http_exception_handler():
    """Create standardized HTTP exception handler for FastAPI"""

    async def http_exception_handler(_request: Request, exc: Exception) -> JSONResponse:
        if isinstance(exc, MysticError):
            status_code = _get_status_code_for_error_code(exc.error_code)
            return JSONResponse(status_code=status_code, content=exc.to_dict())

        if isinstance(exc, HTTPException):
            return JSONResponse(
                status_code=exc.status_code,
                content={
                    "error": True,
                    "error_code": ErrorCode.UNKNOWN_ERROR.value,
                    "error_type": "HTTPException",
                    "message": exc.detail,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                },
            )

        logger.error(f"Unhandled exception: {exc}\n{traceback.format_exc()}")
        return JSONResponse(
            status_code=500,
            content={
                "error": True,
                "error_code": ErrorCode.UNKNOWN_ERROR.value,
                "error_type": "InternalServerError",
                "message": "An unexpected error occurred",
                "timestamp": datetime.now(timezone.utc).isoformat(),
            },
        )

    return http_exception_handler


def _get_status_code_for_error_code(error_code: ErrorCode) -> int:
    status_code_map = {
        # General errors
        ErrorCode.UNKNOWN_ERROR: 500,
        ErrorCode.VALIDATION_ERROR: 400,
        ErrorCode.CONFIGURATION_ERROR: 500,
        ErrorCode.TIMEOUT_ERROR: 408,
        # Database errors
        ErrorCode.DATABASE_CONNECTION_ERROR: 503,
        ErrorCode.DATABASE_QUERY_ERROR: 500,
        ErrorCode.DATABASE_TRANSACTION_ERROR: 500,
        # External API errors
        ErrorCode.API_CONNECTION_ERROR: 503,
        ErrorCode.API_RATE_LIMIT_ERROR: 429,
        ErrorCode.API_AUTHENTICATION_ERROR: 401,
        ErrorCode.API_TIMEOUT_ERROR: 408,
        ErrorCode.API_RESPONSE_ERROR: 502,
        # Trading errors
        ErrorCode.TRADING_ORDER_ERROR: 500,
        ErrorCode.TRADING_BALANCE_ERROR: 400,
        ErrorCode.TRADING_SYMBOL_ERROR: 400,
        ErrorCode.TRADING_EXCHANGE_ERROR: 503,
        # Market data errors
        ErrorCode.MARKET_DATA_ERROR: 503,
        ErrorCode.MARKET_DATA_FETCH_ERROR: 503,
        ErrorCode.MARKET_DATA_PARSING_ERROR: 500,
        # AI/ML errors
        ErrorCode.AI_MODEL_ERROR: 500,
        ErrorCode.AI_PREDICTION_ERROR: 500,
        ErrorCode.AI_TRAINING_ERROR: 500,
        # Authentication/Authorization errors
        ErrorCode.AUTHENTICATION_ERROR: 401,
        ErrorCode.AUTHORIZATION_ERROR: 403,
        ErrorCode.TOKEN_ERROR: 401,
        # Rate limiting errors
        ErrorCode.RATE_LIMIT_ERROR: 429,
        ErrorCode.RATE_LIMIT_EXCEEDED: 429,
    }
    return status_code_map.get(error_code, 500)


def safe_execute(func: Callable, *args, **kwargs) -> Any | MysticError:
    """Safely execute a function and return result or exception"""
    try:
        return func(*args, **kwargs)
    except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
        return MysticError(
            message=f"Error executing {func.__name__}",
            error_code=ErrorCode.UNKNOWN_ERROR,
            details={"function": func.__name__},
            original_exception=e,
        )


async def safe_async_execute(func: Callable, *args, **kwargs) -> Any | MysticError:
    """Safely execute an async function and return result or exception"""
    try:
        return await func(*args, **kwargs)
    except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
        return MysticError(
            message=f"Error executing {func.__name__}",
            error_code=ErrorCode.UNKNOWN_ERROR,
            details={"function": func.__name__},
            original_exception=e,
        )
