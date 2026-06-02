#!/usr/bin/env python3
"""
Enhanced Logging Configuration
Provides comprehensive logging setup for better debugging and monitoring
"""

import asyncio
import logging
import logging.handlers
import os
import sys
import time
from functools import wraps
from pathlib import Path
from typing import ClassVar


class ColoredFormatter(logging.Formatter):
    """Colored formatter for console output"""

    # ANSI color codes
    COLORS: ClassVar[dict[str, str]] = {
        "DEBUG": "\033[36m",  # Cyan
        "INFO": "\033[32m",  # Green
        "WARNING": "\033[33m",  # Yellow
        "ERROR": "\033[31m",  # Red
        "CRITICAL": "\033[35m",  # Magenta
        "RESET": "\033[0m",  # Reset
    }

    def format(self, record: logging.LogRecord) -> str:
        if record.levelname in self.COLORS:
            record.levelname = f"{self.COLORS[record.levelname]}{record.levelname}{self.COLORS['RESET']}"

        return super().format(record)


class EnhancedLogger:
    """Enhanced logger with structured logging and multiple outputs"""

    def __init__(self, name: str, log_level: str = "INFO", log_dir: str = "logs") -> None:
        self.name = name
        self.log_level = getattr(logging, log_level.upper())
        self.log_dir = Path(log_dir)

        # Safely create log directory - non-blocking, won't prevent startup
        try:
            self.log_dir.mkdir(exist_ok=True)
        except (OSError, PermissionError, ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
            # Log directory creation failed - continue with console logging only
            # This prevents startup blocking if disk is full or permissions are wrong
            logger.info(f"WARNING: Could not create log directory '{log_dir}': {e}. Continuing with console logging only.")
            self.log_dir = None  # Disable file logging

        # Create logger
        self.logger = logging.getLogger(name)
        self.logger.setLevel(self.log_level)

        # Prevent duplicate handlers
        if self.logger.handlers:
            return

        self._setup_handlers()

    def _setup_handlers(self) -> None:
        """Setup console and file handlers"""

        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(self.log_level)

        console_format = ColoredFormatter(
            "%(asctime)s | %(levelname)-8s | %(name)-20s | %(message)s",
            datefmt="%H:%M:%S",
        )
        console_handler.setFormatter(console_format)

        # Optional file handler (gated by env; default disabled)
        file_handler = None
        if os.environ.get("MYSTIC_FILE_LOGGING", "0").strip() == "1" and self.log_dir is not None:
            try:
                log_file = self.log_dir / f"{self.name}.log"
                file_handler = (
                    logging.FileHandler(log_file, encoding="utf-8")
                    if os.name == "nt"
                    else logging.handlers.RotatingFileHandler(
                        log_file,
                        maxBytes=1 * 1024 * 1024,  # 1MB (reduced from 10MB)
                        backupCount=2,  # Reduced from 5
                    )
                )
                file_handler.setLevel(self.log_level)
            except (OSError, PermissionError, ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
                # File handler creation failed - continue with console logging only
                # This prevents startup blocking if disk is full, file is locked, or permissions are wrong
                logger.info(f"WARNING: Could not create file handler for '{self.name}': {e}. Continuing with console logging only.")
                file_handler = None  # Fallback to console only

        file_format = logging.Formatter(
            "%(asctime)s | %(levelname)-8s | %(name)-20s | %(funcName)-15s:%(lineno)-4d | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        # Only set formatter if a file handler was actually created
        if file_handler is not None:
            file_handler.setFormatter(file_format)

        # Optional error file handler (gated by env; default disabled)
        error_handler = None
        if os.environ.get("MYSTIC_FILE_LOGGING", "0").strip() == "1" and self.log_dir is not None:
            try:
                error_log_file = self.log_dir / f"{self.name}_errors.log"
                if os.name == "nt":
                    error_handler = logging.FileHandler(error_log_file, encoding="utf-8")
                else:
                    error_handler = logging.handlers.RotatingFileHandler(
                        error_log_file,
                        maxBytes=1 * 1024 * 1024,  # 1MB (reduced from 5MB)
                        backupCount=2,  # Reduced from 3
                    )
                error_handler.setLevel(logging.ERROR)
                error_handler.setFormatter(file_format)
            except (OSError, PermissionError, ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
                # Error file handler creation failed - continue without it
                # This prevents startup blocking if disk is full, file is locked, or permissions are wrong
                logger.info(f"WARNING: Could not create error file handler for '{self.name}': {e}. Continuing without error file logging.")
                error_handler = None  # Fallback to console only

        # Add handlers
        self.logger.addHandler(console_handler)
        if file_handler is not None:
            self.logger.addHandler(file_handler)
        if error_handler is not None:
            self.logger.addHandler(error_handler)

    def get_logger(self) -> logging.Logger:
        """Get the configured logger"""
        return self.logger


def setup_service_logging(service_name: str, log_level: str = "INFO") -> logging.Logger:
    """Setup enhanced logging for a service"""
    enhanced_logger = EnhancedLogger(service_name, log_level)
    return enhanced_logger.get_logger()


def log_function_call(logger: logging.Logger, func_name: str, **kwargs):
    """Log function call with parameters"""
    params = ", ".join([f"{k}={v}" for k, v in kwargs.items()])
    logger.debug(f"CALL: {func_name}({params})")


def log_function_result(logger: logging.Logger, func_name: str, result, duration: float | None = None):
    """Log function result with timing"""
    if duration is not None:
        logger.debug(f"RESULT: {func_name} -> {result} (took {duration:.3f}s)")
    else:
        logger.debug(f"RESULT: {func_name} -> {result}")


def log_error_with_context(logger: logging.Logger, error: Exception, context: dict | None = None):
    """Log error with additional context"""
    context_str = ""
    if context:
        context_str = f" | Context: {context}"

    logger.error(f"ERROR: {type(error).__name__}: {error}{context_str}", exc_info=True)


def log_performance_metrics(logger: logging.Logger, operation: str, metrics: dict):
    """Log performance metrics"""
    metrics_str = ", ".join([f"{k}={v}" for k, v in metrics.items()])
    logger.info(f"PERFORMANCE: {operation} | {metrics_str}")


def performance_logger(operation_name: str):
    """
    Decorator for logging operation performance.
    Measures execution time and logs it.

    Usage:
        @performance_logger("my_operation")
        async def my_function(request: Request, call_next):
            ...
    """

    def decorator(func):
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            start_time = time.time()
            try:
                result = await func(*args, **kwargs)
                duration = time.time() - start_time
            except Exception as e:
                duration = time.time() - start_time
                logger.exception(f"PERFORMANCE: {operation_name} failed after {duration:.3f}s - {e}")
                raise
            else:
                logger.debug(f"PERFORMANCE: {operation_name} completed in {duration:.3f}s")
                return result

        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                duration = time.time() - start_time
            except Exception as e:
                duration = time.time() - start_time
                logger.exception(f"PERFORMANCE: {operation_name} failed after {duration:.3f}s - {e}")
                raise
            else:
                logger.debug(f"PERFORMANCE: {operation_name} completed in {duration:.3f}s")
                return result

        # Return appropriate wrapper based on function type
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        return sync_wrapper

    return decorator


def log_api_call(
    logger: logging.Logger,
    method: str,
    url: str,
    status_code: int | None = None,
    response_time: float | None = None,
    error: str | None = None,
):
    """Log API call details"""
    if error:
        logger.error(f"API_ERROR: {method} {url} | Error: {error}")
    else:
        logger.info(f"API_CALL: {method} {url} | Status: {status_code} | Time: {response_time:.3f}s")


def log_websocket_event(logger: logging.Logger, event: str, details: dict | None = None):
    """Log WebSocket events"""
    details_str = ""
    if details:
        details_str = f" | Details: {details}"

    logger.info(f"WEBSOCKET: {event}{details_str}")


def log_cache_operation(
    logger: logging.Logger,
    operation: str,
    key: str,
    hit: bool | None = None,
    ttl: int | None = None,
    size: int | None = None,
):
    """Log cache operations"""
    hit_str = f" | Hit: {hit}" if hit is not None else ""
    ttl_str = f" | TTL: {ttl}s" if ttl is not None else ""
    size_str = f" | Size: {size}" if size is not None else ""

    logger.debug(f"CACHE: {operation} | Key: {key}{hit_str}{ttl_str}{size_str}")


def log_ai_model_operation(
    logger: logging.Logger,
    operation: str,
    model_name: str,
    symbol: str | None = None,
    accuracy: float | None = None,
    duration: float | None = None,
):
    """Log AI model operations"""
    symbol_str = f" | Symbol: {symbol}" if symbol else ""
    accuracy_str = f" | Accuracy: {accuracy:.3f}" if accuracy is not None else ""
    duration_str = f" | Duration: {duration:.3f}s" if duration is not None else ""

    logger.info(f"AI_MODEL: {operation} | Model: {model_name}{symbol_str}{accuracy_str}{duration_str}")


def log_connection_pool_status(logger: logging.Logger, pool_name: str, metrics: dict):
    """Log connection pool status"""
    metrics_str = ", ".join([f"{k}={v}" for k, v in metrics.items()])
    logger.info(f"CONNECTION_POOL: {pool_name} | {metrics_str}")


def log_system_health(logger: logging.Logger, component: str, status: str, details: dict | None = None):
    """Log system health status"""
    details_str = ""
    if details:
        details_str = f" | Details: {details}"

    logger.info(f"HEALTH: {component} | Status: {status}{details_str}")


# Global logger instances for common services
def get_service_logger(service_name: str) -> logging.Logger:
    """Get a logger for a specific service"""
    return setup_service_logging(service_name)


# Common service loggers
backend_logger = get_service_logger("backend")
ai_logger = get_service_logger("ai_service")
websocket_logger = get_service_logger("websocket")
cache_logger = get_service_logger("cache")
api_logger = get_service_logger("api")
database_logger = get_service_logger("database")

# Default logger for this module
logger = logging.getLogger(__name__)

if __name__ == "__main__":
    # Test the logging system
    test_logger = get_service_logger("test")

    test_logger.debug("This is a debug message")
    test_logger.info("This is an info message")
    test_logger.warning("This is a warning message")
    test_logger.error("This is an error message")

    # Test structured logging
    log_function_call(test_logger, "test_function", param1="value1", param2=42)
    log_function_result(test_logger, "test_function", "success", 0.123)
    log_api_call(test_logger, "GET", "https://api.example.com/data", 200, 0.456)
    log_websocket_event(test_logger, "connected", {"url": "wss://example.com"})
    log_cache_operation(test_logger, "get", "user:123", hit=True, ttl=300)
    log_ai_model_operation(test_logger, "predict", "BTCUSDT_classifier", "BTCUSDT", 0.89, 0.234)
    log_connection_pool_status(test_logger, "http_client", {"active": 5, "idle": 10})
    log_system_health(test_logger, "database", "healthy", {"connections": 3})

    logger.info("Enhanced logging test completed. Check the logs/ directory for output files.")
