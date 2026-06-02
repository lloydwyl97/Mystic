"""
Logging Configuration - All Live Data, No Fallback/Hardcoded Data

This module provides logging configuration for the backend API (port 8000).
All logging:
- Logs live operations from backend endpoints (port 8000)
- Captures live API requests and responses
- No fallback/hardcoded log data - all logs from live operations
- Configures log levels and handlers for live backend services

Live Data Sources:
- Live API requests/responses from backend (port 8000)
- Live application events and errors
- Live database operations
- All logs are from live operations - no mock/test data

Endpoint References:
- Backend API: Port 8000 (logs all live API requests)
- Logs collected via Promtail and forwarded to Loki for monitoring
"""

import logging
import sys
from typing import Any

# HTTP status code constants for log_request_response
HTTP_CLIENT_ERROR_MIN = 400
HTTP_CLIENT_ERROR_MAX = 499
HTTP_SERVER_ERROR_MIN = 500


def setup_logging(log_level: str = "INFO") -> None:
    """
    Configure application logging for live backend operations (port 8000).

    Sets up logging configuration for:
    - Live API requests/responses
    - Live application events
    - Live database operations
    - All logs from live operations - no fallback/hardcoded data

    Args:
        log_level: Logging level (default: "INFO") - from live configuration
    """
    log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    log_level_num = getattr(logging, log_level.upper(), logging.INFO)

    # Configure root logger for live backend operations (port 8000)
    # Guard to avoid duplicate handlers when multiple entry points load
    if not logging.getLogger().handlers:
        logging.basicConfig(level=log_level_num, format=log_format, handlers=[logging.StreamHandler(sys.stdout)])

    # Set specific loggers to appropriate levels (for live backend on port 8000)
    logging.getLogger("uvicorn").setLevel(logging.INFO)
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
    logging.getLogger("uvicorn.error").setLevel(logging.INFO)

    # Suppress noisy third-party loggers (keep focus on live backend operations)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("asyncio").setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger with the specified name for live operations.

    Args:
        name: Logger name (typically module name)

    Returns:
        Logger instance for live backend operations (port 8000)
    """
    return logging.getLogger(name)


def log_request_response(
    logger: logging.Logger,
    request_method: str,
    request_url: str,
    status_code: int,
    processing_time: float,
    extra: dict[str, Any] | None = None,
) -> None:
    """
    Log live API request and response details from backend (port 8000).

    Logs live API requests to backend endpoints:
    - Request method and URL (live endpoint on port 8000)
    - Response status code from live API call
    - Processing time for live operation
    - Additional metadata from live request/response

    All logged data is from live operations - no fallback/hardcoded data.

    Args:
        logger: Logger instance
        request_method: HTTP method (GET, POST, etc.) from live request
        request_url: Request URL (live endpoint on port 8000)
        status_code: HTTP status code from live response
        processing_time: Processing time in seconds for live operation
        extra: Additional metadata from live request/response (optional)
    """
    log_data = {
        "method": request_method,
        "url": request_url,
        "status_code": status_code,
        "processing_time_ms": round(processing_time * 1000, 2),
    }

    if extra:
        log_data.update(extra)

    log_message = f"{request_method} {request_url} - {status_code} - {log_data['processing_time_ms']}ms"

    # Log based on status code from live response (client errors: 4xx, server errors: 5xx)
    if HTTP_CLIENT_ERROR_MIN <= status_code <= HTTP_CLIENT_ERROR_MAX:
        logger.warning(log_message, extra=log_data)
    elif status_code >= HTTP_SERVER_ERROR_MIN:
        logger.error(log_message, extra=log_data)
    else:
        logger.info(log_message, extra=log_data)
