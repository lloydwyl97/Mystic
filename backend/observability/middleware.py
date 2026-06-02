from __future__ import annotations

import logging
import time
from collections.abc import Callable
from typing import Any

from fastapi import Request, Response

# Direct imports for production
from prometheus_client import Counter, Histogram
from starlette.middleware.base import BaseHTTPMiddleware

# Try to use enhanced logging if available
try:
    from backend.utils.enhanced_logging import log_operation_performance
except ImportError:
    log_operation_performance = None

logger = logging.getLogger(__name__)

# HTTP metrics
HTTP_REQUESTS_TOTAL = Counter(
    "http_requests_total",
    "Total HTTP requests",
    ["method", "route", "status_code"],
)

HTTP_REQUEST_DURATION_SECONDS = Histogram(
    "http_request_duration_seconds",
    "HTTP request duration in seconds",
    ["method", "route"],
    buckets=(0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2, 5, 10),
)

HTTP_EXCEPTIONS_TOTAL = Counter(
    "http_exceptions_total",
    "Total HTTP exceptions",
    ["method", "route", "exception_type"],
)


class ObservabilityMiddleware(BaseHTTPMiddleware):
    """Middleware for collecting HTTP observability metrics."""

    def __init__(self, app: Any, enable_logging: bool = True) -> None:
        super().__init__(app)
        self.enable_logging = enable_logging

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Process request and collect metrics."""
        start_time = time.perf_counter()

        # Extract route template for low-cardinality metrics
        route_template = self._get_route_template(request)
        method = request.method

        try:
            response = await call_next(request)
            status_code = str(response.status_code)

            # Record successful request metrics
            HTTP_REQUESTS_TOTAL.labels(method=method, route=route_template, status_code=status_code).inc()

            # Record request duration
            duration = time.perf_counter() - start_time
            HTTP_REQUEST_DURATION_SECONDS.labels(method=method, route=route_template).observe(duration)

            # Log structured request info if enabled
            if self.enable_logging:
                self._log_request(request, response, duration)
        except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
            # Record exception metrics
            exception_type = type(e).__name__
            HTTP_EXCEPTIONS_TOTAL.labels(method=method, route=route_template, exception_type=exception_type).inc()

            # Log exception
            logger.exception("HTTP exception in %s %s", method, route_template)

            # Re-raise the exception
            raise
        else:
            return response

    def _get_route_template(self, request: Request) -> str:
        """Extract route template for low-cardinality metrics."""
        # Use the route path template if available, otherwise use the raw path
        if hasattr(request, "route") and request.route:
            return request.route.path
        return request.url.path

    def _log_request(self, request: Request, response: Response, duration: float) -> None:
        """Log structured request information."""
        try:
            # Try to use enhanced logging if available
            if log_operation_performance is not None:
                log_operation_performance("http_request")(
                    lambda: {
                        "method": request.method,
                        "path": request.url.path,
                        "status_code": response.status_code,
                        "duration_seconds": duration,
                        "user_agent": request.headers.get("user-agent", ""),
                    },
                )()
            else:
                # Fallback to standard logging
                logger.info("%s %s -> %s (%.3fs)", request.method, request.url.path, response.status_code, duration)
        except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
            logger.warning("Failed to log request: %s", e)
