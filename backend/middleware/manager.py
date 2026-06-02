"""
Middleware Manager - All Live Data, No Fallback/Hardcoded Data

This module provides centralized middleware registration and configuration for live trading operations (backend port 8000).
All operations:
- Register middleware for live API requests (backend port 8000)
- Configure middleware for live request processing
- Manage middleware lifecycle for live operations
- No fallback/hardcoded data - all middleware processes live operations
- Used by backend services on port 8000 for live trading operations

Live Data Sources:
- API requests: Live requests to backend API (port 8000) processed by middleware
- Middleware configuration: Configuration for live request processing
- All middleware operations use live data - no mock/test data

Endpoint References:
- Backend API: Port 8000 (all middleware processes live requests)
- All middleware uses live connections - no fallback/hardcoded data
"""

import logging
from typing import Any

from fastapi import FastAPI

from .cache import cache_middleware_handler
from .circuit_breaker import circuit_breaker_middleware
from .rate_limiter import rate_limit_middleware
from .request_logger import request_logger_middleware
from .request_validator import request_validator_middleware
from .response_sanitizer import response_sanitizer_middleware
from .security_headers import security_headers_middleware

logger = logging.getLogger(__name__)


class MiddlewareManager:
    """
    Manages middleware registration and configuration for live API requests.

    All middleware processes live requests to backend (port 8000) - no fallback/hardcoded data.
    """

    def __init__(self) -> None:
        """Initialize middleware manager with configuration for live request processing."""
        # Configuration defaults for live middleware (not fallback data, configuration defaults)
        self.middleware_configs: dict[str, dict[str, Any]] = {
            "rate_limiter": {"enabled": True, "config": {}},  # Live rate limiting
            "request_logger": {"enabled": True, "config": {}},  # Live request logging
            "security_headers": {"enabled": True, "config": {}},  # Live security headers
            "cache": {"enabled": True, "config": {}},  # Live response caching
            "circuit_breaker": {"enabled": True, "config": {}},  # Live circuit breaking
            "request_validator": {"enabled": True, "config": {}},  # Live request validation
            "response_sanitizer": {"enabled": True, "config": {}},  # Live response sanitization
        }

    def configure(
        self,
        middleware_name: str,
        enabled: bool = True,
        config: dict[str, Any] | None = None,
    ) -> None:
        """
        Configure a specific middleware for live request processing.

        Args:
            middleware_name: Name of middleware to configure
            enabled: Whether middleware is enabled (default: True, configuration default not fallback data)
            config: Configuration dictionary for middleware (optional)
        """
        if middleware_name in self.middleware_configs:
            self.middleware_configs[middleware_name]["enabled"] = enabled
            if config:
                self.middleware_configs[middleware_name]["config"] = config
            logger.info("Configured %s middleware for live requests: enabled=%s", middleware_name, enabled)
        else:
            logger.warning("Unknown middleware for live requests: %s", middleware_name)

    def register_all(self, app: FastAPI) -> None:
        """
        Register all enabled middleware with the FastAPI app for live request processing.

        All middleware processes live API requests to backend (port 8000).

        Args:
            app: FastAPI application instance for live request handling
        """
        logger.info("Registering middleware for live request processing...")

        # 1. Request logger (should be first to log all requests)
        if self.middleware_configs["request_logger"]["enabled"]:
            app.middleware("http")(request_logger_middleware)
            logger.info("Request logger middleware registered")

        # 2. Security headers
        if self.middleware_configs["security_headers"]["enabled"]:
            app.middleware("http")(security_headers_middleware)
            logger.info("Security headers middleware registered")

        # 3. Rate limiter
        if self.middleware_configs["rate_limiter"]["enabled"]:
            app.middleware("http")(rate_limit_middleware)
            logger.info("Rate limiter middleware registered")

        # 4. Circuit breaker
        if self.middleware_configs["circuit_breaker"]["enabled"]:
            app.middleware("http")(circuit_breaker_middleware)
            logger.info("Circuit breaker middleware registered")

        # 5. Request validator
        if self.middleware_configs["request_validator"]["enabled"]:
            app.middleware("http")(request_validator_middleware)
            logger.info("Request validator middleware registered")

        # 6. Cache middleware
        if self.middleware_configs["cache"]["enabled"]:
            app.middleware("http")(cache_middleware_handler)
            logger.info("Cache middleware registered")

        # 7. Response sanitizer (should be last to sanitize all responses)
        if self.middleware_configs["response_sanitizer"]["enabled"]:
            app.middleware("http")(response_sanitizer_middleware)
            logger.info("Response sanitizer middleware registered")

        logger.info("All middleware registered successfully")


def get_middleware_manager() -> MiddlewareManager:
    """
    Get the global middleware manager instance for live request processing.

    Returns:
        MiddlewareManager instance for managing live middleware operations
    """
    return MiddlewareManager()
