"""
Middleware Package - All Live Data, No Fallback/Hardcoded Data

This package contains all middleware components for live trading operations (backend port 8000).
All middleware:
- Processes live API requests on backend (port 8000)
- Handles live rate limiting, caching, circuit breaking, and security
- Logs live request/response data
- No fallback/hardcoded data - all middleware processes live operations
- Used by backend services on port 8000 for live trading operations

Live Data Sources:
- API requests: Live requests to backend API (port 8000)
- Rate limiting: Live request rate tracking and limiting
- Caching: Live data caching from backend API responses
- Circuit breaking: Live service health monitoring
- Security: Live request validation and response sanitization
- Metrics: Live operation metrics collection
- All middleware processes live data - no mock/test data

Endpoint References:
- Backend API: Port 8000 (all middleware processes live requests)
- All middleware uses live connections - no fallback/hardcoded data
"""

from .cache import cache_middleware_handler
from .circuit_breaker import circuit_breaker_middleware
from .manager import get_middleware_manager
from .rate_limiter import rate_limit_middleware
from .request_logger import request_logger_middleware
from .request_validator import request_validator_middleware
from .response_sanitizer import response_sanitizer_middleware
from .security_headers import security_headers_middleware

__all__ = [
    "cache_middleware_handler",
    "circuit_breaker_middleware",
    "get_middleware_manager",
    "rate_limit_middleware",
    "request_logger_middleware",
    "request_validator_middleware",
    "response_sanitizer_middleware",
    "security_headers_middleware",
]
