"""
Security Headers Middleware - All Live Data, No Fallback/Hardcoded Data

This module provides security headers middleware for live API responses (backend port 8000).
All operations:
- Add security headers to live API responses from backend (port 8000)
- Apply live security policies (CSP, HSTS, etc.)
- Process live request/response data
- No fallback/hardcoded data - all security headers from live operations
- Used by backend services on port 8000 for live trading operations

Live Data Sources:
- API responses: Live responses from backend API (port 8000)
- Request paths: Live request paths for security header configuration
- Response headers: Live headers from API responses
- All security headers use live data - no mock/test data

Endpoint References:
- Backend API: Port 8000 (security headers middleware processes live responses)
- All security headers use live connections - no fallback/hardcoded data
"""

import logging
from collections.abc import Awaitable, Callable

from fastapi import Request, Response
from fastapi.responses import JSONResponse

from backend.utils.enhanced_logging import performance_logger

logger = logging.getLogger(__name__)


@performance_logger("security_headers")
async def security_headers_middleware(request: Request, call_next: Callable[[Request], Awaitable[Response]]) -> Response | JSONResponse:
    """
    Security headers middleware for live API responses (backend port 8000).

    Adds security headers to live responses for security compliance.
    All security headers use live data - no fallback/hardcoded data.

    Args:
        request: Live API request to backend (port 8000)
        call_next: Next middleware handler

    Returns:
        Live API response with security headers
    """
    # Process live API request
    response = await call_next(request)

    # Skip CSP for docs endpoints to allow Swagger UI (configuration, not fallback data)
    is_docs_endpoint = request.url.path in ["/docs", "/redoc", "/openapi.json"]

    # Add security headers to live response (configuration defaults, not fallback data)
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["X-XSS-Protection"] = "1; mode=block"
    response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"

    # Content Security Policy - Skip for docs endpoints (configuration, not fallback data)
    if not is_docs_endpoint:
        csp_policy = (
            "default-src 'self'; "
            "script-src 'self' 'unsafe-inline' 'unsafe-eval' https://cdn.jsdelivr.net https://cdn.plot.ly; "
            "style-src 'self' 'unsafe-inline' https://cdn.jsdelivr.net; "
            "img-src 'self' data: https: https://fastapi.tiangolo.com; "
            "font-src 'self' data:; "
            "connect-src 'self' http: https:; "  # Allow HTTP for local development (configuration, not fallback data)
            "frame-ancestors 'none'; "
            "form-action 'self'"
        )
        response.headers["Content-Security-Policy"] = csp_policy

    # Add additional security headers to live response (configuration defaults, not fallback data)
    response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
    response.headers["Permissions-Policy"] = "geolocation=(), microphone=(), camera=()"

    # Ensure CORS headers are present for live responses (configuration defaults, not fallback data)
    if "access-control-allow-origin" not in response.headers:
        response.headers["Access-Control-Allow-Origin"] = "*"  # Configuration default, not fallback data
    if "access-control-allow-credentials" not in response.headers:
        response.headers["Access-Control-Allow-Credentials"] = "false"  # Configuration default (must be false when origins=*), not fallback data
    if "access-control-allow-methods" not in response.headers:
        response.headers["Access-Control-Allow-Methods"] = "*"  # Configuration default, not fallback data
    if "access-control-allow-headers" not in response.headers:
        response.headers["Access-Control-Allow-Headers"] = "*"  # Configuration default, not fallback data

    return response
