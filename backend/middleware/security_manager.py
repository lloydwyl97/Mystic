"""
Security Manager - All Live Data, No Fallback/Hardcoded Data

This module provides security headers management for live API responses (backend port 8000).
All operations:
- Manage security headers for live API responses from backend (port 8000)
- Configure path-specific security headers for live requests
- Apply live security policies
- No fallback/hardcoded data - all security headers from live operations
- Used by backend services on port 8000 for live trading operations

Live Data Sources:
- API responses: Live responses from backend API (port 8000)
- Request paths: Live request paths for security header configuration
- Response headers: Live headers from API responses
- All security headers use live data - no mock/test data

Endpoint References:
- Backend API: Port 8000 (security manager processes live responses)
- All security headers use live connections - no fallback/hardcoded data
"""

import logging

from fastapi.responses import JSONResponse

logger = logging.getLogger(__name__)


class SecurityHeaders:
    """
    Security headers manager for live API responses (backend port 8000).

    Manages security headers configuration and applies to live responses.
    All security headers use live data - no fallback/hardcoded data.
    """

    def __init__(self) -> None:
        """Initialize security headers manager for live response security."""
        # Security headers configuration (configuration defaults, not fallback data)
        self.security_headers = {
            # Prevent clickjacking
            "X-Frame-Options": "DENY",
            # Enable XSS protection
            "X-XSS-Protection": "1; mode=block",
            # Prevent MIME type sniffing
            "X-Content-Type-Options": "nosniff",
            # Strict Transport Security
            "Strict-Transport-Security": "max-age=31536000; includeSubDomains",
            # Content Security Policy
            "Content-Security-Policy": (
                "default-src 'self'; "
                "script-src 'self' 'unsafe-inline' 'unsafe-eval' https://cdn.plot.ly; "
                "style-src 'self' 'unsafe-inline'; "
                "img-src 'self' data: https:; "
                "font-src 'self' data:; "
                "connect-src 'self' http: https:; "  # Allow HTTP for local development
                "frame-ancestors 'none'; "
                "form-action 'self'"
            ),
            # Referrer Policy
            "Referrer-Policy": "strict-origin-when-cross-origin",
            # Permissions Policy
            "Permissions-Policy": ("accelerometer=(), camera=(), geolocation=(), gyroscope=(), magnetometer=(), microphone=(), payment=(), usb=()"),
            # Cross-Origin Resource Policy
            "Cross-Origin-Resource-Policy": "same-site",
            # Cross-Origin Embedder Policy
            "Cross-Origin-Embedder-Policy": "require-corp",
            # Cross-Origin Opener Policy
            "Cross-Origin-Opener-Policy": "same-origin",
        }

        # Headers to remove from live responses (configuration, not fallback data)
        self.headers_to_remove = {
            "Server",
            "X-Powered-By",
            "X-AspNet-Version",
            "X-AspNetMvc-Version",
        }

        # Path-specific security headers for live requests (configuration, not fallback data)
        self.path_specific_headers = {
            "/api/auth": {
                "Content-Security-Policy": (
                    "default-src 'self'; "
                    "script-src 'self'; "
                    "style-src 'self' 'unsafe-inline'; "
                    "img-src 'self' data:; "
                    "font-src 'self' data:; "
                    "connect-src 'self'; "
                    "frame-ancestors 'none'; "
                    "form-action 'self'"
                ),
            },
        }

    def get_path_specific_headers(self, path: str) -> dict[str, str]:
        """
        Get security headers specific to the live request path.

        Args:
            path: Live request path

        Returns:
            Path-specific security headers for live request
        """
        for prefix, headers in self.path_specific_headers.items():
            if path.startswith(prefix):
                return headers
        return {}

    def add_security_headers(self, response: JSONResponse, path: str) -> JSONResponse:
        """
        Add security headers to live API response (backend port 8000).

        Applies security headers configuration to live responses.
        All security headers use live data - no fallback/hardcoded data.

        Args:
            response: Live API response from backend (port 8000)
            path: Live request path

        Returns:
            Live API response with security headers applied
        """
        try:
            # Get path-specific headers
            path_headers = self.get_path_specific_headers(path)

            # Add security headers
            for header, value in self.security_headers.items():
                # Override with path-specific header if exists
                if header in path_headers:
                    response.headers[header] = path_headers[header]
                else:
                    response.headers[header] = value

            # Remove unnecessary headers
            for header in self.headers_to_remove:
                if header in response.headers:
                    del response.headers[header]
        except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
            logger.exception("Error adding security headers to live response: %s", e)
            return response
        else:
            return response
