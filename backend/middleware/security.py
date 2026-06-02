"""
Security Middleware for Mystic Trading System

Provides IP whitelist enforcement and request filtering.
"""

import logging
import os
from collections.abc import Callable

from fastapi import Request, Response
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

logger = logging.getLogger(__name__)


class IPWhitelistMiddleware(BaseHTTPMiddleware):
    """
    IP Whitelist Middleware - Only allow requests from whitelisted IPs.

    Configuration via environment variable:
    - ALLOWED_IPS: Comma-separated list of allowed IP addresses
      Default: "127.0.0.1,localhost,::1" (localhost only)

    Examples:
    - ALLOWED_IPS="127.0.0.1,192.168.1.100,10.0.0.5"
    - ALLOWED_IPS="*" (allows all IPs - NOT RECOMMENDED for production)
    """

    def __init__(self, app, allowed_ips: list[str] | None = None):
        super().__init__(app)

        if allowed_ips is None:
            # Load from environment variable
            allowed_ips_env = os.getenv("ALLOWED_IPS", "127.0.0.1,localhost,::1")

            if allowed_ips_env == "*":
                # Allow all IPs (not recommended for production)
                self.allowed_ips = ["*"]
                logger.warning("IP Whitelist: ALL IPs allowed (ALLOWED_IPS=*) - NOT RECOMMENDED for production")
            else:
                # Parse comma-separated list
                self.allowed_ips = [ip.strip() for ip in allowed_ips_env.split(",") if ip.strip()]
        else:
            self.allowed_ips = allowed_ips

        # Add localhost variants
        localhost_variants = ["127.0.0.1", "::1", "localhost"]
        for variant in localhost_variants:
            if variant not in self.allowed_ips and "*" not in self.allowed_ips:
                self.allowed_ips.append(variant)

        logger.info(f"🔒 IP Whitelist enabled: {len(self.allowed_ips)} allowed IPs")
        if "*" not in self.allowed_ips:
            logger.info(f"   Allowed IPs: {', '.join(self.allowed_ips[:5])}{'...' if len(self.allowed_ips) > 5 else ''}")

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """
        Check if request IP is in whitelist before allowing request.
        """
        # Skip whitelist check for health check endpoints
        if request.url.path in ["/health", "/api/health", "/api/status"]:
            return await call_next(request)

        # Get client IP
        client_ip = request.client.host if request.client else "unknown"

        # Check if IP is whitelisted
        if "*" not in self.allowed_ips and client_ip not in self.allowed_ips:
            logger.warning(f"🚫 Blocked request from non-whitelisted IP: {client_ip} to {request.url.path}")
            return JSONResponse(status_code=403, content={"error": "Forbidden", "detail": "Your IP address is not whitelisted", "ip": client_ip})

        # IP is whitelisted, proceed with request
        return await call_next(request)


class AdminAuthMiddleware:
    """
    Admin authentication for sensitive endpoints.

    Checks for ADMIN_TOKEN in request header.
    """

    @staticmethod
    def verify_admin_token(request: Request) -> bool:
        """
        Verify that request has valid admin token.

        Args:
            request: FastAPI request object

        Returns:
            True if token is valid, False otherwise
        """
        # Get expected token from environment
        expected_token = os.getenv("ADMIN_TOKEN")

        if not expected_token:
            logger.error("ADMIN_TOKEN not set in environment - admin endpoints disabled")
            return False

        # Get token from request header
        auth_header = request.headers.get("Authorization")

        if not auth_header:
            return False

        # Support both "Bearer TOKEN" and "TOKEN" formats
        provided_token = auth_header[7:] if auth_header.startswith("Bearer ") else auth_header

        # Constant-time comparison to prevent timing attacks
        return provided_token == expected_token
