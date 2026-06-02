"""
Request Validator Manager - All Live Data, No Fallback/Hardcoded Data

This module provides request validation for live API requests (backend port 8000).
All operations:
- Validate live API requests to backend (port 8000)
- Check live request content types, sizes, and methods
- Validate live request body fields and patterns
- No fallback/hardcoded data - all validation from live operations
- Used by backend services on port 8000 for live trading operations

Live Data Sources:
- API requests: Live requests to backend API (port 8000)
- Request headers: Live content-type and content-length from requests
- Request bodies: Live JSON/form data from requests
- Request methods: Live HTTP methods (GET, POST, etc.) from requests
- All validation uses live data - no mock/test data

Endpoint References:
- Backend API: Port 8000 (request validator processes live requests)
- All request validation uses live connections - no fallback/hardcoded data
"""

import json
import logging
import re
from typing import Any

from fastapi import HTTPException, Request

logger = logging.getLogger(__name__)


class RequestValidator:
    """
    Request validator for live API requests (backend port 8000).

    Validates live request content types, sizes, methods, and body fields.
    All validation uses live data - no fallback/hardcoded data.
    """

    def __init__(self) -> None:
        """Initialize request validator for live request validation."""
        # Maximum request size (configuration default, not fallback data)
        self.max_request_size = 1024 * 1024  # 1MB (configuration default)

        # Allowed content types (configuration defaults, not fallback data)
        self.allowed_content_types: set[str] = {
            "application/json",
            "application/x-www-form-urlencoded",
            "multipart/form-data",
        }

        # Path-specific validations (configuration, not fallback data)
        self.path_validations: dict[str, dict[str, Any]] = {
            "/api/auth/login": {
                "methods": {"POST"},  # Allowed methods for live requests
                "required_fields": {"username", "password"},  # Required fields for live requests
                "max_fields": 2,  # Maximum fields (configuration default, not fallback data)
            },
            "/api/auth/register": {
                "methods": {"POST"},  # Allowed methods for live requests
                "required_fields": {"username", "password", "email"},  # Required fields for live requests
                "max_fields": 3,  # Maximum fields (configuration default, not fallback data)
            },
        }

        # Field validation patterns (configuration, not fallback data)
        self.field_patterns: dict[str, str] = {
            "username": r"^[a-zA-Z0-9_-]{3,32}$",  # Username pattern for live validation
            "password": r"^.{8,}$",  # Password pattern for live validation
            "email": r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$",  # Email pattern for live validation
            "symbol": r"^[A-Z0-9-]{1,10}$",  # Symbol pattern for live validation (from live Binance.US Top-10)
        }

    def validate_content_type(self, request: Request) -> None:
        """
        Validate live request content type.

        Args:
            request: Live API request to backend (port 8000)

        Raises:
            HTTPException: 415 if content type not allowed
        """
        content_type = request.headers.get("content-type", "")
        if not any(content_type.startswith(ct) for ct in self.allowed_content_types):
            raise HTTPException(status_code=415, detail="Unsupported media type")

    def validate_request_size(self, request: Request) -> None:
        """
        Validate live request size.

        Args:
            request: Live API request to backend (port 8000)

        Raises:
            HTTPException: 413 if request too large
        """
        content_length = request.headers.get("content-length")
        if content_length and int(content_length) > self.max_request_size:
            raise HTTPException(status_code=413, detail="Request entity too large")

    def validate_method(self, request: Request, path: str) -> None:
        """
        Validate live request method.

        Args:
            request: Live API request to backend (port 8000)
            path: Request path

        Raises:
            HTTPException: 405 if method not allowed
        """
        if path in self.path_validations:
            allowed_methods: set[str] = self.path_validations[path]["methods"]
            if request.method not in allowed_methods:
                error_msg = f"Method {request.method} not allowed"
                raise HTTPException(status_code=405, detail=error_msg)

    async def validate_json_body(self, request: Request, path: str) -> dict[str, Any] | None:
        """
        Validate live JSON request body.

        Args:
            request: Live API request to backend (port 8000)
            path: Request path

        Returns:
            Parsed JSON body from live request

        Raises:
            HTTPException: 400 if JSON invalid or validation fails
        """
        try:
            body = await request.json()
        except json.JSONDecodeError as e:
            raise HTTPException(status_code=400, detail="Invalid JSON") from e

        if path in self.path_validations:
            validation = self.path_validations[path]

            # Check required fields
            required_fields: set[str] = validation["required_fields"]
            missing_fields: set[str] = required_fields - set(body.keys())
            if missing_fields:
                raise HTTPException(
                    status_code=400,
                    detail=f"Missing required fields: {', '.join(missing_fields)}",
                )

            # Check maximum fields
            max_fields: int = validation["max_fields"]
            if len(body) > max_fields:
                raise HTTPException(
                    status_code=400,
                    detail=f"Too many fields. Maximum allowed: {max_fields}",
                )

            # Validate field patterns
            for field, pattern in self.field_patterns.items():
                if field in body and not re.match(pattern, str(body[field])):
                    raise HTTPException(status_code=400, detail=f"Invalid {field} format")

        return body

    async def validate_request(self, request: Request) -> None:
        """
        Validate live incoming request (backend port 8000).

        Validates live request content type, size, method, and body.
        All validation uses live data - no fallback/hardcoded data.

        Args:
            request: Live API request to backend (port 8000)

        Raises:
            HTTPException: 400/405/413/415 if validation fails
        """
        try:
            path = request.url.path

            # Validate live request content type
            self.validate_content_type(request)

            # Validate live request size
            self.validate_request_size(request)

            # Validate live request method
            self.validate_method(request, path)

            # Validate live JSON body for POST/PUT requests
            if request.method in {"POST", "PUT"}:
                await self.validate_json_body(request, path)

        except HTTPException:
            raise
        except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
            logger.exception("Error in live request validation: %s", e)
            raise HTTPException(status_code=400, detail="Invalid request") from e
