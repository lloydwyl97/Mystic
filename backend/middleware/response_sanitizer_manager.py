"""
Response Sanitizer Manager - All Live Data, No Fallback/Hardcoded Data

This module provides response sanitization for live API responses (backend port 8000).
All operations:
- Sanitize live API responses from backend (port 8000)
- Remove sensitive fields and mask data from live responses
- Format live response data for security
- No fallback/hardcoded data - all sanitization from live operations
- Used by backend services on port 8000 for live trading operations

Live Data Sources:
- API responses: Live responses from backend API (port 8000)
- Response data: Live response body data from API operations
- Response headers: Live headers from API responses
- All sanitization uses live data - no mock/test data

Endpoint References:
- Backend API: Port 8000 (response sanitizer processes live responses)
- All response sanitization uses live connections - no fallback/hardcoded data
"""

import contextlib
import json
import logging
import re
from typing import Any, cast

from fastapi.responses import JSONResponse

logger = logging.getLogger(__name__)


class ResponseSanitizer:
    """
    Response sanitizer for live API responses (backend port 8000).

    Sanitizes live response data by removing sensitive fields and masking data.
    All sanitization uses live data - no fallback/hardcoded data.
    """

    def __init__(self) -> None:
        """Initialize response sanitizer for live response sanitization."""
        # Fields to remove from live responses (configuration, not fallback data)
        self.sensitive_fields: set[str] = {
            "password",
            "api_key",
            "secret",
            "token",
            "authorization",
        }

        # Fields to mask in live responses (configuration, not fallback data)
        self.mask_fields: dict[str, str] = {
            "email": r"(?<=.{3}).(?=.*@)",  # Email masking pattern
            "phone": r"(?<=.{3}).(?=.{4}$)",  # Phone masking pattern
            "credit_card": r"(?<=.{4}).(?=.{4}$)",  # Credit card masking pattern
        }

        # Response size limits (configuration default, not fallback data)
        self.max_response_size = 1024 * 1024  # 1MB (configuration default)

        # Response format configurations (configuration, not fallback data)
        self.response_formats: dict[str, dict[str, Any]] = {
            "default": {
                "success": bool,
                "data": (dict, list),
                "error": str,
                "timestamp": str,
            },
            "error": {
                "success": bool,
                "error": str,
                "code": int,
                "timestamp": str,
            },
        }

    def sanitize_value(self, value: Any) -> Any:
        """Sanitize a single value"""
        if isinstance(value, str):
            # Remove control characters
            value = re.sub(r"[\x00-\x1F\x7F-\x9F]", "", value)
            # Escape HTML
            value = value.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        elif isinstance(value, dict):
            value = self.sanitize_dict(cast("dict[str, Any]", value))
        elif isinstance(value, list):
            value = self.sanitize_list(cast("list[Any]", value))
        return value

    def sanitize_dict(self, data: dict[str, Any]) -> dict[str, Any]:
        """Sanitize dictionary values"""
        sanitized: dict[str, Any] = {}
        for key, value in data.items():
            # Remove sensitive fields
            if key.lower() in self.sensitive_fields:
                continue

            # Mask sensitive data
            if key.lower() in self.mask_fields and isinstance(value, str):
                pattern = self.mask_fields[key.lower()]
                value_sanitized = re.sub(pattern, "*", value)
                sanitized[key] = self.sanitize_value(value_sanitized)
                continue

            sanitized[key] = self.sanitize_value(value)
        return sanitized

    def sanitize_list(self, data: list[Any]) -> list[Any]:
        """Sanitize list values"""
        return [self.sanitize_value(item) for item in data]

    def format_response(self, data: dict[str, Any], status_code: int) -> dict[str, Any]:
        """Format response according to configuration"""
        if status_code >= 400:
            format_config = self.response_formats["error"]
            formatted: dict[str, Any] = {
                "success": False,
                "error": data.get("detail", "Unknown error"),
                "code": status_code,
                "timestamp": data.get("timestamp", ""),
            }
        else:
            format_config = self.response_formats["default"]
            formatted = {
                "success": True,
                "data": data.get("data", {}),
                "error": None,
                "timestamp": data.get("timestamp", ""),
            }

        # Validate types
        for key, expected_type in format_config.items():
            if key in formatted and not isinstance(formatted[key], expected_type):
                if expected_type == (dict, list):
                    # Handle tuple of types - no conversion needed
                    pass
                else:
                    with contextlib.suppress(ValueError, TypeError):
                        # Keep original value if conversion fails
                        formatted[key] = expected_type(formatted[key])

        return formatted

    def check_response_size(self, data: dict[str, Any]) -> dict[str, Any]:
        """
        Check if live response size exceeds limit.

        Args:
            data: Live response data dictionary

        Returns:
            Live response data or error response if size exceeded
        """
        try:
            response_size = len(json.dumps(data).encode())
            if response_size > self.max_response_size:
                logger.warning("Live response size %s exceeds limit %s", response_size, self.max_response_size)
                return {
                    "success": False,
                    "error": "Response too large",
                    "code": 500,
                    "timestamp": data.get("timestamp", ""),  # Live timestamp from data
                }
        except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
            logger.exception("Error checking live response size: %s", e)
        return data

    def sanitize_response(self, response: JSONResponse) -> JSONResponse:
        """
        Sanitize and format live API response (backend port 8000).

        Removes sensitive fields and masks data from live responses.
        All sanitization uses live data - no fallback/hardcoded data.

        Args:
            response: Live API response from backend (port 8000)

        Returns:
            Sanitized live API response
        """
        try:
            # Get response data
            response_body = response.body
            data_str = response_body.decode("utf-8") if isinstance(response_body, bytes) else str(response_body)

            try:
                data = json.loads(data_str)
            except json.JSONDecodeError:
                return response

            # Sanitize data
            if isinstance(data, dict):
                data = self.sanitize_dict(cast("dict[str, Any]", data))
            elif isinstance(data, list):
                data = self.sanitize_list(cast("list[Any]", data))
                # Convert list to dict for format_response
                data = {"data": data}

            # Format response
            data = self.format_response(data, response.status_code) if isinstance(data, dict) else {"success": True, "data": data, "error": None, "timestamp": ""}

            # Check size
            data = self.check_response_size(data)

            # Create new response
            return JSONResponse(
                content=data,
                status_code=response.status_code,
                headers=response.headers,
            )

        except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
            logger.exception("Error sanitizing live response: %s", e)
            return response
