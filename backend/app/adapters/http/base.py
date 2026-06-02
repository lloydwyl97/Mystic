"""
HTTP Client Interface - All Live Data, No Fallback/Hardcoded Data

This module defines the abstract HTTP client interface for making live API calls to external services.
All HTTP operations:
- Connect to live API endpoints (backend port 8000, external APIs)
- No fallback/hardcoded data - all requests to live endpoints
- Support for GET, POST, PUT, DELETE operations
- Abstract interface allows multiple HTTP client implementations

Live Data Sources:
- Backend API endpoints on port 8000 for internal services
- External exchange APIs (Binance.US) for live market data
- All HTTP calls are to live endpoints - no mock/test data

Endpoint References:
- Backend API: http://localhost:8000 (or configured backend URL)
- External APIs: Configured via environment variables
- All connections use live endpoints - no fallback/hardcoded URLs
"""

from abc import ABC, abstractmethod
from typing import Any


class HTTPClient(ABC):
    """
    Abstract HTTP client interface for live API connections.

    All implementations must:
    - Connect to live endpoints (backend port 8000, external APIs)
    - Make real HTTP requests - no mock/test data
    - Support standard HTTP methods for live data operations
    - Provide health checks for live connections
    """

    @abstractmethod
    async def get(self, url: str, params: dict[str, Any] | None = None, headers: dict[str, str] | None = None) -> dict[str, Any]:
        """
        Make a GET request to live endpoint.

        Args:
            url: Live endpoint URL (e.g., http://localhost:8000/api/...)
            params: Query parameters (optional)
            headers: Request headers (optional)

        Returns:
            Response dictionary from live endpoint
        """

    @abstractmethod
    async def post(
        self,
        url: str,
        data: dict[str, Any] | None = None,
        json: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        """
        Make a POST request to live endpoint.

        Args:
            url: Live endpoint URL (e.g., http://localhost:8000/api/...)
            data: Form data (optional)
            json: JSON data (optional)
            headers: Request headers (optional)

        Returns:
            Response dictionary from live endpoint
        """

    @abstractmethod
    async def put(
        self,
        url: str,
        data: dict[str, Any] | None = None,
        json: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        """
        Make a PUT request to live endpoint.

        Args:
            url: Live endpoint URL (e.g., http://localhost:8000/api/...)
            data: Form data (optional)
            json: JSON data (optional)
            headers: Request headers (optional)

        Returns:
            Response dictionary from live endpoint
        """

    @abstractmethod
    async def delete(self, url: str, headers: dict[str, str] | None = None) -> dict[str, Any]:
        """
        Make a DELETE request to live endpoint.

        Args:
            url: Live endpoint URL (e.g., http://localhost:8000/api/...)
            headers: Request headers (optional)

        Returns:
            Response dictionary from live endpoint
        """

    @abstractmethod
    async def health_check(self) -> dict[str, Any]:
        """
        Check HTTP client health for live connections.

        Returns:
            Dictionary with health status of live HTTP client connection
        """

    @abstractmethod
    async def close(self) -> None:
        """
        Close HTTP client connection.

        Ensures proper cleanup of live HTTP connections.
        """
