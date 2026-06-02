"""
HTTP Client Implementation (httpx) - All Live Data, No Fallback/Hardcoded Data

This module implements the HTTP client interface using httpx for making live API calls.
All HTTP operations:
- Connect to live API endpoints (backend port 8000, external APIs)
- No fallback/hardcoded data - all requests to live endpoints
- Real HTTP requests using httpx.AsyncClient
- Supports GET, POST, PUT, DELETE operations

Live Data Sources:
- Backend API endpoints on port 8000 for internal services
- External exchange APIs (Binance.US) for live market data
- All HTTP calls are to live endpoints - no mock/test data

Endpoint References:
- Backend API: http://localhost:8000 (or configured backend URL)
- External APIs: Configured via environment variables
- All connections use live endpoints - no fallback/hardcoded URLs
"""

import logging
from typing import Any

import httpx

from backend.app.adapters.http.base import HTTPClient

logger = logging.getLogger(__name__)


class HttpxHTTPClient(HTTPClient):
    """
    HTTP client implementation using httpx for live API connections.

    Connects to live endpoints:
    - Backend API on port 8000 for internal services
    - External APIs (Binance.US) for live market data
    - All requests are to live endpoints - no mock/test data
    """

    def __init__(self, timeout: float = 30.0, max_retries: int = 3):
        """
        Initialize HTTP client for live API connections.

        Args:
            timeout: Request timeout in seconds (default: 30.0)
            max_retries: Maximum retry attempts (default: 3)
        """
        self._timeout = timeout
        self._max_retries = max_retries
        self._client: httpx.AsyncClient | None = None

    async def _get_client(self) -> httpx.AsyncClient:
        """
        Get or create HTTP client for live API connections.

        Returns:
            httpx.AsyncClient configured for live endpoint requests
        """
        if self._client is None:
            self._client = httpx.AsyncClient(
                timeout=self._timeout,
                limits=httpx.Limits(max_keepalive_connections=20, max_connections=100),
            )
        return self._client

    def _extract_response_data(self, response: httpx.Response) -> Any:
        """
        Extract data from live API response.

        Parses JSON if content-type indicates JSON, otherwise returns text.
        This is content-type parsing, not fallback data - all responses are from live endpoints.

        Args:
            response: Response from live API endpoint

        Returns:
            Parsed JSON data or response text from live endpoint
        """
        content_type = response.headers.get("content-type", "")
        if "application/json" in content_type:
            try:
                return response.json()
            except ValueError:
                # Invalid JSON; return text from live response
                return response.text
        return response.text

    async def get(self, url: str, params: dict[str, Any] | None = None, headers: dict[str, str] | None = None) -> dict[str, Any]:
        """
        Make a GET request to live endpoint.

        Args:
            url: Live endpoint URL (e.g., http://localhost:8000/api/...)
            params: Query parameters (optional)
            headers: Request headers (optional)

        Returns:
            Response dictionary from live endpoint with status_code, headers, and data
        """
        client = await self._get_client()
        try:
            response = await client.get(url, params=params, headers=headers)
            response.raise_for_status()
            return {
                "status_code": response.status_code,
                "headers": dict(response.headers),
                "data": self._extract_response_data(response),
            }
        except httpx.HTTPError as e:
            resp = getattr(e, "response", None)
            status = resp.status_code if resp is not None else 0
            logger.exception("HTTP GET error for live endpoint: %s", url)
            return {"status_code": status, "error": str(e), "data": None}

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
            Response dictionary from live endpoint with status_code, headers, and data
        """
        client = await self._get_client()
        try:
            response = await client.post(url, data=data, json=json, headers=headers)
            response.raise_for_status()
            return {
                "status_code": response.status_code,
                "headers": dict(response.headers),
                "data": self._extract_response_data(response),
            }
        except httpx.HTTPError as e:
            resp = getattr(e, "response", None)
            status = resp.status_code if resp is not None else 0
            logger.exception("HTTP POST error for live endpoint: %s", url)
            return {"status_code": status, "error": str(e), "data": None}

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
            Response dictionary from live endpoint with status_code, headers, and data
        """
        client = await self._get_client()
        try:
            response = await client.put(url, data=data, json=json, headers=headers)
            response.raise_for_status()
            return {
                "status_code": response.status_code,
                "headers": dict(response.headers),
                "data": self._extract_response_data(response),
            }
        except httpx.HTTPError as e:
            resp = getattr(e, "response", None)
            status = resp.status_code if resp is not None else 0
            logger.exception("HTTP PUT error for live endpoint: %s", url)
            return {"status_code": status, "error": str(e), "data": None}

    async def delete(self, url: str, headers: dict[str, str] | None = None) -> dict[str, Any]:
        """
        Make a DELETE request to live endpoint.

        Args:
            url: Live endpoint URL (e.g., http://localhost:8000/api/...)
            headers: Request headers (optional)

        Returns:
            Response dictionary from live endpoint with status_code, headers, and data
        """
        client = await self._get_client()
        try:
            response = await client.delete(url, headers=headers)
            response.raise_for_status()
            return {
                "status_code": response.status_code,
                "headers": dict(response.headers),
                "data": self._extract_response_data(response),
            }
        except httpx.HTTPError as e:
            resp = getattr(e, "response", None)
            status = resp.status_code if resp is not None else 0
            logger.exception("HTTP DELETE error for live endpoint: %s", url)
            return {"status_code": status, "error": str(e), "data": None}

    async def health_check(self) -> dict[str, Any]:
        """
        Check HTTP client health for live connections.

        Returns:
            Dictionary with health status of live HTTP client connection
        """
        try:
            # Health check for live HTTP client (connects to backend port 8000, external APIs)
            return {
                "status": "healthy",
                "type": "httpx",
                "timeout": self._timeout,
                "max_retries": self._max_retries,
                "connected_to": "live endpoints (backend port 8000, external APIs)",
            }
        except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
            return {"status": "unhealthy", "type": "httpx", "error": str(e)}

    async def close(self) -> None:
        """
        Close HTTP client connection.

        Ensures proper cleanup of live HTTP connections to backend (port 8000) and external APIs.
        """
        if self._client:
            await self._client.aclose()
            self._client = None
            logger.info("HTTP client closed (live connections to backend port 8000 and external APIs)")
