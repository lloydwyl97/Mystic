"""
Canonical HTTP Client for Mystic Trading Platform
Single source of truth for all HTTP operations with unified policies
"""

import asyncio
import logging
import time
from typing import Any, Optional

import httpx

logger = logging.getLogger(__name__)


class CanonicalHTTPClient:
    """
    Single canonical HTTP client with unified policies for all outbound requests.
    Replaces all duplicate httpx clients and ad-hoc AsyncClient usage.
    """

    _instance: Optional["CanonicalHTTPClient"] = None
    _client: httpx.AsyncClient | None = None
    _lock: asyncio.Lock | None = None  # Lazy init to avoid event loop issues
    _semaphore: asyncio.Semaphore | None = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    async def get_client(self) -> httpx.AsyncClient:
        """Get the singleton HTTP client, creating it if necessary."""
        if self._client is None:
            # Lazy init lock to avoid event loop issues at module import
            if self._lock is None:
                self._lock = asyncio.Lock()
            async with self._lock:
                if self._client is None:
                    # Unified connection limits and timeouts
                    limits = httpx.Limits(
                        max_connections=100,  # Total connections
                        max_keepalive_connections=50,  # Keep-alive connections
                    )

                    # Further optimized timeout configuration for Binance US
                    timeout = httpx.Timeout(
                        connect=2.0,  # Reduced connection timeout
                        read=10.0,  # Further reduced read timeout for faster responses
                        write=2.0,  # Reduced write timeout
                        pool=2.0,  # Reduced pool timeout
                    )

                    # Standard headers and configuration
                    headers = {
                        "User-Agent": "MysticTradingPlatform/1.0",
                        "Accept": "application/json",
                        "Accept-Encoding": "gzip, deflate",
                        "Connection": "keep-alive",
                    }

                    # Create transport with retries but without pool_limits (not supported in this httpx version)
                    transport = httpx.AsyncHTTPTransport(retries=3)

                    self._client = httpx.AsyncClient(
                        limits=limits,
                        timeout=timeout,
                        follow_redirects=True,
                        verify=True,
                        headers=headers,
                        http2=False,  # Disable HTTP/2 for compatibility
                        transport=transport,
                    )

                    # Single concurrency semaphore for all requests
                    self._semaphore = asyncio.Semaphore(50)  # Max 50 concurrent requests

                    logger.info("Canonical HTTP client created with unified policies")

        return self._client

    def get_semaphore(self) -> asyncio.Semaphore:
        """Get the concurrency control semaphore."""
        if self._semaphore is None:
            self._semaphore = asyncio.Semaphore(50)
        return self._semaphore

    async def make_request(self, method: str, url: str, **kwargs) -> httpx.Response:
        """
        Make an HTTP request using the canonical client with concurrency control.
        Includes retry logic and performance monitoring.
        FIX 3: Binance URLs go through resilience_service binance_api circuit breaker (DNS/outage).
        """

        async def _do_request() -> httpx.Response:
            client = await self.get_client()
            semaphore = self.get_semaphore()
            async with semaphore:
                start_time = time.time()
                retry_count = 0
                max_retries = 2
                retry_delay = 0.5
                while True:
                    try:
                        response = await asyncio.wait_for(
                            client.request(method, url, **kwargs),
                            timeout=15.0,
                        )
                        break
                    except (
                        httpx.ConnectError,
                        httpx.ReadError,
                        asyncio.TimeoutError,
                    ) as e:
                        retry_count += 1
                        if retry_count > max_retries:
                            logger.exception(f"Max retries exceeded for {method} {url}: {e}")
                            raise
                        logger.warning(f"Retrying request ({retry_count}/{max_retries}) after error: {e}")
                        await asyncio.sleep(retry_delay * retry_count)
                duration = time.time() - start_time
                if duration > 1.0:
                    logger.warning(f"Slow request: {method} {url} took {duration:.2f}s")
                return response

        if "binance" in url.lower():
            from backend.services.resilience_service import get_resilience_service

            resilience = await get_resilience_service()
            return await resilience.execute_with_resilience(
                _do_request,
                operation_name="binance_http",
                circuit_breaker="binance_api",
                use_fallback=False,
            )
        try:
            return await _do_request()
        except asyncio.TimeoutError as e:
            logger.exception(f"Request timeout: {method} {url}")
            raise httpx.TimeoutException(f"Request timeout: {method} {url}") from e
        except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
            logger.exception(f"Request error: {method} {url} - {e}")
            raise

    async def get_json(self, url: str, **kwargs) -> dict[str, Any]:
        """
        Make a GET request and return JSON data using canonical client.
        """
        response = await self.make_request("GET", url, **kwargs)
        response.raise_for_status()
        return response.json()

    async def post_json(self, url: str, data: dict[str, Any], **kwargs) -> dict[str, Any]:
        """
        Make a POST request with JSON data using canonical client.
        """
        kwargs.setdefault("json", data)
        response = await self.make_request("POST", url, **kwargs)
        response.raise_for_status()
        return response.json()

    async def close(self):
        """Close the HTTP client."""
        if self._client:
            await self._client.aclose()
            self._client = None
            logger.info("Canonical HTTP client closed")


# Global canonical client instance
canonical_http_client = CanonicalHTTPClient()


# Convenience functions for backward compatibility
async def get_http_client() -> httpx.AsyncClient:
    """Get the canonical HTTP client."""
    return await canonical_http_client.get_client()


async def make_request(method: str, url: str, **kwargs) -> httpx.Response:
    """
    Make an HTTP request using the canonical client.
    """
    return await canonical_http_client.make_request(method, url, **kwargs)


async def get_json(url: str, **kwargs) -> dict[str, Any]:
    """
    Make a GET request and return JSON data using canonical client.
    """
    return await canonical_http_client.get_json(url, **kwargs)


async def post_json(url: str, data: dict[str, Any], **kwargs) -> dict[str, Any]:
    """
    Make a POST request with JSON data using canonical client.
    """
    return await canonical_http_client.post_json(url, data, **kwargs)


async def close_http_client():
    """Close the canonical HTTP client."""
    await canonical_http_client.close()
