#!/usr/bin/env python3
"""
Endpoint Health Check Client for Mystic Trading Platform

Provides comprehensive health checking for the Python dashboard with proper
async/sync handling, correct timeout usage, and alignment with actual API endpoints.
Windows/Python 3.12+ compatible with live data only (no placeholders).
"""

import asyncio
import logging
import os
import sys
import time
from collections.abc import Iterable
from typing import Any

import httpx

# Import from single source of truth
try:
    from backend.config.trading_universe import TRADING_SYMBOLS
except (ImportError, ModuleNotFoundError, AttributeError, ValueError, TypeError, RuntimeError) as e:
    msg = f"Failed to import TRADING_SYMBOLS from trading_universe: {e}"
    raise RuntimeError(msg) from e

# Configure logging for Windows compatibility
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger("mystic.health_check")

# All Live Data, No Fallback/Hardcoded Data
BASE = os.environ.get("MYSTIC_BACKEND")
if not BASE:
    msg = "MYSTIC_BACKEND environment variable is required - no fallback/hardcoded backend URL"
    raise RuntimeError(msg)
BASE = BASE.rstrip("/")

# Top-10 symbols from trading_universe (live data)
TOP10 = tuple(TRADING_SYMBOLS)

# Proper httpx timeout configuration
DEFAULT_TIMEOUT = httpx.Timeout(connect=5.0, read=15.0, write=5.0, pool=5.0)


class HealthCheckClient:
    """Health check client with proper async/sync handling and error management"""

    def __init__(self, base_url: str = BASE) -> None:
        self.base_url = base_url.rstrip("/")
        self.client: httpx.AsyncClient | None = None
        self.results: dict[str, Any] = {
            "total_checks": 0,
            "passed": 0,
            "failed": 0,
            "errors": [],
        }

    async def __aenter__(self):
        """Async context manager entry"""
        self.client = httpx.AsyncClient(
            timeout=DEFAULT_TIMEOUT,
            headers={"User-Agent": "mystic-health-check/1.0"},
            follow_redirects=True,
        )
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        if self.client:
            await self.client.aclose()

    def _url(self, path: str) -> str:
        """Build full URL from path"""
        return f"{self.base_url}/{path.lstrip('/')}"

    async def _check_endpoint(
        self,
        path: str,
        expected_status: Iterable[int] = (200,),
        expect_json: bool = True,
        description: str = "",
    ) -> dict[str, Any]:
        """Check a single endpoint and return results"""
        if not self.client:
            msg = "Client not initialized"
            raise RuntimeError(msg)

        self.results["total_checks"] += 1
        check_result = {
            "path": path,
            "description": description or path,
            "status": "unknown",
            "status_code": None,
            "response_time_ms": None,
            "error": None,
            "data": None,
        }

        try:
            loop = asyncio.get_running_loop()
            start_time = loop.time()
            response = await self.client.get(self._url(path))
            end_time = loop.time()

            check_result["status_code"] = response.status_code
            check_result["response_time_ms"] = round((end_time - start_time) * 1000, 2)

            if response.status_code in expected_status:
                check_result["status"] = "passed"
                self.results["passed"] += 1

                if expect_json:
                    content_type = response.headers.get("Content-Type", "").lower()
                    if "application/json" in content_type:
                        try:
                            check_result["data"] = response.json()
                        except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
                            check_result["status"] = "failed"
                            check_result["error"] = f"Invalid JSON: {e}"
                            self.results["failed"] += 1
                    else:
                        check_result["status"] = "failed"
                        check_result["error"] = f"Expected JSON, got {content_type}"
                        self.results["failed"] += 1
                else:
                    check_result["data"] = {"content_length": len(response.content)}
            else:
                check_result["status"] = "failed"
                check_result["error"] = f"Status {response.status_code} not in {tuple(expected_status)}"
                self.results["failed"] += 1
        except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
            check_result["status"] = "failed"
            check_result["error"] = str(e)
            self.results["failed"] += 1
        else:
            if check_result["status"] == "failed":
                self.results["errors"].append(f"{path}: {check_result['error']}")
            return check_result

    async def run_health_checks(self) -> dict[str, Any]:
        """Run comprehensive health checks against actual API endpoints"""
        logger.info("Starting health checks for Mystic Trading Platform...")

        checks = []

        # Core health endpoints (based on actual debug_router implementation)
        checks.append(
            await self._check_endpoint(
                "/api/health",
                expected_status=(200, 503),  # 503 if services not ready
                description="Core health check",
            )
        )

        checks.append(
            await self._check_endpoint(
                "/api/ready",
                expected_status=(200, 503),  # 503 if not ready
                description="Readiness check",
            )
        )

        checks.append(await self._check_endpoint("/api/test", expected_status=(200,), description="Test endpoint"))

        # API documentation endpoints
        checks.append(
            await self._check_endpoint(
                "/api/docs",
                expected_status=(200,),
                expect_json=False,
                description="API documentation",
            )
        )

        checks.append(
            await self._check_endpoint(
                "/api/openapi.json",
                expected_status=(200,),
                description="OpenAPI specification",
            )
        )

        # Try to get supported symbols from the API
        try:
            # Check if we can get coin state for a known symbol
            symbol = TOP10[0]  # BTCUSDT
            checks.append(
                await self._check_endpoint(
                    f"/api/coinstate/{symbol}",
                    expected_status=(200, 404),  # 404 if symbol not supported
                    description=f"Coin state for {symbol}",
                )
            )

            # Check if we can get coins list
            checks.append(
                await self._check_endpoint(
                    "/api/coins",
                    expected_status=(200,),
                    description="Supported coins list",
                )
            )

        except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
            logger.warning(f"Symbol-related checks failed: {e}")

        # Portfolio endpoints (if available)
        checks.append(
            await self._check_endpoint(
                "/api/portfolio/performance",
                expected_status=(200, 404),  # 404 if not implemented
                description="Portfolio performance",
            )
        )

        checks.append(
            await self._check_endpoint(
                "/api/portfolio/positions",
                expected_status=(200, 404),  # 404 if not implemented
                description="Portfolio positions",
            )
        )

        # Trading endpoints (if available)
        checks.append(
            await self._check_endpoint(
                "/api/trading/status",
                expected_status=(200, 404),  # 404 if not implemented
                description="Trading status",
            )
        )

        # Analytics endpoints (if available)
        checks.append(
            await self._check_endpoint(
                "/api/analytics/market-overview",
                expected_status=(200, 404),  # 404 if not implemented
                description="Market analytics",
            )
        )

        # System endpoints
        checks.append(
            await self._check_endpoint(
                "/api/system/status",
                expected_status=(200, 404),  # 404 if not implemented
                description="System status",
            )
        )

        return {
            "summary": self.results,
            "checks": checks,
            "timestamp": asyncio.get_running_loop().time(),
        }


async def run_async() -> dict[str, Any]:
    """Run health checks asynchronously"""
    async with HealthCheckClient() as client:
        return await client.run_health_checks()


def run() -> dict[str, Any]:
    """Run health checks synchronously (Windows-friendly)"""
    try:
        # Use asyncio.run for clean event loop management
        return asyncio.run(run_async())
    except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
        logger.exception("Health check failed")
        return {
            "summary": {
                "total_checks": 0,
                "passed": 0,
                "failed": 1,
                "errors": [f"Health check execution failed: {e}"],
            },
            "checks": [],
            "timestamp": time.time(),
        }


def print_results(results: dict[str, Any]) -> None:
    """Print health check results in a readable format"""
    summary = results["summary"]
    checks = results["checks"]

    logger.info(f"\n{'=' * 60}")
    logger.info("MYSTIC TRADING PLATFORM HEALTH CHECK RESULTS")
    logger.info(f"{'=' * 60}")
    logger.info(f"Total Checks: {summary['total_checks']}")
    logger.info(f"Passed: {summary['passed']}")
    logger.error(f"Failed: {summary['failed']}")
    logger.info(f"Success Rate: {(summary['passed'] / max(summary['total_checks'], 1)) * 100:.1f}%")

    if summary["errors"]:
        logger.error("\nERRORS:")
        for error in summary["errors"]:
            logger.error(f"  [ERROR] {error}")

    logger.info("\nDETAILED RESULTS:")
    for check in checks:
        status_icon = "[OK]" if check["status"] == "passed" else "[ERROR]"
        response_time = f" ({check['response_time_ms']}ms)" if check["response_time_ms"] else ""
        logger.info(f"  {status_icon} {check['description']}: {check['status_code']}{response_time}")

        if check["error"]:
            logger.error(f"    Error: {check['error']}")

    logger.info(f"{'=' * 60}")

    # Return appropriate exit code
    if summary["failed"] > 0:
        logger.error("[ERROR] HEALTH CHECK FAILED")
        return False
    logger.info("[OK] HEALTH CHECK PASSED")
    return True


if __name__ == "__main__":
    try:
        results = run()
        success = print_results(results)
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        logger.info("\nHealth check interrupted by user")
        sys.exit(1)
    except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError):
        logger.exception("Health check failed with exception")
        sys.exit(1)
