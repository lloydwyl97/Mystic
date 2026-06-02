"""
Startup Health Checks for Critical Services

This module provides health check functions to validate that critical
dependencies are available before starting the trading system.
"""

import asyncio
import logging
import os
from typing import Any

logger = logging.getLogger(__name__)


async def check_redis_health() -> dict[str, Any]:
    """
    Check if Redis is available and responding.

    Returns:
        dict with 'healthy', 'error' keys
    """
    try:
        import redis.asyncio as redis

        redis_url = os.getenv("REDIS_URL", "redis://localhost:6379/0")
        client = redis.from_url(redis_url, decode_responses=True)

        # Test ping
        await client.ping()

        # Test basic operations
        await client.set("health_check", "ok", ex=60)
        value = await client.get("health_check")

        await client.aclose()

        if value != "ok":
            return {"healthy": False, "service": "Redis", "error": "Redis write/read test failed"}

        return {
            "healthy": True,
            "service": "Redis",
            "url": redis_url.split("@")[-1] if "@" in redis_url else redis_url,  # Hide password
        }

    except Exception as e:
        return {"healthy": False, "service": "Redis", "error": str(e)}


async def check_database_health() -> dict[str, Any]:
    """
    Check if SQLite database is accessible and writable.

    Returns:
        dict with 'healthy', 'error' keys
    """
    try:
        from backend.database_schema import DATABASE_PATH, execute_read, execute_write

        # Test read
        result = await execute_read("SELECT 1", fetchone=True)
        if not result or result[0] != 1:
            return {"healthy": False, "service": "Database", "error": "Database read test failed"}

        # Test write
        await execute_write("""
            CREATE TABLE IF NOT EXISTS health_check (
                id INTEGER PRIMARY KEY,
                timestamp TEXT
            )
        """)

        return {"healthy": True, "service": "Database", "path": DATABASE_PATH}

    except Exception as e:
        return {"healthy": False, "service": "Database", "error": str(e)}


async def check_portfolio_engine_health() -> dict[str, Any]:
    """
    Check if Portfolio Engine can be initialized.

    Returns:
        dict with 'healthy', 'error' keys
    """
    try:
        from backend.services.portfolio_engine import get_portfolio_engine, is_portfolio_engine_initialized

        # Check if already initialized
        if is_portfolio_engine_initialized():
            engine = get_portfolio_engine()
            return {"healthy": True, "service": "Portfolio Engine", "status": "initialized", "positions": len(engine.open_positions), "cash_balance": engine.cash_balance}

        # Try to initialize
        from backend.services.portfolio_engine import initialize_portfolio_engine

        engine = await initialize_portfolio_engine()

        if not engine:
            return {"healthy": False, "service": "Portfolio Engine", "error": "Failed to initialize"}

        return {"healthy": True, "service": "Portfolio Engine", "status": "initialized", "positions": len(engine.open_positions), "cash_balance": engine.cash_balance}

    except Exception as e:
        return {"healthy": False, "service": "Portfolio Engine", "error": str(e)}


async def wait_for_redis(timeout: float = 30.0, retry_interval: float = 2.0) -> bool:
    """
    Wait for Redis to become available.

    Args:
        timeout: Maximum time to wait in seconds
        retry_interval: Time between retries in seconds

    Returns:
        True if Redis is available, False if timeout
    """
    start_time = asyncio.get_event_loop().time()

    while (asyncio.get_event_loop().time() - start_time) < timeout:
        result = await check_redis_health()
        if result["healthy"]:
            logger.info(f"Redis is available: {result.get('url', 'N/A')}")
            return True

        logger.warning(f"⏳ Waiting for Redis... ({result.get('error', 'unknown error')})")
        await asyncio.sleep(retry_interval)

    logger.error(f"Redis did not become available within {timeout}s")
    return False


async def wait_for_database(timeout: float = 10.0, retry_interval: float = 1.0) -> bool:
    """
    Wait for database to become available.

    Args:
        timeout: Maximum time to wait in seconds
        retry_interval: Time between retries in seconds

    Returns:
        True if database is available, False if timeout
    """
    start_time = asyncio.get_event_loop().time()

    while (asyncio.get_event_loop().time() - start_time) < timeout:
        result = await check_database_health()
        if result["healthy"]:
            logger.info(f"Database is available: {result.get('path', 'N/A')}")
            return True

        logger.warning(f"⏳ Waiting for database... ({result.get('error', 'unknown error')})")
        await asyncio.sleep(retry_interval)

    logger.error(f"Database did not become available within {timeout}s")
    return False


async def run_all_health_checks() -> dict[str, Any]:
    """
    Run all health checks and return comprehensive status.

    Returns:
        dict with 'healthy' (bool) and 'checks' (list) keys
    """
    logger.info("🏥 Running startup health checks...")

    checks = []
    all_healthy = True

    # Check Redis
    redis_result = await check_redis_health()
    checks.append(redis_result)
    if not redis_result["healthy"]:
        all_healthy = False
        logger.error(f"{redis_result['service']} FAILED: {redis_result.get('error', 'unknown error')}")
    else:
        logger.info(f"{redis_result['service']}: OK")

    # Check Database
    db_result = await check_database_health()
    checks.append(db_result)
    if not db_result["healthy"]:
        all_healthy = False
        logger.error(f"{db_result['service']} FAILED: {db_result.get('error', 'unknown error')}")
    else:
        logger.info(f"{db_result['service']}: OK")

    # Check Portfolio Engine (only if Redis and DB are healthy)
    if all_healthy:
        engine_result = await check_portfolio_engine_health()
        checks.append(engine_result)
        if not engine_result["healthy"]:
            all_healthy = False
            logger.error(f"{engine_result['service']} FAILED: {engine_result.get('error', 'unknown error')}")
        else:
            logger.info(f"{engine_result['service']}: OK ({engine_result.get('status', 'unknown')})")
    else:
        logger.warning("Skipping Portfolio Engine check (dependencies not healthy)")

    if all_healthy:
        logger.info("All health checks passed!")
    else:
        logger.error("Some health checks failed - review logs above")

    return {"healthy": all_healthy, "checks": checks, "timestamp": asyncio.get_event_loop().time()}
