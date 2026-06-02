#!/usr/bin/env python3
"""
CRYPTO AUTOENGINE API Endpoints
Main API for frontend integration
"""

import logging
import time
from typing import Any

# Use absolute imports
from autobuy_system import AutobuyManager
from data_fetchers import DataFetcherManager
from fastapi import (
    APIRouter,
    HTTPException,
    Query,
    Request,
    status,
)

from backend.endpoints.strategies.strategy_manager import StrategyManager
from backend.services.canonical_cache import canonical_cache
from backend.services.portfolio_service import PortfolioService
from backend.services.redis_service import get_redis_service

logger = logging.getLogger(__name__)

# Create router
router = APIRouter(prefix="", tags=["crypto-autoengine"])

# Manager state - using dict to avoid global keyword
_managers_state: dict[str, Any] = {
    "cache": None,
    "data_fetcher_manager": None,
    "strategy_manager": None,
    "autobuy_manager": None,
    "redis_client": None,
}


async def initialize_managers(redis_client_param: Any | None) -> None:
    """Initialize all managers with proper Redis integration"""
    # Store Redis client globally for idempotency operations
    _managers_state["redis_client"] = redis_client_param

    # Use Redis-backed cache when available, fallback to canonical_cache
    if _managers_state["redis_client"] is not None:
        try:
            # Test Redis connection
            _managers_state["redis_client"].ping()
            logger.info("Redis client available, using Redis-backed cache")
            # Initialize canonical cache with Redis backend
            await canonical_cache.initialize()
            _managers_state["cache"] = canonical_cache
        except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
            logger.warning(f"Redis client failed ping test: {e}, falling back to in-memory cache")
            _managers_state["cache"] = canonical_cache
    else:
        logger.warning("Redis client is None, using in-memory cache only")
        _managers_state["cache"] = canonical_cache

    _managers_state["data_fetcher_manager"] = DataFetcherManager(_managers_state["cache"])
    _managers_state["strategy_manager"] = StrategyManager(_managers_state["cache"])
    _managers_state["autobuy_manager"] = AutobuyManager(_managers_state["cache"], _managers_state["strategy_manager"])

    logger.info("CRYPTO AUTOENGINE managers initialized")


def _get_health_error_detail(manager_type: str) -> dict[str, Any]:
    """Get detailed health information for error messages"""
    health = check_system_health()
    return {
        "error_code": f"{manager_type.upper()}_NOT_READY",
        "message": f"{manager_type.title()} manager not initialized",
        "health_status": health,
        "troubleshooting": "Check system startup logs and ensure all required services are running",
    }


def _check_manager_health(manager_type: str, manager_instance: Any) -> None:
    """Check manager health and raise detailed error if not ready"""
    if not manager_instance:
        health_detail = _get_health_error_detail(manager_type)
        raise HTTPException(status_code=503, detail=health_detail)

    # Additional readiness checks for specific managers
    if manager_type == "strategy" and hasattr(manager_instance, "get_strategy_status"):
        try:
            status = manager_instance.get_strategy_status()
            if not status.get("ready", False):
                health_detail = _get_health_error_detail(manager_type)
                health_detail["message"] = f"{manager_type.title()} manager not ready"
                raise HTTPException(status_code=503, detail=health_detail)
        except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
            health_detail = _get_health_error_detail(manager_type)
            health_detail["message"] = f"{manager_type.title()} manager not ready"
            raise HTTPException(status_code=503, detail=health_detail) from e


def check_system_health() -> dict[str, Any]:
    """Check if all managers are properly initialized"""
    health_status = {
        "cache_initialized": _managers_state["cache"] is not None,
        "data_fetcher_initialized": _managers_state["data_fetcher_manager"] is not None,
        "strategy_manager_initialized": _managers_state["strategy_manager"] is not None,
        "autobuy_manager_initialized": _managers_state["autobuy_manager"] is not None,
        "redis_available": _managers_state["redis_client"] is not None,
    }

    if _managers_state["redis_client"]:
        try:
            _managers_state["redis_client"].ping()
            health_status["redis_connected"] = True
        except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError):
            health_status["redis_connected"] = False
    else:
        health_status["redis_connected"] = False

    health_status["all_ready"] = all(
        [
            health_status["cache_initialized"],
            health_status["data_fetcher_initialized"],
            health_status["strategy_manager_initialized"],
            health_status["autobuy_manager_initialized"],
        ],
    )

    return health_status


async def startup_event_handler() -> None:
    """FastAPI startup event handler to ensure managers are initialized"""
    try:
        redis_service = get_redis_service()
        client = redis_service.redis_client if redis_service.redis_client else None
        await initialize_managers(client)
    except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
        logger.exception(f"CRYPTO AUTOENGINE startup failed: {e}")
        raise

    # Verify initialization - move outside try to avoid TRY301
    health = check_system_health()
    if not health["all_ready"]:
        logger.error(f"Manager initialization failed: {health}")
        msg = "Failed to initialize required managers"
        raise RuntimeError(msg)

    logger.info("CRYPTO AUTOENGINE startup completed successfully")


def _handle_idempotency(idempotency_key: str, request: Request) -> None:
    """Handle idempotency check and set with single consistent path"""
    if not idempotency_key:
        raise HTTPException(
            status_code=428,
            detail={
                "error_code": "MISSING_IDEMPOTENCY_KEY",
                "message": "Idempotency-Key header required",
            },
        )

    # Try Redis first if available
    if _managers_state["redis_client"]:
        try:
            # Check if already exists
            exists = _managers_state["redis_client"].exists(f"idempotency:{idempotency_key}")
            # Set with TTL
            set_result = _managers_state["redis_client"].set(f"idempotency:{idempotency_key}", "1", nx=True, ex=300) if not exists else None
        except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
            logger.warning(f"Redis idempotency failed, falling back to in-memory: {e}")
            exists = False
            set_result = None

        # Validate and raise outside try to avoid TRY301
        if exists:
            raise HTTPException(
                status_code=status.HTTP_208_ALREADY_REPORTED,
                detail={
                    "error_code": "ALREADY_REPORTED",
                    "message": "Request already processed",
                },
            )

        if set_result is False:
            # Key was set by another process between check and set
            raise HTTPException(
                status_code=status.HTTP_208_ALREADY_REPORTED,
                detail={
                    "error_code": "ALREADY_REPORTED",
                    "message": "Request already processed",
                },
            )

        if set_result:
            return  # Success with Redis

    # Fallback to in-memory if Redis fails
    try:
        idem_store: set[str] = getattr(request.app.state, "idempotency_keys", set())
        in_memory_exists = idempotency_key in idem_store
        if not in_memory_exists:
            idem_store.add(idempotency_key)
    except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
        logger.warning(f"In-memory idempotency failed: {e}")
        in_memory_exists = False

    # Validate and raise outside try to avoid TRY301
    if in_memory_exists:
        raise HTTPException(
            status_code=status.HTTP_208_ALREADY_REPORTED,
            detail={
                "error_code": "ALREADY_REPORTED",
                "message": "Request already processed",
            },
        )

    request.app.state.idempotency_keys = idem_store


def _check_rate_limit(user_key: str, endpoint: str, limit: int = 5, window_seconds: int = 60) -> bool:
    """Check if user has exceeded rate limit for endpoint"""
    if not _managers_state["redis_client"]:
        return False  # Deny if Redis not available

    try:
        key = f"rate_limit:{endpoint}:{user_key}"
        current_count = _managers_state["redis_client"].get(key)

        # Normalize Redis return type - could be bytes, string, or None
        if current_count is None:
            # First request in window
            _managers_state["redis_client"].setex(key, window_seconds, 1)
            return True

        # Handle different Redis return types safely
        if isinstance(current_count, bytes):
            current_count = int(current_count.decode())
        elif isinstance(current_count, str):
            current_count = int(current_count)
        else:
            current_count = int(current_count)

        if current_count >= limit:
            return False

        # Increment counter
        _managers_state["redis_client"].incr(key)
    except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
        logger.warning(f"Rate limit check failed: {e}")
        return False  # Fail-closed: deny if Redis fails
    else:
        return True


def _acquire_trading_lock(user_key: str, request: Request | None = None, lock_ttl: int = 30) -> bool:
    """
    Acquire a trading lock for a given user_key. Returns True if lock acquired, False otherwise.
    Uses Redis when available, falls back to in-memory per-app state.
    """
    if not user_key:
        return False

    # Try Redis-backed lock
    if _managers_state["redis_client"]:
        try:
            lock_key = f"trading_lock:{user_key}"
            # Redis set with NX and expiration
            return bool(_managers_state["redis_client"].set(lock_key, "1", nx=True, ex=lock_ttl))
        except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
            logger.warning(f"Redis trading lock failed, falling back to in-memory: {e}")

    # Fallback to in-memory lock stored on the FastAPI app state
    try:
        if request is None:
            return False  # Fail closed: no request context to enforce lock
        locks = getattr(request.app.state, "trading_locks", set())
        if user_key in locks:
            return False
        locks.add(user_key)
        request.app.state.trading_locks = locks
    except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
        logger.warning(f"In-memory trading lock failed: {e}")
        return False
    else:
        return True


@router.get(
    "/portfolio/performance",
    operation_id="get_portfolio_performance_api_portfolio_performance_get",
)
async def get_portfolio_performance() -> dict[str, Any]:
    """Get portfolio performance data from live services - NO FALLBACK ZEROS"""
    try:
        portfolio_service = PortfolioService()
        performance = await portfolio_service.get_performance()

        if performance:
            return {"data": performance}

        # NO FALLBACK ZEROS - return explicit unavailable status
        return {
            "status": "unavailable",
            "message": "Portfolio performance data not available",
            "data": None,
            "timestamp": time.time(),
        }
    except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
        logger.exception(f"Error getting portfolio performance: {e}")
        return {
            "status": "error",
            "message": "Portfolio performance service error",
            "error": str(e),
            "data": None,
            "timestamp": time.time(),
        }


@router.get(
    "/portfolio/risk-metrics",
    operation_id="get_portfolio_risk_metrics_api_portfolio_risk_metrics_get",
)
async def get_portfolio_risk_metrics() -> dict[str, Any]:
    """Get portfolio risk metrics from live services - NO FALLBACK ZEROS"""
    try:
        portfolio_service = PortfolioService()
        risk_metrics = await portfolio_service.get_risk_metrics()

        if risk_metrics:
            return {"metrics": risk_metrics}

        # NO FALLBACK ZEROS - return explicit unavailable status
        return {
            "status": "unavailable",
            "message": "Portfolio risk metrics not available",
            "metrics": None,
            "timestamp": time.time(),
        }
    except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
        logger.exception(f"Error getting portfolio risk metrics: {e}")
        return {
            "status": "error",
            "message": "Portfolio risk metrics service error",
            "error": str(e),
            "metrics": None,
            "timestamp": time.time(),
        }


@router.get(
    "/portfolio/allocation",
    operation_id="get_portfolio_allocation_api_portfolio_allocation_get",
)
async def get_portfolio_allocation() -> dict[str, Any]:
    """Get portfolio allocation from live services - NO FALLBACK ZEROS"""
    try:
        portfolio_service = PortfolioService()
        allocation_data = await portfolio_service.get_allocation()

        if allocation_data:
            return {"allocation": allocation_data}

        # NO FALLBACK ZEROS - return explicit unavailable status
        return {
            "status": "unavailable",
            "message": "Portfolio allocation data not available",
            "allocation": None,
            "timestamp": time.time(),
        }
    except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
        logger.exception(f"Error getting portfolio allocation: {e}")
        return {
            "status": "error",
            "message": "Portfolio allocation service error",
            "error": str(e),
            "allocation": None,
            "timestamp": time.time(),
        }


@router.get(
    "/portfolio/asset-performance",
    operation_id="get_portfolio_asset_performance_api_portfolio_asset_performance_get",
)
async def get_portfolio_asset_performance() -> dict[str, Any]:
    """Get asset performance data from live services - NO FALLBACK ZEROS"""
    try:
        portfolio_service = PortfolioService()
        asset_performance = await portfolio_service.get_asset_performance()

        if asset_performance:
            return {"performance": asset_performance}

        # NO FALLBACK ZEROS - return explicit unavailable status
        return {
            "status": "unavailable",
            "message": "Portfolio asset performance data not available",
            "performance": None,
            "timestamp": time.time(),
        }
    except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
        logger.exception(f"Error getting portfolio asset performance: {e}")
        return {
            "status": "error",
            "message": "Portfolio asset performance service error",
            "error": str(e),
            "performance": None,
            "timestamp": time.time(),
        }


@router.get(
    "/portfolio/positions",
    operation_id="get_portfolio_positions_api_portfolio_positions_get",
)
async def get_portfolio_positions(
    limit: int = Query(100, ge=1, le=1000, description="Number of positions to return"),
    offset: int = Query(0, ge=0, description="Number of positions to skip"),
) -> dict[str, Any]:
    """Get current portfolio positions from live data with pagination"""
    try:
        portfolio_service = PortfolioService()
        positions = await portfolio_service.get_positions()

        # Apply pagination to positions
        total_count = len(positions) if positions else 0
        paginated_positions = positions[offset : offset + limit] if positions else []

        return {
            "positions": paginated_positions,
            "pagination": {
                "limit": limit,
                "offset": offset,
                "total": total_count,
                "has_more": offset + limit < total_count,
            },
            "timestamp": time.time(),
        }
    except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
        logger.exception(f"Error getting live portfolio positions: {e}")
        return {
            "status": "error",
            "message": "Portfolio positions service error",
            "error": str(e),
            "positions": [],
            "pagination": {
                "limit": limit,
                "offset": offset,
                "total": 0,
                "has_more": False,
            },
            "timestamp": time.time(),
        }


@router.get(
    "/portfolio/insights",
    operation_id="get_portfolio_insights_api_portfolio_insights_get",
)
async def get_portfolio_insights() -> dict[str, Any]:
    """Get portfolio insights from live services - NO FALLBACK ZEROS"""
    try:
        portfolio_service = PortfolioService()
        insights = await portfolio_service.get_insights()

        if insights:
            return {"insights": insights}

        # NO FALLBACK ZEROS - return explicit unavailable status
        return {
            "status": "unavailable",
            "message": "Portfolio insights not available",
            "insights": None,
            "timestamp": time.time(),
        }
    except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
        logger.exception(f"Error getting portfolio insights: {e}")
        return {
            "status": "error",
            "message": "Portfolio insights service error",
            "error": str(e),
            "insights": None,
            "timestamp": time.time(),
        }


@router.get(
    "/portfolio/monthly-returns",
    operation_id="get_portfolio_monthly_returns_api_portfolio_monthly_returns_get",
)
async def get_portfolio_monthly_returns() -> dict[str, Any]:
    """Get monthly returns data from live services - NO FALLBACK ZEROS"""
    try:
        portfolio_service = PortfolioService()
        monthly_returns = await portfolio_service.get_monthly_returns()

        if monthly_returns:
            return {"data": monthly_returns}

        # NO FALLBACK ZEROS - return explicit unavailable status
        return {
            "status": "unavailable",
            "message": "Portfolio monthly returns data not available",
            "data": None,
            "timestamp": time.time(),
        }
    except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
        logger.exception(f"Error getting portfolio monthly returns: {e}")
        return {
            "status": "error",
            "message": "Portfolio monthly returns service error",
            "error": str(e),
            "data": None,
            "timestamp": time.time(),
        }


@router.get(
    "/portfolio/drawdown",
    operation_id="get_portfolio_drawdown_api_portfolio_drawdown_get",
)
async def get_portfolio_drawdown() -> dict[str, Any]:
    """Get drawdown analysis from live portfolio data - NO FALLBACK ZEROS"""
    try:
        portfolio_service = PortfolioService()
        drawdown_data = await portfolio_service.get_drawdown_analysis()

        if drawdown_data:
            return {"data": drawdown_data}

        # NO FALLBACK ZEROS - return explicit unavailable status
        return {
            "status": "unavailable",
            "message": "Portfolio drawdown data not available",
            "data": None,
            "timestamp": time.time(),
        }
    except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
        logger.exception(f"Error getting portfolio drawdown: {e}")
        return {
            "status": "error",
            "message": "Portfolio drawdown service error",
            "error": str(e),
            "data": None,
            "timestamp": time.time(),
        }


# REMOVED: Fake autobuy status - dashboard uses LIVE data from /autobuy/status in autobuy_endpoints.py

# REMOVED: Fake AI strategies - dashboard uses LIVE data from AI endpoints

# REMOVED: Fake market data - dashboard uses LIVE Binance US data from /market/live in market_data_endpoints.py
