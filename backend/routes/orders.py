"""
Orders Router - Order Management

Order placement, management, advanced orders, and cancellation.
- Live data only (no fabricated fallbacks)
"""

import contextlib
import time
from datetime import datetime, timezone
from typing import Any

from fastapi import APIRouter, Depends, Header, HTTPException, status

from backend.config.redis_config import get_redis_client as get_actual_redis_client
from backend.observability.order_metrics import (
    ACTIVE_ORDERS_GAUGE,
    ADV_ORDERS_CREATE_LATENCY_SECONDS,
    ORDERS_CANCEL_LATENCY_SECONDS,
    ORDERS_CANCELLED_TOTAL,
    ORDERS_CREATE_LATENCY_SECONDS,
    ORDERS_CREATED_TOTAL,
    ORDERS_ERRORS_TOTAL,
    ORDERS_GET_LATENCY_SECONDS,
)
from backend.services.order_service import order_service
from backend.utils.enhanced_logging import get_service_logger

# Lazy import for optional risk service (may not be available in all deployments)
try:
    from backend.services.risk_service import get_risk_service as _get_risk_service_impl
except ImportError:
    _get_risk_service_impl = None  # type: ignore[assignment, misc]

# Alias for backward compatibility
get_risk_service = _get_risk_service_impl

try:
    from backend.services.order_manager_service import (
        order_manager_service,
    )

    ORDER_MANAGER_AVAILABLE = True
except (ImportError, ModuleNotFoundError, AttributeError):
    ORDER_MANAGER_AVAILABLE = False
    order_manager_service = None

router = APIRouter()
logger = get_service_logger("orders")

_ORDER_PLACEMENT_RETIRED_DETAIL = (
    "HTTP order placement is retired. Mystic DAY trades execute only via portfolio_engine integration (execute_buy_fifo / execute_sell_fifo). Dashboard is read-only for orders."
)


def _reject_order_placement() -> None:
    raise HTTPException(status_code=410, detail=_ORDER_PLACEMENT_RETIRED_DETAIL)


def get_redis_client() -> Any:
    """Get Redis client."""
    try:
        return get_actual_redis_client()
    except Exception as e:
        logger.exception("Error getting Redis client")
        raise HTTPException(status_code=500, detail="Redis service unavailable") from e


# Module-level dependency to avoid function call in default argument
_get_redis_client_dep = Depends(get_redis_client)


# ============================================================================
# Helpers
# ============================================================================


def _log_info(message: str, **kwargs: Any) -> None:
    """Log info message."""
    if kwargs:
        logger.info(f"{message} | {kwargs}", extra={"extra_fields": kwargs})
    else:
        logger.info(message)


def _log_warning(message: str, **kwargs: Any) -> None:
    """Log warning message."""
    if kwargs:
        logger.warning(f"{message} | {kwargs}", extra={"extra_fields": kwargs})
    else:
        logger.warning(message)


def _log_error(message: str, error: Exception | None = None, **kwargs: Any) -> None:
    """Log error message."""
    if error is not None:
        exc_tuple = (type(error), error, error.__traceback__)
        if kwargs:
            logger.error(f"{message} | {kwargs}", exc_info=exc_tuple, extra={"extra_fields": kwargs})
        else:
            logger.error(message, exc_info=exc_tuple)
    elif kwargs:
        logger.error(f"{message} | {kwargs}", extra={"extra_fields": kwargs})
    else:
        logger.error(message)


def _idempotency_guard(redis_client: Any, key: str, ttl_sec: int = 300) -> bool:
    """
    Idempotency guard using Redis SETNX + EXPIRE.
    Returns True if key was set (first time), False if present.
    """
    try:
        created = bool(redis_client.setnx(key, "1"))
        if created:
            with contextlib.suppress(Exception):
                redis_client.expire(key, ttl_sec)
    except Exception:
        # If Redis unavailable, don't block request
        return True
    else:
        return created


# ============================================================================
# ORDER MANAGEMENT ENDPOINTS
# ============================================================================


@router.get("/api/orders")
async def get_orders(
    status: str | None = None,
    symbol: str | None = None,
    limit: int = 100,
    offset: int = 0,
    _redis_client: Any = _get_redis_client_dep,
) -> Any:
    """Get all orders with optional filtering."""
    t0 = time.perf_counter()
    try:
        if ORDER_MANAGER_AVAILABLE and order_manager_service:
            orders = await order_manager_service.get_orders(
                status=status,
                symbol=symbol,
                limit=limit,
                offset=offset,
            )
        else:
            orders = await order_service.get_orders(
                status=status,
                symbol=symbol,
                limit=limit,
                offset=offset,
            )

        try:
            if isinstance(orders, list):
                active_count = sum(1 for o in orders if o.get("status") == "active")
                ACTIVE_ORDERS_GAUGE.set(active_count)
        except Exception:
            pass

        ORDERS_GET_LATENCY_SECONDS.observe(time.perf_counter() - t0)
        _log_info(
            "Fetched orders",
            status=status or "any",
            symbol=symbol or "any",
            limit=limit,
            offset=offset,
        )
    except Exception as e:
        ORDERS_ERRORS_TOTAL.labels(operation="get_orders").inc()
        ORDERS_GET_LATENCY_SECONDS.observe(time.perf_counter() - t0)
        _log_error(
            "Error getting orders",
            e,
            status=status,
            symbol=symbol,
            limit=limit,
            offset=offset,
        )
        raise HTTPException(status_code=500, detail=f"Error getting orders: {e}") from e
    else:
        return orders


@router.post("/api/orders", status_code=status.HTTP_201_CREATED)
async def create_order(
    order_data: dict[str, Any],
    redis_client: Any = _get_redis_client_dep,
    idempotency_key: str | None = Header(
        default=None,
        convert_underscores=False,
        alias="Idempotency-Key",
    ),
) -> dict[str, Any]:
    """Retired: order placement is not part of the supported Mystic surface."""
    _reject_order_placement()


@router.get("/api/orders/{order_id}")
async def get_order(order_id: str) -> dict[str, Any]:
    """Get a specific order by ID."""
    t0 = time.perf_counter()
    # Validate order existence outside try to avoid TRY301
    try:
        order = await order_service.get_order(order_id)
    except Exception as e:
        logger.exception(f"Error getting order: {e}")
        raise HTTPException(status_code=500, detail=f"Error getting order: {e}") from e

    if not order:
        raise HTTPException(status_code=404, detail="Order not found")

    try:
        ORDERS_GET_LATENCY_SECONDS.observe(time.perf_counter() - t0)
        _log_info(
            "Fetched order",
            order_id=order_id,
            status=order.get("status"),
        )
    except HTTPException:
        ORDERS_GET_LATENCY_SECONDS.observe(time.perf_counter() - t0)
        raise
    except Exception as e:
        ORDERS_ERRORS_TOTAL.labels(operation="get_order").inc()
        ORDERS_GET_LATENCY_SECONDS.observe(time.perf_counter() - t0)
        _log_error("Error getting order", e, order_id=order_id)
        raise HTTPException(status_code=500, detail=f"Error getting order: {e}") from e
    else:
        return order


@router.post("/api/orders/{order_id}/cancel")
async def cancel_order(order_id: str) -> dict[str, Any]:
    """Cancel a specific order."""
    t0 = time.perf_counter()
    try:
        result = await order_service.cancel_order(order_id)
        ORDERS_CANCELLED_TOTAL.inc()
        ORDERS_CANCEL_LATENCY_SECONDS.observe(time.perf_counter() - t0)
        _log_info("Order cancelled", order_id=order_id, result=bool(result))
        return {
            "status": "success",
            "message": f"Order {order_id} cancelled successfully",
            "result": result,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
    except Exception as e:
        ORDERS_ERRORS_TOTAL.labels(operation="cancel_order").inc()
        ORDERS_CANCEL_LATENCY_SECONDS.observe(time.perf_counter() - t0)
        _log_error("Error cancelling order", e, order_id=order_id)
        raise HTTPException(status_code=500, detail=f"Error cancelling order: {e}") from e


# ============================================================================
# ADVANCED ORDER ENDPOINTS
# ============================================================================


@router.post("/orders/advanced", status_code=status.HTTP_201_CREATED)
async def place_advanced_order(
    order_data: dict[str, Any],
    idempotency_key: str | None = Header(
        default=None,
        convert_underscores=False,
        alias="Idempotency-Key",
    ),
    redis_client: Any = _get_redis_client_dep,
) -> dict[str, Any]:
    """Retired: order placement is not part of the supported Mystic surface."""
    _reject_order_placement()


# ============================================================================
# RISK MANAGEMENT ENDPOINTS
# ============================================================================


@router.get("/risk/parameters")
async def get_risk_parameters() -> dict[str, Any]:
    """Get current risk management parameters from service."""

    def _ensure_risk_service():
        if _get_risk_service_impl is None:
            raise ImportError("Risk service not available")
        return _get_risk_service_impl()

    try:
        risk_service = _ensure_risk_service()
        params = await risk_service.get_risk_parameters()

        if not isinstance(params, dict):
            return {
                "status": "error",
                "error": "Invalid risk parameters format",
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }

        result: dict[str, Any] = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

        # Only include keys that exist in live data
        if params.get("max_position_size") is not None:
            result["max_position_size"] = params["max_position_size"]
        if params.get("max_daily_loss") is not None:
            result["max_daily_loss"] = params["max_daily_loss"]
        if params.get("max_drawdown") is not None:
            result["max_drawdown"] = params["max_drawdown"]
        if params.get("max_leverage") is not None:
            result["max_leverage"] = params["max_leverage"]
        if params.get("correlation_limit") is not None:
            result["correlation_limit"] = params["correlation_limit"]

        _log_info("Fetched risk parameters")
    except ImportError:
        return {
            "status": "error",
            "error": "Risk service not available",
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
    except Exception as e:
        ORDERS_ERRORS_TOTAL.labels(operation="get_risk_parameters").inc()
        _log_error("Error getting risk parameters", e)
        raise HTTPException(
            status_code=500,
            detail=f"Error getting risk parameters: {e}",
        ) from e
    else:
        return result


@router.post("/risk/parameters")
async def update_risk_parameters(
    risk_data: dict[str, Any],
) -> dict[str, Any]:
    """Update risk management parameters."""
    if _get_risk_service_impl is None:
        raise HTTPException(status_code=503, detail="Risk service not available")
    risk_service = _get_risk_service_impl()
    # Validate max position size outside try to avoid TRY301
    max_position_size = risk_data.get("max_position_size")
    if max_position_size is not None and max_position_size > 0.1:
        raise HTTPException(status_code=400, detail="Max position size too high")

    try:
        risk_service = get_risk_service()

        result = await risk_service.update_risk_parameters(risk_data)

        _log_info("Updated risk parameters", keys=list(risk_data.keys()))
        return {
            "status": "success",
            "message": "Risk parameters updated successfully",
            "parameters": result,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
    except ImportError:
        return {
            "status": "error",
            "error": "Risk service not available",
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
    except HTTPException:
        raise
    except Exception as e:
        ORDERS_ERRORS_TOTAL.labels(operation="update_risk_parameters").inc()
        _log_error("Error updating risk parameters", e)
        raise HTTPException(
            status_code=500,
            detail=f"Error updating risk parameters: {e}",
        ) from e


@router.post("/risk/position-size")
async def calculate_position_size(
    data: dict[str, Any],
) -> dict[str, Any]:
    """Calculate optimal position size from risk service."""

    def _ensure_risk_service():
        if _get_risk_service_impl is None:
            raise ImportError("Risk service not available")
        return _get_risk_service_impl()

    try:
        risk_service = _ensure_risk_service()
        position_size = await risk_service.calculate_position_size(data)

        if position_size is None:
            return {
                "status": "error",
                "error": "Position size calculation failed",
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }

        result: dict[str, Any] = {
            "status": "success",
            "position_size": position_size,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

        current_price = data.get("current_price")
        if current_price is not None:
            try:
                price = float(current_price)
                if price > 0:
                    result["quantity"] = position_size / price
            except (ValueError, TypeError):
                pass

        risk_score = data.get("risk_score")
        if risk_score is not None:
            result["risk_score"] = risk_score

        _log_info(
            "Calculated position size",
            symbol=data.get("symbol"),
            position_size=position_size,
        )
    except ImportError:
        return {
            "status": "error",
            "error": "Risk service not available",
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
    except Exception as e:
        ORDERS_ERRORS_TOTAL.labels(operation="calculate_position_size").inc()
        _log_error("Error calculating position size", e)
        raise HTTPException(
            status_code=500,
            detail=f"Error calculating position size: {e}",
        ) from e
    else:
        return result


@router.get("/api/orders/active")
async def get_active_orders(
    symbol: str | None = None,
    limit: int = 100,
    offset: int = 0,
) -> dict[str, Any]:
    """Get active orders for dashboard UI."""
    t0 = time.perf_counter()
    try:
        logger.info(f"Starting get_active_orders: symbol={symbol}, limit={limit}, offset={offset}")

        orders = await order_service.get_orders(
            status="active",
            symbol=symbol,
            limit=limit,
            offset=offset,
        )
        logger.info(f"Got {len(orders) if isinstance(orders, list) else 'non-list'} orders from service")

        try:
            if isinstance(orders, list):
                ACTIVE_ORDERS_GAUGE.set(len(orders))
        except Exception as gauge_error:
            logger.warning(f"Failed to set gauge: {gauge_error}")

        ORDERS_GET_LATENCY_SECONDS.observe(time.perf_counter() - t0)
        order_list = orders if isinstance(orders, list) else []

        response: dict[str, Any] = {
            "orders": order_list,
            "count": len(order_list),
            "timestamp": time.time(),
            "status": "success",
        }
        logger.info(f"Returning response with {len(order_list)} orders")

    except Exception as e:
        ORDERS_ERRORS_TOTAL.labels(operation="get_active_orders").inc()
        ORDERS_GET_LATENCY_SECONDS.observe(time.perf_counter() - t0)
        logger.exception(f"ERROR in get_active_orders: {e}")
        raise HTTPException(status_code=500, detail=f"Error getting active orders: {e}") from e
    else:
        return response


@router.get("/test-simple")
async def test_simple_endpoint() -> dict[str, Any]:
    """Simple test endpoint to verify routing works."""
    return {"message": "Simple endpoint works", "timestamp": time.time()}


@router.post("/api/orders/place", status_code=status.HTTP_201_CREATED)
async def place_order(
    order_data: dict[str, Any],
    redis_client: Any = _get_redis_client_dep,
    idempotency_key: str | None = Header(
        default=None,
        convert_underscores=False,
        alias="Idempotency-Key",
    ),
) -> dict[str, Any]:
    """Retired: order placement is not part of the supported Mystic surface."""
    _reject_order_placement()
