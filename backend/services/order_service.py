"""
Order Service

Handles order operations and order management.
"""

import logging
from typing import Any

# Import from single source of truth
try:
    from backend.config.trading_universe import EXCHANGE_ID
except (ImportError, ModuleNotFoundError, AttributeError, ValueError, TypeError, RuntimeError) as e:
    msg = f"Failed to import trading_universe: {e}"
    raise RuntimeError(msg) from e

from backend.services.execution_mode_service import is_live_execution_allowed_sync
from backend.services.live_trading_service import LiveTradingService

logger = logging.getLogger(__name__)


class OrderService:
    """Service for managing orders."""

    def __init__(self) -> None:
        self.orders = []
        self.order_history = []

    async def get_orders(
        self,
        status: str | None = None,
        symbol: str | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[dict[str, Any]]:
        """Get current orders with live data."""
        try:
            trading_service = LiveTradingService()
            result = await trading_service.get_open_orders()
        except Exception as e:
            logger.exception(f"Error fetching open orders from LiveTradingService: {e}")
            # Return empty list instead of failing
            return []

        if result.get("status") == "success":
            # Flatten orders across exchanges (binance_us only)
            all_orders: list[dict[str, Any]] = []
            for exchange_orders in result.get("orders", {}).values():
                if isinstance(exchange_orders, list):
                    all_orders.extend(exchange_orders)

            # Apply filters
            if status:
                all_orders = [o for o in all_orders if o.get("status", "").lower() == status.lower()]
            if symbol:
                all_orders = [o for o in all_orders if o.get("symbol", "").upper() == symbol.upper()]

            # Apply pagination
            # total_count = len(all_orders)  # Unused
            return all_orders[offset : offset + limit]

        logger.error(f"LiveTradingService error: {result.get('message', 'Unknown error')}")
        return []

    async def get_open_orders(self) -> list[dict[str, Any]]:
        """Alias to get_orders()."""
        return await self.get_orders()

    async def get_order(self, order_id: str) -> dict[str, Any] | None:
        """Get a specific order by ID."""
        try:
            # Get all orders and search for matching ID
            all_orders = await self.get_orders()

            for order in all_orders:
                # Check both 'order_id' and 'id' fields for compatibility
                if order.get("order_id") == order_id or order.get("id") == order_id or order.get("clientOrderId") == order_id:
                    return order
        except Exception as e:
            logger.exception(f"Error getting order {order_id}: {e}")
            return None
        else:
            # Order not found if we reach here
            return None

    async def get_trade_history(
        self,
        limit: int = 100,
        offset: int = 0,
        symbol: str | None = None,
        strategy: str | None = None,
    ) -> dict[str, Any]:
        """Get detailed trade history with optional filtering."""
        trading_service = LiveTradingService()
        result = await trading_service.get_trade_history(symbol, limit)

        if result.get("status") == "success":
            all_trades: list[dict[str, Any]] = []
            for exchange_trades in result.get("trades", {}).values():
                if isinstance(exchange_trades, list):
                    all_trades.extend(exchange_trades)

            # Apply filters
            if symbol:
                all_trades = [t for t in all_trades if t.get("symbol") == symbol]
            if strategy:
                all_trades = [t for t in all_trades if t.get("strategy") == strategy]

            total_count = len(all_trades)
            trades = all_trades[offset : offset + limit]

            return {
                "trades": trades,
                "total_count": total_count,
                "limit": limit,
                "offset": offset,
                "has_more": offset + limit < total_count,
            }
        logger.error(f"LiveTradingService error: {result.get('message', 'Unknown error')}")
        return {
            "trades": [],
            "total_count": 0,
            "limit": limit,
            "offset": offset,
            "has_more": False,
        }

    async def create_order(self, order_data: dict[str, Any]) -> dict[str, Any]:
        """Create a new order (Binance.US)."""
        symbol = order_data.get("symbol", "")
        order_type = order_data.get("type", "MARKET")
        side = order_data.get("side", "BUY")
        amount = float(order_data.get("quantity", 0.0) or 0.0)
        price = order_data.get("price")

        if not is_live_execution_allowed_sync():
            logger.error(f"Live execution disabled. Order rejected: {symbol} {side} {amount}")
            return {"status": "error", "message": "Live execution is disabled"}

        trading_service = LiveTradingService()
        result = await trading_service.place_order(
            EXCHANGE_ID,  # standardize key
            symbol,
            order_type,
            side,
            amount,
            price,
        )

        if result.get("status") == "success":
            return result.get("order", {})
        logger.error(f"Order creation failed: {result.get('message', 'Unknown error')}")
        return {}


# Global instance
order_service = OrderService()
