"""
Live Trading Endpoints - Core Live Trading API

Provides the essential live trading endpoints that were missing:
- /api/trading/orders - Live order management
- /api/trading/portfolio - Live portfolio tracking
- /api/trading/balance - Live account balance

All endpoints use existing services and require admin authentication.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone
from typing import Any

from fastapi import APIRouter, Body, Depends, HTTPException

from backend.services.admin_auth import require_admin_key
from backend.services.live_trading_service import LiveTradingService
from backend.services.portfolio_service import get_portfolio_service

# LiveTradingService is already imported above

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/trading", tags=["live-trading"])

# Module-level dependencies to avoid function calls in default arguments
_require_admin_dep = Depends(require_admin_key)
_request_body = Body(...)


# Lazy-loaded LiveTradingService instance
class _TradingServiceManager:
    _instance: LiveTradingService | None = None

    @classmethod
    def get_instance(cls) -> LiveTradingService:
        if cls._instance is None:
            cls._instance = LiveTradingService()
        return cls._instance


def _get_trading_service() -> LiveTradingService:
    """Get LiveTradingService instance."""
    return _TradingServiceManager.get_instance()


@router.get("/orders")
async def get_live_orders(_: None = _require_admin_dep) -> dict[str, Any]:
    """
    Get all live trading orders from Binance.US.

    Returns active orders, order history, and order status.
    Requires admin authentication.
    """
    try:
        # Get live trading service instance
        trading_service = _get_trading_service()

        # Fetch open orders
        open_orders = await trading_service.get_open_orders()

        # Fetch recent order history (last 100 orders)
        order_history = await trading_service.get_trade_history(limit=50)

        return {
            "status": "success",
            "open_orders": open_orders.get("orders", []),
            "order_history": order_history.get("trades", []),
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "source": "binance_us_live",
        }

    except Exception as e:
        logger.exception(f"Failed to get live orders: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to retrieve live orders: {e!s}") from e


@router.get("/portfolio")
async def get_live_portfolio(_: None = _require_admin_dep) -> dict[str, Any]:
    """
    Get live portfolio positions and holdings from Binance.US.

    Returns current positions, unrealized P&L, and portfolio metrics.
    Requires admin authentication.
    """
    try:
        # Get live trading service instance
        trading_service = _get_trading_service()

        # Get account balance (includes positions)
        account_data = await trading_service.get_account_balance()

        # Get portfolio service for additional analytics
        portfolio_service = get_portfolio_service()

        # Get portfolio summary from portfolio service
        portfolio_summary = {}
        try:
            # Try to get portfolio data if available
            portfolio_data = await portfolio_service.get_portfolio_summary()
            portfolio_summary = portfolio_data or {}
        except Exception as e:
            logger.warning(f"Could not get portfolio summary: {e}")

        return {
            "status": "success",
            "account_balance": account_data.get("balance", {}),
            "positions": account_data.get("positions", []),
            "portfolio_summary": portfolio_summary,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "source": "binance_us_live",
        }

    except Exception as e:
        logger.exception(f"Failed to get live portfolio: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to retrieve live portfolio: {e!s}") from e


@router.get("/balance")
async def get_live_balance(_: None = _require_admin_dep) -> dict[str, Any]:
    """
    Get live account balance from Binance.US.

    Returns account balances, available funds, and account status.
    Requires admin authentication.
    """
    try:
        # Get live trading service instance
        trading_service = _get_trading_service()

        # Get account balance
        balance_data = await trading_service.get_account_balance()

        # Get account balances specifically
        account_balances = await trading_service.get_account_balances()

        return {
            "status": "success",
            "balance": balance_data.get("balance", {}),
            "account_balances": account_balances.get("balances", []),
            "total_usd_value": balance_data.get("total_usd", 0),
            "free_usd": balance_data.get("free_usd", 0),
            "locked_usd": balance_data.get("locked_usd", 0),
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "source": "binance_us_live",
        }

    except Exception as e:
        logger.exception(f"Failed to get live balance: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to retrieve live balance: {e!s}") from e


@router.post("/execute_live_market_buy")
async def execute_live_market_buy(payload: dict[str, Any] = _request_body, _: None = _require_admin_dep) -> dict[str, Any]:
    """
    Execute a live market buy order on Binance.US.

    Requires admin authentication and valid API keys.
    """
    trading_service = _get_trading_service()
    symbol = str(payload.get("symbol") or "").upper()
    quote_usd = float(payload.get("quote_usd") or 0)

    if not symbol or quote_usd <= 0:
        raise HTTPException(status_code=400, detail="symbol and positive quote_usd required")

    try:
        # Use Binance API directly with quoteOrderQty for market buy
        if trading_service.binance:
            pair = symbol.replace("USDT", "/USDT") if "USDT" in symbol else symbol + "/USDT"
            order = await asyncio.to_thread(
                trading_service.binance.create_order,
                symbol=pair,
                type="market",
                side="buy",
                amount=None,  # Not used with quoteOrderQty
                price=None,
                params={"quoteOrderQty": quote_usd},
            )
            order_result = {
                "status": "success",
                "order": {
                    "id": order.get("id"),
                    "symbol": order.get("symbol"),
                    "type": order.get("type"),
                    "side": order.get("side"),
                    "amount": order.get("amount"),
                    "price": order.get("price"),
                    "clientOrderId": order.get("clientOrderId"),
                    "status": order.get("status"),
                    "timestamp": order.get("timestamp"),
                },
            }
        else:
            order_result = {"status": "error", "message": "Binance client not available"}

        if order_result.get("status") == "success":
            _audit_trade("live_market_buy", {"symbol": symbol, "quote_usd": quote_usd})
            return {"status": "success", "order": order_result.get("order", {})}
        else:
            _raise_order_error(f"Order failed: {order_result.get('message', 'Unknown error')}")

    except HTTPException:
        raise
    except Exception as e:
        _audit_trade("live_market_buy_fail", {"symbol": symbol, "quote_usd": quote_usd, "error": str(e)}, success=False)
        raise HTTPException(status_code=502, detail=f"order failed: {e}") from e


@router.post("/execute_live_market_sell")
async def execute_live_market_sell(payload: dict[str, Any] = _request_body, _: None = _require_admin_dep) -> dict[str, Any]:
    """
    Execute a live market sell order on Binance.US.

    Requires admin authentication and valid API keys.
    """
    trading_service = _get_trading_service()
    symbol = str(payload.get("symbol") or "").upper()
    quantity = float(payload.get("quantity") or 0)

    if not symbol or quantity <= 0:
        raise HTTPException(status_code=400, detail="symbol and positive quantity required")

    try:
        # Use Binance API directly for market sell
        if trading_service.binance:
            pair = symbol.replace("USDT", "/USDT") if "USDT" in symbol else symbol + "/USDT"
            order = await asyncio.to_thread(
                trading_service.binance.create_order,
                symbol=pair,
                type="market",
                side="sell",
                amount=quantity,  # For sell, amount is the quantity to sell
                price=None,
            )
            order_result = {
                "status": "success",
                "order": {
                    "id": order.get("id"),
                    "symbol": order.get("symbol"),
                    "type": order.get("type"),
                    "side": order.get("side"),
                    "amount": order.get("amount"),
                    "price": order.get("price"),
                    "clientOrderId": order.get("clientOrderId"),
                    "status": order.get("status"),
                    "timestamp": order.get("timestamp"),
                },
            }
        else:
            order_result = {"status": "error", "message": "Binance client not available"}

        if order_result.get("status") == "success":
            _audit_trade("live_market_sell", {"symbol": symbol, "quantity": quantity})
            return {"status": "success", "order": order_result.get("order", {})}
        else:
            _raise_order_error(f"Order failed: {order_result.get('message', 'Unknown error')}")

    except HTTPException:
        raise
    except Exception as e:
        _audit_trade("live_market_sell_fail", {"symbol": symbol, "quantity": quantity, "error": str(e)}, success=False)
        raise HTTPException(status_code=502, detail=f"order failed: {e}") from e


def _raise_order_error(message: str) -> None:
    """Raise HTTPException for order errors."""
    raise HTTPException(status_code=502, detail=message)


def _audit_trade(event: str, details: dict[str, Any], success: bool = True) -> None:
    """Audit live trading events."""
    try:
        # Basic audit logging - could be enhanced with security middleware
        logger.info(f"LIVE TRADE AUDIT: {event} - Success: {success} - Details: {details}")
    except Exception as e:
        logger.warning(f"Failed to audit trade event: {e}")
