"""
Live Trading Endpoints — read-only Binance.US account inspection (admin).

Order placement is NOT supported here. All DAY execution goes through
portfolio_engine (execute_buy_fifo / execute_sell_fifo) in the external
integration process. See CANONICAL_SYSTEM.md.
"""

from __future__ import annotations

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
    """Retired: live orders must not bypass portfolio_engine gates."""
    raise HTTPException(
        status_code=410,
        detail=(
            "POST /api/trading/execute_live_market_buy is retired. "
            "Live DAY execution must go through portfolio_engine when LIVE mode is armed."
        ),
    )


@router.post("/execute_live_market_sell")
async def execute_live_market_sell(payload: dict[str, Any] = _request_body, _: None = _require_admin_dep) -> dict[str, Any]:
    """Retired: live orders must not bypass portfolio_engine gates."""
    raise HTTPException(
        status_code=410,
        detail=(
            "POST /api/trading/execute_live_market_sell is retired. "
            "Live DAY exits must go through portfolio_engine monitor loop."
        ),
    )


logger = logging.getLogger(__name__)
