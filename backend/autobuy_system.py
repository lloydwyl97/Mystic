#!/usr/bin/env python3
"""
Autobuy System for Mystic Trading Platform

Handles automated buying of cryptocurrencies based on signals and strategies.
Now integrated with AI training, model versioning, and experimental services.
"""

import asyncio
import contextlib
import logging
import threading
import time
from datetime import datetime, timezone
from typing import Any

from backend.ai_training_pipeline import get_ai_training_pipeline
from backend.experimental_integration import get_experimental_integration
from backend.modules.ai.ai_model_versioning import get_ai_model_versioning
from backend.services.mystic_signal_engine import mystic_signal_engine
from backend.services.task_manager import task_manager
from backend.websocket_manager import websocket_manager

AI_MODULES_AVAILABLE = True

# Initialize logger first
logger = logging.getLogger(__name__)

# AutoBuy system timing constants
AUTOBUY_CHECK_INTERVAL = 5.0  # Check for autobuy opportunities every 5 seconds
AUTOBUY_ERROR_RECOVERY_DELAY = 10.0  # Delay after errors
AUTOBUY_REBALANCE_INTERVAL = 3600.0  # 1 hour rebalance interval
AUTOBUY_ERROR_REBALANCE_DELAY = 300.0  # 5 minutes delay on rebalance errors

# Ensure logging is configured
if not logger.handlers:
    # Check if root logger is configured
    root_logger = logging.getLogger()
    if not root_logger.handlers:
        # Configure root logger
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            handlers=[logging.StreamHandler()],
        )
    else:
        # Just add handler to this logger
        handler = logging.StreamHandler()
        formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)

    class SafeExperimentalIntegration:
        def __init__(self) -> None:
            self.status = "degraded"
            self.last_run_time = None

        async def start(self):
            self.last_run_time = time.time()
            logger.info("[OK] Experimental Integration started (degraded mode)")

        async def stop(self):
            logger.info("[OK] Experimental Integration stopped (degraded mode)")

        async def get_experimental_influence(self, _symbol):
            self.last_run_time = time.time()
            return {
                "signal_type": "hold",
                "confidence": 0.0,  # No confidence in degraded mode
                "strength": 0.0,  # No strength in degraded mode
                "reasoning": "Degraded mode - no experimental influence available",
            }

        def get_status(self):
            return {
                "status": "degraded",
                "last_run_time": self.last_run_time,
                "message": "Experimental Integration running in degraded mode",
            }

    class SafeAIModelVersioning:
        def __init__(self) -> None:
            self.status = "degraded"
            self.active_model = None
            self.last_run_time = None

        async def auto_optimize_models(self):
            self.last_run_time = time.time()
            return {
                "actions_taken": [],
                "message": "Model optimization not available in degraded mode",
            }

        async def update_model_performance(self, model, _data):
            self.last_run_time = time.time()
            logger.debug(f"Model performance update (degraded mode): {model}")

        async def evaluate_model_performance(self, _model):
            self.last_run_time = time.time()
            return {
                "recommendation": "hold",
                "message": "Model evaluation not available in degraded mode",
            }

        def get_status(self):
            return {
                "status": "degraded",
                "active_model": self.active_model,
                "last_run_time": self.last_run_time,
                "message": "AI Model Versioning running in degraded mode",
            }

    # Functions are imported from their respective modules above


CORE_MODULES_AVAILABLE = True


class TradeOrder:
    """Trade order representation"""

    def __init__(
        self,
        symbol: str,
        side: str,
        amount: float,
        price: float,
        confidence: float,
        mystic_factors: dict[str, Any] | None = None,
    ) -> None:
        self.symbol = symbol
        self.side = side  # 'buy' or 'sell'
        self.amount = amount  # base asset amount
        self.price = price  # price per base asset in quote currency (e.g., USD)
        self.confidence = confidence
        self.mystic_factors = mystic_factors or {}
        self.status = "pending"
        self.timestamp = time.time()
        self.execution_price: float | None = None
        # quote_amount is price * base amount (USD amount)
        self.quote_amount: float | None = None
        self.base_amount: float = amount

    def mark_executed(self, execution_price: float) -> None:
        self.execution_price = execution_price
        self.quote_amount = execution_price * self.base_amount
        self.status = "executed"
        self.timestamp = time.time()

    def to_dict(self) -> dict[str, Any]:
        return {
            "symbol": self.symbol,
            "side": self.side,
            "amount": self.amount,
            "price": self.price,
            "confidence": self.confidence,
            "mystic_factors": self.mystic_factors,
            "status": self.status,
            "timestamp": datetime.fromtimestamp(self.timestamp, tz=timezone.utc).isoformat(),
            "execution_price": self.execution_price,
            "quote_amount": self.quote_amount,
        }


class AutobuySystem:
    """Simplified AutobuySystem implementation that preserves public API"""

    def __init__(self, cache: Any, strategy_manager: Any) -> None:
        self.cache = cache
        self.strategy_manager = strategy_manager
        self.is_running = False
        self.pending_orders: dict[str, TradeOrder] = {}
        self.executed_orders: list[TradeOrder] = []
        self.total_trades = 0
        self.total_profit = 0.0
        self.last_rebalance_time = 0.0
        # AI and experimental integrations (safe fallbacks available above)
        try:
            self.ai_model_versioning = get_ai_model_versioning()
        except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError):
            self.ai_model_versioning = None

        try:
            self.ai_training_pipeline = get_ai_training_pipeline(cache)
        except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError):
            self.ai_training_pipeline = None

        try:
            self.experimental_integration = get_experimental_integration()
        except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError):
            self.experimental_integration = None

        # Websocket and mystic engine may be callables returning instances in fallback
        try:
            ws = websocket_manager if "websocket_manager" in globals() else None
            self.websocket = ws() if callable(ws) else ws
        except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError):
            self.websocket = None

        try:
            me = mystic_signal_engine if "mystic_signal_engine" in globals() else None
            self.mystic_engine = me() if callable(me) else me
        except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError):
            self.mystic_engine = None

    async def start(self) -> None:
        """Start the autobuy loop"""
        if self.is_running:
            logger.warning("AutobuySystem already running")
            return

        logger.info("[OK] AutobuySystem starting")
        self.is_running = True

        # Start AI and experimental subsystems if available
        try:
            if hasattr(self, "ai_training_pipeline") and self.ai_training_pipeline:
                if asyncio.iscoroutinefunction(self.ai_training_pipeline.start):
                    await self.ai_training_pipeline.start()
                else:
                    # call sync start if present
                    maybe = getattr(self.ai_training_pipeline, "start", None)
                    if callable(maybe):
                        maybe()
        except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
            logger.exception(f"[ERROR] Error starting AI training pipeline: {e}")

        try:
            if hasattr(self, "experimental_integration") and self.experimental_integration and hasattr(self.experimental_integration, "start"):
                if asyncio.iscoroutinefunction(self.experimental_integration.start):
                    await self.experimental_integration.start()
                else:
                    maybe = getattr(self.experimental_integration, "start", None)
                    if callable(maybe):
                        maybe()
        except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
            logger.exception(f"[ERROR] Error starting experimental integration: {e}")

        # Main loop
        try:
            while self.is_running:
                try:
                    await self._run_cycle()
                except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as inner:
                    logger.exception(f"[ERROR] Error during autobuy cycle: {inner}")
                    # recovery delay
                    await asyncio.sleep(AUTOBUY_ERROR_RECOVERY_DELAY)
                await asyncio.sleep(AUTOBUY_CHECK_INTERVAL)
        finally:
            logger.info("[OK] AutobuySystem stopped main loop")
            self.is_running = False

    async def stop(self) -> None:
        """Stop the autobuy loop"""
        if not self.is_running:
            logger.info("AutobuySystem not running")
            return

        logger.info("[OK] Stopping AutobuySystem")
        self.is_running = False

        # attempt to stop AI/experimental components gracefully
        try:
            if hasattr(self, "ai_training_pipeline") and self.ai_training_pipeline and hasattr(self.ai_training_pipeline, "stop"):
                maybe_stop = self.ai_training_pipeline.stop
                if asyncio.iscoroutinefunction(maybe_stop):
                    await maybe_stop()
                else:
                    maybe_stop()
        except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
            logger.exception(f"[ERROR] Error stopping AI training pipeline: {e}")

        try:
            if hasattr(self, "experimental_integration") and self.experimental_integration and hasattr(self.experimental_integration, "stop"):
                maybe_stop = self.experimental_integration.stop
                if asyncio.iscoroutinefunction(maybe_stop):
                    await maybe_stop()
                else:
                    maybe_stop()
        except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
            logger.exception(f"[ERROR] Error stopping experimental integration: {e}")

    async def _run_cycle(self) -> None:
        """Perform a single check/execution cycle"""
        # Run strategies (if available)
        try:
            strategy_results = self.strategy_manager.run_all_strategies() if self.strategy_manager and hasattr(self.strategy_manager, "run_all_strategies") else {}
        except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
            logger.exception(f"[ERROR] Error running strategies: {e}")
            strategy_results = {}

        # Attempt to obtain a mystic signal if available
        # mystic_signal = None  # Commented out unused variable
        try:
            if self.mystic_engine and hasattr(self.mystic_engine, "generate_comprehensive_signal"):
                maybe = self.mystic_engine.generate_comprehensive_signal
                if asyncio.iscoroutinefunction(maybe):
                    # mystic_signal = await maybe()  # Commented out unused variable
                    await maybe()
                else:
                    # mystic_signal = maybe()  # Commented out unused variable
                    maybe()
        except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
            logger.debug(f"[DEBUG] No mystic signal available: {e}")
            # mystic_signal = None  # Commented out unused variable

        # Experimental influence may be requested per-symbol in a fuller impl.
        # experimental_influence = None  # Commented out unused variable
        try:
            if self.experimental_integration and hasattr(self.experimental_integration, "get_experimental_influence"):
                maybe = self.experimental_integration.get_experimental_influence
                # Not awaiting since we don't know symbol; safe call omitted
                # In a fuller impl we'd iterate symbols and call per-symbol
                # experimental_influence = {"status": "available"}  # Commented out unused variable
        except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError):
            # experimental_influence = None  # Commented out unused variable
            pass

        # In a full system, we'd analyze signals and place orders. Here we simply log.
        logger.debug(f"[DEBUG] Ran strategies: {len(strategy_results) if strategy_results is not None else 0} results")

        # Periodic rebalance/maintenance
        now = time.time()
        if now - self.last_rebalance_time > AUTOBUY_REBALANCE_INTERVAL:
            try:
                await self._rebalance()
                self.last_rebalance_time = now
            except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
                logger.exception(f"[ERROR] Rebalance error: {e}")
                await asyncio.sleep(AUTOBUY_ERROR_REBALANCE_DELAY)

    async def _rebalance(self) -> None:
        """Placeholder for rebalancing logic"""
        logger.info("[OK] Rebalance check executed (placeholder)")

    def place_order(
        self,
        symbol: str,
        side: str,
        amount: float,
        price: float,
        confidence: float = 0.0,
        mystic_factors: dict[str, Any] | None = None,
    ) -> TradeOrder:
        """Place an order synchronously (simulated immediate execution)"""
        order = TradeOrder(symbol, side, amount, price, confidence, mystic_factors)
        self.pending_orders[symbol] = order
        # Simulate immediate execution for simplicity
        try:
            self._execute_order(order)
        except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
            logger.exception(f"[ERROR] Failed to execute order for {symbol}: {e}")
        return order

    def _execute_order(self, order: TradeOrder) -> None:
        """Execute order immediately (simulation) and update stats"""
        execution_price = order.price
        order.mark_executed(execution_price)

        # Update profit/trades according to simplified logic consistent with drawdown calc
        if order.side == "buy":
            # USD spent reduces profit
            self.total_profit -= order.quote_amount or 0.0
        else:  # sell
            self.total_profit += order.quote_amount or 0.0

        self.total_trades += 1
        # Move to executed orders
        self.executed_orders.append(order)
        # Remove from pending if present
        self.pending_orders.pop(order.symbol, None)

        logger.info(f"[OK] Executed order: {order.symbol} {order.side} {order.base_amount}@{order.execution_price}")

    def get_recent_orders(self, limit: int = 10) -> list[dict[str, Any]]:
        """Return recent executed orders as dicts"""
        recent = self.executed_orders[-limit:]
        return [o.to_dict() for o in reversed(recent)]

    def get_trading_stats(self) -> dict[str, Any]:
        """Return aggregate trading statistics"""
        total_trades = self.total_trades
        total_profit = self.total_profit

        # Simplified win rate: count sells as wins (placeholder)
        wins = len([o for o in self.executed_orders if o.side == "sell" and (o.quote_amount or 0) > 0])
        win_rate = (wins / max(total_trades, 1)) * 100 if total_trades > 0 else 0.0

        average_profit = (total_profit / max(total_trades, 1)) if total_trades > 0 else 0.0

        return {
            "total_trades": total_trades,
            "total_profit": total_profit,
            "win_rate": win_rate,
            "average_profit": average_profit,
            "pending_orders": len(self.pending_orders),
            "executed_orders": len(self.executed_orders),
        }

    def cancel_pending_order(self, symbol: str) -> bool:
        """Cancel a single pending order"""
        if symbol in self.pending_orders:
            self.pending_orders.pop(symbol, None)
            logger.info(f"[OK] Cancelled pending order for {symbol}")
            return True
        logger.debug(f"[DEBUG] No pending order to cancel for {symbol}")
        return False

    def cancel_all_pending_orders(self) -> None:
        count = len(self.pending_orders)
        self.pending_orders.clear()
        logger.info(f"[OK] Cancelled all pending orders ({count})")

    def _calculate_max_drawdown(self) -> float:
        """Calculate maximum drawdown from executed trades"""
        try:
            if not self.executed_orders:
                return 0.0

            # Calculate cumulative profit over time
            cumulative_profits = []
            running_profit = 0.0

            for order in sorted(self.executed_orders, key=lambda x: x.timestamp):
                if order.status == "executed" and order.execution_price:
                    # Simplified profit calculation using base quantity consistently
                    if order.side == "buy":
                        running_profit -= order.quote_amount or 0.0  # USD amount spent
                    else:  # sell
                        running_profit += order.quote_amount or 0.0  # USD amount received

                    cumulative_profits.append(running_profit)

            if not cumulative_profits:
                return 0.0

            # Calculate maximum drawdown
            peak = cumulative_profits[0]
            max_drawdown = 0.0

            for profit in cumulative_profits:
                peak = max(peak, profit)
                drawdown = (peak - profit) / max(peak, 1.0)
                max_drawdown = max(max_drawdown, drawdown)
        except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
            logger.exception(f"[ERROR] Error calculating max drawdown: {e}")
            return 0.0
        else:
            return max_drawdown


class AutobuyManager:
    """Manages the autobuy system"""

    def __init__(self, cache: Any, strategy_manager: Any) -> None:
        self.cache = cache
        self.strategy_manager = strategy_manager
        self.autobuy_system = AutobuySystem(cache, strategy_manager)
        self.task: asyncio.Task | None = None

    async def start(self):
        """Start the autobuy manager"""
        if self.task and not self.task.done():
            logger.warning("Autobuy system already running")
            return

        self.task = await task_manager.create_task(self.autobuy_system.start(), name="autobuy_system:start")
        logger.info("Autobuy manager started")

    async def stop(self):
        """Stop the autobuy manager"""
        if self.task:
            await self.autobuy_system.stop()
            self.task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self.task
            self.task = None
            logger.info("Autobuy manager stopped")

    def get_status(self) -> dict[str, Any]:
        """Get autobuy system status"""
        return {
            "is_running": self.autobuy_system.is_running,
            "task_running": self.task is not None and not self.task.done(),
            "trading_stats": self.autobuy_system.get_trading_stats(),
        }

    def get_recent_orders(self, limit: int = 10) -> list[dict[str, Any]]:
        """Get recent trade orders"""
        return self.autobuy_system.get_recent_orders(limit)

    def cancel_pending_order(self, symbol: str) -> bool:
        """Cancel a pending order"""
        return self.autobuy_system.cancel_pending_order(symbol)

    def cancel_all_pending_orders(self):
        """Cancel all pending orders"""
        self.autobuy_system.cancel_all_pending_orders()

    def start_trading(self) -> None:
        """Start trading (sync wrapper for API compatibility)"""
        try:
            # Check if we're in an async context
            try:
                loop = asyncio.get_running_loop()
                if not self.task or self.task.done():
                    self.task = loop.create_task(self.start())
                    logger.info("[OK] Autobuy system started via sync wrapper")
                else:
                    logger.warning("[WARNING] Autobuy system already running")
            except RuntimeError:
                # No running loop, create one
                def run_in_thread():
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    try:
                        if not self.task or self.task.done():
                            self.task = loop.create_task(self.start())
                            loop.run_until_complete(self.task)
                    finally:
                        loop.close()

                thread = threading.Thread(target=run_in_thread, daemon=True)
                thread.start()
                logger.info("[OK] Autobuy system started in background thread")
        except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
            logger.exception(f"[ERROR] Failed to start trading: {e}")

    def stop_trading(self) -> None:
        """Stop trading (sync wrapper for API compatibility)"""
        try:
            if self.task and not self.task.done():
                # Schedule the stop task
                try:
                    loop = asyncio.get_running_loop()
                    task = loop.create_task(self.stop())
                    if not hasattr(self, "_tasks"):
                        self._tasks: list[asyncio.Task[Any]] = []
                    self._tasks.append(task)
                    logger.info("[OK] Autobuy system stop requested")
                except RuntimeError:
                    # No running loop, just mark as stopped
                    self.autobuy_system.is_running = False
                    logger.info("[OK] Autobuy system marked as stopped")
            else:
                logger.warning("[WARNING] Autobuy system not running")
        except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
            logger.exception(f"[ERROR] Failed to stop trading: {e}")

    def get_trading_info(self) -> dict[str, Any]:
        """Get trading information for API compatibility"""
        return {
            "active": self.autobuy_system.is_running,
            "status": self.get_status(),
            "stats": self.autobuy_system.get_trading_stats(),
        }

    def cancel_orders(self, symbol: str) -> dict[str, Any]:
        """Cancel orders for a specific symbol (API compatibility method)"""
        success = self.autobuy_system.cancel_pending_order(symbol)
        return {
            "success": success,
            "symbol": symbol,
            "message": f"Order cancelled for {symbol}" if success else f"No pending order found for {symbol}",
        }

    def cancel_all_orders(self) -> dict[str, Any]:
        """Cancel all orders (API compatibility method)"""
        symbols_before = list(self.autobuy_system.pending_orders.keys())
        self.autobuy_system.cancel_all_pending_orders()
        symbols_after = list(self.autobuy_system.pending_orders.keys())

        successful_symbols = [s for s in symbols_before if s not in symbols_after]
        failed_symbols = [s for s in symbols_before if s in symbols_after]

        return {
            "success": len(failed_symbols) == 0,
            "successful_symbols": successful_symbols,
            "failed_symbols": failed_symbols,
            "total_cancelled": len(successful_symbols),
            "message": f"Cancelled {len(successful_symbols)} orders",
        }
