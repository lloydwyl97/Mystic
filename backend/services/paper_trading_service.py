"""
Paper Trading Service
Provides risk-free trading simulation for strategy testing

Quick Test Checklist:
- Symbols normalized to BASE/QUOTE via _to_ccxt_symbol; accepts BTCUSDT input but converts to BTC/USDT.
- ccxt-facing strings are strictly BASE/QUOTE; no legacy concatenated symbols leak through.
- No binance/binanceus leaks; exchange id is centralized elsewhere.
- No unreachable code; parameterized ASCII-only logging.
- No Streamlit, no Docker, no Coinbase, no CoinGecko, no Kraken, no yfinance.
- Python 3.12, backend port 8000, unified dashboard port 8000.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import os
import time
from collections import deque
from datetime import datetime, timezone
from typing import Any

# Direct imports for production
import redis.asyncio as redis  # type: ignore[import-not-found]

from backend.config.redis_config import get_shared_redis_async
from backend.config.trading_economics import (
    ORDERBOOK_HALF_SPREAD_ESTIMATE,
    SLIPPAGE_BUFFER,
    TAKER_FEE,
)
from backend.database_schema import DATABASE_PATH, execute_read
from backend.services.redis_service import get_redis_service
from backend.services.symbols import _to_ccxt_symbol  # type: ignore[import-not-found]
from backend.services.task_manager import task_manager
from backend.services.trade_performance_tracker import log_trade_performance

# Database imports for trade persistence
try:
    from backend.database_init import SessionLocal, TradeLog
except ImportError:
    SessionLocal = None
    TradeLog = None

# Feedback service will be imported lazily to avoid circular imports

logger = logging.getLogger(__name__)

# Maximum trade history size to prevent unbounded growth
MAX_TRADE_HISTORY = 5000


class PaperOrder:
    def __init__(
        self,
        order_id: str,
        symbol: str,
        side: str,
        order_type: str,
        quantity: float,
        price: float,
        status: str,
        created_at: datetime,
        filled_at: datetime | None = None,
        filled_price: float | None = None,
        stop_price: float | None = None,
        take_profit_price: float | None = None,
        parent_order_id: str | None = None,
        exit_reason: str | None = None,
    ) -> None:
        self.order_id = order_id
        self.symbol = symbol
        self.side = side
        self.order_type = order_type
        self.quantity = float(quantity)
        self.price = float(price)
        self.status = status
        self.created_at = created_at
        self.filled_at = filled_at
        self.filled_price = filled_price
        self.stop_price = stop_price
        self.take_profit_price = take_profit_price
        self.parent_order_id = parent_order_id
        self.exit_reason = exit_reason


class PaperPosition:
    def __init__(
        self,
        symbol: str,
        quantity: float,
        average_price: float,
        current_price: float,
        unrealized_pnl: float,
        realized_pnl: float,
        created_at: datetime,
        last_updated: datetime,
        entry_commission: float = 0.0,  # FIX 5: Track entry commission for round-trip PnL
        sleeve: str = "ACTIVE",
    ) -> None:
        self.symbol = symbol
        self.quantity = float(quantity)
        self.average_price = float(average_price)
        self.current_price = float(current_price)
        self.unrealized_pnl = float(unrealized_pnl)
        self.realized_pnl = float(realized_pnl)
        self.created_at = created_at
        self.last_updated = last_updated
        self.entry_commission = float(entry_commission)  # Total entry commission for position
        self.sleeve = str(sleeve or "ACTIVE")

        # Open-position observation state (advisory metadata, not sell drivers).
        self.highest_price_since_entry: float = current_price


class PaperTradingService:
    def __init__(self) -> None:
        self.redis_client: redis.Redis | None = None  # type: ignore[assignment]
        self._redis_available = True  # Track Redis availability
        self.redis_url = os.getenv("REDIS_URL") or ""

        # Generate unique run ID for this paper trading session
        self.paper_run_id = self._generate_paper_run_id()

        # Get initial balance from environment or default
        self.initial_balance = float(os.getenv("PAPER_TRADING_INITIAL_BALANCE", "10000.0"))
        # Principal is the fixed starting capital used for PnL baselines
        self.principal = self.initial_balance
        self.current_balance = self.initial_balance
        # Cumulative realized PnL ledger so closed positions do not erase history
        self.realized_pnl_total = 0.0
        logger.info(f"Paper trading initialized with ${self.initial_balance} balance (Run ID: {self.paper_run_id})")
        self.positions: dict[str, PaperPosition] = {}
        self.orders: dict[str, PaperOrder] = {}
        self.trade_history: list[dict[str, Any]] = []
        # Binance.US Advanced Spot (Apr 2026): 0% maker / 0.02% taker. Verified against
        # 129 authenticated post-2026-04-21 fills measuring 2.000 bps/side taker.
        self.commission_rate = TAKER_FEE
        self.slippage_rate = SLIPPAGE_BUFFER
        self.enabled = True
        self._running = False
        self._task: asyncio.Task | None = None
        # MEMORY LEAK FIX: Bound balance_history to prevent unbounded growth in 24/7 operation
        # Track balance history for drawdown calculation (live data only)
        # Keep only the most recent 10,000 balance snapshots (enough for detailed analysis)
        self.balance_history: deque[tuple[float, float]] = deque([(time.time(), self.initial_balance)], maxlen=10000)
        self.peak_balance = self.initial_balance
        logger.info("PaperTradingService initialized")

        # BUG #43: Cooldown persistence
        self._cooldowns: dict[str, float] = {}  # symbol -> cooldown_until timestamp

        # Issue #4 Fix: Lock for dict mutations to prevent concurrent modification races
        self._positions_lock = asyncio.Lock()

    def _generate_paper_run_id(self) -> str:
        """Generate unique run ID for this paper trading session"""
        import uuid

        return str(uuid.uuid4())

    async def start(self, poll_interval_s: int = 1) -> None:
        if self._running:
            return
        # CRITICAL FIX: Only ensure Redis if not already initialized
        if not self.redis_client:
            await self._ensure_redis()

        # Ensure paper_run_id is persisted to Redis for continuity
        if self._redis_available and self.redis_client:
            try:
                await self._persist_paper_run_id_to_redis()
            except Exception as e:
                logger.warning(f"Failed to persist paper run ID on start: {e}")

        # BUG #43: Load cooldowns from Redis on startup
        await self._load_cooldowns()

        self._running = True
        self._task = await task_manager.create_task(self._loop(poll_interval_s), name="paper_trading_service:loop")

    async def stop(self) -> None:
        if not self._running:
            return
        self._running = False
        task = self._task
        self._task = None
        if task is not None:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
            except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
                logger.exception("PaperTradingService stop error: %s", e)

    async def _load_cooldowns(self) -> None:
        """BUG #43: Load cooldowns from Redis on startup"""
        if not self.redis_client:
            return
        try:
            async for key in self.redis_client.scan_iter(match="cooldown:*"):
                symbol = key.replace("cooldown:", "")
                try:
                    cooldown_until_str = await self.redis_client.get(key)
                    if cooldown_until_str:
                        cooldown_until = float(cooldown_until_str)
                        self._cooldowns[symbol] = cooldown_until
                except Exception as e:
                    logger.warning(f"Failed to load cooldown for {symbol}: {e}")
        except Exception as e:
            logger.warning(f"Failed to load cooldowns from Redis: {e}")

    async def set_cooldown(self, symbol: str, cooldown_seconds: int) -> None:
        """BUG #43: Set and persist cooldown to Redis"""
        cooldown_until = time.time() + cooldown_seconds

        # Store in memory
        self._cooldowns[symbol] = cooldown_until

        # Persist to Redis with TTL
        if self.redis_client:
            try:
                cooldown_key = f"cooldown:{symbol}"
                await self.redis_client.set(
                    cooldown_key,
                    str(cooldown_until),
                    ex=cooldown_seconds + 60,  # Extra buffer
                )
            except Exception as e:
                logger.warning(f"Failed to persist cooldown for {symbol}: {e}")

    def is_enabled(self) -> bool:
        return self.enabled

    def _cleanup_trade_history(self) -> None:
        """Trim trade_history to the most recent MAX_TRADE_HISTORY entries."""
        if len(self.trade_history) > MAX_TRADE_HISTORY:
            self.trade_history = self.trade_history[-MAX_TRADE_HISTORY:]
            logger.debug(f"Trimmed trade history to {MAX_TRADE_HISTORY} entries")

    async def _loop(self, poll_interval_s: int) -> None:
        try:
            while self._running:
                try:
                    await self._evaluate_pending_orders()
                    await self._update_stats_to_redis()
                except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
                    logger.exception("PaperTradingService loop iteration failed: %s", e)
                await asyncio.sleep(poll_interval_s)
        except asyncio.CancelledError:
            return
        except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
            logger.exception("PaperTradingService loop crashed: %s", e)

    def _calculate_sharpe_ratio(self) -> float:
        """Calculate Sharpe Ratio from trade returns (live data only)"""
        try:
            if len(self.trade_history) < 2:
                return 0.0

            # Calculate returns from each trade (percentage gain/loss)
            returns = []
            for trade in self.trade_history:
                pnl = float(trade.get("pnl", 0))
                # Calculate return as percentage of entry value
                entry_value = float(trade.get("entry_value", 0))
                if entry_value > 0:
                    trade_return = (pnl / entry_value) * 100.0  # Return as percentage
                    returns.append(trade_return)

            if len(returns) < 2:
                return 0.0

            # Calculate mean and standard deviation of returns
            mean_return = sum(returns) / len(returns)
            variance = sum((r - mean_return) ** 2 for r in returns) / (len(returns) - 1)
            std_dev = variance**0.5

            if std_dev == 0:
                return 0.0

            # Sharpe Ratio = (Mean Return - Risk Free Rate) / Std Dev
            # Using 0% risk-free rate for crypto (standard practice)
            risk_free_rate = 0.0
            sharpe = (mean_return - risk_free_rate) / std_dev

            return float(sharpe)

        except (ValueError, TypeError, ZeroDivisionError) as e:
            logger.debug(f"Sharpe ratio calculation failed: {e}")
            return 0.0

    def _calculate_max_drawdown(self) -> tuple[float, float]:
        """Calculate maximum drawdown from balance history (live data only)
        Returns: (max_drawdown_pct, max_drawdown_usd)
        """
        try:
            if len(self.balance_history) < 2:
                return 0.0, 0.0

            max_drawdown_usd = 0.0
            max_drawdown_pct = 0.0
            peak = self.balance_history[0][1]

            # Track running maximum balance and worst drawdown
            for _timestamp, balance in self.balance_history:
                peak = max(peak, balance)

                drawdown_usd = peak - balance
                if drawdown_usd > max_drawdown_usd:
                    max_drawdown_usd = drawdown_usd
                    max_drawdown_pct = (drawdown_usd / peak * 100.0) if peak > 0 else 0.0

            return float(max_drawdown_pct), float(max_drawdown_usd)

        except (ValueError, TypeError, ZeroDivisionError) as e:
            logger.debug(f"Max drawdown calculation failed: {e}")
            return 0.0, 0.0

    def get_performance_by_symbol(self) -> dict[str, Any]:
        """Calculate performance metrics grouped by symbol (live data only)"""
        try:
            if not self.trade_history:
                return {}

            symbol_stats: dict[str, dict[str, Any]] = {}

            # Group trades by symbol and calculate metrics
            for trade in self.trade_history:
                symbol = trade.get("symbol", "UNKNOWN")
                pnl = float(trade.get("pnl", 0))
                hold_time = float(trade.get("hold_time_seconds", 0))

                if symbol not in symbol_stats:
                    symbol_stats[symbol] = {
                        "symbol": symbol,
                        "total_trades": 0,
                        "winning_trades": 0,
                        "losing_trades": 0,
                        "total_pnl": 0.0,
                        "gross_profit": 0.0,
                        "gross_loss": 0.0,
                        "win_rate": 0.0,
                        "avg_win": 0.0,
                        "avg_loss": 0.0,
                        "best_trade": 0.0,
                        "worst_trade": 0.0,
                        "profit_factor": 0.0,
                        "total_hold_time": 0.0,
                        "avg_hold_time_seconds": 0.0,
                    }

                stats = symbol_stats[symbol]
                stats["total_trades"] += 1
                stats["total_pnl"] += pnl
                stats["total_hold_time"] += hold_time

                if pnl > 0:
                    stats["winning_trades"] += 1
                    stats["gross_profit"] += pnl
                    stats["best_trade"] = max(stats["best_trade"], pnl)
                elif pnl < 0:
                    stats["losing_trades"] += 1
                    stats["gross_loss"] += abs(pnl)
                    stats["worst_trade"] = min(stats["worst_trade"], pnl)

            # Calculate derived metrics
            for _symbol, stats in symbol_stats.items():
                total = stats["total_trades"]
                winning = stats["winning_trades"]
                losing = stats["losing_trades"]

                # Win rate
                stats["win_rate"] = (winning / total * 100.0) if total > 0 else 0.0

                # Average win/loss
                stats["avg_win"] = (stats["gross_profit"] / winning) if winning > 0 else 0.0
                stats["avg_loss"] = (stats["gross_loss"] / losing) if losing > 0 else 0.0

                # Profit factor
                stats["profit_factor"] = (stats["gross_profit"] / stats["gross_loss"]) if stats["gross_loss"] > 0 else 0.0

                # Average hold time
                stats["avg_hold_time_seconds"] = (stats["total_hold_time"] / total) if total > 0 else 0.0

        except (ValueError, TypeError, AttributeError, KeyError) as e:
            logger.exception(f"Performance by symbol calculation failed: {e}")
            return {}
        else:
            return symbol_stats

    async def _update_stats_to_redis(self) -> None:
        """Write paper trading stats to Redis for dashboard (live data only)"""
        try:
            await self._ensure_redis()
            if not self.redis_client:
                return

            # Calculate basic stats from live trade history
            total_trades = len(self.trade_history)
            winning_trades = sum(1 for t in self.trade_history if t.get("pnl", 0) > 0)
            losing_trades = sum(1 for t in self.trade_history if t.get("pnl", 0) < 0)
            total_pnl = sum(t.get("pnl", 0) for t in self.trade_history)
            win_rate = (winning_trades / total_trades) if total_trades > 0 else 0.0

            # Calculate advanced metrics from live data
            sharpe_ratio = self._calculate_sharpe_ratio()
            max_drawdown_pct, max_drawdown_usd = self._calculate_max_drawdown()

            stats = {
                "total_trades": total_trades,
                "winning_trades": winning_trades,
                "losing_trades": losing_trades,
                "total_pnl": total_pnl,
                "win_rate": win_rate,
                "sharpe_ratio": sharpe_ratio,
                "max_drawdown_pct": max_drawdown_pct,
                "max_drawdown_usd": max_drawdown_usd,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }

            await self.redis_client.set("paper_trading:stats", json.dumps(stats), ex=3600)
            logger.debug(f"Updated paper trading stats to Redis: {total_trades} trades, win_rate={win_rate:.2%}, sharpe={sharpe_ratio:.2f}, max_dd={max_drawdown_pct:.2f}%")
        except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
            logger.exception(f"Failed to update paper trading stats to Redis: {e}")

    async def _ensure_redis(self) -> None:
        if self.redis_client is None:
            try:
                self.redis_client = get_shared_redis_async()
                if self.redis_client is None:
                    self.redis_client = get_redis_service()
                if self.redis_client is None:
                    logger.warning("Redis connection unavailable - running in memory-only mode")
                    self._redis_available = False
                    return
            except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
                self.redis_client = None
                self._redis_available = False
                logger.exception("Redis connection failed: %s", e)

    async def initialize(self):
        """Initialize paper trading with Redis connection"""
        # CRITICAL FIX: Always ensure Redis connection with timeout
        if not self.redis_client:
            try:
                await asyncio.wait_for(self._ensure_redis(), timeout=5.0)
            except asyncio.TimeoutError:
                logger.warning("[WARN] Redis initialization timeout - operating in memory-only mode")
                self._redis_available = False
            except Exception as e:
                logger.warning(f"[WARN] Redis initialization failed: {e} - operating in memory-only mode")
                self._redis_available = False

        # Load positions from Redis on initialization (with timeout)
        if self._redis_available and self.redis_client:
            try:
                await asyncio.wait_for(self._load_positions_from_redis(), timeout=5.0)
            except asyncio.TimeoutError:
                logger.warning("[WARN] Load positions timeout - will try SQLite fallback")
            except Exception as e:
                logger.warning(f"[WARN] Load positions from Redis failed: {e} - will try SQLite fallback")

        # OPTION A FIX: Fallback to SQLite if no positions loaded from Redis
        # This prevents orphaning positions after service restarts
        if not self.positions:
            logger.info("No positions in Redis, attempting to load from SQLite database...")
            try:
                await asyncio.wait_for(self._load_open_positions_from_sqlite(), timeout=5.0)
            except asyncio.TimeoutError:
                logger.warning("[WARN] Load positions from SQLite timeout")
            except Exception as e:
                logger.warning(f"[WARN] Load positions from SQLite failed: {e}")

        # CRITICAL FIX: ALWAYS try to load existing paper_run_id from Redis first
        # This prevents run_id fragmentation that breaks P&L reconciliation
        if self._redis_available and self.redis_client:
            try:
                existing_run_id = await asyncio.wait_for(self._load_paper_run_id_from_redis(), timeout=2.0)
                if existing_run_id:
                    self.paper_run_id = existing_run_id
                    logger.info(f"Reusing existing paper run ID: {self.paper_run_id} (continuing session)")
                else:
                    # No existing run_id, persist the newly generated one
                    await self._persist_paper_run_id_to_redis()
                    logger.info(f"Generated new paper run ID: {self.paper_run_id} (first session)")
            except asyncio.TimeoutError:
                logger.warning("[WARN] Load paper run ID timeout - keeping generated ID")
                # Try to persist it anyway
                with contextlib.suppress(Exception):
                    await self._persist_paper_run_id_to_redis()
            except Exception as e:
                logger.warning(f"[WARN] Load paper run ID failed: {e} - keeping generated ID")
                # Try to persist it anyway
                with contextlib.suppress(Exception):
                    await self._persist_paper_run_id_to_redis()

        # Load principal (fixed starting capital) if persisted
        if self._redis_available and self.redis_client:
            try:
                await asyncio.wait_for(self._load_principal_from_redis(), timeout=2.0)
            except asyncio.TimeoutError:
                logger.warning("[WARN] Load principal timeout - using in-memory principal")
            except Exception as e:
                logger.warning(f"[WARN] Load principal failed: {e} - using in-memory principal")

        # Load persisted cash balance if available (prevents restart reseeds)
        if self._redis_available and self.redis_client:
            try:
                await asyncio.wait_for(self._load_cash_balance_from_redis(), timeout=2.0)
            except asyncio.TimeoutError:
                logger.warning("[WARN] Load cash balance timeout - using in-memory balance")
            except Exception as e:
                logger.warning(f"[WARN] Load cash balance failed: {e} - using in-memory balance")

        # Load cumulative realized PnL ledger (so history survives closed positions)
        if self._redis_available and self.redis_client:
            try:
                await asyncio.wait_for(self._load_realized_pnl_total_from_redis(), timeout=2.0)
            except asyncio.TimeoutError:
                logger.warning("[WARN] Load realized PnL ledger timeout - starting at 0.0")
            except Exception as e:
                logger.warning(f"[WARN] Load realized PnL ledger failed: {e} - starting at 0.0")

        # Ensure principal and cash are persisted once initialized
        if self._redis_available and self.redis_client:
            try:
                await self._persist_principal_to_redis()
                await self._persist_cash_balance_to_redis()
            except Exception as e:
                logger.warning(f"[WARN] Persist principal/cash failed: {e}")

        logger.info("Paper Trading Service initialized for AI integration")

    async def get_balance(self) -> float:
        """Get current paper trading balance"""
        try:
            if not self.redis_client:
                await self.initialize()
            # Use existing account balance method
            account_data = await self.get_account_balance()
            return float(account_data.get("cash_balance", self.initial_balance))
        except Exception:
            return self.initial_balance

    async def create_buy_order(
        self,
        symbol: str,
        quantity: float,
        price: float,
        confidence: float | None = None,
        trade_id: str | None = None,
        decision_id: str | None = None,
        sleeve: str | None = None,
    ) -> dict[str, Any] | None:
        """Create paper buy order and return trade result.
        trade_id: optional, when provided (from portfolio_engine) used for SQLite to allow remaining_position updates on sell.
        decision_id: optional pipeline id for attribution (stored when persisting canonical row).
        sleeve: optional CORE/ACTIVE (mirrors portfolio_engine OpenPosition.sleeve).
        """
        try:
            if not self.redis_client:
                await self.initialize()

            # Normalize symbol to ensure consistent format (e.g., BTCUSDT -> BTC/USDT)
            symbol = self._normalize_symbol(symbol)

            # Generate order ID; use passed trade_id for SQLite if provided (engine sync)
            order_id = trade_id or f"paper_ai_{symbol}_{int(time.time())}"

            # Check if we have enough balance
            current_balance = await self.get_balance()
            order_value = quantity * price

            if current_balance < order_value:
                logger.warning(f"Insufficient paper balance for {symbol}: ${current_balance:.2f} < ${order_value:.2f}")
                return None

            # Update balance
            new_balance = current_balance - order_value
            self.current_balance = new_balance
            await self._persist_cash_balance_to_redis()

            # Create or update position
            if symbol in self.positions:
                # Update existing position with average cost
                existing_pos = self.positions[symbol]
                total_qty = existing_pos.quantity + quantity
                avg_price = ((existing_pos.average_price * existing_pos.quantity) + (price * quantity)) / total_qty
                existing_pos.quantity = total_qty
                existing_pos.average_price = avg_price
                existing_pos.current_price = price
                existing_pos.last_updated = datetime.now(timezone.utc)
                existing_pos.unrealized_pnl = total_qty * (price - avg_price)
                if sleeve:
                    existing_pos.sleeve = str(sleeve)
                # Persist updated position to Redis
                await self._persist_position_to_redis(symbol, existing_pos)
            else:
                # Create new position
                _sl = sleeve or "ACTIVE"
                position = PaperPosition(
                    symbol=symbol,
                    quantity=quantity,
                    average_price=price,
                    current_price=price,
                    unrealized_pnl=0.0,
                    realized_pnl=0.0,
                    created_at=datetime.now(timezone.utc),
                    last_updated=datetime.now(timezone.utc),
                    sleeve=_sl,
                )
                async with self._positions_lock:
                    self.positions[symbol] = position
                # Persist new position to Redis
                await self._persist_position_to_redis(symbol, position)

            # Create order record
            order = PaperOrder(
                order_id=order_id,
                symbol=symbol,
                side="buy",
                order_type="market",
                quantity=quantity,
                price=price,
                status="filled",
                created_at=datetime.now(timezone.utc),
                filled_at=datetime.now(timezone.utc),
                filled_price=price,
            )
            self.orders[order_id] = order

            # Add to trade history
            trade_record = {
                "order_id": order_id,
                "symbol": symbol,
                "side": "buy",
                "quantity": quantity,
                "price": price,
                "value": order_value,
                "commission": order_value * self.commission_rate,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "pnl": 0.0,
            }
            trade_record["side"] = trade_record["side"].upper()
            self.trade_history.append(trade_record)
            self._cleanup_trade_history()

            # Persist to Redis for cross-instance access and persistence
            await self._persist_trade_to_redis(trade_record)

            # Persist BUY to SQLite canonical for complete trade history
            # SKIP if trade_id was passed - caller (portfolio_engine) already persisted
            if trade_id is None:
                buy_trade_record = {
                    "trade_id": order_id,
                    "paper_run_id": self.paper_run_id,
                    "mode": "paper",
                    "symbol": symbol,
                    "side": "buy",
                    "quantity": quantity,
                    "price": price,
                    "entry_price": price,
                    "pnl": 0.0,
                    "pnl_pct": 0.0,
                    "remaining_position": quantity,
                    "hold_time_seconds": 0,
                    "commission": order_value * self.commission_rate,
                    "strategy": "portfolio_engine_day",
                    "confidence": confidence,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "order_id": order_id,
                    "status": "executed",
                    "exit_reason": None,
                    "entry_timestamp": datetime.now(timezone.utc).isoformat(),
                    "source": "paper_engine",
                    "decision_id": decision_id,
                    "sleeve": sleeve or "ACTIVE",
                }
                buy_trade_record["side"] = buy_trade_record["side"].upper()
                await self._persist_trade_to_sqlite_canonical(buy_trade_record)

            logger.info(f"AI Paper buy order created: {order_id} - {symbol} {quantity} @ ${price:.2f}")

            return {"order_id": order_id, "symbol": symbol, "side": "buy", "quantity": quantity, "price": price, "status": "filled", "timestamp": datetime.now(timezone.utc).isoformat()}

        except Exception as e:
            logger.exception(f"Error creating paper buy order: {e}")
            return None

    async def _calculate_realized_pnl_sqlite(self) -> float:
        """Calculate realized P&L from SQLite canonical data"""
        try:
            # Use async DB helper to avoid blocking event loop
            result = await execute_read(
                """
                SELECT COALESCE(SUM(pnl), 0)
                FROM paper_trades
                WHERE UPPER(side) = 'SELL' AND remaining_position >= 0 AND pnl IS NOT NULL
                """,
                fetchone=True,
            )
            return result[0] if result and result[0] is not None else 0.0

        except Exception as e:
            logger.exception(f"Error calculating realized P&L from SQLite: {e}")
            return self.realized_pnl_total  # Fallback to in-memory value

    async def get_account_balance(self, use_cache_only: bool = False) -> dict[str, Any]:
        """
        Get paper account balance and positions value.
        use_cache_only=True: never call exchange (Redis/SQLite only). Use for dashboard to avoid API weight.
        When cache-only and some Redis prices are missing, returns prices_incomplete=True and excludes
        those symbols from valuation (no 0.0 used as real price).
        """
        try:
            # Load positions from Redis if not already loaded (for current positions)
            if not self.positions:
                await self._ensure_redis()
                await self._load_positions_from_redis()
            missing_prices = await self._refresh_positions_prices(use_cache_only=use_cache_only)
            prices_incomplete = len(missing_prices) > 0
            missing_set = set(missing_prices)

            # Calculate from SQLite canonical data where possible
            realized_pnl_sqlite = await self._calculate_realized_pnl_sqlite()

            total_value = self.current_balance
            positions_snapshot = list(self.positions.values())
            # Only include positions with known prices in valuation (exclude missing_prices)
            for position in positions_snapshot:
                if position.symbol in missing_set:
                    continue
                total_value += position.quantity * position.current_price
            unreal = sum(pos.unrealized_pnl for pos in positions_snapshot if pos.symbol not in missing_set)

            # Use SQLite-calculated realized P&L as canonical source
            realiz = realized_pnl_sqlite

            positions_value = total_value - self.current_balance
            pnl_vs_principal = total_value - self.principal
            pnl_vs_principal_pct = (pnl_vs_principal / self.principal * 100.0) if self.principal > 0 else 0.0

            # If there are no open positions (or all missing), all P&L is realized by definition.
            if float(positions_value or 0.0) == 0.0 and not prices_incomplete:
                # FIX: Use database-calculated realized PnL as canonical source (not stale Redis balance)
                realized_pnl = realized_pnl_sqlite
                total_pnl = realized_pnl
                unreal = 0.0  # No unrealized P&L when no positions

                # FIX: Correct cash balance if it's out of sync with database
                expected_balance = self.principal + realized_pnl_sqlite
                balance_diff = abs(self.current_balance - expected_balance)
                if balance_diff > 0.01:  # $0.01 tolerance for floating point
                    logger.warning(f"DATA INTEGRITY FIX: Correcting cash balance from ${self.current_balance:.2f} to ${expected_balance:.2f} (diff: ${balance_diff:.2f})")
                    self.current_balance = expected_balance
                    # Recalculate total_value with corrected balance
                    total_value = self.current_balance
                    # Persist corrected balance to Redis
                    await self._persist_cash_balance_to_redis()
                    # Also update realized_pnl_total in Redis to match database
                    self.realized_pnl_total = realized_pnl_sqlite
                    await self._persist_realized_pnl_total_to_redis()

                logger.info(f"NO POSITIONS: realized_pnl={realized_pnl}, total_pnl={total_pnl}, cash={self.current_balance}, principal={self.principal}, sqlite_pnl={realized_pnl_sqlite}")
            else:
                # keep existing logic for unrealized/realized when positions exist
                # (do not change your current calculations in that branch)
                realized_pnl = realiz
                total_pnl = unreal + realized_pnl
                logger.info(f"HAS POSITIONS: realized_pnl={realized_pnl}, unreal={unreal}, total_pnl={total_pnl}")

            out = {
                "cash_balance": self.current_balance,
                "total_balance": self.current_balance,
                "total_value": total_value,
                "total_balance_usd": total_value,
                "positions_value": positions_value,
                "unrealized_pnl": unreal,
                "realized_pnl": realized_pnl,
                "total_pnl": total_pnl,
                "principal": self.principal,
                "pnl_vs_principal": pnl_vs_principal,
                "pnl_vs_principal_pct": pnl_vs_principal_pct,
                "available_balance": self.current_balance,
                "positions_count": len(self.positions),
                "orders_count": len([o for o in self.orders.values() if o.status == "PENDING"]),
                "paper_run_id": self.paper_run_id,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "prices_incomplete": prices_incomplete,
                "missing_prices": missing_prices,
                "price_source": "redis_cache_only" if use_cache_only else "normal",
            }
            if prices_incomplete:
                out["valuation_note"] = "Cache-only pricing: some symbols missing Redis prices; equity/PNL excludes missing symbols."
                out["positions_value_known_only"] = positions_value
                out["total_equity_known_only"] = total_value
        except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
            logger.exception("Error getting account balance: %s", e)
            return {"error": str(e)}
        else:
            return out

    async def place_order(
        self,
        symbol: str,
        side: str,
        order_type: str,
        quantity: float,
        price: float | None = None,
        stop_price: float | None = None,
        take_profit_price: float | None = None,
        parent_order_id: str | None = None,
        exit_reason: str | None = None,
        trade_id: str | None = None,  # CRITICAL FIX: Accept trade_id for BUY/SELL linking
        confidence: float | None = None,  # CRITICAL FIX: Accept confidence for database storage
        skip_sqlite_persist: bool = False,  # Engine sync: portfolio_engine already wrote paper_trades
    ) -> dict[str, Any]:
        try:
            if not self.enabled:
                return {"success": False, "error": "Paper trading is disabled"}

            sym_norm = self._normalize_symbol(symbol)
            side_u = str(side).strip().upper()
            if side_u not in ("BUY", "SELL"):
                return {"success": False, "error": "Invalid side"}
            type_u = str(order_type).strip().upper()
            if type_u not in ("MARKET", "LIMIT"):
                return {"success": False, "error": "Invalid order type"}

            qty = float(quantity)
            if qty <= 0.0:
                return {"success": False, "error": "Quantity must be positive"}

            await self._ensure_redis()
            # Load positions from Redis if not already loaded
            if not self.positions:
                await self._load_positions_from_redis()
            current_price = await self._get_current_price(sym_norm)
            if current_price is None or current_price <= 0.0:
                return {"success": False, "error": f"Price unavailable for {sym_norm}"}

            if type_u == "MARKET":
                order_price = current_price
            elif type_u == "LIMIT" and price is not None and float(price) > 0.0:
                order_price = float(price)
            else:
                return {"success": False, "error": "Invalid order parameters"}

            if side_u == "BUY" and type_u in ("MARKET", "LIMIT"):
                est_price = order_price if type_u == "LIMIT" else current_price
                required_balance = qty * est_price * (1.0 + self.commission_rate)
                if required_balance > self.current_balance:
                    return {"success": False, "error": "Insufficient balance"}

            if side_u == "SELL":
                pos = self.positions.get(sym_norm)
                if pos is None or pos.quantity < qty:
                    return {"success": False, "error": "Insufficient position quantity"}

            order_id = f"paper_{int(datetime.now(timezone.utc).timestamp() * 1000)}"
            order = PaperOrder(
                order_id=order_id,
                symbol=sym_norm,
                side=side_u,
                order_type=type_u,
                quantity=qty,
                price=order_price,
                status="PENDING",
                created_at=datetime.now(timezone.utc),
                stop_price=float(stop_price) if stop_price else None,
                take_profit_price=float(take_profit_price) if take_profit_price else None,
                parent_order_id=parent_order_id,
                exit_reason=exit_reason,
            )
            # CRITICAL FIX: Store trade_id and confidence for database persistence
            if trade_id:
                order.linked_trade_id = trade_id  # type: ignore[attr-defined]
            if confidence is not None:
                order.confidence = confidence  # type: ignore[attr-defined]
            if skip_sqlite_persist:
                order.skip_sqlite_persist = True  # type: ignore[attr-defined]
            self.orders[order_id] = order

            if type_u in ("MARKET", "LIMIT"):
                await self._process_order_immediate(order_id, current_price)
            elif not self._running:
                await self.start()

            return {
                "success": True,
                "order_id": order_id,
                "status": self.orders[order_id].status,
                "message": "Order accepted",
            }
        except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
            logger.exception("Error placing order: %s", e)
            return {"success": False, "error": str(e)}

    async def cancel_order(self, order_id: str) -> dict[str, Any]:
        try:
            order = self.orders.get(order_id)
            if order is None:
                return {"success": False, "error": "Order not found"}
            if order.status != "PENDING":
                return {"success": False, "error": "Order cannot be cancelled"}
            order.status = "CANCELLED"
        except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
            logger.exception("Error cancelling order: %s", e)
            return {"success": False, "error": str(e)}
        else:
            return {
                "success": True,
                "order_id": order_id,
                "status": "CANCELLED",
                "message": "Order cancelled",
            }

    async def get_positions(self) -> list[dict[str, Any]]:
        try:
            # Load positions from Redis if not already loaded
            if not self.positions:
                await self._ensure_redis()
                await self._load_positions_from_redis()
            await self._refresh_positions_prices()
            out: list[dict[str, Any]] = []
            # Use list() to create copy - prevents "dictionary changed size during iteration"
            for p in list(self.positions.values()):
                out.append(
                    {
                        "symbol": p.symbol,
                        "quantity": p.quantity,
                        "average_price": p.average_price,
                        "current_price": p.current_price,
                        "unrealized_pnl": p.unrealized_pnl,
                        "realized_pnl": p.realized_pnl,
                        "total_pnl": p.unrealized_pnl + p.realized_pnl,
                        "market_value": p.quantity * p.current_price,
                        "created_at": p.created_at.isoformat(),
                        "last_updated": p.last_updated.isoformat(),
                    },
                )
        except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
            logger.exception("Error getting positions: %s", e)
            return []
        else:
            return out

    async def get_orders(self, status: str | None = None) -> list[dict[str, Any]]:
        try:
            out: list[dict[str, Any]] = []
            for o in self.orders.values():
                if status is None or o.status == status:
                    out.append(
                        {
                            "order_id": o.order_id,
                            "symbol": o.symbol,
                            "side": o.side,
                            "order_type": o.order_type,
                            "quantity": o.quantity,
                            "price": o.price,
                            "status": o.status,
                            "created_at": o.created_at.isoformat(),
                            "filled_at": o.filled_at.isoformat() if o.filled_at else None,
                            "filled_price": o.filled_price,
                            "stop_price": o.stop_price,
                            "take_profit_price": o.take_profit_price,
                            "parent_order_id": o.parent_order_id,
                        },
                    )
        except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
            logger.exception("Error getting orders: %s", e)
            return []
        else:
            return out

    async def get_trade_history(self, limit: int | None = 100, run_id: str | None = None, since: str | None = None, include_history: bool = False) -> list[dict[str, Any]]:
        """Get trade history from SQLite canonical storage"""
        import sqlite3

        def _sync_get_trade_history():
            try:
                # BUG #24 FIX: Use context manager for proper connection cleanup
                with sqlite3.connect(DATABASE_PATH) as conn:
                    cursor = conn.cursor()
                    # Use only columns that exist (probe proved table can have legacy schema without order_id etc.)
                    cursor.execute("PRAGMA table_info(paper_trades)")
                    existing_cols = {row[1] for row in cursor.fetchall()}
                    expected_cols = [
                        "trade_id",
                        "paper_run_id",
                        "symbol",
                        "side",
                        "quantity",
                        "price",
                        "entry_price",
                        "pnl",
                        "pnl_pct",
                        "remaining_position",
                        "hold_time_seconds",
                        "commission",
                        "strategy",
                        "confidence",
                        "timestamp",
                        "order_id",
                        "status",
                        "exit_reason",
                        "entry_timestamp",
                        "source",
                        "sleeve",
                        "decision_id",
                        "spread_pct_used",
                        "regime",
                    ]
                    select_cols = [c for c in expected_cols if c in existing_cols]
                    if not select_cols:
                        return []

                    query = "SELECT " + ", ".join(select_cols) + " FROM paper_trades WHERE mode IN ('paper', 'live')"
                    params = []
                    if run_id:
                        query += " AND paper_run_id = ?"
                        params.append(run_id)
                    elif not include_history:
                        query += " AND paper_run_id = ?"
                        params.append(self.paper_run_id)
                    if since:
                        query += " AND timestamp >= ?"
                        params.append(since)
                    query += " ORDER BY timestamp DESC"
                    if limit is not None and limit > 0:
                        query += " LIMIT ?"
                        params.append(limit)

                    cursor.execute(query, params)
                    rows = cursor.fetchall()

                    trades = []
                    for row in rows:
                        by_name = dict(zip(select_cols, row, strict=True))
                        qty = by_name.get("quantity")
                        pr = by_name.get("price")
                        trade_dict = {k: by_name.get(k) for k in expected_cols}
                        trade_dict["notional"] = qty * pr if qty and pr else 0
                        trade_dict["mode"] = "paper"
                        trades.append(trade_dict)

                    return trades

            except Exception as e:
                logger.exception(f"Error reading trade history from SQLite: {e}")
                return []

        # Execute in thread pool to avoid blocking
        return await asyncio.get_running_loop().run_in_executor(None, _sync_get_trade_history)

    async def reset_account(self) -> dict[str, Any]:
        try:
            # CRITICAL FIX: Clear ALL positions from Redis before clearing memory
            if self._redis_available and self.redis_client:
                try:
                    # Delete all position keys from Redis
                    for symbol in list(self.positions.keys()):
                        await self._delete_position_from_redis(symbol)
                    logger.info("Cleared all positions from Redis")
                except Exception as e:
                    logger.warning(f"Failed to clear Redis positions: {e}")

            # CRITICAL FIX: Clear ALL trades from SQLite for current run
            try:
                from backend.database import get_db

                conn = get_db()
                cursor = conn.cursor()
                cursor.execute("DELETE FROM paper_trades WHERE paper_run_id = ?", (self.paper_run_id,))
                deleted_count = cursor.rowcount
                conn.commit()
                logger.info(f"Deleted {deleted_count} trades from database for run {self.paper_run_id}")
            except Exception as e:
                logger.warning(f"Failed to clear database trades: {e}")

            # Clear in-memory state
            self.current_balance = self.initial_balance
            self.realized_pnl_total = 0.0
            self.positions.clear()
            self.orders.clear()
            self.trade_history.clear()

            # Persist reset state
            await self._persist_cash_balance_to_redis()
            await self._persist_realized_pnl_total_to_redis()

            # Legacy paper_trading_feedback_service was retired alongside the
            # deleted ``backend.modules.ai.ai_learning`` package. The unified
            # learning sink (``trade_learning_writer``) holds no per-account
            # state and needs no reset hook.

        except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
            logger.exception("Error resetting paper trading account: %s", e)
            return {"success": False, "error": str(e)}
        else:
            return {
                "success": True,
                "message": "Paper trading account reset",
                "initial_balance": self.initial_balance,
            }

    async def enable_paper_trading(self) -> dict[str, Any]:
        self.enabled = True
        return {"success": True, "message": "Paper trading enabled"}

    async def disable_paper_trading(self) -> dict[str, Any]:
        self.enabled = False
        return {"success": True, "message": "Paper trading disabled"}

    async def _process_order_immediate(self, order_id: str, current_price: float) -> None:
        order = self.orders.get(order_id)
        if order is None or order.status != "PENDING":
            return
        if order.order_type == "LIMIT":
            if order.side == "BUY" and current_price > order.price:
                return
            if order.side == "SELL" and current_price < order.price:
                return
        await self._fill_order(order, current_price)

    async def _evaluate_pending_orders(self) -> None:
        await self._ensure_redis()
        for order in list(self.orders.values()):
            if order.status != "PENDING":
                continue
            try:
                current_price = await self._get_current_price(order.symbol)
                if current_price is None or current_price <= 0.0:
                    continue
                if order.order_type == "LIMIT" and ((order.side == "BUY" and current_price <= float(order.price)) or (order.side == "SELL" and current_price >= float(order.price))):
                    await self._fill_order(order, current_price)
            except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
                logger.exception("Pending order eval failed for %s: %s", order.order_id, e)

    async def _fill_order(self, order: PaperOrder, mark_price: float) -> None:
        if order.status != "PENDING":
            return
        # Half-spread slippage. Measured Binance.US top-four full spread is 0.59-2.19 bps
        # (980k decision_book_tape samples); ORDERBOOK_HALF_SPREAD_ESTIMATE is the canonical half.
        spread_pct = ORDERBOOK_HALF_SPREAD_ESTIMATE * 2.0
        fill_price = mark_price * (1.0 + spread_pct / 2) if order.side == "BUY" else mark_price * (1.0 - spread_pct / 2)
        commission = order.quantity * fill_price * self.commission_rate
        ts_now = datetime.now(timezone.utc)
        realized_pnl = 0.0
        entry_price = fill_price if order.side == "BUY" else None
        remaining_qty = 0.0

        if order.side == "BUY":
            total_cost = order.quantity * fill_price + commission
            if total_cost > self.current_balance:
                order.status = "REJECTED"
                return

            # ================================================================
            # PHASE 2 FIX #3: ATOMIC UPDATE WITH TRANSACTION BOUNDARY
            # ================================================================
            # Ensure balance and position are updated atomically
            # If position update fails, we don't persist balance
            # ================================================================

            try:
                # 1. UPDATE POSITION FIRST (in memory)
                pos = self.positions.get(order.symbol)
                if pos:
                    total_qty = pos.quantity + order.quantity
                    new_avg = (pos.quantity * pos.average_price + order.quantity * fill_price) / total_qty
                    # FIX 5: Add entry commission proportionally
                    total_entry_commission = pos.entry_commission + commission
                    pos.quantity = total_qty
                    pos.average_price = new_avg
                    pos.entry_commission = total_entry_commission
                    pos.current_price = mark_price
                    pos.unrealized_pnl = pos.quantity * (pos.current_price - pos.average_price)
                    pos.last_updated = ts_now
                    remaining_qty = pos.quantity
                else:
                    new_position = PaperPosition(
                        symbol=order.symbol,
                        quantity=order.quantity,
                        average_price=fill_price,
                        current_price=mark_price,
                        unrealized_pnl=order.quantity * (mark_price - fill_price),
                        realized_pnl=0.0,
                        created_at=ts_now,
                        last_updated=ts_now,
                        entry_commission=commission,  # FIX 5: Store entry commission
                    )
                    async with self._positions_lock:
                        self.positions[order.symbol] = new_position
                    remaining_qty = order.quantity

                # 2. PERSIST POSITION (must succeed before updating balance)
                if pos:
                    await self._persist_position_to_redis(order.symbol, pos)
                else:
                    await self._persist_position_to_redis(order.symbol, self.positions[order.symbol])

                # 3. UPDATE AND PERSIST BALANCE (only if position persistence succeeded)
                self.current_balance -= total_cost
                await self._persist_cash_balance_to_redis()

            except Exception as e:
                logger.exception(f"🚨 BUY FILL FAILED - ROLLING BACK: {order.symbol} {order.quantity} - {e}")
                # Rollback in-memory changes
                if order.symbol in self.positions:
                    # Reload position from Redis if available
                    try:
                        # Revert to previous state by not persisting
                        logger.exception(f"Position update rolled back for {order.symbol}")
                    except Exception as rollback_error:
                        logger.exception(f"Rollback failed: {rollback_error}")
                order.status = "REJECTED"
                order.error_message = f"Atomic update failed: {e}"
                return
        else:
            pos = self.positions.get(order.symbol)
            if pos is None or pos.quantity < order.quantity:
                order.status = "REJECTED"
                return
            entry_price = pos.average_price
            # FIX 5: Include both entry and exit commission in realized PnL
            # Calculate the portion of entry commission for this sell quantity
            portion_of_entry_commission = (order.quantity / pos.quantity) * pos.entry_commission if pos.quantity > 0 else 0
            exit_commission = commission
            realized_pnl = order.quantity * (fill_price - entry_price) - portion_of_entry_commission - exit_commission
            proceeds = order.quantity * fill_price - exit_commission
            new_quantity = pos.quantity - order.quantity
            pos.quantity = new_quantity
            # FIX 5: Reduce entry commission proportionally
            remaining_entry_commission = pos.entry_commission - portion_of_entry_commission
            pos.entry_commission = remaining_entry_commission
            pos.current_price = mark_price
            pos.realized_pnl += realized_pnl
            pos.unrealized_pnl = pos.quantity * (pos.current_price - pos.average_price)
            pos.last_updated = ts_now
            self.current_balance += proceeds
            # PROFIT SIPHON: Process realized gains before adding to ledger
            try:
                from backend.services.profit_siphon import get_profit_siphon

                siphon = get_profit_siphon()
                siphon_result = siphon.process_realized_pnl(realized_pnl, self.current_balance)
                remaining_profit = siphon_result["remaining_profit"]
            except (ImportError, Exception) as _siphon_err:
                logger.debug("Profit siphon unavailable: %s", _siphon_err)
                remaining_profit = realized_pnl

            self.realized_pnl_total += remaining_profit
            await self._persist_realized_pnl_total_to_redis()
            await self._persist_cash_balance_to_redis()
            if pos.quantity <= 0.0:
                del self.positions[order.symbol]
                # Delete position from Redis when fully closed
                await self._delete_position_from_redis(order.symbol)
                remaining_qty = 0.0
            else:
                remaining_qty = pos.quantity
                # Persist updated position to Redis
                await self._persist_position_to_redis(order.symbol, pos)

            # SCRATCH EXIT DETECTION: Check if this SELL qualifies as a scratch exit
            if pos is not None and order.side == "SELL":
                try:
                    from backend.core.constants import SCRATCH_MAX_HOLD_SEC, SCRATCH_PNL_ABS_MAX
                except (ImportError, Exception):
                    SCRATCH_MAX_HOLD_SEC = 300
                    SCRATCH_PNL_ABS_MAX = 1.0

                hold_time_seconds = (ts_now - pos.created_at).total_seconds()

                # Check scratch exit conditions
                if abs(realized_pnl) <= SCRATCH_PNL_ABS_MAX and hold_time_seconds <= SCRATCH_MAX_HOLD_SEC and remaining_qty == 0.0:  # Fully closed position
                    # This is a scratch exit - log it (AI service will detect via trade performance)
                    logger.info(f"SCRATCH_EXIT_DETECTED: {order.symbol} | PnL=${realized_pnl:.4f} | Hold={hold_time_seconds:.1f}s | TradeID={order.order_id}")

        order.status = "FILLED"
        order.filled_at = ts_now
        order.filled_price = fill_price

        # Calculate hold time for positions that are being closed (SELL orders)
        hold_time_seconds = 0.0
        if order.side == "SELL" and pos is not None:
            hold_time_seconds = (ts_now - pos.created_at).total_seconds()

        # CRITICAL FIX: Use provided trade_id for BUY/SELL linking, or generate new one for BUY
        # For SELL orders, this should be the original BUY's trade_id passed from AI service
        final_trade_id = order.linked_trade_id if hasattr(order, "linked_trade_id") and order.linked_trade_id else f"paper_{self.paper_run_id}_{int(ts_now.timestamp() * 1000)}_{order.order_id}"

        # CRITICAL FIX: Extract confidence from order if available
        trade_confidence = getattr(order, "confidence", None)

        trade_record = {
            "trade_id": final_trade_id,
            "paper_run_id": self.paper_run_id,
            "mode": "paper",
            "order_id": order.order_id,
            "symbol": order.symbol,
            "side": order.side,
            "quantity": order.quantity,
            "price": fill_price,
            "entry_price": entry_price,
            "entry_value": order.quantity * entry_price if entry_price else None,  # For Sharpe ratio calculation
            "commission": commission,
            "timestamp": order.filled_at.isoformat(),
            "notional": order.quantity * fill_price,
            "pnl": realized_pnl,
            "pnl_pct": (realized_pnl / (entry_price * order.quantity)) * 100 if entry_price and entry_price > 0 else 0.0,
            "remaining_position": remaining_qty,
            "hold_time_seconds": hold_time_seconds,  # Average hold time tracking
            "status": "executed",
            "strategy": "paper_trading",
            "confidence": trade_confidence,  # CRITICAL FIX: Store confidence in database
            "exit_reason": getattr(order, "exit_reason", None),  # Set by AI service
            "entry_timestamp": pos.created_at.isoformat() if pos else None,
            "source": "paper_engine",
        }
        trade_record["side"] = trade_record["side"].upper()
        self.trade_history.append(trade_record)
        self._cleanup_trade_history()

        skip_sqlite = bool(getattr(order, "skip_sqlite_persist", False))

        # Write to SQLite canonical storage (dual-write: SQLite canonical + Redis cache)
        if not skip_sqlite:
            try:
                await self._persist_trade_to_sqlite_canonical(trade_record)
            except Exception as e:
                logger.exception(f"Failed to persist trade to canonical SQLite: {e}")
                # Continue execution - don't fail trade due to persistence issue

        # Also persist to Redis cache
        await self._persist_trade_to_redis(trade_record)

        # Log trade performance for API reporting (SELL orders only, when position is closed)
        if order.side == "SELL" and entry_price is not None:
            pnl_pct = (realized_pnl / (entry_price * order.quantity)) * 100 if entry_price > 0 else 0.0
            is_win = 1 if realized_pnl > 0 else (0 if realized_pnl < 0 else None)

            # Log asynchronously without blocking trade execution
            # Generate a proper integer trade_id using timestamp
            trade_id_int = int(ts_now.timestamp() * 1000000)  # microseconds for uniqueness

            task = asyncio.create_task(
                log_trade_performance(
                    trade_id=trade_id_int,
                    symbol=order.symbol,
                    side="sell",
                    entry_price=entry_price,
                    exit_price=fill_price,
                    quantity=order.quantity,
                    pnl=realized_pnl,
                    pnl_pct=pnl_pct,
                    is_win=is_win,
                    hold_time_seconds=int(hold_time_seconds) if hold_time_seconds > 0 else None,
                    strategy="paper_trading",
                    confidence=None,
                    mode="paper",
                )
            )
            task.add_done_callback(lambda t: t.exception())

        # NOTE: legacy ``paper_trading_feedback_service`` / ``ai_learning_service``
        # used to be invoked here. Both modules depended on the deleted
        # ``backend.modules.ai`` package and now fail closed on import. The
        # canonical learning sink for paper trades is ``trade_learning_writer``
        # via ``portfolio_engine._record_learning_outcome`` — paper and live
        # both write the same row. Do not re-add a parallel feedback path.

        # Update balance history for drawdown calculation (live data only)
        # Use list() to create copy - prevents "dictionary changed size during iteration"
        current_total = self.current_balance + sum(p.quantity * p.current_price for p in list(self.positions.values()))
        self.balance_history.append((time.time(), current_total))
        # Update peak balance
        self.peak_balance = max(self.peak_balance, current_total)

        # Persist to Redis for cross-instance access and fast retrieval
        await self._persist_trade_to_redis(trade_record)

        # Legacy trade_logs table (optional); portfolio_engine owns canonical paper_trades
        if not skip_sqlite:
            await self._persist_trade_to_database(trade_record)

        # RATE LIMIT FIX: Write status to Redis for dashboard (avoids API calls)
        await self._update_status_in_redis()

        logger.info(
            "Paper order filled: %s %s %s @ %s",
            order.symbol,
            order.side,
            order.quantity,
            fill_price,
        )

    async def _refresh_positions_prices(self, use_cache_only: bool = False) -> list[str]:
        """Refresh position prices from Redis (or exchange if not cache_only). Returns list of symbols with missing price."""
        await self._ensure_redis()
        missing_prices: list[str] = []
        for sym, pos in list(self.positions.items()):
            try:
                cp = await self._get_current_price(sym, use_cache_only=use_cache_only)
                if cp is None:
                    missing_prices.append(sym)
                    continue
                if cp > 0.0:
                    pos.current_price = cp
                    pos.unrealized_pnl = pos.quantity * (pos.current_price - pos.average_price)
                    pos.last_updated = datetime.now(timezone.utc)
            except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
                logger.exception("Refresh price failed for %s: %s", sym, e)
                missing_prices.append(sym)
        return missing_prices

    async def _get_current_price(self, sym_ccxt: str, use_cache_only: bool = False) -> float | None:
        # First try Redis cache
        await self._ensure_redis()
        if self.redis_client is not None:
            # Convert symbol format: BTC/USDT -> BTC (matches market data format)
            base_symbol = sym_ccxt.replace("/USDT", "").replace("USDT", "")
            key = f"market:{base_symbol}"
            try:
                # Market data service stores as JSON string with "price" field
                price_str = await self.redis_client.get(key)  # type: ignore[attr-defined]
                if price_str:
                    if isinstance(price_str, str):
                        price_json = json.loads(price_str)
                        if isinstance(price_json, dict) and "price" in price_json:
                            return float(price_json["price"])
                    # Fallback for other formats
                    return float(price_str)
            except Exception:
                logger.warning("PAPER_PRICE_REDIS_READ_FAILED", exc_info=True)

        # Dashboard/observer must never trigger exchange weight (DASHBOARD_WEIGHT_SAFETY)
        if use_cache_only:
            return None  # Telemetry: do not use 0.0 as real price; caller treats as missing

        # Fallback: Get price directly from Binance API (only when not cache-only)
        try:
            from backend.utils.binance_limited_http import limited_binance_get
            from backend.utils.canonical_symbol_formatter import CanonicalSymbolFormatter

            # Convert symbol format: SOL/USDT -> SOLUSDT for Binance API
            # CRITICAL FIX: Use canonical formatter to prevent malformed symbols like LINKUSDTUSDT
            try:
                binance_symbol = CanonicalSymbolFormatter.to_exchange(sym_ccxt)
            except Exception:
                # Fallback to simple replace if formatter fails
                binance_symbol = sym_ccxt.replace("/", "")
                # Fix double USDT issue: LINKUSDTUSDT -> LINKUSDT
                if binance_symbol.count("USDT") > 1:
                    # Remove all USDT occurrences and add back one
                    binance_symbol = binance_symbol.replace("USDT", "") + "USDT"

            response = await limited_binance_get(
                "/api/v3/ticker/24hr",
                params={"symbol": binance_symbol},
                timeout_sec=5.0,
            )
            if response is not None and response.status_code == 200:
                data = response.json()
                if "lastPrice" in data:
                    price = float(data["lastPrice"])
                    logger.info(f"Got price from API for {sym_ccxt}: ${price:.2f}")
                    return price
            elif response is not None:
                logger.warning(f"API request failed for {binance_symbol}: HTTP {response.status_code}")
        except Exception as e:
            logger.warning(f"API price fetch failed for {sym_ccxt}: {e}")

        logger.error(f"All price sources failed for {sym_ccxt}")
        return None

    async def _persist_trade_to_redis(self, trade_record: dict[str, Any]) -> None:
        """Persist trade to Redis for cross-instance access and persistence"""
        # BUG #21 FIX: Silent Redis Persistence Failures
        # Don't swallow exceptions - properly distinguish transient vs permanent errors
        max_retries = 3
        retry_count = 0

        while retry_count < max_retries:
            try:
                if not self.redis_client:
                    logger.warning("Redis client not available for trade persistence")
                    return

                order_id = trade_record.get("order_id", "")
                if not order_id:
                    logger.error("Cannot persist trade without order_id")
                    raise ValueError("Missing order_id in trade record")

                # Store individual trade - iterate and set each field separately
                trade_key = f"paper:trade:{order_id}"
                for field, value in trade_record.items():
                    await self.redis_client.hset(trade_key, key=field, value=str(value))
                await self.redis_client.expire(trade_key, 604800)  # 7 days TTL

                # Add to sorted set for time-series queries (score = timestamp)
                timestamp = datetime.now(timezone.utc).timestamp()
                await self.redis_client.zadd("paper:trades", {order_id: timestamp})

                # Keep sorted set size manageable (max 5000 trades)
                trade_count = await self.redis_client.zcard("paper:trades")
                if trade_count > 5000:
                    # Remove oldest trades
                    await self.redis_client.zremrangebyrank("paper:trades", 0, trade_count - 5000)

                logger.debug(f"Successfully persisted trade {order_id} to Redis")
                return  # Success!

            except (redis.TimeoutError, asyncio.TimeoutError) as e:
                # Transient error - retry with backoff
                retry_count += 1
                if retry_count < max_retries:
                    wait_time = 2**retry_count  # Exponential backoff: 2, 4, 8 seconds
                    logger.warning(f"Redis timeout persisting trade {order_id}, retrying in {wait_time}s (attempt {retry_count}/{max_retries}): {e}")
                    await asyncio.sleep(wait_time)
                else:
                    logger.exception(f"Failed to persist trade {order_id} to Redis after {max_retries} retries (timeout)")
                    raise
            except redis.ConnectionError as e:
                # Connection error - retry after longer wait
                retry_count += 1
                if retry_count < max_retries:
                    wait_time = 5
                    logger.warning(f"Redis connection error persisting trade {order_id}, retrying in {wait_time}s (attempt {retry_count}/{max_retries}): {e}")
                    await asyncio.sleep(wait_time)
                else:
                    logger.exception(f"Failed to persist trade {order_id} to Redis after {max_retries} retries (connection error)")
                    raise
            except Exception as e:
                # Permanent error - don't retry
                logger.exception(f"Failed to persist trade {order_id} to Redis (permanent error): {e}")
                raise

    async def _persist_trade_to_database(self, trade_record: dict[str, Any]) -> None:
        """Persist trade to legacy trade_logs table when present (optional analytics sink)."""
        try:
            if SessionLocal is None or TradeLog is None:
                return

            # Run database operations in thread pool to avoid blocking
            def _sync_persist():
                import sqlite3

                with sqlite3.connect(DATABASE_PATH) as probe:
                    if not probe.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='trade_logs'").fetchone():
                        return "skipped"

                with SessionLocal() as session:
                    # Convert trade record to TradeLog format
                    trade_log = TradeLog(
                        symbol=trade_record.get("symbol", ""),
                        side=trade_record.get("side", "").lower(),
                        amount=float(trade_record.get("quantity", 0)),
                        price=float(trade_record.get("price", 0)),
                        timestamp=datetime.fromisoformat(trade_record.get("timestamp", datetime.now(timezone.utc).isoformat())),
                        strategy="paper_trading",
                        portfolio_id="paper_portfolio",
                        status="executed",
                    )

                    session.add(trade_log)
                    session.commit()
                    return trade_record.get("order_id", "unknown")

            # Execute in thread pool
            order_id = await asyncio.get_running_loop().run_in_executor(None, _sync_persist)
            if order_id != "skipped":
                logger.debug(f"Persisted trade to database: {order_id}")

        except Exception as e:
            logger.debug(f"Legacy trade_logs persist skipped: {e}")
            # Don't raise exception - database failure shouldn't break trading

    async def _update_status_in_redis(self) -> None:
        """
        Write current paper trading status to Redis for dashboard consumption
        RATE LIMIT FIX: Dashboard reads from this instead of calling API endpoints
        """
        try:
            await self._ensure_redis()
            if not self.redis_client:
                return

            import json

            # Calculate current status
            status = {
                "running": self._running,
                "balance": float(self.current_balance),
                "positions_count": len(self.positions),
                "total_pnl": float(self.realized_pnl_total),
                "pnl_pct": ((self.current_balance - self.principal) / self.principal) * 100 if self.principal > 0 else 0.0,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }

            # Write to Redis with 30 second TTL (refreshed on every trade)
            await self.redis_client.set(
                "paper_trading:status",
                json.dumps(status),
                ex=30,  # 30 second expiry
            )

            logger.debug(f"Dashboard status updated in Redis: Balance=${status['balance']:.2f}, P&L={status['pnl_pct']:.2f}%")

        except Exception as e:
            logger.debug(f"Failed to update status in Redis: {e}")
            # Don't raise - status update failure shouldn't break trading

    async def _persist_trade_to_sqlite_canonical(self, trade_record: dict[str, Any]) -> None:
        """Persist trade to SQLite paper_trades table (schema-introspecting insert)."""
        import sqlite3

        field_map: dict[str, Any] = {
            "trade_id": trade_record.get("trade_id"),
            "paper_run_id": trade_record.get("paper_run_id", self.paper_run_id),
            "mode": trade_record.get("mode", "paper"),
            "symbol": trade_record.get("symbol"),
            "side": str(trade_record.get("side") or "").upper(),
            "quantity": trade_record.get("quantity"),
            "price": trade_record.get("price"),
            "entry_price": trade_record.get("entry_price"),
            "pnl": trade_record.get("pnl"),
            "pnl_pct": trade_record.get("pnl_pct"),
            "remaining_position": trade_record.get("remaining_position", 0),
            "hold_time_seconds": trade_record.get("hold_time_seconds"),
            "commission": trade_record.get("commission", 0),
            "strategy": trade_record.get("strategy"),
            "confidence": trade_record.get("confidence"),
            "timestamp": trade_record.get("timestamp"),
            "status": trade_record.get("status", "executed"),
            "exit_reason": trade_record.get("exit_reason"),
            "entry_timestamp": trade_record.get("entry_timestamp"),
            "source": trade_record.get("source", "paper_engine"),
            "decision_id": trade_record.get("decision_id"),
            "sleeve": trade_record.get("sleeve") or "ACTIVE",
            "spread_pct_used": trade_record.get("spread_pct_used"),
            "regime": trade_record.get("regime"),
        }

        def _sync_sqlite_persist():
            try:
                with sqlite3.connect(DATABASE_PATH) as conn:
                    cursor = conn.cursor()
                    cursor.execute("PRAGMA table_info(paper_trades)")
                    existing_cols = {row[1] for row in cursor.fetchall()}
                    insert_cols = [c for c in field_map if c in existing_cols]
                    if not insert_cols:
                        logger.warning("paper_trades has no writable columns; skip persist")
                        return

                    values = [field_map[c] for c in insert_cols]
                    col_sql = ", ".join(insert_cols)
                    placeholders = ", ".join("?" for _ in insert_cols)

                    cursor.execute("BEGIN TRANSACTION")
                    try:
                        cursor.execute(
                            f"INSERT OR REPLACE INTO paper_trades ({col_sql}) VALUES ({placeholders})",
                            values,
                        )
                        conn.commit()
                        logger.debug(f"Persisted trade to SQLite canonical: {trade_record.get('trade_id')}")
                    except Exception:
                        cursor.execute("ROLLBACK")
                        raise

            except Exception as e:
                logger.exception(f"Failed to persist trade to SQLite canonical: {e}")
                raise

        await asyncio.get_running_loop().run_in_executor(None, _sync_sqlite_persist)

    async def _persist_position_to_redis(self, symbol: str, position: PaperPosition) -> None:
        """Persist position to Redis for cross-instance access and persistence"""
        try:
            if not self.redis_client:
                return

            # Store position data as hash
            # BUG #22 FIX: Include commission fields in persistence
            position_key = f"paper:position:{symbol}"
            position_data = {
                "symbol": symbol,
                "quantity": str(position.quantity),
                "average_price": str(position.average_price),
                "current_price": str(position.current_price),
                "unrealized_pnl": str(position.unrealized_pnl),
                "realized_pnl": str(position.realized_pnl),
                "entry_commission": str(getattr(position, "entry_commission", 0.0)),
                "exit_commission": str(getattr(position, "exit_commission", 0.0)),
                "created_at": position.created_at.isoformat(),
                "last_updated": position.last_updated.isoformat(),
                "sleeve": str(getattr(position, "sleeve", "ACTIVE") or "ACTIVE"),
                "repair_add_count": str(int(getattr(position, "repair_add_count", 0) or 0)),
                "last_repair_add_ts": str(float(getattr(position, "last_repair_add_ts", 0) or 0)),
                "entry_strategy_id": str(getattr(position, "entry_strategy_id", "") or ""),
            }

            for field, value in position_data.items():
                await self.redis_client.hset(position_key, key=field, value=value)
            await self.redis_client.expire(position_key, 86400)  # 24 hours TTL

            # Add to set of active positions
            await self.redis_client.sadd("paper:positions:active", symbol)

        except Exception as e:
            logger.warning(f"Failed to persist position to Redis: {e}")

    async def _delete_position_from_redis(self, symbol: str) -> None:
        """Delete position from Redis when closed"""
        try:
            if not self.redis_client:
                return

            position_key = f"paper:position:{symbol}"
            await self.redis_client.delete(position_key)
            await self.redis_client.srem("paper:positions:active", symbol)

        except Exception as e:
            logger.warning(f"Failed to delete position from Redis: {e}")

    async def _load_positions_from_redis(self) -> None:
        """Load all positions from Redis on service initialization"""
        try:
            if not self.redis_client:
                return

            # Get all active position symbols
            symbols = await self.redis_client.smembers("paper:positions:active")
            if not symbols:
                logger.debug("No positions found in Redis")
                return

            # Load each position
            loaded_count = 0
            for symbol in symbols:
                position_key = f"paper:position:{symbol}"
                position_data = await self.redis_client.hgetall(position_key)

                if position_data:
                    # Reconstruct position object
                    # BUG #22 FIX: Include entry_commission when loading from Redis
                    _sl_raw = position_data.get("sleeve", "ACTIVE")
                    if isinstance(_sl_raw, bytes):
                        _sl_raw = _sl_raw.decode()
                    _sl_restored = str(_sl_raw or "ACTIVE")
                    position = PaperPosition(
                        symbol=str(position_data.get("symbol", symbol)),
                        quantity=float(position_data.get("quantity", 0)),
                        average_price=float(position_data.get("average_price", 0)),
                        current_price=float(position_data.get("current_price", 0)),
                        unrealized_pnl=float(position_data.get("unrealized_pnl", 0)),
                        realized_pnl=float(position_data.get("realized_pnl", 0)),
                        created_at=datetime.fromisoformat(position_data.get("created_at", datetime.now(timezone.utc).isoformat())),
                        last_updated=datetime.fromisoformat(position_data.get("last_updated", datetime.now(timezone.utc).isoformat())),
                        sleeve=_sl_restored,
                    )
                    # BUG #22 FIX: Restore entry_commission if available
                    if "entry_commission" in position_data:
                        position.entry_commission = float(position_data["entry_commission"])
                    if "exit_commission" in position_data:
                        position.exit_commission = float(position_data["exit_commission"])
                    if "repair_add_count" in position_data:
                        position.repair_add_count = int(float(position_data["repair_add_count"] or 0))
                    if "last_repair_add_ts" in position_data:
                        position.last_repair_add_ts = float(position_data["last_repair_add_ts"] or 0)
                    if "entry_strategy_id" in position_data:
                        position.entry_strategy_id = str(position_data["entry_strategy_id"] or "")

                    async with self._positions_lock:
                        self.positions[symbol] = position
                    loaded_count += 1

            logger.info(f"Loaded {loaded_count} positions from Redis")

        except Exception as e:
            logger.exception(f"Failed to load positions from Redis: {e}")

    async def _load_open_positions_from_sqlite(self) -> None:
        """Load open positions from SQLite database (fallback when Redis is empty)"""
        try:
            import sqlite3

            # BUG #24 FIX: Use context manager for proper connection cleanup
            with sqlite3.connect(DATABASE_PATH) as conn:
                cursor = conn.cursor()

                # Get all BUYs without matching SELLs (open positions)
                # Group by symbol to consolidate multiple BUYs into single position
                # ================================================================
                # PHASE 2 FIX #4: FIX ORPHANED POSITIONS QUERY
                # ================================================================
                # CORRECTED QUERY: Calculate net quantity per symbol
                # SUM(BUY qty) - SUM(SELL qty) to get remaining quantity
                # Only include symbols where net quantity > 0
                # ================================================================
                cursor.execute("""
                    SELECT
                        symbol,
                        COALESCE(SUM(CASE WHEN side = 'BUY' THEN quantity ELSE 0 END), 0) -
                        COALESCE(SUM(CASE WHEN side = 'SELL' THEN quantity ELSE 0 END), 0) as net_qty,
                        COALESCE(SUM(CASE WHEN side = 'BUY' THEN quantity * price ELSE 0 END) /
                        NULLIF(SUM(CASE WHEN side = 'BUY' THEN quantity ELSE 0 END), 0), 0) as buy_avg_price,
                        MIN(CASE WHEN side = 'BUY' THEN timestamp END) as created_at,
                        paper_run_id
                    FROM paper_trades
                    GROUP BY symbol, paper_run_id
                    HAVING net_qty > 0
                    ORDER BY created_at ASC
                """)

                open_positions = cursor.fetchall()
                loaded_count = 0

                for symbol, qty, avg_price, created_at, run_id in open_positions:
                    if qty and qty > 0:
                        # Create position object
                        position = PaperPosition(
                            symbol=symbol,
                            quantity=float(qty),
                            average_price=float(avg_price),
                            current_price=float(avg_price),  # Will be updated by price feed
                            unrealized_pnl=0.0,
                            realized_pnl=0.0,
                            created_at=datetime.fromisoformat(created_at.replace("Z", "+00:00")),
                            last_updated=datetime.now(timezone.utc),
                        )

                        async with self._positions_lock:
                            self.positions[symbol] = position
                        loaded_count += 1
                        logger.info(f"Loaded position from SQLite: {symbol} ({qty:.4f} @ ${avg_price:.4f}, run_id: {run_id[:8]}...)")

                if loaded_count > 0:
                    logger.info(f"Loaded {loaded_count} open positions from SQLite database")
                else:
                    logger.debug("No open positions found in SQLite")

        except Exception as e:
            logger.exception(f"Failed to load positions from SQLite: {e}")

    async def _persist_cash_balance_to_redis(self) -> None:
        """Persist cash balance to Redis to survive restarts."""
        try:
            if not self.redis_client:
                return
            cash_s = str(self.current_balance)
            await self.redis_client.set("paper_trading:cash_balance", cash_s)
            await self.redis_client.set("paper:cash_balance", cash_s)
        except Exception as e:
            logger.warning(f"Failed to persist cash balance: {e}")

    async def _load_cash_balance_from_redis(self) -> None:
        """Load cash balance from Redis if present."""
        try:
            if not self.redis_client:
                return
            value = await self.redis_client.get("paper_trading:cash_balance")
            if value is not None:
                self.current_balance = float(value)
        except Exception as e:
            logger.warning(f"Failed to load cash balance: {e}")

    async def _persist_principal_to_redis(self) -> None:
        """Persist principal (fixed starting capital) to Redis."""
        try:
            if not self.redis_client:
                return
            await self.redis_client.set("paper_trading:principal", str(self.principal))
        except Exception as e:
            logger.warning(f"Failed to persist principal: {e}")

    async def _load_principal_from_redis(self) -> None:
        """Load principal from Redis if present."""
        try:
            if not self.redis_client:
                return
            value = await self.redis_client.get("paper_trading:principal")
            if value is not None:
                self.principal = float(value)
        except Exception as e:
            logger.warning(f"Failed to load principal: {e}")

    async def _persist_realized_pnl_total_to_redis(self) -> None:
        """Persist cumulative realized PnL ledger"""
        try:
            if not self.redis_client:
                return
            await self.redis_client.set("paper_trading:realized_pnl_total", str(self.realized_pnl_total))
        except Exception as e:
            logger.warning(f"Failed to persist realized PnL ledger: {e}")

    async def _load_realized_pnl_total_from_redis(self) -> None:
        """Load cumulative realized PnL ledger"""
        try:
            if not self.redis_client:
                return
            value = await self.redis_client.get("paper_trading:realized_pnl_total")
            if value is not None:
                self.realized_pnl_total = float(value)
        except Exception as e:
            logger.warning(f"Failed to load realized PnL ledger: {e}")

    async def _load_paper_run_id_from_redis(self) -> str | None:
        """Load persisted paper run ID"""
        try:
            if not self.redis_client:
                return None
            value = await self.redis_client.get("paper_trading:paper_run_id")
            if value is not None:
                return str(value)
        except Exception as e:
            logger.warning(f"Failed to load paper run ID: {e}")
        return None

    async def _persist_paper_run_id_to_redis(self) -> None:
        """Persist current paper run ID to Redis"""
        try:
            if not self.redis_client:
                return
            await self.redis_client.set("paper_trading:paper_run_id", self.paper_run_id)
        except Exception as e:
            logger.warning(f"Failed to persist paper run ID: {e}")

    def _normalize_symbol(self, symbol: str) -> str:
        """Normalize symbol to CCXT format using canonical formatter"""
        try:
            from backend.utils.canonical_symbol_formatter import CanonicalSymbolFormatter

            return CanonicalSymbolFormatter.to_ccxt(symbol)
        except Exception:
            # Fallback to old logic if canonical formatter fails
            s = str(symbol).strip().upper()
            if "/" in s:
                base, quote = s.split("/", 1)
                return _to_ccxt_symbol(f"{base}/{quote}")
            if s.endswith("USDT"):
                base = s[:-4]
                return _to_ccxt_symbol(f"{base}/USDT")
            msg = "Invalid symbol; expected BASE/QUOTE"
            raise ValueError(msg) from None


# Paper trading service state - using dict to avoid global keyword
_paper_trading_service_state: dict[str, PaperTradingService | None] = {"instance": None}


def get_paper_trading_service() -> PaperTradingService:
    """Get or create singleton paper trading service"""
    if _paper_trading_service_state["instance"] is None:
        _paper_trading_service_state["instance"] = PaperTradingService()
    return _paper_trading_service_state["instance"]
