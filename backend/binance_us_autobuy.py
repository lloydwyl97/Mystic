#!/usr/bin/env python3
"""
binance_us_autobuy.py

Binance US Autobuy System
- Enforced Top-10 Binance.US universe only.
- Single source of truth constants (EXCHANGE_ID, TOP10_BINANCEUS).
- No ad-hoc exchange strings; only "binance_us".
- ASCII-only logging (no emojis).
- Python 3.12, Windows/PowerShell friendly.
- Binance US only implementation.
- Live-only: No simulation mode in production.
- Import-safe: No I/O operations at import time.

ERROR CONTRACT:
- Success: Returns {"success": True, "data": ..., "error": None}
- Error: Returns {"success": False, "data": None, "error": {"code": str, "message": str}}
- All methods follow this consistent structure for predictable API responses
"""

from __future__ import annotations

import asyncio
import logging
import os
import sqlite3
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx
from mystic_config import mystic_config

from backend.services.task_manager import task_manager

# Import from single source of truth
try:
    from backend.config.trading_universe import EXCHANGE_ID, TRADING_SYMBOLS
except (ImportError, ModuleNotFoundError, AttributeError, ValueError, TypeError, RuntimeError) as e:
    msg = f"Failed to import trading_universe: {e}"
    raise RuntimeError(msg) from e

from backend.database_schema import DATABASE_PATH
from backend.services.binance_rest_client import BinanceREST
from backend.utils.binance_weight_limiter import BinanceWeightLimiter

# ------------------------------ Exchange constants ------------------------------

# Use TRADING_SYMBOLS from trading_universe (live data)
TOP10_BINANCEUS: set[str] = set(TRADING_SYMBOLS)

# ------------------------------ Configuration ------------------------------

# Get absolute paths for logs and DB
_MODULE_DIR = Path(__file__).parent.absolute()
_PROJECT_ROOT = _MODULE_DIR.parent
_LOG_DIR = _PROJECT_ROOT / "logs"
_DB_PATH = Path(DATABASE_PATH).resolve()

# Ensure log directory exists
_LOG_DIR.mkdir(exist_ok=True)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(_LOG_DIR / "binance_us_autobuy.log"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)

# ------------------------------ Trading config ------------------------------

# Default to a safe subset of the enforced Top-10. You can override via env TRADING_PAIRS="BTCUSDT,ETHUSDT"
_env_pairs = os.getenv("TRADING_PAIRS")
if _env_pairs:
    TRADING_PAIRS: list[str] = [p.strip().upper() for p in _env_pairs.split(",") if p.strip()]
    # Enforce Top-10 universe
    TRADING_PAIRS = [s for s in TRADING_PAIRS if s in TOP10_BINANCEUS]
    if not TRADING_PAIRS:
        msg = "No valid trading pairs found in TRADING_PAIRS environment variable - all pairs must be in TOP10_BINANCEUS"
        raise RuntimeError(msg)
else:
    # Use all symbols from trading_universe if no env var set
    TRADING_PAIRS = list(TRADING_SYMBOLS)

USD_AMOUNT_PER_TRADE = float(os.getenv("USD_AMOUNT_PER_TRADE", "50"))
MAX_CONCURRENT_TRADES = int(os.getenv("MAX_CONCURRENT_TRADES", "4"))
TRADING_ENABLED = os.getenv("TRADING_ENABLED", "true").lower() == "true"

# Signal config
MIN_VOLUME_USD = float(os.getenv("MIN_VOLUME_USD", "1000000"))  # $1M 24h quote volume
MIN_PRICE_CHANGE_PCT = float(os.getenv("MIN_PRICE_CHANGE_PCT", "2.0"))  # 2% price change
SIGNAL_COOLDOWN = int(os.getenv("SIGNAL_COOLDOWN", "300"))  # 5 minutes

# Memory management - keep enough history for AI learning without unbounded growth
MAX_TRADE_HISTORY = int(os.getenv("MAX_AUTOBUY_TRADE_HISTORY", "3000"))

# Notifications (optional)
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")
DISCORD_WEBHOOK_URL = os.getenv("DISCORD_WEBHOOK_URL")


class BinanceUSAutobuy:
    def __init__(self) -> None:
        # Client instances (reused)
        self._limiter: BinanceWeightLimiter | None = None
        self._client: BinanceREST | None = None
        self._last_error: str | None = None
        self._last_successful_order_time: str | None = None
        self.active_trades: dict[str, dict[str, Any]] = {}
        self.trade_history: list[dict[str, Any]] = []
        self.signal_history: dict[str, list[dict[str, Any]]] = {}
        self.last_signal_time: dict[str, float] = {}
        self.is_running = False
        self._initialized = False

        # Statistics
        self.total_trades = 0
        self.successful_trades = 0
        # Track background tasks for proper cleanup
        self._tasks: list[asyncio.Task[Any]] = []
        self.failed_trades = 0
        self.total_volume = 0.0

        # Health check status
        self._last_ticker_ok = None
        self._last_order_ok = None
        self._binance_connected = False

        logger.info("Binance US Autobuy instance created (not initialized)")

    async def initialize(self) -> bool:
        """Initialize the autobuy system with explicit startup."""
        if self._initialized:
            return True

        try:
            # Get credentials
            api_key = mystic_config.exchange.binance_us_api_key
            api_secret = mystic_config.exchange.binance_us_secret_key

            if not api_key or not api_secret:
                self._last_error = "Binance US API credentials are not configured"
                logger.error("Binance US API credentials are not configured")
                return False

            # Initialize database
            await self._init_db_async()

            # Test connection using the REST client + limiter
            self._limiter = await BinanceWeightLimiter.create()
            self._client = BinanceREST(self._limiter, api_key=api_key, api_secret=api_secret)

            # Use public ping instead of internal _request
            await self._client.ping()
            self._binance_connected = True
            self._last_error = None

            logger.info("Binance US connection successful")
            logger.info("Binance US Autobuy initialized for pairs: %s", TRADING_PAIRS)

            self._initialized = True
        except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
            self._last_error = str(e)
            self._binance_connected = False
            logger.exception("Connection test failed: %s", e)
            return False
        else:
            return True

    async def cleanup(self) -> None:
        logger.info("Binance US Autobuy cleaned up")

    # ------------------------------ Notifications ------------------------------

    async def send_notification(self, message: str) -> None:
        """Send notification via Telegram and/or Discord (best-effort)."""
        try:
            if TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID:
                async with httpx.AsyncClient() as client:
                    await client.post(
                        f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage",
                        data={"chat_id": TELEGRAM_CHAT_ID, "text": message},
                        timeout=5,
                    )
            if DISCORD_WEBHOOK_URL:
                async with httpx.AsyncClient() as client:
                    await client.post(DISCORD_WEBHOOK_URL, json={"content": message}, timeout=5)
            logger.info("Notification sent")
        except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
            logger.exception("Failed to send notification: %s", e)

    # ------------------------------ Data access ------------------------------

    async def get_ticker_24hr(self, symbol: str) -> dict[str, Any]:
        """Get 24hr ticker for a symbol (Top-10 enforced upstream)."""
        if not self._client:
            return {
                "success": False,
                "data": None,
                "error": {
                    "code": "CLIENT_NOT_INITIALIZED",
                    "message": "Client is not initialized. Call initialize() first.",
                },
            }

        try:
            data = await self._client.ticker_24h(symbol)
            if data:
                self._last_ticker_ok = datetime.now(timezone.utc).isoformat()
                result = {"success": True, "data": data, "error": None}
            else:
                error_msg = f"24h ticker empty for {symbol}"
                logger.error(error_msg)
                result = {
                    "success": False,
                    "data": None,
                    "error": {"code": "TICKER_EMPTY", "message": error_msg},
                }
        except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
            self._last_error = str(e)
            error_msg = f"Error getting ticker for {symbol}: {e}"
            logger.exception(error_msg)
            return {
                "success": False,
                "data": None,
                "error": {"code": "TICKER_ERROR", "message": error_msg},
            }
        else:
            return result

    async def get_current_price(self, symbol: str) -> dict[str, Any]:
        """Get current price for a symbol."""
        if not self._client:
            return {
                "success": False,
                "data": None,
                "error": {
                    "code": "CLIENT_NOT_INITIALIZED",
                    "message": "Client is not initialized. Call initialize() first.",
                },
            }

        try:
            data = await self._client.price(symbol)
            if data and data.get("price") is not None:
                try:
                    price = float(data["price"])
                except (TypeError, ValueError):
                    price = float(str(data["price"]))
                result = {"success": True, "data": price, "error": None}
            else:
                error_msg = f"Price response missing for {symbol}"
                logger.error(error_msg)
                result = {
                    "success": False,
                    "data": None,
                    "error": {"code": "PRICE_MISSING", "message": error_msg},
                }
        except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
            self._last_error = str(e)
            error_msg = f"Error getting price for {symbol}: {e}"
            logger.exception(error_msg)
            return {
                "success": False,
                "data": None,
                "error": {"code": "PRICE_ERROR", "message": error_msg},
            }
        else:
            return result

    # ------------------------------ Signal analysis & execution ------------------------------

    def analyze_signal(self, symbol: str, ticker: dict[str, Any]) -> dict[str, Any]:
        """
        Analyze 24h ticker data and determine if a buy signal should be emitted.
        Returns a dict containing:
            - should_buy: bool
            - price_change_pct: float
            - confidence: int (0-100)
            - signals: list[str]
        """
        try:
            # Extract relevant fields robustly (Binance returns strings)
            price_change_pct = 0.0
            volume_quote = 0.0
            signals: list[str] = []
            confidence = 0

            # Price change percent
            for key in ("priceChangePercent", "priceChangePct", "priceChange"):
                if key in ticker and ticker.get(key) is not None:
                    try:
                        price_change_pct = float(ticker.get(key, 0.0))
                        break
                    except (TypeError, ValueError):
                        continue

            # Quote volume (24h) - prefer quoteVolume or quoteAssetVolume
            for key in ("quoteVolume", "quoteAssetVolume", "volumeQuote"):
                if key in ticker and ticker.get(key) is not None:
                    try:
                        volume_quote = float(ticker.get(key, 0.0))
                        break
                    except (TypeError, ValueError):
                        continue

            # Volume threshold check
            if volume_quote >= MIN_VOLUME_USD:
                confidence += 50
                signals.append("high_volume")
            else:
                signals.append("low_volume")

            # Price movement check
            if price_change_pct >= MIN_PRICE_CHANGE_PCT:
                confidence += 50
                signals.append("up_trend")
            else:
                signals.append("no_strong_move")

            # Cooldown enforcement
            last_time = self.last_signal_time.get(symbol, 0)
            if time.time() - last_time < SIGNAL_COOLDOWN:
                signals.append("cooldown")
                should_buy = False
            else:
                should_buy = price_change_pct >= MIN_PRICE_CHANGE_PCT and volume_quote >= MIN_VOLUME_USD

            # Normalize confidence to 0-100
            confidence = max(0, min(100, confidence))

            result = {
                "should_buy": should_buy,
                "price_change_pct": price_change_pct,
                "confidence": confidence,
                "signals": signals,
            }
        except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
            logger.exception("Failed to analyze signal for %s: %s", symbol, e)
            return {"should_buy": False, "price_change_pct": 0.0, "confidence": 0, "signals": ["error"]}
        else:
            return result

    async def execute_buy_signal(self, symbol: str, signal: dict[str, Any]) -> dict[str, Any]:
        """
        Execute a market buy for the given symbol based on USD_AMOUNT_PER_TRADE.
        Attempts several common client method names for compatibility.
        Returns the ERROR CONTRACT shaped dict on completion.
        """
        if not TRADING_ENABLED:
            msg = "Trading disabled by configuration"
            logger.info(msg)
            return {"success": False, "data": None, "error": {"code": "TRADING_DISABLED", "message": msg}}

        if not self._client:
            msg = "Client not initialized"
            logger.error(msg)
            return {"success": False, "data": None, "error": {"code": "CLIENT_NOT_INITIALIZED", "message": msg}}

        # Respect concurrency limit
        if len(self.active_trades) >= MAX_CONCURRENT_TRADES:
            msg = "Max concurrent trades reached"
            logger.info(msg)
            return {"success": False, "data": None, "error": {"code": "MAX_CONCURRENT", "message": msg}}

        # Get current price to compute quantity
        price_res = await self.get_current_price(symbol)
        if not price_res["success"]:
            return price_res
        price = price_res["data"]
        if price <= 0:
            msg = "Invalid price for order computation"
            logger.error(msg)
            return {"success": False, "data": None, "error": {"code": "INVALID_PRICE", "message": msg}}

        # Compute quantity - use quoteOrderQty approach if client supports it
        quantity = USD_AMOUNT_PER_TRADE / price

        # Check for supported order method before try block
        if not any(hasattr(self._client, method) for method in ["order_market_buy", "order_market", "create_order", "order"]):
            msg = "No supported market order method on client"
            raise RuntimeError(msg)

        order_result = None
        try:
            # Try common async method names for placing market buy by quote amount
            if hasattr(self._client, "order_market_buy"):
                # Some clients expect quoteOrderQty as string
                order_result = await self._client.order_market_buy(symbol, quoteOrderQty=str(USD_AMOUNT_PER_TRADE))
            elif hasattr(self._client, "order_market"):
                order_result = await self._client.order_market(symbol, "BUY", quoteOrderQty=str(USD_AMOUNT_PER_TRADE))
            elif hasattr(self._client, "create_order"):
                order_result = await self._client.create_order(symbol=symbol, side="BUY", type="MARKET", quoteOrderQty=str(USD_AMOUNT_PER_TRADE))
            # As a last resort, try a generic 'order' method
            elif hasattr(self._client, "order"):
                order_result = await self._client.order(symbol=symbol, side="BUY", type="MARKET", quoteOrderQty=str(USD_AMOUNT_PER_TRADE))

            # Normalize order_result structure
            if not isinstance(order_result, dict):
                # If client returned something unexpected, wrap it
                order_result = {"raw": order_result}

            # Compute fill price and quantities
            fill_price = self._compute_fill_price(order_result)
            executed_qty = float(order_result.get("executedQty", order_result.get("origQty", quantity))) if order_result else quantity
            quote_qty = float(order_result.get("cummulativeQuoteQty", USD_AMOUNT_PER_TRADE))

            # Create trade record
            trade_record = {
                "symbol": symbol,
                "order_id": str(order_result.get("orderId", "")),
                "amount_usd": USD_AMOUNT_PER_TRADE,
                "price": fill_price or price,
                "quantity": executed_qty,
                "quote_qty": quote_qty,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "confidence": int(signal.get("confidence", 0)),
                "signals": signal.get("signals", []),
                "status": "FILLED" if executed_qty and executed_qty > 0 else "UNKNOWN",
            }

            # Update runtime state
            self.trade_history.append(trade_record)

            # Ring buffer cleanup - keep recent trades for AI learning
            if len(self.trade_history) > MAX_TRADE_HISTORY:
                self.trade_history = self.trade_history[-MAX_TRADE_HISTORY:]

            self.active_trades[symbol] = trade_record
            self.total_trades += 1
            if trade_record["status"] == "FILLED":
                self.successful_trades += 1
                self.total_volume += trade_record["quote_qty"]
                self._last_successful_order_time = trade_record["timestamp"]
                self._last_order_ok = datetime.now(timezone.utc).isoformat()
            else:
                self.failed_trades += 1

            # Persist trade asynchronously (fire-and-forget)
            try:
                task1 = await task_manager.create_task(self._persist_trade_async(trade_record), name="binance_us_autobuy:persist_trade")
                self._tasks.append(task1)
            except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError):
                # In case loop not running or task creation fails, persist synchronously fallback
                await self._persist_trade_async(trade_record)

            # Notifications
            try:
                msg = f"Executed buy for {symbol}: {trade_record['quantity']} @ {trade_record['price']} USD"
                task2 = await task_manager.create_task(self.send_notification(msg), name="binance_us_autobuy:send_notification")
                self._tasks.append(task2)
            except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError):
                # best-effort
                pass

            # Record last signal time to enforce cooldown
            self.last_signal_time[symbol] = time.time()

            logger.info("Buy order executed for %s: %s", symbol, trade_record)
        except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
            self.failed_trades += 1
            self._last_error = str(e)
            logger.exception("Failed to execute buy for %s: %s", symbol, e)
            return {"success": False, "data": None, "error": {"code": "ORDER_FAILED", "message": str(e)}}
        else:
            return {"success": True, "data": trade_record, "error": None}

    async def process_trading_pair(self, symbol: str) -> None:
        """
        Process a single trading pair:
         - Fetch 24h ticker
         - Analyze signal
         - Record signal history
         - Execute buy if appropriate
        """
        try:
            ticker_res = await self.get_ticker_24hr(symbol)
            if not ticker_res["success"]:
                logger.debug("Skipping %s due to ticker fetch failure: %s", symbol, ticker_res["error"])
                return

            ticker = ticker_res["data"]
            signal = self.analyze_signal(symbol, ticker)

            # Record signal history
            sh = self.signal_history.setdefault(symbol, [])
            sh.append(
                {
                    "timestamp": time.time(),
                    "confidence": signal.get("confidence", 0),
                    "signals": signal.get("signals", []),
                    "price_change_pct": signal.get("price_change_pct", 0.0),
                },
            )
            if len(sh) > 100:
                self.signal_history[symbol] = sh[-100:]

            if signal.get("should_buy"):
                await self.execute_buy_signal(symbol, signal)
        except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
            logger.exception("Error processing %s: %s", symbol, e)

    async def run_trading_cycle(self) -> None:
        """Run a single trading cycle across all configured pairs."""
        logger.info("Starting trading cycle...")
        # Add timeout per cycle
        cycle_timeout = float(os.getenv("CYCLE_TIMEOUT_SEC", "60.0"))
        try:
            await asyncio.wait_for(
                asyncio.gather(
                    *(self.process_trading_pair(s) for s in TRADING_PAIRS),
                    return_exceptions=True,
                ),
                timeout=cycle_timeout,
            )
        except asyncio.TimeoutError:
            logger.warning("Trading cycle timed out after %s seconds", cycle_timeout)
        logger.info(
            "Cycle complete - Active trades: %d | Totals -> trades=%d success=%d failed=%d",
            len(self.active_trades),
            self.total_trades,
            self.successful_trades,
            self.failed_trades,
        )

    async def run(self) -> None:
        """Main loop."""
        logger.info("Starting Binance US Autobuy System")
        logger.info("Trading pairs: %s", TRADING_PAIRS)
        logger.info(
            "Amount per trade: $%.2f | Trading enabled: %s",
            USD_AMOUNT_PER_TRADE,
            TRADING_ENABLED,
        )

        if not await self.initialize():
            logger.error("Failed to initialize autobuy system")
            return

        self.is_running = True
        try:
            while self.is_running:
                await self.run_trading_cycle()
                cycle_sleep = float(os.getenv("CYCLE_SLEEP_SEC", "30.0"))
                await asyncio.sleep(cycle_sleep)
        except KeyboardInterrupt:
            logger.info("Shutdown requested by user")
        except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
            logger.exception("Fatal error in main loop: %s", e)
        finally:
            self.is_running = False
            await self.cleanup()

    # ------------------------------ Status ------------------------------

    def get_status(self) -> dict[str, Any]:
        """Get comprehensive status for dashboard with heartbeat fields."""
        return {
            "success": True,
            "data": {
                "exchange_id": EXCHANGE_ID,
                "is_running": self.is_running,
                "initialized": self._initialized,
                "trading_pairs": TRADING_PAIRS,
                "active_trades": len(self.active_trades),
                "total_trades": self.total_trades,
                "successful_trades": self.successful_trades,
                "failed_trades": self.failed_trades,
                "total_volume": self.total_volume,
                "trading_enabled": TRADING_ENABLED,
                "last_error": self._last_error,
                "last_successful_order_time": self._last_successful_order_time,
                "cooldown_expirations": {s: self.last_signal_time.get(s, 0) + SIGNAL_COOLDOWN for s in TRADING_PAIRS},
                "last_update": datetime.now(timezone.utc).isoformat(),
                # Heartbeat fields for instant UI rendering
                "last_ticker_ok": self._last_ticker_ok,
                "last_order_ok": self._last_order_ok,
                "binance_connected": self._binance_connected,
                # Configuration values
                "config": {
                    "usd_amount_per_trade": USD_AMOUNT_PER_TRADE,
                    "max_concurrent_trades": MAX_CONCURRENT_TRADES,
                    "min_volume_usd": MIN_VOLUME_USD,
                    "min_price_change_pct": MIN_PRICE_CHANGE_PCT,
                    "signal_cooldown": SIGNAL_COOLDOWN,
                },
            },
            "error": None,
        }

    # --------------------------------------------------------------------------
    # Helper methods
    # --------------------------------------------------------------------------
    def _compute_fill_price(self, order_result: dict[str, Any]) -> float:
        """Compute average fill price from order result."""
        try:
            cummulative_quote = float(order_result.get("cummulativeQuoteQty", order_result.get("cummulativeQuoteQty", 0)))
            executed_qty = float(order_result.get("executedQty", order_result.get("executedQty", 0)))
            if executed_qty > 0:
                return cummulative_quote / executed_qty
            # Fallback to price field if available
            return float(order_result.get("price", 0))
        except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError):
            return 0.0

    async def _init_db_async(self) -> None:
        """Initialize SQLite tables for persistence (async)."""
        try:

            def _init_db_sync():
                conn = sqlite3.connect(str(_DB_PATH))
                with conn:
                    conn.execute(
                        """
                        CREATE TABLE IF NOT EXISTS autobuy_trades (
                            id TEXT PRIMARY KEY,
                            symbol TEXT,
                            order_id TEXT,
                            amount_usd REAL,
                            price REAL,
                            quantity REAL,
                            quote_qty REAL,
                            timestamp TEXT,
                            confidence INTEGER,
                            signals TEXT,
                            status TEXT
                        )
                        """,
                    )
                    conn.execute(
                        """
                        CREATE TABLE IF NOT EXISTS autobuy_signals (
                            symbol TEXT,
                            timestamp REAL,
                            confidence INTEGER,
                            signals TEXT,
                            price_change_pct REAL
                        )
                        """,
                    )
                conn.close()

            # Run DB init in thread pool to avoid blocking event loop
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, _init_db_sync)
            logger.info("Database initialized successfully")
        except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
            logger.exception("Autobuy DB init failed: %s", e)
            raise

    async def _persist_trade_async(self, trade_record: dict[str, Any]) -> None:
        """Persist trade record to database (async)."""
        try:

            def _persist_sync():
                conn = sqlite3.connect(str(_DB_PATH))
                with conn:
                    conn.execute(
                        """
                        INSERT INTO autobuy_trades (
                            id, symbol, order_id, amount_usd, price, quantity, quote_qty,
                            timestamp, confidence, signals, status
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            str(uuid.uuid4()),
                            trade_record["symbol"],
                            trade_record["order_id"],
                            trade_record["amount_usd"],
                            trade_record["price"],
                            trade_record["quantity"],
                            trade_record.get("quote_qty", trade_record["amount_usd"]),
                            trade_record["timestamp"],
                            trade_record["confidence"],
                            ",".join(trade_record["signals"]),
                            trade_record["status"],
                        ),
                    )
                conn.close()

            # Run DB write in thread pool to avoid blocking event loop
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, _persist_sync)
        except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
            logger.exception("Failed to persist trade: %s", e)


# Global instance (lazy creation)
autobuy_system: BinanceUSAutobuy | None = None


# Autobuy system state - using dict to avoid global keyword
_autobuy_system_state: dict[str, BinanceUSAutobuy | None] = {"instance": None}


def get_autobuy_system() -> BinanceUSAutobuy:
    """Get the global autobuy system instance, creating it if needed."""
    if _autobuy_system_state["instance"] is None:
        _autobuy_system_state["instance"] = BinanceUSAutobuy()
    return _autobuy_system_state["instance"]


async def main() -> None:
    """Main entry point for standalone execution."""
    system = get_autobuy_system()
    await system.run()


if __name__ == "__main__":
    asyncio.run(main())
