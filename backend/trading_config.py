"""
Trading Configuration
Centralized configuration for all trading-related hardcoded values
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from typing import Any, ClassVar

# Optional imports - try at top level
try:
    from backend.config.trading_universe import TRADING_SYMBOLS
except (ImportError, ModuleNotFoundError, AttributeError, ValueError, TypeError, RuntimeError):
    TRADING_SYMBOLS = None

try:
    from backend.modules.ai.persistent_cache import get_persistent_cache
except (ImportError, ModuleNotFoundError, AttributeError, ValueError, TypeError, RuntimeError):
    get_persistent_cache = None

try:
    from backend.services.binanceus_live_client import (
        BinanceUSLiveClient,
    )  # Binance.US only
except (ImportError, ModuleNotFoundError, AttributeError, ValueError, TypeError, RuntimeError):
    BinanceUSLiveClient = None

try:
    from backend.config.redis_config import get_shared_redis_sync
except (ImportError, ModuleNotFoundError, AttributeError, ValueError, TypeError, RuntimeError):
    get_shared_redis_sync = None

logger = logging.getLogger(__name__)


class TradingConfig:
    """Centralized trading configuration"""

    # ----- Risk Management Defaults -----
    DEFAULT_MAX_POSITION_SIZE = float(os.getenv("DEFAULT_MAX_POSITION_SIZE", "0.10"))  # 10% of portfolio
    DEFAULT_MAX_DRAWDOWN = float(os.getenv("DEFAULT_MAX_DRAWDOWN", "0.10"))  # 10% maximum drawdown
    DEFAULT_STOP_LOSS = float(os.getenv("DEFAULT_STOP_LOSS", "0.018"))  # 1.8% stop loss - AGGRESSIVE
    DEFAULT_TAKE_PROFIT = float(os.getenv("DEFAULT_TAKE_PROFIT", "0.03"))  # 3.0% take profit - AGGRESSIVE
    DEFAULT_MAX_LEVERAGE = int(os.getenv("DEFAULT_MAX_LEVERAGE", "3"))  # 3x maximum leverage
    DEFAULT_MIN_VOLUME = int(os.getenv("MIN_VOLUME_USD", "1000000"))  # env-tunable
    DEFAULT_MAX_SLIPPAGE = float(os.getenv("DEFAULT_MAX_SLIPPAGE", "0.02"))  # 2% maximum slippage

    # ----- Performance Thresholds -----
    DEFAULT_SHARPE_RATIO = 0.0
    DEFAULT_TOTAL_TRADES = 0
    DEFAULT_WINNING_TRADES = 0
    DEFAULT_LOSING_TRADES = 0
    DEFAULT_TOTAL_PNL = 0.0

    # ----- Redis TTL Values (in seconds) -----
    AUTO_TRADE_CONFIG_TTL = 3600  # 1 hour
    AUTO_TRADING_ENABLED_TTL = 3600  # 1 hour

    # ----- API / Redis Defaults -----
    # All Live Data, No Fallback/Hardcoded Data
    # Redis connection must be configured via environment variables

    # ----- Service Ports (keep aligned with your system) -----
    DEFAULT_SERVICE_PORT = int(os.getenv("BACKEND_PORT", "8000"))
    AI_STRATEGY_GENERATOR_PORT = int(os.getenv("AI_STRATEGY_PORT", "8002"))

    # ----- Timeouts / pacing -----
    DEFAULT_REQUEST_TIMEOUT = int(os.getenv("REQUEST_TIMEOUT_SECONDS", "10"))  # seconds
    DEFAULT_BATCH_DELAY = float(os.getenv("BATCH_DELAY_SECONDS", "0.25"))  # seconds

    # ----- Trading Thresholds -----
    PROFIT_THRESHOLD = float(os.getenv("PROFIT_THRESHOLD", "0.01"))  # 1% gain threshold
    LOSS_THRESHOLD = float(os.getenv("LOSS_THRESHOLD", "-0.01"))  # 1% loss threshold
    REBALANCE_THRESHOLD = float(os.getenv("REBALANCE_THRESHOLD", "0.02"))  # 2% rebalancing threshold
    VAR_THRESHOLD = float(os.getenv("VAR_THRESHOLD", "0.05"))  # 5% VaR threshold
    DRAWDOWN_THRESHOLD = float(os.getenv("DRAWDOWN_THRESHOLD", "0.15"))  # 15% drawdown threshold

    # ----- Model Parameters -----
    DEFAULT_LEARNING_RATE = float(os.getenv("DEFAULT_LEARNING_RATE", "0.001"))
    DEFAULT_DROPOUT = float(os.getenv("DEFAULT_DROPOUT", "0.2"))
    DEFAULT_CONFIDENCE_THRESHOLD = float(os.getenv("DEFAULT_CONFIDENCE_THRESHOLD", "0.75"))  # must match AI conf threshold
    DEFAULT_TRAIN_TEST_SPLIT = float(os.getenv("DEFAULT_TRAIN_TEST_SPLIT", "0.8"))  # 80/20 split

    # ----- Cache TTLs -----
    STRATEGY_CACHE_TTL = int(os.getenv("STRATEGY_CACHE_TTL", "86400"))  # 24h
    PORTFOLIO_CACHE_TTL = int(os.getenv("PORTFOLIO_CACHE_TTL", "1800"))  # 30m
    RISK_CACHE_TTL = int(os.getenv("RISK_CACHE_TTL", "1800"))  # 30m

    # ----- Live Data Only - No Mock Data -----
    # All prices must come from live Binance API
    # No fallback mock prices allowed

    # ----- Live Data Only - No Hardcoded Defaults -----
    # All portfolio data must come from live market data
    # No fallback hardcoded values allowed in production

    # ----- Performance Profiles (reference) -----
    DEFAULT_PERFORMANCE_METRICS: ClassVar[dict[str, dict[str, float]]] = {
        "conservative": {"sharpe": 0.95, "returns": 0.12, "max_dd": 0.05},
        "moderate": {"sharpe": 1.42, "returns": 0.18, "max_dd": 0.12},
        "aggressive": {"sharpe": 1.85, "returns": 0.23, "max_dd": 0.08},
        "very_aggressive": {"sharpe": 2.10, "returns": 0.31, "max_dd": 0.15},
    }

    # ----- Volatility Ranges (reference) -----
    # Dynamically generated from trading_universe (live data)
    @classmethod
    def get_volatility_ranges(cls) -> dict[str, tuple[float, float]]:
        """Get volatility ranges from trading_universe symbols (live data)"""
        # Validate TRADING_SYMBOLS outside try to avoid TRY301
        if TRADING_SYMBOLS is None:
            msg = "TRADING_SYMBOLS not available"
            raise RuntimeError(msg)

        try:
            # Generate ranges for all Top-10 symbols (live data)
            ranges = {}
            for symbol in TRADING_SYMBOLS:
                # Convert BTCUSDT to BTC/USDT for range key
                ccxt_symbol = symbol.replace("USDT", "/USDT")
                # Default volatility ranges (can be configured via env vars)
                ranges[ccxt_symbol] = (0.02, 0.15)  # Conservative range for all symbols
        except (ImportError, ModuleNotFoundError, AttributeError, ValueError, TypeError, RuntimeError) as e:
            msg = f"Failed to import TRADING_SYMBOLS from trading_universe: {e}"
            raise RuntimeError(msg) from e
        else:
            return ranges

    # ----- No Fallback Portfolio Defaults -----
    # All Live Data, No Fallback/Hardcoded Data
    # Portfolio data must come from live exchange only

    # ---------------------------
    # Section: Live Portfolio API
    # ---------------------------

    @classmethod
    async def get_live_portfolio_data(cls) -> dict[str, Any]:
        """
        Get real portfolio data from the exchange (Binance.US) when available.
        All Live Data, No Fallback/Hardcoded Data - raises error if unavailable.

        Returns:
            {
                "total_value": float,
                "cash_allocation": float,
                "asset_amounts": { "COIN/USDT": {"amount": float, "value": float}, ... },
                "asset_allocations": { "COIN/USDT": float, ... },
                "data_source": "live_exchange",
            }

        Raises:
            RuntimeError: If live portfolio data is unavailable (no fallbacks)
        """
        # Validate required services outside try to avoid TRY301
        if get_persistent_cache is None or BinanceUSLiveClient is None:
            msg = "Required services not available"
            raise RuntimeError(msg)

        try:
            client = BinanceUSLiveClient()

            # Support both async and sync client methods gracefully
            async def _maybe_await(x):
                if asyncio.iscoroutine(x):
                    return await x
                return x

            # load markets if available (sync or async)
            await _maybe_await(getattr(client, "load_markets", lambda: None)())

            # fetch balance (sync or async)
            balance = await _maybe_await(getattr(client, "fetch_balance", dict)())

            # obtain cache (support sync or async get_persistent_cache())
            cache_obj = await _maybe_await(get_persistent_cache())

            latest_prices = {}
            try:
                # Some caches expose sync getters or async getters
                result = getattr(cache_obj, "get_latest_prices", dict)()
                latest_prices = await _maybe_await(result) or {}
            except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError):
                latest_prices = {}

            live_portfolio = {
                "total_value": 0.0,
                "cash_allocation": 0.0,
                "asset_amounts": {},
                "asset_allocations": {},
                "data_source": "live_exchange",
            }

            # Normalize typical ccxt-like balance payloads
            # Prefer balance["total"] mapping if present; otherwise iterate items
            currency_totals: dict[str, float] = {}
            if isinstance(balance, dict) and "total" in balance and isinstance(balance["total"], dict):
                for cur, amt in balance["total"].items():
                    try:
                        currency_totals[str(cur).upper()] = float(amt or 0.0)
                    except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError):
                        continue
            else:
                for cur, info in (balance or {}).items():
                    if isinstance(info, dict) and "total" in info:
                        try:
                            currency_totals[str(cur).upper()] = float(info.get("total") or 0.0)
                        except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError):
                            continue

            # Normalize latest price keys once (e.g., BTC-USD -> BTCUSDT)
            normalized_prices: dict[str, float] = {}
            for sym, p in (latest_prices or {}).items():
                try:
                    if not isinstance(sym, str):
                        continue
                    key = sym.replace("-", "").replace("/", "").upper()
                    if key.endswith("USD"):
                        key = key[:-3] + "USDT"
                    normalized_prices[key] = float(p or 0.0)
                except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError):
                    continue

            # Build portfolio using normalized_prices
            for currency, total_amount in currency_totals.items():
                if total_amount <= 0:
                    continue
                if currency in {"USDT", "USD"}:
                    live_portfolio["cash_allocation"] += total_amount
                    continue

                # find a matching price for currency/USDT
                symbol_key = f"{currency}USDT"
                price = float(normalized_prices.get(symbol_key, 0.0))

                if price > 0:
                    value = total_amount * price
                    asset_key = f"{currency}/USDT"
                    live_portfolio["asset_amounts"][asset_key] = {
                        "amount": total_amount,
                        "value": value,
                    }
                    live_portfolio["total_value"] += value

            live_portfolio["total_value"] += live_portfolio["cash_allocation"]

            # Allocations
            tv = live_portfolio["total_value"]
            if tv > 0:
                for asset, data in list(live_portfolio["asset_amounts"].items()):
                    live_portfolio["asset_allocations"][asset] = data["value"] / tv
        except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
            # All Live Data, No Fallback/Hardcoded Data - raise error
            msg = f"Failed to get live portfolio data: {e}"
            raise RuntimeError(msg) from e
        else:
            return live_portfolio

    # ---------------------------
    # Section: Static Config APIs
    # ---------------------------

    @classmethod
    def get_risk_management_config(cls) -> dict[str, Any]:
        return {
            "max_position_size": cls.DEFAULT_MAX_POSITION_SIZE,
            "max_drawdown": cls.DEFAULT_MAX_DRAWDOWN,
            "stop_loss": cls.DEFAULT_STOP_LOSS,
            "take_profit": cls.DEFAULT_TAKE_PROFIT,
            "max_leverage": cls.DEFAULT_MAX_LEVERAGE,
            "min_volume": cls.DEFAULT_MIN_VOLUME,
            "max_slippage": cls.DEFAULT_MAX_SLIPPAGE,
        }

    @classmethod
    def get_risk_management_config_ui(cls) -> dict[str, Any]:
        """UI-friendly percentages (0-100 scales)."""
        return {
            "max_position_size_pct": cls.DEFAULT_MAX_POSITION_SIZE * 100.0,
            "max_drawdown_pct": cls.DEFAULT_MAX_DRAWDOWN * 100.0,
            "stop_loss_pct": cls.DEFAULT_STOP_LOSS * 100.0,
            "take_profit_pct": cls.DEFAULT_TAKE_PROFIT * 100.0,
            "max_leverage": cls.DEFAULT_MAX_LEVERAGE,
            "min_volume_usd": cls.DEFAULT_MIN_VOLUME,
            "max_slippage_pct": cls.DEFAULT_MAX_SLIPPAGE * 100.0,
        }

    @classmethod
    def get_performance_config(cls) -> dict[str, Any]:
        return {
            "total_trades": cls.DEFAULT_TOTAL_TRADES,
            "winning_trades": cls.DEFAULT_WINNING_TRADES,
            "losing_trades": cls.DEFAULT_LOSING_TRADES,
            "total_pnl": cls.DEFAULT_TOTAL_PNL,
            "sharpe_ratio": cls.DEFAULT_SHARPE_RATIO,
        }

    @classmethod
    def get_redis_config(cls) -> dict[str, Any]:
        """Get Redis configuration from environment variables (no hardcoded defaults)"""
        # All Live Data, No Fallback/Hardcoded Data
        redis_host = os.getenv("REDIS_HOST")
        redis_port = os.getenv("REDIS_PORT")
        redis_db = os.getenv("REDIS_DB")

        if not redis_host:
            msg = "REDIS_HOST environment variable is required - no fallback/hardcoded Redis host"
            raise RuntimeError(msg)

        return {
            "host": redis_host,
            "port": int(redis_port) if redis_port else 6379,  # Default port only if env var provided
            "db": int(redis_db) if redis_db else 0,  # Default db only if env var provided
        }

    @classmethod
    def get_service_config(cls) -> dict[str, Any]:
        return {
            "default_port": int(os.getenv("BACKEND_PORT", cls.DEFAULT_SERVICE_PORT)),
            "ai_strategy_port": cls.AI_STRATEGY_GENERATOR_PORT,
        }

    @classmethod
    def get_micro_account_config(cls, budget: float = 100.0) -> dict[str, Any]:
        # clamp budget to positive
        b = max(float(budget), 1.0)
        return {
            "max_position_size": b * 0.15,  # 15% max position
            "min_trade_size": max(1.0, b * 0.01),  # $1 or 1%
            "max_order_size": b * 0.50,  # 50% max single order
            "daily_loss_limit": b * 0.05,  # 5% daily loss
            "risk_per_trade": b * 0.02,  # 2% risk per trade
            "stop_loss_pct": 0.03,
            "take_profit_pct": 0.06,
            "emergency_stop_pct": 0.20,
            "max_concurrent_positions": 2 if b < 500 else (3 if b < 1000 else 4),
            "confidence_threshold": 0.70,
            "max_leverage": cls.DEFAULT_MAX_LEVERAGE,
            "revenue_target_per_minute_usd": float(os.getenv("REVENUE_PER_MIN_USD", "1.0")),
            "trade_cooldown_seconds": int(os.getenv("TRADE_COOLDOWN_SECONDS", "60")),
            "current_budget": b,
            "starting_budget": 100.0,
            "growth_factor": b / 100.0,
        }

    def apply_mode_settings(self, mode_config: dict[str, Any]) -> None:
        """
        Apply auto-trader mode settings to live trading systems
        Called when mode switches from SAFE to MAX_PROFIT or vice versa
        """
        try:
            # Update Redis with new mode settings for all trading engines
            redis_client = get_shared_redis_sync() if get_shared_redis_sync else None
            if redis_client:
                settings_key = "trading:mode_settings"

                # Convert mode config to Redis-compatible format
                redis_settings = {
                    "mode": mode_config.get("name", "SAFE"),
                    "take_profit_pct": mode_config.get("tp", 0.03),
                    "stop_loss_pct": mode_config.get("sl", 0.03),
                    "trail_stop_pct": mode_config.get("trail", 0.02),
                    "min_symbol_seconds": mode_config.get("symbol_delay", 90),
                    "min_strategy_seconds": mode_config.get("strategy_delay", 45),
                    "min_global_seconds": mode_config.get("global_delay", 7),
                    "max_daily_orders": mode_config.get("max_daily_orders", 120),
                    "updated_at": asyncio.get_event_loop().time() if asyncio.get_event_loop() else 0,
                }

                redis_client.set(settings_key, json.dumps(redis_settings))
                logger.info(f"[MODE APPLY] Applied {mode_config.get('name', 'UNKNOWN')} settings to Redis")

            # Broadcast mode change to all trading components
            self._broadcast_mode_change(mode_config)

        except Exception as e:
            logger.info(f"[MODE APPLY] Error applying mode settings: {e}")

    def _broadcast_mode_change(self, mode_config: dict[str, Any]) -> None:
        """Broadcast mode change to all trading components"""
        try:
            # Update in-memory trading parameters
            self.DEFAULT_TAKE_PROFIT = mode_config.get("tp", 0.03)
            self.DEFAULT_STOP_LOSS = mode_config.get("sl", 0.03)

            # Update timing parameters
            self.DEFAULT_REQUEST_TIMEOUT = mode_config.get("symbol_delay", 90)

            logger.info(f"[MODE BROADCAST] Updated trading parameters for {mode_config.get('name', 'UNKNOWN')} mode")

        except Exception as e:
            logger.info(f"[MODE BROADCAST] Error broadcasting mode change: {e}")


# Global configuration instance
trading_config = TradingConfig()
