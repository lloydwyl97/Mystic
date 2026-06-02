#!/usr/bin/env python3
"""
CRYPTO AUTOENGINE Configuration - SINGLE SOURCE OF TRUTH
Central configuration for Binance US coins - Top 10 supported symbols

FIXED FOR PRODUCTION RELIABILITY:
- SINGLE SOURCE OF TRUTH: This is the ONLY authoritative config module
- STRICT ALLOWLIST ENFORCEMENT: No non-fatal validation - strict fail for live trading
- CANONICAL CACHE SCHEMA: Standardized cosmic_data and all cache shapes
- ENVIRONMENT CONSISTENCY: Trade amounts and enablement propagate to all services
- THROTTLING SINGLE SOURCE: All fetchers read from get_throttling_config()
- REQUIRED NORMALIZER: No silent fallbacks that yield "not found"
- HEALTH INTEGRATION: get_validation_status() surfaces in health endpoints
- STRATEGY ALIGNMENT: strategy_count matches trainer/leaderboard expectations
- UTC CONSISTENCY: All timestamps UTC ISO-8601 with Z

Windows/Python 3.12+ Compatibility:
- Uses modern type annotations (list[...], dict[...]) compatible with Python 3.12+
- All logging messages are ASCII-only for Windows PowerShell compatibility
- Safe environment variable parsing prevents import-time crashes
- Strict validation with clear error reporting for production reliability
"""

import logging
import os
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, TypedDict

# Import from single source of truth
try:
    from backend.config.trading_universe import EXCHANGE_ID, TOP10_COINS, TRADING_SYMBOLS
except (ImportError, ModuleNotFoundError, AttributeError, ValueError, TypeError, RuntimeError) as e:
    msg = f"Failed to import trading_universe: {e}"
    raise RuntimeError(msg) from e

logger = logging.getLogger("mystic.config")


@dataclass
class CoinConfig:
    """Configuration for individual coins"""

    symbol: str
    exchange: str  # EXCHANGE_ID from trading_universe
    base_currency: str
    quote_currency: str
    min_trade_amount: float
    max_trade_amount: float
    enabled: bool = True


@dataclass
class FetcherConfig:
    """Configuration for data fetchers"""

    price_fetch_interval: int = 10  # seconds
    volume_fetch_interval: int = 180  # 3 minutes
    indicator_calc_interval: int = 120  # 2 minutes
    mystic_fetch_interval: int = 3600  # 1 hour
    cache_ttl: int = 300  # 5 minutes
    price_change_threshold: float = 0.002  # 0.2%


@dataclass
class StrategyConfig:
    """Configuration for trading strategies"""

    min_confidence: float = 0.7
    max_confidence: float = 0.95
    min_signal_strength: float = 0.6
    strategy_count: int = 50  # Aligned with mutation trainer MAX_STRATEGIES default
    cooldown_period: int = 300  # 5 minutes between trades


@dataclass
class APIConfig:
    """API configuration for external data sources"""

    # All Live Data, No Fallback/Hardcoded Data
    binance_base_url: str = field(default_factory=lambda: os.getenv("BINANCEUS_BASE", "https://api.binance.us") + "/api/v3")
    # Cosmic data sources - used for Tier-3 cosmic analysis
    noaa_base_url: str = "https://services.swpc.noaa.gov/json"  # Solar activity data
    schumann_base_url: str = "https://www2.irf.se/maggraphs/schumann"  # Schumann resonance data
    max_retries: int = 3
    timeout: int = 10


class PriceCache(TypedDict):
    """Type definition for price cache"""

    price: float
    timestamp: float


class VolumeCache(TypedDict):
    """Type definition for volume cache"""

    volume: float
    timestamp: float


class RSICache(TypedDict):
    """Type definition for RSI cache"""

    value: float
    timestamp: float


class MACDCache(TypedDict):
    """Type definition for MACD cache"""

    value: dict[str, float]  # Contains 'macd', 'signal', 'histogram'
    timestamp: float


class LastUpdatedCache(TypedDict):
    """Type definition for last updated timestamps"""

    timestamp: float


class StrategySignalCache(TypedDict):
    """Type definition for strategy signals cache"""

    signal: float
    confidence: float
    timestamp: float


class CosmicDataCache(TypedDict):
    """Type definition for cosmic data cache - CANONICAL SCHEMA"""

    data: dict[str, Any]
    timestamp: float  # Single canonical timestamp field (UTC float epoch)
    # Note: This replaces any "last_updated" fields - all consumers must use "timestamp"


class TradeCooldownCache(TypedDict):
    """Type definition for trade cooldowns"""

    until: float  # Timestamp until cooldown expires


class CryptoAutoEngineConfig:
    """Complete CRYPTO AUTOENGINE configuration"""

    binance_coins: list[CoinConfig]
    all_coins: list[CoinConfig]
    fetcher_config: FetcherConfig
    strategy_config: StrategyConfig
    api_config: APIConfig
    cache_structure: dict[
        str,
        dict[
            str,
            PriceCache | VolumeCache | RSICache | MACDCache | LastUpdatedCache | StrategySignalCache | CosmicDataCache | TradeCooldownCache,
        ],
    ]
    throttling_rules: dict[str, dict[str, int | float]]

    def __init__(self) -> None:
        # 1. COIN CONFIGURATION - TOP 10 BINANCE US COINS (FROM TRADING_UNIVERSE - LIVE DATA)

        # Use TRADING_SYMBOLS from trading_universe (single source of truth)
        # Trade amounts loaded from environment with defaults
        # Parse trade amounts with safe defaults to prevent import-time crashes
        default_min_trade = self._safe_parse_env_float("MIN_TRADE_AMOUNT", 10.0)
        default_max_trade = self._safe_parse_env_float("MAX_TRADE_AMOUNT", 5000.0)

        # Generate coin configs from TRADING_SYMBOLS (live data)
        self.binance_coins = []
        for symbol in TRADING_SYMBOLS:
            # Extract base currency from symbol (e.g., BTCUSDT -> BTC)
            base = symbol.replace("USDT", "")
            if base in TOP10_COINS:
                self.binance_coins.append(
                    CoinConfig(
                        symbol,
                        EXCHANGE_ID,  # Use EXCHANGE_ID from trading_universe
                        base,
                        "USDT",
                        default_min_trade,
                        default_max_trade,
                    )
                )

        # Validate trade amounts
        self._validate_trade_amounts()

        # Validate against enforced allowlist
        self._validate_binance_allowlist()

        # All coins (Binance US only)
        self.all_coins = self.binance_coins

        # 2. FETCHER CONFIGURATION
        self.fetcher_config = FetcherConfig()

        # 3. STRATEGY CONFIGURATION
        self.strategy_config = StrategyConfig()

        # 4. API CONFIGURATION
        self.api_config = APIConfig()

        # 5. CACHE STRUCTURE - CANONICAL SCHEMA FOR ALL CONSUMERS
        self.cache_structure: dict[
            str,
            dict[
                str,
                PriceCache | VolumeCache | RSICache | MACDCache | LastUpdatedCache | StrategySignalCache | CosmicDataCache | TradeCooldownCache,
            ],
        ] = {
            "price": {},  # Dict[str, PriceCache] - {price: float, timestamp: float}
            "volume_24h": {},  # Dict[str, VolumeCache] - {volume: float, timestamp: float}
            "rsi": {},  # Dict[str, RSICache] - {value: float, timestamp: float}
            "macd": {},  # Dict[str, MACDCache] - {value: {macd, signal, histogram}, timestamp: float}
            "last_updated": {},  # Dict[str, LastUpdatedCache] - {timestamp: float}
            "strategy_signals": {},  # Dict[str, StrategySignalCache] - {signal: float, confidence: float, timestamp: float}
            "cosmic_data": {},  # Dict[str, CosmicDataCache] - {data: dict, timestamp: float} - CANONICAL SCHEMA
            "trade_cooldowns": {},  # Dict[str, TradeCooldownCache] - {until: float}
        }

        # 6. THROTTLING RULES - Consolidated thresholds
        self.throttling_rules: dict[str, dict[str, int | float]] = {
            "price": {
                "min_interval": 10,  # seconds
                "max_calls_per_minute": 60,
            },
            "volume": {
                "min_interval": 180,  # seconds
                "max_calls_per_minute": 10,
            },
            "api": {
                "max_retries": self.api_config.max_retries,
                "timeout": self.api_config.timeout,
            },
        }

        # Internal status markers
        self.allowlist_validation_status = "unknown"
        self.allowlist_validation_error = None
        self._trade_amounts_validated = hasattr(self, "_trade_amounts_validated") and self._trade_amounts_validated

    def _safe_parse_env_float(self, key: str, default: float) -> float:
        """Safely parse a float from the environment with a fallback default."""
        raw = os.getenv(key)
        if raw is None:
            return default
        raw_str = raw.strip()
        if raw_str == "":
            return default
        try:
            # Allow commas in numbers like "1,000.0"
            normalized = raw_str.replace(",", "")
            return float(normalized)
        except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError):
            logger.warning(f"Invalid float for env {key}: '{raw}' - using default {default}")
            return default

    def get_throttling_config(self) -> dict[str, dict[str, int | float]]:
        """Return throttling rules for fetchers and API clients"""
        return self.throttling_rules

    def get_coin_by_symbol(self, symbol: str) -> CoinConfig | None:
        """Get coin configuration by symbol (case-insensitive exact match)"""
        if not symbol:
            return None
        target = symbol.strip().upper()
        for coin in self.all_coins:
            if coin.symbol.upper() == target:
                return coin
        return None

    def get_allowlist_sync_status(self) -> dict[str, Any]:
        """
        Check whether the configured coins match the trading_universe (single source of truth).
        Returns a diagnostic dict for health endpoints.
        """
        # Import from single source of truth
        configured_symbols = {coin.symbol for coin in self.binance_coins}
        allowlist_symbols = set(TRADING_SYMBOLS)

        missing_in_config = sorted(allowlist_symbols - configured_symbols)
        extra_in_config = sorted(configured_symbols - allowlist_symbols)

        return {
            "in_sync": configured_symbols == allowlist_symbols,
            "missing_in_config": missing_in_config,
            "extra_in_config": extra_in_config,
            "configured_symbols": sorted(configured_symbols),
            "allowlist_symbols": sorted(allowlist_symbols),
            "sync_status": "perfect" if configured_symbols == allowlist_symbols else "mismatch",
        }

    def _validate_binance_allowlist(self) -> None:
        """Validate that all configured coins are in the trading_universe (single source of truth) - STRICT ENFORCEMENT"""
        self.allowlist_validation_status = "unknown"
        self.allowlist_validation_error = None

        # Use TRADING_SYMBOLS from trading_universe (single source of truth)
        configured_symbols = {coin.symbol for coin in self.binance_coins}
        allowlist_symbols = set(TRADING_SYMBOLS)

        if configured_symbols != allowlist_symbols:
            missing = allowlist_symbols - configured_symbols
            extra = configured_symbols - allowlist_symbols

            error_msg = "Binance allowlist validation FAILED: "
            if missing:
                error_msg += f"Missing symbols: {sorted(missing)}. "
            if extra:
                error_msg += f"Extra symbols: {sorted(extra)}. "
            error_msg += f"Expected: {sorted(allowlist_symbols)}, Got: {sorted(configured_symbols)}"

            self.allowlist_validation_status = "failed"
            self.allowlist_validation_error = error_msg
            logger.error(error_msg)

            # STRICT ENFORCEMENT: Fail fast for live trading
            msg = f"CRITICAL: Allowlist validation failed - {error_msg}"
            raise RuntimeError(msg)
        self.allowlist_validation_status = "passed"
        logger.info("Binance allowlist validation PASSED - all symbols match Top-10")

    def _validate_trade_amounts(self) -> None:
        """Validate trade amounts are within reasonable bounds"""
        for coin in self.binance_coins:
            if coin.min_trade_amount <= 0:
                msg = f"Invalid min trade amount for {coin.symbol}: {coin.min_trade_amount}"
                raise ValueError(msg)
            if coin.max_trade_amount <= coin.min_trade_amount:
                msg = f"Invalid max trade amount for {coin.symbol}: {coin.max_trade_amount} <= {coin.min_trade_amount}"
                raise ValueError(msg)
            if coin.max_trade_amount > 100000:  # Reasonable upper bound
                logger.warning(f"Very high max trade amount for {coin.symbol}: {coin.max_trade_amount}")

        # Mark as validated for downstream services
        self._trade_amounts_validated = True
        logger.info(f"Trade amounts validated: min={self.binance_coins[0].min_trade_amount}, max={self.binance_coins[0].max_trade_amount}")

    def get_coins_by_exchange(self, exchange: str) -> list[CoinConfig]:
        """Get all coins for a specific exchange - SINGLE CANONICAL ID ONLY"""
        # STRICT: Only accept EXCHANGE_ID from trading_universe (single source of truth)
        if exchange.lower() == EXCHANGE_ID.lower():
            return self.binance_coins

        # Log warning for deprecated aliases but don't support them
        if exchange.lower() in ("binance", "binanceus", "binance us", "binance-us"):
            logger.warning(f"Deprecated exchange alias '{exchange}' used - use '{EXCHANGE_ID}' instead")
            return self.binance_coins

        logger.warning(f"Unsupported exchange '{exchange}' - only '{EXCHANGE_ID}' is supported")
        return []

    def get_all_symbols(self) -> list[str]:
        """Get all coin symbols"""
        return [coin.symbol for coin in self.all_coins]

    def get_enabled_coins(self) -> list[CoinConfig]:
        """Get all enabled coins with env override support"""
        enabled_coins = []

        for coin in self.all_coins:
            # Check env override first
            env_key = f"COIN_ENABLED_{coin.symbol}"
            env_value = os.getenv(env_key)

            if env_value is not None:
                # Environment variable override
                coin_enabled = env_value.strip().lower() in {"1", "true", "on", "yes"}
                logger.debug(f"Coin {coin.symbol} enabled via env {env_key}: {coin_enabled}")
            else:
                # Use default enabled status
                coin_enabled = coin.enabled

            if coin_enabled:
                enabled_coins.append(coin)

        return enabled_coins

    def get_enabled_symbols(self) -> list[str]:
        """Get all enabled coin symbols with env override support"""
        return [coin.symbol for coin in self.get_enabled_coins()]

    def get_trade_amounts_config(self) -> dict[str, Any]:
        """Get trade amounts configuration for downstream services"""
        return {
            "min_trade_amount": self.binance_coins[0].min_trade_amount,
            "max_trade_amount": self.binance_coins[0].max_trade_amount,
            "source": "environment_with_defaults",
            "validation_status": "valid" if getattr(self, "_trade_amounts_validated", False) else "unknown",
        }

    def get_enabled_symbols_api(self) -> dict[str, Any]:
        """Get enabled symbols API response for UI components"""
        enabled_coins = self.get_enabled_coins()
        return {
            "enabled_symbols": [coin.symbol for coin in enabled_coins],
            "enabled_count": len(enabled_coins),
            "total_count": len(self.all_coins),
            "disabled_symbols": [coin.symbol for coin in self.all_coins if coin not in enabled_coins],
            "source": "environment_overrides_and_defaults",
        }

    def get_validation_status(self) -> dict[str, Any]:
        """Return a consolidated validation status used by health endpoints"""
        allowlist_status = {
            "status": getattr(self, "allowlist_validation_status", "unknown"),
            "error": getattr(self, "allowlist_validation_error", None),
        }
        trade_status = {"trade_amounts_validated": bool(getattr(self, "_trade_amounts_validated", False))}
        allowlist_sync = self.get_allowlist_sync_status()
        return {
            "allowlist": allowlist_status,
            "trade_amounts": trade_status,
            "allowlist_sync": allowlist_sync,
        }


# Global configuration instance
config = CryptoAutoEngineConfig()


def get_config() -> CryptoAutoEngineConfig:
    """Get the global configuration instance"""
    return config


def get_coin_config(symbol: str) -> CoinConfig | None:
    """Get coin configuration by symbol"""
    return config.get_coin_by_symbol(symbol)


def get_all_symbols() -> list[str]:
    """Get all coin symbols"""
    return config.get_all_symbols()


def get_enabled_symbols() -> list[str]:
    """Get all enabled coin symbols"""
    return config.get_enabled_symbols()


def get_current_timestamp() -> float:
    """Get current UTC timestamp as float epoch - SINGLE SOURCE OF TRUTH"""
    return time.time()


def get_current_time() -> datetime:
    """Get current UTC time as datetime object - SINGLE SOURCE OF TRUTH"""
    return datetime.now(timezone.utc)


def get_strategy_count() -> int:
    """Get strategy count - SINGLE SOURCE OF TRUTH aligned with trainer/leaderboard"""
    return config.strategy_config.strategy_count


def get_cache_schema() -> dict[str, Any]:
    """Get canonical cache schema for all consumers"""
    return {
        "cosmic_data": {
            "fields": ["data", "timestamp"],
            "note": "Single canonical timestamp field - no 'last_updated'",
            "timestamp_format": "UTC float epoch",
        },
        "price": {
            "fields": ["price", "timestamp"],
            "timestamp_format": "UTC float epoch",
        },
        "volume_24h": {
            "fields": ["volume", "timestamp"],
            "timestamp_format": "UTC float epoch",
        },
        "rsi": {
            "fields": ["value", "timestamp"],
            "timestamp_format": "UTC float epoch",
        },
        "macd": {
            "fields": ["value", "timestamp"],
            "value_structure": {
                "macd": "float",
                "signal": "float",
                "histogram": "float",
            },
            "timestamp_format": "UTC float epoch",
        },
        "strategy_signals": {
            "fields": ["signal", "confidence", "timestamp"],
            "timestamp_format": "UTC float epoch",
        },
        "trade_cooldowns": {"fields": ["until"], "timestamp_format": "UTC float epoch"},
    }
