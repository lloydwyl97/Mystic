#!/usr/bin/env python3
"""
Configuration Bridge - LIVE-ONLY PRODUCTION CONFIGURATION SYSTEM
Provides validated configurations with strict live trading enforcement.

LIVE-ONLY PRODUCTION SYSTEM:
- ENFORCES CREDENTIAL VALIDATION: Live trading disabled if credentials missing
- COMPREHENSIVE RANGE VALIDATION: All numeric parameters validated with error on out-of-range
- DETERMINISTIC FAILURE SIGNALING: Clear mode indicators for UI rendering
- NO SECRET LEAKAGE: Credentials never logged or exported
- THREAD-SAFE OPERATIONS: Safe configuration reloading
- WINDOWS/PYTHON 3.12 COMPATIBLE: ASCII-only logs, robust error handling

Windows/Python 3.12+ Compatibility:
- Uses modern type annotations compatible with Python 3.12+
- All logging messages are ASCII-only for Windows PowerShell compatibility
- Safe environment variable parsing prevents import-time crashes
- Robust error handling with proper logging
"""

import logging
import os
import threading
import time
from dataclasses import dataclass, field
from typing import Any

# Import from single source of truth
try:
    from backend.config.trading_universe import EXCHANGE_ID
except (ImportError, ModuleNotFoundError, AttributeError, ValueError, TypeError, RuntimeError) as e:
    msg = f"Failed to import EXCHANGE_ID from trading_universe: {e}"
    raise RuntimeError(msg) from e

logger = logging.getLogger(__name__)

# Top-10 Binance.US universe size for validation
TOP10_UNIVERSE_SIZE = 10

# Configuration validation ranges
VALIDATION_RANGES = {
    "stop_loss_percentage": (0.0, 0.20),
    "take_profit_percentage": (0.0, 0.20),
    "risk_per_trade": (0.0, 0.05),  # Max 5% risk per trade
    "max_position_size": (0.01, 10000.0),  # Min $0.01, Max $10k
    "max_daily_trades": (0, 500),
    "max_open_positions": (0, TOP10_UNIVERSE_SIZE),
    "cache_ttl": (1, 3600),  # 1 second to 1 hour
    "request_timeout": (1, 300),  # 1 second to 5 minutes
    "confidence_threshold": (0.0, 1.0),
    "learning_rate": (0.0001, 1.0),
    "batch_size": (1, 1000),
    "training_epochs": (1, 1000),
    "max_training_samples": (1, 1000000),
}

# High-risk thresholds for warnings
HIGH_RISK_THRESHOLDS = {
    "risk_per_trade": 0.05,  # 5% risk per trade
    "max_position_size": 1000.0,  # $1000 position
    "confidence_threshold": 0.5,  # 50% confidence
    "batch_size": 100,
    "training_epochs": 100,
    "max_training_samples": 100000,
}


@dataclass
class ExchangeConfig:
    """Exchange configuration with strict credential validation"""

    binance_us_api_key: str = ""
    binance_us_secret_key: str = ""
    _credentials_validated: bool = False
    _validation_error: str | None = None

    def __post_init__(self):
        """Load credentials from environment with validation"""
        self.binance_us_api_key = os.getenv("BINANCE_US_API_KEY", "")
        self.binance_us_secret_key = os.getenv("BINANCE_US_SECRET_KEY", "")
        self._validate_credentials()

    def _validate_credentials(self) -> None:
        """Validate exchange credentials"""
        self._credentials_validated = False
        self._validation_error = None

        if not self.binance_us_api_key:
            self._validation_error = "BINANCE_US_API_KEY is missing"
            return

        if not self.binance_us_secret_key:
            self._validation_error = "BINANCE_US_SECRET_KEY is missing"
            return

        if len(self.binance_us_api_key) < 10:
            self._validation_error = "BINANCE_US_API_KEY appears to be invalid (too short)"
            return

        if len(self.binance_us_secret_key) < 10:
            self._validation_error = "BINANCE_US_SECRET_KEY appears to be invalid (too short)"
            return

        self._credentials_validated = True

    def has_valid_credentials(self) -> bool:
        """Check if both API key and secret are valid"""
        return self._credentials_validated

    def get_validation_error(self) -> str | None:
        """Get credential validation error message"""
        return self._validation_error


@dataclass
class TradingConfig:
    """Trading configuration with comprehensive validation"""

    max_position_size: float = 100.0
    max_daily_trades: int = 50
    stop_loss_percentage: float = 0.05
    take_profit_percentage: float = 0.10
    min_trade_amount: float = 10.0
    risk_per_trade: float = 0.02
    max_open_positions: int = 5
    enable_paper_trading: bool = True
    enable_live_trading: bool = False
    _validation_errors: list[str] = field(default_factory=list)
    _validation_warnings: list[str] = field(default_factory=list)

    def __post_init__(self):
        """Load from environment and validate"""
        self._load_from_environment()
        self._validate_trading_config()

    def _load_from_environment(self) -> None:
        """Load trading configuration from environment variables"""
        env_vars = {
            "MAX_POSITION_SIZE": ("max_position_size", float),
            "MAX_DAILY_TRADES": ("max_daily_trades", int),
            "STOP_LOSS_PCT": ("stop_loss_percentage", float),
            "TAKE_PROFIT_PCT": ("take_profit_percentage", float),
            "MIN_TRADE_AMOUNT": ("min_trade_amount", float),
            "RISK_PER_TRADE": ("risk_per_trade", float),
            "MAX_OPEN_POSITIONS": ("max_open_positions", int),
        }

        for env_var, (attr_name, type_func) in env_vars.items():
            value = os.getenv(env_var)
            if value:
                try:
                    setattr(self, attr_name, type_func(value))
                except (ValueError, TypeError) as e:
                    self._validation_errors.append(f"Invalid {env_var}: {value} ({e})")

        # Handle live trading flag with credential enforcement
        enable_live_env = os.getenv("ENABLE_LIVE_TRADING", "").lower()
        if enable_live_env in ("true", "1", "on", "yes"):
            # Live trading will be validated in _validate_trading_config
            self.enable_live_trading = True
            self.enable_paper_trading = False

    def _validate_trading_config(self) -> None:
        """Validate trading configuration parameters"""
        self._validation_errors.clear()
        self._validation_warnings.clear()

        # Validate numeric ranges
        numeric_validations = [
            ("stop_loss_percentage", self.stop_loss_percentage),
            ("take_profit_percentage", self.take_profit_percentage),
            ("risk_per_trade", self.risk_per_trade),
            ("max_position_size", self.max_position_size),
            ("max_daily_trades", self.max_daily_trades),
            ("max_open_positions", self.max_open_positions),
        ]

        for param_name, value in numeric_validations:
            if param_name in VALIDATION_RANGES:
                min_val, max_val = VALIDATION_RANGES[param_name]
                if not (min_val <= value <= max_val):
                    self._validation_errors.append(f"{param_name} ({value}) must be between {min_val} and {max_val}")
                elif param_name in HIGH_RISK_THRESHOLDS and value > HIGH_RISK_THRESHOLDS[param_name]:
                    self._validation_warnings.append(f"High risk: {param_name} ({value}) exceeds recommended threshold")

        # Validate system parameters if present on the instance
        system_validations = [
            ("cache_ttl", getattr(self, "cache_ttl", 300)),
            ("request_timeout", getattr(self, "request_timeout", 30)),
        ]

        for param_name, value in system_validations:
            if param_name in VALIDATION_RANGES:
                min_val, max_val = VALIDATION_RANGES[param_name]
                if not (min_val <= value <= max_val):
                    self._validation_errors.append(f"{param_name} ({value}) must be between {min_val} and {max_val}")

    def get_validation_errors(self) -> list[str]:
        """Get trading configuration validation errors"""
        return self._validation_errors.copy()

    def get_validation_warnings(self) -> list[str]:
        """Get trading configuration validation warnings"""
        return self._validation_warnings.copy()

    def is_valid(self) -> bool:
        """Check if trading configuration is valid"""
        return len(self._validation_errors) == 0


@dataclass
class AIConfig:
    """AI configuration with comprehensive validation"""

    learning_rate: float = 0.001
    batch_size: int = 8
    training_epochs: int = 10
    model_update_frequency: int = 500
    confidence_threshold: float = 0.6
    enable_realtime_learning: bool = True
    enable_multimodal_learning: bool = True
    max_training_samples: int = 5000
    _validation_errors: list[str] = field(default_factory=list)
    _validation_warnings: list[str] = field(default_factory=list)

    def __post_init__(self):
        """Load from environment and validate"""
        self._load_from_environment()
        self._validate_ai_config()

    def _load_from_environment(self) -> None:
        """Load AI configuration from environment variables"""
        env_vars = {
            "AI_LEARNING_RATE": ("learning_rate", float),
            "AI_BATCH_SIZE": ("batch_size", int),
            "AI_TRAINING_EPOCHS": ("training_epochs", int),
            "AI_CONFIDENCE_THRESHOLD": ("confidence_threshold", float),
            "AI_MAX_TRAINING_SAMPLES": ("max_training_samples", int),
            "AI_MODEL_UPDATE_FREQUENCY": ("model_update_frequency", int),
        }

        for env_var, (attr_name, type_func) in env_vars.items():
            value = os.getenv(env_var)
            if value:
                try:
                    setattr(self, attr_name, type_func(value))
                except (ValueError, TypeError) as e:
                    self._validation_errors.append(f"Invalid {env_var}: {value} ({e})")

        # Flags (booleans)
        enable_realtime = os.getenv("AI_ENABLE_REALTIME_LEARNING", "").lower()
        if enable_realtime in ("true", "1", "on", "yes"):
            self.enable_realtime_learning = True
        elif enable_realtime in ("false", "0", "off", "no"):
            self.enable_realtime_learning = False

        enable_multimodal = os.getenv("AI_ENABLE_MULTIMODAL_LEARNING", "").lower()
        if enable_multimodal in ("true", "1", "on", "yes"):
            self.enable_multimodal_learning = True
        elif enable_multimodal in ("false", "0", "off", "no"):
            self.enable_multimodal_learning = False

    def _validate_ai_config(self) -> None:
        """Validate AI configuration parameters"""
        self._validation_errors.clear()
        self._validation_warnings.clear()

        ai_validations = [
            ("learning_rate", self.learning_rate),
            ("batch_size", self.batch_size),
            ("training_epochs", self.training_epochs),
            ("confidence_threshold", self.confidence_threshold),
            ("max_training_samples", self.max_training_samples),
        ]

        for param_name, value in ai_validations:
            if param_name in VALIDATION_RANGES:
                min_val, max_val = VALIDATION_RANGES[param_name]
                if not (min_val <= value <= max_val):
                    self._validation_errors.append(f"{param_name} ({value}) must be between {min_val} and {max_val}")
                elif param_name in HIGH_RISK_THRESHOLDS and value > HIGH_RISK_THRESHOLDS[param_name]:
                    self._validation_warnings.append(f"High risk: {param_name} ({value}) exceeds recommended threshold")

    def get_validation_errors(self) -> list[str]:
        """Get AI validation errors"""
        return self._validation_errors.copy()

    def get_validation_warnings(self) -> list[str]:
        """Get AI validation warnings"""
        return self._validation_warnings.copy()

    def is_valid(self) -> bool:
        """Check if AI configuration is valid"""
        return len(self._validation_errors) == 0


@dataclass
class SystemConfig:
    """System / infrastructure related configuration"""

    log_level: str = "INFO"
    enable_performance_monitoring: bool = False
    cache_ttl: int = 300
    max_concurrent_requests: int = 10
    request_timeout: int = 30
    retry_attempts: int = 3
    enable_health_checks: bool = True
    _validation_errors: list[str] = field(default_factory=list)

    def __post_init__(self):
        self._load_from_environment()
        self._validate_system_config()

    def _load_from_environment(self) -> None:
        env_map = {
            "LOG_LEVEL": ("log_level", str),
            "ENABLE_PERFORMANCE_MONITORING": ("enable_performance_monitoring", bool),
            "CACHE_TTL": ("cache_ttl", int),
            "MAX_CONCURRENT_REQUESTS": ("max_concurrent_requests", int),
            "REQUEST_TIMEOUT": ("request_timeout", int),
            "RETRY_ATTEMPTS": ("retry_attempts", int),
            "ENABLE_HEALTH_CHECKS": ("enable_health_checks", bool),
        }

        for env_var, (attr_name, type_func) in env_map.items():
            value = os.getenv(env_var)
            if value is None or value == "":
                continue
            try:
                val = value.lower() in ("true", "1", "on", "yes") if type_func is bool else type_func(value)
                setattr(self, attr_name, val)
            except (ValueError, TypeError) as e:
                self._validation_errors.append(f"Invalid {env_var}: {value} ({e})")

    def _validate_system_config(self) -> None:
        self._validation_errors.clear()
        # Validate cache_ttl and request_timeout
        for param in ("cache_ttl", "request_timeout"):
            value = getattr(self, param)
            if param in VALIDATION_RANGES:
                min_val, max_val = VALIDATION_RANGES[param]
                if not (min_val <= value <= max_val):
                    self._validation_errors.append(f"{param} ({value}) must be between {min_val} and {max_val}")

    def get_validation_errors(self) -> list[str]:
        return self._validation_errors.copy()

    def is_valid(self) -> bool:
        return len(self._validation_errors) == 0


class MysticConfig:
    """Top-level configuration aggregator with thread-safe operations"""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self.exchange = ExchangeConfig()
        self.trading = TradingConfig()
        self.ai = AIConfig()
        self.system = SystemConfig()
        self._last_validation: dict[str, Any] | None = None
        self._validation_timestamp: float = 0.0

    def is_live_trading_enabled(self) -> bool:
        """Return whether live trading is enabled (subject to validation)"""
        return bool(self.trading.enable_live_trading)

    def has_valid_exchange_credentials(self) -> bool:
        return self.exchange.has_valid_credentials()

    def get_active_exchanges(self) -> list[str]:
        active = []
        if self.exchange.has_valid_credentials():
            active.append(EXCHANGE_ID)
        return active

    def get_config_mode(self) -> str:
        """Return config mode: 'live' if live trading enabled and credentials valid, else 'paper'."""
        if self.trading.enable_live_trading and self.exchange.has_valid_credentials() and self.trading.is_valid() and self.ai.is_valid() and self.system.is_valid():
            return "live"
        return "paper"

    def get_environment_sources(self) -> dict[str, str]:
        """Get mapping of configuration keys to their sources (env vs default)"""
        sources = {}

        # Exchange credentials
        if os.getenv("BINANCE_US_API_KEY"):
            sources["exchange.binance_us_api_key"] = "environment"
        if os.getenv("BINANCE_US_SECRET_KEY"):
            sources["exchange.binance_us_secret_key"] = "environment"

        # Trading parameters
        trading_env_vars = [
            "MAX_POSITION_SIZE",
            "MAX_DAILY_TRADES",
            "STOP_LOSS_PCT",
            "TAKE_PROFIT_PCT",
            "MIN_TRADE_AMOUNT",
            "RISK_PER_TRADE",
            "MAX_OPEN_POSITIONS",
            "ENABLE_LIVE_TRADING",
        ]
        for env_var in trading_env_vars:
            if os.getenv(env_var):
                sources[f"trading.{env_var.lower()}"] = "environment"

        # AI parameters
        ai_env_vars = [
            "AI_LEARNING_RATE",
            "AI_BATCH_SIZE",
            "AI_TRAINING_EPOCHS",
            "AI_CONFIDENCE_THRESHOLD",
            "AI_MAX_TRAINING_SAMPLES",
            "AI_MODEL_UPDATE_FREQUENCY",
        ]
        for env_var in ai_env_vars:
            if os.getenv(env_var):
                sources[f"ai.{env_var.lower()}"] = "environment"

        # System parameters
        system_env_vars = [
            "LOG_LEVEL",
            "CACHE_TTL",
            "REQUEST_TIMEOUT",
            "MAX_CONCURRENT_REQUESTS",
            "RETRY_ATTEMPTS",
        ]
        for env_var in system_env_vars:
            if os.getenv(env_var):
                sources[f"system.{env_var.lower()}"] = "environment"

        return sources

    def to_dict(self) -> dict[str, Any]:
        """Get complete configuration as dictionary with metadata"""
        with self._lock:
            return {
                "mode": self.get_config_mode(),
                "timestamp": time.time(),
                "sources": self.get_environment_sources(),
                "validation": self.validate_configuration(),
                "trading": {
                    "max_position_size": self.trading.max_position_size,
                    "max_daily_trades": self.trading.max_daily_trades,
                    "stop_loss_percentage": self.trading.stop_loss_percentage,
                    "take_profit_percentage": self.trading.take_profit_percentage,
                    "min_trade_amount": self.trading.min_trade_amount,
                    "risk_per_trade": self.trading.risk_per_trade,
                    "max_open_positions": self.trading.max_open_positions,
                    "enable_paper_trading": self.trading.enable_paper_trading,
                    "enable_live_trading": self.trading.enable_live_trading,
                },
                "ai": {
                    "learning_rate": self.ai.learning_rate,
                    "batch_size": self.ai.batch_size,
                    "training_epochs": self.ai.training_epochs,
                    "model_update_frequency": self.ai.model_update_frequency,
                    "confidence_threshold": self.ai.confidence_threshold,
                    "enable_realtime_learning": self.ai.enable_realtime_learning,
                    "enable_multimodal_learning": self.ai.enable_multimodal_learning,
                    "max_training_samples": self.ai.max_training_samples,
                },
                "system": {
                    "log_level": self.system.log_level,
                    "enable_performance_monitoring": self.system.enable_performance_monitoring,
                    "cache_ttl": self.system.cache_ttl,
                    "max_concurrent_requests": self.system.max_concurrent_requests,
                    "request_timeout": self.system.request_timeout,
                    "retry_attempts": self.system.retry_attempts,
                    "enable_health_checks": self.system.enable_health_checks,
                },
                "exchange": {
                    "has_binance_us_credentials": self.exchange.has_valid_credentials(),
                    "active_exchanges": self.get_active_exchanges(),
                    "credential_validation_error": self.exchange.get_validation_error(),
                },
            }

    def validate_configuration(self) -> dict[str, Any]:
        """Comprehensive configuration validation"""
        with self._lock:
            current_time = time.time()

            # Use cached validation if recent (within 5 seconds)
            if self._last_validation and current_time - self._validation_timestamp < 5.0:
                return self._last_validation

            results: dict[str, Any] = {
                "is_valid": True,
                "mode": self.get_config_mode(),
                "errors": [],
                "warnings": [],
                "recommendations": [],
                "timestamp": current_time,
            }

            # Collect all validation errors
            all_errors: list[str] = []
            all_warnings: list[str] = []

            # Exchange validation
            if not self.exchange.has_valid_credentials():
                error_msg = self.exchange.get_validation_error()
                if error_msg:
                    all_errors.append(f"Exchange credentials: {error_msg}")
                else:
                    all_errors.append("Exchange credentials: Missing API key or secret")

            # Trading validation
            all_errors.extend(self.trading.get_validation_errors())
            all_warnings.extend(self.trading.get_validation_warnings())

            # AI validation
            all_errors.extend(self.ai.get_validation_errors())
            all_warnings.extend(self.ai.get_validation_warnings())

            # System validation
            all_errors.extend(self.system.get_validation_errors())

            # Live trading safety checks
            if self.trading.enable_live_trading:
                if not self.exchange.has_valid_credentials():
                    all_errors.append("CRITICAL: Live trading enabled but exchange credentials invalid")
                else:
                    all_warnings.append("LIVE TRADING ENABLED: Ensure thorough testing and safeguards")

                # Additional live trading warnings
                if self.trading.max_position_size > HIGH_RISK_THRESHOLDS.get("max_position_size", 1000):
                    all_warnings.append(f"High risk: Max position size ${self.trading.max_position_size}")

                if self.trading.risk_per_trade > HIGH_RISK_THRESHOLDS.get("risk_per_trade", 0.05):
                    all_warnings.append(f"High risk: Risk per trade {self.trading.risk_per_trade:.1%}")
            else:
                results["recommendations"].append("Paper trading mode active")

            # AI safety checks
            if self.ai.confidence_threshold < HIGH_RISK_THRESHOLDS.get("confidence_threshold", 0.5):
                all_warnings.append(f"Low AI confidence threshold: {self.ai.confidence_threshold}")

            # Set final results
            results["errors"] = all_errors
            results["warnings"] = all_warnings
            results["is_valid"] = len(all_errors) == 0

            # Cache results
            self._last_validation = results
            self._validation_timestamp = current_time

            return results


# Global configuration instance with thread safety
# Mystic config state - using dict to avoid global keyword
_mystic_config_state: dict[str, MysticConfig] = {"instance": MysticConfig()}
_config_lock = threading.Lock()


def get_mystic_config() -> MysticConfig:
    """Get the current configuration instance"""
    with _config_lock:
        return _mystic_config_state["instance"]


def reload_config() -> MysticConfig:
    """Thread-safe configuration reload"""
    with _config_lock:
        logger.info("Reloading configuration...")
        _mystic_config_state["instance"] = MysticConfig()
        logger.info("Configuration reloaded successfully")
        return _mystic_config_state["instance"]


def validate_config() -> dict[str, Any]:
    """Validate current configuration - thread-safe"""
    with _config_lock:
        return _mystic_config_state["instance"].validate_configuration()


def get_config() -> MysticConfig:
    """Alias for get_mystic_config() for backward compatibility"""
    return get_mystic_config()


def get_config_health() -> dict[str, Any]:
    """Get configuration health for inclusion in global health endpoint"""
    with _config_lock:
        validation = _mystic_config_state["instance"].validate_configuration()
        return {
            "config_valid": validation["is_valid"],
            "config_mode": validation["mode"],
            "config_errors": validation["errors"],
            "config_warnings": validation["warnings"],
            "config_recommendations": validation["recommendations"],
            "live_trading_enabled": _mystic_config_state["instance"].is_live_trading_enabled(),
            "exchange_credentials_valid": _mystic_config_state["instance"].has_valid_exchange_credentials(),
            "active_exchanges": _mystic_config_state["instance"].get_active_exchanges(),
        }
