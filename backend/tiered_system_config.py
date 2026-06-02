#!/usr/bin/env python3
"""
Configuration for Tiered Signal System
Defines all settings and parameters for the three-tier signal architecture
"""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass, fields
from typing import Any

# Import from single source of truth
try:
    from backend.config.trading_universe import TRADING_SYMBOLS
except (ImportError, ModuleNotFoundError, AttributeError, ValueError, TypeError, RuntimeError) as e:
    msg = f"Failed to import TRADING_SYMBOLS from trading_universe: {e}"
    raise RuntimeError(msg) from e


@dataclass
class Tier1Config:
    """Tier 1: Real-Time Signals Configuration"""

    price_fetch_interval: int = 5  # 5-10 seconds
    momentum_fetch_interval: int = 10  # 10-15 seconds
    orderbook_fetch_interval: int = 15  # 15 seconds
    cache_ttl: int = 30  # seconds
    max_retries: int = 3
    retry_delay: int = 1

    # Supported coins per exchange
    # Supported coins for Binance US
    binance_coins: list[str] | None = None

    def __post_init__(self):
        if self.binance_coins is None:
            # All Live Data, No Fallback/Hardcoded Data
            self.binance_coins = list(TRADING_SYMBOLS)

        # Basic sanity checks
        for iv in (
            self.price_fetch_interval,
            self.momentum_fetch_interval,
            self.orderbook_fetch_interval,
        ):
            if iv <= 0:
                msg = "Tier1Config intervals must be positive"
                raise ValueError(msg)
        if self.cache_ttl <= 0:
            msg = "Tier1Config.cache_ttl must be positive"
            raise ValueError(msg)
        if self.max_retries < 0 or self.retry_delay <= 0:
            msg = "Tier1Config retry settings invalid"
            raise ValueError(msg)


@dataclass
class Tier2Config:
    """Tier 2: Tactical Strategy Configuration"""

    rsi_fetch_interval: int = 180  # 3 minutes
    volume_fetch_interval: int = 120  # 2 minutes
    volatility_fetch_interval: int = 300  # 5 minutes
    cache_ttl: int = 600  # 10 minutes

    # Technical indicator parameters
    rsi_period: int = 14
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9

    # Thresholds for signal generation
    rsi_oversold: float = 30.0
    rsi_overbought: float = 70.0
    macd_bullish: float = 0.001
    macd_bearish: float = -0.001

    def __post_init__(self):
        if any(
            iv <= 0
            for iv in (
                self.rsi_fetch_interval,
                self.volume_fetch_interval,
                self.volatility_fetch_interval,
            )
        ):
            msg = "Tier2Config intervals must be positive"
            raise ValueError(msg)
        if self.cache_ttl <= 0:
            msg = "Tier2Config.cache_ttl must be positive"
            raise ValueError(msg)


@dataclass
class Tier3Config:
    """Tier 3: Mystic/Cosmic Configuration"""

    schumann_fetch_interval: int = 3600  # 1 hour
    solar_fetch_interval: int = 3600  # 1 hour
    pineal_fetch_interval: int = 7200  # 2 hours
    cache_ttl: int = 7200  # 2 hours
    max_retries: int = 3
    retry_delay: int = 60

    # Cosmic alignment thresholds
    cosmic_alignment_min: float = 60.0
    earth_frequency_ideal: float = 7.83  # Hz

    def __post_init__(self):
        if any(
            iv <= 0
            for iv in (
                self.schumann_fetch_interval,
                self.solar_fetch_interval,
                self.pineal_fetch_interval,
            )
        ):
            msg = "Tier3Config intervals must be positive"
            raise ValueError(msg)
        if self.cache_ttl <= 0:
            msg = "Tier3Config.cache_ttl must be positive"
            raise ValueError(msg)
        if self.max_retries < 0 or self.retry_delay <= 0:
            msg = "Tier3Config retry settings invalid"
            raise ValueError(msg)


@dataclass
class TradeEngineConfig:
    """Trade Decision Engine Configuration"""

    decision_interval: int = 5  # 3-10 seconds
    cache_ttl: int = 60  # 1 minute
    min_confidence: float = 0.75  # Match AI_CONFIDENCE_THRESHOLD config
    max_confidence: float = 0.95

    # Signal thresholds
    price_deviation_threshold: float = 0.02  # 2%
    volume_spike_threshold: float = 0.2  # 20%
    momentum_flip_threshold: float = 0.05  # 5%

    # Trading thresholds
    rsi_oversold: float = 30.0
    rsi_overbought: float = 70.0
    macd_bullish: float = 0.001
    macd_bearish: float = -0.001
    cosmic_alignment_min: float = 60.0
    volatility_max: float = 80.0

    def __post_init__(self):
        if self.decision_interval <= 0 or self.cache_ttl <= 0:
            msg = "TradeEngineConfig intervals must be positive"
            raise ValueError(msg)
        if not (0.0 <= self.min_confidence <= 1.0 and 0.0 <= self.max_confidence <= 1.0 and self.max_confidence >= self.min_confidence):
            msg = "TradeEngineConfig confidence thresholds invalid"
            raise ValueError(msg)


@dataclass
class UnifiedManagerConfig:
    """Unified Signal Manager Configuration"""

    sync_interval: int = 10  # Sync all tiers every 10 seconds
    health_check_interval: int = 60  # Health check every minute
    cache_ttl: int = 300  # 5 minutes
    auto_restart: bool = True
    max_restart_attempts: int = 3

    def __post_init__(self):
        if any(iv <= 0 for iv in (self.sync_interval, self.health_check_interval, self.cache_ttl)):
            msg = "UnifiedManagerConfig intervals must be positive"
            raise ValueError(msg)
        if self.max_restart_attempts < 0:
            msg = "UnifiedManagerConfig.max_restart_attempts must be >= 0"
            raise ValueError(msg)


@dataclass
class RedisConfig:
    """Redis Configuration - All Live Data, No Fallback/Hardcoded Data"""

    url: str = ""
    host: str = ""
    port: int = 6379
    db: int = 0
    password: str | None = None
    decode_responses: bool = True
    socket_timeout: int = 5
    socket_connect_timeout: int = 5

    def __post_init__(self):
        # All Live Data, No Fallback/Hardcoded Data
        # Redis connection must be configured via environment variables
        if not self.url:
            redis_url = os.getenv("REDIS_URL")
            if redis_url:
                self.url = redis_url
            else:
                redis_host = os.getenv("REDIS_HOST")
                if not redis_host:
                    msg = "REDIS_URL or REDIS_HOST environment variable is required - no fallback/hardcoded Redis host"
                    raise RuntimeError(msg)
                self.host = redis_host
                self.port = int(os.getenv("REDIS_PORT", "6379"))
                self.db = int(os.getenv("REDIS_DB", "0"))
                self.url = f"redis://{self.host}:{self.port}/{self.db}"
        elif not self.host:
            # Parse URL if provided
            if self.url.startswith("redis://"):
                parts = self.url.replace("redis://", "").split("/")
                host_port = parts[0].split(":")
                self.host = host_port[0] if host_port else ""
                self.port = int(host_port[1]) if len(host_port) > 1 else 6379
                self.db = int(parts[1]) if len(parts) > 1 else 0

        if self.port <= 0 or self.db < 0:
            msg = "RedisConfig port/db must be non-negative and port > 0"
            raise ValueError(msg)
        if self.socket_timeout <= 0 or self.socket_connect_timeout <= 0:
            msg = "RedisConfig socket timeouts must be positive"
            raise ValueError(msg)


@dataclass
class APIConfig:
    """API Configuration"""

    binance_base_url: str = "https://api.binance.us/api/v3"
    noaa_base_url: str = "https://services.swpc.noaa.gov/json"
    schumann_base_url: str = "https://www2.irf.se/maggraphs/schumann"

    # Rate limiting
    requests_per_minute: int = 60
    max_concurrent_requests: int = 10

    def __post_init__(self):
        if self.requests_per_minute <= 0 or self.max_concurrent_requests <= 0:
            msg = "APIConfig rate limits must be positive"
            raise ValueError(msg)


@dataclass
class LoggingConfig:
    """Logging Configuration"""

    level: str = "INFO"
    log_format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    file_handler: str = "tiered_system.log"
    max_file_size: int = 10 * 1024 * 1024  # 10MB
    backup_count: int = 5

    def __post_init__(self):
        if self.max_file_size <= 0 or self.backup_count < 0:
            msg = "LoggingConfig limits must be positive/non-negative"
            raise ValueError(msg)


class TieredSystemConfig:
    """Complete configuration for the tiered signal system"""

    def __init__(self) -> None:
        self.tier1 = Tier1Config()
        self.tier2 = Tier2Config()
        self.tier3 = Tier3Config()
        self.trade_engine = TradeEngineConfig()
        self.unified_manager = UnifiedManagerConfig()
        self.redis = RedisConfig()
        self.api = APIConfig()
        self.logging = LoggingConfig()

    def to_dict(self) -> dict[str, Any]:
        """Convert configuration to a deep dictionary (safe for JSON)."""
        return {
            "tier1": asdict(self.tier1),
            "tier2": asdict(self.tier2),
            "tier3": asdict(self.tier3),
            "trade_engine": asdict(self.trade_engine),
            "unified_manager": asdict(self.unified_manager),
            "redis": asdict(self.redis),
            "api": asdict(self.api),
            "logging": asdict(self.logging),
        }

    @classmethod
    def _filter_fields(cls, datacls, data: dict[str, Any]) -> dict[str, Any]:
        """Filter a dict to only known dataclass fields to avoid TypeErrors."""
        valid = {f.name for f in fields(datacls)}
        return {k: v for k, v in (data or {}).items() if k in valid}

    @classmethod
    def from_dict(cls, config_dict: dict[str, Any]) -> TieredSystemConfig:
        """Create configuration from dictionary (ignores unknown keys)."""
        cfg = cls()

        if "tier1" in config_dict:
            cfg.tier1 = Tier1Config(**cls._filter_fields(Tier1Config, config_dict["tier1"]))
        if "tier2" in config_dict:
            cfg.tier2 = Tier2Config(**cls._filter_fields(Tier2Config, config_dict["tier2"]))
        if "tier3" in config_dict:
            cfg.tier3 = Tier3Config(**cls._filter_fields(Tier3Config, config_dict["tier3"]))
        if "trade_engine" in config_dict:
            cfg.trade_engine = TradeEngineConfig(**cls._filter_fields(TradeEngineConfig, config_dict["trade_engine"]))
        if "unified_manager" in config_dict:
            cfg.unified_manager = UnifiedManagerConfig(**cls._filter_fields(UnifiedManagerConfig, config_dict["unified_manager"]))
        if "redis" in config_dict:
            cfg.redis = RedisConfig(**cls._filter_fields(RedisConfig, config_dict["redis"]))
        if "api" in config_dict:
            cfg.api = APIConfig(**cls._filter_fields(APIConfig, config_dict["api"]))
        if "logging" in config_dict:
            cfg.logging = LoggingConfig(**cls._filter_fields(LoggingConfig, config_dict["logging"]))

        return cfg

    def get_optimized_config(self) -> TieredSystemConfig:
        """Get optimized configuration for high-frequency trading"""
        optimized = TieredSystemConfig()

        # Optimize Tier 1 for maximum speed
        optimized.tier1.price_fetch_interval = 3
        optimized.tier1.momentum_fetch_interval = 5
        optimized.tier1.orderbook_fetch_interval = 10
        optimized.tier1.cache_ttl = 15

        # Optimize Tier 2 for faster analysis
        optimized.tier2.rsi_fetch_interval = 60
        optimized.tier2.volume_fetch_interval = 60
        optimized.tier2.volatility_fetch_interval = 120

        # Optimize Tier 3 for more frequent cosmic checks
        optimized.tier3.schumann_fetch_interval = 1800
        optimized.tier3.solar_fetch_interval = 1800
        optimized.tier3.pineal_fetch_interval = 3600

        # Optimize trade engine for faster decisions
        optimized.trade_engine.decision_interval = 3

        # Optimize unified manager
        optimized.unified_manager.sync_interval = 5
        optimized.unified_manager.health_check_interval = 30

        return optimized

    def get_conservative_config(self) -> TieredSystemConfig:
        """Get conservative configuration for lower resource usage"""
        conservative = TieredSystemConfig()

        # Conservative Tier 1 settings
        conservative.tier1.price_fetch_interval = 10
        conservative.tier1.momentum_fetch_interval = 15
        conservative.tier1.orderbook_fetch_interval = 30
        conservative.tier1.cache_ttl = 60

        # Conservative Tier 2 settings
        conservative.tier2.rsi_fetch_interval = 300
        conservative.tier2.volume_fetch_interval = 300
        conservative.tier2.volatility_fetch_interval = 600

        # Conservative Tier 3 settings
        conservative.tier3.schumann_fetch_interval = 7200
        conservative.tier3.solar_fetch_interval = 7200
        conservative.tier3.pineal_fetch_interval = 14400

        # Conservative trade engine
        conservative.trade_engine.decision_interval = 10

        # Conservative unified manager
        conservative.unified_manager.sync_interval = 30
        conservative.unified_manager.health_check_interval = 120

        return conservative


# Default configuration instance
default_config = TieredSystemConfig()

# Predefined configurations
optimized_config = default_config.get_optimized_config()
conservative_config = default_config.get_conservative_config()


def get_config(config_type: str = "default") -> TieredSystemConfig:
    """Get configuration by type ('default' | 'optimized' | 'conservative')."""
    configs = {
        "default": default_config,
        "optimized": optimized_config,
        "conservative": conservative_config,
    }
    return configs.get(config_type, default_config)
