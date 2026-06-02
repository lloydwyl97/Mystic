"""
Unified Configuration System
Consolidates config into a single source of truth with MINIMAL IMPACT.
Environment variables are the primary source.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# Import from single source of truth
try:
    from backend.config.trading_universe import EXCHANGE_ID, TRADING_SYMBOLS
except (ImportError, ModuleNotFoundError, AttributeError, ValueError, TypeError, RuntimeError) as e:
    msg = f"Failed to import trading_universe: {e}"
    raise RuntimeError(msg) from e

# ---------- helpers ----------
TRUE_SET = {"1", "true", "yes", "on", "y", "t"}
FALSE_SET = {"0", "false", "no", "off", "n", "f"}


def _get_redis_host() -> str:
    """Get Redis host from environment (no fallback)"""
    host = os.getenv("REDIS_HOST")
    if not host:
        msg = "REDIS_HOST environment variable is required - no fallback/hardcoded Redis host"
        raise RuntimeError(msg)
    return host


def _get_redis_port() -> int:
    """Get Redis port from environment"""
    port = os.getenv("REDIS_PORT")
    if port:
        return getenv_int("REDIS_PORT", 6379)
    return 6379  # Default port only if not specified


def _get_redis_db() -> int:
    """Get Redis DB from environment"""
    db = os.getenv("REDIS_DB")
    if db:
        return getenv_int("REDIS_DB", 0)
    return 0  # Default DB only if not specified


def getenv_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    v = raw.strip().lower()
    if v in TRUE_SET:
        return True
    if v in FALSE_SET:
        return False
    logging.warning(f"Malformed boolean for {name}='{raw}', using default={default}")
    return default  # fall back if weird value


def getenv_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError):
        logging.warning(f"Malformed int for {name}='{os.getenv(name)}', using default={default}")
        return default


def getenv_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError):
        logging.warning(f"Malformed float for {name}='{os.getenv(name)}', using default={default}")
        return default


def getenv_list(name: str, default: list[str]) -> list[str]:
    raw = os.getenv(name)
    if not raw:
        return default
    # split by comma/space and strip
    # NOTE: This helper is for symbol-like lists. Do not use for values where spaces are significant.
    parts = [p.strip() for p in raw.replace(" ", ",").split(",") if p.strip()]
    return parts or default


@dataclass
class UnifiedConfig:
    """Single source of truth configuration"""

    # ---------- Redis ----------
    # All Live Data, No Fallback/Hardcoded Data
    redis_host: str = field(default_factory=_get_redis_host)
    redis_port: int = field(default_factory=_get_redis_port)
    redis_db: int = field(default_factory=_get_redis_db)

    # ---------- Trading ----------
    # NOTE: Explicit opt-in: defaults False to prevent unintended live actions
    trading_enabled: bool = field(default_factory=lambda: getenv_bool("TRADING_ENABLED", False))
    live_mode: bool = field(default_factory=lambda: getenv_bool("LIVE_MODE", False))
    max_order_notional: float = field(default_factory=lambda: getenv_float("MAX_ORDER_NOTIONAL_USD", 250.0))
    # Align with micro-account 2% risk-per-trade semantics by default
    position_size_pct: float = field(default_factory=lambda: getenv_float("POSITION_SIZE_PCT", 0.02))  # 2%

    # ---------- AI ----------
    ai_enabled: bool = field(default_factory=lambda: getenv_bool("AI_ENABLED", True))
    # Resolve to absolute path to avoid CWD issues under Windows services
    ai_model_path: str = field(default_factory=lambda: str(Path(os.getenv("AI_MODEL_PATH", "models")).resolve()))
    ai_training_enabled: bool = field(default_factory=lambda: getenv_bool("AI_TRAINING_ENABLED", True))

    # ---------- Service Ports ----------
    # Core services
    backend_port: int = field(default_factory=lambda: getenv_int("BACKEND_PORT", 8000))  # FastAPI backend
    ai_service_port: int = field(default_factory=lambda: getenv_int("AI_SERVICE_PORT", 8001))  # Separate process default to avoid collision
    ai_strategy_port: int = field(default_factory=lambda: getenv_int("AI_STRATEGY_PORT", 8002))  # AI strategy
    dashboard_port: int = field(default_factory=lambda: getenv_int("DASHBOARD_PORT", 8000))  # If served by backend, equals backend_port

    # Mining/Pool services (per your ports)
    btc_miner_port: int = field(default_factory=lambda: getenv_int("BTC_MINER_PORT", 8099))
    eth_miner_port: int = field(default_factory=lambda: getenv_int("ETH_MINER_PORT", 8100))
    mining_pool_port: int = field(default_factory=lambda: getenv_int("MINING_POOL_PORT", 8101))

    # ---------- API / Market Data ----------
    # Live-only rule: only Binance.US is supported
    external_apis_enabled: bool = field(default_factory=lambda: getenv_bool("EXTERNAL_APIS_ENABLED", False))
    binance_us_enabled: bool = field(default_factory=lambda: getenv_bool("BINANCE_US_ENABLED", True))

    # Hard rules: featured exchange/symbols (from trading_universe - live data)
    featured_exchange: str = field(default_factory=lambda: os.getenv("FEATURED_EXCHANGE", EXCHANGE_ID))
    featured_symbols: list[str] = field(default_factory=lambda: getenv_list("FEATURED_SYMBOLS", list(TRADING_SYMBOLS)))

    # ---------- Auto-Buy ----------
    autobuy_enabled: bool = field(default_factory=lambda: getenv_bool("AUTOBUY_ENABLED", True))
    rsi_threshold: float = field(default_factory=lambda: getenv_float("RSI_THRESHOLD", 30.0))
    min_quantity_usd: float = field(default_factory=lambda: getenv_float("MIN_QUANTITY_USD", 10.0))
    max_quantity_usd: float = field(default_factory=lambda: getenv_float("MAX_QUANTITY_USD", 1000.0))

    # ---------- Optional normalization ----------
    def __post_init__(self):
        # keep position size sane (0..1)
        if self.position_size_pct < 0:
            self.position_size_pct = 0.0
        if self.position_size_pct > 1:
            self.position_size_pct = 1.0

    # ---------- Back-compat dict ----------
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for backward compatibility"""
        return {
            "redis": {
                "host": self.redis_host,
                "port": self.redis_port,
                "db": self.redis_db,
            },
            "trading": {
                "enabled": self.trading_enabled,
                "live_mode": self.live_mode,
                "max_order_notional": self.max_order_notional,
                "position_size_pct": self.position_size_pct,
            },
            "ai": {
                "enabled": self.ai_enabled,
                "model_path": self.ai_model_path,
                "training_enabled": self.ai_training_enabled,
            },
            "ports": {
                "backend": self.backend_port,
                "ai_service": self.ai_service_port,
                "ai_strategy": self.ai_strategy_port,
                "dashboard": self.dashboard_port,
                "btc_miner": self.btc_miner_port,
                "eth_miner": self.eth_miner_port,
                "mining_pool": self.mining_pool_port,
            },
            "apis": {
                "external_enabled": self.external_apis_enabled,  # kept for back-compat; defaults False
                "binance_us_enabled": self.binance_us_enabled,
                "featured_exchange": self.featured_exchange,
                "featured_symbols": self.featured_symbols,
            },
            "autobuy": {
                "enabled": self.autobuy_enabled,
                "rsi_threshold": self.rsi_threshold,
                "min_quantity_usd": self.min_quantity_usd,
                "max_quantity_usd": self.max_quantity_usd,
            },
        }


# Unified config state - using dict to avoid global keyword
_unified_config_state: dict[str, UnifiedConfig | None] = {"instance": None}


def get_config() -> UnifiedConfig:
    """Get unified configuration instance"""
    if _unified_config_state["instance"] is None:
        _unified_config_state["instance"] = UnifiedConfig()
    return _unified_config_state["instance"]


def refresh_config() -> UnifiedConfig:
    """Refresh the singleton from current environment (Windows-friendly)."""
    _unified_config_state["instance"] = UnifiedConfig()
    return _unified_config_state["instance"]


# ---------- Backward-compat shims (no code changes elsewhere) ----------
def get_trading_config():
    return get_config().to_dict()["trading"]


def get_ai_config():
    return get_config().to_dict()["ai"]


def get_redis_config():
    return get_config().to_dict()["redis"]


def get_ports_config():
    return get_config().to_dict()["ports"]


def get_apis_config():
    return get_config().to_dict()["apis"]


def get_autobuy_config():
    return get_config().to_dict()["autobuy"]


def get_service_config():
    ports = get_ports_config()
    return {
        "default_port": ports.get("backend"),
        "ai_strategy_port": ports.get("ai_strategy"),
        "dashboard_port": ports.get("dashboard"),
    }
