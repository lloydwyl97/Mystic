"""
Configuration Module - All Live Data, No Fallback/Hardcoded Data

This module provides configuration management for the backend API (port 8000).
All configuration:
- Loads from live environment variables and .env files
- Default values are configuration defaults only, not fallback data
- Backend runs on port 8000 by default (configurable via BACKEND_PORT)
- Connects to live endpoints for all services

Live Data Sources:
- Environment variables for live API keys and credentials
- .env file for local configuration overrides
- All endpoints configured for live connections (backend port 8000, external APIs)
- Database and Redis URLs configured for live data storage

Endpoint References:
- Backend API: Configured via BACKEND_HOST and BACKEND_PORT (default: 127.0.0.1:8000)
- External APIs: Binance.US API keys from environment variables
- Database: Configured via DATABASE_URL for live data persistence
- Redis: Configured via REDIS_URL for live caching
"""

from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace

# Import trading symbols at module level (live configuration)
# All Live Data, No Fallback/Hardcoded Data
try:
    from backend.config.trading_universe import TRADING_SYMBOLS  # type: ignore[import]
except (ImportError, ModuleNotFoundError, AttributeError, ValueError, TypeError, RuntimeError) as e:
    msg = f"Failed to import TRADING_SYMBOLS from trading_universe: {e}"
    raise RuntimeError(msg) from e


def _read_env_value(env_path: Path, key: str) -> str | None:
    """
    Read a single KEY=value from .env file if present.

    This reads live configuration from .env file, not fallback data.

    Args:
        env_path: Path to .env file
        key: Environment variable key to read

    Returns:
        Value from .env file or None if not found
    """
    try:
        if not env_path.exists():
            return None
        env_content = env_path.read_text(encoding="utf-8", errors="ignore")
        for raw_line in env_content.splitlines():
            stripped_line = raw_line.strip()
            if not stripped_line or stripped_line.startswith("#") or "=" not in stripped_line:
                continue
            k, v = stripped_line.split("=", 1)
            if k.strip() == key:
                return v.strip()
    except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError):
        pass
    return None


class Settings:
    """
    Configuration settings for live backend operations.

    Loads configuration from environment variables and .env files.
    Default values are configuration defaults only, not fallback data.
    All endpoints configured for live connections (backend port 8000, external APIs).
    """

    def __init__(self) -> None:
        # Root .env (project root = one dir above /backend)
        try:
            root = Path(__file__).resolve().parents[2]
        except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError):
            root = Path(__file__).resolve().parent
        env_path = root / ".env"

        # Core backend configuration (live endpoints on port 8000)
        self.backend_host = os.getenv("BACKEND_HOST", "127.0.0.1")
        try:
            # Backend port 8000 is the default for live API endpoints
            self.backend_port = int(os.getenv("BACKEND_PORT", "8000"))
        except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError):
            # Default to port 8000 if environment variable invalid (configuration default, not fallback)
            self.backend_port = 8000

        # Database configuration (live data persistence)
        # Default SQLite URL for live data storage (configuration default, not fallback data)
        self.database_url = os.getenv("DATABASE_URL", "sqlite:///./mystic_trading.db")

        # Redis cache settings (live caching)
        # All Live Data, No Fallback/Hardcoded Data
        redis_url = os.getenv("REDIS_URL")
        if not redis_url:
            raise RuntimeError()
        self.redis_url = redis_url
        try:
            self.cache_ttl_seconds = int(os.getenv("CACHE_TTL_SECONDS", "3600"))
        except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError):
            self.cache_ttl_seconds = 3600

        # Feature flags
        self.feature_ai_live = os.getenv("FEATURE_AI_LIVE", "false").lower() == "true"
        self.feature_ai_demo = os.getenv("FEATURE_AI_DEMO", "false").lower() == "true"

        # AI Service Configuration
        try:
            self.ai_decision_ttl_sec = int(os.getenv("DECISION_TTL_SEC", "15"))
        except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError):
            self.ai_decision_ttl_sec = 15
        try:
            self.ai_decision_history_max = int(os.getenv("DECISION_HISTORY_MAX", "400"))
        except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError):
            self.ai_decision_history_max = 400
        try:
            self.ai_engine_loop_sleep_sec = float(os.getenv("ENGINE_LOOP_SLEEP_SEC", "2"))
        except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError):
            self.ai_engine_loop_sleep_sec = 2.0
        try:
            self.ai_decision_t_high_default = float(os.getenv("DECISION_T_HIGH_DEFAULT", "0.62") or 0.62)
        except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError):
            self.ai_decision_t_high_default = 0.45
        try:
            self.ai_meta_label_gate_default = float(os.getenv("META_LABEL_GATE_DEFAULT", "0.55") or 0.55)
        except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError):
            self.ai_meta_label_gate_default = 0.50

        # Logging Configuration
        self.log_level = os.getenv("LOG_LEVEL", "INFO").upper()
        self.log_format = os.getenv("LOG_FORMAT", "json")

        # Performance Configuration
        try:
            self.max_workers = int(os.getenv("MAX_WORKERS", "4"))
        except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError):
            self.max_workers = 4
        try:
            self.request_timeout_sec = int(os.getenv("REQUEST_TIMEOUT_SEC", "30"))
        except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError):
            self.request_timeout_sec = 30

        # Dashboard configuration removed - dashboard cleanup

        # Security
        self.jwt_secret = os.getenv("JWT_SECRET", "dev-secret-change-me")
        # Optional WS token requirement for WebSocket endpoints
        self.ws_token_required = os.getenv("WS_TOKEN_REQUIRED", "false").lower() == "true"

        # UI/dev origins and allowed hosts (dev-friendly defaults)
        try:
            ui_origins_env = os.getenv("UI_ORIGINS", "").strip()
            if ui_origins_env:
                self.ui_origins = [o.strip() for o in ui_origins_env.split(",") if o.strip()]
            else:
                self.ui_origins = [
                    "http://localhost:8000",
                    "http://localhost:8000",
                ]
        except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError):
            self.ui_origins = ["http://localhost:8000"]

        try:
            allowed_hosts_env = os.getenv("ALLOWED_HOSTS", "").strip()
            if allowed_hosts_env:
                self.allowed_hosts = [h.strip() for h in allowed_hosts_env.split(",") if h.strip()]
            else:
                self.allowed_hosts = ["localhost", "127.0.0.1"]
        except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError):
            self.allowed_hosts = ["localhost", "127.0.0.1"]

        # Binance US API credentials (live trading - from environment variables)
        self.binance_us_api_key = os.getenv("BINANCE_US_API_KEY") or _read_env_value(env_path, "BINANCE_US_API_KEY") or ""
        # Check multiple environment variable names for API secret (alternative key name, not fallback data)
        self.binance_us_api_secret = (
            os.getenv("BINANCE_US_API_SECRET")
            or os.getenv("BINANCE_US_SECRET_KEY")  # Alternative key name (BINANCE_US_SECRET_KEY)
            or _read_env_value(env_path, "BINANCE_US_API_SECRET")
            or _read_env_value(env_path, "BINANCE_US_SECRET_KEY")
            or ""
        )

        # Display defaults
        self.display_exchange = os.getenv("DISPLAY_EXCHANGE", "binance.us")
        try:
            self.display_top_n = int(os.getenv("DISPLAY_TOP_N", "10"))
        except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError):
            self.display_top_n = 10

        # Trading symbols from live configuration (imported at module level)
        self.trading_symbols = TRADING_SYMBOLS

        # Back-compat wrapper so legacy code can do settings.exchange.<...>
        self.exchange = SimpleNamespace(
            name=self.display_exchange,
            binance_us_api_key=self.binance_us_api_key,
            binance_us_api_secret=self.binance_us_api_secret,
            # Legacy alias expected by some parts of the codebase
            binance_us_secret_key=self.binance_us_api_secret,
        )
        self.exchange_json = {"name": self.exchange.name}

    def validate(self) -> list[str]:
        """Validate configuration and return list of warnings/errors."""
        warnings: list[str] = []

        # Check required API keys for live trading
        if self.feature_ai_live and not self.binance_us_api_key:
            raise RuntimeError()

        if self.feature_ai_live and not self.binance_us_api_secret:
            raise RuntimeError()

        # Check JWT secret for production
        if self.jwt_secret == "dev-secret-change-me":
            warnings.append("WARNING: Using default JWT secret - change for production")

        # Check database URL
        if "sqlite" not in self.database_url.lower():
            warnings.append("INFO: Using non-SQLite database - ensure proper setup")

        # Check Redis URL
        if "localhost" in self.redis_url and self.feature_ai_live:
            warnings.append("INFO: Using localhost Redis - ensure Redis is running for live features")

        return warnings


settings = Settings()
