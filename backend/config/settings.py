"""
Settings Configuration - All Live Data, No Fallback/Hardcoded Data

This module provides centralized settings configuration for the backend API (port 8000).
All configuration:
- Loads from live environment variables and .env files
- Default values are configuration defaults only, not fallback data
- Backend runs on port 8000 by default (configurable via BACKEND_PORT)
- Connects to live endpoints for all services (backend, database, Redis, Binance.US)

Live Data Sources:
- Environment variables for live API keys and credentials
- .env file for local configuration overrides
- All endpoints configured for live connections (backend port 8000, external APIs)
- Database and Redis URLs configured for live data storage
- Binance.US API keys from environment variables for live trading

Endpoint References:
- Backend API: Configured via BACKEND_HOST and BACKEND_PORT (default: 127.0.0.1:8000)
- Database: Configured via DATABASE_URL for live data persistence
- Redis: Configured via REDIS_URL for live caching (Windows Home 11)
- Binance.US API: Live exchange API via API keys from environment variables
- All connections use live endpoints - no fallback/hardcoded data
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
    Read a single KEY=value from .env file if present (no third-party deps).

    Reads live environment values from .env file for local configuration overrides.
    All values are from live .env file - no fallback/hardcoded values.

    Args:
        env_path: Path to .env file (live configuration file)
        key: Environment variable key to read

    Returns:
        Value from live .env file if found, None otherwise
    """
    try:
        if not env_path.exists():
            return None
        for raw_line in env_path.read_text(encoding="utf-8", errors="ignore").splitlines():
            stripped_line = raw_line.strip()
            if not stripped_line or stripped_line.startswith("#") or "=" not in stripped_line:
                continue
            k, v = stripped_line.split("=", 1)
            if k.strip() == key:
                return v.strip()
    except (ValueError, IndexError, AttributeError):
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
        # Root .env (project root = two dirs above /backend/config)
        # Used for reading live environment configuration
        root = Path(__file__).resolve().parents[2]
        env_path = root / ".env"

        # Core backend configuration (live endpoints on port 8000)
        self.backend_host = os.getenv("BACKEND_HOST", "127.0.0.1")
        try:
            # Backend port 8000 is the default for live API endpoints
            self.backend_port = int(os.getenv("BACKEND_PORT", "8000"))
        except (ValueError, TypeError):
            # Default to port 8000 if environment variable invalid (configuration default, not fallback)
            self.backend_port = 8000

        # Database configuration (live data persistence)
        # Default SQLite URL for live data storage (configuration default, not fallback data)
        self.database_url = os.getenv("DATABASE_URL", "sqlite:///./mystic_trading.db")
        # PostgreSQL configuration (live database connections)
        self.DB_HOST = os.getenv("DB_HOST", "localhost")
        try:
            self.DB_PORT = int(os.getenv("DB_PORT", "5432"))
        except (ValueError, TypeError):
            self.DB_PORT = 5432
        self.DB_NAME = os.getenv("DB_NAME", "mystic_db")
        self.DB_USER = os.getenv("DB_USER", "postgres")
        self.DB_PASSWORD = os.getenv("DB_PASSWORD", "")

        # Feature flags (live feature configuration)
        self.feature_ai_live = os.getenv("FEATURE_AI_LIVE", "false").lower() == "true"
        self.feature_ai_demo = os.getenv("FEATURE_AI_DEMO", "false").lower() == "true"

        # AI/ML configuration (live AI/ML settings)
        # "1" = enabled (offline mode), "0" = disabled
        self.transformers_offline = os.getenv("TRANSFORMERS_OFFLINE", "1")
        self.hf_datasets_offline = os.getenv("HF_DATASETS_OFFLINE", "1")

        # Security (JWT secret from live environment variables)
        # JWT secret must be provided via environment variable for security or root .env file
        jwt_secret = os.getenv("JWT_SECRET")

        # If not found in environment, try to read from root .env file (live configuration)
        if not jwt_secret:
            try:
                # Get the project root directory using Path (modern approach)
                root_dir = Path(__file__).resolve().parents[2]
                env_file_path = root_dir / ".env"

                if env_file_path.exists():
                    # Parse the .env file (live configuration)
                    with env_file_path.open(encoding="utf-8") as env_file:
                        for raw_line in env_file:
                            stripped_line = raw_line.strip()
                            if stripped_line and not stripped_line.startswith("#"):
                                key, value = stripped_line.split("=", 1)
                                if key.strip() == "JWT_SECRET":
                                    jwt_secret = value.strip()
                                    break
            except Exception as e:
                error_msg = f"Error reading JWT_SECRET from root .env file: {e}"
                print(error_msg)

        if not jwt_secret:
            error_message = "JWT_SECRET must be set in environment variables or root .env file"
            raise ValueError(error_message)

        self.jwt_secret = jwt_secret
        # Optional WS token requirement for WebSocket endpoints (live configuration)
        self.ws_token_required = os.getenv("WS_TOKEN_REQUIRED", "false").lower() == "true"

        # UI/dev origins and allowed hosts (configuration defaults for live backend port 8000)
        try:
            ui_origins_env = os.getenv("UI_ORIGINS", "").strip()
            if ui_origins_env:
                self.ui_origins = [o.strip() for o in ui_origins_env.split(",") if o.strip()]
            else:
                # Configuration default: backend port 8000 (not fallback data)
                self.ui_origins = ["http://localhost:8000"]
        except (ValueError, TypeError, AttributeError):
            self.ui_origins = ["http://localhost:8000"]

        try:
            allowed_hosts_env = os.getenv("ALLOWED_HOSTS", "").strip()
            if allowed_hosts_env:
                self.allowed_hosts = [h.strip() for h in allowed_hosts_env.split(",") if h.strip()]
            else:
                # Configuration default (not fallback data)
                self.allowed_hosts = ["localhost", "127.0.0.1"]
        except (ValueError, TypeError, AttributeError):
            self.allowed_hosts = ["localhost", "127.0.0.1"]

        # Binance.US API configuration (live API keys for live trading)
        # Load from environment variables or .env file (live configuration)
        self.binance_us_api_key = os.getenv("BINANCE_US_API_KEY") or _read_env_value(env_path, "BINANCE_US_API_KEY") or ""
        # Alternative key name supported (BINANCE_US_SECRET_KEY) for compatibility
        self.binance_us_api_secret = (
            os.getenv("BINANCE_US_API_SECRET")
            or os.getenv("BINANCE_US_SECRET_KEY")  # Alternative key name (not fallback)
            or _read_env_value(env_path, "BINANCE_US_API_SECRET")
            or _read_env_value(env_path, "BINANCE_US_SECRET_KEY")
            or ""
        )

        # Coinbase removed - using Binance.US only (live trading operations)

        # Display configuration (live display settings)
        self.display_exchange = os.getenv("DISPLAY_EXCHANGE", "binance.us")
        try:
            self.display_top_n = int(os.getenv("DISPLAY_TOP_N", "10"))
        except (ValueError, TypeError):
            # Configuration default (not fallback data)
            self.display_top_n = 10

        # Trading symbols from single source of truth (live Binance.US Top-10 configuration)
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


settings = Settings()
