#!/usr/bin/env python3
"""
API Key Bridge - Standardizes API key access across the platform
Ensures all components can access API keys regardless of variable name variations
"""

import logging

from backend.utils.binance_credentials import (
    get_binance_us_api_key as _resolve_api_key,
)
from backend.utils.binance_credentials import (
    get_binance_us_secret_key as _resolve_secret_key,
)
from backend.utils.binance_credentials import (
    sync_binance_us_env_aliases,
)

logger = logging.getLogger(__name__)


class APIKeyBridge:
    """
    Provides standardized access to API keys across different naming conventions.
    Canonical env names: BINANCE_US_API_KEY / BINANCE_US_SECRET_KEY.
    """

    def __init__(self) -> None:
        self._keys_cache = {}
        self._load_keys()

    def _load_keys(self) -> None:
        """Load and standardize all API keys from environment variables"""

        binance_us_key = _resolve_api_key() or None
        binance_us_secret = _resolve_secret_key() or None

        self._keys_cache = {
            "binance_us_api_key": binance_us_key,
            "binance_us_secret_key": binance_us_secret,
        }

        found_keys = [k for k, v in self._keys_cache.items() if v]
        if found_keys:
            logger.info("[OK] API Key Bridge loaded keys for: %s", ", ".join(found_keys))
        else:
            logger.warning("[WARN] No API keys found in environment variables")

    def get_binance_us_api_key(self) -> str | None:
        """Get Binance US API key"""
        return self._keys_cache.get("binance_us_api_key")

    def get_binance_us_secret_key(self) -> str | None:
        """Get Binance US secret key"""
        return self._keys_cache.get("binance_us_secret_key")

    def has_binance_us_credentials(self) -> bool:
        """Check if Binance US credentials are available"""
        return bool(self.get_binance_us_api_key() and self.get_binance_us_secret_key())

    def set_environment_variables(self) -> None:
        """Publish canonical credentials into legacy alias env vars for old readers."""
        sync_binance_us_env_aliases()


# Global instance
# API key bridge state - using dict to avoid global keyword
_api_key_bridge_state: dict[str, APIKeyBridge] = {"instance": APIKeyBridge()}


def get_api_key_bridge() -> APIKeyBridge:
    """Get the global API key bridge instance"""
    return _api_key_bridge_state["instance"]


def initialize_api_keys() -> None:
    """Initialize and standardize API keys across the platform"""
    _api_key_bridge_state["instance"] = APIKeyBridge()
    _api_key_bridge_state["instance"].set_environment_variables()
    logger.info("[OK] API Key Bridge initialized and environment variables standardized")


# Auto-initialize when imported
initialize_api_keys()
