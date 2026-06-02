#!/usr/bin/env python3
"""
API Key Bridge - Standardizes API key access across the platform
Ensures all components can access API keys regardless of variable name variations
"""

import logging
import os

logger = logging.getLogger(__name__)


class APIKeyBridge:
    """
    Provides standardized access to API keys across different naming conventions.
    This bridge ensures backward compatibility while standardizing access.
    """

    def __init__(self) -> None:
        self._keys_cache = {}
        self._load_keys()

    def _load_keys(self) -> None:
        """Load and standardize all API keys from environment variables"""

        # Binance US API Key - try all possible variations
        binance_us_key = os.getenv("BINANCE_US_API_KEY") or os.getenv("BINANCEUS_API_KEY") or os.getenv("BINANCE_API_KEY")

        # Binance US Secret Key - try all possible variations
        binance_us_secret = os.getenv("BINANCE_US_SECRET_KEY") or os.getenv("BINANCEUS_API_SECRET") or os.getenv("BINANCE_SECRET_KEY")

        # Store in standardized format
        self._keys_cache = {
            "binance_us_api_key": binance_us_key,
            "binance_us_secret_key": binance_us_secret,
        }

        # Log which keys were found (without exposing the keys)
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
        """
        Set standardized environment variables for backward compatibility.
        This ensures all existing code continues to work.
        """

        # Set Binance US variables in all formats for compatibility
        api_key = self.get_binance_us_api_key()
        if api_key:
            os.environ["BINANCE_US_API_KEY"] = api_key
            os.environ["BINANCEUS_API_KEY"] = api_key
            os.environ["BINANCE_API_KEY"] = api_key

        secret_key = self.get_binance_us_secret_key()
        if secret_key:
            os.environ["BINANCE_US_SECRET_KEY"] = secret_key
            os.environ["BINANCEUS_API_SECRET"] = secret_key
            os.environ["BINANCE_SECRET_KEY"] = secret_key


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
