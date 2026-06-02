"""
Cache or Redis Client - Live Configuration Only

Provides cache access with proper fallback chain to avoid duplicates.
Uses canonical_cache system as the primary cache to ensure consistency.
All configuration values come from live config - no hardcoded values.
"""

from __future__ import annotations

import logging
import os
from typing import Any

# Live service imports - only imported when needed to avoid circular dependencies
try:
    from backend.services.canonical_cache import canonical_cache
except (ImportError, AttributeError, ValueError, TypeError, RuntimeError):
    canonical_cache = None  # type: ignore[assignment,misc]

try:
    from backend.modules.ai.poller import get_cache as _ai_get_cache
except (ImportError, AttributeError, ValueError, TypeError, RuntimeError):
    _ai_get_cache = None  # type: ignore[assignment,misc]

# Import live configuration
try:
    from backend.config_bridge import get_mystic_config

    _mystic_config = get_mystic_config()
except (ImportError, AttributeError, ValueError, TypeError, RuntimeError):
    _mystic_config = None

logger = logging.getLogger(__name__)

# --- Live Configuration Helpers -------------------------------------------------------------------


def _should_use_ai_poller_cache() -> bool:
    """Determine if AI poller cache should be used from live config."""
    if _mystic_config:
        try:
            value = _mystic_config
            if hasattr(value, "cache") and hasattr(value.cache, "use_ai_poller_cache"):
                use_ai = value.cache.use_ai_poller_cache
                if isinstance(use_ai, bool):
                    return use_ai
        except (AttributeError, ValueError, TypeError):
            pass

    use_ai = os.getenv("CACHE_USE_AI_POLLER_CACHE", "").strip().lower()
    if use_ai in ("true", "1", "yes"):
        return True
    return use_ai not in ("false", "0", "no")


def _should_use_canonical_cache() -> bool:
    """Determine if canonical cache should be used from live config."""
    if _mystic_config:
        try:
            value = _mystic_config
            if hasattr(value, "cache") and hasattr(value.cache, "use_canonical_cache"):
                use_canonical = value.cache.use_canonical_cache
                if isinstance(use_canonical, bool):
                    return use_canonical
        except (AttributeError, ValueError, TypeError):
            pass

    use_canonical = os.getenv("CACHE_USE_CANONICAL_CACHE", "").strip().lower()
    if use_canonical in ("true", "1", "yes"):
        return True
    return use_canonical not in ("false", "0", "no")


_singleton_cache: Any | None = None


def _build_fallback_cache() -> Any:
    """Build fallback cache instance."""

    class _DataCache:
        def __init__(self) -> None:
            self.binance: dict[str, Any] = {}
            # Coinbase and CoinGecko removed - using Binance US only
            self.last_update: dict[str, Any] = {}

    return _DataCache()


def get_cache() -> Any:
    """Get cache instance with proper fallback chain to avoid duplicates.

    Priority order:
    1. Canonical cache (if enabled) - to avoid duplicates
    2. AI poller cache (if enabled and available)
    3. Fallback in-memory cache

    Returns:
        Cache instance (canonical_cache, AI poller cache, or fallback)
    """
    global _singleton_cache
    if _singleton_cache is not None:
        return _singleton_cache

    # Try canonical cache first to avoid duplicates
    if _should_use_canonical_cache():
        # Validate canonical_cache outside try to avoid TRY301
        if canonical_cache is None:
            logger.warning("canonical_cache not available, trying fallback")
        else:
            try:
                _singleton_cache = canonical_cache
                logger.info("Using canonical_cache to avoid duplicates")
            except (ImportError, AttributeError, ValueError, TypeError, RuntimeError) as e:
                logger.warning(f"Canonical cache unavailable: {e}, trying fallback")
            else:
                return _singleton_cache

    # Try AI poller cache if enabled
    if _should_use_ai_poller_cache():
        # Validate _ai_get_cache outside try to avoid TRY301
        if _ai_get_cache is None:
            logger.warning("_ai_get_cache not available, using fallback")
        else:
            try:
                _singleton_cache = _ai_get_cache()
                logger.info("Using AI poller cache")
            except (ImportError, AttributeError, ValueError, TypeError, KeyError, IndexError, RuntimeError) as e:
                logger.warning(f"AI poller cache unavailable: {e}, using fallback")
            else:
                return _singleton_cache

    # Fallback to basic in-memory cache
    _singleton_cache = _build_fallback_cache()
    logger.info("Using fallback in-memory cache")
    return _singleton_cache
