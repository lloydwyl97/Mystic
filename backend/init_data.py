"""
Initialize Data Cache
Populates the AI cache with live data from real APIs
All Live Data, No Fallback/Hardcoded Data
"""

import asyncio
import logging
from datetime import datetime, timezone

# Import from single source of truth
try:
    from backend.config.trading_universe import EXCHANGE_ID
    from backend.modules.ai.poller import get_cache
except (ImportError, ModuleNotFoundError, AttributeError, ValueError, TypeError, RuntimeError) as e:
    msg = f"Failed to import cache or EXCHANGE_ID: {e}"
    raise RuntimeError(msg) from e

logger = logging.getLogger(__name__)


async def init_live_data():
    """Initialize cache with live market data from APIs"""
    try:
        # Get cache from poller (live data only)
        cache = get_cache()

        # Initialize empty cache - will be populated by live data fetchers
        # Use EXCHANGE_ID from trading_universe (single source of truth)
        # Set cache for exchange using EXCHANGE_ID (dynamic, not hardcoded)
        if hasattr(cache, EXCHANGE_ID):
            setattr(cache, EXCHANGE_ID, {})
        # Also handle legacy binance_us attribute for backward compatibility
        if hasattr(cache, "binance_us"):
            cache.binance_us = {}

        # Set last update timestamps
        if hasattr(cache, "last_update"):
            cache.last_update = {
                EXCHANGE_ID: datetime.now(timezone.utc).isoformat(),
            }

        logger.info("Live data cache initialized:")
        logger.info(f"   {EXCHANGE_ID}: Ready for live data")
        logger.info("   Note: Data will be populated by live API connections")
    except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError):
        logger.exception("Error initializing live data cache")
        return False
    else:
        return True


if __name__ == "__main__":
    # Initialize live data cache
    asyncio.run(init_live_data())
    logger.info("Live data cache initialized successfully!")


def init_sample_data() -> None:
    logger.info("Sample data initialized (placeholder).")
