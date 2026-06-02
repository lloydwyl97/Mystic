"""
Initialize Persistent Data Cache
Initializes the persistent cache without mock market data.
Prepares symbol universe for supported assets and writes a heartbeat.
"""

from __future__ import annotations

import asyncio
import inspect
import logging
from collections.abc import Sequence
from datetime import datetime, timezone

from backend.modules.ai.persistent_cache import get_persistent_cache

# Import from single source of truth
try:
    from backend.config.trading_universe import EXCHANGE_ID, TRADING_SYMBOLS
except (ImportError, ModuleNotFoundError, AttributeError, ValueError, TypeError, RuntimeError) as e:
    msg = f"Failed to import trading_universe: {e}"
    raise RuntimeError(msg) from e

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Use TRADING_SYMBOLS from trading_universe (live data)
USDT_PAIRS: tuple[str, ...] = tuple(TRADING_SYMBOLS)


async def _try_call(obj: object, candidates: Sequence[str], *args, **kwargs) -> bool:
    for name in candidates:
        fn = getattr(obj, name, None)
        if callable(fn):
            try:
                result = fn(*args, **kwargs)
                if inspect.isawaitable(result):
                    await result
            except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
                logger.debug("Method %s failed: %s", name, e)
            else:
                return True
        else:
            return False
    return False


async def init_persistent_sample_data() -> bool:
    """
    Initialize persistent cache safely:
    - Register symbol universe for binance_us
    - Write a heartbeat/metadata
    - Do not write any mock prices or external exchange data
    """
    try:
        cache = get_persistent_cache()

        # 1) Register symbol universe from trading_universe (live data)
        universe_args = (EXCHANGE_ID, list(USDT_PAIRS))
        registered = (
            await _try_call(cache, ("update_exchange_universe",), *universe_args)
            or await _try_call(cache, ("set_symbol_universe",), EXCHANGE_ID, list(USDT_PAIRS))
            or await _try_call(cache, ("set_supported_symbols",), EXCHANGE_ID, list(USDT_PAIRS))
            or await _try_call(cache, ("update_symbol_list",), EXCHANGE_ID, list(USDT_PAIRS))
        )

        if registered:
            logger.info("Registered %d symbols for %s", len(USDT_PAIRS), EXCHANGE_ID)
        else:
            logger.info("No symbol-universe setter found on cache. Skipped registering symbols.")

        # 2) Write a heartbeat/metadata without injecting prices (no mock data)
        heartbeat_payload = {
            "exchange": EXCHANGE_ID,
            "symbols": list(USDT_PAIRS),
            "initialized_at": datetime.now(timezone.utc).isoformat(),
            "note": "Initialized without mock prices",
        }

        wrote_meta = (
            await _try_call(cache, ("update_metadata",), heartbeat_payload)
            or await _try_call(cache, ("set_metadata",), heartbeat_payload)
            or await _try_call(cache, ("write_heartbeat",), heartbeat_payload)
        )
        if wrote_meta:
            logger.info("Wrote cache heartbeat metadata")
        else:
            logger.info("No metadata/heartbeat writer found on cache. Skipped metadata write.")

        # 3) Explicitly do NOT write any price data here
        #    No Coinbase/Coingecko/Kraken, and no fabricated prices.

        logger.info("Persistent cache initialization completed without mock data")
    except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError):
        logger.exception("Error initializing persistent cache")
        return False
    else:
        return True


if __name__ == "__main__":
    ok = asyncio.run(init_persistent_sample_data())
    if ok:
        logger.info("Persistent cache initialized successfully")
    else:
        logger.error("Persistent cache initialization failed")
