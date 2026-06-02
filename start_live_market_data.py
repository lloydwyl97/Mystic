#!/usr/bin/env python3
"""Start LiveMarketDataService ticker/OHLCV loops in a dedicated process."""

from __future__ import annotations

import asyncio
import logging
import signal
import sys

from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
logger = logging.getLogger("live_market_data")


async def _main() -> int:
    from backend.services.live_market_data import live_market_data_service

    if live_market_data_service is None:
        logger.error("live_market_data_service not available")
        return 1

    await live_market_data_service.start()
    logger.info("LiveMarketDataService ticker/ohlcv loops running")

    stop_event = asyncio.Event()

    def _stop(*_a):
        logger.info("live_market_data: stop signal received")
        stop_event.set()

    loop = asyncio.get_running_loop()
    for sig in (signal.SIGTERM, signal.SIGINT):
        try:
            loop.add_signal_handler(sig, _stop)
        except NotImplementedError:
            pass

    await stop_event.wait()
    await live_market_data_service.stop()
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(_main()))
