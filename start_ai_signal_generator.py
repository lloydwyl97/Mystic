#!/usr/bin/env python3
"""Start RealTimeAISignalGenerator (DAY bundle writer) in a dedicated process."""

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
logger = logging.getLogger("ai_signal_generator")


async def _main() -> int:
    from backend.services.ai_signal_generator import get_signal_generator

    signal_gen = get_signal_generator()
    if signal_gen is None:
        logger.error("AI Signal Generator not available")
        return 1

    await signal_gen.start()
    logger.info("RealTimeAISignalGenerator running (DAY bundle primary writer)")

    stop_event = asyncio.Event()

    def _stop(*_a):
        logger.info("ai_signal_generator: stop signal received")
        stop_event.set()

    loop = asyncio.get_running_loop()
    for sig in (signal.SIGTERM, signal.SIGINT):
        try:
            loop.add_signal_handler(sig, _stop)
        except NotImplementedError:
            pass

    while not stop_event.is_set():
        if not signal_gen.is_running:
            logger.warning("Signal generator stopped unexpectedly — restarting")
            await signal_gen.start()
        try:
            await asyncio.wait_for(stop_event.wait(), timeout=60.0)
        except asyncio.TimeoutError:
            pass

    await signal_gen.stop()
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(_main()))
