#!/usr/bin/env python3
"""
Start the canonical AI Market Context loop.

Publishes ai_context:{symbol} and persists ai_context_snapshots for every
traded symbol on AI_CONTEXT_LOOP_SEC cadence. Required for the
ai_signal_generator's canonical context multiplier. Run as its own process.
"""

from __future__ import annotations

import asyncio
import logging
import signal
import sys
from pathlib import Path

try:
    from dotenv import load_dotenv

    load_dotenv(Path(__file__).resolve().parent / ".env", override=False)
except Exception:
    pass

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
logger = logging.getLogger("ai_market_context")


async def _main() -> int:
    from backend.services.ai_market_context import get_market_context_service

    svc = get_market_context_service()
    await svc.start()

    stop_event = asyncio.Event()

    def _stop(*_a):
        logger.info("ai_market_context: stop signal received")
        stop_event.set()

    loop = asyncio.get_running_loop()
    for sig in (signal.SIGTERM, signal.SIGINT):
        try:
            loop.add_signal_handler(sig, _stop)
        except NotImplementedError:
            pass

    await stop_event.wait()
    await svc.stop()
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(_main()))
