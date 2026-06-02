#!/usr/bin/env python3
"""
Start AI Learning Service — continuous training/retraining loop only.
Runs AITrainingDataPipeline (collection + continuous_learning_loop) in a separate process.
"""

# CRITICAL: Load .env FIRST before any imports
from dotenv import load_dotenv

load_dotenv()

# CRITICAL FIX: Force IPv4 for all connections (Binance US requirement)
import socket as _socket

_orig_getaddrinfo = _socket.getaddrinfo


def _ipv4_only_getaddrinfo(host, port, family=0, type=0, proto=0, flags=0):
    return _orig_getaddrinfo(host, port, _socket.AF_INET, type, proto, flags)


_socket.getaddrinfo = _ipv4_only_getaddrinfo

import asyncio
import logging
import sys

if not logging.getLogger().handlers:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(name)-20s | %(message)s",
        datefmt="%H:%M:%S",
    )
logger = logging.getLogger(__name__)

# Ensure backend is on path when run from mystic/
if __name__ == "__main__":
    sys.path.insert(0, ".")


async def main() -> None:
    pipeline = None
    try:
        from backend.ai_training_pipeline import get_ai_training_pipeline

        pipeline = get_ai_training_pipeline()
        if pipeline is None:
            logger.error("AI training pipeline not available")
            return
        await pipeline.start()
        logger.info("TRAINING LOOP STARTED")
        while True:
            await asyncio.sleep(60)
    except KeyboardInterrupt:
        logger.info("Shutting down AI learning service...")
        if pipeline is not None:
            pipeline.is_running = False
        logger.info("AI learning service stopped")
    except Exception as e:
        logger.exception("Error in AI learning service: %s", e)
        if pipeline is not None:
            pipeline.is_running = False


if __name__ == "__main__":
    asyncio.run(main())
