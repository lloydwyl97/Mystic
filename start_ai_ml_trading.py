#!/usr/bin/env python3
"""
Start AI ML Trading Service (using trained 124-feature models)
This replaces start_ai_live_trading.py with ML-based signal generator
"""

# CRITICAL: Load .env FIRST before any backend imports
from dotenv import load_dotenv

load_dotenv()

# CRITICAL: Force IPv4 BEFORE any backend imports (Binance US requirement)
import socket as _socket

_original_getaddrinfo = _socket.getaddrinfo


def _force_ipv4_getaddrinfo(*args, **kwargs):
    """Filter to only return IPv4 addresses."""
    responses = _original_getaddrinfo(*args, **kwargs)
    ipv4_responses = [r for r in responses if r[0] == _socket.AF_INET]
    return ipv4_responses if ipv4_responses else responses


_socket.getaddrinfo = _force_ipv4_getaddrinfo

# CRITICAL FIX: Windows ProactorEventLoop has bugs with async Redis connections
# Must be set BEFORE any asyncio operations
import sys

if sys.platform == "win32":
    import asyncio

    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

import asyncio
import logging

from backend.services.ai_live_autobuy_service import get_ai_live_autobuy_service
from backend.services.ai_signal_generator import get_signal_generator


def _normalize_non_root_stream_handlers() -> None:
    """Collapse duplicate per-logger stream handlers into root only."""
    manager = logging.Logger.manager
    for _, obj in manager.loggerDict.items():
        if not isinstance(obj, logging.Logger):
            continue
        removed = False
        for h in list(obj.handlers):
            if isinstance(h, logging.StreamHandler):
                obj.removeHandler(h)
                try:
                    h.close()
                except Exception:
                    pass
                removed = True
        if removed:
            obj.propagate = True


# Normalize root logging to a single stream handler to avoid duplicated output.
root = logging.getLogger()
for h in list(root.handlers):
    if isinstance(h, logging.StreamHandler):
        root.removeHandler(h)
        try:
            h.close()
        except Exception:
            pass
if not root.handlers:
    logging.basicConfig(level=logging.INFO, stream=sys.stdout)
_normalize_non_root_stream_handlers()
logger = logging.getLogger(__name__)


async def main():
    """Start the AI ML trading service with trained models"""
    signal_gen = None
    try:
        # Ensure pipeline_decisions table exists before any signal generator writes
        try:
            from backend.services.pipeline_schema_init import initialize_pipeline_schema

            initialize_pipeline_schema()
        except Exception as e:
            logger.warning("Pipeline schema init (non-fatal): %s", e)

        # Start AI Signal Generator (uses trained 124-feature ML models)
        logger.info("🧠 Starting AI Signal Generator (ML-based with trained models)...")
        signal_gen = get_signal_generator()

        if signal_gen is None:
            logger.error("AI Signal Generator not available - check dependencies")
            return

        await signal_gen.start()
        logger.info("✅ AI Signal Generator started - using trained ML models (124 features)")

        # Start the autobuy service (consumes signals from Redis)
        logger.info("🤖 Starting AI Live AutoBuy Service...")
        autobuy_service = get_ai_live_autobuy_service()
        await autobuy_service.start()

        logger.info("✅ AI Live AutoBuy Service started successfully!")
        logger.info(f"   Enabled: {autobuy_service.enabled}")
        logger.info(f"   Confidence Threshold: {autobuy_service.min_confidence}")
        logger.info("🚀 Service is now running with ML models and will process AI signals for live trading...")
        logger.info("📊 Signals are generated using RandomForest with 124 features + trained models")

        # Keep the service running
        while True:
            await asyncio.sleep(60)  # Check every minute

            # Monitor autobuy service
            if not autobuy_service._running:
                logger.warning("AutoBuy service stopped unexpectedly, restarting...")
                await autobuy_service.start()

            # Monitor signal generator
            if signal_gen and not signal_gen.is_running:
                logger.warning("Signal Generator stopped unexpectedly, restarting...")
                await signal_gen.start()

    except KeyboardInterrupt:
        logger.info("🛑 Shutting down AI ML Trading Service...")
        if signal_gen:
            await signal_gen.stop()
        try:
            svc = get_ai_live_autobuy_service()
            await svc.stop()
        except Exception:
            pass
    except Exception as e:
        logger.exception(f"❌ Error starting AI ML Trading Service: {e}")
        if signal_gen:
            await signal_gen.stop()
        try:
            svc = get_ai_live_autobuy_service()
            await svc.stop()
        except Exception:
            pass


if __name__ == "__main__":
    asyncio.run(main())
