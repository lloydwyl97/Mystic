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


_SCALP_INGEST_INTERVAL_SEC = 300  # ingest scalp outcomes every 5 minutes
_DAY_SHADOW_RESOLVE_INTERVAL_SEC = 300  # resolve DAY shadow rejects on closed bars


async def _scalp_ingest_loop() -> None:
    """Periodic background ingestion of closed SCALP outcomes into scalp_learning_outcomes."""
    while True:
        try:
            from backend.services.ai_learning_ingestion import ingest_scalp_outcomes

            result = ingest_scalp_outcomes()
            if result.get("ingested", 0) > 0:
                logger.info("SCALP_OUTCOME_INGEST ingested=%d skipped=%d errors=%d", result.get("ingested", 0), result.get("skipped", 0), result.get("errors", 0))
        except Exception as exc:
            logger.debug("SCALP_OUTCOME_INGEST_SKIPPED %s", exc)
        await asyncio.sleep(_SCALP_INGEST_INTERVAL_SEC)


async def _day_shadow_resolve_loop() -> None:
    """Periodic closed-bar resolution of DAY gate shadow rejects (no orders)."""
    while True:
        try:
            from backend.database_schema import DATABASE_PATH
            from backend.services.day_gate_telemetry import resolve_shadow_rejects_async

            result = await resolve_shadow_rejects_async(DATABASE_PATH, max_rows=40)
            if int(result.get("resolved") or 0) > 0:
                logger.info(
                    "DAY_SHADOW_RESOLVE resolved=%s scanned=%s expired=%s",
                    result.get("resolved"),
                    result.get("scanned"),
                    result.get("expired"),
                )
        except Exception as exc:
            logger.debug("DAY_SHADOW_RESOLVE_SKIPPED %s", exc)
        await asyncio.sleep(_DAY_SHADOW_RESOLVE_INTERVAL_SEC)


async def _scalp_shadow_resolve_loop() -> None:
    """Periodic closed-bar resolution of SCALP gate shadow rejects (no orders)."""
    while True:
        try:
            from backend.services.binance_scalp.config import get_scalp_config
            from backend.services.scalp_gate_telemetry import resolve_shadow_rejects_async

            db = get_scalp_config().database_path
            result = await resolve_shadow_rejects_async(db, max_rows=40)
            if int(result.get("resolved") or 0) > 0:
                logger.info(
                    "SCALP_SHADOW_RESOLVE resolved=%s scanned=%s expired=%s",
                    result.get("resolved"),
                    result.get("scanned"),
                    result.get("expired"),
                )
        except Exception as exc:
            logger.debug("SCALP_SHADOW_RESOLVE_SKIPPED %s", exc)
        await asyncio.sleep(_DAY_SHADOW_RESOLVE_INTERVAL_SEC)


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
        asyncio.ensure_future(_scalp_ingest_loop())
        asyncio.ensure_future(_day_shadow_resolve_loop())
        asyncio.ensure_future(_scalp_shadow_resolve_loop())
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
