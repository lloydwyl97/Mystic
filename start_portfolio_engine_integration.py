#!/usr/bin/env python3
"""
Start Portfolio Engine Integration
Standalone launcher for isolated operation
"""

# CRITICAL: Load .env FIRST before any imports
import os

from dotenv import load_dotenv

load_dotenv()
# Launch-discipline proof: effective env seen by this process before engine imports.
print(
    "LAUNCH_ENV_ENGINE_KEYS "
    f"MYSTIC_CHURN_RATIO_LIMIT={os.getenv('MYSTIC_CHURN_RATIO_LIMIT')!r} "
    f"MYSTIC_SPREAD_SPIKE_THRESHOLD={os.getenv('MYSTIC_SPREAD_SPIKE_THRESHOLD')!r} "
    f"MYSTIC_SPREAD_BLOCK_BARS={os.getenv('MYSTIC_SPREAD_BLOCK_BARS')!r} "
    f"MAX_SPREAD_PCT={os.getenv('MAX_SPREAD_PCT')!r}",
    flush=True,
)

# CRITICAL FIX: Force IPv4 for all connections (IPv6 broken on many servers)
import socket as _socket

_orig_getaddrinfo = _socket.getaddrinfo


def _ipv4_only_getaddrinfo(host, port, family=0, type=0, proto=0, flags=0):
    return _orig_getaddrinfo(host, port, _socket.AF_INET, type, proto, flags)


_socket.getaddrinfo = _ipv4_only_getaddrinfo

# CRITICAL FIX: Windows ProactorEventLoop has bugs with async Redis connections
# Must be set BEFORE any asyncio operations
import sys
from pathlib import Path

if sys.platform == "win32":
    import asyncio

    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

import asyncio
import logging
import sys

# ONE sink: stdout only (caller redirects stdout/stderr to /tmp/mystic_portfolio.log)
# Remove any FileHandler for that path to avoid duplicate writes
TARGET_LOG = "/tmp/mystic_portfolio.log"
FMT = "%(asctime)s | %(levelname)-8s | %(name)-20s | %(message)s"
DFMT = "%H:%M:%S"


def _remove_file_handler_for_target(log: logging.Logger, target: str) -> None:
    """Remove and close any handler that writes to target file."""
    for h in list(log.handlers):
        if getattr(h, "baseFilename", None) == target:
            log.removeHandler(h)
            try:
                h.close()
            except Exception:
                pass


def _normalize_non_root_stream_handlers(exempt_loggers: set[str] | None = None) -> None:
    """
    Remove per-logger stream handlers so logs flow through one sink.
    Keep handlers only for explicitly exempted loggers.
    """
    exempt = exempt_loggers or set()
    manager = logging.Logger.manager
    for name, obj in manager.loggerDict.items():
        if not isinstance(obj, logging.Logger):
            continue
        if name in exempt:
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


# Root: remove file handlers for target and collapse to one StreamHandler only.
root = logging.getLogger()
_remove_file_handler_for_target(root, TARGET_LOG)
for h in list(root.handlers):
    if isinstance(h, logging.StreamHandler):
        root.removeHandler(h)
        try:
            h.close()
        except Exception:
            pass
if not root.handlers:
    logging.basicConfig(level=logging.INFO, format=FMT, datefmt=DFMT, stream=sys.stdout)

logger = logging.getLogger(__name__)

# Integration logger: StreamHandler only; no FileHandler for target; no propagation
_int_log = logging.getLogger("backend.services.portfolio_engine_integration")
_int_log.propagate = False
_remove_file_handler_for_target(_int_log, TARGET_LOG)
for h in list(_int_log.handlers):
    _int_log.removeHandler(h)
    try:
        h.close()
    except Exception:
        pass
_sh = logging.StreamHandler(sys.stdout)
_sh.setFormatter(logging.Formatter(FMT, datefmt=DFMT))
_sh.setLevel(logging.INFO)
_int_log.addHandler(_sh)
# Collapse duplicate stream handlers created by imported modules.
_normalize_non_root_stream_handlers(exempt_loggers={"backend.services.portfolio_engine_integration"})


def _proof_logging_config() -> None:
    """Proof: print handler list + propagate for root and integration logger."""
    root = logging.getLogger()
    log = logging.getLogger("backend.services.portfolio_engine_integration")
    print("INTEGRATION handlers:", [(type(h).__name__, getattr(h, "baseFilename", None)) for h in log.handlers])
    print("INTEGRATION propagate:", log.propagate)
    print("ROOT handlers:", [(type(h).__name__, getattr(h, "baseFilename", None)) for h in root.handlers])
    print("ROOT propagate:", root.propagate)


PID_FILE = "/tmp/mystic_pe_integration.pid"


def _check_singleton() -> None:
    """Abort if another PE integration process is already running."""
    if os.path.exists(PID_FILE):
        try:
            with open(PID_FILE) as f:
                old_pid = int(f.read().strip())
            os.kill(old_pid, 0)
            logger.error("ABORT: PE integration already running (PID %d). Kill it first.", old_pid)
            sys.exit(1)
        except (ProcessLookupError, ValueError):
            pass
    with open(PID_FILE, "w") as f:
        f.write(str(os.getpid()))
    import atexit

    def _unlink_pid_file() -> None:
        p = Path(PID_FILE)
        if p.exists():
            p.unlink()

    atexit.register(_unlink_pid_file)


async def main():
    """Start the Portfolio Engine Integration"""
    _check_singleton()
    integration = None
    try:
        logger.info("Warming up Redis...")
        from backend.config.redis_config import SharedRedisState

        redis_client = SharedRedisState.get_async_client()
        if not redis_client:
            raise RuntimeError("Redis async client unavailable during startup")
        redis_ready = False
        for attempt in range(5):
            try:
                await asyncio.wait_for(redis_client.ping(), timeout=30.0)
                logger.info(f"Redis ready (attempt {attempt + 1})")
                redis_ready = True
                break
            except Exception:
                if attempt < 4:
                    await asyncio.sleep(3)
        if not redis_ready:
            raise RuntimeError("Redis warmup failed after retries")

        logger.info("=" * 70)
        logger.info("STARTING PORTFOLIO ENGINE INTEGRATION")
        logger.info("=" * 70)

        # Import the service
        from backend.services.portfolio_engine_integration import start_portfolio_integration

        # Imported modules may attach extra stream handlers; normalize again.
        _normalize_non_root_stream_handlers(exempt_loggers={"backend.services.portfolio_engine_integration"})

        logger.info("Initializing Portfolio Engine Integration...")

        # Start the service
        integration = await start_portfolio_integration()

        logger.info("=" * 70)
        logger.info("PORTFOLIO ENGINE INTEGRATION STARTED SUCCESSFULLY!")
        logger.info("=" * 70)
        logger.info(f"Service Running: {integration.is_running}")
        logger.info(f"Monitor Task: {integration._monitor_task}")
        logger.info(f"Bar Processor Task: {integration._bar_processor_task}")
        logger.info("=" * 70)
        logger.info("Integration is now executing AI trades...")
        logger.info("Press Ctrl+C to stop")
        logger.info("=" * 70)

        # Keep running
        while integration.is_running:
            await asyncio.sleep(60)

    except KeyboardInterrupt:
        logger.info("\nShutting down Portfolio Engine Integration...")
    except Exception as e:
        logger.exception(f"Error in Portfolio Engine Integration: {e}")
    finally:
        if integration is not None:
            try:
                await integration.stop()
            except Exception as stop_err:
                logger.warning("Integration stop failed: %s", stop_err)


if __name__ == "__main__":
    asyncio.run(main())
