#!/usr/bin/env python3
"""
Database Initialization Script for Mystic Trading Platform
Creates and initializes the database with required tables and (optional) reference data.
Rules respected:
- No mock data insertion.
- Logging only (ASCII), no emojis or special characters.
- Windows PowerShell friendly, Python 3.12 compatible.
"""

from __future__ import annotations

import asyncio
import importlib
import inspect
import logging
import os
import sys
from collections.abc import Callable
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("db_init")

# Import database bootstrap function
try:
    from backend.database_init import bootstrap_database  # type: ignore[import-not-found]
except ImportError:
    bootstrap_database = None  # type: ignore[assignment]


def _resolve_database_url() -> str | None:
    """
    Resolve DATABASE_URL from environment (single source of truth).
    Returns the URL string or None if not found.
    """
    # All Live Data, No Fallback/Hardcoded Data
    # Get DATABASE_URL from environment only
    env_url = os.getenv("DATABASE_URL")
    if env_url and env_url.strip():
        return env_url

    return None


def init_database() -> bool:
    """
    Initialize the database by invoking database.init_db() and database.create_tables().
    """
    logger.info("Initializing database")

    # Direct imports for production
    db_url = _resolve_database_url()
    if db_url:
        logger.info("Database URL detected")
    else:
        logger.warning("DATABASE_URL not found in config or environment")

    # Initialize DB and create tables
    success = bootstrap_database()
    if success:
        logger.info("Database initialized and tables created")
    else:
        logger.error("Database initialization failed")
    return success


def create_initial_data() -> bool:
    """
    Optionally create initial reference data (no mock data).
    Will attempt to call init_data.init_reference_data() if present.
    Skips silently if module or function is absent.
    """
    logger.info("Checking for reference data initializer")

    try:
        try:
            init_data = importlib.import_module("init_data")
        except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError):
            logger.info("init_data module not found; skipping reference data initialization")
            return True  # Not critical

        fn: Callable[[], object] | None = getattr(init_data, "init_reference_data", None)  # type: ignore[attr-defined]
        if not callable(fn):
            logger.info("init_reference_data() not found; skipping reference data initialization")
            return True

        logger.info("Running reference data initialization")
        try:
            # If the function is a coroutine function, run it directly.
            if inspect.iscoroutinefunction(fn):
                asyncio.run(fn())  # type: ignore[arg-type]
            else:
                # Call the function; if it returns an awaitable, run it, otherwise assume synchronous.
                result = fn()
                if inspect.isawaitable(result):
                    asyncio.run(result)  # type: ignore[arg-type]
        except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
            logger.exception("Reference data initialization failed while running initializer: %s", e)
            return False
        else:
            logger.info("Reference data initialization completed")
            return True
    except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
        logger.exception("Reference data initialization failed: %s", e)
        return False


def main() -> int:
    """
    Main entry: ensure we are in the correct working directory, then init DB and optional data.
    """
    logger.info("Starting database initialization procedure")

    # If run from repo root, switch into backend directory when needed
    if not Path("main.py").exists() and Path("backend/main.py").exists():
        try:
            os.chdir("backend")
            logger.info("Changed working directory to backend/")
        except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
            logger.exception("Failed to change directory to backend/: %s", e)
            return 1

    if not init_database():
        logger.error("Database initialization failed")
        return 1

    # Optional reference data (no mock data). Controlled by env var INIT_REFERENCE_DATA (default true).
    init_refs = os.getenv("INIT_REFERENCE_DATA", "true").strip().lower() == "true"
    if init_refs:
        create_initial_data()
    else:
        logger.info("INIT_REFERENCE_DATA disabled; skipping reference data initialization")

    logger.info("Database initialization completed successfully")
    return 0


if __name__ == "__main__":
    sys.exit(main())
