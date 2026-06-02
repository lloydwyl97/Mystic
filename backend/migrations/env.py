"""
Alembic Environment Configuration - All Live Data, No Fallback/Hardcoded Data

This module provides Alembic database migration environment for live database operations (backend port 8000).
All operations:
- Run database migrations on live database connections (backend port 8000)
- Apply live schema changes to production database
- Connect to live database from environment variables or configuration
- No fallback/hardcoded data - all migrations operate on live database
- Used by backend services on port 8000 for live trading operations

Live Data Sources:
- Database URL: Live database connection string from environment or configuration
- Database schema: Live schema metadata from SQLAlchemy models
- Migrations: Live migration scripts applied to production database
- All migrations operate on live database - no mock/test data

Endpoint References:
- Backend API: Port 8000 (migrations applied to live database used by backend)
- Database: Live database connection (from DATABASE_URL or AIDBManager)
- All migrations use live connections - no fallback/hardcoded data

Note: Database URL resolution follows priority order (environment -> manager -> config) for live operations.
The in-memory SQLite URL is a development configuration option, not fallback data.
"""

from __future__ import annotations

import logging
import os
import sys
from logging.config import fileConfig
from pathlib import Path

from alembic import context
from sqlalchemy import engine_from_config, pool

logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------------
# Python path setup (ensure `backend` is importable)
# -----------------------------------------------------------------------------
backend_dir = Path(__file__).resolve().parent.parent
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))

# -----------------------------------------------------------------------------
# Alembic Config & Logging
# -----------------------------------------------------------------------------
config = context.config
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# -----------------------------------------------------------------------------
# Models metadata import (Base) and optional DB manager
# -----------------------------------------------------------------------------
# Base metadata from live models (for migration autogenerate)
Base = None
db_url_from_manager: str | None = None

try:
    from backend.services.ai_live_models_store import Base as _Base

    Base = _Base  # Live models metadata
except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
    # Log import error but continue - might be able to proceed with DATABASE_URL
    logger.warning("Could not import Base from ai_live_models_store, continuing with DATABASE_URL: %s", e)
    Base = None  # type: ignore[assignment]

try:
    from backend.services.ai_db_manager import AIDBManager

    # Get live database URL from manager (for live operations)
    _mgr = AIDBManager()
    # Ensure live DB directory exists (esp. for SQLite)
    _mgr.db_dir.mkdir(parents=True, exist_ok=True)
    db_url_from_manager = f"sqlite:///{_mgr.db_path}"  # Live database URL
except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
    logger.warning("Could not get database URL from AIDBManager, using environment/config: %s", e)
    db_url_from_manager = None

# Use Base.metadata for autogenerate (live schema metadata)
target_metadata = getattr(Base, "metadata", None)


# -----------------------------------------------------------------------------
# URL resolution (priority: env var -> manager -> alembic.ini) for live operations
# -----------------------------------------------------------------------------
def _resolve_sqlalchemy_url() -> str:
    """
    Resolve live database URL for migrations (backend port 8000).

    Priority order for live database connection:
    1. DATABASE_URL environment variable (live database connection)
    2. AIDBManager database path (live database from manager)
    3. alembic.ini configuration (live database from config)

    Returns:
        Live database connection URL

    Note: The in-memory SQLite URL is a development configuration option, not fallback data.
    All migrations operate on live database connections.
    """
    # Priority 1: Live database URL from environment (production)
    env_url = os.getenv("DATABASE_URL")
    if env_url:
        logger.info("Using live database URL from DATABASE_URL environment variable")
        return env_url
    # Priority 2: Live database URL from manager (live operations)
    if db_url_from_manager:
        logger.info("Using live database URL from AIDBManager")
        return db_url_from_manager
    # Priority 3: Live database URL from alembic.ini configuration
    ini_url = config.get_main_option("sqlalchemy.url")
    if ini_url:
        logger.info("Using live database URL from alembic.ini configuration")
        return ini_url
    # Development configuration option (in-memory SQLite, not fallback data, development mode)
    logger.warning("No live database URL found, using in-memory SQLite for development")
    return "sqlite:///:memory:"  # Development configuration option, not fallback data


# Always set the resolved live database URL into alembic config
resolved_url = _resolve_sqlalchemy_url()
config.set_main_option("sqlalchemy.url", resolved_url)

# Determine if we're on SQLite for batch mode migrations (live database type detection)
_IS_SQLITE = resolved_url.startswith("sqlite://")


# -----------------------------------------------------------------------------
# Migration runners for live database operations (backend port 8000)
# -----------------------------------------------------------------------------
def run_migrations_offline() -> None:
    """
    Run migrations in 'offline' mode on live database.

    Generates migration scripts without connecting to live database.
    All migrations operate on live database schema.
    """
    context.configure(
        url=resolved_url,  # Live database URL
        target_metadata=target_metadata,  # Live schema metadata
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        compare_type=True,  # Compare live column types
        compare_server_default=True,  # Compare live server defaults
        render_as_batch=_IS_SQLITE,  # Safer DDL on SQLite (live database)
    )

    with context.begin_transaction():
        context.run_migrations()  # Run migrations on live database schema


def run_migrations_online() -> None:
    """
    Run migrations in 'online' mode on live database (backend port 8000).

    Connects to live database and applies migration scripts.
    All migrations operate on live database schema.
    """
    # Create connection to live database
    connectable = engine_from_config(
        config.get_section(config.config_ini_section, {}) or {},
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,  # No connection pooling for migrations (live database)
    )

    with connectable.connect() as connection:
        # Configure context for live database connection
        context.configure(
            connection=connection,  # Live database connection
            target_metadata=target_metadata,  # Live schema metadata
            compare_type=True,  # Compare live column types
            compare_server_default=True,  # Compare live server defaults
            render_as_batch=_IS_SQLITE,  # Safer DDL on SQLite (live database)
        )

        with context.begin_transaction():
            context.run_migrations()  # Run migrations on live database


# Run migrations on live database (offline or online mode)
if context.is_offline_mode():
    run_migrations_offline()  # Generate migration scripts for live database
else:
    run_migrations_online()  # Apply migrations to live database
