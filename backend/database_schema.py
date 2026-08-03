"""
Database Schema for Unified Trade Data Architecture

This module defines the schema for the unified trade data architecture
where SQLite serves as the canonical source for paper trades.

BUG-004 Fix: Provides serialized SQLite write access with WAL mode + busy_timeout.
"""

import asyncio
import contextlib
import logging
import os
import sqlite3
import sys
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# =============================================================================
# SHARED DATABASE ACCESS LAYER (BUG-004 Fix)
# =============================================================================

# Centralized database path (absolute path to avoid CWD issues)
PROJECT_ROOT = Path(__file__).parent.parent
DATABASE_PATH = os.getenv("DATABASE_PATH", str(PROJECT_ROOT / "mystic_trading.db"))

# Write serialization lock to prevent concurrent writes
_write_lock = asyncio.Lock()
_paper_schema_initialized = False

# Schema initialization lock - ensures only one service initializes schema at startup
_schema_init_lock = asyncio.Lock()
_schema_initialization_complete = False


def get_db_connection(timeout: float = 5.0) -> sqlite3.Connection:
    """
    Get a properly configured SQLite connection (BUG-004 Fix).

    Uses sqlite_runtime.connect_rw (WAL once-per-process + busy_timeout).
    """
    from backend.utils.sqlite_runtime import connect_rw

    conn = connect_rw(DATABASE_PATH)
    conn.row_factory = sqlite3.Row
    with contextlib.suppress(Exception):
        conn.execute(f"PRAGMA busy_timeout={int(max(0.1, float(timeout)) * 1000)}")
    return conn


async def execute_write(query: str, params: tuple | dict | None = None, fetch: bool = False) -> Any:
    """
    Execute a write query with serialization (BUG-004 Fix).

    All writes are serialized through a single asyncio.Lock to prevent
    database corruption from concurrent writes.

    Uses BEGIN IMMEDIATE to acquire write lock immediately.

    Args:
        query: SQL query (INSERT, UPDATE, DELETE)
        params: Query parameters
        fetch: If True, fetch and return results (for INSERT ... RETURNING)

    Returns:
        For fetch=True: query results
        For fetch=False: lastrowid
    """
    async with _write_lock:
        # Execute in thread pool to avoid blocking event loop
        return await asyncio.to_thread(_execute_write_sync, query, params, fetch)


def _execute_write_sync(query: str, params: tuple | dict | None = None, fetch: bool = False) -> Any:
    """
    Synchronous implementation of write execution.
    Called from execute_write via asyncio.to_thread.
    Uses bounded lock retries + busy_timeout (sqlite_runtime).
    """
    from backend.utils.sqlite_runtime import is_locked_error, run_locked_retry

    ensure_paper_trading_schema_initialized()

    def _once() -> Any:
        conn = None
        try:
            conn = get_db_connection(timeout=5.0)
            cursor = conn.cursor()
            cursor.execute("BEGIN IMMEDIATE")
            if params:
                cursor.execute(query, params)
            else:
                cursor.execute(query)
            conn.commit()
            if fetch:
                return cursor.fetchall()
            return cursor.lastrowid
        except Exception:
            if conn:
                with contextlib.suppress(Exception):
                    conn.rollback()
            raise
        finally:
            if conn:
                conn.close()

    try:
        return run_locked_retry(_once)
    except sqlite3.OperationalError as e:
        if is_locked_error(e):
            logger.warning("Database locked during write after retries: %s", e)
        else:
            logger.exception("Database write error: %s", e)
        raise
    except Exception as e:
        logger.exception("Database write error: %s", e)
        raise


async def execute_read(query: str, params: tuple | dict | None = None, fetchone: bool = False) -> Any:
    """
    Execute a read query without blocking event loop (BUG-010 Fix).

    Reads use asyncio.to_thread to avoid blocking the FastAPI event loop.
    Reads do NOT need the write lock (WAL mode allows concurrent reads).

    Args:
        query: SQL SELECT query
        params: Query parameters
        fetchone: If True, return single row; if False, return all rows

    Returns:
        Query results (list of rows or single row)
    """
    return await asyncio.to_thread(_execute_read_sync, query, params, fetchone)


def _execute_read_sync(query: str, params: tuple | dict | None = None, fetchone: bool = False) -> Any:
    """
    Synchronous implementation of read execution.
    Called from execute_read via asyncio.to_thread.
    """
    conn = None
    try:
        ensure_paper_trading_schema_initialized()
        conn = get_db_connection(timeout=5.0)
        cursor = conn.cursor()

        if params:
            cursor.execute(query, params)
        else:
            cursor.execute(query)

        if fetchone:
            return cursor.fetchone()
        else:
            return cursor.fetchall()

    except Exception as e:
        logger.exception(f"Database read error: {e}")
        raise
    finally:
        if conn:
            conn.close()


async def execute_batch_write(query: str, params_list: list[tuple | dict]) -> int:
    """
    Execute a batch of write queries (e.g., bulk insert) with serialization.

    All writes in the batch are executed in a single transaction.

    Args:
        query: SQL query template
        params_list: List of parameter tuples/dicts for each execution

    Returns:
        Number of rows affected
    """
    async with _write_lock:
        return await asyncio.to_thread(_execute_batch_write_sync, query, params_list)


def _execute_batch_write_sync(query: str, params_list: list[tuple | dict]) -> int:
    """
    Synchronous implementation of batch write execution.
    Called from execute_batch_write via asyncio.to_thread.
    """
    conn = None
    try:
        ensure_paper_trading_schema_initialized()
        conn = get_db_connection(timeout=5.0)
        cursor = conn.cursor()

        # Begin transaction with IMMEDIATE
        cursor.execute("BEGIN IMMEDIATE")

        cursor.executemany(query, params_list)

        conn.commit()
        return cursor.rowcount

    except Exception as e:
        if conn:
            conn.rollback()
        logger.exception(f"Database batch write error: {e}")
        raise
    finally:
        if conn:
            conn.close()


# =============================================================================
# LEGACY DATABASE FUNCTIONS (Updated to use new access layer)
# =============================================================================

logger = logging.getLogger(__name__)


def create_operational_state_table(db_path: str = DATABASE_PATH) -> bool:
    """
    Create operational_state table for persistent operational state.

    This table is the authoritative source for:
    - Circuit breaker states
    - Profit siphon locked_profit_reserve
    - Risk gates and trading paused states
    - Dedupe markers and last processed decisions
    """
    conn = sqlite3.connect(db_path)
    try:
        cursor = conn.cursor()

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS operational_state (
                key TEXT PRIMARY KEY,
                value_json TEXT NOT NULL,
                updated_ts INTEGER NOT NULL
            )
        """)

        conn.commit()
        logger.info("Operational state table created successfully")
        return True
    except Exception as e:
        logger.exception(f"Failed to create operational state table: {e}")
        return False
    finally:
        conn.close()


def create_paper_trades_table(db_path: str = DATABASE_PATH) -> bool:
    """
    Create the unified paper_trades table for canonical paper trading data.

    This table serves as the single source of truth for all paper trading operations.
    Redis becomes cache-only, not authoritative.
    """
    conn = sqlite3.connect(db_path)
    try:
        cursor = conn.cursor()

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS paper_trades (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                trade_id TEXT UNIQUE NOT NULL,        -- Unique across all restarts/sessions
                paper_run_id TEXT NOT NULL,            -- Session identifier for run separation
                mode TEXT DEFAULT 'paper',             -- Always 'paper' for this table
                symbol TEXT NOT NULL,
                side TEXT NOT NULL,                    -- 'buy' or 'sell'
                quantity REAL NOT NULL,
                price REAL NOT NULL,
                entry_price REAL,                      -- For sells: original buy price
                pnl REAL,                              -- For sells: realized profit/loss
                pnl_pct REAL,                          -- For sells: percentage gain/loss
                remaining_position REAL DEFAULT 0,     -- For tracking partial fills
                hold_time_seconds INTEGER,             -- For sells: time position was held
                commission REAL DEFAULT 0,
                strategy TEXT,
                confidence REAL,
                timestamp TEXT NOT NULL,
                order_id TEXT,
                status TEXT DEFAULT 'executed',
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                updated_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)

        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_paper_trades_run_id
            ON paper_trades(paper_run_id)
        """)

        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_paper_trades_timestamp
            ON paper_trades(timestamp)
        """)

        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_paper_trades_symbol
            ON paper_trades(symbol)
        """)

        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_paper_trades_trade_id
            ON paper_trades(trade_id)
        """)

        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_paper_trades_side
            ON paper_trades(side)
        """)

        conn.commit()
        logger.info(" Paper trades table created successfully")
        return True

    except Exception as e:
        logger.exception(f" Failed to create paper trades table: {e}")
        return False
    finally:
        conn.close()


def ensure_paper_trades_columns(conn) -> bool:
    """
    Ensure paper_trades table has required columns for exit_reason and entry_timestamp.
    Adds columns if missing (idempotent).
    """
    try:
        cursor = conn.cursor()

        # Check existing columns
        cursor.execute("PRAGMA table_info(paper_trades)")
        existing_columns = [row[1] for row in cursor.fetchall()]

        columns_to_add = [
            ("pnl", "REAL"),
            ("pnl_pct", "REAL"),
            ("remaining_position", "REAL DEFAULT 0"),
            ("entry_price", "REAL"),
            ("hold_time_seconds", "INTEGER"),
            ("commission", "REAL DEFAULT 0"),
            ("strategy", "TEXT"),
            ("strategy_id", "TEXT"),
            ("confidence", "REAL"),
            ("exit_reason", "TEXT"),
            ("entry_timestamp", "TEXT"),
            ("decision_id", "TEXT"),
            ("source", "TEXT"),
        ]

        for col_name, col_type in columns_to_add:
            if col_name not in existing_columns:
                cursor.execute(f"ALTER TABLE paper_trades ADD COLUMN {col_name} {col_type}")
                logger.info(f" Added {col_name} column to paper_trades")

        conn.commit()

    except Exception as e:
        logger.exception(f" Failed to ensure paper_trades columns: {e}")
        return False
    else:
        return True


def initialize_paper_trading_schema(db_path: str = DATABASE_PATH) -> bool:
    """
    Initialize the complete paper trading schema.
    Call this on application startup.
    """
    logger.info(" Initializing paper trading database schema...")

    # Create operational_state table first
    operational_success = create_operational_state_table(db_path)
    if not operational_success:
        logger.error(" Failed to create operational_state table")
        return False

    success = create_paper_trades_table(db_path)

    if success:
        # Ensure additional columns exist
        try:
            import sqlite3

            conn = sqlite3.connect(db_path)
            try:
                column_success = ensure_paper_trades_columns(conn)
                if not column_success:
                    logger.error(" Failed to add required columns to paper_trades")
                    return False
            finally:
                conn.close()

        except Exception as e:
            logger.exception(f" Failed to ensure paper_trades columns: {e}")
            return False

        logger.info(" Paper trading schema initialized successfully")
    else:
        logger.error(" Paper trading schema initialization failed")

    return success


def ensure_paper_trading_schema_initialized() -> None:
    """Best-effort, idempotent schema initialization for paper_trades."""
    global _paper_schema_initialized
    if _paper_schema_initialized:
        return
    try:
        if initialize_paper_trading_schema(DATABASE_PATH):
            _paper_schema_initialized = True
    except Exception as e:
        logger.exception(f" Failed to initialize paper trading schema: {e}")


async def initialize_all_schemas() -> bool:
    """
    CENTRALIZED SCHEMA INITIALIZATION (Issue #6 Fix)

    This function MUST be called ONCE at startup BEFORE any service
    starts trading. It initializes all schemas in a serialized, atomic way.

    Returns:
        True if all schemas initialized successfully, False otherwise
    """
    global _schema_initialization_complete

    async with _schema_init_lock:
        if _schema_initialization_complete:
            logger.debug("Schema initialization already complete")
            return True

        logger.info("[SCHEMA_INIT] Starting centralized schema initialization...")

        try:
            # Run all schema initialization synchronously (single thread)
            ensure_paper_trading_schema_initialized()
            logger.info("[SCHEMA_INIT] Paper trading schema initialized")

            # Validate all critical tables exist
            if not validate_paper_trades_table(DATABASE_PATH):
                logger.error("[SCHEMA_INIT] Paper trades table validation failed")
                return False

            logger.info("[SCHEMA_INIT] Schema initialization complete - all tables valid")
            _schema_initialization_complete = True
            return True

        except Exception as e:
            logger.exception(f"[SCHEMA_INIT] CRITICAL: Schema initialization failed: {e}")
            return False


def validate_paper_trades_table(db_path: str = DATABASE_PATH) -> bool:
    """
    Validate that the paper_trades table exists and has correct structure.
    """
    conn = sqlite3.connect(db_path)
    try:
        cursor = conn.cursor()

        cursor.execute("""
            SELECT name FROM sqlite_master
            WHERE type='table' AND name='paper_trades'
        """)

        if not cursor.fetchone():
            logger.error(" paper_trades table does not exist")
            return False

        cursor.execute("PRAGMA table_info(paper_trades)")
        columns = cursor.fetchall()
        column_names = [col[1] for col in columns]

        required_columns = ["trade_id", "paper_run_id", "mode", "symbol", "side", "quantity", "price", "timestamp"]

        missing_columns = [col for col in required_columns if col not in column_names]

        if missing_columns:
            logger.error(f" Missing required columns: {missing_columns}")
            return False

        logger.info("Paper trades table validation passed")
        return True

    except Exception as e:
        logger.exception(f" Paper trades table validation failed: {e}")
        return False
    finally:
        conn.close()


# =============================================================================
# OPERATIONAL STATE HELPER FUNCTIONS
# =============================================================================


async def set_state(key: str, json_obj: Any) -> None:
    """
    Set operational state in SQLite (authoritative source).

    Args:
        key: State key (e.g. 'locked_profit_reserve', 'risk:daily_loss_freeze')
        json_obj: JSON-serializable object to store
    """
    import json
    import time

    value_json = json.dumps(json_obj)
    updated_ts = int(time.time())

    await execute_write("INSERT OR REPLACE INTO operational_state (key, value_json, updated_ts) VALUES (?, ?, ?)", (key, value_json, updated_ts))


async def get_state(key: str) -> Any | None:
    """
    Get operational state from SQLite (authoritative source).

    Args:
        key: State key to retrieve

    Returns:
        Deserialized JSON object or None if not found
    """
    import json

    row = await execute_read("SELECT value_json FROM operational_state WHERE key = ?", (key,), fetchone=True)

    if row:
        return json.loads(row[0])
    return None


async def delete_state(key: str) -> None:
    """
    Delete operational state from SQLite.

    Args:
        key: State key to delete
    """
    await execute_write("DELETE FROM operational_state WHERE key = ?", (key,))


# Run initialization if called directly
if __name__ == "__main__":
    success = initialize_paper_trading_schema()
    if success:
        logger.info(" Paper trading schema initialized successfully!")
    else:
        logger.info(" Paper trading schema initialization failed!")
        sys.exit(1)
