"""DAY V2 database migrations.

Idempotent add-column and create-table migrations for the SCALP V2 / DAY V2
engine identity schema. All migrations are reversible (columns added with
DEFAULT values; tables created with IF NOT EXISTS).

Usage:
    from backend.services.day_v2.migrations import apply_all_migrations
    apply_all_migrations("/path/to/mystic_trading.db")

Each migration is identified by a unique string key. Already-applied
migrations are detected by catching OperationalError from duplicate-column
attempts (SQLite does not have IF NOT EXISTS for ALTER TABLE ADD COLUMN).
"""

from __future__ import annotations

import logging
import sqlite3
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Individual migrations (idempotent — safe to call multiple times)
# ---------------------------------------------------------------------------


def _column_exists(conn: sqlite3.Connection, table: str, column: str) -> bool:
    """Return True iff the column exists in the table."""
    try:
        rows = conn.execute(f"PRAGMA table_info({table})").fetchall()
        return any(r[1] == column for r in rows)
    except Exception:
        return False


def _table_exists(conn: sqlite3.Connection, table: str) -> bool:
    """Return True iff the table exists."""
    try:
        row = conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?",
            (table,),
        ).fetchone()
        return row is not None
    except Exception:
        return False


def _add_column_if_missing(
    conn: sqlite3.Connection,
    table: str,
    column: str,
    column_def: str,
) -> bool:
    """Add a column to a table if it does not already exist. Returns True if added."""
    if not _table_exists(conn, table):
        logger.debug("migration: table %r does not exist yet; skipping %r", table, column)
        return False
    if _column_exists(conn, table, column):
        return False
    try:
        conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {column_def}")
        conn.commit()
        logger.info("migration: added column %r.%r", table, column)
        return True
    except sqlite3.OperationalError as exc:
        # Race condition: another process added the column simultaneously.
        if "duplicate column" in str(exc).lower():
            return False
        raise


def migration_engine_id_positions(conn: sqlite3.Connection) -> dict[str, Any]:
    """Add engine_id and scalp_opportunity_id to portfolio_engine_positions."""
    results = {}
    results["engine_id"] = _add_column_if_missing(
        conn,
        "portfolio_engine_positions",
        "engine_id",
        "TEXT DEFAULT 'LEGACY_DAY_LIVE'",
    )
    results["scalp_opportunity_id"] = _add_column_if_missing(
        conn,
        "portfolio_engine_positions",
        "scalp_opportunity_id",
        "TEXT DEFAULT ''",
    )
    return results


def migration_engine_id_paper_trades(conn: sqlite3.Connection) -> dict[str, Any]:
    """Add engine_id and scalp_opportunity_id to paper_trades."""
    results = {}
    results["engine_id"] = _add_column_if_missing(
        conn,
        "paper_trades",
        "engine_id",
        "TEXT DEFAULT 'LEGACY_DAY_LIVE'",
    )
    results["scalp_opportunity_id"] = _add_column_if_missing(
        conn,
        "paper_trades",
        "scalp_opportunity_id",
        "TEXT DEFAULT ''",
    )
    return results


def migration_engine_id_trailing_buy_intents(conn: sqlite3.Connection) -> dict[str, Any]:
    """Add engine_id and scalp_opportunity_id to day_trailing_buy_intents."""
    results = {}
    results["engine_id"] = _add_column_if_missing(
        conn,
        "day_trailing_buy_intents",
        "engine_id",
        "TEXT DEFAULT 'LEGACY_DAY_LIVE'",
    )
    results["scalp_opportunity_id"] = _add_column_if_missing(
        conn,
        "day_trailing_buy_intents",
        "scalp_opportunity_id",
        "TEXT DEFAULT ''",
    )
    return results


def migration_day_v2_shadow_observations(conn: sqlite3.Connection) -> dict[str, Any]:
    """Create the day_v2_shadow_observations table (read-only accumulator)."""
    created = False
    if not _table_exists(conn, "day_v2_shadow_observations"):
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS day_v2_shadow_observations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                opportunity_id TEXT,
                state TEXT NOT NULL,
                exit_role_evaluated TEXT,
                exit_should_fire INTEGER,
                net_pnl_pct REAL,
                hold_minutes REAL,
                bar_ts TEXT,
                recorded_at TEXT DEFAULT (datetime('now'))
            )
            """
        )
        conn.commit()
        logger.info("migration: created table day_v2_shadow_observations")
        created = True
    return {"created": created}


# ---------------------------------------------------------------------------
# Migration registry and runner
# ---------------------------------------------------------------------------

_MIGRATIONS: list[tuple[str, Any]] = [
    ("engine_id_positions", migration_engine_id_positions),
    ("engine_id_paper_trades", migration_engine_id_paper_trades),
    ("engine_id_trailing_buy_intents", migration_engine_id_trailing_buy_intents),
    ("day_v2_shadow_observations", migration_day_v2_shadow_observations),
]


def apply_all_migrations(db_path: str) -> dict[str, Any]:
    """Apply all DAY V2 / SCALP V2 migrations to the given SQLite database.

    Idempotent — safe to call at startup on every run. Returns a dict mapping
    migration name -> result dict.
    """
    logger.info("day_v2.migrations: applying %d migrations to %s", len(_MIGRATIONS), db_path)
    results: dict[str, Any] = {}
    try:
        with sqlite3.connect(db_path) as conn:
            for name, fn in _MIGRATIONS:
                try:
                    result = fn(conn)
                    results[name] = result
                    logger.debug("migration %r: %s", name, result)
                except Exception as exc:
                    logger.error("migration %r failed: %s", name, exc)
                    results[name] = {"error": str(exc)}
    except Exception as exc:
        logger.error("day_v2.migrations: could not connect to %s: %s", db_path, exc)
        results["_connection_error"] = str(exc)
    return results
