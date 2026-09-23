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


def migration_legacy_exit_only(conn: sqlite3.Connection) -> dict[str, Any]:
    """Open legacy lots stay exit-only. Closed history is not relabeled."""
    cols = {str(r[1]) for r in conn.execute("PRAGMA table_info(portfolio_engine_positions)")}
    if "engine_id" not in cols:
        return {"updated": 0, "skipped": "no engine_id column"}
    cur = conn.execute(
        """
        UPDATE portfolio_engine_positions
        SET engine_id='LEGACY_EXIT_ONLY'
        WHERE UPPER(COALESCE(status,'ACTIVE')) IN ('ACTIVE','DUST_PENDING')
          AND COALESCE(engine_id,'') IN ('','LEGACY_DAY_LIVE')
        """
    )
    conn.commit()
    return {"updated": int(cur.rowcount or 0)}


# ---------------------------------------------------------------------------
# DAY_STRUCTURAL_PULLBACK_V1 migrations and helpers
# ---------------------------------------------------------------------------

_PULLBACK_V1_TELEMETRY_COLS: list[tuple[str, str]] = [
    ("policy_version", "TEXT DEFAULT 'LEGACY'"),
    ("structural_entry_level", "REAL DEFAULT 0"),
    ("pullback_reached", "INTEGER DEFAULT 0"),
    ("red_5m_seen", "INTEGER DEFAULT 0"),
    ("reversal_candle_ts", "REAL DEFAULT 0"),
    ("reversal_level", "REAL DEFAULT 0"),
    ("freq_limit_state", "TEXT DEFAULT ''"),
]

_DAY_V2_OPPORTUNITY_STATE_DDL = """
CREATE TABLE IF NOT EXISTS day_v2_opportunity_state (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    opportunity_id TEXT NOT NULL UNIQUE,
    symbol TEXT NOT NULL,
    setup TEXT NOT NULL,
    policy_version TEXT NOT NULL DEFAULT 'DAY_STRUCTURAL_PULLBACK_V1',
    state TEXT NOT NULL DEFAULT 'ACTIVE',
    consumed INTEGER NOT NULL DEFAULT 0,
    fill_count INTEGER NOT NULL DEFAULT 0,
    created_at REAL NOT NULL,
    consumed_at REAL,
    trade_id TEXT DEFAULT '',
    notes TEXT DEFAULT '',
    updated_at REAL NOT NULL
)
"""


def run_day_v2_migrations(db_path: str) -> dict[str, Any]:
    """Apply DAY_STRUCTURAL_PULLBACK_V1 schema additions. Idempotent.

    Adds telemetry columns to day_trailing_buy_intents and creates
    day_v2_opportunity_state table. Safe to call at every startup.
    """
    results: dict[str, Any] = {}
    try:
        with sqlite3.connect(db_path) as conn:
            # Telemetry columns on intents table
            for col, col_def in _PULLBACK_V1_TELEMETRY_COLS:
                added = _add_column_if_missing(conn, "day_trailing_buy_intents", col, col_def)
                results[f"col_{col}"] = "added" if added else "exists"
            # Opportunity state table
            if not _table_exists(conn, "day_v2_opportunity_state"):
                conn.execute(_DAY_V2_OPPORTUNITY_STATE_DDL)
                conn.commit()
                results["day_v2_opportunity_state"] = "created"
            else:
                results["day_v2_opportunity_state"] = "exists"
    except Exception as exc:
        logger.error("run_day_v2_migrations failed: %s", exc)
        results["_error"] = str(exc)
    return results


def is_opportunity_consumed(db_path: str, opportunity_id: str) -> bool:
    """Return True if the opportunity has been consumed (filled) in this session.

    Fails open (returns False) on any DB error so a DB issue never blocks trading.
    SQLite is the authoritative durable record — Redis loss cannot re-arm a consumed
    opportunity.
    """
    if not opportunity_id:
        return False
    try:
        with sqlite3.connect(db_path) as conn:
            row = conn.execute(
                "SELECT consumed FROM day_v2_opportunity_state WHERE opportunity_id=?",
                (opportunity_id,),
            ).fetchone()
        if row is None:
            return False
        return bool(int(row[0]))
    except Exception as exc:
        logger.warning("is_opportunity_consumed DB error (fail-open): %s", exc)
        return False


def consume_opportunity(
    db_path: str,
    opportunity_id: str,
    symbol: str,
    setup: str,
    *,
    trade_id: str = "",
) -> None:
    """Mark an opportunity as consumed (filled) in the durable SQLite record.

    Upserts: inserts the row if it doesn't exist, then marks it consumed.
    Idempotent: safe to call multiple times for the same opportunity.
    """
    import time as _time

    now = _time.time()
    try:
        with sqlite3.connect(db_path) as conn:
            conn.execute(
                """
                INSERT INTO day_v2_opportunity_state
                    (opportunity_id, symbol, setup, state, consumed, fill_count,
                     created_at, consumed_at, trade_id, updated_at)
                VALUES (?, ?, ?, 'CONSUMED', 1, 1, ?, ?, ?, ?)
                ON CONFLICT(opportunity_id) DO UPDATE SET
                    state='CONSUMED',
                    consumed=1,
                    fill_count=fill_count+1,
                    consumed_at=excluded.consumed_at,
                    trade_id=CASE WHEN excluded.trade_id != '' THEN excluded.trade_id ELSE trade_id END,
                    updated_at=excluded.updated_at
                """,
                (opportunity_id, symbol, setup, now, now, trade_id, now),
            )
            conn.commit()
    except Exception as exc:
        logger.error("consume_opportunity failed opp=%s: %s", opportunity_id, exc)


def cancel_old_policy_intents(db_path: str) -> int:
    """Cancel WAIT_DIP/TRAIL_LOW DAY_V2 intents with LEGACY policy_version and no order_id.

    Called at startup to retire intents armed before DAY_STRUCTURAL_PULLBACK_V1 was
    deployed. Returns the number of intents canceled.
    """
    import time as _time

    now = _time.time()
    canceled = 0
    try:
        with sqlite3.connect(db_path) as conn:
            cols = {r[1] for r in conn.execute("PRAGMA table_info(day_trailing_buy_intents)").fetchall()}
            if "policy_version" not in cols:
                # Column not yet added — nothing to cancel
                return 0
            cur = conn.execute(
                """
                UPDATE day_trailing_buy_intents
                SET status='CANCELED',
                    cancel_reason='LEGACY_POLICY_SUPERSEDED_BY_DAY_STRUCTURAL_PULLBACK_V1',
                    updated_at=?
                WHERE engine_id='DAY_V2'
                  AND status IN ('WAIT_DIP', 'TRAIL_LOW')
                  AND COALESCE(order_id, '') = ''
                  AND COALESCE(policy_version, 'LEGACY') = 'LEGACY'
                """,
                (now,),
            )
            conn.commit()
            canceled = int(cur.rowcount or 0)
    except Exception as exc:
        logger.warning("cancel_old_policy_intents failed: %s", exc)
    return canceled


_MIGRATIONS: list[tuple[str, Any]] = [
    ("engine_id_positions", migration_engine_id_positions),
    ("engine_id_paper_trades", migration_engine_id_paper_trades),
    ("engine_id_trailing_buy_intents", migration_engine_id_trailing_buy_intents),
    ("day_v2_shadow_observations", migration_day_v2_shadow_observations),
    ("legacy_exit_only", migration_legacy_exit_only),
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
