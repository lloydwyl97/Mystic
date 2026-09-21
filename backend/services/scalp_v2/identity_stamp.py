"""Stamp engine identity onto rows after an insert that omits the columns.

INSERT OR REPLACE drops columns it does not name back to their defaults, so
the stamp runs in the same transaction after the write.
"""

from __future__ import annotations

import sqlite3


def _ensure(conn: sqlite3.Connection, table: str) -> None:
    cols = {str(r[1]) for r in conn.execute(f"PRAGMA table_info({table})")}
    if "engine_id" not in cols:
        conn.execute(f"ALTER TABLE {table} ADD COLUMN engine_id TEXT DEFAULT 'LEGACY_DAY_LIVE'")
    if "scalp_opportunity_id" not in cols:
        conn.execute(f"ALTER TABLE {table} ADD COLUMN scalp_opportunity_id TEXT DEFAULT ''")


def stamp_engine(
    conn: sqlite3.Connection,
    table: str,
    key_col: str,
    key: str,
    engine_id: str,
    opportunity_id: str,
) -> None:
    if not key:
        return
    _ensure(conn, table)
    conn.execute(
        f"UPDATE {table} SET engine_id=?, scalp_opportunity_id=? WHERE {key_col}=?",
        (engine_id or "LEGACY_DAY_LIVE", opportunity_id or "", key),
    )
