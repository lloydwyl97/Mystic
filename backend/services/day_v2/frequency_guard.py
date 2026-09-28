"""DAY V2 frequency guard.

Enforces fill-rate limits to prevent over-trading:

  - Per-symbol cap: max 2 DAY_V2 fills in any rolling 24-hour window.
  - Total cap:      max 8 DAY_V2 fills in any rolling 24-hour window.

The window is computed from arm_ts (the time the intent was armed), not
from updated_at, so the count accurately reflects when entries were decided.

Filled intents are those with status='FILLED' and engine_id='DAY_V2'.
Direct entries create no intent, so DAY_V2 BUY rows in paper_trades are also
counted; the larger of the two counts applies (intent fills also write a BUY row).

Fail-open: any DB error returns (True, "DB_ERROR_FAIL_OPEN") so a transient
SQLite failure never silently blocks all trading.
"""

from __future__ import annotations

import logging
import sqlite3
import time

logger = logging.getLogger(__name__)

_PER_SYMBOL_CAP: int = 2
_TOTAL_CAP: int = 8
_WINDOW_SEC: float = 86400.0  # 24 hours
_DAY_V2_ENGINE_ID: str = "DAY_V2"

# Public aliases — tests and callers use these names
DAY_V2_MAX_FILLS_PER_SYMBOL_24H: int = _PER_SYMBOL_CAP
DAY_V2_MAX_FILLS_TOTAL_24H: int = _TOTAL_CAP
DAY_ENTRY_FREQUENCY_LIMIT: str = "DAY_ENTRY_FREQUENCY_LIMIT"


def _count(conn: sqlite3.Connection, sql: str, params: list[object]) -> int:
    """COUNT(*) query; a table missing on this database counts as zero."""
    try:
        row = conn.execute(sql, params).fetchone()
    except sqlite3.OperationalError as exc:
        if "no such table" not in str(exc):
            raise
        return 0
    return int(row[0]) if row else 0


_SYMBOL_MATCH = " AND UPPER(REPLACE(REPLACE(symbol, '/', ''), '-', '')) = UPPER(REPLACE(REPLACE(?, '/', ''), '-', ''))"


def _filled_count(conn: sqlite3.Connection, symbol: str | None, cutoff: float) -> int:
    intents_sql = "SELECT COUNT(*) FROM day_trailing_buy_intents WHERE engine_id=? AND status='FILLED' AND arm_ts >= ?"
    buys_sql = """
        SELECT COUNT(*) FROM paper_trades
        WHERE engine_id=? AND UPPER(side)='BUY' AND status='executed'
          AND (is_synthetic IS NULL OR is_synthetic != 1)
          AND julianday(timestamp) >= julianday(?, 'unixepoch')
    """
    params: list[object] = [_DAY_V2_ENGINE_ID, cutoff]
    if symbol is not None:
        intents_sql += _SYMBOL_MATCH
        buys_sql += _SYMBOL_MATCH
        params.append(symbol)
    return max(_count(conn, intents_sql, params), _count(conn, buys_sql, params))


def check_frequency_limit(db_path: str, symbol: str) -> tuple[bool, str]:
    """Return (allowed, reason).

    allowed=True  means this symbol may proceed to arm a new intent.
    allowed=False means a cap has been reached; reason describes which one.

    The window is [now - 86400, now].
    """
    cutoff = time.time() - _WINDOW_SEC
    try:
        with sqlite3.connect(db_path) as conn:
            sym_count = _filled_count(conn, symbol, cutoff)
            if sym_count >= _PER_SYMBOL_CAP:
                return (
                    False,
                    f"{DAY_ENTRY_FREQUENCY_LIMIT}:SYMBOL:{symbol}:{sym_count}/{_PER_SYMBOL_CAP}_in_24h",
                )

            total_count = _filled_count(conn, None, cutoff)
            if total_count >= _TOTAL_CAP:
                return (
                    False,
                    f"{DAY_ENTRY_FREQUENCY_LIMIT}:TOTAL:{total_count}/{_TOTAL_CAP}_in_24h",
                )

        return True, "OK"

    except Exception as exc:
        logger.warning("check_frequency_limit DB error (fail-open): %s", exc)
        return True, "DB_ERROR_FAIL_OPEN"
