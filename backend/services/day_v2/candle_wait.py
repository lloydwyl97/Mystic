"""Wait for the exact closed 15m candle. Do not evaluate the prior bar as current."""

from __future__ import annotations

import sqlite3
import time
from pathlib import Path
from typing import Any

from backend.config.canonical_candle_intervals import stale_after_sec

PENDING_CANDLE = "PENDING_CANDLE"
MISSING_COMPLETED_CANDLE_TIMEOUT = "MISSING_COMPLETED_CANDLE_TIMEOUT"
_FIFTEEN_MIN = 900


def evaluate_candle_gate(
    *,
    completed_bar_count: int,
    minimum_bars: int,
    latest_open: float | None,
    required_open: float,
    executable_price: float,
    book_age_sec: float | None,
    book_stale_sec: float,
    now: float,
) -> dict[str, Any]:
    candle_ready = completed_bar_count >= minimum_bars and latest_open is not None and float(latest_open) + 1.0 >= float(required_open)
    price_ready = executable_price > 0 and (book_age_sec is None or book_age_sec <= book_stale_sec)
    if not candle_ready:
        close_ts = float(required_open) + _FIFTEEN_MIN
        if float(now) - close_ts > float(stale_after_sec("15m")):
            return {"action": "timeout", "reason": MISSING_COMPLETED_CANDLE_TIMEOUT}
        return {"action": "pending", "reason": PENDING_CANDLE}
    if not price_ready:
        return {"action": "reject", "reason": "MISSING_EXECUTABLE_PRICE"}
    return {"action": "proceed", "reason": "READY", "price": float(executable_price)}


def _ensure(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS day_v2_candle_pending (
            symbol TEXT PRIMARY KEY,
            bar_open REAL NOT NULL,
            started_at REAL NOT NULL
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS day_v2_bar_claims (
            symbol TEXT NOT NULL,
            bar_open REAL NOT NULL,
            result TEXT NOT NULL,
            claimed_at REAL NOT NULL,
            PRIMARY KEY (symbol, bar_open)
        )
        """
    )


def note_pending(db_path: str | Path, symbol: str, bar_open: float, *, now: float | None = None) -> None:
    moment = float(now if now is not None else time.time())
    conn = sqlite3.connect(str(db_path), timeout=30)
    try:
        _ensure(conn)
        conn.execute(
            """
            INSERT INTO day_v2_candle_pending(symbol, bar_open, started_at)
            VALUES (?,?,?)
            ON CONFLICT(symbol) DO UPDATE SET bar_open=excluded.bar_open
            """,
            (str(symbol), float(bar_open), moment),
        )
        conn.commit()
    finally:
        conn.close()


def pending_symbols(db_path: str | Path) -> set[str]:
    conn = sqlite3.connect(str(db_path), timeout=30)
    try:
        _ensure(conn)
        rows = conn.execute("SELECT symbol FROM day_v2_candle_pending").fetchall()
        return {str(r[0]) for r in rows}
    finally:
        conn.close()


def clear_pending(db_path: str | Path, symbol: str) -> None:
    conn = sqlite3.connect(str(db_path), timeout=30)
    try:
        _ensure(conn)
        conn.execute("DELETE FROM day_v2_candle_pending WHERE symbol=?", (str(symbol),))
        conn.commit()
    finally:
        conn.close()


def claim_bar(db_path: str | Path, symbol: str, bar_open: float, result: str) -> bool:
    """Claim this symbol and closed bar once. A second call does not insert."""
    conn = sqlite3.connect(str(db_path), timeout=30)
    try:
        _ensure(conn)
        cur = conn.execute(
            """
            INSERT OR IGNORE INTO day_v2_bar_claims(symbol, bar_open, result, claimed_at)
            VALUES (?,?,?,?)
            """,
            (str(symbol), float(bar_open), str(result), time.time()),
        )
        conn.execute("DELETE FROM day_v2_candle_pending WHERE symbol=?", (str(symbol),))
        conn.commit()
        return int(cur.rowcount or 0) == 1
    finally:
        conn.close()


def claim_result(db_path: str | Path, symbol: str, bar_open: float) -> str | None:
    conn = sqlite3.connect(str(db_path), timeout=30)
    try:
        _ensure(conn)
        row = conn.execute(
            "SELECT result FROM day_v2_bar_claims WHERE symbol=? AND bar_open=?",
            (str(symbol), float(bar_open)),
        ).fetchone()
        return None if row is None else str(row[0])
    finally:
        conn.close()
