"""A venue fill is owned even when the open-lot transaction does not commit.

The row is written before that transaction. Restart books it once from the
stored quantity. It never sends another order and never invents a size.
"""

from __future__ import annotations

import sqlite3
from datetime import datetime, timezone
from typing import Any

PENDING_BOOK = "PENDING_BOOK"
RECOVERED_ACTIVE_LOT = "RECOVERED_ACTIVE_LOT"
QUARANTINED_FILLED_ASSET = "QUARANTINED_FILLED_ASSET"
_OPEN = (PENDING_BOOK, QUARANTINED_FILLED_ASSET)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def ensure(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS filled_asset_custody (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            engine_id TEXT NOT NULL,
            symbol TEXT NOT NULL,
            trade_id TEXT NOT NULL,
            client_order_id TEXT NOT NULL DEFAULT '',
            exchange_order_id TEXT NOT NULL DEFAULT '',
            quantity REAL NOT NULL,
            fill_price REAL NOT NULL,
            fee REAL NOT NULL DEFAULT 0,
            cash_debit REAL NOT NULL DEFAULT 0,
            status TEXT NOT NULL,
            cash_applied INTEGER NOT NULL DEFAULT 0,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            UNIQUE(engine_id, trade_id)
        )
        """
    )


def remember(
    db_path: str,
    *,
    engine_id: str,
    symbol: str,
    trade_id: str,
    quantity: float,
    fill_price: float,
    fee: float = 0.0,
    cash_debit: float = 0.0,
    client_order_id: str = "",
    exchange_order_id: str = "",
) -> None:
    """Record a venue fill before the open transaction. A repeat is a no-op."""
    if float(quantity) <= 0 or not str(trade_id or "") or not str(engine_id or ""):
        raise ValueError("custody requires engine, trade id and a positive fill quantity")
    conn = sqlite3.connect(db_path)
    try:
        ensure(conn)
        stamp = _now()
        conn.execute(
            """
            INSERT INTO filled_asset_custody (
                engine_id, symbol, trade_id, client_order_id, exchange_order_id,
                quantity, fill_price, fee, cash_debit, status, cash_applied, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 0, ?, ?)
            ON CONFLICT(engine_id, trade_id) DO NOTHING
            """,
            (
                str(engine_id),
                str(symbol),
                str(trade_id),
                str(client_order_id or ""),
                str(exchange_order_id or ""),
                float(quantity),
                float(fill_price),
                float(fee or 0.0),
                float(cash_debit or 0.0),
                PENDING_BOOK,
                stamp,
                stamp,
            ),
        )
        conn.commit()
    finally:
        conn.close()


def open_rows(db_path: str) -> list[dict[str, Any]]:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        ensure(conn)
        rows = conn.execute(
            "SELECT * FROM filled_asset_custody WHERE status IN (?, ?) ORDER BY id",
            _OPEN,
        ).fetchall()
        return [dict(row) for row in rows]
    finally:
        conn.close()


def mark(db_path: str, *, engine_id: str, trade_id: str, status: str, cash_applied: int | None = None) -> None:
    conn = sqlite3.connect(db_path)
    try:
        ensure(conn)
        if cash_applied is None:
            conn.execute(
                "UPDATE filled_asset_custody SET status=?, updated_at=? WHERE engine_id=? AND trade_id=?",
                (status, _now(), str(engine_id), str(trade_id)),
            )
        else:
            conn.execute(
                "UPDATE filled_asset_custody SET status=?, cash_applied=?, updated_at=? WHERE engine_id=? AND trade_id=?",
                (status, int(cash_applied), _now(), str(engine_id), str(trade_id)),
            )
        conn.commit()
    finally:
        conn.close()


def position_owned(conn: sqlite3.Connection, *, engine_id: str, trade_id: str, exchange_order_id: str) -> bool:
    row = conn.execute(
        """
        SELECT 1 FROM portfolio_engine_positions
        WHERE engine_id=? AND (
            trade_id=? OR (entry_order_id=? AND entry_order_id!='')
        ) AND IFNULL(quantity, 0) > 0
        """,
        (str(engine_id), str(trade_id), str(exchange_order_id or "")),
    ).fetchone()
    return row is not None
