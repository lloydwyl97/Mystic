"""Idempotent accounting repair. Does not delete fills or rewrite P&L columns.

Reconcile mirrors and manual rows with no venue order stop counting toward
realized P&L via counts_toward_realized=0. The original pnl columns stay.
Missing venue trade ids are written only when the caller supplies venue-proven
ids. Sibling orders that were merged into one Mystic row are stored as
supplements so their commission is not dropped and is not double-booked.
"""

from __future__ import annotations

import json
import sqlite3
from typing import Any


def _cols(conn: sqlite3.Connection, table: str) -> set[str]:
    return {str(r[1]) for r in conn.execute(f"PRAGMA table_info({table})")}


def _add(conn: sqlite3.Connection, table: str, name: str, decl: str) -> None:
    if name not in _cols(conn, table):
        conn.execute(f"ALTER TABLE {table} ADD COLUMN {name} {decl}")


def ensure_repair_schema(conn: sqlite3.Connection) -> None:
    if "paper_trades" in {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}:
        _add(conn, "paper_trades", "counts_toward_realized", "INTEGER DEFAULT 1")
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS live_fill_supplements (
            exchange_order_id TEXT PRIMARY KEY,
            symbol TEXT NOT NULL,
            side TEXT NOT NULL,
            trade_ids_json TEXT NOT NULL,
            qty REAL NOT NULL,
            price REAL NOT NULL,
            fee_amount REAL NOT NULL,
            fee_asset TEXT,
            taker_or_maker TEXT,
            parent_order_id TEXT,
            note TEXT NOT NULL,
            created_at TEXT DEFAULT (datetime('now'))
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS documented_balance_residuals (
            symbol TEXT PRIMARY KEY,
            exchange_qty REAL NOT NULL,
            position_qty REAL NOT NULL,
            residual_qty REAL NOT NULL,
            note TEXT NOT NULL,
            updated_at TEXT DEFAULT (datetime('now'))
        )
        """
    )
    if "live_exchange_fills" in {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}:
        _add(conn, "live_exchange_fills", "venue_order_ids_json", "TEXT DEFAULT '[]'")
        _add(conn, "live_exchange_fills", "taker_or_maker", "TEXT")


def exclude_duplicate_realized(conn: sqlite3.Connection) -> int:
    """Stop reconcile mirrors and venue-less manual rows from adding realized P&L again."""
    ensure_repair_schema(conn)
    cur = conn.execute(
        """
        UPDATE paper_trades
        SET counts_toward_realized=0
        WHERE side='SELL'
          AND COALESCE(counts_toward_realized, 1)=1
          AND (
            exit_reason IN ('EXCHANGE_RECONCILE_CLOSE','HUMAN_MANUAL_SELL')
            OR exit_type IN ('EXCHANGE_RECONCILE_CLOSE','HUMAN_MANUAL_SELL')
          )
        """
    )
    return int(cur.rowcount or 0)


def record_supplements(conn: sqlite3.Connection, rows: list[dict[str, Any]]) -> int:
    """Insert venue orders that were merged away. Existing order ids are left as-is."""
    ensure_repair_schema(conn)
    added = 0
    for row in rows:
        oid = str(row.get("exchange_order_id") or "").strip()
        if not oid:
            continue
        cur = conn.execute(
            """
            INSERT OR IGNORE INTO live_fill_supplements(
                exchange_order_id, symbol, side, trade_ids_json, qty, price,
                fee_amount, fee_asset, taker_or_maker, parent_order_id, note
            ) VALUES (?,?,?,?,?,?,?,?,?,?,?)
            """,
            (
                oid,
                str(row.get("symbol") or ""),
                str(row.get("side") or ""),
                json.dumps(row.get("trade_ids") or []),
                float(row.get("qty") or 0),
                float(row.get("price") or 0),
                float(row.get("fee_amount") or 0),
                str(row.get("fee_asset") or ""),
                str(row.get("taker_or_maker") or ""),
                str(row.get("parent_order_id") or ""),
                str(row.get("note") or "merged venue order"),
            ),
        )
        added += int(cur.rowcount or 0)
    return added


def record_residual(conn: sqlite3.Connection, symbol: str, exchange_qty: float, position_qty: float, note: str) -> None:
    ensure_repair_schema(conn)
    residual = float(exchange_qty) - float(position_qty)
    conn.execute(
        """
        INSERT INTO documented_balance_residuals(symbol, exchange_qty, position_qty, residual_qty, note, updated_at)
        VALUES (?,?,?,?,?, datetime('now'))
        ON CONFLICT(symbol) DO UPDATE SET
            exchange_qty=excluded.exchange_qty,
            position_qty=excluded.position_qty,
            residual_qty=excluded.residual_qty,
            note=excluded.note,
            updated_at=datetime('now')
        """,
        (symbol, float(exchange_qty), float(position_qty), residual, note),
    )


def apply_trade_id_backfill(conn: sqlite3.Connection, by_order: dict[str, dict[str, Any]]) -> int:
    """Write venue-proven trade ids onto empty fill rows. Does not invent ids."""
    ensure_repair_schema(conn)
    updated = 0
    for order_id, payload in by_order.items():
        trade_ids = [str(x) for x in (payload.get("trade_ids") or []) if str(x)]
        if not trade_ids:
            continue
        cur = conn.execute(
            """
            UPDATE live_exchange_fills
            SET fill_ids_json=?,
                venue_trade_ids_json=?,
                venue_order_ids_json=?,
                fill_count=?,
                taker_or_maker=COALESCE(taker_or_maker, ?)
            WHERE exchange_order_id=?
              AND COALESCE(venue_trade_ids_json,'[]') IN ('[]','','null')
            """,
            (
                json.dumps(trade_ids),
                json.dumps(trade_ids),
                json.dumps(payload.get("order_ids") or [order_id]),
                len(trade_ids),
                str(payload.get("taker_or_maker") or ""),
                str(order_id),
            ),
        )
        updated += int(cur.rowcount or 0)
    return updated
