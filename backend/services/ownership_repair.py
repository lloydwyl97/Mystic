"""Metadata-only repairs for closes that were not strategy decisions.

Every row is copied to ``ownership_repair_backup`` before it changes. Price,
quantity, fees, P&L, timestamps and venue ids of a trade are never touched.
"""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from typing import Any

from backend.services.day_v2.live_exit_evaluator import day_v2_recorded_exit_reason
from backend.services.live_close_integrity import CLASS_MANUAL_UNMATCHED, CLASSIFICATION_TABLE, ensure_live_close_tables
from backend.services.scalp_v2.exit_evaluator import scalp_v2_recorded_exit_reason

BACKUP_TABLE = "ownership_repair_backup"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def ensure_backup_table(conn: sqlite3.Connection) -> None:
    conn.execute(
        f"""
        CREATE TABLE IF NOT EXISTS {BACKUP_TABLE} (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            table_name TEXT NOT NULL,
            row_key TEXT NOT NULL,
            row_json TEXT NOT NULL,
            action TEXT NOT NULL,
            reason TEXT NOT NULL,
            backed_up_at TEXT NOT NULL
        )
        """
    )


def _backup(
    conn: sqlite3.Connection,
    table: str,
    key_col: str,
    key: Any,
    action: str,
    reason: str,
    *,
    table_label: str | None = None,
) -> dict[str, Any] | None:
    ensure_backup_table(conn)
    conn.row_factory = sqlite3.Row
    row = conn.execute(f"SELECT * FROM {table} WHERE {key_col}=?", (key,)).fetchone()
    if row is None:
        return None
    data = dict(row)
    conn.execute(
        f"INSERT INTO {BACKUP_TABLE} (table_name, row_key, row_json, action, reason, backed_up_at) VALUES (?,?,?,?,?,?)",
        (table_label or table, f"{key_col}={key}", json.dumps(data, default=str), action, reason, _now()),
    )
    return data


def reclassify_unmatched_sell(conn: sqlite3.Connection, paper_trade_id: int, reason: str) -> bool:
    """A venue SELL of quantity no strategy lot owned: out of strategy P&L and learning."""
    row = _backup(conn, "paper_trades", "id", paper_trade_id, "reclassify_manual_unmatched", reason)
    if row is None or str(row.get("side") or "").upper() != "SELL":
        return False
    conn.execute(
        "UPDATE paper_trades SET counts_toward_realized=0, exit_type=?, exit_reason=? WHERE id=?",
        (CLASS_MANUAL_UNMATCHED, reason, paper_trade_id),
    )
    ensure_live_close_tables(conn)
    conn.execute(
        f"""
        INSERT INTO {CLASSIFICATION_TABLE}
        (trade_id, accounting_class, symbol, timestamp, exchange_order_id, reason, classified_at)
        VALUES (?,?,?,?,?,?,?)
        ON CONFLICT(trade_id) DO UPDATE SET
            accounting_class=excluded.accounting_class,
            reason=excluded.reason,
            classified_at=excluded.classified_at
        """,
        (
            str(row["trade_id"]),
            CLASS_MANUAL_UNMATCHED,
            row.get("symbol"),
            row.get("timestamp"),
            str(row.get("order_id") or "") or None,
            reason,
            _now(),
        ),
    )
    return True


def strategy_label_for(raw_exit_reason: str) -> str:
    return scalp_v2_recorded_exit_reason(raw_exit_reason) or day_v2_recorded_exit_reason(raw_exit_reason)


def relabel_generic_manual_exit(conn: sqlite3.Connection, paper_trade_id: int) -> str | None:
    """Replace a generic MANUAL_EXIT label with the strategy reason the row itself recorded."""
    conn.row_factory = sqlite3.Row
    row = conn.execute("SELECT exit_reason, explainability_json FROM paper_trades WHERE id=?", (paper_trade_id,)).fetchone()
    if row is None or str(row["exit_reason"] or "").upper() != "MANUAL_EXIT":
        return None
    try:
        raw = str(json.loads(row["explainability_json"] or "{}").get("raw_exit_reason") or "")
    except (TypeError, ValueError):
        raw = ""
    label = strategy_label_for(raw)
    if not label:
        return None
    _backup(conn, "paper_trades", "id", paper_trade_id, "relabel_strategy_exit", f"raw_exit_reason={raw}")
    conn.execute("UPDATE paper_trades SET exit_reason=?, exit_type=? WHERE id=?", (label, label, paper_trade_id))
    return label


def exclude_learning_rows(conn: sqlite3.Connection, table: str, key_col: str, keys: list[Any], reason: str) -> int:
    """Remove derived learning rows of a non-strategy close. The trade row itself stays."""
    removed = 0
    for key in keys:
        if _backup(conn, table, key_col, key, "exclude_from_learning", reason) is None:
            continue
        removed += conn.execute(f"DELETE FROM {table} WHERE {key_col}=?", (key,)).rowcount or 0
    return removed


def revert_bandit_observation(
    conn: sqlite3.Connection,
    arm_key: str,
    *,
    win: bool,
    weight_now: float,
    pnl: float,
    restore_last: tuple[float, str] | None = None,
    reason: str,
) -> bool:
    """Undo one observation. ``weight_now`` is its weight after any decay applied since."""
    if _backup(conn, "day_outcome_bandit_arms", "arm_key", arm_key, "revert_bandit_observation", reason) is None:
        return False
    side, count = ("alpha", "wins") if win else ("beta", "losses")
    conn.execute(
        f"""
        UPDATE day_outcome_bandit_arms
        SET {side}=MAX(1.0, {side}-?), {count}=MAX(0, {count}-1), n_obs=MAX(0, n_obs-1), total_pnl=total_pnl-?
        WHERE arm_key=?
        """,
        (float(weight_now), float(pnl), arm_key),
    )
    if restore_last is not None:
        conn.execute(
            "UPDATE day_outcome_bandit_arms SET last_pnl=?, last_exit_reason=? WHERE arm_key=?",
            (float(restore_last[0]), str(restore_last[1]), arm_key),
        )
    return True


def set_lot_quantity(conn: sqlite3.Connection, trade_id: str, quantity: float, reason: str) -> bool:
    """Set a stored lot to its fill-proven quantity."""
    if _backup(conn, "portfolio_engine_positions", "trade_id", trade_id, "set_lot_quantity", reason) is None:
        return False
    conn.execute(
        """
        UPDATE portfolio_engine_positions
        SET quantity=?, dust_qty_canonical=CASE WHEN status='DUST_PENDING' THEN ? ELSE dust_qty_canonical END
        WHERE trade_id=?
        """,
        (float(quantity), float(quantity), trade_id),
    )
    return True
