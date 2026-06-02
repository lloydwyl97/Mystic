#!/usr/bin/env python3
"""One-shot repair for duplicate XRP cycle-2 sells (paper_trades 2876-2880)."""

from __future__ import annotations

import json
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path

DB_PATH = Path("/home/mystic/mystic/mystic_trading.db")

DUP_PAPER_TRADE_IDS = (2876, 2877, 2878, 2879, 2880)
DUP_AUDIT_IDS = (2698, 2699, 2700, 2701, 2702)
DUP_CLOSE_LEDGER_IDS = (32, 33, 34, 35, 36)
DUP_LEARNING_IDS = (37, 38, 39, 40, 41)

KEEP_PAPER_SELL_ID = 2875
KEEP_AUDIT_ID = 2697
KEEP_CLOSE_LEDGER_ID = 31
KEEP_LEARNING_ID = 36

VOID_REASON = "DUPLICATE_XRP_SELL_IDEMPOTENCY_REPAIR"


def _audit_post_cash(conn: sqlite3.Connection, audit_id: int) -> float | None:
    row = conn.execute(
        "SELECT post_ledger_json FROM portfolio_engine_audit WHERE id = ?",
        (audit_id,),
    ).fetchone()
    if not row or not row[0]:
        return None
    data = json.loads(row[0])
    return float(data.get("cash_balance", 0))


def main() -> int:
    if not DB_PATH.exists():
        print(f"ERROR: database not found: {DB_PATH}", file=sys.stderr)
        return 1

    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    try:
        before = conn.execute("SELECT cash_balance, positions_value, total_equity, realized_pnl, unrealized_pnl, principal FROM portfolio_engine_ledger WHERE id = 1").fetchone()
        print("LEDGER_BEFORE", dict(before) if before else None)

        dup_pnl = conn.execute(
            "SELECT COALESCE(SUM(pnl), 0) FROM paper_trades WHERE id IN ({})".format(",".join("?" * len(DUP_PAPER_TRADE_IDS))),
            DUP_PAPER_TRADE_IDS,
        ).fetchone()[0]

        target_cash = _audit_post_cash(conn, KEEP_AUDIT_ID)
        if target_cash is None:
            target_cash = float(before["cash_balance"]) - float(
                conn.execute(
                    "SELECT COALESCE(SUM(quantity * price), 0) FROM paper_trades WHERE id IN ({})".format(",".join("?" * len(DUP_PAPER_TRADE_IDS))),
                    DUP_PAPER_TRADE_IDS,
                ).fetchone()[0]
            )

        target_realized = float(before["realized_pnl"]) - float(dup_pnl or 0)

        conn.execute("BEGIN IMMEDIATE")

        conn.execute(
            f"DELETE FROM paper_trades WHERE id IN ({','.join('?' * len(DUP_PAPER_TRADE_IDS))})",
            DUP_PAPER_TRADE_IDS,
        )
        conn.execute(
            f"DELETE FROM portfolio_engine_audit WHERE id IN ({','.join('?' * len(DUP_AUDIT_IDS))})",
            DUP_AUDIT_IDS,
        )
        conn.execute(
            f"DELETE FROM position_close_ledger WHERE id IN ({','.join('?' * len(DUP_CLOSE_LEDGER_IDS))})",
            DUP_CLOSE_LEDGER_IDS,
        )
        conn.execute(
            f"DELETE FROM trade_learning_outcomes WHERE id IN ({','.join('?' * len(DUP_LEARNING_IDS))})",
            DUP_LEARNING_IDS,
        )

        positions = conn.execute("SELECT symbol, quantity, entry_price FROM portfolio_engine_positions ORDER BY symbol").fetchall()
        pos_cost = sum(float(r["quantity"]) * float(r["entry_price"]) for r in positions)

        positions_value = float(before["positions_value"])
        total_equity = float(target_cash) + float(positions_value)
        unrealized_pnl = float(positions_value) - float(pos_cost)

        conn.execute(
            """
            UPDATE portfolio_engine_ledger SET
                cash_balance = ?,
                positions_value = ?,
                total_equity = ?,
                realized_pnl = ?,
                unrealized_pnl = ?,
                last_updated = ?
            WHERE id = 1
            """,
            (
                float(target_cash),
                float(positions_value),
                float(total_equity),
                float(target_realized),
                float(unrealized_pnl),
                datetime.now(timezone.utc).isoformat(),
            ),
        )

        conn.execute(
            """
            UPDATE portfolio_engine_scoreboard_daily SET
                trades = 1,
                wins = 1,
                losses = 0,
                win_rate = 1.0,
                realized_pnl = (
                    SELECT COALESCE(pnl, 0) FROM paper_trades WHERE id = ?
                ),
                updated_at = ?
            WHERE date = '2026-05-30'
            """,
            (KEEP_PAPER_SELL_ID, datetime.now(timezone.utc).isoformat()),
        )

        conn.commit()

        after = conn.execute("SELECT cash_balance, positions_value, total_equity, realized_pnl, unrealized_pnl, principal FROM portfolio_engine_ledger WHERE id = 1").fetchone()
        print("LEDGER_AFTER", dict(after) if after else None)
        print("DELETED paper_trades", DUP_PAPER_TRADE_IDS)
        print("DELETED audit", DUP_AUDIT_IDS)
        print("DELETED position_close_ledger", DUP_CLOSE_LEDGER_IDS)
        print("DELETED trade_learning_outcomes", DUP_LEARNING_IDS)
        print("KEPT paper_trades", KEEP_PAPER_SELL_ID)
        print("dup_pnl_removed", float(dup_pnl or 0))
        return 0
    except Exception as exc:
        conn.rollback()
        print(f"ERROR: {exc}", file=sys.stderr)
        raise
    finally:
        conn.close()


if __name__ == "__main__":
    raise SystemExit(main())
