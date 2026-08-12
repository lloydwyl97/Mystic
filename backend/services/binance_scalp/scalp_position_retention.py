"""Retention and housekeeping for scalp_paper_positions CLOSED history rows."""

from __future__ import annotations

import logging
import os
import sqlite3
import time
from typing import Any

from backend.utils.sqlite_runtime import connect_rw

logger = logging.getLogger(__name__)

_DEFAULT_KEEP_CLOSED = int(os.getenv("SCALP_CLOSED_POSITIONS_KEEP", "50") or "50")
_LAST_PURGE_AT = 0.0


def purge_closed_scalp_positions(
    db_path: str,
    *,
    keep_closed: int | None = None,
) -> dict[str, Any]:
    """
    Delete old CLOSED rows from scalp_paper_positions; keep newest N by id.
    OPEN rows are never touched.
    """
    keep = max(10, keep_closed if keep_closed is not None else _DEFAULT_KEEP_CLOSED)
    with connect_rw(db_path) as conn:
        closed_n = int(conn.execute("SELECT COUNT(*) FROM scalp_paper_positions WHERE status='CLOSED'").fetchone()[0] or 0)
        if closed_n <= keep:
            return {"deleted": 0, "remaining_closed": closed_n, "keep_closed": keep}

        row = conn.execute(
            """
            SELECT id FROM scalp_paper_positions
            WHERE status='CLOSED'
            ORDER BY id DESC
            LIMIT 1 OFFSET ?
            """,
            (keep - 1,),
        ).fetchone()
        if not row:
            return {"deleted": 0, "remaining_closed": closed_n, "keep_closed": keep}

        cutoff_id = int(row[0])
        cur = conn.execute(
            "DELETE FROM scalp_paper_positions WHERE status='CLOSED' AND id < ?",
            (cutoff_id,),
        )
        deleted = int(cur.rowcount or 0)
        conn.commit()
        remaining = int(conn.execute("SELECT COUNT(*) FROM scalp_paper_positions WHERE status='CLOSED'").fetchone()[0] or 0)
        logger.info(
            "SCALP_POSITION_RETENTION: deleted=%s remaining_closed=%s keep=%s cutoff_id=%s",
            deleted,
            remaining,
            keep,
            cutoff_id,
        )
        return {
            "deleted": deleted,
            "remaining_closed": remaining,
            "keep_closed": keep,
            "cutoff_id": cutoff_id,
        }


def reconcile_scalp_ledger_equity(db_path: str) -> dict[str, Any]:
    """Heal scalp ledger from closed trades when flat; always align total_equity."""
    with connect_rw(db_path) as conn:
        row = conn.execute("SELECT principal, cash_balance, positions_value, realized_pnl, total_equity FROM scalp_paper_ledger WHERE id=1").fetchone()
        if not row:
            return {"skipped": True, "reason": "no_ledger_row"}
        principal = float(row[0] or 0)
        cash = float(row[1] or 0)
        pos_val = float(row[2] or 0)
        realized = float(row[3] or 0)
        equity = float(row[4] or 0)
        open_n = int(conn.execute("SELECT COUNT(*) FROM scalp_paper_positions WHERE status='OPEN'").fetchone()[0] or 0)
        sell_pnl = float(conn.execute("SELECT COALESCE(SUM(pnl_usd), 0) FROM scalp_paper_trades WHERE UPPER(side)='SELL'").fetchone()[0] or 0.0)
        out: dict[str, Any] = {"open_positions": open_n, "sell_pnl": sell_pnl}
        if open_n == 0 and principal > 0:
            expected_cash = principal + sell_pnl
            expected_realized = sell_pnl
            if abs(cash - expected_cash) >= 0.01 or abs(realized - expected_realized) >= 0.01 or abs(pos_val) >= 0.01:
                conn.execute(
                    """
                    UPDATE scalp_paper_ledger SET
                        cash_balance=?, positions_value=0, realized_pnl=?, unrealized_pnl=0,
                        total_equity=?, updated_at=datetime('now')
                    WHERE id=1
                    """,
                    (expected_cash, expected_realized, expected_cash),
                )
                conn.commit()
                logger.warning(
                    "SCALP_LEDGER_HEAL: flat-book cash %.4f->%.4f realized %.4f->%.4f (principal=%.4f sell_pnl=%.4f)",
                    cash,
                    expected_cash,
                    realized,
                    expected_realized,
                    principal,
                    sell_pnl,
                )
                out.update(
                    {
                        "healed": True,
                        "mode": "flat_book_from_trades",
                        "before": {"cash": cash, "realized": realized, "equity": equity},
                        "after": {"cash": expected_cash, "realized": expected_realized, "equity": expected_cash},
                    }
                )
                return out
        expected_equity = cash + pos_val
        if abs(equity - expected_equity) < 0.01:
            out.update({"skipped": True, "reason": "already_aligned", "total_equity": equity})
            return out
        conn.execute(
            "UPDATE scalp_paper_ledger SET total_equity=?, updated_at=datetime('now') WHERE id=1",
            (expected_equity,),
        )
        conn.commit()
        logger.warning(
            "SCALP_LEDGER_HEAL: total_equity %.4f -> %.4f (cash=%.4f pos=%.4f)",
            equity,
            expected_equity,
            cash,
            pos_val,
        )
        out.update({"healed": True, "mode": "equity_only", "before": equity, "after": expected_equity})
        return out


def maybe_run_scalp_position_housekeeping(
    db_path: str,
    *,
    min_interval_sec: float = 3600.0,
) -> dict[str, Any] | None:
    """Purge old CLOSED position rows + heal ledger equity (at most once per hour)."""
    global _LAST_PURGE_AT
    now = time.time()
    if now - _LAST_PURGE_AT < min_interval_sec:
        return None
    out: dict[str, Any] = {}
    try:
        out["purge"] = purge_closed_scalp_positions(db_path)
        # Flat-book trade reconciliation first, then equity identity.
        out["ledger"] = reconcile_scalp_ledger_equity(db_path)
        _LAST_PURGE_AT = now
        return out
    except Exception as exc:
        logger.warning("scalp position housekeeping failed: %s", exc)
        return None


__all__ = [
    "maybe_run_scalp_position_housekeeping",
    "purge_closed_scalp_positions",
    "reconcile_scalp_ledger_equity",
]
