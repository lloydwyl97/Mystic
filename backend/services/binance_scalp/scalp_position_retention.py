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
    """Heal scalp_paper_ledger.total_equity = cash_balance + positions_value."""
    with connect_rw(db_path) as conn:
        row = conn.execute("SELECT cash_balance, positions_value, total_equity FROM scalp_paper_ledger WHERE id=1").fetchone()
        if not row:
            return {"skipped": True, "reason": "no_ledger_row"}
        cash, pos_val, equity = float(row[0] or 0), float(row[1] or 0), float(row[2] or 0)
        expected = cash + pos_val
        if abs(equity - expected) < 0.01:
            return {"skipped": True, "reason": "already_aligned", "total_equity": equity}
        conn.execute(
            "UPDATE scalp_paper_ledger SET total_equity=?, updated_at=datetime('now') WHERE id=1",
            (expected,),
        )
        conn.commit()
        logger.warning(
            "SCALP_LEDGER_HEAL: total_equity %.4f -> %.4f (cash=%.4f pos=%.4f)",
            equity,
            expected,
            cash,
            pos_val,
        )
        return {"healed": True, "before": equity, "after": expected}


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
