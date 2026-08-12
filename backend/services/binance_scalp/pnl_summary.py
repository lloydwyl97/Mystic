"""Read-only scalp PnL summary — isolated from Mystic DAY portfolio_engine ledger."""

from __future__ import annotations

import sqlite3
from datetime import datetime, timezone
from typing import Any

from backend.database_schema import DATABASE_PATH


def build_scalp_pnl_summary(db_path: str = DATABASE_PATH) -> dict[str, Any]:
    """DAY and scalp PnL must stay separate; this is scalp-only.

    Short busy timeout — safe for runner publish path; GET /status must not call this.
    """
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    out: dict[str, Any] = {
        "engine": "scalp",
        "today": {"sells": 0, "realized_pnl_usd": 0.0},
        "all_time": {"sells": 0, "realized_pnl_usd": 0.0},
        "open_positions": 0,
    }
    try:
        from backend.utils.sqlite_runtime import connect_ro

        with connect_ro(db_path, timeout_sec=1.5) as conn:
            row = conn.execute(
                """
                SELECT COUNT(*), COALESCE(SUM(pnl_usd), 0)
                FROM scalp_paper_trades WHERE side='SELL' AND date(created_at) = date(?)
                """,
                (today,),
            ).fetchone()
            if row:
                out["today"] = {"sells": int(row[0] or 0), "realized_pnl_usd": round(float(row[1] or 0.0), 2)}
            row2 = conn.execute("SELECT COUNT(*), COALESCE(SUM(pnl_usd), 0) FROM scalp_paper_trades WHERE side='SELL'").fetchone()
            if row2:
                out["all_time"] = {"sells": int(row2[0] or 0), "realized_pnl_usd": round(float(row2[1] or 0.0), 2)}
            out["open_positions"] = int(conn.execute("SELECT COUNT(*) FROM scalp_paper_positions WHERE status='OPEN'").fetchone()[0] or 0)
    except sqlite3.Error:
        pass
    return out


__all__ = ["build_scalp_pnl_summary"]
