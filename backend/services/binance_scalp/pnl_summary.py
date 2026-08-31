"""Read-only scalp PnL summary — isolated from Mystic DAY portfolio_engine ledger."""

from __future__ import annotations

import sqlite3
from datetime import datetime, timezone
from typing import Any


def _default_scalp_money_db() -> str:
    from backend.services.binance_scalp.config import get_scalp_config

    return get_scalp_config().database_path


def build_scalp_pnl_summary(db_path: str | None = None) -> dict[str, Any]:
    """DAY and scalp PnL must stay separate; this is scalp-only.

    Short busy timeout — safe for runner publish path; GET /status must not call this.
    """
    db_path = db_path or _default_scalp_money_db()
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
                FROM scalp_paper_trades
                WHERE side='SELL' AND strategy_id='structural_lp' AND date(created_at) = date(?)
                """,
                (today,),
            ).fetchone()
            if row:
                out["today"] = {"sells": int(row[0] or 0), "realized_pnl_usd": round(float(row[1] or 0.0), 2)}
            row2 = conn.execute("SELECT COUNT(*), COALESCE(SUM(pnl_usd), 0) FROM scalp_paper_trades WHERE side='SELL' AND strategy_id='structural_lp'").fetchone()
            if row2:
                out["all_time"] = {"sells": int(row2[0] or 0), "realized_pnl_usd": round(float(row2[1] or 0.0), 2)}
            legacy = conn.execute("SELECT COUNT(*), COALESCE(SUM(pnl_usd), 0) FROM scalp_paper_trades WHERE side='SELL' AND IFNULL(strategy_id,'') != 'structural_lp'").fetchone()
            if legacy:
                out["legacy_ranking_book"] = {
                    "sells": int(legacy[0] or 0),
                    "realized_pnl_usd": round(float(legacy[1] or 0.0), 2),
                    "mixed": False,
                    "note": "retired ranking book — not included in structural PnL",
                }
            out["open_positions"] = int(conn.execute("SELECT COUNT(*) FROM scalp_paper_positions WHERE status='OPEN' AND strategy_id='structural_lp'").fetchone()[0] or 0)
            out["legacy_open_positions"] = int(conn.execute("SELECT COUNT(*) FROM scalp_paper_positions WHERE status='OPEN' AND IFNULL(strategy_id,'') != 'structural_lp'").fetchone()[0] or 0)
            out["book"] = "structural_lp"
            out["fill_model"] = "structural_event_queue_v1"
    except sqlite3.Error:
        pass
    return out


__all__ = ["build_scalp_pnl_summary"]
