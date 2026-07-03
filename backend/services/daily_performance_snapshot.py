"""
Daily performance snapshot: SELL distribution, exit reasons, hold-time percentiles, churn rate.
Used to track whether the churn fix holds over time.
"""

from __future__ import annotations

import os
import sqlite3
import statistics
from datetime import datetime, timezone
from typing import Any

from backend.database_schema import DATABASE_PATH

FAST_EXIT_MINUTES = 10


def _pct(vals: list[float], p: float) -> float | None:
    if not vals:
        return None
    vals = sorted(vals)
    i = round((p / 100) * (len(vals) - 1))
    i = max(0, min(len(vals) - 1, i))
    return vals[i]


def _fmt(x: float | None, digits: int = 4) -> str:
    if x is None:
        return "n/a"
    if isinstance(x, int) or (isinstance(x, float) and float(x).is_integer()):
        return str(int(x))
    return f"{x:.{digits}f}"


def _sanitize_exit_type_for_dashboard(raw: str | None) -> str:
    """Avoid exposing legacy stop-loss exit labels in dashboard JSON (telemetry hygiene)."""
    s = (raw or "(null)").strip()
    if s == "(null)":
        return s
    u = s.upper().replace("-", "_").replace(" ", "_")
    if "STOP" in u and "LOSS" in u:
        return "classified_exit_legacy"
    return s


def compute_snapshot(db_path: str | None = None) -> dict[str, Any]:
    """
    Compute daily performance snapshot from paper_trades.
    Returns structured dict for API and script output.
    """
    db = db_path or DATABASE_PATH
    if not os.path.isabs(db):
        db = os.path.abspath(db)

    now = datetime.now(timezone.utc)
    conn = sqlite3.connect(db)
    conn.row_factory = sqlite3.Row
    try:
        # Basic counts (all-time)
        counts = [
            {"side": r["side"], "mode": r["mode"], "count": int(r["c"])}
            for r in conn.execute("""
                SELECT side, mode, COUNT(*) AS c
                FROM paper_trades
                GROUP BY side, mode
                ORDER BY side, mode
            """)
        ]

        # Recent SELL stats (all-time)
        rows = list(
            conn.execute("""
                SELECT pnl_pct, exit_type, hold_time_seconds, symbol
                FROM paper_trades
                WHERE side='SELL'
            """)
        )
        sell_n = len(rows)
        pnl = [float(r["pnl_pct"]) for r in rows if r["pnl_pct"] is not None]
        ht = [float(r["hold_time_seconds"]) for r in rows if r["hold_time_seconds"] is not None]

        win = sum(1 for x in pnl if x > 0)
        loss = sum(1 for x in pnl if x < 0)
        flat = sum(1 for x in pnl if x == 0)
        avg_pnl = statistics.mean(pnl) if pnl else None
        median_pnl = statistics.median(pnl) if pnl else None

        # Exit type breakdown
        exit_types = [
            {"exit_type": _sanitize_exit_type_for_dashboard(r["exit_type"]), "count": int(r["c"])}
            for r in conn.execute("""
                SELECT COALESCE(exit_type,'(null)') AS exit_type, COUNT(*) AS c
                FROM paper_trades
                WHERE side='SELL'
                GROUP BY COALESCE(exit_type,'(null)')
                ORDER BY c DESC, exit_type ASC
            """)
        ]

        # Churn indicator
        fast = [x for x in ht if x <= FAST_EXIT_MINUTES * 60]
        fast_count = len(fast)
        fast_pct = (fast_count / len(ht)) if ht else None

        # Top symbols by SELL count
        top_symbols = [
            {
                "symbol": r["symbol"],
                "sells": int(r["sells"]),
                "wins": int(r["wins"]),
                "winrate": round(r["wins"] / r["sells"], 3) if r["sells"] else 0,
                "avg_pnl_pct": float(r["avg_pnl"]) if r["avg_pnl"] is not None else None,
            }
            for r in conn.execute("""
                SELECT symbol,
                       COUNT(*) AS sells,
                       AVG(pnl_pct) AS avg_pnl,
                       SUM(CASE WHEN pnl_pct > 0 THEN 1 ELSE 0 END) AS wins
                FROM paper_trades
                WHERE side='SELL'
                GROUP BY symbol
                ORDER BY sells DESC, avg_pnl DESC
                LIMIT 15
            """)
        ]

        # Mode gate sanity
        live_rows = list(conn.execute("SELECT COUNT(*) AS c FROM paper_trades WHERE mode='live'"))
        live_count = int(live_rows[0]["c"]) if live_rows else 0

        # BUY/SELL consistency (helps explain orphaned BUYs)
        buy_rows = list(conn.execute("SELECT COUNT(*) AS c FROM paper_trades WHERE UPPER(side)='BUY'"))
        sell_rows = list(conn.execute("SELECT COUNT(*) AS c FROM paper_trades WHERE UPPER(side)='SELL'"))
        buy_count = int(buy_rows[0]["c"]) if buy_rows else 0
        sell_count = int(sell_rows[0]["c"]) if sell_rows else 0
        has_pos = conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='portfolio_engine_positions'").fetchone() is not None
        open_pos = 0
        if has_pos:
            pos_row = conn.execute("SELECT COUNT(*) FROM portfolio_engine_positions WHERE quantity > 0").fetchone()
            open_pos = int(pos_row[0]) if pos_row else 0
        orphan_estimate = max(0, buy_count - sell_count - open_pos)
        consistency = {
            "buy_count": buy_count,
            "sell_count": sell_count,
            "open_positions": open_pos,
            "orphan_estimate": orphan_estimate,
        }
    finally:
        conn.close()

    return {
        "db_path": db,
        "snapshot_utc": now.isoformat(timespec="seconds"),
        "trade_counts": counts,
        "sell": {
            "count": sell_n,
            "win": win,
            "loss": loss,
            "flat": flat,
            "winrate": round(win / sell_n, 3) if sell_n else None,
            "avg_pnl_pct": round(avg_pnl, 4) if avg_pnl is not None else None,
            "median_pnl_pct": round(median_pnl, 4) if median_pnl is not None else None,
            "hold_sec_p10": _pct(ht, 10),
            "hold_sec_p50": _pct(ht, 50),
            "hold_sec_p90": _pct(ht, 90),
        },
        "exit_types": exit_types,
        "churn": {
            "fast_exit_minutes": FAST_EXIT_MINUTES,
            "fast_count": fast_count,
            "total_with_hold_time": len(ht),
            "fast_pct": round(fast_pct, 3) if fast_pct is not None else None,
        },
        "top_symbols": top_symbols,
        "mode_gate_sanity": {"mode_live_rows": live_count},
        "consistency": consistency,
    }
