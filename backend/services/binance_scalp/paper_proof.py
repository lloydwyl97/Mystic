"""SCALP paper-proof readiness — genuine-pass closes only, no live enable."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

# Conservative go/no-go before any live conversation.
MIN_CLOSED_SELLS = 20
MIN_NET_PNL_USD = 5.0
MIN_WIN_RATE = 0.45
MIN_PROFIT_FACTOR = 1.15


def compute_paper_proof(db_path: str | Path) -> dict[str, Any]:
    """Summarize closed SCALP paper sells for Option B proof gate."""
    sells = 0
    wins = 0
    gross_win = 0.0
    gross_loss = 0.0
    net = 0.0
    by_setup: dict[str, dict[str, float]] = {}
    max_hold = 0
    net_profit_exits = 0

    with sqlite3.connect(str(db_path), timeout=15) as conn:
        conn.row_factory = sqlite3.Row
        try:
            rows = conn.execute(
                """
                SELECT pnl_usd, exit_reason, diagnostics_json
                FROM scalp_paper_trades
                WHERE UPPER(side)='SELL'
                ORDER BY id ASC
                """
            ).fetchall()
        except sqlite3.OperationalError:
            rows = []

    for row in rows:
        sells += 1
        pnl = float(row["pnl_usd"] or 0.0)
        net += pnl
        er = str(row["exit_reason"] or "")
        if "NET_PROFIT" in er.upper():
            net_profit_exits += 1
        if "MAX_HOLD" in er.upper():
            max_hold += 1
        if pnl > 0:
            wins += 1
            gross_win += pnl
        else:
            gross_loss += abs(pnl)
        setup = "unknown"
        raw = row["diagnostics_json"]
        if raw:
            try:
                d = json.loads(raw) if isinstance(raw, str) else dict(raw or {})
                setup = str(d.get("setup_name") or (d.get("entry_setup_signal") or {}).get("setup_name") or d.get("scalp_setup") or "unknown")
            except Exception:
                pass
        bucket = by_setup.setdefault(setup, {"n": 0.0, "pnl": 0.0, "wins": 0.0})
        bucket["n"] += 1
        bucket["pnl"] += pnl
        if pnl > 0:
            bucket["wins"] += 1

    win_rate = (wins / sells) if sells else 0.0
    pf = (gross_win / gross_loss) if gross_loss > 0 else (999.0 if gross_win > 0 else 0.0)
    ready = sells >= MIN_CLOSED_SELLS and net >= MIN_NET_PNL_USD and win_rate >= MIN_WIN_RATE and pf >= MIN_PROFIT_FACTOR
    return {
        "closed_sells": sells,
        "wins": wins,
        "win_rate": round(win_rate, 4),
        "net_pnl_usd": round(net, 4),
        "profit_factor": round(pf, 4) if pf < 900 else None,
        "net_profit_exits": net_profit_exits,
        "max_hold_exits": max_hold,
        "by_setup": {
            k: {
                "n": int(v["n"]),
                "pnl": round(v["pnl"], 4),
                "wins": int(v["wins"]),
            }
            for k, v in by_setup.items()
        },
        "ready_for_live_discussion": ready,
        "thresholds": {
            "min_closed_sells": MIN_CLOSED_SELLS,
            "min_net_pnl_usd": MIN_NET_PNL_USD,
            "min_win_rate": MIN_WIN_RATE,
            "min_profit_factor": MIN_PROFIT_FACTOR,
        },
        "live_blocked": True,
    }


__all__ = ["compute_paper_proof", "MIN_CLOSED_SELLS", "MIN_NET_PNL_USD"]
