#!/usr/bin/env python3
"""Purge closed scalp_paper_positions rows and report book health (no strategy changes)."""

from __future__ import annotations

import json
import sqlite3
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.services.binance_scalp.config import get_scalp_config


def main() -> int:
    cfg = get_scalp_config()
    db = cfg.database_path
    with sqlite3.connect(db) as conn:
        open_n = conn.execute("SELECT COUNT(*) FROM scalp_paper_positions WHERE status='OPEN'").fetchone()[0]
        closed_n = conn.execute("SELECT COUNT(*) FROM scalp_paper_positions WHERE status='CLOSED'").fetchone()[0]
        ledger = conn.execute("SELECT cash_balance, positions_value, realized_pnl, total_equity FROM scalp_paper_ledger WHERE id=1").fetchone()
        deleted = 0
        if closed_n > 0:
            deleted = conn.execute("DELETE FROM scalp_paper_positions WHERE status='CLOSED'").rowcount
            conn.commit()
        open_after = conn.execute("SELECT COUNT(*) FROM scalp_paper_positions WHERE status='OPEN'").fetchone()[0]

    report = {
        "database": db,
        "open_before": open_n,
        "closed_purged": deleted,
        "open_after": open_after,
        "ledger": {
            "cash_balance": ledger[0] if ledger else None,
            "positions_value": ledger[1] if ledger else None,
            "realized_pnl": ledger[2] if ledger else None,
            "total_equity": ledger[3] if ledger else None,
        },
        "pass": open_after == open_n and open_after <= cfg.max_open_positions,
    }
    print(json.dumps(report, indent=2))
    return 0 if report["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
