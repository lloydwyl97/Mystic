#!/usr/bin/env python3
"""One-off repair of reconcile-booked HUMAN_MANUAL_SELL closes (fixed in 408fc0f).

No human ever sold Mystic inventory: every HUMAN_MANUAL_SELL row was written by
the vanished-lot path from a stale balance snapshot, with no venue order behind it.
Metadata only; every changed row is copied to ownership_repair_backup first.
Dry run by default; pass --apply with the service stopped.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.services.engine_lot_ownership import fill_owned_quantity
from backend.services.ownership_repair import _backup, ensure_backup_table, exclude_learning_rows, reclassify_unmatched_sell

FALSE_CLOSE_REASON = "false_human_manual_sell:stale_balance_snapshot"

LEARNING_QUERIES = [
    ("trade_learning_outcomes", "SELECT id FROM trade_learning_outcomes WHERE close_reason='HUMAN_MANUAL_SELL'"),
    ("scalp_learning_outcomes", "SELECT id FROM scalp_learning_outcomes WHERE exit_reason='HUMAN_MANUAL_SELL'"),
    ("ai_outcome_training_rows", "SELECT id FROM ai_outcome_training_rows WHERE outcome_class='MANUAL_LOSS'"),
    ("position_close_ledger", "SELECT id FROM position_close_ledger WHERE close_reason='HUMAN_MANUAL_SELL'"),
    ("day_outcome_attribution", "SELECT id FROM day_outcome_attribution WHERE exit_reason='HUMAN_MANUAL_SELL'"),
    ("market_role_trade_outcomes", "SELECT id FROM market_role_trade_outcomes WHERE exit_reason='HUMAN_MANUAL_SELL'"),
]

# Arms whose latest observation is a false close: pnl 0 -> loss of weight 1,
# scaled by the 0.92 window decay applied in the same update when n_obs > 40.
BANDIT_ARMS_QUERY = "SELECT * FROM day_outcome_bandit_arms WHERE last_exit_reason='HUMAN_MANUAL_SELL' AND ABS(COALESCE(last_pnl,0)) < 1e-12"


def _unsold_zeroed_buys(conn: sqlite3.Connection) -> list[dict]:
    """BUY rows zeroed while their own fills still own quantity and no lot holds them."""
    open_tids = {r[0] for r in conn.execute("SELECT trade_id FROM portfolio_engine_positions")}
    out = []
    rows = conn.execute(
        """SELECT trade_id, symbol, engine_id, timestamp FROM paper_trades
           WHERE UPPER(side)='BUY' AND COALESCE(mode,'')='live' AND COALESCE(remaining_position,0)=0
             AND timestamp >= '2026-09-12'"""
    ).fetchall()
    for tid, sym, eng, ts in rows:
        if tid in open_tids:
            continue
        owned = fill_owned_quantity(conn, tid, sym)
        if owned is not None and float(owned) > 0:
            out.append({"trade_id": tid, "symbol": sym, "engine": eng, "bought": ts, "fill_owned": str(owned)})
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default="mystic_trading.db")
    ap.add_argument("--apply", action="store_true")
    args = ap.parse_args()

    conn = sqlite3.connect(args.db, timeout=30)
    conn.row_factory = sqlite3.Row
    sells = conn.execute(
        """SELECT id, symbol, timestamp, quantity, order_id, counts_toward_realized FROM paper_trades
           WHERE UPPER(side)='SELL' AND (exit_type='HUMAN_MANUAL_SELL' OR exit_reason='HUMAN_MANUAL_SELL') ORDER BY id"""
    ).fetchall()
    if any(r["order_id"] for r in sells):
        print("ABORT: a HUMAN_MANUAL_SELL row carries a venue order id")
        return 2
    learning = {t: [r[0] for r in conn.execute(q)] for t, q in LEARNING_QUERIES}
    arms = [dict(r) for r in conn.execute(BANDIT_ARMS_QUERY)]
    report: dict = {
        "apply": args.apply,
        "sell_rows": [r["id"] for r in sells],
        "sell_rows_counting_toward_realized": [r["id"] for r in sells if r["counts_toward_realized"]],
        "learning": {t: len(k) for t, k in learning.items()},
        "bandit_arms": [{k: a[k] for k in ("arm_key", "n_obs", "alpha", "beta", "losses")} for a in arms],
        "zeroed_buys_not_restored": _unsold_zeroed_buys(conn),
    }
    if not args.apply:
        print(json.dumps(report, indent=2, default=str))
        return 0

    conn.execute("BEGIN IMMEDIATE")
    ensure_backup_table(conn)
    for r in sells:
        assert reclassify_unmatched_sell(conn, int(r["id"]), FALSE_CLOSE_REASON)
    report["learning_removed"] = {t: exclude_learning_rows(conn, t, "id", keys, FALSE_CLOSE_REASON) for t, keys in learning.items()}
    for a in arms:
        w = 0.92 if int(a["n_obs"]) > 40 else 1.0
        _backup(conn, "day_outcome_bandit_arms", "arm_key", a["arm_key"], "revert_bandit_observation", FALSE_CLOSE_REASON)
        conn.execute(
            """UPDATE day_outcome_bandit_arms
               SET beta=MAX(1.0, beta-?), losses=MAX(0, losses-1), n_obs=MAX(0, n_obs-1), last_exit_reason=?
               WHERE arm_key=?""",
            (w, FALSE_CLOSE_REASON, a["arm_key"]),
        )
    conn.commit()
    print(json.dumps(report, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
