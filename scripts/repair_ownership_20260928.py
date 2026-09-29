#!/usr/bin/env python3
"""One-off accounting repair for the 2026-09-28 BTC protected-inventory oversell.

Metadata only. Every changed row is copied to ownership_repair_backup first.
Dry run by default; pass --apply with the service stopped.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.services.engine_lot_ownership import UNMATCHED_OVERSELL_REASON, fill_owned_quantity
from backend.services.ownership_repair import (
    ensure_backup_table,
    exclude_learning_rows,
    reclassify_unmatched_sell,
    relabel_generic_manual_exit,
    revert_bandit_observation,
    set_lot_quantity,
)

OVERSELL_ROW = 2642
FALSE_CLOSE_REASON = "false_human_manual_sell:stale_balance_snapshot"

# (table, key column, keys, reason)
LEARNING_ROWS = [
    ("trade_learning_outcomes", "id", [7911], UNMATCHED_OVERSELL_REASON),
    ("ai_outcome_training_rows", "id", [2552], UNMATCHED_OVERSELL_REASON),
    ("day_outcome_attribution", "id", [2431], UNMATCHED_OVERSELL_REASON),
    ("position_close_ledger", "id", [2626], UNMATCHED_OVERSELL_REASON),
    ("market_role_trade_outcomes", "id", [1208], UNMATCHED_OVERSELL_REASON),
    ("trade_learning_outcomes", "id", [7906], FALSE_CLOSE_REASON),
    ("scalp_learning_outcomes", "id", [5450], FALSE_CLOSE_REASON),
    ("ai_outcome_training_rows", "id", [2547], FALSE_CLOSE_REASON),
    ("day_outcome_attribution", "id", [2546], FALSE_CLOSE_REASON),
]

BANDITS = [
    {
        "arm_key": "BTC/USDT|RANGE_BOUNCE|range",
        "expect": {"n_obs": 20, "last_exit_reason": "MANUAL_EXIT"},
        "win": False,
        "weight_now": 1.0 + 0.3090673108272597 / 12.0,
        "pnl": 0.3090673108272597,
        "restore_last": (0.012812359739139856, "TRAILING_STOP_EXIT"),
        "reason": UNMATCHED_OVERSELL_REASON,
    },
    {
        # w=1.0 at n=131, then decayed x0.92 at n=131..134.
        "arm_key": "XRP/USDT|UNKNOWN|range",
        "expect": {"n_obs": 134, "last_updated": 1790623008.650892},
        "win": False,
        "weight_now": 0.92**4,
        "pnl": 0.0,
        "restore_last": None,
        "reason": FALSE_CLOSE_REASON,
    },
]

# Adaptive-weight buckets whose last EMA step came from a removed close.
WEIGHT_BUCKETS = [
    ("day", "BTCUSDT", "neutral::RANGE_BOUNCE", "2026-09-28T18:08:4", UNMATCHED_OVERSELL_REASON),
    ("scalp", "XRPUSDT", "unknown::NO_CLEAR_THESIS", "2026-09-28T16:21:3", FALSE_CLOSE_REASON),
]

DUST_LOTS = [
    ("mystic_ETH/USDT_1790568969440", "ETH/USDT"),
    ("scalp_v2_BTCUSDT_1790608985645", "BTC/USDT"),
]


def _restore_weights(conn: sqlite3.Connection, apply: bool, report: dict) -> None:
    from backend.services.ownership_repair import _backup

    for sid, sym, regime, stamp, reason in WEIGHT_BUCKETS:
        rows = conn.execute(
            "SELECT component_name, weight, previous_weight, updated_at FROM ai_strategy_score_weights WHERE LOWER(strategy_id)=? AND UPPER(symbol)=? AND LOWER(regime)=LOWER(?)",
            (sid, sym, regime),
        ).fetchall()
        for comp, weight, prev, updated in rows:
            ok = str(updated or "").startswith(stamp)
            report["weights"].append({"bucket": f"{sid}/{sym}/{regime}", "component": comp, "weight": weight, "restore_to": prev if ok else None, "last_update": updated})
            if ok and apply:
                key = f"{sid}|{sym}|{regime}|{comp}"
                conn.execute("CREATE TEMP VIEW IF NOT EXISTS _w AS SELECT strategy_id||'|'||symbol||'|'||regime||'|'||component_name AS k, * FROM ai_strategy_score_weights")
                _backup(conn, "_w", "k", key, "restore_adaptive_weight", reason, table_label="ai_strategy_score_weights")
                conn.execute(
                    "UPDATE ai_strategy_score_weights SET weight=previous_weight WHERE LOWER(strategy_id)=? AND UPPER(symbol)=? AND LOWER(regime)=LOWER(?) AND component_name=?",
                    (sid, sym, regime, comp),
                )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default="mystic_trading.db")
    ap.add_argument("--apply", action="store_true")
    args = ap.parse_args()

    conn = sqlite3.connect(args.db, timeout=30)
    conn.row_factory = sqlite3.Row
    report: dict = {"apply": args.apply, "relabel": [], "unrelabeled": [], "learning": [], "bandits": [], "weights": [], "dust": []}

    row = conn.execute("SELECT id, order_id, quantity, pnl, exit_reason FROM paper_trades WHERE id=?", (OVERSELL_ROW,)).fetchone()
    if row is None or str(row["order_id"]) != "1845996508":
        print("ABORT: row 2642 does not match order 1845996508")
        return 2
    report["row_2642"] = dict(row)

    manual = conn.execute(
        """SELECT id, engine_id, symbol, timestamp, quantity, order_id, explainability_json FROM paper_trades
           WHERE UPPER(side)='SELL' AND COALESCE(mode,'')='live' AND exit_reason='MANUAL_EXIT' AND id != ? ORDER BY id""",
        (OVERSELL_ROW,),
    ).fetchall()

    for tid, sym in DUST_LOTS:
        lot = conn.execute("SELECT quantity, status, engine_id FROM portfolio_engine_positions WHERE trade_id=?", (tid,)).fetchone()
        owned = fill_owned_quantity(conn, tid, sym)
        report["dust"].append({"trade_id": tid, "booked": lot["quantity"] if lot else None, "fill_owned": str(owned) if owned is not None else None})

    for b in BANDITS:
        arm = conn.execute("SELECT * FROM day_outcome_bandit_arms WHERE arm_key=?", (b["arm_key"],)).fetchone()
        match = arm is not None and all((abs(float(arm[k]) - v) < 1e-6) if isinstance(v, float) else arm[k] == v for k, v in b["expect"].items())
        report["bandits"].append({"arm": b["arm_key"], "before": dict(arm) if arm else None, "preconditions_hold": match})

    if not args.apply:
        for r in manual:
            raw = json.loads(r["explainability_json"] or "{}").get("raw_exit_reason")
            report["relabel"].append({"id": r["id"], "engine": r["engine_id"], "symbol": r["symbol"], "raw": raw})
        _restore_weights(conn, False, report)
        print(json.dumps(report, indent=2, default=str))
        return 0

    if not all(b["preconditions_hold"] for b in report["bandits"]):
        print("ABORT: bandit arm changed since the audit")
        print(json.dumps(report["bandits"], indent=2, default=str))
        return 3

    conn.execute("BEGIN IMMEDIATE")
    ensure_backup_table(conn)
    assert reclassify_unmatched_sell(conn, OVERSELL_ROW, UNMATCHED_OVERSELL_REASON)
    for r in manual:
        label = relabel_generic_manual_exit(conn, int(r["id"]))
        (report["relabel"] if label else report["unrelabeled"]).append({"id": r["id"], "engine": r["engine_id"], "label": label})
    for table, col, keys, reason in LEARNING_ROWS:
        report["learning"].append({"table": table, "keys": keys, "removed": exclude_learning_rows(conn, table, col, keys, reason)})
    for b in BANDITS:
        revert_bandit_observation(conn, b["arm_key"], win=b["win"], weight_now=b["weight_now"], pnl=b["pnl"], restore_last=b["restore_last"], reason=b["reason"])
    _restore_weights(conn, True, report)
    for tid, sym in DUST_LOTS:
        owned = fill_owned_quantity(conn, tid, sym)
        lot = conn.execute("SELECT quantity FROM portfolio_engine_positions WHERE trade_id=?", (tid,)).fetchone()
        if owned is not None and lot is not None and float(lot["quantity"]) > float(owned) + 1e-12:
            set_lot_quantity(conn, tid, float(owned), "dust_reconcile_quantity_inflation")
    conn.commit()

    from backend.services.market_role_outcome_learner import _recompute_stats

    _recompute_stats(args.db, "BTC/USDT", "day")
    print(json.dumps(report, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
