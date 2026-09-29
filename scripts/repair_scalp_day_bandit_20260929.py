#!/usr/bin/env python3
"""Undo DAY bandit observations that are the latest update from a SCALP_V2 close.

A SCALP close used to call record_bandit_outcome. Only an arm whose latest
observation still matches one SCALP sell (pnl, exit reason, timestamp) is
reversed. Every other SCALP close is reported as UNRESOLVED_LEGACY_ATTRIBUTION.
Dry run by default; pass --apply with the service stopped.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.services.day_outcome_bandit import PRIOR_ALPHA, PRIOR_BETA, SLIDING_WINDOW, _is_win, _weight, invert_latest_bandit_update
from backend.services.ownership_repair import _backup, relabel_generic_manual_exit

REASON = "scalp_close_updated_day_bandit"
UNRESOLVED = "UNRESOLVED_LEGACY_ATTRIBUTION"


def _forward(alpha: float, beta: float, n_obs: int, pnl: float, exit_reason: str) -> tuple[float, float, int]:
    win = _is_win(pnl, exit_reason)
    weight = _weight(pnl)
    if win:
        alpha += weight
    else:
        beta += weight
    n_obs += 1
    if n_obs > SLIDING_WINDOW:
        alpha = PRIOR_ALPHA + (alpha - PRIOR_ALPHA) * 0.92
        beta = PRIOR_BETA + (beta - PRIOR_BETA) * 0.92
    return alpha, beta, n_obs


def _epoch(timestamp: str) -> float | None:
    try:
        return datetime.fromisoformat(str(timestamp)).timestamp()
    except ValueError:
        return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default="mystic_trading.db")
    ap.add_argument("--apply", action="store_true")
    args = ap.parse_args()
    conn = sqlite3.connect(args.db, timeout=30)
    conn.row_factory = sqlite3.Row
    sells = conn.execute(
        """SELECT symbol, pnl, exit_reason, timestamp, trade_id FROM paper_trades
           WHERE UPPER(side)='SELL' AND engine_id='SCALP_V2' AND COALESCE(pnl, 0) IS NOT NULL"""
    ).fetchall()
    arms = [dict(r) for r in conn.execute("SELECT * FROM day_outcome_bandit_arms")]
    reversible = []
    decayed = []
    for arm in arms:
        for sell in sells:
            pnl = float(sell["pnl"] or 0.0)
            stamp = _epoch(sell["timestamp"])
            if stamp is None or abs(float(arm["last_pnl"] or 0.0) - pnl) > 1e-6:
                continue
            if str(arm["last_exit_reason"] or "") != str(sell["exit_reason"] or ""):
                continue
            if abs(float(arm["last_updated"] or 0.0) - stamp) > 30.0:
                continue
            if not str(arm["arm_key"]).startswith(str(sell["symbol"]) + "|"):
                continue
            restored = invert_latest_bandit_update(
                alpha=float(arm["alpha"]),
                beta=float(arm["beta"]),
                wins=int(arm["wins"]),
                losses=int(arm["losses"]),
                total_pnl=float(arm["total_pnl"] or 0.0),
                n_obs=int(arm["n_obs"]),
                pnl=pnl,
                exit_reason=str(sell["exit_reason"] or ""),
            )
            if restored is None:
                decayed.append({"arm": arm["arm_key"], "sell_trade_id": sell["trade_id"], "reason": "counts_cannot_hold_observation"})
                break
            back_a, back_b, back_n = _forward(float(restored["alpha"]), float(restored["beta"]), int(restored["n_obs"]), pnl, str(sell["exit_reason"] or ""))
            if abs(back_a - float(arm["alpha"])) > 1e-6 or abs(back_b - float(arm["beta"])) > 1e-6 or back_n != int(arm["n_obs"]):
                decayed.append({"arm": arm["arm_key"], "sell_trade_id": sell["trade_id"], "reason": "round_trip_failed"})
                break
            reversible.append({"arm": arm["arm_key"], "sell_trade_id": sell["trade_id"], "pnl": pnl, "n_obs": int(arm["n_obs"]), "restored_n_obs": restored["n_obs"]})
            break
    matched_trades = {r["sell_trade_id"] for r in reversible + decayed}
    report = {
        "apply": args.apply,
        "scalp_sells": len(sells),
        "reversible_latest_observations": reversible,
        "identified_but_decayed_not_reversed": decayed,
        UNRESOLVED: len(sells) - len(matched_trades),
    }
    print(json.dumps(report, indent=2, default=str))
    if not args.apply:
        return 0
    manual = conn.execute("SELECT id FROM paper_trades WHERE upper(side)='SELL' AND exit_reason='MANUAL_EXIT'").fetchall()
    conn.execute("BEGIN IMMEDIATE")
    for item in reversible:
        arm = conn.execute("SELECT * FROM day_outcome_bandit_arms WHERE arm_key=?", (item["arm"],)).fetchone()
        if arm is None or int(arm["n_obs"]) != int(item["n_obs"]) or abs(float(arm["last_pnl"] or 0) - item["pnl"]) > 1e-6:
            print("ABORT: arm changed", item["arm"])
            conn.rollback()
            return 3
        restored = invert_latest_bandit_update(
            alpha=float(arm["alpha"]),
            beta=float(arm["beta"]),
            wins=int(arm["wins"]),
            losses=int(arm["losses"]),
            total_pnl=float(arm["total_pnl"] or 0.0),
            n_obs=int(arm["n_obs"]),
            pnl=item["pnl"],
            exit_reason=str(arm["last_exit_reason"] or ""),
        )
        if restored is None:
            print("ABORT: inverse rejected", item["arm"])
            conn.rollback()
            return 3
        _backup(conn, "day_outcome_bandit_arms", "arm_key", item["arm"], "invert_latest_scalp_observation", REASON)
        conn.execute(
            """UPDATE day_outcome_bandit_arms
               SET alpha=?, beta=?, wins=?, losses=?, total_pnl=?, n_obs=?, last_exit_reason=?
               WHERE arm_key=?""",
            (restored["alpha"], restored["beta"], restored["wins"], restored["losses"], restored["total_pnl"], restored["n_obs"], REASON, item["arm"]),
        )
    relabeled = []
    for row in manual:
        label = relabel_generic_manual_exit(conn, int(row["id"]))
        if label:
            relabeled.append({"id": row["id"], "label": label})
    conn.commit()
    print("reversed", len(reversible), "relabeled", relabeled, "unresolved", report[UNRESOLVED])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
