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

from backend.services.day_outcome_bandit import LOSS_PNL_SCALE, MAX_WEIGHT, SLIDING_WINDOW, WIN_PNL_SCALE, _is_win, arm_key
from backend.services.ownership_repair import revert_bandit_observation

REASON = "scalp_close_updated_day_bandit"
UNRESOLVED = "UNRESOLVED_LEGACY_ATTRIBUTION"


def _weight(pnl: float, n_obs: int) -> float:
    scale = WIN_PNL_SCALE if pnl >= 0 else LOSS_PNL_SCALE
    w = min(MAX_WEIGHT, 1.0 + abs(pnl) / scale)
    return w * (0.92 if n_obs > SLIDING_WINDOW else 1.0)


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
            item = {
                "arm": arm["arm_key"],
                "sell_trade_id": sell["trade_id"],
                "pnl": pnl,
                "n_obs": int(arm["n_obs"]),
                "win": _is_win(pnl, sell["exit_reason"]),
                "weight_now": _weight(pnl, int(arm["n_obs"])),
            }
            # A windowed update also decays every older observation. That mix
            # cannot be inverted from the stored arm, so it is not reversed.
            (reversible if int(arm["n_obs"]) <= SLIDING_WINDOW else decayed).append(item)
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
    conn.execute("BEGIN IMMEDIATE")
    for item in reversible:
        arm = conn.execute("SELECT n_obs, last_pnl, last_exit_reason FROM day_outcome_bandit_arms WHERE arm_key=?", (item["arm"],)).fetchone()
        if arm is None or int(arm["n_obs"]) != int(item["n_obs"]) or abs(float(arm["last_pnl"] or 0) - item["pnl"]) > 1e-6:
            print("ABORT: arm changed", item["arm"])
            conn.rollback()
            return 3
        revert_bandit_observation(conn, item["arm"], win=item["win"], weight_now=item["weight_now"], pnl=item["pnl"], restore_last=None, reason=REASON)
        conn.execute("UPDATE day_outcome_bandit_arms SET last_exit_reason=? WHERE arm_key=?", (REASON, item["arm"]))
    conn.commit()
    print("reversed", len(reversible), "unresolved", report[UNRESOLVED])
    print("sample arm check", arm_key("SOL/USDT", "UNKNOWN", "range"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
