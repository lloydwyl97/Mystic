#!/usr/bin/env python3
"""Restore SCALP V2 lots the reconcile falsely closed as HUMAN_MANUAL_SELL.

Their coins were re-imported as protected inventory. Each lot is rebuilt from its
own BUY row at its fill-owned quantity, and the protected row gives that quantity
back. Every changed row is copied to ownership_repair_backup first.
Dry run by default; pass --apply with the service stopped.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.services.engine_lot_ownership import fill_owned_quantity
from backend.services.ownership_repair import _backup, ensure_backup_table
from backend.services.portfolio_engine import PortfolioEngine
from backend.services.scalp_v2.exit_calibration import SCALP_V2_ENGINE_ID

REASON = "restore_false_closed_scalp_lot:stale_balance_snapshot"
LOTS = [
    ("scalp_v2_SOLUSDT_1790604179476", "SOL/USDT"),
    ("scalp_v2_XRPUSDT_1790612481481", "XRP/USDT"),
]
MIN_PROTECTED_NOTIONAL = 1.0


def _plan(conn: sqlite3.Connection, trade_id: str, symbol: str) -> dict:
    buy = conn.execute(
        """SELECT id, price, quantity, remaining_position, order_id, COALESCE(fees_paid,0) AS fee, COALESCE(atr_at_entry,0) AS atr,
                  COALESCE(scalp_opportunity_id,'') AS opp, COALESCE(decision_id,'') AS decision, COALESCE(entry_timestamp, timestamp) AS ts
           FROM paper_trades WHERE trade_id=? AND UPPER(side)='BUY' AND paper_run_id='scalp_v2_live' AND mode='live'""",
        (trade_id,),
    ).fetchone()
    owned = fill_owned_quantity(conn, trade_id, symbol)
    lot = conn.execute("SELECT trade_id FROM portfolio_engine_positions WHERE engine_id=? AND symbol=?", (SCALP_V2_ENGINE_ID, symbol)).fetchone()
    prot = conn.execute("SELECT quantity, cost_price FROM protected_external_inventory WHERE symbol=?", (symbol,)).fetchone()
    plan = {"trade_id": trade_id, "symbol": symbol, "fill_owned": str(owned) if owned is not None else None}
    if buy is None or not buy["order_id"]:
        plan["abort"] = "no live SCALP BUY row with a venue order id"
    elif owned is None or float(owned) <= 0:
        plan["abort"] = "fills do not prove unsold quantity"
    elif lot is not None:
        plan["abort"] = f"SCALP lot already open ({lot['trade_id']})"
    elif prot is None or float(prot["quantity"]) + 1e-9 < float(owned):
        plan["abort"] = "protected inventory does not hold the lot quantity"
    else:
        rest = float(prot["quantity"]) - float(owned)
        plan.update(
            {
                "buy": dict(buy),
                "restore_qty": float(owned),
                "protected_before": float(prot["quantity"]),
                "protected_after": rest if rest * float(prot["cost_price"]) >= MIN_PROTECTED_NOTIONAL else 0.0,
            }
        )
    return plan


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default="mystic_trading.db")
    ap.add_argument("--apply", action="store_true")
    args = ap.parse_args()

    conn = sqlite3.connect(args.db, timeout=30)
    conn.row_factory = sqlite3.Row
    plans = [_plan(conn, tid, sym) for tid, sym in LOTS]
    print(json.dumps({"apply": args.apply, "plans": plans}, indent=2, default=str))
    if not args.apply:
        return 0
    if any("abort" in p for p in plans):
        print("ABORT: a lot failed its preconditions")
        return 2

    conn.execute("BEGIN IMMEDIATE")
    ensure_backup_table(conn)
    now = datetime.now(timezone.utc).isoformat()
    for p in plans:
        buy, sym, qty = p["buy"], p["symbol"], p["restore_qty"]
        _backup(conn, "paper_trades", "id", buy["id"], "restore_remaining_position", REASON)
        conn.execute("UPDATE paper_trades SET remaining_position=? WHERE id=?", (qty, buy["id"]))
        PortfolioEngine._scalp_v2_write_position_row(
            conn,
            symbol=sym,
            quantity=qty,
            fill_price=float(buy["price"]),
            fee=float(buy["fee"]),
            order_id=str(buy["order_id"]),
            atr=float(buy["atr"]),
            opportunity_id=str(buy["opp"]),
            decision_id=str(buy["decision"]),
            reservation_id="",
            client_order_id="",
            trade_id=p["trade_id"],
            entry_time=datetime.fromisoformat(str(buy["ts"])).timestamp(),
            timestamp=now,
        )
        _backup(conn, "protected_external_inventory", "symbol", sym, "return_to_strategy_lot", REASON)
        if p["protected_after"] > 0:
            conn.execute("UPDATE protected_external_inventory SET quantity=?, updated_at=strftime('%s','now') WHERE symbol=?", (p["protected_after"], sym))
        else:
            conn.execute("DELETE FROM protected_external_inventory WHERE symbol=?", (sym,))
    conn.commit()
    print("restored:", [dict(r) for r in conn.execute("SELECT symbol, engine_id, trade_id, quantity, status FROM portfolio_engine_positions")])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
