"""Read-only: did any order/trade exist for bars the ledger labelled final_decision='execute'?"""

from __future__ import annotations

import json
import sqlite3
import sys

DB = sys.argv[1] if len(sys.argv) > 1 else "mystic_trading.db"


def main() -> None:
    c = sqlite3.connect(DB, timeout=15)
    c.row_factory = sqlite3.Row

    print("=== final_decision distribution (all time) ===")
    for r in c.execute("SELECT final_decision, COUNT(*) n FROM day_decision_records GROUP BY final_decision ORDER BY n DESC"):
        print(f"  {r['final_decision']!s:<14} {r['n']}")

    print("\n=== rows labelled 'execute' today, cross-checked against real fills ===")
    rows = list(
        c.execute(
            "SELECT decision_id, created_at, symbol, requested_size, approved_size, detail_json "
            "FROM day_decision_records WHERE final_decision='execute' "
            "AND created_at >= date('now') ORDER BY created_at DESC LIMIT 12"
        )
    )
    if not rows:
        print("  (none today)")
    for r in rows:
        did = r["decision_id"]
        bar = ""
        try:
            bar = str(json.loads(r["detail_json"] or "{}").get("bar_timestamp", ""))
        except Exception:
            pass
        hits = []
        for tbl, col in (
            ("paper_trades", "decision_id"),
            ("portfolio_engine_positions", "decision_id"),
            ("portfolio_engine_orders", "decision_id"),
        ):
            try:
                n = c.execute(
                    f"SELECT COUNT(*) FROM {tbl} WHERE {col}=?",
                    (did,),
                ).fetchone()[0]
            except sqlite3.OperationalError as e:
                n = f"n/a({e.args[0][:22]})"
            hits.append(f"{tbl}={n}")
        print(f"  {r['created_at'][11:19]} {r['symbol']:<9} req={r['requested_size']} appr={r['approved_size']} bar={bar}")
        print("      " + "  ".join(str(h) for h in hits))

    print("\n=== any BUY fill at all in the last 6h? ===")
    for tbl in ("paper_trades", "portfolio_engine_positions"):
        try:
            cols = [x[1] for x in c.execute(f"PRAGMA table_info({tbl})")]
            tcol = next(
                (x for x in ("created_at", "timestamp", "opened_at", "entry_time") if x in cols),
                None,
            )
            if not tcol:
                print(f"  {tbl}: no time column among {cols[:8]}")
                continue
            n = c.execute(f"SELECT COUNT(*) FROM {tbl} WHERE {tcol} >= datetime('now','-6 hours')").fetchone()[0]
            print(f"  {tbl:<30} rows_last_6h={n} (time col {tcol})")
        except sqlite3.OperationalError as e:
            print(f"  {tbl}: {e}")


if __name__ == "__main__":
    main()
