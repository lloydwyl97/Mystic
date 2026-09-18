"""Read-only: do FILLED trailing-buy intents actually persist a trade row?

Positions appeared at 14:32 and 14:34 while the newest paper_trades BUY row was
from 00:24, which would mean a fill reached the exchange without being recorded.
Confirm or refute that before treating it as a defect.
"""

from __future__ import annotations

import sqlite3
import sys

DB = sys.argv[1] if len(sys.argv) > 1 else "mystic_trading.db"


def main() -> None:
    c = sqlite3.connect(DB, timeout=20)
    c.row_factory = sqlite3.Row

    print("=== intent status histogram ===")
    for r in c.execute("SELECT status, COUNT(*) n, MAX(updated_at) latest FROM day_trailing_buy_intents GROUP BY status ORDER BY n DESC"):
        print(f"  {r['status']!s:<14} n={r['n']:<5} latest={r['latest']}")

    print("\n=== newest FILLED intents ===")
    rows = list(
        c.execute(
            "SELECT symbol, intent_id, decision_id, status, order_id, client_order_id, "
            "fill_id, trade_id, reservation_id, order_accepted, quantity, arm_ask, "
            "lowest_ask, min_dip_bps, rebound_bps, round_trip_cost_bps, "
            "created_at, updated_at "
            "FROM day_trailing_buy_intents WHERE status='FILLED' "
            "ORDER BY rowid DESC LIMIT 6"
        )
    )
    if not rows:
        print("  none")
    for r in rows:
        d = {k: v for k, v in dict(r).items() if v not in (None, "", 0)}
        print("  " + " | ".join(f"{k}={v}" for k, v in d.items()))

    print("\n=== does each FILLED intent have a trade row? ===")
    for r in rows:
        iid = str(r["intent_id"])[:16]
        for key in ("trade_id", "order_id", "decision_id"):
            val = r[key]
            if not val:
                print(f"  intent {iid} {key} is EMPTY on the intent itself")
                continue
            n = c.execute(
                f"SELECT COUNT(*) FROM paper_trades WHERE {key}=?",
                (str(val),),
            ).fetchone()[0]
            flag = "" if n else "   <-- NO TRADE ROW"
            print(f"  intent {iid} {key}={val} -> paper_trades rows={n}{flag}")

    print("\n=== newest paper_trades rows, any side ===")
    for r in c.execute("SELECT symbol, side, quantity, price, order_id, status, mode, created_at FROM paper_trades ORDER BY created_at DESC LIMIT 8"):
        print("  " + " | ".join(f"{k}={v}" for k, v in dict(r).items()))

    print("\n=== positions vs their trade rows ===")
    for r in c.execute("SELECT symbol, quantity, entry_price, entry_time, trade_id, status, entry_decision_id FROM portfolio_engine_positions"):
        tid = str(r["trade_id"] or "")
        n = c.execute("SELECT COUNT(*) FROM paper_trades WHERE trade_id=?", (tid,)).fetchone()[0] if tid else 0
        print(f"  {r['symbol']:<10} qty={r['quantity']:<12} status={r['status']!s:<14} trade_id={tid[:34]:<34} trade_rows={n} entry_decision_id={r['entry_decision_id']}")


if __name__ == "__main__":
    main()
