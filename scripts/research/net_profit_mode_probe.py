"""Read-only: were the realized NET_PROFIT_EXIT fills live or paper money?

Account equity is small, so the headline +$1,110 must be attributed to the
right ledger before it is used to justify anything.
"""

from __future__ import annotations

import sqlite3
import sys

DB = sys.argv[1] if len(sys.argv) > 1 else "mystic_trading.db"


def main() -> None:
    c = sqlite3.connect(DB, timeout=20)
    c.row_factory = sqlite3.Row
    cols = {r["name"] for r in c.execute("PRAGMA table_info(paper_trades)")}
    mode_col = next((x for x in ("mode", "trading_mode", "execution_mode") if x in cols), None)
    print(f"  mode column: {mode_col}")

    if mode_col:
        print("\n  NET_PROFIT_EXIT by mode:")
        for r in c.execute(
            f"SELECT COALESCE({mode_col},'(null)') m, COUNT(*) n, ROUND(COALESCE(SUM(pnl),0),2) p, "
            f"MIN(created_at) f, MAX(created_at) l FROM paper_trades "
            "WHERE UPPER(side)='SELL' AND exit_reason='NET_PROFIT_EXIT' GROUP BY m ORDER BY n DESC"
        ):
            print(f"    {r['m']:<10} n={r['n']:<5} pnl={r['p']:<10} {str(r['f'])[:19]} -> {str(r['l'])[:19]}")

        print("\n  ALL sell exits by mode:")
        for r in c.execute(f"SELECT COALESCE({mode_col},'(null)') m, COUNT(*) n, ROUND(COALESCE(SUM(pnl),0),2) p FROM paper_trades WHERE UPPER(side)='SELL' GROUP BY m ORDER BY n DESC"):
            print(f"    {r['m']:<10} n={r['n']:<5} pnl={r['p']}")

        print("\n  live-only exit reason breakdown:")
        for r in c.execute(
            f"SELECT COALESCE(exit_reason,'(null)') e, COUNT(*) n, ROUND(COALESCE(SUM(pnl),0),2) p, "
            f"MAX(created_at) l FROM paper_trades WHERE UPPER(side)='SELL' "
            f"AND UPPER(COALESCE({mode_col},''))='LIVE' GROUP BY e ORDER BY n DESC"
        ):
            print(f"    {r['e'][:32]:<32} n={r['n']:<5} pnl={r['p']:<10} last={str(r['l'])[:19]}")

    print("\n  ledger:")
    try:
        for r in c.execute("SELECT principal, cash_balance, positions_value, realized_pnl, total_equity FROM portfolio_engine_ledger WHERE id=1"):
            print(f"    {dict(r)}")
    except sqlite3.OperationalError as e:
        print(f"    unavailable: {e}")


if __name__ == "__main__":
    main()
