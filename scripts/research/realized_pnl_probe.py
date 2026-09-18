"""Read-only: reconcile portfolio_engine_ledger.realized_pnl against the trade record.

The ledger row and the API disagreed by roughly $976. Work out which one matches the
sum of actual closed trades, and whether the gap is DAY-only, scalp contamination, or
a stale ledger column.
"""

from __future__ import annotations

import sqlite3
import sys

DB = sys.argv[1] if len(sys.argv) > 1 else "mystic_trading.db"


def cols(c: sqlite3.Connection, table: str) -> list[str]:
    return [r[1] for r in c.execute(f"PRAGMA table_info({table})")]


def main() -> None:
    c = sqlite3.connect(DB, timeout=20)
    c.row_factory = sqlite3.Row

    print("=== portfolio_engine_ledger row ===")
    lcols = cols(c, "portfolio_engine_ledger")
    print("  columns: " + ", ".join(lcols))
    for r in c.execute("SELECT * FROM portfolio_engine_ledger"):
        for k, v in dict(r).items():
            print(f"  {k:<26} {v}")

    print("\n=== paper_trades shape ===")
    pcols = cols(c, "paper_trades")
    print("  columns: " + ", ".join(pcols))

    side = "side" if "side" in pcols else None
    pnl = next((x for x in ("realized_pnl", "pnl", "profit", "net_pnl") if x in pcols), None)
    strat = next((x for x in ("strategy", "engine", "source", "mode") if x in pcols), None)
    print(f"  using side={side} pnl={pnl} strategy_col={strat}")

    if pnl:
        print("\n=== sum of realized pnl over paper_trades ===")
        tot = c.execute(f"SELECT SUM({pnl}), COUNT(*) FROM paper_trades").fetchone()
        print(f"  all rows:            sum={tot[0]} n={tot[1]}")
        if side:
            for s in ("SELL", "BUY"):
                r = c.execute(
                    f"SELECT SUM({pnl}), COUNT(*) FROM paper_trades WHERE UPPER({side})=?",
                    (s,),
                ).fetchone()
                print(f"  {s:<20} sum={r[0]} n={r[1]}")
        if strat:
            print(f"\n  grouped by {strat}:")
            for r in c.execute(f"SELECT {strat} g, SUM({pnl}) s, COUNT(*) n FROM paper_trades GROUP BY {strat} ORDER BY n DESC"):
                print(f"    {r['g']!s:<24} sum={r['s']} n={r['n']}")

    print("\n=== scalp trades kept separate? ===")
    for t in ("scalp_paper_trades", "scalp_trades"):
        try:
            n = c.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
            tcols = cols(c, t)
            pc = next((x for x in ("realized_pnl", "pnl", "net_pnl") if x in tcols), None)
            s = c.execute(f"SELECT SUM({pc}) FROM {t}").fetchone()[0] if pc else None
            print(f"  {t:<24} rows={n} pnl_col={pc} sum={s}")
        except sqlite3.OperationalError:
            print(f"  {t:<24} (absent)")

    print("\n=== portfolio_engine_audit (fees/slippage) ===")
    try:
        acols = cols(c, "portfolio_engine_audit")
        print("  columns: " + ", ".join(acols))
        n = c.execute("SELECT COUNT(*) FROM portfolio_engine_audit").fetchone()[0]
        print(f"  rows={n}")
        for cand in ("fee", "fees", "fee_usd", "slippage", "slippage_usd"):
            if cand in acols:
                s = c.execute(f"SELECT SUM({cand}) FROM portfolio_engine_audit").fetchone()[0]
                print(f"  SUM({cand}) = {s}")
    except sqlite3.OperationalError as e:
        print(f"  {e}")

    print("\n=== open positions (unrealized side) ===")
    try:
        poscols = cols(c, "portfolio_engine_positions")
        print("  columns: " + ", ".join(poscols))
        for r in c.execute("SELECT * FROM portfolio_engine_positions"):
            d = dict(r)
            keep = {k: d[k] for k in ("symbol", "quantity", "entry_price", "entry_time", "status") if k in d}
            print("  " + "  ".join(f"{k}={v}" for k, v in keep.items()))
    except sqlite3.OperationalError as e:
        print(f"  {e}")


if __name__ == "__main__":
    main()
