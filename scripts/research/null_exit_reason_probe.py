"""Read-only: the 72 live SELL rows with no exit_reason are the largest live loss.

-$128.73 of a -$152.76 live total sits in rows that never recorded why they sold.
Find out what closed them.
"""

from __future__ import annotations

import json
import sqlite3
import sys

DB = sys.argv[1] if len(sys.argv) > 1 else "mystic_trading.db"


def main() -> None:
    c = sqlite3.connect(DB, timeout=20)
    c.row_factory = sqlite3.Row
    cols = {r["name"] for r in c.execute("PRAGMA table_info(paper_trades)")}

    sel = [x for x in ("symbol", "quantity", "price", "pnl", "pnl_pct", "exit_type", "status", "order_id", "created_at", "strategy") if x in cols]
    rows = list(
        c.execute(
            f"SELECT {', '.join(sel)}, explainability_json e, diagnostics_json d FROM paper_trades "
            "WHERE UPPER(side)='SELL' AND UPPER(COALESCE(mode,''))='LIVE' "
            "AND (exit_reason IS NULL OR TRIM(exit_reason)='') ORDER BY created_at DESC"
        )
    )
    print(f"  null-reason live sells: {len(rows)}")

    print("\n  by exit_type:")
    for r in c.execute(
        "SELECT COALESCE(exit_type,'(null)') t, COUNT(*) n, ROUND(COALESCE(SUM(pnl),0),2) p "
        "FROM paper_trades WHERE UPPER(side)='SELL' AND UPPER(COALESCE(mode,''))='LIVE' "
        "AND (exit_reason IS NULL OR TRIM(exit_reason)='') GROUP BY t ORDER BY n DESC"
    ):
        print(f"    {r['t']:<26} n={r['n']:<5} pnl={r['p']}")

    print("\n  by symbol:")
    for r in c.execute(
        "SELECT symbol s, COUNT(*) n, ROUND(COALESCE(SUM(pnl),0),2) p FROM paper_trades "
        "WHERE UPPER(side)='SELL' AND UPPER(COALESCE(mode,''))='LIVE' "
        "AND (exit_reason IS NULL OR TRIM(exit_reason)='') GROUP BY s ORDER BY p"
    ):
        print(f"    {r['s']:<12} n={r['n']:<5} pnl={r['p']}")

    print("\n  by strategy:")
    try:
        for r in c.execute(
            "SELECT COALESCE(strategy,'(null)') s, COUNT(*) n, ROUND(COALESCE(SUM(pnl),0),2) p "
            "FROM paper_trades WHERE UPPER(side)='SELL' AND UPPER(COALESCE(mode,''))='LIVE' "
            "AND (exit_reason IS NULL OR TRIM(exit_reason)='') GROUP BY s ORDER BY n DESC"
        ):
            print(f"    {r['s']:<24} n={r['n']:<5} pnl={r['p']}")
    except sqlite3.OperationalError:
        pass

    print("\n  worst 10 individually:")
    for r in sorted(rows, key=lambda x: float(x["pnl"] or 0))[:10]:
        d = dict(r)
        print(f"    {str(d.get('created_at'))[:19]} {d.get('symbol')!s:<10} pnl={d.get('pnl')} qty={d.get('quantity')} price={d.get('price')} type={d.get('exit_type')} status={d.get('status')}")

    print("\n  dust write-off rate over time:")
    for r in c.execute(
        "SELECT substr(created_at,1,10) d, COUNT(*) n, ROUND(COALESCE(SUM(pnl),0),2) p "
        "FROM paper_trades WHERE UPPER(side)='SELL' AND UPPER(COALESCE(mode,''))='LIVE' "
        "AND exit_type='DUST_WRITEOFF' GROUP BY d ORDER BY d DESC LIMIT 14"
    ):
        print(f"    {r['d']} n={r['n']:<4} pnl={r['p']}")

    print("\n  stored raw reasons inside the json blobs:")
    tally: dict[str, int] = {}
    for r in rows:
        blob: dict = {}
        for col in ("e", "d"):
            try:
                blob.update(json.loads(r[col] or "{}"))
            except Exception:
                pass
        for k in ("exit_reason_raw", "raw_exit_reason", "exit_trigger", "canonical_exit_reason", "exit_type"):
            v = blob.get(k)
            if v:
                tally[f"{k}={v}"] = tally.get(f"{k}={v}", 0) + 1
    if tally:
        for k, v in sorted(tally.items(), key=lambda x: -x[1])[:12]:
            print(f"    n={v:<5} {k}")
    else:
        print("    nothing stored -- these rows carry no reason anywhere")


if __name__ == "__main__":
    main()
