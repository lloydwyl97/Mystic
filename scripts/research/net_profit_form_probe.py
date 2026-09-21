"""Read-only: which net-profit form earned the money, target_hit or profit_floor?

The path-aware restore has two candidate conditions, mirroring the ladder:
  A) target_hit    -- mark reached the resolved profit target, net >= floor*0.45
  B) profit_floor  -- net >= per-coin floor (0.004) and the thesis exit says sell

Form B collides with an existing contract test that asserts a 0.45% winner with
its target unreached must hold. Form A does not. So: how much of the realized
+$1,110 sits above the target, versus merely above the bare floor?
"""

from __future__ import annotations

import sqlite3
import sys

DB = sys.argv[1] if len(sys.argv) > 1 else "mystic_trading.db"
FLOOR = 0.004


def main() -> None:
    c = sqlite3.connect(DB, timeout=20)
    c.row_factory = sqlite3.Row

    cols = [r["name"] for r in c.execute("PRAGMA table_info(paper_trades)")]
    interesting = [x for x in cols if any(k in x.lower() for k in ("detail", "meta", "note", "json", "reason"))]
    print(f"  candidate detail columns: {interesting}")

    print(f"\n  NET_PROFIT_EXIT realized pnl_pct distribution (floor={FLOOR}):")
    buckets = (
        (-99.0, FLOOR * 0.45, f"below floor*0.45 ({FLOOR * 0.45})"),
        (FLOOR * 0.45, FLOOR, f"between {FLOOR * 0.45} and floor"),
        (FLOOR, 0.006, "floor to 0.6%"),
        (0.006, 0.010, "0.6% to 1.0%"),
        (0.010, 99.0, "1.0% and up"),
    )
    for lo, hi, label in buckets:
        r = c.execute(
            "SELECT COUNT(*) n, ROUND(COALESCE(SUM(pnl),0),2) p FROM paper_trades WHERE UPPER(side)='SELL' AND exit_reason='NET_PROFIT_EXIT' AND pnl_pct >= ? AND pnl_pct < ?",
            (lo, hi),
        ).fetchone()
        print(f"    {label:<34} n={r['n']:<4} pnl={r['p']}")

    r = c.execute("SELECT MIN(pnl_pct) a, MAX(pnl_pct) b, ROUND(AVG(pnl_pct),5) c, COUNT(*) n FROM paper_trades WHERE UPPER(side)='SELL' AND exit_reason='NET_PROFIT_EXIT'").fetchone()
    print(f"    range min={r['a']} max={r['b']} avg={r['c']} n={r['n']}")

    for col in interesting:
        if col in ("exit_reason",):
            continue
        try:
            rows = list(
                c.execute(
                    f"SELECT {col} v, COUNT(*) n, ROUND(COALESCE(SUM(pnl),0),2) p FROM paper_trades WHERE UPPER(side)='SELL' AND exit_reason='NET_PROFIT_EXIT' GROUP BY v ORDER BY n DESC LIMIT 12"
                )
            )
        except sqlite3.OperationalError:
            continue
        if not rows:
            continue
        print(f"\n  NET_PROFIT_EXIT grouped by {col}:")
        for row in rows:
            val = str(row["v"])[:80] if row["v"] is not None else "(null)"
            print(f"    n={row['n']:<4} pnl={row['p']:<10} {val}")

    print("\n  how many would Form A alone still catch?")
    print("    (needs target price per trade; approximating with pnl_pct >= floor*0.45)")
    r = c.execute(
        "SELECT COUNT(*) n, ROUND(COALESCE(SUM(pnl),0),2) p FROM paper_trades WHERE UPPER(side)='SELL' AND exit_reason='NET_PROFIT_EXIT' AND pnl_pct >= ?",
        (FLOOR * 0.45,),
    ).fetchone()
    print(f"    n={r['n']} pnl={r['p']}")


if __name__ == "__main__":
    main()
