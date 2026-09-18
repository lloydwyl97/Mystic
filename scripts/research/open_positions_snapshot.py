"""Read-only: open DAY positions and how far each sits from its profit target."""

from __future__ import annotations

import sqlite3
import sys

DB = sys.argv[1] if len(sys.argv) > 1 else "mystic_trading.db"


def main() -> None:
    c = sqlite3.connect(DB, timeout=20)
    c.row_factory = sqlite3.Row
    cols = {r["name"] for r in c.execute("PRAGMA table_info(portfolio_engine_positions)")}
    want = [
        x
        for x in (
            "symbol",
            "quantity",
            "entry_price",
            "current_price",
            "highest_price",
            "trailing_stop_price",
            "thesis_target_level",
            "take_profit_1_price",
            "status",
            "unrealized_pnl",
        )
        if x in cols
    ]
    status_col = "status" if "status" in cols else None
    q = f"SELECT {', '.join(want)} FROM portfolio_engine_positions"
    if status_col:
        q += f" WHERE UPPER(COALESCE({status_col},'OPEN'))='OPEN'"
    rows = list(c.execute(q))
    print(f"  open positions: {len(rows)}")
    for r in rows:
        d = dict(r)
        entry = float(d.get("entry_price") or 0)
        mark = float(d.get("current_price") or 0)
        tgt = 0.0
        for k in ("take_profit_1_price", "thesis_target_level"):
            v = float(d.get(k) or 0)
            if v > entry > 0:
                tgt = v if tgt == 0 else min(tgt, v)
        line = f"    {d.get('symbol')} qty={d.get('quantity')} entry={entry} mark={mark}"
        if tgt and entry:
            line += f" target={tgt} target_dist={(tgt - entry) / entry * 100:.3f}%"
            if mark:
                line += f" mark_vs_target={'AT/ABOVE' if mark >= tgt else 'below'}"
        line += f" trail={d.get('trailing_stop_price')} high={d.get('highest_price')}"
        print(line)

    print("\n  positions table row count (any status):")
    print("   ", c.execute("SELECT COUNT(*) FROM portfolio_engine_positions").fetchone()[0])
    if status_col:
        for r in c.execute(f"SELECT {status_col} s, COUNT(*) n FROM portfolio_engine_positions GROUP BY s ORDER BY n DESC"):
            print(f"    {r['s']}: {r['n']}")


if __name__ == "__main__":
    main()
