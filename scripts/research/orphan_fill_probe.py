"""Read-only: locate every trace of a specific accepted order id across the database.

The newest FILLED trailing-buy intent carried a real Binance.US order id but no
trade_id, no fill_id and no paper_trades row. Before calling that a persistence
failure, search every table for the id in case the fill landed somewhere else.
"""

from __future__ import annotations

import sqlite3
import sys

DB = sys.argv[1] if len(sys.argv) > 1 else "mystic_trading.db"
NEEDLES = sys.argv[2:] or ["1587098371", "tb336a886a2a2c4cd4", "res_545f8ed975aa4e44"]


def main() -> None:
    c = sqlite3.connect(DB, timeout=30)
    c.row_factory = sqlite3.Row

    tables = [r[0] for r in c.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")]
    print(f"searching {len(tables)} tables for {NEEDLES}\n")

    for needle in NEEDLES:
        print(f"=== {needle} ===")
        hits = 0
        for t in tables:
            try:
                cols = [x[1] for x in c.execute(f"PRAGMA table_info({t})")]
            except sqlite3.OperationalError:
                continue
            if not cols:
                continue
            # Only text-comparable columns; skip huge blob tables by sampling cheaply.
            where = " OR ".join(f'CAST("{x}" AS TEXT) = ?' for x in cols)
            try:
                n = c.execute(
                    f'SELECT COUNT(*) FROM "{t}" WHERE {where}',
                    tuple([needle] * len(cols)),
                ).fetchone()[0]
            except sqlite3.OperationalError:
                continue
            if n:
                hits += 1
                print(f"  {t:<40} rows={n}")
                for row in c.execute(
                    f'SELECT * FROM "{t}" WHERE {where} LIMIT 2',
                    tuple([needle] * len(cols)),
                ):
                    d = {k: v for k, v in dict(row).items() if v not in (None, "", 0)}
                    trimmed = {k: (str(v)[:70] + "...") if len(str(v)) > 70 else v for k, v in list(d.items())[:14]}
                    print("      " + " | ".join(f"{k}={v}" for k, v in trimmed.items()))
        if not hits:
            print("  NOT FOUND IN ANY TABLE")
        print()


if __name__ == "__main__":
    main()
