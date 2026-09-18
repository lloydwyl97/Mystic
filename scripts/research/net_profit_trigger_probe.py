"""Read-only: read the stored exit trigger/raw reason for realized net-profit exits."""

from __future__ import annotations

import json
import sqlite3
import sys
from collections import Counter

DB = sys.argv[1] if len(sys.argv) > 1 else "mystic_trading.db"

KEYS = (
    "exit_trigger",
    "exit_reason_full",
    "exit_reason_raw",
    "raw_exit_reason",
    "canonical_exit_reason",
    "exit_reason_canonical",
    "exit_skeleton_exit_tag",
    "exit_type",
    "exit_quality_label",
)


def main() -> None:
    c = sqlite3.connect(DB, timeout=20)
    c.row_factory = sqlite3.Row
    rows = list(c.execute("SELECT explainability_json e, diagnostics_json d, pnl_pct FROM paper_trades WHERE UPPER(side)='SELL' AND exit_reason='NET_PROFIT_EXIT'"))
    tallies: dict[str, Counter[str]] = {k: Counter() for k in KEYS}
    for r in rows:
        blob: dict = {}
        for col in ("e", "d"):
            try:
                blob.update(json.loads(r[col] or "{}"))
            except Exception:
                pass
        for k in KEYS:
            v = blob.get(k)
            tallies[k][str(v)[:60] if v is not None else "(null)"] += 1

    print(f"  rows: {len(rows)}\n")
    for k in KEYS:
        nonnull = sum(v for key, v in tallies[k].items() if key != "(null)")
        if not nonnull:
            continue
        print(f"  {k}:")
        for val, n in tallies[k].most_common(8):
            print(f"    n={n:<5} {val}")
        print()


if __name__ == "__main__":
    main()
