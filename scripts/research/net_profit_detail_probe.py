"""Read-only: did the realized net-profit exits fire as target_hit or profit_floor?

Form A (target_hit) keeps the existing "do not clip an unreached winner" contract
test green. Form B (bare profit floor) breaks it. Decide from the stored exit
detail rather than from an argument.
"""

from __future__ import annotations

import json
import sqlite3
import sys
from collections import Counter

DB = sys.argv[1] if len(sys.argv) > 1 else "mystic_trading.db"


def walk_keys(obj, prefix="", out=None):
    if out is None:
        out = set()
    if isinstance(obj, dict):
        for k, v in obj.items():
            out.add(f"{prefix}{k}")
            if isinstance(v, (dict, list)) and len(prefix) < 40:
                walk_keys(v, f"{prefix}{k}.", out)
    return out


def main() -> None:
    c = sqlite3.connect(DB, timeout=20)
    c.row_factory = sqlite3.Row

    rows = list(c.execute("SELECT explainability_json e, diagnostics_json d, pnl_pct, pnl, symbol FROM paper_trades WHERE UPPER(side)='SELL' AND exit_reason='NET_PROFIT_EXIT'"))
    print(f"  net-profit sell rows: {len(rows)}")

    allkeys: set[str] = set()
    for r in rows[:5]:
        for col in ("e", "d"):
            try:
                allkeys |= walk_keys(json.loads(r[col] or "{}"))
            except Exception:
                pass
    hits = sorted(k for k in allkeys if any(t in k.lower() for t in ("detail", "exit", "target", "reason", "tp")))
    print(f"  keys mentioning detail/exit/target/reason/tp:\n    {hits}")

    details: Counter[str] = Counter()
    target_evidence = []
    for r in rows:
        blob = {}
        for col in ("e", "d"):
            try:
                blob.update(json.loads(r[col] or "{}"))
            except Exception:
                pass
        det = ""
        for k in ("exit_detail", "detail", "exit_reason_detail", "managed_detail"):
            if blob.get(k):
                det = str(blob[k])
                break
        details[det or "(no detail stored)"] += 1
        tgt = None
        for k in ("target_price", "effective_target", "take_profit_1_price", "thesis_target_level"):
            v = blob.get(k)
            if v:
                tgt = float(v)
                break
        ent = blob.get("entry_price") or blob.get("entry")
        if tgt and ent:
            ent = float(ent)
            if ent > 0:
                target_evidence.append(((tgt - ent) / ent, float(r["pnl_pct"] or 0)))

    print("\n  stored exit detail values:")
    for k, v in details.most_common():
        print(f"    n={v:<5} {k[:70]}")

    if target_evidence:
        print(f"\n  target distance vs realized pnl_pct ({len(target_evidence)} rows with both):")
        near = sum(1 for t, p in target_evidence if abs(t - p) <= 0.0015)
        print(f"    realized within 15 bps of the target distance: {near}/{len(target_evidence)}")
        for t, p in target_evidence[:10]:
            print(f"      target={t:.5f} realized={p:.5f} delta={p - t:+.5f}")
    else:
        print("\n  no rows carried both a target and an entry price")


if __name__ == "__main__":
    main()
