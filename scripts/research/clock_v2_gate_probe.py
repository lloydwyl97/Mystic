"""Read-only probe: what did production itself record as the gate state?

Answers Part 2 empirically instead of by inference. For every decision group,
compares the selected symbol's recorded gate fields against the reconstruction,
so the hard-blocker set is derived from production's own point-in-time record.
"""

from __future__ import annotations

import json
import sqlite3
import sys
from collections import Counter

DB = sys.argv[1] if len(sys.argv) > 1 else "mystic_trading.db"

conn = sqlite3.connect(f"file:{DB}?mode=ro", uri=True)
conn.row_factory = sqlite3.Row

print("=== live schema ===")
for t in ("day_decision_candidate_records", "day_path_clock_v2_candidate_artifact"):
    cols = [r[1] for r in conn.execute(f"PRAGMA table_info({t})")]
    print(f"{t}: {len(cols)} cols")
    print(f"  {cols}")

groups = [
    dict(r)
    for r in conn.execute(
        "SELECT decision_group_id, selected_symbol, lifecycle_state, contract_json "
        "FROM day_decision_group_records ORDER BY created_at"
    )
]

sel_gates: Counter = Counter()
sel_missing: Counter = Counter()
group_slot: Counter = Counter()
exclusion_by_selected: Counter = Counter()
examples: list[dict] = []

for g in groups:
    try:
        c = json.loads(g["contract_json"] or "{}")
    except (TypeError, ValueError):
        continue
    cands = c.get("candidates") or []
    if not cands:
        continue
    sel = g["selected_symbol"]
    row = next((x for x in cands if str(x.get("symbol", "")).replace("/", "") == str(sel or "").replace("/", "")), None)
    if row is None:
        sel_missing["SELECTED_NOT_IN_CONTRACT_CANDIDATES"] += 1
        continue

    group_slot[f"slots_used={c.get('slots_used')}/{c.get('slot_count')}"] += 1
    key = (
        f"path_input_valid={row.get('path_input_valid')}",
        f"slot_available={row.get('slot_available')}",
        f"symbol_already_open={row.get('symbol_already_open')}",
        f"eligible={row.get('eligible')}",
    )
    sel_gates[key] += 1
    exclusion_by_selected[str(row.get("exclusion_reason"))] += 1
    if len(examples) < 3 and not row.get("eligible"):
        gate_keys = (
            "path_input_valid",
            "path_invalid_reason",
            "slot_available",
            "symbol_already_open",
            "eligible",
            "exclusion_reason",
            "rank_position",
            "final_rank_score",
            "path_ev",
        )
        examples.append(
            {
                "group": g["decision_group_id"],
                "selected": sel,
                "lifecycle": c.get("final_lifecycle_state") or g["lifecycle_state"],
                "gates": {k: row.get(k) for k in gate_keys},
                "group_state": {k: c.get(k) for k in ("slots_used", "slot_count", "open_symbols", "execute_authorized")},
            }
        )

print("\n=== selected symbol: production-recorded gate combinations ===")
for k, n in sel_gates.most_common():
    print(f"  {n:>4}  " + "  ".join(k))

print("\n=== selected symbol: recorded exclusion_reason ===")
for k, n in exclusion_by_selected.most_common():
    print(f"  {n:>4}  {k}")

print("\n=== group slot occupancy at decision time ===")
for k, n in group_slot.most_common(8):
    print(f"  {n:>4}  {k}")

if sel_missing:
    print("\n=== selected symbol absent from contract candidates ===")
    for k, n in sel_missing.items():
        print(f"  {n:>4}  {k}")

print("\n=== examples where selected symbol was recorded ineligible ===")
print(json.dumps(examples, indent=2, default=str))
conn.close()
