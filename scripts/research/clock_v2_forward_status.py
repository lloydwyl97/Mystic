"""Read-only status of the corrected clock-v2 capture and the v5 contract.

Reports what the running service has actually written since the correction went
live: corrected action fields on new groups, the partition registry, the planned
v5 experiment row, the 3h label table, and the v5 readiness counters.
"""

from __future__ import annotations

import json
import sqlite3
import sys
from collections import Counter

DB = sys.argv[1] if len(sys.argv) > 1 else "mystic_trading.db"
conn = sqlite3.connect(f"file:{DB}?mode=ro", uri=True)
conn.row_factory = sqlite3.Row


def table_exists(name: str) -> bool:
    return bool(conn.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (name,)).fetchone())


print("=== corrected action fields on decision candidate records ===")
tot = conn.execute("SELECT COUNT(*) FROM day_decision_candidate_records").fetchone()[0]
filled_rows = conn.execute(
    "SELECT COUNT(*) FROM day_decision_candidate_records WHERE action_contract_version IS NOT NULL"
).fetchone()[0]
print(f"  rows total={tot:,}  with corrected contract={filled_rows:,}")
for r in conn.execute(
    "SELECT action_available, action_unavailable_reason, COUNT(*) n FROM day_decision_candidate_records "
    "WHERE action_contract_version IS NOT NULL GROUP BY 1,2 ORDER BY n DESC LIMIT 8"
):
    print(f"    available={r[0]!s:<5} reason={r[1]!s:<34} n={r[2]}")

print("\n=== newest groups written under the corrected contract ===")
rows = conn.execute(
    "SELECT g.decision_group_id, g.selected_symbol, g.lifecycle_state, c.symbol, c.eligible, "
    "c.exclusion_reason, c.action_available, c.action_unavailable_reason, c.legacy_rank_candidate_present, "
    "c.legacy_final_rank_score, c.legacy_final_rank_score_valid, c.production_selected, "
    "c.execution_resolvable_candidate_present "
    "FROM day_decision_candidate_records c JOIN day_decision_group_records g "
    "ON g.decision_group_id = c.decision_group_id "
    "WHERE c.action_contract_version IS NOT NULL ORDER BY c.created_at DESC LIMIT 12"
).fetchall()
for r in rows:
    star = " <== SELECTED" if r["production_selected"] else ""
    print(
        f"  {r['decision_group_id']} {r['symbol']:<8} v1_eligible={r['eligible']!s:<5} "
        f"v1_reason={r['exclusion_reason']!s:<19} avail={r['action_available']!s:<5} "
        f"legacy_member={r['legacy_rank_candidate_present']!s:<5} "
        f"legacy_rank={r['legacy_final_rank_score']!s:<8} valid={r['legacy_final_rank_score_valid']!s:<5}"
        f"{star}"
    )

print("\n=== selected-action invariant on corrected rows ===")
bad = conn.execute(
    "SELECT COUNT(*) FROM day_decision_candidate_records "
    "WHERE production_selected = 1 AND action_available = 0"
).fetchone()[0]
sel = conn.execute(
    "SELECT COUNT(*) FROM day_decision_candidate_records WHERE production_selected = 1"
).fetchone()[0]
print(f"  production_selected rows={sel}  recorded unavailable={bad}  (must be 0)")

print("\n=== clock-v2 partition registry ===")
if table_exists("day_clock_v2_partition_registry"):
    for r in conn.execute("SELECT * FROM day_clock_v2_partition_registry"):
        d = dict(r)
        print(f"  {json.dumps({k: d[k] for k in list(d)[:8]}, default=str)}")
else:
    print("  ABSENT")

print("\n=== clock-v2 artifact partitions ===")
if table_exists("day_path_clock_v2_candidate_artifact"):
    for r in conn.execute(
        "SELECT clock_v2_partition, COUNT(DISTINCT decision_group_id) g, COUNT(*) rows "
        "FROM day_path_clock_v2_candidate_artifact GROUP BY 1"
    ):
        print(f"  partition={r[0]!s:<22} groups={r[1]:<5} rows={r[2]}")

print("\n=== planned v5 in experiment registry ===")
if table_exists("day_experiment_registry"):
    for r in conn.execute(
        "SELECT experiment_id, timestamp, result, promoted FROM day_experiment_registry ORDER BY timestamp"
    ):
        mark = "  <== NEW v5" if "v5" in str(r[0]) else ""
        print(f"  {r[0]:<34} ts={r[1]:<28} result={r[2]!s:<18} promoted={r[3]}{mark}")
else:
    print("  ABSENT")

print("\n=== 3h v5 label table ===")
if table_exists("day_clock_v2_outcome_labels"):
    n = conn.execute("SELECT COUNT(*) FROM day_clock_v2_outcome_labels").fetchone()[0]
    valid = conn.execute("SELECT COUNT(*) FROM day_clock_v2_outcome_labels WHERE label_valid=1").fetchone()[0]
    reasons = Counter(
        r[0] for r in conn.execute("SELECT label_invalid_reason FROM day_clock_v2_outcome_labels WHERE label_valid=0")
    )
    print(f"  rows={n:,} valid={valid:,}")
    for k, v in reasons.most_common(5):
        print(f"    invalid: {k} = {v}")
else:
    print("  ABSENT (no labelling cycle has run yet)")

print("\n=== generic 4H lock (existence check only, outcomes not read) ===")
if table_exists("day_forward_lock_registry"):
    for r in conn.execute(
        "SELECT experiment_id, inspected FROM day_forward_lock_registry ORDER BY experiment_id"
    ):
        print(f"  {r[0]:<40} inspected={r[1]}")
else:
    print("  ABSENT")

conn.close()
