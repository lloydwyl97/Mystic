"""Read-only preflight for the clock-v2 eligibility correction.

Validates against real production data, without writing to it:
  1. the ALTER TABLE migrations apply cleanly to the real production schema
     (replayed on a scratch database built from the live schema, never the live file)
  2. every historical decision group reconstructs deterministically under the
     corrected contract, and the selected-action invariant holds afterwards

Usage: venv/bin/python3 scripts/research/clock_v2_preflight_validate.py [db_path]
"""

from __future__ import annotations

import json
import sqlite3
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from backend.services.day_clock_v2_action_contract import (
    NO_SCORED_CANDIDATE,
    RECONSTRUCTED_PIT,
    reconstruct_group_action_state,
)
from backend.services.day_decision_observability import (
    TABLE_CANDIDATES,
    _ensure_schema,
)
from backend.services.day_path_clock_v2_capture import (
    TABLE_ARTIFACT,
    ensure_artifact_schema,
)

MIGRATED_TABLES = (TABLE_CANDIDATES, TABLE_ARTIFACT)


def _ro(db_path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    return conn


def check_migration(db_path: str, scratch: Path) -> dict:
    """Replay the real production schema into a scratch DB and migrate that."""
    src = _ro(db_path)
    try:
        ddl = [
            r[0]
            for r in src.execute(
                "SELECT sql FROM sqlite_master WHERE sql IS NOT NULL AND tbl_name IN (?,?)",
                MIGRATED_TABLES,
            )
        ]
        before = {t: [r[1] for r in src.execute(f"PRAGMA table_info({t})")] for t in MIGRATED_TABLES}
    finally:
        src.close()

    scratch.unlink(missing_ok=True)
    dst = sqlite3.connect(str(scratch))
    try:
        for stmt in ddl:
            dst.execute(stmt)
        dst.commit()
    finally:
        dst.close()

    _ensure_schema(str(scratch))
    ensure_artifact_schema(str(scratch))

    dst = sqlite3.connect(str(scratch))
    try:
        after = {t: [r[1] for r in dst.execute(f"PRAGMA table_info({t})")] for t in MIGRATED_TABLES}
    finally:
        dst.close()
    scratch.unlink(missing_ok=True)

    return {
        "tables": {
            t: {
                "columns_before": len(before[t]),
                "columns_after": len(after[t]),
                "added": [c for c in after[t] if c not in before[t]],
                "dropped": [c for c in before[t] if c not in after[t]],
            }
            for t in MIGRATED_TABLES
        }
    }


def check_reconstruction(db_path: str) -> dict:
    conn = _ro(db_path)
    try:
        groups = conn.execute(
            "SELECT decision_group_id, selected_symbol, bar_timestamp FROM day_decision_group_records ORDER BY bar_timestamp"
        ).fetchall()
        by_group: dict[str, list[dict]] = {}
        for row in conn.execute(f"SELECT * FROM {TABLE_CANDIDATES}"):
            by_group.setdefault(row["decision_group_id"], []).append(dict(row))
        clock_v2_ids = {
            r[0] for r in conn.execute(f"SELECT DISTINCT decision_group_id FROM {TABLE_ARTIFACT}")
        }
        filled = {
            r[0]
            for r in conn.execute(
                "SELECT decision_group_id FROM day_decision_group_records WHERE lifecycle_state IN ('filled','FILLED')"
            )
        }
    finally:
        conn.close()

    status = Counter()
    invariant = Counter()
    defective_before = []
    unresolved = []
    fabricated_rank = 0

    for g in groups:
        gid = g["decision_group_id"]
        rows = by_group.get(gid, [])
        payload = {
            "decision_group_id": gid,
            "selected_symbol": g["selected_symbol"],
            "lifecycle_state": "filled" if gid in filled else "none",
            "candidates": [
                {
                    "symbol": r.get("symbol"),
                    "eligible": bool(r.get("eligible")),
                    "exclusion_reason": r.get("exclusion_reason"),
                    "path_input_valid": r.get("path_input_valid"),
                    "path_ev": r.get("path_ev"),
                    "final_rank_score": r.get("final_rank_score"),
                }
                for r in rows
            ],
        }
        sel = next((r for r in rows if r.get("symbol") == g["selected_symbol"]), None)
        was_defective = bool(sel) and not sel.get("eligible") and sel.get("exclusion_reason") == NO_SCORED_CANDIDATE

        out = reconstruct_group_action_state(payload)
        status[out["reconstruction_status"]] += 1
        invariant["pass" if out["selected_action_invariant"]["pass"] else "fail"] += 1
        fabricated_rank += sum(
            1 for r in out["rows"] if r.get("legacy_final_rank_score_valid") is False and r["symbol"] != "HOLD"
        )
        if was_defective:
            fixed = next((r for r in out["rows"] if r["symbol"] == g["selected_symbol"]), None)
            defective_before.append(
                {
                    "group": gid,
                    "symbol": g["selected_symbol"],
                    "clock_v2": gid in clock_v2_ids,
                    "filled": gid in filled,
                    "available_after": None if fixed is None else fixed["action_available"],
                }
            )
        if not out["selected_action_invariant"]["pass"]:
            unresolved.append({"group": gid, "violations": out["selected_action_invariant"]["violations"]})

    repaired = [d for d in defective_before if d["available_after"] is True]
    return {
        "groups_total": len(groups),
        "clock_v2_groups": len(clock_v2_ids),
        "reconstruction_status": dict(status),
        "pit_reconstructable": status.get(RECONSTRUCTED_PIT, 0),
        "selected_action_invariant": dict(invariant),
        "defective_selected_before": len(defective_before),
        "defective_repaired": len(repaired),
        "defective_by_symbol": Counter(d["symbol"] for d in defective_before).most_common(),
        "defective_filled": sum(1 for d in defective_before if d["filled"]),
        "fabricated_rank_scores_nulled": fabricated_rank,
        "unresolved_invariant_failures": unresolved[:5],
    }


def main() -> int:
    db = sys.argv[1] if len(sys.argv) > 1 else "mystic_trading.db"
    scratch = Path("/tmp/clock_v2_migration_scratch.db")
    report = {
        "db": db,
        "migration": check_migration(db, scratch),
        "reconstruction": check_reconstruction(db),
        "production_db_written": False,
    }
    print(json.dumps(report, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
