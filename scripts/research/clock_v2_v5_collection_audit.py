"""Read-only v5 DEVELOPMENT collection audit.

Reports only point-in-time metadata allowed under the v5 DEVELOPMENT partition.
Never prints P&L, MFE, MAE, markouts, best-coin, or regret.
"""

from __future__ import annotations

import json
import sqlite3
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

_here = Path(__file__).resolve()
for _parent in _here.parents:
    if (_parent / "backend" / "services").is_dir():
        sys.path.insert(0, str(_parent))
        break
else:
    sys.path.insert(0, str(Path.cwd()))

from backend.services.day_4h_entry_features import COINS, HOLD_SYMBOL
from backend.services.day_clock_v2_action_contract import selected_action_invariant
from backend.services.day_clock_v2_labels import TABLE_V5_LABELS
from backend.services.day_clock_v2_partition import CLOCK_V2_V5_DEVELOPMENT_START, DEVELOPMENT
from backend.services.day_path_clock_v2 import REQUIRED_CLOCK_V2_FIELDS_V5
from backend.services.day_path_clock_v2_capture import (
    TABLE_ARTIFACT,
    group_completeness_v5,
)
from backend.services.day_path_clock_v2_readiness import evaluate_clock_v2_v5_readiness

COVERAGE_FIELDS = (
    "p_buy",
    "legacy_path_ev",
    "ret_5m",
    "ret_15m",
    "ret_30m",
    "realized_vol_10m",
    "btc_rel_ret_5m",
    "rel_volume_15m",
    "production_4h_break_true_at_decision",
    "distance_to_4h_break_bps",
    "4h_range_position",
    "spread_bps",
    "estimated_all_in_cost_bps",
)
HORIZON_SEC = 10800


def _ro(db: str) -> sqlite3.Connection:
    conn = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    return conn


def _parse(raw: object) -> datetime | None:
    if raw in (None, ""):
        return None
    try:
        dt = datetime.fromisoformat(str(raw).replace("Z", "+00:00"))
    except ValueError:
        return None
    return dt.replace(tzinfo=timezone.utc) if dt.tzinfo is None else dt.astimezone(timezone.utc)


def _load_dev_groups(db: str) -> list[dict]:
    conn = _ro(db)
    try:
        arts = [dict(r) for r in conn.execute(f"SELECT * FROM {TABLE_ARTIFACT}")]
        groups = {r["decision_group_id"]: dict(r) for r in conn.execute("SELECT decision_group_id, selected_symbol, created_at, schema_version, feature_schema FROM day_decision_group_records")}
        label_rows = 0
        label_valid = 0
        if conn.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (TABLE_V5_LABELS,)).fetchone():
            label_rows = conn.execute(f"SELECT COUNT(*) FROM {TABLE_V5_LABELS}").fetchone()[0]
            label_valid = conn.execute(f"SELECT COUNT(*) FROM {TABLE_V5_LABELS} WHERE label_valid=1").fetchone()[0]
    finally:
        conn.close()

    by_gid: dict[str, list[dict]] = {}
    for art in arts:
        feats = {}
        raw = art.get("feature_json")
        if raw:
            try:
                feats = json.loads(raw)
            except (TypeError, ValueError):
                feats = {}
        art["features"] = feats
        by_gid.setdefault(str(art["decision_group_id"]), []).append(art)

    start = _parse(CLOCK_V2_V5_DEVELOPMENT_START)
    out = []
    for gid, rows in by_gid.items():
        when_raw = rows[0].get("decision_timestamp") or rows[0].get("created_at")
        when = _parse(when_raw)
        part = rows[0].get("clock_v2_partition") or ""
        if part and part != DEVELOPMENT:
            continue
        if when is None or start is None or when < start:
            continue
        out.append({"gid": gid, "when": when, "arts": rows, "meta": groups.get(gid) or {}})
    out.sort(key=lambda g: g["when"])
    return out, label_rows, label_valid


def main() -> int:
    db = sys.argv[1] if len(sys.argv) > 1 else "mystic_trading.db"
    now = datetime.now(timezone.utc)
    groups, label_rows, label_valid = _load_dev_groups(db)
    snap = evaluate_clock_v2_v5_readiness(db)

    print(f"now_utc={now.isoformat()}")
    print(f"development_start={CLOCK_V2_V5_DEVELOPMENT_START}")
    print(f"v5_development_groups={len(groups)}")
    print()

    statuses = Counter()
    invariant_fail = 0
    available_complete = 0
    available_total = 0
    coverage = Counter()
    coverage_denom = 0
    mature = 0
    pending = 0
    per_group = []

    for g in groups:
        arts = g["arts"]
        comp = group_completeness_v5(arts)
        statuses[comp["status"]] += 1
        available_complete += int(comp["available_action_complete"])
        available_total += int(comp["available_action_total"])
        selected = g["meta"].get("selected_symbol") or HOLD_SYMBOL
        inv = selected_action_invariant(
            rows=[
                {
                    "symbol": a.get("symbol"),
                    "action_available": a.get("action_available"),
                    "action_unavailable_reason": a.get("action_unavailable_reason"),
                }
                for a in arts
            ],
            selected_symbol=selected,
        )
        if not inv.get("pass"):
            invariant_fail += 1
        age = (now - g["when"]).total_seconds()
        is_mature = age >= HORIZON_SEC
        if is_mature:
            mature += 1
        else:
            pending += 1
        missing: list[str] = []
        schema_versions = sorted({str(a.get("feature_schema_version") or "") for a in arts})
        contract_versions = sorted({str(a.get("feature_contract_version") or "") for a in arts})
        action_states = {}
        for a in arts:
            sym = str(a.get("symbol"))
            action_states[sym] = {
                "action_available": a.get("action_available"),
                "action_unavailable_reason": a.get("action_unavailable_reason"),
                "legacy_rank_candidate_present": a.get("legacy_rank_candidate_present"),
                "production_selected": bool(a.get("production_selected")) or (sym == selected),
            }
            if bool(a.get("action_available")) and a.get("action_available") is not None and sym != HOLD_SYMBOL:
                coverage_denom += 1
                feats = a.get("features") or {}
                for field in COVERAGE_FIELDS:
                    if feats.get(field) is not None:
                        coverage[field] += 1
                miss = [n for n in REQUIRED_CLOCK_V2_FIELDS_V5 if feats.get(n) is None]
                if miss:
                    missing.append(f"{sym}:{','.join(miss)}")
            reasons = a.get("missingness_reasons_json")
            if reasons:
                try:
                    parsed = json.loads(reasons) if isinstance(reasons, str) else reasons
                except (TypeError, ValueError):
                    parsed = None
                if parsed:
                    missing.append(f"{sym}:reasons={parsed}")
        per_group.append(
            {
                "group_id": g["gid"],
                "decision_timestamp": g["when"].isoformat(),
                "production_selected": selected,
                "action_availability": action_states,
                "feature_schema_version": schema_versions,
                "feature_contract_version": contract_versions,
                "completeness": comp["status"],
                "missingness": missing,
                "3h_mature": is_mature,
            }
        )

    print("=== GROUPS (metadata only) ===")
    print(json.dumps(per_group, indent=2, default=str))
    print()
    print("=== INVARIANT ===")
    print(f"groups_checked={len(groups)} violations={invariant_fail}")
    print()
    print("=== COMPLETENESS (DEVELOPMENT only) ===")
    print(f"FEATURE_COMPLETE={statuses.get('FEATURE_COMPLETE', 0)} / 190")
    print(f"FEATURE_PARTIAL={statuses.get('FEATURE_PARTIAL', 0)}")
    print(f"UNUSABLE={statuses.get('UNUSABLE', 0)}")
    print(f"complete_available_rows={available_complete} / available_rows={available_total}")
    print("coverage_of_available_coin_rows:")
    for field in COVERAGE_FIELDS:
        n = coverage[field]
        pct = (100.0 * n / coverage_denom) if coverage_denom else 0.0
        print(f"  {field}: {n}/{coverage_denom} ({pct:.0f}%)")
    print()
    print("=== LABEL MATURITY (counts only, no values) ===")
    print(f"v5_groups={len(groups)}")
    print(f"3h_mature_groups={mature}")
    print(f"pending_groups={pending}")
    print(f"label_table_rows={label_rows} label_valid_rows={label_valid}")
    if groups:
        first_plus = groups[0]["when"].timestamp() + HORIZON_SEC
        if now.timestamp() < first_plus:
            print("maturity_correctly_zero=true (now is before first_group + 10800s)")
        else:
            print("maturity_correctly_zero=false")
    else:
        print("maturity_correctly_zero=true (no v5 groups yet)")
    print()
    print("=== READINESS ===")
    keys = (
        "DATA_READINESS",
        "feature_complete_development_groups",
        "required_feature_complete_groups",
        "fully_comparable_development_groups",
        "required_fully_comparable_groups",
        "FEATURE_COMPLETE_GROUPS",
        "FULLY_COMPARABLE_GROUPS",
        "AUTHORITATIVE_CALIBRATION_FILLS",
        "authoritative_calibration_fills",
        "required_calibration_fills",
        "NON_HOLD_SELECTED_GROUPS",
        "EXECUTE_AUTHORIZED_GROUPS",
        "ACTUAL_LOGICAL_BUY_FILLS",
        "groups_with_valid_selected_v2_label",
        "groups_with_complete_execution_provenance",
        "chronological_span_days",
        "target_horizon_sec",
        "target_horizon_status",
        "accounting_pass",
        "future_data_violations",
        "selected_action_invariant_violations",
        "train",
        "promoted",
        "final_test_status",
        "clock_v2_v5_development_start",
    )
    print(json.dumps({k: snap.get(k) for k in keys}, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
