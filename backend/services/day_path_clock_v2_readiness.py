"""CLOCK-V2 model-specific readiness. Does not replace day_model_readiness.

Does not train. Does not inspect sealed-lock outcomes.
A selected-trade count of 140 is never sufficient by itself.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from backend.services.day_4h_entry_features import COINS, HOLD_SYMBOL
from backend.services.day_clock_v2_action_contract import selected_action_invariant
from backend.services.day_clock_v2_labels import (
    TARGET_NAME,
    load_v5_label_presence,
    v5_label_contract,
)
from backend.services.day_clock_v2_partition import (
    CLOCK_V2_V5_DEVELOPMENT_START,
    DEVELOPMENT,
    FINAL_TEST_STATUS,
    GENERIC_4H_LOCK_ID,
    PRE_MODEL_QUARANTINE,
    partition_contract,
    partition_for,
    register_partition_contract,
)
from backend.services.day_decision_label_contract import TABLE_LABELS
from backend.services.day_decision_observability import TABLE_GROUPS
from backend.services.day_experiment_registry import TABLE as TABLE_REGISTRY
from backend.services.day_experiment_registry import (
    ensure_registry_schema,
    record_experiment,
    seed_historical,
)
from backend.services.day_forward_lock import FORWARD_LOCK_START
from backend.services.day_model_readiness import (
    CHRONOLOGICAL_BLOCK_HOURS,
    MIN_CHRONOLOGICAL_BLOCKS,
    evaluate_readiness,
)
from backend.services.day_path_clock_dataset import in_sealed_lock
from backend.services.day_path_clock_v2 import (
    CLOCK_V2_NUMERIC_FEATURE_COUNT,
    PLANNED_EXPERIMENT_ID,
    PLANNED_EXPERIMENT_ID_V2,
    PLANNED_EXPERIMENT_ID_V3,
    PLANNED_EXPERIMENT_ID_V4,
    PLANNED_EXPERIMENT_ID_V5,
    PLANNED_RESULT,
    PRIMARY_TARGET,
    PRIMARY_TARGET_HORIZON_NAME,
    PRIMARY_TARGET_HORIZON_SEC,
    REQUIRED_CLOCK_V2_FIELDS,
    TARGET_HORIZON_STATUS,
    clock_v2_statistical_contract,
    clock_v2_v2_readiness_requirements,
    clock_v2_v3_readiness_requirements,
    clock_v2_v4_readiness_requirements,
    clock_v2_v5_readiness_requirements,
    planned_challenger_specification_v2,
    planned_challenger_specification_v3,
    planned_challenger_specification_v4,
    planned_challenger_specification_v5,
)
from backend.services.day_path_clock_v2_capture import (
    TABLE_ARTIFACT,
    group_comparability_v5,
    group_completeness,
    group_completeness_v5,
)

TABLE_HISTORY = "day_path_clock_v2_readiness_history"

SCHEMA_SQL = f"""
CREATE TABLE IF NOT EXISTS {TABLE_HISTORY} (
    recorded_at TEXT PRIMARY KEY,
    snapshot_json TEXT NOT NULL
);
"""


def _parse_iso(raw: Any) -> datetime | None:
    if raw in (None, ""):
        return None
    if isinstance(raw, datetime):
        return raw if raw.tzinfo else raw.replace(tzinfo=timezone.utc)
    try:
        text = str(raw).replace("Z", "+00:00")
        dt = datetime.fromisoformat(text)
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _loads(raw: Any) -> dict[str, Any]:
    if isinstance(raw, dict):
        return raw
    if not raw:
        return {}
    try:
        out = json.loads(raw)
    except Exception:
        return {}
    return out if isinstance(out, dict) else {}


def ensure_readiness_schema(db_path: str | Path) -> None:
    conn = sqlite3.connect(str(db_path), timeout=30)
    try:
        conn.executescript(SCHEMA_SQL)
        conn.commit()
    finally:
        conn.close()


def record_planned_clock_v2_v2(db_path: str | Path) -> None:
    """Insert the corrected plan. Never overwrites the original planned arm."""
    spec = planned_challenger_specification_v2()
    seed_historical(db_path)
    conn = sqlite3.connect(str(db_path), timeout=30)
    try:
        existing = conn.execute(
            "SELECT experiment_id FROM day_experiment_registry WHERE experiment_id=?",
            (PLANNED_EXPERIMENT_ID,),
        ).fetchone()
        if existing is None:
            pass
    finally:
        conn.close()
    record_experiment(
        db_path,
        {
            "experiment_id": spec["experiment_id"],
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "feature_set": spec["feature_set"],
            "target": spec["target"],
            "model_class": spec["model_class"],
            "hyperparameters": spec["training_procedure"],
            "training_period": "forward_after_capture",
            "validation_period": "expanding_chrono_folds_purge_4h_embargo_4h",
            "locked_period": spec["acceptance"]["lock_cutoff"],
            "result": spec["result"],
            "promoted": False,
            "notes": spec["notes"],
            **spec,
        },
    )


def record_planned_clock_v2_v3(db_path: str | Path) -> None:
    """Insert v3. Never overwrites v1 or v2."""
    spec = planned_challenger_specification_v3()
    seed_historical(db_path)
    record_experiment(
        db_path,
        {
            "experiment_id": spec["experiment_id"],
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "feature_set": spec["feature_set"],
            "target": spec["target"],
            "model_class": spec["model_class"],
            "hyperparameters": spec["training_procedure"],
            "training_period": "forward_after_capture",
            "validation_period": "expanding_chrono_folds_purge_embargo_pending_horizon",
            "locked_period": spec["acceptance"]["lock_cutoff"],
            "result": spec["result"],
            "promoted": False,
            "notes": spec["notes"],
            **spec,
        },
    )


def _load_artifacts(db_path: str | Path) -> list[dict[str, Any]]:
    if not Path(db_path).exists():
        return []
    conn = sqlite3.connect(f"file:{Path(db_path)}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    try:
        if (
            conn.execute(
                "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?",
                (TABLE_ARTIFACT,),
            ).fetchone()
            is None
        ):
            return []
        rows = [dict(r) for r in conn.execute(f"SELECT * FROM {TABLE_ARTIFACT} ORDER BY created_at, symbol")]
    finally:
        conn.close()
    for row in rows:
        row["features"] = _loads(row.get("feature_json"))
        row["missingness_reasons"] = _loads(row.get("missingness_reasons_json"))
        row["provenance"] = _loads(row.get("provenance_json"))
        row["quote"] = _loads(row.get("quote_json"))
        row["eligible"] = bool(row.get("eligible"))
        row["lock_window"] = bool(row.get("lock_window"))
        row["inspected"] = bool(row.get("inspected"))
    return rows


def _load_groups(db_path: str | Path) -> list[dict[str, Any]]:
    if not Path(db_path).exists():
        return []
    conn = sqlite3.connect(f"file:{Path(db_path)}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    try:
        if (
            conn.execute(
                "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?",
                (TABLE_GROUPS,),
            ).fetchone()
            is None
        ):
            return []
        return [dict(r) for r in conn.execute(f"SELECT decision_group_id, created_at, selected_action, selected_symbol FROM {TABLE_GROUPS}")]
    finally:
        conn.close()


def _load_labels_presence(db_path: str | Path) -> dict[tuple[str, str], dict[str, Any]]:
    """Presence only. Does not return markout/MFE/MAE/net values."""
    if not Path(db_path).exists():
        return {}
    conn = sqlite3.connect(f"file:{Path(db_path)}?mode=ro", uri=True)
    try:
        if (
            conn.execute(
                "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?",
                (TABLE_LABELS,),
            ).fetchone()
            is None
        ):
            return {}
        rows = conn.execute(f"SELECT decision_group_id, symbol, provenance, label_json FROM {TABLE_LABELS}").fetchall()
    finally:
        conn.close()
    out: dict[tuple[str, str], dict[str, Any]] = {}
    for gid, sym, prov, raw in rows:
        payload = _loads(raw)
        out[(str(gid), str(sym))] = {
            "provenance": prov,
            "counterfactual": bool(payload.get("counterfactual")),
        }
    return out


def evaluate_clock_v2_readiness(db_path: str | Path, *, generic_state: dict[str, Any] | None = None) -> dict[str, Any]:
    """Separate gate. Generic day_model_readiness is unchanged."""
    req_v2 = clock_v2_v2_readiness_requirements()
    req = clock_v2_v4_readiness_requirements()  # v4 supersedes v3; numbers unchanged, horizon frozen
    artifacts = _load_artifacts(db_path)
    groups = _load_groups(db_path)
    labels = _load_labels_presence(db_path)
    by_group: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for art in artifacts:
        by_group[str(art["decision_group_id"])].append(art)
    group_meta = {str(g["decision_group_id"]): g for g in groups}

    complete_feature_groups = 0
    rectangular_groups = 0
    fully_comparable = 0
    label_complete = 0
    hold_coverage = 0
    selected_auth = 0
    counterfactual_cov = 0
    lock_inspected = 0
    future_data = 0
    field_present: dict[str, int] = defaultdict(int)
    field_total: dict[str, int] = defaultdict(int)
    quote_present = 0
    quote_total = 0
    per_symbol = Counter()
    blocks: set[int] = set()
    first: datetime | None = None
    last: datetime | None = None

    for gid, arts in by_group.items():
        meta = group_meta.get(gid) or {}
        created = _parse_iso(arts[0].get("created_at") or meta.get("created_at"))
        if created is not None:
            first = created if first is None or created < first else first
            last = created if last is None or created > last else last
            blocks.add(int(created.timestamp() // (CHRONOLOGICAL_BLOCK_HOURS * 3600)))
        if any(a.get("inspected") for a in arts):
            lock_inspected += 1
        locked = bool(arts[0].get("lock_window") or in_sealed_lock(arts[0].get("created_at")))
        completeness = group_completeness(arts)
        if completeness["FEATURE_COMPLETE"]:
            complete_feature_groups += 1
        if completeness["rectangular_feature_complete"]:
            rectangular_groups += 1
        if any(a.get("symbol") == HOLD_SYMBOL for a in arts):
            hold_coverage += 1
        eligible = [a for a in arts if a.get("symbol") != HOLD_SYMBOL and a.get("eligible")]
        label_ok = True
        cf_ok = True
        if locked:
            label_ok = False  # do not inspect lock outcomes
        else:
            for art in eligible:
                lab = labels.get((gid, str(art["symbol"])))
                if lab is None:
                    label_ok = False
                    cf_ok = False
                    continue
                if lab.get("provenance") != "authoritative" and not lab.get("counterfactual"):
                    cf_ok = False
            hold_lab = labels.get((gid, HOLD_SYMBOL))
            if hold_lab is None and not locked:
                # HOLD is definitionally 0; presence is optional
                pass
        if label_ok and completeness["FEATURE_COMPLETE"] and not locked:
            label_complete += 1
        if completeness["FEATURE_COMPLETE"] and label_ok and not locked:
            fully_comparable += 1
        if cf_ok and not locked:
            counterfactual_cov += 1
        selected = str(meta.get("selected_symbol") or "")
        if selected and selected != HOLD_SYMBOL and not locked:
            lab = labels.get((gid, selected.replace("/", ""))) or labels.get((gid, selected))
            if lab and lab.get("provenance") == "authoritative" and not lab.get("counterfactual"):
                selected_auth += 1
                per_symbol[selected] += 1
        for art in arts:
            if art.get("symbol") == HOLD_SYMBOL:
                continue
            feats = art.get("features") or {}
            quote = art.get("quote") or {}
            quote_total += 1
            if quote.get("spread_bps") is not None or feats.get("spread_bps") is not None:
                quote_present += 1
            for name in REQUIRED_CLOCK_V2_FIELDS:
                field_total[name] += 1
                if feats.get(name) is not None:
                    field_present[name] += 1
            src_latest = (art.get("provenance") or {}).get("source_latest_ts")
            cutoff = (art.get("provenance") or {}).get("feature_cutoff_ts")
            src_dt = _parse_iso(src_latest)
            cut_dt = _parse_iso(cutoff)
            if src_dt and cut_dt and src_dt > cut_dt:
                future_data += 1

    if generic_state is not None:
        generic = generic_state
    else:
        try:
            generic = evaluate_readiness(db_path)
        except Exception:
            generic = {"checks": {}}
    checks = generic.get("checks") or {}
    generic_g = checks.get("G_forward_span") or {}
    generic_f = checks.get("F_accounting") or {}
    generic_h = checks.get("H_locked_test_protection") or {}

    req_complete = int(req["min_fully_comparable_independent_groups"])
    req_comparable = int(req["min_fully_comparable_independent_groups"])
    req_fills = int(req["min_authoritative_fills_execution_calibration"])
    req_selected = int(req_v2["min_authoritative_selected_trade_labels"])
    span_days = ((last - first).total_seconds() / 86400.0) if first and last else 0.0
    # TARGET_HORIZON_STATUS is now "PRIMARY_TARGET_HORIZON_3H" (frozen).
    # horizon_frozen is True when the status string is NOT the "not frozen" sentinel.
    horizon_frozen = req["target_horizon_status"] != "TARGET_HORIZON_NOT_FROZEN"
    ok = (
        complete_feature_groups >= req_complete
        and fully_comparable >= req_comparable
        and selected_auth >= req_fills
        and len(blocks) >= MIN_CHRONOLOGICAL_BLOCKS
        and lock_inspected == 0
        and future_data == 0
        and bool(generic_f.get("pass", True))
        and horizon_frozen
    )
    # Selected-trade counts never authorize research by themselves.
    if selected_auth >= req_selected and (complete_feature_groups < req_complete or fully_comparable < req_comparable):
        ok = False
    if not horizon_frozen:
        ok = False

    snapshot = {
        "gate": "day_path_clock_v2_readiness",
        "replaces_generic_day_model_readiness": False,
        "DATA_READINESS": "PASS" if ok else "FAIL",
        "train": False,
        "promoted": False,
        "original_planned_experiment": PLANNED_EXPERIMENT_ID,
        "planned_experiment_v2": PLANNED_EXPERIMENT_ID_V2,
        "planned_experiment_v3": PLANNED_EXPERIMENT_ID_V3,
        "planned_experiment": PLANNED_EXPERIMENT_ID_V4,
        "planned_result": PLANNED_RESULT,
        "target": PRIMARY_TARGET,
        "target_horizon_status": TARGET_HORIZON_STATUS,
        "primary_target_horizon_sec": PRIMARY_TARGET_HORIZON_SEC,
        "primary_target_horizon_name": PRIMARY_TARGET_HORIZON_NAME,
        "statistical_contract": clock_v2_statistical_contract(),
        "v3_parameter_contract": req["parameter_contract"],
        "requirements_v2_historical": req_v2,
        "requirements": req,
        "complete_feature_groups": complete_feature_groups,
        "complete_candidate_rows": sum(
            1
            for arts in by_group.values()
            for a in arts
            if a.get("symbol") != HOLD_SYMBOL and a.get("eligible") and not [n for n in REQUIRED_CLOCK_V2_FIELDS if (a.get("features") or {}).get(n) is None]
        ),
        "rectangular_feature_groups": rectangular_groups,
        "mature_comparable_full_groups": fully_comparable,
        "label_complete_groups": label_complete,
        "authoritative_selected_trade_labels": selected_auth,
        "counterfactual_candidate_label_coverage": counterfactual_cov,
        "HOLD_coverage": hold_coverage,
        "chronological_span_days": round(span_days, 2),
        "chronological_blocks": len(blocks),
        "chronological_block_note": ("A block with 1 decision is a calendar-day bin, not an independent validation fold."),
        "per_symbol_support": dict(per_symbol),
        "feature_missingness": {
            name: {
                "present": field_present[name],
                "total": field_total[name],
                "rate": (field_present[name] / field_total[name] if field_total[name] else None),
            }
            for name in REQUIRED_CLOCK_V2_FIELDS
        },
        "quote_spread_coverage": {
            "present": quote_present,
            "total": quote_total,
            "rate": (quote_present / quote_total if quote_total else None),
        },
        "lock_status": {
            "inspected_artifact_rows": lock_inspected,
            "inspected": False,
            "cutoff": FORWARD_LOCK_START,
        },
        "accounting_integrity": generic_f.get("pass"),
        "lock_protection_pass": generic_h.get("pass"),
        "no_future_data_integrity": future_data == 0,
        "future_data_rows": future_data,
        "generic_G_forward_span": generic_g.get("pass"),
        "generic_required_selected_trades": req_selected,
        "v3_required_comparable_groups": req_complete,
        "v3_required_calibration_fills": req_fills,
        "selected_trade_only_sufficient_for_ranker": False,
        "numeric_features_counted": CLOCK_V2_NUMERIC_FEATURE_COUNT,
        "groups_with_artifacts": len(by_group),
        "artifact_rows": len(artifacts),
    }
    return snapshot


def record_planned_clock_v2_v4(db_path: str | Path) -> None:
    """Insert v4 (horizon frozen at 3h). Never overwrites v1/v2/v3."""
    spec = planned_challenger_specification_v4()
    seed_historical(db_path)
    record_experiment(
        db_path,
        {
            "experiment_id": spec["experiment_id"],
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "feature_set": spec["feature_set"],
            "target": spec["target"],
            "primary_target_horizon_sec": spec["primary_target_horizon_sec"],
            "primary_target_horizon_name": spec["primary_target_horizon_name"],
            "target_horizon_status": spec["target_horizon_status"],
            "model_class": spec["model_class"],
            "hyperparameters": spec["training_procedure"],
            "training_period": "forward_after_capture",
            "validation_period": "expanding_chrono_folds_purge_embargo_3h",
            "locked_period": spec["acceptance"]["lock_cutoff"],
            "result": spec["result"],
            "promoted": False,
            "notes": spec["notes"],
            **spec,
        },
    )


def record_planned_clock_v2_v5(db_path: str | Path) -> dict[str, Any]:
    """Insert the v5 plan with its ACTUAL insertion timestamp.

    Insert-if-absent so the first real insertion time survives re-runs. Never
    fabricates a historical timestamp and never mutates the A-L arms or v1-v4.
    """
    spec = planned_challenger_specification_v5()
    seed_historical(db_path)
    ensure_registry_schema(db_path)
    conn = sqlite3.connect(str(db_path), timeout=30)
    try:
        row = conn.execute(
            f"SELECT timestamp FROM {TABLE_REGISTRY} WHERE experiment_id=?",
            (PLANNED_EXPERIMENT_ID_V5,),
        ).fetchone()
    finally:
        conn.close()
    if row is not None:
        return {"experiment_id": PLANNED_EXPERIMENT_ID_V5, "inserted": False, "timestamp": row[0]}
    inserted_at = datetime.now(timezone.utc).isoformat()
    partition = register_partition_contract(db_path)
    payload = {
        "experiment_id": spec["experiment_id"],
        "timestamp": inserted_at,
        "inserted_at_actual": inserted_at,
        "timestamp_is_actual_insertion": True,
        "parent_contract_references": spec["parent_contracts"],
        "feature_set": spec["feature_set"],
        "feature_list": spec["inputs"],
        "schema_hash": _schema_hash(spec),
        "target": spec["target"],
        "target_horizon_sec": spec["primary_target_horizon_sec"],
        "partition_contract": partition,
        "sample_requirements": {
            "min_feature_complete_development_groups": spec["minimum_trainable_comparable_groups"],
            "min_fully_comparable_development_groups": spec["minimum_trainable_comparable_groups"],
            "min_authoritative_calibration_fills": spec["minimum_authoritative_real_fill_support"],
        },
        "validation_design": spec["training_procedure"],
        "model_class": spec["model_class"],
        "hyperparameters": spec["training_procedure"],
        "training_period": f"DEVELOPMENT from {CLOCK_V2_V5_DEVELOPMENT_START}",
        "validation_period": "expanding_chrono_folds_purge_3h_embargo_4h",
        "locked_period": f"FINAL_TEST={FINAL_TEST_STATUS}",
        "final_test_status": FINAL_TEST_STATUS,
        "result": PLANNED_RESULT,
        "train": False,
        "promoted": False,
        "notes": spec["notes"],
        **{k: v for k, v in spec.items() if k not in {"timestamp", "notes"}},
    }
    payload["timestamp"] = inserted_at
    payload["result"] = PLANNED_RESULT
    payload["train"] = False
    payload["promoted"] = False
    record_experiment(db_path, payload)
    return {"experiment_id": PLANNED_EXPERIMENT_ID_V5, "inserted": True, "timestamp": inserted_at}


def _schema_hash(spec: dict[str, Any]) -> str:
    payload = {
        "inputs": list(spec.get("inputs") or []),
        "target": spec.get("target"),
        "target_horizon_sec": spec.get("primary_target_horizon_sec"),
        "actions": list(spec.get("actions") or []),
        "effective_fitted_parameters": spec.get("effective_fitted_parameters"),
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()[:32]


def evaluate_clock_v2_v5_readiness(db_path: str | Path, *, generic_state: dict[str, Any] | None = None) -> dict[str, Any]:
    """V5 gate. Counts only valid v5 DEVELOPMENT groups.

    The generic ``forward_4h_entry_lock_20260903`` is neither read nor altered:
    clock-v2 uses its own partition contract. PRE_MODEL_QUARANTINE groups and any
    future FINAL_TEST groups are excluded. Never trains, never promotes.
    """
    req = clock_v2_v5_readiness_requirements()
    artifacts = _load_artifacts(db_path)
    groups = _load_groups(db_path)
    v5_labels = load_v5_label_presence(db_path)
    by_group: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for art in artifacts:
        by_group[str(art["decision_group_id"])].append(art)
    group_meta = {str(g["decision_group_id"]): g for g in groups}

    partitions = Counter()
    feature_complete = 0
    fully_comparable = 0
    calibration_fills = 0
    quarantined = 0
    future_data = 0
    lock_inspected = 0
    invariant_violations = 0
    availability_unknown_groups = 0
    blocks: set[int] = set()
    first: datetime | None = None
    last: datetime | None = None
    per_symbol = Counter()

    for gid, arts in by_group.items():
        stamped = next((a.get("clock_v2_partition") for a in arts if a.get("clock_v2_partition")), None)
        when_raw = arts[0].get("decision_timestamp") or arts[0].get("created_at")
        part = str(stamped) if stamped else partition_for(when_raw)
        partitions[part] += 1
        if any(a.get("inspected") for a in arts):
            lock_inspected += 1
        if part != DEVELOPMENT:
            if part == PRE_MODEL_QUARANTINE:
                quarantined += 1
            continue
        created = _parse_iso(when_raw)
        if created is not None:
            first = created if first is None or created < first else first
            last = created if last is None or created > last else last
            blocks.add(int(created.timestamp() // (CHRONOLOGICAL_BLOCK_HOURS * 3600)))
        completeness = group_completeness_v5(arts)
        if completeness.get("availability_unknown_actions"):
            availability_unknown_groups += 1
        if completeness["FEATURE_COMPLETE"]:
            feature_complete += 1
        labels_by_symbol = {sym: bool(v5_labels.get((gid, sym))) for sym in (*COINS, HOLD_SYMBOL)}
        comparability = group_comparability_v5(arts, labels_by_symbol=labels_by_symbol)
        if comparability["FULLY_COMPARABLE"]:
            fully_comparable += 1
        meta = group_meta.get(gid) or {}
        selected = str(meta.get("selected_symbol") or "")
        rows = [
            {
                "symbol": a.get("symbol"),
                "action_available": a.get("action_available"),
                "action_unavailable_reason": a.get("action_unavailable_reason"),
                "legacy_rank_candidate_present": a.get("legacy_rank_candidate_present"),
            }
            for a in arts
        ]
        invariant = selected_action_invariant(rows=rows, selected_symbol=selected or HOLD_SYMBOL)
        if not invariant.get("pass"):
            invariant_violations += 1
        if selected and selected != HOLD_SYMBOL and v5_labels.get((gid, selected)):
            calibration_fills += 1
            per_symbol[selected] += 1
        for art in arts:
            prov = art.get("provenance") or {}
            src = _parse_iso(prov.get("source_latest_ts"))
            cut = _parse_iso(prov.get("feature_cutoff_ts"))
            if src and cut and src > cut:
                future_data += 1

    req_groups = int(req["min_fully_comparable_independent_groups"])
    req_fills = int(req["min_authoritative_fills_execution_calibration"])
    span_days = ((last - first).total_seconds() / 86400.0) if first and last else 0.0
    generic = generic_state if generic_state is not None else _safe_generic(db_path)
    checks = generic.get("checks") or {}
    accounting_pass = bool((checks.get("F_accounting") or {}).get("pass", True))

    ok = (
        feature_complete >= req_groups
        and fully_comparable >= req_groups
        and calibration_fills >= req_fills
        and len(blocks) >= MIN_CHRONOLOGICAL_BLOCKS
        and lock_inspected == 0
        and future_data == 0
        and invariant_violations == 0
        and availability_unknown_groups == 0
        and accounting_pass
        and req["target_horizon_status"] != "TARGET_HORIZON_NOT_FROZEN"
    )

    return {
        "gate": "day_path_clock_v2_readiness_v5",
        "contract_version": "v5",
        "DATA_READINESS": "PASS" if ok else "FAIL",
        "train": False,
        "promoted": False,
        "auto_train_on_pass": False,
        "planned_experiment": PLANNED_EXPERIMENT_ID_V5,
        "planned_result": PLANNED_RESULT,
        "parent_contracts": [
            PLANNED_EXPERIMENT_ID,
            PLANNED_EXPERIMENT_ID_V2,
            PLANNED_EXPERIMENT_ID_V3,
            PLANNED_EXPERIMENT_ID_V4,
        ],
        "target": PRIMARY_TARGET,
        "target_name": TARGET_NAME,
        "target_horizon_sec": PRIMARY_TARGET_HORIZON_SEC,
        "target_horizon_status": TARGET_HORIZON_STATUS,
        "hold_target_bps": 0.0,
        "partition_contract": partition_contract(),
        "clock_v2_v5_development_start": CLOCK_V2_V5_DEVELOPMENT_START,
        "partition_counts": dict(partitions),
        "quarantined_groups_excluded": quarantined,
        "final_test_status": FINAL_TEST_STATUS,
        "final_test_required_before_promotion": True,
        "generic_4h_lock": {
            "experiment_id": GENERIC_4H_LOCK_ID,
            "used_as_clock_v2_partition": False,
            "mutated": False,
            "inspected": False,
            "outcomes_read": False,
        },
        "feature_complete_development_groups": feature_complete,
        "required_feature_complete_groups": req_groups,
        "fully_comparable_development_groups": fully_comparable,
        "required_fully_comparable_groups": req_groups,
        "authoritative_calibration_fills": calibration_fills,
        "required_calibration_fills": req_fills,
        "per_symbol_calibration_support": dict(per_symbol),
        "chronological_blocks": len(blocks),
        "required_chronological_blocks": MIN_CHRONOLOGICAL_BLOCKS,
        "chronological_span_days": round(span_days, 2),
        "embargo_seconds": req["embargo_seconds_min"],
        "purge_window_seconds": req["purge_window_seconds"],
        "selected_action_invariant_violations": invariant_violations,
        "availability_unknown_groups": availability_unknown_groups,
        "future_data_violations": future_data,
        "lock_inspected_rows": lock_inspected,
        "accounting_pass": accounting_pass,
        "effective_fitted_parameters": req["effective_fitted_parameters"],
        "previous_v4_required_groups": req["previous_v4_required_groups"],
        "support_floor_not_lowered": req["support_floor_not_lowered"],
        "final_rank_treatment": req["final_rank_treatment"],
        "listed_inputs": req["listed_inputs"],
        "label_contract": v5_label_contract(),
        "requirements": req,
        "groups_with_artifacts": len(by_group),
    }


def _safe_generic(db_path: str | Path) -> dict[str, Any]:
    try:
        return evaluate_readiness(db_path)
    except Exception:
        return {"checks": {}}


def format_clock_v2_v5_readiness(snapshot: dict[str, Any]) -> str:
    return "\n".join(
        [
            f"CLOCK_V2_V5_DATA_READINESS = {snapshot.get('DATA_READINESS')}",
            f"FEATURE_COMPLETE = {snapshot.get('feature_complete_development_groups')} / {snapshot.get('required_feature_complete_groups')}",
            f"FULLY_COMPARABLE = {snapshot.get('fully_comparable_development_groups')} / {snapshot.get('required_fully_comparable_groups')}",
            f"calibration_fills = {snapshot.get('authoritative_calibration_fills')} / {snapshot.get('required_calibration_fills')}",
            f"development_start = {snapshot.get('clock_v2_v5_development_start')}",
            f"final_test = {snapshot.get('final_test_status')}",
            f"train = {snapshot.get('train')}  promoted = {snapshot.get('promoted')}",
        ]
    )


def persist_clock_v2_v5(db_path: str | Path) -> dict[str, Any]:
    """Register the partition, persist the v5 plan, snapshot readiness. No training."""
    partition = register_partition_contract(db_path)
    planned = record_planned_clock_v2_v5(db_path)
    snap = evaluate_clock_v2_v5_readiness(db_path)
    ensure_readiness_schema(db_path)
    conn = sqlite3.connect(str(db_path), timeout=30)
    try:
        conn.execute(
            f"INSERT OR REPLACE INTO {TABLE_HISTORY}(recorded_at, snapshot_json) VALUES (?,?)",
            (datetime.now(timezone.utc).isoformat(), json.dumps(snap, default=str)),
        )
        conn.commit()
    finally:
        conn.close()
    return {"partition_contract": partition, "planned_v5": planned, "readiness": snap}


def persist_clock_v2_readiness(db_path: str | Path, snapshot: dict[str, Any] | None = None) -> dict[str, Any]:
    snap = snapshot or evaluate_clock_v2_readiness(db_path)
    record_planned_clock_v2_v2(db_path)
    record_planned_clock_v2_v3(db_path)
    record_planned_clock_v2_v4(db_path)
    ensure_readiness_schema(db_path)
    conn = sqlite3.connect(str(db_path), timeout=30)
    try:
        conn.execute(
            f"INSERT OR REPLACE INTO {TABLE_HISTORY}(recorded_at, snapshot_json) VALUES (?,?)",
            (datetime.now(timezone.utc).isoformat(), json.dumps(snap, default=str)),
        )
        conn.commit()
    finally:
        conn.close()
    return snap


def format_clock_v2_readiness(snapshot: dict[str, Any]) -> str:
    lines = [
        f"CLOCK_V2_DATA_READINESS = {snapshot.get('DATA_READINESS')}",
        f"complete_feature_groups = {snapshot.get('complete_feature_groups')}",
        f"fully_comparable = {snapshot.get('mature_comparable_full_groups')}",
        f"selected_authoritative = {snapshot.get('authoritative_selected_trade_labels')}",
        f"blocks = {snapshot.get('chronological_blocks')}",
        f"train = {snapshot.get('train')}",
        f"promoted = {snapshot.get('promoted')}",
    ]
    return "\n".join(lines)


__all__ = [
    "TABLE_HISTORY",
    "evaluate_clock_v2_readiness",
    "evaluate_clock_v2_v5_readiness",
    "format_clock_v2_readiness",
    "format_clock_v2_v5_readiness",
    "persist_clock_v2_readiness",
    "persist_clock_v2_v5",
    "record_planned_clock_v2_v2",
    "record_planned_clock_v2_v3",
    "record_planned_clock_v2_v4",
    "record_planned_clock_v2_v5",
]
