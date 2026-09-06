"""CLOCK-V2 v5 feature schema, completeness, comparability, readiness, registry.

Never trains, never promotes, never inspects the sealed 4H lock.
"""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timedelta, timezone

import pytest

from backend.services.day_clock_v2_labels import (
    INVALID_IMMATURE,
    INVALID_NOT_AVAILABLE,
    TABLE_V5_LABELS,
    TARGET_NAME,
    build_v5_label,
    hold_label,
    v5_label_contract,
)
from backend.services.day_clock_v2_partition import (
    CLOCK_V2_V5_DEVELOPMENT_START,
    DEVELOPMENT,
    PRE_MODEL_QUARANTINE,
)
from backend.services.day_experiment_registry import TABLE as TABLE_REGISTRY
from backend.services.day_path_clock_v2 import (
    CLOCK_V2_EFFECTIVE_FITTED_PARAMETERS,
    CLOCK_V2_V5_EFFECTIVE_FITTED_PARAMETERS,
    CLOCK_V2_V5_REQUIRED_CALIBRATION_FILLS,
    CLOCK_V2_V5_REQUIRED_GROUPS,
    EVENTS_PER_PARAMETER,
    PLANNED_EXPERIMENT_ID_V4,
    PLANNED_EXPERIMENT_ID_V5,
    PRIMARY_TARGET_HORIZON_SEC,
    REQUIRED_CLOCK_V2_FIELDS,
    REQUIRED_CLOCK_V2_FIELDS_V5,
    clock_v2_v4_readiness_requirements,
    clock_v2_v5_feature_schema,
    clock_v2_v5_readiness_requirements,
    planned_challenger_specification_v4,
    planned_challenger_specification_v5,
)
from backend.services.day_path_clock_v2_capture import (
    group_comparability_v5,
    group_completeness_v5,
)
from backend.services.day_path_clock_v2_readiness import (
    evaluate_clock_v2_v5_readiness,
    persist_clock_v2_v5,
    record_planned_clock_v2_v5,
)

COINS = ("BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT")


def _feats(**over):
    base = dict.fromkeys(REQUIRED_CLOCK_V2_FIELDS_V5, 0.1)
    base["symbol"] = "ETHUSDT"
    base["commission_rt_bps"] = 8.0
    base["expected_slippage_bps"] = 2.0
    base.update(over)
    return base


def _artifact(symbol, *, available=True, reason=None, complete=True, partition=DEVELOPMENT):
    feats = _feats(symbol=symbol)
    if not complete:
        feats["ret_15m"] = None
    if symbol == "HOLD":
        feats = {
            "symbol": "HOLD",
            "legacy_path_ev": 0.0,
            "final_rank_score": 0.0,
            "spread_bps": 0.0,
            "estimated_all_in_cost_bps": 0.0,
        }
    return {
        "symbol": symbol,
        "features": feats,
        "eligible": True,
        "action_available": available,
        "action_unavailable_reason": reason,
        "clock_v2_partition": partition,
        "decision_timestamp": "2026-09-06T01:00:00+00:00",
        "created_at": "2026-09-06T01:00:00+00:00",
    }


def _group(**kw):
    return [_artifact(s, **kw) for s in COINS] + [_artifact("HOLD")]


# --- v5 feature schema / final-rank treatment ---


def test_v5_schema_removes_final_rank_score():
    assert "final_rank_score" in REQUIRED_CLOCK_V2_FIELDS
    assert "final_rank_score" not in REQUIRED_CLOCK_V2_FIELDS_V5
    schema = clock_v2_v5_feature_schema()
    assert schema["removed_from_v4"] == ["final_rank_score"]
    assert schema["ignored_obsolete_feature_json_fields"] == ["final_rank_score"]
    assert "final_rank_score" not in schema["inputs"]
    from backend.services.day_path_clock_v2 import v5_listed_features_from_blob

    blob = {"p_buy": 0.4, "legacy_path_ev": 0.001, "final_rank_score": 0.001, "ret_5m": 0.01}
    exported = v5_listed_features_from_blob(blob)
    assert "final_rank_score" not in exported
    assert exported["legacy_path_ev"] == 0.001


def test_v5_keeps_all_action_definable_inputs_only():
    schema = clock_v2_v5_feature_schema()
    assert "p_buy" in schema["inputs"]
    assert "legacy_path_ev" in schema["inputs"]
    assert "p_buy" in schema["all_action_inputs_verified"]


def test_v4_spec_is_not_mutated_by_v5():
    v4 = planned_challenger_specification_v4()
    assert "final_rank_score" in v4["inputs"]
    assert v4["experiment_id"] == PLANNED_EXPERIMENT_ID_V4
    assert clock_v2_v4_readiness_requirements()["numeric_features_counted"] == 14


def test_v5_parameter_count_and_support_floor_not_lowered():
    assert CLOCK_V2_V5_EFFECTIVE_FITTED_PARAMETERS == 18
    assert CLOCK_V2_EFFECTIVE_FITTED_PARAMETERS == 19
    v4_floor = CLOCK_V2_EFFECTIVE_FITTED_PARAMETERS * EVENTS_PER_PARAMETER
    expected = max(v4_floor, CLOCK_V2_V5_EFFECTIVE_FITTED_PARAMETERS * EVENTS_PER_PARAMETER)
    assert expected == CLOCK_V2_V5_REQUIRED_GROUPS
    assert CLOCK_V2_V5_REQUIRED_GROUPS == 190
    assert v4_floor <= CLOCK_V2_V5_REQUIRED_GROUPS
    assert CLOCK_V2_V5_REQUIRED_CALIBRATION_FILLS == 50


def test_v5_readiness_requirements_are_frozen_and_block_auto_training():
    req = clock_v2_v5_readiness_requirements()
    assert req["min_feature_complete_groups"] == 190
    assert req["min_fully_comparable_independent_groups"] == 190
    assert req["min_authoritative_fills_execution_calibration"] == 50
    assert req["embargo_seconds_min"] >= 4 * 3600
    assert req["purge_window_seconds"] == PRIMARY_TARGET_HORIZON_SEC
    assert req["hold_target_bps"] == 0.0
    assert req["auto_train_forbidden"] is True
    assert req["train_on_readiness_pass"] is False
    assert req["require_final_test_contract_before_promotion"] is True
    assert req["support_floor_not_lowered"] is True
    assert req["counted_partition"] == DEVELOPMENT


def test_v5_spec_is_planned_not_run():
    spec = planned_challenger_specification_v5()
    assert spec["experiment_id"] == PLANNED_EXPERIMENT_ID_V5
    assert spec["result"] == "PLANNED_NOT_RUN"
    assert spec["train"] is False
    assert spec["promoted"] is False
    assert spec["live_gate"] is False
    assert PLANNED_EXPERIMENT_ID_V4 in spec["parent_contracts"]
    assert spec["primary_target_horizon_sec"] == PRIMARY_TARGET_HORIZON_SEC == 10800


# --- feature completeness over real actions ---


def test_feature_complete_requires_every_available_action():
    out = group_completeness_v5(_group())
    assert out["FEATURE_COMPLETE"] is True
    assert out["available_action_total"] == 4


def test_legacy_unscored_action_cannot_disappear_from_completeness():
    """A legacy-unscored but production-available coin must still carry features."""
    arts = _group()
    arts[1] = _artifact("ETHUSDT", complete=False)
    arts[1]["eligible"] = False
    out = group_completeness_v5(arts)
    assert out["FEATURE_COMPLETE"] is False
    assert "ETHUSDT" in out["available_actions"]


def test_genuinely_unavailable_action_may_be_excluded():
    arts = _group()
    arts[1] = _artifact("ETHUSDT", available=False, reason="DUPLICATE_SAME_SYMBOL", complete=False)
    out = group_completeness_v5(arts)
    assert out["FEATURE_COMPLETE"] is True
    assert out["unavailable_actions"] == ["ETHUSDT"]
    assert out["available_action_total"] == 3


def test_unknown_availability_blocks_completeness():
    arts = _group()
    arts[1] = _artifact("ETHUSDT", available=None, reason="AVAILABILITY_UNKNOWN_NO_PATH_TELEMETRY")
    out = group_completeness_v5(arts)
    assert out["FEATURE_COMPLETE"] is False
    assert out["availability_unknown_actions"] == ["ETHUSDT"]


def test_hold_is_always_an_available_action():
    assert "HOLD" in group_completeness_v5(_group())["available_actions"]


# --- comparability ---


def test_fully_comparable_requires_labels_for_every_available_action():
    arts = _group()
    labels = dict.fromkeys((*COINS, "HOLD"), True)
    assert group_comparability_v5(arts, labels_by_symbol=labels)["FULLY_COMPARABLE"] is True
    labels["SOLUSDT"] = False
    out = group_comparability_v5(arts, labels_by_symbol=labels)
    assert out["FULLY_COMPARABLE"] is False
    assert out["missing_label_actions"] == ["SOLUSDT"]


def test_comparability_requires_uniform_cost_methodology():
    arts = _group()
    arts[2]["features"]["expected_slippage_bps"] = None
    labels = dict.fromkeys((*COINS, "HOLD"), True)
    out = group_comparability_v5(arts, labels_by_symbol=labels)
    assert out["methodology_uniform"] is False
    assert out["FULLY_COMPARABLE"] is False


def test_production_exit_never_substitutes_for_the_fixed_horizon_target():
    out = group_comparability_v5(_group(), labels_by_symbol=dict.fromkeys((*COINS, "HOLD"), True))
    assert out["production_exit_substituted_for_target"] is False
    assert out["hold_target_bps"] == 0.0
    assert v5_label_contract()["production_exit_may_replace_target"] is False


# --- 3h label contract ---


def test_label_contract_is_3h_and_separate_from_generic_4h_labels():
    contract = v5_label_contract()
    assert contract["target_horizon_sec"] == 10800
    assert contract["target_name"] == TARGET_NAME == "executable_net_bps_3h"
    assert contract["table"] == TABLE_V5_LABELS != "day_decision_outcome_labels"
    assert contract["separate_from_generic_4h_labels"] is True
    assert contract["generic_4h_lock_read"] is False
    assert contract["identical_methodology_for_every_action"] is True


def test_hold_label_is_exactly_zero():
    lab = hold_label(decision_group_id="g", decision_ts="2026-09-06T01:00:00+00:00")
    assert lab["executable_net_bps_3h"] == 0.0
    assert lab["label_valid"] is True


def test_label_is_immature_before_the_horizon(tmp_path):
    now = datetime(2026, 9, 6, 2, 0, tzinfo=timezone.utc)
    lab = build_v5_label(
        db_path=tmp_path / "x.db",
        decision_group_id="g",
        symbol="ETHUSDT",
        decision_ts=(now - timedelta(hours=1)).isoformat(),
        action_available=True,
        entry_px=100.0,
        now=now,
    )
    assert lab["label_valid"] is False
    assert lab["label_invalid_reason"] == INVALID_IMMATURE


def test_unavailable_action_is_not_labeled(tmp_path):
    lab = build_v5_label(
        db_path=tmp_path / "x.db",
        decision_group_id="g",
        symbol="ETHUSDT",
        decision_ts="2026-09-06T01:00:00+00:00",
        action_available=False,
        entry_px=100.0,
        now=datetime(2026, 9, 7, tzinfo=timezone.utc),
    )
    assert lab["label_invalid_reason"] == INVALID_NOT_AVAILABLE


# --- readiness uses its own partition ---


def _seed_artifacts(db, groups):
    from backend.services.day_path_clock_v2_capture import TABLE_ARTIFACT, ensure_artifact_schema

    ensure_artifact_schema(db)
    conn = sqlite3.connect(str(db))
    try:
        for gid, arts in groups.items():
            for art in arts:
                conn.execute(
                    f"""INSERT OR REPLACE INTO {TABLE_ARTIFACT}(
                        decision_group_id, symbol, created_at, decision_timestamp,
                        feature_schema_version, feature_contract_version, eligible,
                        feature_json, clock_v2_partition, action_available, lock_window, inspected
                    ) VALUES (?,?,?,?,?,?,?,?,?,?,0,0)""",
                    (
                        gid,
                        art["symbol"],
                        art["created_at"],
                        art["decision_timestamp"],
                        "day_path_clock_v2",
                        "day_path_clock_v2_capture_1",
                        1,
                        json.dumps(art["features"]),
                        art["clock_v2_partition"],
                        None if art["action_available"] is None else int(bool(art["action_available"])),
                    ),
                )
        conn.commit()
    finally:
        conn.close()


def test_readiness_excludes_quarantined_groups(tmp_path):
    db = tmp_path / "r.db"
    dev = _group()
    quar = [dict(a, clock_v2_partition=PRE_MODEL_QUARANTINE, decision_timestamp="2026-09-05T10:00:00+00:00") for a in _group()]
    _seed_artifacts(db, {"g_dev": dev, "g_quar": quar})
    snap = evaluate_clock_v2_v5_readiness(db)
    assert snap["partition_counts"].get(PRE_MODEL_QUARANTINE) == 1
    assert snap["quarantined_groups_excluded"] == 1
    assert snap["feature_complete_development_groups"] == 1


def test_readiness_reports_fail_below_floor_and_never_trains(tmp_path):
    db = tmp_path / "r2.db"
    _seed_artifacts(db, {"g1": _group()})
    snap = evaluate_clock_v2_v5_readiness(db)
    assert snap["DATA_READINESS"] == "FAIL"
    assert snap["train"] is False
    assert snap["promoted"] is False
    assert snap["auto_train_on_pass"] is False
    assert snap["required_feature_complete_groups"] == 190
    assert snap["required_calibration_fills"] == 50


def test_v5_readiness_does_not_query_4h_outcome_labels(tmp_path, monkeypatch):
    db = tmp_path / "no4h.db"
    _seed_artifacts(db, {"g1": _group()})
    queries: list[str] = []
    real_connect = sqlite3.connect

    def spy(*args, **kwargs):
        conn = real_connect(*args, **kwargs)
        conn.set_trace_callback(lambda sql: queries.append(str(sql)))
        return conn

    monkeypatch.setattr(sqlite3, "connect", spy)
    import backend.services.day_clock_v2_calibration as cal
    import backend.services.day_clock_v2_labels as labs
    import backend.services.day_model_readiness as model
    import backend.services.day_path_clock_v2_readiness as ready

    monkeypatch.setattr(ready.sqlite3, "connect", spy)
    monkeypatch.setattr(labs.sqlite3, "connect", spy)
    monkeypatch.setattr(cal.sqlite3, "connect", spy)
    monkeypatch.setattr(model.sqlite3, "connect", spy)
    snap = evaluate_clock_v2_v5_readiness(db)
    joined = "\n".join(queries)
    assert "day_decision_outcome_labels" not in joined
    assert snap["generic_4h_lock"]["outcomes_read"] is False
    assert snap["generic_4h_lock"]["inspected"] is False


def test_readiness_leaves_generic_4h_lock_untouched(tmp_path):
    db = tmp_path / "r3.db"
    _seed_artifacts(db, {"g1": _group()})
    snap = evaluate_clock_v2_v5_readiness(db)
    lock = snap["generic_4h_lock"]
    assert lock["used_as_clock_v2_partition"] is False
    assert lock["mutated"] is False
    assert lock["inspected"] is False
    assert lock["outcomes_read"] is False
    assert snap["final_test_status"] == "NOT_YET_CREATED"
    assert snap["clock_v2_v5_development_start"] == CLOCK_V2_V5_DEVELOPMENT_START


def test_readiness_never_reports_lock_inspection(tmp_path):
    db = tmp_path / "r4.db"
    _seed_artifacts(db, {"g1": _group()})
    assert evaluate_clock_v2_v5_readiness(db)["lock_inspected_rows"] == 0


# --- experiment registry persistence ---


def test_planned_v5_is_persisted_with_actual_timestamp(tmp_path):
    db = tmp_path / "e.db"
    before = datetime.now(timezone.utc)
    out = record_planned_clock_v2_v5(db)
    assert out["inserted"] is True
    stamped = datetime.fromisoformat(out["timestamp"])
    assert stamped >= before
    conn = sqlite3.connect(str(db))
    try:
        row = conn.execute(
            f"SELECT timestamp, result, promoted, meta_json FROM {TABLE_REGISTRY} WHERE experiment_id=?",
            (PLANNED_EXPERIMENT_ID_V5,),
        ).fetchone()
    finally:
        conn.close()
    assert row is not None
    assert row[1] == "PLANNED_NOT_RUN"
    assert row[2] == 0
    meta = json.loads(row[3])
    assert meta["train"] is False
    assert meta["promoted"] is False
    assert meta["timestamp_is_actual_insertion"] is True
    assert meta["schema_hash"]
    assert meta["feature_list"] == list(REQUIRED_CLOCK_V2_FIELDS_V5)
    assert meta["target_horizon_sec"] == 10800
    assert meta["partition_contract"]["clock_v2_v5_development_start"] == CLOCK_V2_V5_DEVELOPMENT_START
    assert meta["sample_requirements"]["min_fully_comparable_development_groups"] == 190
    assert meta["final_test_status"] == "NOT_YET_CREATED"
    assert meta["validation_design"]["embargo_seconds_min"] >= 4 * 3600


def test_planned_v5_insertion_timestamp_is_not_rewritten(tmp_path):
    db = tmp_path / "e2.db"
    first = record_planned_clock_v2_v5(db)
    second = record_planned_clock_v2_v5(db)
    assert second["inserted"] is False
    assert second["timestamp"] == first["timestamp"]


def test_historical_experiment_arms_are_not_mutated(tmp_path):
    db = tmp_path / "e3.db"
    record_planned_clock_v2_v5(db)
    conn = sqlite3.connect(str(db))
    try:
        rows = dict(conn.execute(f"SELECT experiment_id, timestamp FROM {TABLE_REGISTRY}").fetchall())
        promoted = conn.execute(f"SELECT COUNT(*) FROM {TABLE_REGISTRY} WHERE promoted=1").fetchone()[0]
    finally:
        conn.close()
    assert rows["A"] == "2026-09-01"
    assert rows["L"] == "2026-09-03"
    assert rows[PLANNED_EXPERIMENT_ID_V4] == "2026-09-04"
    assert rows["M_clock_v2_planned_20260905"] == "2026-09-05"
    assert promoted == 0


def test_persist_cycle_registers_partition_and_snapshots(tmp_path):
    db = tmp_path / "e4.db"
    out = persist_clock_v2_v5(db)
    assert out["partition_contract"]["clock_v2_v5_development_start"] == CLOCK_V2_V5_DEVELOPMENT_START
    assert out["planned_v5"]["inserted"] is True
    assert out["readiness"]["train"] is False
    assert out["readiness"]["promoted"] is False


def test_research_cycle_is_fail_open_and_never_trains(tmp_path):
    from backend.services.day_clock_v2_research_cycle import run_clock_v2_v5_cycle

    out = run_clock_v2_v5_cycle(tmp_path / "cycle.db")
    assert out["errors"] == 0
    assert out["readiness"] in {"PASS", "FAIL"}
    assert out["planned_inserted"] is True


def test_research_cycle_tolerates_bad_path():
    from backend.services.day_clock_v2_research_cycle import run_clock_v2_v5_cycle

    assert run_clock_v2_v5_cycle("")["labels_written"] == 0


def test_label_table_exists_before_the_first_label_matures(tmp_path):
    """The v5 label authority must be present as soon as the contract is live."""
    from backend.services.day_clock_v2_labels import run_v5_label_batch

    db = tmp_path / "labels.db"
    out = run_v5_label_batch(db)
    assert out["labels_written"] == 0
    conn = sqlite3.connect(str(db))
    try:
        found = conn.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (TABLE_V5_LABELS,)).fetchone()
    finally:
        conn.close()
    assert found is not None


# --- retention protections ---


def test_clock_v2_authority_tables_are_protected_from_retention():
    from backend.services.sqlite_large_table_retention import PROTECTED_TABLES, RETENTION_POLICIES

    for table in (
        "day_experiment_registry",
        "day_forward_lock_registry",
        "day_path_clock_v2_candidate_artifact",
        "day_clock_v2_partition_registry",
        "day_clock_v2_outcome_labels",
        "day_clock_v2_outcome_labels_history",
    ):
        assert table in PROTECTED_TABLES
    policy_tables = {p.table for p in RETENTION_POLICIES}
    for table in ("paper_trades",):
        assert table in policy_tables  # orders/fills keep a long window, never dropped outright
    assert "day_clock_v2_outcome_labels" not in policy_tables


def test_new_retention_policies_are_documented_and_bounded():
    from backend.services.sqlite_large_table_retention import (
        RETENTION_JUSTIFICATION,
        RETENTION_POLICIES,
    )

    by_table = {p.table: p for p in RETENTION_POLICIES}
    micro = by_table["microstructure_feature_snapshots"]
    assert micro.keep_days == 14
    assert micro.ts_column == "ts_utc"
    assert micro.cutoff_format == "epoch_seconds"
    assert micro.keep_days * 86400 > 4 * 3600  # exceeds the longest research horizon
    assert "microstructure_feature_snapshots" in RETENTION_JUSTIFICATION
    assert by_table["scalp_shadow_rejects"].keep_days == 30


def test_epoch_seconds_cutoff_is_numeric():
    from backend.services.sqlite_large_table_retention import _cutoff_value

    policy = next(p for p in __import__("backend.services.sqlite_large_table_retention", fromlist=["x"]).RETENTION_POLICIES if p.cutoff_format == "epoch_seconds")
    assert isinstance(_cutoff_value(policy), float)


def test_retention_dry_run_reports_without_deleting(tmp_path):
    from backend.services.sqlite_large_table_retention import retention_dry_run

    db = tmp_path / "ret.db"
    conn = sqlite3.connect(str(db))
    try:
        conn.execute("CREATE TABLE microstructure_feature_snapshots (id INTEGER PRIMARY KEY, ts_utc REAL)")
        old = (datetime.now(timezone.utc) - timedelta(days=40)).timestamp()
        new = datetime.now(timezone.utc).timestamp()
        conn.executemany("INSERT INTO microstructure_feature_snapshots(ts_utc) VALUES (?)", [(old,)] * 5 + [(new,)] * 3)
        conn.commit()
        report = retention_dry_run(db)
        remaining = conn.execute("SELECT COUNT(*) FROM microstructure_feature_snapshots").fetchone()[0]
    finally:
        conn.close()
    entry = report["tables"]["microstructure_feature_snapshots"]
    assert report["dry_run"] is True
    assert entry["rows_to_delete"] == 5
    assert entry["status"] == "would_delete"
    assert entry["justification"]
    assert remaining == 8


def test_hourly_logrotate_runner_exists_so_the_size_rule_is_not_cosmetic():
    """logrotate.timer is daily; a 20 MB cap needs an hourly check to mean anything."""
    from pathlib import Path

    text = Path(__file__).resolve().parents[1].joinpath("deploy/cron.hourly-mystic-logrotate").read_text()
    assert "/etc/logrotate.d/mystic" in text
    assert "--state" in text


def test_logrotate_config_is_bounded_and_uses_copytruncate():
    from pathlib import Path

    text = Path(__file__).resolve().parents[1].joinpath("deploy/logrotate-mystic").read_text()
    assert "copytruncate" in text
    assert "compress" in text
    assert "size 20M" in text
    assert "rotate 5" in text
    assert "/home/mystic/mystic/logs/*.log" in text
    # copytruncate copies to a file nothing writes to, so compress on the same pass.
    assert "delaycompress" not in text


@pytest.mark.parametrize("field", ["train", "promoted"])
def test_no_v5_surface_ever_enables_training(field):
    assert planned_challenger_specification_v5()[field] is False
