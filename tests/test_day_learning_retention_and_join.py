"""Retention safety and the open-position label-join rule.

Two independent guarantees are pinned here:

* an authoritative production label is required only once a fill has actually closed, so a
  trade still holding an open position is excluded from the denominator rather than counted
  as a missing label;
* retention may never age out sealed research authority, and may never delete learning rows
  that a forward lock still needs in order to rebuild its dataset.
"""

from __future__ import annotations

import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from backend.services.day_model_readiness import (
    OPEN_POSITION_STATUSES,
    check_production_label_integrity,
    open_fill_trade_ids,
    readiness_progress,
)
from backend.services.sqlite_large_table_retention import (
    LOCK_DEPENDENT_TABLES,
    PROTECTED_TABLES,
    RETENTION_POLICIES,
    effective_cutoff,
    lock_floor,
    retention_dry_run,
    run_large_table_retention,
    storage_report,
)

NOW = datetime(2026, 9, 4, 2, 0, tzinfo=timezone.utc)
ACCOUNTING_AUTHORITY = (
    "portfolio_engine_ledger",
    "portfolio_engine_positions",
    "portfolio_engine_orders",
    "portfolio_engine_audit",
    "position_close_ledger",
)


def _iso(dt: datetime) -> str:
    return dt.isoformat()


def _state(*, status: str, created: datetime, provenance: str) -> dict:
    """Minimal readiness state for one traded group holding one position row."""
    return {
        "groups": {
            "g1": {
                "decision_group_id": "g1",
                "created_at": _iso(created),
                "selected_action": "BUY_ETHUSDT",
                "selected_symbol": "ETHUSDT",
                "execute_authorized": 1,
                "fill_trade_id": "mystic_ETH/USDT_1",
            }
        },
        "labels": {"g1": {"ETHUSDT": {"provenance": provenance}}},
        "book": [
            {
                "symbol": "ETH/USDT",
                "quantity": 0.0257,
                "entry_price": 2515.42,
                "status": status,
                "trade_id": "mystic_ETH/USDT_1",
            }
        ],
    }


# ------------------------------------------------------------------------------------
# open-position join rule
# ------------------------------------------------------------------------------------
def test_open_position_is_excluded_rather_than_counted_as_a_missing_label():
    """The exact production case: matured by clock, still ACTIVE, so not yet labelable."""
    state = _state(status="ACTIVE", created=NOW - timedelta(hours=6), provenance="reconstructed")
    result = check_production_label_integrity(state, now=NOW.timestamp())

    assert result["open_trades_excluded"] == 1
    assert result["open_trade_groups"] == ["g1"]
    assert result["matured_traded_groups"] == 0
    assert result["unjoined_groups"] == []


def test_a_closed_fill_still_requires_an_authoritative_label():
    """Excluding open trades must not become a loophole for genuinely missing labels."""
    state = _state(status="ACTIVE", created=NOW - timedelta(hours=6), provenance="reconstructed")
    state["book"] = []  # position closed, no book row remains

    result = check_production_label_integrity(state, now=NOW.timestamp())

    assert result["open_trades_excluded"] == 0
    assert result["matured_traded_groups"] == 1
    assert result["authoritative_joins"] == 0
    assert result["unjoined_groups"] == ["g1"]
    assert result["pass"] is False


def test_closed_fill_with_authoritative_label_reaches_full_coverage():
    state = _state(status="ACTIVE", created=NOW - timedelta(hours=6), provenance="authoritative")
    state["book"] = []

    result = check_production_label_integrity(state, now=NOW.timestamp())

    assert result["join_rate"] == 1.0
    assert result["pass"] is True


def test_dust_tail_stays_in_the_denominator():
    """A dust tail means the paying tranche already sold; the exit is real and labelable."""
    state = _state(status="DUST_PENDING", created=NOW - timedelta(hours=6), provenance="authoritative")

    result = check_production_label_integrity(state, now=NOW.timestamp())

    assert "DUST_PENDING" not in OPEN_POSITION_STATUSES
    assert result["open_trades_excluded"] == 0
    assert result["matured_traded_groups"] == 1
    assert result["join_rate"] == 1.0


def test_open_fill_trade_ids_ignores_blank_and_closed_rows():
    state = _state(status="ACTIVE", created=NOW, provenance="reconstructed")
    state["book"].append({"symbol": "XRP/USDT", "status": "DUST_PENDING", "trade_id": "dust_1"})
    state["book"].append({"symbol": "SOL/USDT", "status": "ACTIVE", "trade_id": ""})

    assert open_fill_trade_ids(state) == {"mystic_ETH/USDT_1"}


def test_unmatured_open_trade_is_not_double_counted():
    state = _state(status="ACTIVE", created=NOW - timedelta(minutes=30), provenance="reconstructed")

    result = check_production_label_integrity(state, now=NOW.timestamp())

    assert result["matured_traded_groups"] == 0
    assert result["open_trades_excluded"] == 0


# ------------------------------------------------------------------------------------
# retention policy shape
# ------------------------------------------------------------------------------------
def test_sealed_research_authority_is_never_on_a_deletion_timer():
    policy_tables = {p.table for p in RETENTION_POLICIES}
    assert set(PROTECTED_TABLES) == {
        "day_experiment_registry",
        "day_forward_lock_registry",
        "day_path_clock_feature_snapshots",
        "day_path_clock_readiness_history",
        "day_path_clock_v2_candidate_artifact",
        "day_path_clock_v2_readiness_history",
        "day_clock_v2_partition_registry",
        "day_clock_v2_outcome_labels",
    }
    assert not (policy_tables & PROTECTED_TABLES)


def test_retention_never_targets_accounting_or_fill_authority():
    policy_tables = {p.table for p in RETENTION_POLICIES}
    for table in ACCOUNTING_AUTHORITY:
        assert table not in policy_tables, f"{table} must never be aged out"


def test_learning_tables_keep_ninety_days():
    """Part 13: the window is frozen. A shrink must be a deliberate, reviewed change."""
    by_table = {p.table: p for p in RETENTION_POLICIES}
    for table in LOCK_DEPENDENT_TABLES:
        assert by_table[table].keep_days == 90


# ------------------------------------------------------------------------------------
# retention behaviour against a real database
# ------------------------------------------------------------------------------------
def _build_db(path: Path, *, lock_cutoff: str | None, oldest_days: int) -> None:
    conn = sqlite3.connect(path)
    conn.execute("CREATE TABLE day_decision_group_records (decision_group_id TEXT, created_at TEXT)")
    conn.execute("CREATE TABLE day_decision_outcome_labels (decision_group_id TEXT, created_at TEXT)")
    conn.execute("CREATE TABLE day_experiment_registry (experiment_id TEXT, timestamp TEXT)")
    conn.execute("CREATE TABLE day_forward_lock_registry (experiment_id TEXT, created_at TEXT, dataset_cutoff TEXT, training_start TEXT, locked_test_start TEXT)")
    for offset in (oldest_days, 200, 5):
        stamp = _iso(NOW - timedelta(days=offset))
        conn.execute("INSERT INTO day_decision_group_records VALUES (?,?)", (f"g{offset}", stamp))
        conn.execute("INSERT INTO day_decision_outcome_labels VALUES (?,?)", (f"g{offset}", stamp))
    conn.execute("INSERT INTO day_experiment_registry VALUES (?,?)", ("A", _iso(NOW - timedelta(days=400))))
    if lock_cutoff:
        conn.execute(
            "INSERT INTO day_forward_lock_registry VALUES (?,?,?,?,?)",
            ("lock_1", _iso(NOW - timedelta(days=300)), lock_cutoff, lock_cutoff, None),
        )
    conn.commit()
    conn.close()


def test_dry_run_reports_without_deleting(tmp_path):
    db = tmp_path / "dry.db"
    _build_db(db, lock_cutoff=None, oldest_days=365)

    before = sqlite3.connect(db).execute("SELECT COUNT(*) FROM day_decision_group_records").fetchone()[0]
    report = retention_dry_run(db)
    after = sqlite3.connect(db).execute("SELECT COUNT(*) FROM day_decision_group_records").fetchone()[0]

    assert report["dry_run"] is True
    assert before == after, "dry run must not delete anything"
    groups = report["tables"]["day_decision_group_records"]
    assert groups["rows_to_delete"] == 2  # 365d and 200d old, 5d retained
    assert groups["oldest_row_retained"] is not None
    assert groups["newest_row_to_delete"] is not None
    assert groups["estimated_bytes_reclaimed"] >= 0


def test_dry_run_marks_protected_tables_and_never_plans_their_deletion(tmp_path):
    db = tmp_path / "prot.db"
    _build_db(db, lock_cutoff=None, oldest_days=365)

    report = retention_dry_run(db)

    for table in PROTECTED_TABLES:
        entry = report["tables"].get(table)
        if entry is not None:
            assert entry["status"] == "protected"
            assert entry["rows_to_delete"] == 0


def test_lock_floor_clamps_the_cutoff_back(tmp_path):
    """A lock reaching further back than 90 days must hold retention at its own cutoff."""
    db = tmp_path / "floor.db"
    floor_iso = _iso(NOW - timedelta(days=300))
    _build_db(db, lock_cutoff=floor_iso, oldest_days=365)

    conn = sqlite3.connect(db)
    try:
        assert lock_floor(conn) == floor_iso
        policy = next(p for p in RETENTION_POLICIES if p.table == "day_decision_group_records")
        cutoff, floor = effective_cutoff(conn, policy)
        assert floor == floor_iso
        assert cutoff == floor_iso, "cutoff must be clamped back to the lock floor"
    finally:
        conn.close()


def test_lock_protected_rows_survive_a_real_retention_run(tmp_path):
    db = tmp_path / "enforce.db"
    floor_iso = _iso(NOW - timedelta(days=300))
    _build_db(db, lock_cutoff=floor_iso, oldest_days=365)

    run_large_table_retention(db)

    conn = sqlite3.connect(db)
    try:
        remaining = [r[0] for r in conn.execute("SELECT created_at FROM day_decision_group_records")]
        assert all(r >= floor_iso for r in remaining), "no row the lock depends on may be deleted"
        assert any(r == _iso(NOW - timedelta(days=200)) for r in remaining), "200d row is inside the lock window"
        assert conn.execute("SELECT COUNT(*) FROM day_experiment_registry").fetchone()[0] == 1
        assert conn.execute("SELECT COUNT(*) FROM day_forward_lock_registry").fetchone()[0] == 1
    finally:
        conn.close()


def test_without_a_lock_the_plain_ninety_day_cutoff_applies(tmp_path):
    db = tmp_path / "nolock.db"
    _build_db(db, lock_cutoff=None, oldest_days=365)

    run_large_table_retention(db)

    conn = sqlite3.connect(db)
    try:
        remaining = [r[0] for r in conn.execute("SELECT created_at FROM day_decision_group_records")]
        assert remaining == [_iso(NOW - timedelta(days=5))]
    finally:
        conn.close()


def test_retention_is_idempotent(tmp_path):
    db = tmp_path / "idem.db"
    _build_db(db, lock_cutoff=None, oldest_days=365)

    run_large_table_retention(db)
    first = sqlite3.connect(db).execute("SELECT COUNT(*) FROM day_decision_group_records").fetchone()[0]
    second_run = run_large_table_retention(db)
    second = sqlite3.connect(db).execute("SELECT COUNT(*) FROM day_decision_group_records").fetchone()[0]

    assert first == second
    assert second_run["tables"]["day_decision_group_records"]["deleted"] == 0


def test_cutoff_is_utc_and_deterministic(tmp_path):
    db = tmp_path / "utc.db"
    _build_db(db, lock_cutoff=None, oldest_days=365)

    report = retention_dry_run(db)
    cutoff = report["tables"]["day_decision_group_records"]["cutoff"]

    parsed = datetime.fromisoformat(cutoff)
    assert parsed.tzinfo is not None
    assert parsed.utcoffset() == timedelta(0), "cutoff must be UTC, not local time"


def test_dry_run_on_a_missing_database_is_not_an_exception(tmp_path):
    report = retention_dry_run(tmp_path / "nope.db")
    assert "error" in report
    assert report["tables"] == {}


# ------------------------------------------------------------------------------------
# storage monitoring
# ------------------------------------------------------------------------------------
def test_storage_report_bands_are_observability_only(tmp_path):
    db = tmp_path / "size.db"
    _build_db(db, lock_cutoff=None, oldest_days=365)

    report = storage_report(db)

    assert report["severity"] in {"OK", "WARNING", "CRITICAL"}
    assert "not a trading gate" in report["severity_note"]
    for horizon in (30, 60, 90):
        assert f"projection_{horizon}d_gib" in report
    assert report["db_bytes"] > 0


def test_storage_report_on_missing_database_is_not_an_exception(tmp_path):
    assert "error" in storage_report(tmp_path / "gone.db")


# ------------------------------------------------------------------------------------
# readiness progress
# ------------------------------------------------------------------------------------
def test_progress_reports_counts_not_a_countdown_date():
    report = {
        "ready": False,
        "reasons_not_ready": ["G_forward_span"],
        "checks": {
            "G_forward_span": {
                "pass": False,
                "mature_authoritative_trade_labels": 2,
                "required_mature_trade_labels": 140,
                "chronological_blocks": 2,
                "required_chronological_blocks": 5,
                "effective_window_start": "2026-09-03T22:02:47+00:00",
                "lock_cutoff": "2026-09-03T00:00:00+00:00",
            },
        },
    }

    progress = readiness_progress(report)

    assert progress["mature_events"] == "2 / 140"
    assert progress["chronological_blocks"] == "2 / 5"
    assert progress["events_remaining"] == 138
    assert progress["effective_feature_start"] == "2026-09-03T22:02:47+00:00"
    forbidden = {"eta", "ready_by", "ready_on", "estimated_ready", "countdown", "days_remaining"}
    assert not (forbidden & set(progress)), "progress must be counts, never a projected date"


@pytest.mark.parametrize("ready", [True, False])
def test_snapshot_never_implies_training_is_authorized(ready):
    from backend.services.day_model_readiness import format_snapshot

    text = format_snapshot({"ready": ready, "reasons_not_ready": [], "checks": {}, "generated_at": "x"})

    assert f"READY_FOR_MODEL_TRAINING = {'true' if ready else 'false'}" in text
    assert "does not authorize model" in text
