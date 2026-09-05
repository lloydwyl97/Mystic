"""CLOCK-V2 dataset partition authority. Research only — never alters trading.

The generic ``forward_4h_entry_lock_20260903`` is an open-ended forward lock for
the 4H-entry experiment. It is deliberately NOT used as the clock-v2 partition:
because it has no end, every clock-v2 group is inside it, so the clock-v2
readiness gate could never observe a labeled comparable group. This module gives
clock-v2 its own partition without reading, mutating, or unsealing that lock.

PRE_MODEL_QUARANTINE  captured under broken action-eligibility semantics; never
                      used for model fitting
DEVELOPMENT           starts at the predeclared clean v5 boundary; feature and
                      label accumulation plus chronological development
FINAL_TEST            NOT YET CREATED; may only be declared after the model
                      specification is frozen and training is complete, and must
                      then cover FUTURE observations
"""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PARTITION_CONTRACT_VERSION = "day_clock_v2_partition_v1"
TABLE_PARTITION = "day_clock_v2_partition_registry"

PRE_MODEL_QUARANTINE = "PRE_MODEL_QUARANTINE"
DEVELOPMENT = "DEVELOPMENT"
FINAL_TEST = "FINAL_TEST"
PARTITIONS: tuple[str, ...] = (PRE_MODEL_QUARANTINE, DEVELOPMENT, FINAL_TEST)

# Predeclared clean development boundary. Chosen as the next UTC calendar-day
# boundary after the eligibility correction was authored (2026-09-05T22:15Z) and
# BEFORE any outcome was inspected. Never backdate this to gain sample size.
CLOCK_V2_V5_DEVELOPMENT_START = "2026-09-06T00:00:00+00:00"
CLOCK_V2_V5_DEVELOPMENT_START_RATIONALE = (
    "Next UTC calendar-day boundary after the corrected action-availability capture was "
    "authored. Declared before inspecting any label. The 87 capture-v1 groups before it "
    "were recorded under NO_SCORED_CANDIDATE-as-unavailable semantics and are quarantined."
)

# The final test window does not exist yet, by contract.
FINAL_TEST_STATUS = "NOT_YET_CREATED"
FINAL_TEST_START: str | None = None
FINAL_TEST_END: str | None = None
FINAL_TEST_PRECONDITIONS: tuple[str, ...] = (
    "v5 model specification frozen",
    "training complete on DEVELOPMENT only",
    "challenger artifact frozen",
    "window declared strictly in the future relative to the training data",
    "used exactly once",
)

# The generic 4H lock stays exactly as it is.
GENERIC_4H_LOCK_ID = "forward_4h_entry_lock_20260903"
GENERIC_4H_LOCK_ROLE = "unchanged_uninspected_4h_entry_experiment_lock"
GENERIC_4H_LOCK_IS_CLOCK_V2_PARTITION = False


def _parse(raw: Any) -> datetime | None:
    if raw in (None, ""):
        return None
    if isinstance(raw, datetime):
        return raw if raw.tzinfo else raw.replace(tzinfo=timezone.utc)
    if isinstance(raw, (int, float)):
        try:
            return datetime.fromtimestamp(float(raw), tz=timezone.utc)
        except (OverflowError, OSError, ValueError):
            return None
    try:
        dt = datetime.fromisoformat(str(raw).replace("Z", "+00:00"))
    except ValueError:
        return None
    return dt.replace(tzinfo=timezone.utc) if dt.tzinfo is None else dt.astimezone(timezone.utc)


def partition_for(
    decision_ts: Any,
    *,
    development_start: str = CLOCK_V2_V5_DEVELOPMENT_START,
    final_test_start: str | None = FINAL_TEST_START,
    final_test_end: str | None = FINAL_TEST_END,
) -> str:
    """Partition label for one decision timestamp. Never raises."""
    when = _parse(decision_ts)
    dev = _parse(development_start)
    if when is None or dev is None:
        return PRE_MODEL_QUARANTINE
    ft_start = _parse(final_test_start)
    ft_end = _parse(final_test_end)
    if ft_start is not None and when >= ft_start and (ft_end is None or when < ft_end):
        return FINAL_TEST
    if when >= dev:
        return DEVELOPMENT
    return PRE_MODEL_QUARANTINE


def is_development(decision_ts: Any, **kwargs: Any) -> bool:
    return partition_for(decision_ts, **kwargs) == DEVELOPMENT


def partition_contract() -> dict[str, Any]:
    return {
        "contract_version": PARTITION_CONTRACT_VERSION,
        "partitions": list(PARTITIONS),
        "clock_v2_v5_development_start": CLOCK_V2_V5_DEVELOPMENT_START,
        "clock_v2_v5_development_start_rationale": CLOCK_V2_V5_DEVELOPMENT_START_RATIONALE,
        "development_start_backdated": False,
        "pre_model_quarantine": {
            "meaning": "capture-v1 / broken action-eligibility period",
            "used_for_model_fitting": False,
            "reconstruction_allowed_for_audit": True,
            "counts_toward_v5_trainability": False,
        },
        "development": {
            "meaning": "clean forward v5 collection",
            "starts_at": CLOCK_V2_V5_DEVELOPMENT_START,
            "ends_at": None,
            "used_for_model_fitting": True,
            "folds": "expanding chronological, grouped timestamp atomic, purged and embargoed",
        },
        "final_test": {
            "status": FINAL_TEST_STATUS,
            "start": FINAL_TEST_START,
            "end": FINAL_TEST_END,
            "preconditions": list(FINAL_TEST_PRECONDITIONS),
            "must_be_future_relative_to_training": True,
            "earlier_period_as_final_test_forbidden": True,
        },
        "generic_4h_lock": {
            "experiment_id": GENERIC_4H_LOCK_ID,
            "role": GENERIC_4H_LOCK_ROLE,
            "is_clock_v2_partition": GENERIC_4H_LOCK_IS_CLOCK_V2_PARTITION,
            "mutated": False,
            "inspected": False,
            "why_not_reused": (
                "It is open-ended, so every clock-v2 group falls inside it and comparable "
                "labeled groups could never be counted. Clock-v2 needs its own partition; "
                "the 4H lock stays sealed for its own experiment."
            ),
        },
    }


def ensure_partition_schema(db_path: str | Path) -> None:
    conn = sqlite3.connect(str(db_path), timeout=30)
    try:
        conn.executescript(
            f"""
            CREATE TABLE IF NOT EXISTS {TABLE_PARTITION} (
                contract_version TEXT PRIMARY KEY,
                created_at TEXT NOT NULL,
                development_start TEXT NOT NULL,
                final_test_status TEXT NOT NULL,
                final_test_start TEXT,
                final_test_end TEXT,
                generic_4h_lock_id TEXT,
                generic_4h_lock_is_partition INTEGER NOT NULL DEFAULT 0,
                contract_json TEXT NOT NULL
            );
            """
        )
        conn.commit()
    finally:
        conn.close()


def register_partition_contract(db_path: str | Path) -> dict[str, Any]:
    """Persist the partition contract once. Never rewrites an existing row."""
    contract = partition_contract()
    ensure_partition_schema(db_path)
    conn = sqlite3.connect(str(db_path), timeout=30)
    try:
        conn.execute(
            f"""
            INSERT OR IGNORE INTO {TABLE_PARTITION}(
                contract_version, created_at, development_start, final_test_status,
                final_test_start, final_test_end, generic_4h_lock_id,
                generic_4h_lock_is_partition, contract_json
            ) VALUES (?,?,?,?,?,?,?,0,?)
            """,
            (
                PARTITION_CONTRACT_VERSION,
                datetime.now(timezone.utc).isoformat(),
                CLOCK_V2_V5_DEVELOPMENT_START,
                FINAL_TEST_STATUS,
                FINAL_TEST_START,
                FINAL_TEST_END,
                GENERIC_4H_LOCK_ID,
                json.dumps(contract, default=str),
            ),
        )
        conn.commit()
    finally:
        conn.close()
    return contract


def stored_partition_contract(db_path: str | Path) -> dict[str, Any] | None:
    if not Path(db_path).exists():
        return None
    conn = sqlite3.connect(f"file:{Path(db_path)}?mode=ro", uri=True)
    try:
        if conn.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (TABLE_PARTITION,)).fetchone() is None:
            return None
        row = conn.execute(
            f"SELECT created_at, development_start, final_test_status, contract_json FROM {TABLE_PARTITION} WHERE contract_version=?",
            (PARTITION_CONTRACT_VERSION,),
        ).fetchone()
    finally:
        conn.close()
    if row is None:
        return None
    try:
        contract = json.loads(row[3])
    except (TypeError, ValueError):
        contract = {}
    return {
        "created_at": row[0],
        "development_start": row[1],
        "final_test_status": row[2],
        "contract": contract,
    }


def declare_final_test_window(*_args: Any, **_kwargs: Any) -> None:
    """Refuse to create the final test window from this task."""
    raise RuntimeError(
        "FINAL_TEST is NOT_YET_CREATED by contract. It may only be declared after the v5 "
        "specification is frozen and training is complete, and must cover future observations."
    )


__all__ = [
    "CLOCK_V2_V5_DEVELOPMENT_START",
    "DEVELOPMENT",
    "FINAL_TEST",
    "FINAL_TEST_STATUS",
    "GENERIC_4H_LOCK_ID",
    "GENERIC_4H_LOCK_IS_CLOCK_V2_PARTITION",
    "PARTITIONS",
    "PARTITION_CONTRACT_VERSION",
    "PRE_MODEL_QUARANTINE",
    "TABLE_PARTITION",
    "declare_final_test_window",
    "ensure_partition_schema",
    "is_development",
    "partition_contract",
    "partition_for",
    "register_partition_contract",
    "stored_partition_contract",
]
