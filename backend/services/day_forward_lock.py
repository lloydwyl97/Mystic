"""Forward chronological lock metadata. Does not train or select models.

The historical Ocean 66 (2026-08-25 <= BUY < 2026-09-02) is studied.
It is not a fresh locked test.
"""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

TABLE = "day_forward_lock_registry"
LABEL_VERSION = "day_4h_outcome_label_v1"
FEATURE_VERSION = "day_4h_entry_structure_v1"
HISTORICAL_66_START = "2026-08-25T00:00:00+00:00"
HISTORICAL_66_END = "2026-09-02T00:00:00+00:00"
FORWARD_LOCK_START = "2026-09-03T00:00:00+00:00"

SCHEMA_SQL = f"""
CREATE TABLE IF NOT EXISTS {TABLE} (
    experiment_id TEXT PRIMARY KEY,
    created_at TEXT NOT NULL,
    dataset_cutoff TEXT,
    training_start TEXT,
    training_end TEXT,
    validation_start TEXT,
    validation_end TEXT,
    locked_test_start TEXT,
    locked_test_end TEXT,
    feature_version TEXT,
    label_version TEXT,
    inspected INTEGER NOT NULL DEFAULT 0,
    notes TEXT,
    meta_json TEXT
);
"""


def ensure_lock_schema(db_path: str | Path) -> None:
    conn = sqlite3.connect(str(db_path), timeout=30)
    try:
        conn.executescript(SCHEMA_SQL)
        conn.commit()
    finally:
        conn.close()


def default_forward_lock(*, experiment_id: str = "forward_4h_entry_lock_20260903") -> dict[str, Any]:
    return {
        "experiment_id": experiment_id,
        "dataset_cutoff": FORWARD_LOCK_START,
        "training_start": FORWARD_LOCK_START,
        "training_end": None,
        "validation_start": None,
        "validation_end": None,
        "locked_test_start": None,
        "locked_test_end": None,
        "feature_version": FEATURE_VERSION,
        "label_version": LABEL_VERSION,
        "inspected": False,
        "historical_66_excluded": True,
        "historical_66_window": [HISTORICAL_66_START, HISTORICAL_66_END],
        "notes": "New untouched chronological lock. Do not reuse the studied 66. Do not inspect the lock before sealing.",
    }


def register_lock(db_path: str | Path, payload: dict[str, Any] | None = None) -> dict[str, Any]:
    row = default_forward_lock()
    row.update(payload or {})
    ensure_lock_schema(db_path)
    conn = sqlite3.connect(str(db_path), timeout=30)
    try:
        conn.execute(
            f"""
            INSERT OR IGNORE INTO {TABLE}(
                experiment_id, created_at, dataset_cutoff, training_start, training_end,
                validation_start, validation_end, locked_test_start, locked_test_end,
                feature_version, label_version, inspected, notes, meta_json
            ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            """,
            (
                row["experiment_id"],
                datetime.now(timezone.utc).isoformat(),
                row.get("dataset_cutoff"),
                row.get("training_start"),
                row.get("training_end"),
                row.get("validation_start"),
                row.get("validation_end"),
                row.get("locked_test_start"),
                row.get("locked_test_end"),
                row.get("feature_version"),
                row.get("label_version"),
                1 if row.get("inspected") else 0,
                row.get("notes"),
                json.dumps(row, default=str),
            ),
        )
        conn.commit()
    finally:
        conn.close()
    return row


def challenger_export_schema() -> dict[str, Any]:
    """Predeclared future challenger columns. Not a trained model."""
    return {
        "inputs": [
            "symbol",
            "p_buy",
            "path_ev",
            "final_rank_score",
            "production_4h_break_true_at_decision",
            "distance_to_4h_break_bps",
            "4h_range_position",
            "minutes_into_4h_bar",
            "4h_alignment_state",
            "spread_bps",
            "expected_slippage",
            "estimated_all_in_cost_bps",
            "volatility",
            "liquidity",
        ],
        "targets": ["production_exit_net_bps"],
        "train": False,
        "live_gate": False,
    }


__all__ = [
    "FEATURE_VERSION",
    "FORWARD_LOCK_START",
    "HISTORICAL_66_END",
    "HISTORICAL_66_START",
    "LABEL_VERSION",
    "TABLE",
    "challenger_export_schema",
    "default_forward_lock",
    "register_lock",
]
