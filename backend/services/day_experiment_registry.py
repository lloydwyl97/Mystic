"""DAY experiment registry. Research only — does not promote models.

Historical arms already count. This module appends; it does not reset.
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

TABLE = "day_experiment_registry"
LOCKED_TEST_FRAC = 0.20
EMBARGO_SEC = 6 * 3600


@dataclass(frozen=True)
class ExperimentArm:
    experiment_id: str
    timestamp: str
    feature_set: str
    target: str
    model_class: str
    hyperparameters: str
    training_period: str
    validation_period: str
    locked_period: str
    result: str
    promoted: bool
    notes: str


SEED_ARMS: tuple[ExperimentArm, ...] = (
    ExperimentArm("A", "2026-09-01", "current_production_veto", "lifecycle_net", "replay", "", "pre-66", "pre-66", "studied_66", "baseline d319c50 22-bps veto", False, "historical"),
    ExperimentArm("B", "2026-09-01", "honest_same_quality_floor", "lifecycle_net", "replay", "", "pre-66", "pre-66", "studied_66", "rejected_as_live_swap", False, "historical"),
    ExperimentArm("C", "2026-09-01", "honest_pure_net_ev", "lifecycle_net", "replay", "", "pre-66", "pre-66", "studied_66", "research", False, "historical"),
    ExperimentArm("D", "2026-09-01", "calibrated_net_ev", "lifecycle_net", "linear", "", "pre-66", "pre-66", "studied_66", "research", False, "historical"),
    ExperimentArm("E", "2026-09-01", "path_net_favorable_first", "lifecycle_net", "offline_rank", "", "pre-66", "pre-66", "studied_66", "rejected -22.9 vs -14.3 bps", False, "historical"),
    ExperimentArm("F", "2026-09-01", "direct_realized_net_ridge_145", "lifecycle_net", "ridge145", "", "pre-66", "pre-66", "studied_66", "rejected_unstable", False, "historical"),
    ExperimentArm("G", "2026-09-01", "champion_d319c50_grouped", "lifecycle_net", "production", "", "pre-66", "pre-66", "studied_66", "champion", False, "historical"),
    ExperimentArm("H", "2026-09-02", "calibrated_current_score_net_bps", "lifecycle_net", "challenger", "", "studied_66", "studied_66", "studied_66", "not promoted", False, "historical"),
    ExperimentArm("I", "2026-09-02", "pooled_ridge_low_dim_net", "lifecycle_net", "ridge", "", "studied_66", "studied_66", "studied_66", "not promoted", False, "historical"),
    ExperimentArm("J", "2026-09-02", "grouped_ranker", "lifecycle_net", "deferred", "", "studied_66", "studied_66", "studied_66", "deferred", False, "historical"),
    ExperimentArm("K", "2026-09-02", "order_flow_entry", "lifecycle_net", "flow", "", "studied_66", "studied_66", "studied_66", "rejected flow worse than v1", False, "historical"),
    ExperimentArm(
        "L",
        "2026-09-03",
        "4h_entry_structure_ridge",
        "production_exit_net_bps",
        "ridge_small",
        "lambda=8",
        "studied_66",
        "chrono_folds",
        "studied_66",
        "OOS net worse; break-timing better; not promoted",
        False,
        "4h pre-entry research; do not reuse 66 as lock",
    ),
)

SCHEMA_SQL = f"""
CREATE TABLE IF NOT EXISTS {TABLE} (
    experiment_id TEXT PRIMARY KEY,
    timestamp TEXT NOT NULL,
    feature_set TEXT,
    target TEXT,
    model_class TEXT,
    hyperparameters TEXT,
    training_period TEXT,
    validation_period TEXT,
    locked_period TEXT,
    result TEXT,
    promoted INTEGER NOT NULL DEFAULT 0,
    notes TEXT,
    meta_json TEXT
);
"""


def ensure_registry_schema(db_path: str | Path) -> None:
    conn = sqlite3.connect(str(db_path), timeout=30)
    try:
        conn.executescript(SCHEMA_SQL)
        conn.commit()
    finally:
        conn.close()


def seed_historical(db_path: str | Path) -> int:
    """Insert historical arms if missing. Never deletes. Never sets promoted=true."""
    ensure_registry_schema(db_path)
    conn = sqlite3.connect(str(db_path), timeout=30)
    inserted = 0
    try:
        for arm in SEED_ARMS:
            cur = conn.execute(
                f"""
                INSERT OR IGNORE INTO {TABLE}(
                    experiment_id, timestamp, feature_set, target, model_class,
                    hyperparameters, training_period, validation_period, locked_period,
                    result, promoted, notes, meta_json
                ) VALUES (?,?,?,?,?,?,?,?,?,?,0,?,?)
                """,
                (
                    arm.experiment_id,
                    arm.timestamp,
                    arm.feature_set,
                    arm.target,
                    arm.model_class,
                    arm.hyperparameters,
                    arm.training_period,
                    arm.validation_period,
                    arm.locked_period,
                    arm.result,
                    arm.notes,
                    json.dumps(asdict(arm), default=str),
                ),
            )
            inserted += int(cur.rowcount or 0)
        conn.commit()
    finally:
        conn.close()
    return inserted


def record_experiment(db_path: str | Path, payload: dict[str, Any]) -> None:
    ensure_registry_schema(db_path)
    conn = sqlite3.connect(str(db_path), timeout=30)
    try:
        conn.execute(
            f"""
            INSERT OR REPLACE INTO {TABLE}(
                experiment_id, timestamp, feature_set, target, model_class,
                hyperparameters, training_period, validation_period, locked_period,
                result, promoted, notes, meta_json
            ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)
            """,
            (
                str(payload.get("experiment_id") or ""),
                str(payload.get("timestamp") or datetime.now(timezone.utc).isoformat()),
                payload.get("feature_set"),
                payload.get("target"),
                payload.get("model_class"),
                json.dumps(payload.get("hyperparameters") or {}, default=str),
                payload.get("training_period"),
                payload.get("validation_period"),
                payload.get("locked_period"),
                payload.get("result"),
                1 if payload.get("promoted") else 0,
                payload.get("notes"),
                json.dumps(payload, default=str),
            ),
        )
        conn.commit()
    finally:
        conn.close()


def registry(db_path: str | Path | None = None) -> dict[str, Any]:
    arms = [asdict(a) for a in SEED_ARMS]
    stored = 0
    if db_path:
        seed_historical(db_path)
        conn = sqlite3.connect(str(db_path), timeout=30)
        try:
            stored = int(conn.execute(f"SELECT COUNT(*) FROM {TABLE}").fetchone()[0])
        finally:
            conn.close()
    return {
        "locked_test_frac": LOCKED_TEST_FRAC,
        "embargo_sec": EMBARGO_SEC,
        "arm_count": max(len(SEED_ARMS), stored),
        "historical_seed_count": len(SEED_ARMS),
        "promoted_count": 0,
        "arms": arms,
        "reset": False,
    }


__all__ = [
    "SEED_ARMS",
    "TABLE",
    "record_experiment",
    "registry",
    "seed_historical",
]
