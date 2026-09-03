"""
Bounded retention for large Mystic SQLite tables (AI / features / runtime audit).

Uses connect_rw + WAL + busy_timeout + run_locked_retry. Deletes in small batches.
Never runs VACUUM online. Offline VACUUM: scripts/offline_sqlite_vacuum_maintenance.sh (Mystic stopped).
"""

from __future__ import annotations

import logging
import os
import sqlite3
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from backend.utils.sqlite_runtime import connect_rw, run_locked_retry

logger = logging.getLogger(__name__)

DEFAULT_BATCH_SIZE = int(os.getenv("LARGE_TABLE_RETENTION_BATCH_SIZE", "500") or "500")
DEFAULT_MAX_BATCHES_PER_TABLE = int(os.getenv("LARGE_TABLE_RETENTION_MAX_BATCHES", "100") or "100")
DEFAULT_MAX_RUN_SECONDS = float(os.getenv("LARGE_TABLE_RETENTION_MAX_RUN_SEC", "45") or "45")
DEFAULT_INTERVAL_SEC = float(os.getenv("LARGE_TABLE_RETENTION_INTERVAL_SEC", "21600") or "21600")  # 6h
# Long initial delay so retention never competes with post-restart exit/bar writes.
DEFAULT_INITIAL_DELAY_SEC = float(os.getenv("LARGE_TABLE_RETENTION_INITIAL_DELAY_SEC", "900") or "900")


@dataclass(frozen=True)
class RetentionPolicy:
    table: str
    ts_column: str
    keep_days: int
    cutoff_format: str  # "iso_utc" | "feature_ohlcv"


RETENTION_POLICIES: tuple[RetentionPolicy, ...] = (
    RetentionPolicy("ai_inference_log", "ts_utc", 90, "iso_utc"),
    RetentionPolicy("ai_context_snapshots", "ts_utc", 30, "iso_utc"),
    # strategy_runtime_audit writes ~160k rows/day — keep only 3 days (~480k rows max)
    RetentionPolicy("strategy_runtime_audit", "ts_utc", 3, "iso_utc"),
    RetentionPolicy("feature_ohlcv", "ts", 90, "feature_ohlcv"),
    RetentionPolicy("paper_trades", "timestamp", 90, "iso_utc"),
    # Append-only high-frequency logs (created_at tracks insert time).
    RetentionPolicy("ai_live_signals", "created_at", 30, "iso_utc"),
    RetentionPolicy("pipeline_decisions", "created_at", 30, "iso_utc"),
    RetentionPolicy("ai_rank_snapshots", "created_at", 14, "iso_utc"),
    RetentionPolicy("scalp_rejects", "created_at", 7, "iso_utc"),
    RetentionPolicy("ai_feature_samples", "created_at", 14, "iso_utc"),
    RetentionPolicy("decision_book_tape", "ts_utc", 14, "iso_utc"),
    RetentionPolicy("day_decision_group_records", "created_at", 90, "iso_utc"),
    RetentionPolicy("day_decision_candidate_records", "created_at", 90, "iso_utc"),
    RetentionPolicy("day_decision_feature_artifacts", "created_at", 90, "iso_utc"),
    RetentionPolicy("day_decision_outcome_labels", "created_at", 90, "iso_utc"),
    RetentionPolicy("day_experiment_registry", "timestamp", 90, "iso_utc"),
    RetentionPolicy("day_forward_lock_registry", "created_at", 90, "iso_utc"),
)


def _table_exists(conn: sqlite3.Connection, table: str) -> bool:
    row = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=? LIMIT 1",
        (table,),
    ).fetchone()
    return row is not None


def _column_exists(conn: sqlite3.Connection, table: str, column: str) -> bool:
    rows = conn.execute(f"PRAGMA table_info({table})").fetchall()
    return any(r[1] == column for r in rows)


def _cutoff_value(policy: RetentionPolicy) -> str:
    cutoff_dt = datetime.now(timezone.utc) - timedelta(days=policy.keep_days)
    if policy.cutoff_format == "feature_ohlcv":
        return cutoff_dt.strftime("%Y-%m-%d %H:%M:%S.%f")
    return cutoff_dt.isoformat()


def _delete_one_batch(
    conn: sqlite3.Connection,
    policy: RetentionPolicy,
    cutoff: str,
    batch_size: int,
) -> int:
    sql = f"""
        DELETE FROM {policy.table}
        WHERE rowid IN (
            SELECT rowid FROM {policy.table}
            WHERE {policy.ts_column} < ?
            LIMIT ?
        )
    """
    cur = conn.cursor()
    cur.execute(sql, (cutoff, batch_size))
    return int(cur.rowcount or 0)


def run_large_table_retention(
    db_path: str | Path,
    *,
    batch_size: int | None = None,
    max_batches_per_table: int | None = None,
    max_run_seconds: float | None = None,
    unlimited: bool = False,
) -> dict[str, Any]:
    """
    Delete rows older than each policy's retention window.

    Returns summary dict with per-table deleted/skipped counts and reasons.
    """
    path = Path(db_path)
    if not path.is_file():
        return {"error": f"database not found: {path}", "tables": {}}

    batch_sz = max(1, batch_size or DEFAULT_BATCH_SIZE)
    max_batches = max(1, max_batches_per_table or DEFAULT_MAX_BATCHES_PER_TABLE)
    run_budget = max(5.0, max_run_seconds or DEFAULT_MAX_RUN_SECONDS)
    if unlimited:
        max_batches = 10_000_000
        run_budget = 86400.0

    started = time.monotonic()
    summary: dict[str, Any] = {"tables": {}, "total_deleted": 0}

    def _run() -> None:
        with connect_rw(path) as conn:
            for policy in RETENTION_POLICIES:
                if time.monotonic() - started >= run_budget:
                    summary.setdefault("stopped_early", "run_time_budget")
                    break

                entry: dict[str, Any] = {
                    "keep_days": policy.keep_days,
                    "ts_column": policy.ts_column,
                    "deleted": 0,
                    "batches": 0,
                }
                summary["tables"][policy.table] = entry

                if not _table_exists(conn, policy.table):
                    entry["status"] = "skipped"
                    entry["reason"] = "table_missing"
                    logger.info("LARGE_TABLE_RETENTION: skip %s (table missing)", policy.table)
                    continue

                if not _column_exists(conn, policy.table, policy.ts_column):
                    entry["status"] = "skipped"
                    entry["reason"] = f"timestamp_column_missing:{policy.ts_column}"
                    logger.warning(
                        "LARGE_TABLE_RETENTION: skip %s (%s column missing)",
                        policy.table,
                        policy.ts_column,
                    )
                    continue

                cutoff = _cutoff_value(policy)
                entry["cutoff"] = cutoff
                deleted_total = 0
                batches = 0

                while batches < max_batches and time.monotonic() - started < run_budget:
                    conn.execute("BEGIN IMMEDIATE")
                    try:
                        n = _delete_one_batch(conn, policy, cutoff, batch_sz)
                        conn.commit()
                    except Exception:
                        conn.rollback()
                        raise

                    if n <= 0:
                        break
                    deleted_total += n
                    batches += 1
                    if n < batch_sz:
                        break
                    time.sleep(0.01)

                entry["deleted"] = deleted_total
                entry["batches"] = batches
                entry["status"] = "ok" if deleted_total or batches == 0 else "partial"
                summary["total_deleted"] += deleted_total

                if deleted_total:
                    logger.info(
                        "LARGE_TABLE_RETENTION: %s deleted=%d batches=%d cutoff=%s keep_days=%d",
                        policy.table,
                        deleted_total,
                        batches,
                        cutoff,
                        policy.keep_days,
                    )
                else:
                    logger.info(
                        "LARGE_TABLE_RETENTION: %s nothing to delete (keep_days=%d cutoff=%s)",
                        policy.table,
                        policy.keep_days,
                        cutoff,
                    )

    run_locked_retry(_run)
    summary["elapsed_sec"] = round(time.monotonic() - started, 3)
    return summary


def run_offline_integrity_check(db_path: str | Path) -> dict[str, Any]:
    """Offline-only read-only integrity_check. Caller must ensure Mystic is stopped."""
    path = Path(db_path)
    result: dict[str, Any] = {"path": str(path)}

    def _check() -> None:
        conn = sqlite3.connect(str(path), timeout=120.0)
        try:
            conn.execute("PRAGMA busy_timeout=30000;")
            ic = conn.execute("PRAGMA integrity_check").fetchone()
            result["integrity_check"] = ic[0] if ic else "unknown"
        finally:
            conn.close()

    run_locked_retry(_check)
    return result


def run_offline_vacuum_and_integrity(db_path: str | Path) -> dict[str, Any]:
    """Offline-only: integrity_check then VACUUM. Caller must ensure Mystic is stopped."""
    path = Path(db_path)
    result: dict[str, Any] = {"path": str(path)}

    def _vacuum() -> None:
        conn = sqlite3.connect(str(path), timeout=120.0)
        try:
            conn.execute("PRAGMA journal_mode=WAL;")
            conn.execute("PRAGMA busy_timeout=30000;")
            ic = conn.execute("PRAGMA integrity_check").fetchone()
            result["integrity_check"] = ic[0] if ic else "unknown"
            if result["integrity_check"] != "ok":
                return
            conn.execute("VACUUM")
            conn.commit()
            result["vacuum"] = "ok"
        finally:
            conn.close()

    run_locked_retry(_vacuum)
    return result


if __name__ == "__main__":
    import argparse
    import json
    import sys

    from backend.database_schema import DATABASE_PATH

    parser = argparse.ArgumentParser(description="Large-table SQLite retention (offline-capable)")
    parser.add_argument("--db", default=str(DATABASE_PATH))
    parser.add_argument("--unlimited", action="store_true", help="Delete until caught up (offline)")
    parser.add_argument(
        "--integrity-check",
        action="store_true",
        help="Run integrity_check after retention (offline only, no VACUUM)",
    )
    parser.add_argument(
        "--vacuum",
        action="store_true",
        help="Run integrity_check + VACUUM after retention (offline only, Mystic stopped)",
    )
    args = parser.parse_args()

    before = Path(args.db).stat().st_size if Path(args.db).is_file() else 0
    out = run_large_table_retention(args.db, unlimited=args.unlimited)
    after_ret = Path(args.db).stat().st_size if Path(args.db).is_file() else 0
    out["size_before_bytes"] = before
    out["size_after_retention_bytes"] = after_ret

    if args.integrity_check:
        ic = run_offline_integrity_check(args.db)
        out["integrity"] = ic
        if ic.get("integrity_check") != "ok":
            print(json.dumps(out, indent=2))
            sys.exit(f"integrity_check failed: {ic.get('integrity_check')!r}")

    if args.vacuum:
        vac = run_offline_vacuum_and_integrity(args.db)
        out["vacuum_result"] = vac
        out["size_after_vacuum_bytes"] = Path(args.db).stat().st_size
        if vac.get("integrity_check") != "ok":
            print(json.dumps(out, indent=2))
            sys.exit(f"integrity_check failed: {vac.get('integrity_check')!r}")
        if vac.get("vacuum") != "ok":
            print(json.dumps(out, indent=2))
            sys.exit("VACUUM did not complete")

    print(json.dumps(out, indent=2))
    sys.exit(0 if "error" not in out else 1)
