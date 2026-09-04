"""
Bounded retention for large Mystic SQLite tables (AI / features / runtime audit).

Uses connect_rw + WAL + busy_timeout + run_locked_retry. Deletes in small batches.
Never runs VACUUM online. Offline VACUUM: scripts/offline_sqlite_vacuum_maintenance.sh (Mystic stopped).
"""

from __future__ import annotations

import logging
import os
import shutil
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
)

# Sealed research authority. These describe experiments and locks rather than sampling them,
# so ageing them out would silently destroy the record of what was already tried and make a
# prior result impossible to reproduce. They are tiny and must never be deleted on a timer.
PROTECTED_TABLES: frozenset[str] = frozenset(
    {
        "day_experiment_registry",
        "day_forward_lock_registry",
    }
)

# Learning rows a sealed lock may need to reproduce its dataset. Retention on these is
# additionally floored by the oldest cutoff any uninspected lock still depends on.
LOCK_DEPENDENT_TABLES: frozenset[str] = frozenset(
    {
        "day_decision_group_records",
        "day_decision_candidate_records",
        "day_decision_feature_artifacts",
        "day_decision_outcome_labels",
    }
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


def _iso_to_dt(value: Any) -> datetime | None:
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)


def _cutoff_value(policy: RetentionPolicy) -> str:
    cutoff_dt = datetime.now(timezone.utc) - timedelta(days=policy.keep_days)
    if policy.cutoff_format == "feature_ohlcv":
        return cutoff_dt.strftime("%Y-%m-%d %H:%M:%S.%f")
    return cutoff_dt.isoformat()


def lock_floor(conn: sqlite3.Connection) -> str | None:
    """Oldest instant any sealed forward lock still depends on.

    Retention must never delete learning rows at or after this point: the lock's dataset
    could no longer be rebuilt and an already-sealed result would become unreproducible.
    Inspected locks are protected too — a published experiment still has to be auditable.
    """
    if not _table_exists(conn, "day_forward_lock_registry"):
        return None
    cols = [row[1] for row in conn.execute("PRAGMA table_info(day_forward_lock_registry)")]
    wanted = [c for c in ("dataset_cutoff", "training_start", "locked_test_start") if c in cols]
    if not wanted:
        return None
    floors: list[str] = []
    for row in conn.execute(f"SELECT {', '.join(wanted)} FROM day_forward_lock_registry"):
        floors.extend(str(v).strip() for v in row if str(v or "").strip())
    return min(floors) if floors else None


def effective_cutoff(conn: sqlite3.Connection, policy: RetentionPolicy) -> tuple[str, str | None]:
    """Policy cutoff, clamped back to the lock floor for lock-dependent learning tables."""
    cutoff = _cutoff_value(policy)
    if policy.table not in LOCK_DEPENDENT_TABLES:
        return cutoff, None
    floor = lock_floor(conn)
    if floor and floor < cutoff:
        return floor, floor
    return cutoff, floor


def retention_dry_run(db_path: str | Path) -> dict[str, Any]:
    """Report exactly what retention would remove, without deleting anything.

    Read-only. Intended to be run before enabling enforcement on a new table, and safe to
    run at any time against production.
    """
    path = Path(db_path)
    if not path.is_file():
        return {"error": f"database not found: {path}", "tables": {}}

    out: dict[str, Any] = {
        "dry_run": True,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "protected_tables": sorted(PROTECTED_TABLES),
        "tables": {},
        "total_rows_to_delete": 0,
    }
    conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    try:
        out["lock_floor"] = lock_floor(conn)
        for table in sorted(PROTECTED_TABLES):
            if _table_exists(conn, table):
                rows = int(conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0])
                out["tables"][table] = {"status": "protected", "rows": rows, "rows_to_delete": 0}
        for policy in RETENTION_POLICIES:
            entry: dict[str, Any] = {"keep_days": policy.keep_days, "ts_column": policy.ts_column}
            out["tables"][policy.table] = entry
            if not _table_exists(conn, policy.table):
                entry["status"] = "skipped"
                entry["reason"] = "table_missing"
                continue
            if not _column_exists(conn, policy.table, policy.ts_column):
                entry["status"] = "skipped"
                entry["reason"] = f"timestamp_column_missing:{policy.ts_column}"
                continue
            cutoff, floor = effective_cutoff(conn, policy)
            entry["cutoff"] = cutoff
            entry["lock_floor_applied"] = bool(floor and floor <= cutoff)
            entry["rows"] = int(conn.execute(f"SELECT COUNT(*) FROM {policy.table}").fetchone()[0])
            entry["rows_to_delete"] = int(conn.execute(f"SELECT COUNT(*) FROM {policy.table} WHERE {policy.ts_column} < ?", (cutoff,)).fetchone()[0])
            entry["newest_row_to_delete"] = conn.execute(
                f"SELECT MAX({policy.ts_column}) FROM {policy.table} WHERE {policy.ts_column} < ?",
                (cutoff,),
            ).fetchone()[0]
            entry["oldest_row_retained"] = conn.execute(
                f"SELECT MIN({policy.ts_column}) FROM {policy.table} WHERE {policy.ts_column} >= ?",
                (cutoff,),
            ).fetchone()[0]
            try:
                total_bytes = int(conn.execute("SELECT SUM(pgsize) FROM dbstat WHERE name=?", (policy.table,)).fetchone()[0] or 0)
            except sqlite3.Error:
                total_bytes = 0
            entry["table_bytes"] = total_bytes
            entry["estimated_bytes_reclaimed"] = int(total_bytes * entry["rows_to_delete"] / entry["rows"]) if entry["rows"] else 0
            entry["status"] = "would_delete" if entry["rows_to_delete"] else "nothing_to_delete"
            out["total_rows_to_delete"] += entry["rows_to_delete"]
    finally:
        conn.close()
    return out


DISK_WARNING_FREE_GB = float(os.getenv("RETENTION_DISK_WARNING_FREE_GB", "5") or "5")
DISK_CRITICAL_FREE_GB = float(os.getenv("RETENTION_DISK_CRITICAL_FREE_GB", "2") or "2")


def storage_report(db_path: str | Path) -> dict[str, Any]:
    """Disk and learning-table growth, with a severity band.

    Observability only. A rising band is not a trading gate and must never be used as a
    reason to shorten the retention window; that requires separate, explicit evidence.
    """
    path = Path(db_path)
    out: dict[str, Any] = {"db_path": str(path), "generated_at": datetime.now(timezone.utc).isoformat()}
    if not path.is_file():
        return {**out, "error": "database not found"}

    usage = shutil.disk_usage(path.parent)
    free_gb = usage.free / 1024**3
    out["db_bytes"] = path.stat().st_size
    out["db_gib"] = round(out["db_bytes"] / 1024**3, 3)
    out["filesystem_total_gib"] = round(usage.total / 1024**3, 2)
    out["filesystem_free_gib"] = round(free_gb, 2)
    out["filesystem_used_pct"] = round(100.0 * usage.used / usage.total, 1) if usage.total else None

    conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    try:
        learning_bytes = 0
        oldest: str | None = None
        newest: str | None = None
        for table in sorted(LOCK_DEPENDENT_TABLES | PROTECTED_TABLES):
            if not _table_exists(conn, table):
                continue
            try:
                learning_bytes += int(conn.execute("SELECT SUM(pgsize) FROM dbstat WHERE name=?", (table,)).fetchone()[0] or 0)
            except sqlite3.Error:
                pass
            column = "created_at" if _column_exists(conn, table, "created_at") else "timestamp"
            if not _column_exists(conn, table, column):
                continue
            lo, hi = conn.execute(f"SELECT MIN({column}), MAX({column}) FROM {table}").fetchone()
            oldest = min(x for x in (oldest, lo) if x) if (oldest or lo) else None
            newest = max(x for x in (newest, hi) if x) if (newest or hi) else None
    finally:
        conn.close()

    out["learning_table_bytes"] = learning_bytes
    out["learning_oldest"] = oldest
    out["learning_newest"] = newest
    span_days = 0.0
    lo_dt, hi_dt = _iso_to_dt(oldest), _iso_to_dt(newest)
    if lo_dt and hi_dt:
        span_days = max((hi_dt - lo_dt).total_seconds() / 86400.0, 0.0)
    out["learning_span_days"] = round(span_days, 3)
    per_day = learning_bytes / span_days if span_days > 0 else 0.0
    out["learning_bytes_per_day"] = int(per_day)
    out["learning_mb_per_day"] = round(per_day / 1024**2, 3)
    for horizon in (30, 60, 90):
        out[f"projection_{horizon}d_gib"] = round(per_day * horizon / 1024**3, 3)

    if free_gb <= DISK_CRITICAL_FREE_GB:
        out["severity"] = "CRITICAL"
    elif free_gb <= DISK_WARNING_FREE_GB:
        out["severity"] = "WARNING"
    else:
        out["severity"] = "OK"
    out["severity_note"] = "observability only; not a trading gate and not a reason to shorten retention"
    return out


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

                if policy.table in PROTECTED_TABLES:
                    entry["status"] = "skipped"
                    entry["reason"] = "protected_research_authority"
                    logger.info("LARGE_TABLE_RETENTION: skip %s (protected)", policy.table)
                    continue

                cutoff, floor = effective_cutoff(conn, policy)
                entry["cutoff"] = cutoff
                if floor:
                    entry["lock_floor"] = floor
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
        "--dry-run",
        action="store_true",
        help="Report what would be deleted and exit without touching the database",
    )
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

    if args.dry_run:
        print(json.dumps({"retention": retention_dry_run(args.db), "storage": storage_report(args.db)}, indent=2))
        sys.exit(0)

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
