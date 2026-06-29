"""
Append-only audit trail for strategy/runtime proof (Step 1 instrumentation).

Rows link signal emission → consumption → fills via decision_id / paper_trade_id.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from backend.database_schema import DATABASE_PATH
from backend.services.ai_decision_contract import REDIS_TTL_AI_CONTEXT_SEC
from backend.utils.sqlite_runtime import connect_rw, run_locked_retry

logger = logging.getLogger(__name__)

# --- Event types (stable strings for SQL filtering) ---
EVT_SIGNAL_EMITTED = "SIGNAL_EMITTED"
EVT_SIGNAL_CONSUME = "SIGNAL_CONSUME"
EVT_CANDIDATE_ENQUEUED = "CANDIDATE_ENQUEUED"
EVT_BUY_EXECUTED = "BUY_EXECUTED"
EVT_BUY_REJECT = "BUY_REJECT"
EVT_ENTRY_VETO = "ENTRY_VETO"
EVT_SELL_EXECUTED = "SELL_EXECUTED"
EVT_EXIT_SKELETON_EVAL = "EXIT_SKELETON_EVAL"

_CTX_FRESH_EFFECTIVE_LOGGED = False


def get_ctx_fresh_max_age_sec() -> float:
    """
    Max wall-clock age (seconds) for ai_context ts_utc at signal emit / entry gate.

    Env: CTX_FRESH_MAX_AGE_SEC (default 900). Clamped to [60, max(180, REDIS_TTL_AI_CONTEXT_SEC)]
    so we never accept context older than the Redis key TTL window or below one minute.
    """
    global _CTX_FRESH_EFFECTIVE_LOGGED
    raw = os.getenv("CTX_FRESH_MAX_AGE_SEC", "900")
    try:
        v = float(raw)
    except ValueError:
        v = 900.0
    lo = 60.0
    hi = float(max(180, int(REDIS_TTL_AI_CONTEXT_SEC)))
    effective = max(lo, min(hi, v))
    if not _CTX_FRESH_EFFECTIVE_LOGGED:
        _CTX_FRESH_EFFECTIVE_LOGGED = True
        logger.info(
            "CTX_FRESH_MAX_AGE_SEC effective=%.3f env_raw=%r redis_ttl_cap=%.3f",
            effective,
            raw,
            hi,
        )
    return effective


# Import-time snapshot for legacy readers; hot paths should call get_ctx_fresh_max_age_sec().
CTX_FRESH_MAX_AGE_SEC = get_ctx_fresh_max_age_sec()


def sha256_file(path: str | Path) -> str:
    """SHA-256 hex digest of file contents; empty string on failure."""
    p = Path(path)
    if not p.is_file():
        return ""
    h = hashlib.sha256()
    try:
        with p.open("rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                h.update(chunk)
        return h.hexdigest()
    except OSError:
        return ""


def ensure_strategy_runtime_audit_table(db_path: str | Path = DATABASE_PATH) -> None:
    """Idempotent CREATE for strategy_runtime_audit."""
    ddl = """
    CREATE TABLE IF NOT EXISTS strategy_runtime_audit (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        ts_utc TEXT NOT NULL,
        event_type TEXT NOT NULL,
        decision_id TEXT,
        strategy_id TEXT,
        symbol TEXT,
        redis_signal_key TEXT,
        artifact_path TEXT,
        artifact_sha256 TEXT,
        feature_version INTEGER,
        feature_dim INTEGER,
        context_fresh INTEGER NOT NULL DEFAULT 1,
        context_age_sec REAL,
        context_defaulted_json TEXT,
        reject_reason TEXT,
        paper_trade_id TEXT,
        buy_trade_id TEXT,
        sell_trade_id TEXT,
        exit_reason TEXT,
        exit_type TEXT,
        extra_json TEXT
    )
    """
    idx = [
        "CREATE INDEX IF NOT EXISTS ix_sra_decision ON strategy_runtime_audit(decision_id)",
        "CREATE INDEX IF NOT EXISTS ix_sra_ts ON strategy_runtime_audit(ts_utc)",
        "CREATE INDEX IF NOT EXISTS ix_sra_paper_trade ON strategy_runtime_audit(paper_trade_id)",
        "CREATE INDEX IF NOT EXISTS ix_sra_strategy ON strategy_runtime_audit(strategy_id)",
    ]
    try:

        def _op() -> None:
            with connect_rw(db_path) as conn:
                row = conn.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='strategy_runtime_audit'").fetchone()
                if row:
                    return
                conn.execute("BEGIN IMMEDIATE")
                conn.execute(ddl)
                for s in idx:
                    conn.execute(s)
                conn.commit()

        run_locked_retry(_op, max_attempts=8)
    except sqlite3.Error as e:
        logger.warning("ensure_strategy_runtime_audit_table failed: %s", e)


def insert_audit_row_sync(
    *,
    event_type: str,
    decision_id: str | None = None,
    strategy_id: str | None = None,
    symbol: str | None = None,
    redis_signal_key: str | None = None,
    artifact_path: str | None = None,
    artifact_sha256: str | None = None,
    feature_version: int | None = None,
    feature_dim: int | None = None,
    context_fresh: bool | None = None,
    context_age_sec: float | None = None,
    context_defaulted_json: dict[str, Any] | None = None,
    reject_reason: str | None = None,
    paper_trade_id: str | None = None,
    buy_trade_id: str | None = None,
    sell_trade_id: str | None = None,
    exit_reason: str | None = None,
    exit_type: str | None = None,
    extra_json: dict[str, Any] | None = None,
    db_path: str | Path = DATABASE_PATH,
) -> int | None:
    """Insert one audit row; returns row id or None."""
    ensure_strategy_runtime_audit_table(db_path)
    ts = datetime.now(timezone.utc).isoformat()
    cf = 1 if context_fresh else 0 if context_fresh is False else 1
    ctx_def = json.dumps(context_defaulted_json, separators=(",", ":")) if context_defaulted_json else None
    ex = json.dumps(extra_json, separators=(",", ":")) if extra_json else None
    try:

        def _op() -> int | None:
            with connect_rw(db_path) as conn:
                conn.execute("BEGIN IMMEDIATE")
                cur = conn.execute(
                    """
                    INSERT INTO strategy_runtime_audit (
                        ts_utc, event_type, decision_id, strategy_id, symbol, redis_signal_key,
                        artifact_path, artifact_sha256, feature_version, feature_dim,
                        context_fresh, context_age_sec, context_defaulted_json,
                        reject_reason, paper_trade_id, buy_trade_id, sell_trade_id,
                        exit_reason, exit_type, extra_json
                    ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                    """,
                    (
                        ts,
                        event_type,
                        decision_id,
                        strategy_id,
                        symbol,
                        redis_signal_key,
                        artifact_path,
                        artifact_sha256,
                        feature_version,
                        feature_dim,
                        cf,
                        context_age_sec,
                        ctx_def,
                        reject_reason,
                        paper_trade_id,
                        buy_trade_id,
                        sell_trade_id,
                        exit_reason,
                        exit_type,
                        ex,
                    ),
                )
                conn.commit()
                return int(cur.lastrowid) if cur.lastrowid else None

        return run_locked_retry(_op)
    except sqlite3.Error as e:
        logger.warning("strategy_runtime_audit insert failed (%s): %s", event_type, e)
        return None


async def insert_audit_row_async(**kwargs: Any) -> int | None:
    return await asyncio.to_thread(insert_audit_row_sync, **kwargs)


def validate_loaded_slots(
    *,
    models: dict[str, Any],
    model_feature_versions: dict[str, int],
    model_feature_dims: dict[str, int],
    model_artifact_paths: dict[str, str],
    model_artifact_sha256: dict[str, str],
    enabled_strategies: tuple[str, ...],
    min_feature_version: int,
    min_feature_version_by_strategy: dict[str, int] | None = None,
) -> list[str]:
    """
    Runtime invariant check after artifact load. Returns list of violation messages (empty = OK).
    """
    violations: list[str] = []
    if not enabled_strategies:
        violations.append("INVARIANT: no LIVE_AI_STRATEGIES enabled")
    for slot, _model in models.items():
        fv = model_feature_versions.get(slot, 0)
        fd = model_feature_dims.get(slot, 0)
        path = model_artifact_paths.get(slot, "")
        digest = model_artifact_sha256.get(slot, "")
        sid = slot.split(":", 1)[0].strip().lower() if ":" in slot else ""
        need_fv = (min_feature_version_by_strategy or {}).get(sid, min_feature_version)
        if fv < need_fv:
            violations.append(f"INVARIANT: {slot} feature_version={fv} < min={need_fv}")
        if fd not in (124, 145):
            violations.append(f"INVARIANT: {slot} feature_dim={fd} not in {{124,145}}")
        if fv == 2 and fd != 145:
            violations.append(f"INVARIANT: {slot} v2 artifact dim mismatch fv={fv} dim={fd}")
        if fv >= 3 and fd != 145:
            violations.append(f"INVARIANT: {slot} v3+ artifact dim mismatch fv={fv} dim={fd}")
        if fv == 1 and fd != 124:
            violations.append(f"INVARIANT: {slot} v1 artifact dim mismatch fv={fv} dim={fd}")
        if path and not digest:
            violations.append(f"INVARIANT: {slot} artifact hash empty for path={path}")
    if violations:
        for v in violations:
            logger.error("STRATEGY_RUNTIME_INVARIANT %s", v)
    else:
        logger.info(
            "STRATEGY_RUNTIME_INVARIANT_OK slots=%d strategies=%s min_fv=%d",
            len(models),
            ",".join(enabled_strategies),
            min_feature_version,
        )
    return violations


def compute_context_freshness(ctx_age_sec: float | None) -> tuple[bool, float]:
    """Return (fresh_bool, age_sec) for audit columns."""
    if ctx_age_sec is None or ctx_age_sec < 0:
        return False, -1.0
    limit = get_ctx_fresh_max_age_sec()
    return ctx_age_sec <= limit, float(ctx_age_sec)


__all__ = [
    "CTX_FRESH_MAX_AGE_SEC",
    "EVT_BUY_EXECUTED",
    "EVT_BUY_REJECT",
    "EVT_CANDIDATE_ENQUEUED",
    "EVT_ENTRY_VETO",
    "EVT_EXIT_SKELETON_EVAL",
    "EVT_SELL_EXECUTED",
    "EVT_SIGNAL_CONSUME",
    "EVT_SIGNAL_EMITTED",
    "compute_context_freshness",
    "ensure_strategy_runtime_audit_table",
    "get_ctx_fresh_max_age_sec",
    "insert_audit_row_async",
    "insert_audit_row_sync",
    "sha256_file",
    "validate_loaded_slots",
]
