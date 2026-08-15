"""Explicit pre-repair vs post-repair validation boundary.

Historical trades stay in place. New execution after the atomic-OPEN /
SCALP-money-DB deploy is stamped against this cutoff so 60% acceptance
is never computed on mixed books.
"""

from __future__ import annotations

import json
import logging
import os
import sqlite3
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

CUTOFF_LABEL = "atomic-open-scalp-db-20260814"
CUTOFF_TABLE = "repair_validation_cutoff"

_SCHEMA = f"""
CREATE TABLE IF NOT EXISTS {CUTOFF_TABLE} (
    id INTEGER PRIMARY KEY CHECK (id = 1),
    label TEXT NOT NULL,
    cutoff_utc TEXT NOT NULL,
    git_sha TEXT,
    engine TEXT NOT NULL,
    note TEXT
);
"""


def _git_sha(repo_root: str) -> str:
    try:
        out = subprocess.check_output(
            ["git", "-C", repo_root, "rev-parse", "--short", "HEAD"],
            text=True,
            timeout=5,
        )
        return (out or "").strip()
    except Exception:
        return ""


def ensure_validation_cutoff(db_path: str | Path, *, engine: str, repo_root: str | None = None) -> dict[str, Any]:
    """Insert the cutoff row once. Never overwrite an existing boundary."""
    path = str(db_path)
    root = repo_root or str(Path(path).resolve().parent)
    now = datetime.now(timezone.utc).isoformat()
    sha = _git_sha(root)
    conn = sqlite3.connect(path, timeout=15)
    try:
        conn.execute("BEGIN IMMEDIATE")
        conn.executescript(_SCHEMA)
        existing = conn.execute(f"SELECT label, cutoff_utc, git_sha, engine FROM {CUTOFF_TABLE} WHERE id=1").fetchone()
        if existing:
            conn.commit()
            row = {
                "label": existing[0],
                "cutoff_utc": existing[1],
                "git_sha": existing[2],
                "engine": existing[3],
                "created": False,
            }
            logger.info("VALIDATION_CUTOFF_EXISTS engine=%s label=%s cutoff=%s", engine, row["label"], row["cutoff_utc"])
            return row
        conn.execute(
            f"""
            INSERT INTO {CUTOFF_TABLE} (id, label, cutoff_utc, git_sha, engine, note)
            VALUES (1, ?, ?, ?, ?, ?)
            """,
            (
                CUTOFF_LABEL,
                now,
                sha,
                engine,
                "Trades at/after cutoff_utc are clean post-repair. Earlier rows are historical.",
            ),
        )
        conn.commit()
        logger.critical(
            "VALIDATION_CUTOFF_CREATED engine=%s label=%s cutoff_utc=%s git_sha=%s db=%s",
            engine,
            CUTOFF_LABEL,
            now,
            sha,
            path,
        )
        return {"label": CUTOFF_LABEL, "cutoff_utc": now, "git_sha": sha, "engine": engine, "created": True}
    except Exception:
        conn.rollback()
        logger.exception("VALIDATION_CUTOFF_FAILED engine=%s db=%s", engine, path)
        raise
    finally:
        conn.close()


def replace_validation_cutoff(
    db_path: str | Path,
    *,
    engine: str,
    label: str,
    cutoff_utc: str,
    repo_root: str | None = None,
    note: str = "User-marked clean-sample start. Earlier rows stay in the book and are not acceptance.",
) -> dict[str, Any]:
    """Overwrite the stored cutoff. Only when the operator explicitly marks a new start."""
    path = str(db_path)
    root = repo_root or str(Path(path).resolve().parent)
    sha = _git_sha(root)
    conn = sqlite3.connect(path, timeout=15)
    try:
        conn.execute("BEGIN IMMEDIATE")
        conn.executescript(_SCHEMA)
        prior = conn.execute(f"SELECT label, cutoff_utc, git_sha FROM {CUTOFF_TABLE} WHERE id=1").fetchone()
        conn.execute(
            f"""
            INSERT INTO {CUTOFF_TABLE} (id, label, cutoff_utc, git_sha, engine, note)
            VALUES (1, ?, ?, ?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET
                label=excluded.label,
                cutoff_utc=excluded.cutoff_utc,
                git_sha=excluded.git_sha,
                engine=excluded.engine,
                note=excluded.note
            """,
            (label, cutoff_utc, sha, engine, note),
        )
        conn.commit()
        logger.critical(
            "VALIDATION_CUTOFF_REPLACED engine=%s label=%s cutoff_utc=%s prior=%s git_sha=%s db=%s",
            engine,
            label,
            cutoff_utc,
            (prior[1] if prior else None),
            sha,
            path,
        )
        return {
            "label": label,
            "cutoff_utc": cutoff_utc,
            "git_sha": sha,
            "engine": engine,
            "prior_cutoff_utc": prior[1] if prior else None,
            "replaced": True,
        }
    except Exception:
        conn.rollback()
        logger.exception("VALIDATION_CUTOFF_REPLACE_FAILED engine=%s db=%s", engine, path)
        raise
    finally:
        conn.close()


def read_validation_cutoff(db_path: str | Path) -> dict[str, Any] | None:
    path = str(db_path)
    if not Path(path).exists():
        return None
    conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True, timeout=5)
    try:
        row = conn.execute(f"SELECT label, cutoff_utc, git_sha, engine, note FROM {CUTOFF_TABLE} WHERE id=1").fetchone()
        if not row:
            return None
        return {
            "label": row[0],
            "cutoff_utc": row[1],
            "git_sha": row[2],
            "engine": row[3],
            "note": row[4],
        }
    except sqlite3.OperationalError:
        return None
    finally:
        conn.close()


def post_repair_where(timestamp_column: str = "timestamp") -> tuple[str, tuple[str, ...]]:
    """SQL fragment: timestamp_column >= cutoff. Caller supplies cutoff_utc."""
    return f"{timestamp_column} >= ?", ()


# Accounting keeps these rows. Strategy-acceptance stats must exclude them.
RECONCILIATION_EXIT_REASONS = frozenset(
    {
        "RECONCILIATION_MANUAL_EXIT",
        "RECONCILIATION_EXIT",
    }
)
RECONCILIATION_TRADE_IDS = frozenset(
    {
        # Ocean DAY SELL of restored orphan XRP BUY 980. Not an AI-selected entry.
        "983",
    }
)


def is_strategy_acceptance_eligible(
    *,
    exit_reason: str | None = None,
    trade_id: str | None = None,
    extra: dict[str, Any] | None = None,
) -> bool:
    """True when a close may count toward clean AI-strategy WR/net/expectancy."""
    reason = str(exit_reason or "").strip().upper()
    if reason in RECONCILIATION_EXIT_REASONS:
        return False
    extra = extra or {}
    acceptance = str(extra.get("acceptance_class") or extra.get("strategy_acceptance") or "").strip().upper()
    if acceptance in RECONCILIATION_EXIT_REASONS or acceptance == "RECONCILIATION":
        return False
    tid = str(trade_id or extra.get("trade_id") or "").strip()
    if tid in RECONCILIATION_TRADE_IDS:
        return False
    return True


def mark_reconciliation_manual_exit(
    db_path: str | Path,
    *,
    trade_id: int = 983,
    note: str = "Restored orphan inventory flatten; excluded from clean AI-strategy acceptance.",
) -> dict[str, Any]:
    """Stamp an existing close as reconciliation. Does not delete or change cash."""
    path = str(db_path)
    conn = sqlite3.connect(path, timeout=15)
    conn.row_factory = sqlite3.Row
    try:
        row = conn.execute(
            "SELECT id, trade_id, symbol, side, exit_reason, pnl, explainability_json FROM paper_trades WHERE id=?",
            (int(trade_id),),
        ).fetchone()
        if row is None:
            return {"updated": False, "reason": "not_found", "id": trade_id}
        extra = {}
        raw = row["explainability_json"]
        if raw:
            try:
                extra = json.loads(raw) if isinstance(raw, str) else dict(raw)
            except Exception:
                extra = {}
        extra["acceptance_class"] = "RECONCILIATION_MANUAL_EXIT"
        extra["strategy_acceptance_eligible"] = False
        extra["reconciliation_note"] = note
        extra["original_exit_reason"] = row["exit_reason"]
        conn.execute(
            """
            UPDATE paper_trades
            SET exit_reason='RECONCILIATION_MANUAL_EXIT',
                explainability_json=?
            WHERE id=?
            """,
            (json.dumps(extra, separators=(",", ":"), default=str), int(trade_id)),
        )
        conn.commit()
        return {
            "updated": True,
            "id": int(trade_id),
            "symbol": row["symbol"],
            "pnl": row["pnl"],
            "prior_exit_reason": row["exit_reason"],
            "exit_reason": "RECONCILIATION_MANUAL_EXIT",
        }
    finally:
        conn.close()


def clean_strategy_acceptance_sql(
    *,
    exit_reason_column: str = "exit_reason",
    id_column: str = "id",
) -> str:
    """SQL AND-clause excluding reconciliation/manual inventory closes."""
    reasons = ", ".join(f"'{r}'" for r in sorted(RECONCILIATION_EXIT_REASONS))
    ids = ", ".join(str(int(i)) for i in RECONCILIATION_TRADE_IDS if str(i).isdigit())
    return (
        f" AND UPPER(COALESCE({exit_reason_column}, '')) NOT IN ({reasons}) "
        f" AND COALESCE({id_column}, 0) NOT IN ({ids}) "
        f" AND COALESCE(json_extract(explainability_json, '$.acceptance_class'), '') "
        f" NOT IN ('RECONCILIATION_MANUAL_EXIT', 'RECONCILIATION')"
    )
