"""Explicit pre-repair vs post-repair validation boundary.

Historical trades stay in place. New execution after the atomic-OPEN /
SCALP-money-DB deploy is stamped against this cutoff so 60% acceptance
is never computed on mixed books.
"""

from __future__ import annotations

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


def read_validation_cutoff(db_path: str | Path) -> dict[str, Any] | None:
    path = str(db_path)
    if not Path(path).exists():
        return None
    conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True, timeout=5)
    try:
        row = conn.execute(
            f"SELECT label, cutoff_utc, git_sha, engine, note FROM {CUTOFF_TABLE} WHERE id=1"
        ).fetchone()
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
