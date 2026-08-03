"""
SQLite runtime helpers for live write reliability.

Scope: runtime stability only (no schema/strategy changes).
"""

from __future__ import annotations

import contextlib
import os
import random
import sqlite3
import time
from collections.abc import Callable
from pathlib import Path
from typing import TypeVar

T = TypeVar("T")


def _db_timeout_sec() -> float:
    try:
        return float(os.getenv("SQLITE_CONNECT_TIMEOUT_SEC", "10"))
    except (TypeError, ValueError):
        return 10.0


def _busy_timeout_ms() -> int:
    try:
        return int(float(os.getenv("SQLITE_BUSY_TIMEOUT_MS", "10000")))
    except (TypeError, ValueError):
        return 10000


def _locked_retries() -> int:
    try:
        return max(1, int(os.getenv("SQLITE_LOCK_RETRIES", "5")))
    except (TypeError, ValueError):
        return 5


def _base_backoff_sec() -> float:
    try:
        return max(0.01, float(os.getenv("SQLITE_LOCK_RETRY_BASE_SEC", "0.05")))
    except (TypeError, ValueError):
        return 0.05


def is_locked_error(exc: BaseException) -> bool:
    if not isinstance(exc, sqlite3.OperationalError):
        return False
    msg = str(exc).lower()
    return "database is locked" in msg or "database table is locked" in msg


_wal_ready: set[str] = set()


def ensure_wal(db_path: str | Path) -> None:
    """Set WAL once per process — avoid journal_mode churn on every hot connect."""
    key = str(db_path)
    if key in _wal_ready:
        return
    conn = sqlite3.connect(key, timeout=_db_timeout_sec())
    try:
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA synchronous=NORMAL;")
        conn.execute(f"PRAGMA busy_timeout={_busy_timeout_ms()};")
        _wal_ready.add(key)
    finally:
        conn.close()


def connect_rw(db_path: str | Path) -> sqlite3.Connection:
    """Read-write connection for portfolio writer / mutation paths."""
    ensure_wal(db_path)
    conn = sqlite3.connect(str(db_path), timeout=_db_timeout_sec())
    conn.execute("PRAGMA synchronous=NORMAL;")
    conn.execute("PRAGMA foreign_keys=ON;")
    conn.execute(f"PRAGMA busy_timeout={_busy_timeout_ms()};")
    return conn


def connect_ro(db_path: str | Path, *, timeout_sec: float | None = None) -> sqlite3.Connection:
    """Short-lived read connection for GET/dashboard routes — no write txn, no WAL rewrite."""
    timeout = _db_timeout_sec() if timeout_sec is None else max(0.1, float(timeout_sec))
    path = str(db_path)
    try:
        conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True, timeout=timeout)
    except sqlite3.OperationalError:
        conn = sqlite3.connect(path, timeout=timeout)
        with contextlib.suppress(sqlite3.Error):
            conn.execute("PRAGMA query_only=ON;")
    conn.execute(f"PRAGMA busy_timeout={int(timeout * 1000)};")
    return conn


def run_locked_retry(
    op: Callable[[], T],
    *,
    max_attempts: int | None = None,
) -> T:
    attempts = max_attempts or _locked_retries()
    base = _base_backoff_sec()
    for i in range(attempts):
        try:
            return op()
        except sqlite3.OperationalError as exc:
            if not is_locked_error(exc) or i >= attempts - 1:
                raise
            sleep_s = min(0.6, base * (2**i)) + random.uniform(0.0, 0.02)
            time.sleep(sleep_s)
    # Defensive: loop always returns or raises above.
    return op()
