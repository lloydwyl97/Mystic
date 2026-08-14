"""Capture post-exit prices for every new SCALP close.

Offsets: +30s, +1m, +3m, +5m, +max-hold horizon (20m).
Does not change entry/exit decisions. Fills from the live market reader.
"""

from __future__ import annotations

import contextlib
import sqlite3
from datetime import datetime, timezone
from typing import Any

OFFSETS_SEC = (30, 60, 180, 300, 1200)
TABLE = "scalp_post_exit_path"

_CREATE = f"""
CREATE TABLE IF NOT EXISTS {TABLE} (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    trade_id TEXT NOT NULL UNIQUE,
    symbol TEXT NOT NULL,
    setup TEXT NOT NULL DEFAULT '',
    exit_reason TEXT NOT NULL DEFAULT '',
    exit_ts TEXT NOT NULL,
    exit_epoch REAL NOT NULL,
    entry_price REAL,
    exit_price REAL,
    plus_30s REAL,
    plus_60s REAL,
    plus_180s REAL,
    plus_300s REAL,
    plus_1200s REAL,
    hit_target_after INTEGER NOT NULL DEFAULT 0,
    hit_target_after_sec REAL,
    complete INTEGER NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
)
"""


def ensure_post_exit_path_table(db_path: str) -> None:
    with sqlite3.connect(db_path, timeout=10) as conn:
        conn.execute(_CREATE)
        conn.commit()


def schedule_post_exit_path(
    db_path: str,
    *,
    trade_id: str,
    symbol: str,
    setup: str,
    exit_reason: str,
    exit_ts: str,
    exit_epoch: float,
    entry_price: float,
    exit_price: float,
) -> None:
    ensure_post_exit_path_table(db_path)
    with sqlite3.connect(db_path, timeout=10) as conn:
        conn.execute(
            f"""
            INSERT OR IGNORE INTO {TABLE}
            (trade_id, symbol, setup, exit_reason, exit_ts, exit_epoch, entry_price, exit_price)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (trade_id, symbol, setup, exit_reason, exit_ts, exit_epoch, entry_price, exit_price),
        )
        conn.commit()


def fill_due_post_exit_paths(db_path: str, reader: Any, *, now_epoch: float | None = None) -> int:
    ensure_post_exit_path_table(db_path)
    now = float(now_epoch if now_epoch is not None else datetime.now(timezone.utc).timestamp())
    conn = sqlite3.connect(db_path, timeout=10)
    conn.row_factory = sqlite3.Row
    filled = 0
    try:
        rows = list(conn.execute(f"SELECT * FROM {TABLE} WHERE complete=0"))
        for row in rows:
            exit_epoch = float(row["exit_epoch"] or 0)
            age = now - exit_epoch
            updates: dict[str, float] = {}
            col_by_off = {
                30: "plus_30s",
                60: "plus_60s",
                180: "plus_180s",
                300: "plus_300s",
                1200: "plus_1200s",
            }
            due = False
            for off, col in col_by_off.items():
                if row[col] is None and age >= off:
                    due = True
            if not due:
                continue
            snap = None
            with contextlib.suppress(Exception):
                snap = reader.read(str(row["symbol"]))
            if snap is None:
                continue
            mid = float(getattr(snap, "mid", 0) or 0)
            if mid <= 0:
                continue
            sets = []
            vals: list[Any] = []
            for off, col in col_by_off.items():
                if row[col] is None and age >= off:
                    sets.append(f"{col}=?")
                    vals.append(mid)
            entry = float(row["entry_price"] or 0)
            hit = int(row["hit_target_after"] or 0)
            hit_sec = row["hit_target_after_sec"]
            if entry > 0 and mid >= entry * 1.0025 and not hit:
                sets.append("hit_target_after=1")
                sets.append("hit_target_after_sec=?")
                vals.append(age)
            complete = all((row[col] is not None or age >= off) for off, col in col_by_off.items())
            if complete and age >= 1200:
                sets.append("complete=1")
            if not sets:
                continue
            vals.append(row["id"])
            conn.execute(f"UPDATE {TABLE} SET {', '.join(sets)} WHERE id=?", vals)
            filled += 1
        conn.commit()
    finally:
        conn.close()
    return filled
