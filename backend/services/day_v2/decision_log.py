"""Persist one DAY V2 evaluation result per symbol per cycle."""

from __future__ import annotations

import json
import sqlite3
import time
from pathlib import Path


def record_day_decision(
    db_path: str | Path,
    symbol: str,
    result: str,
    *,
    cycle_ts: float | None = None,
    closest: str = "",
    unmet: list[str] | None = None,
) -> None:
    conn = sqlite3.connect(str(db_path), timeout=30)
    try:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS day_v2_decisions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                cycle_ts REAL NOT NULL,
                result TEXT NOT NULL,
                closest TEXT NOT NULL DEFAULT '',
                unmet_json TEXT NOT NULL DEFAULT '[]'
            )
            """
        )
        conn.execute(
            "INSERT INTO day_v2_decisions(symbol, cycle_ts, result, closest, unmet_json) VALUES (?,?,?,?,?)",
            (str(symbol), float(cycle_ts if cycle_ts is not None else time.time()), str(result), str(closest or ""), json.dumps(unmet or [])),
        )
        conn.commit()
    finally:
        conn.close()


def reason_counts(db_path: str | Path, *, since_ts: float) -> dict[str, dict[str, int]]:
    conn = sqlite3.connect(str(db_path), timeout=30)
    try:
        rows = conn.execute(
            "SELECT symbol, result, closest, COUNT(*) FROM day_v2_decisions WHERE cycle_ts>=? GROUP BY symbol, result, closest",
            (float(since_ts),),
        ).fetchall()
    except sqlite3.OperationalError:
        return {}
    finally:
        conn.close()
    out: dict[str, dict[str, int]] = {}
    for sym, result, closest, n in rows:
        label = str(result)
        if closest and result == "NO_SIGNAL":
            label = f"NO_SIGNAL:{closest}"
        bucket = out.setdefault(str(sym), {})
        bucket[label] = bucket.get(label, 0) + int(n)
    return out
