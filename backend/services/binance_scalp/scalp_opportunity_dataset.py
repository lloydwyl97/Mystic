"""Persist every SCALP decision-cycle opportunity, including rejects.

Forward labels are filled later from live mids / 1m bars. Never executes.
"""

from __future__ import annotations

import json
import sqlite3
import time
from datetime import datetime, timezone
from typing import Any

TABLE = "scalp_opportunity_snapshots"
HORIZONS_SEC = (30, 60, 180, 300, 600, 1200)

_CREATE = f"""
CREATE TABLE IF NOT EXISTS {TABLE} (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at TEXT NOT NULL,
    epoch REAL NOT NULL,
    symbol TEXT NOT NULL,
    mid REAL,
    spread_pct REAL,
    impact_pct REAL,
    regime TEXT,
    best_setup TEXT,
    best_passed INTEGER,
    best_reject TEXT,
    rank_score REAL,
    measurements_json TEXT NOT NULL DEFAULT '{{}}',
    signals_json TEXT NOT NULL DEFAULT '[]',
    feature_vector_json TEXT NOT NULL DEFAULT '[]',
    plus_30s_net REAL,
    plus_60s_net REAL,
    plus_180s_net REAL,
    plus_300s_net REAL,
    plus_600s_net REAL,
    plus_1200s_net REAL,
    plus_30s_mfe REAL,
    plus_60s_mfe REAL,
    plus_180s_mfe REAL,
    plus_300s_mfe REAL,
    plus_600s_mfe REAL,
    plus_1200s_mfe REAL,
    plus_30s_mae REAL,
    plus_60s_mae REAL,
    plus_180s_mae REAL,
    plus_300s_mae REAL,
    plus_600s_mae REAL,
    plus_1200s_mae REAL,
    labeled INTEGER NOT NULL DEFAULT 0
)
"""


def ensure_opportunity_table(db_path: str, conn: sqlite3.Connection | None = None) -> None:
    def _apply(c: sqlite3.Connection) -> None:
        c.execute(_CREATE)
        c.execute(f"CREATE INDEX IF NOT EXISTS ix_scalp_opp_epoch ON {TABLE}(epoch, symbol)")

    if conn is not None:
        _apply(conn)
        return
    with sqlite3.connect(db_path, timeout=10) as owned:
        _apply(owned)
        owned.commit()


def record_opportunity_cycle(
    db_path: str,
    *,
    rows: list[dict[str, Any]],
    epoch: float | None = None,
    cost_pct: float = 0.0006,
    conn: sqlite3.Connection | None = None,
) -> int:
    """Persist one evaluate_all() cycle.

    When ``conn`` is the paper-engine tick connection (BEGIN IMMEDIATE),
    reuse it. A second sqlite writer against the same file deadlocks under
    that lock and the snapshot never lands.
    """
    if not rows:
        return 0
    ensure_opportunity_table(db_path, conn=conn)
    now = datetime.now(timezone.utc).isoformat()
    ts = float(epoch if epoch is not None else time.time())
    written = 0
    owned = conn is None
    writer = conn if conn is not None else sqlite3.connect(db_path, timeout=10)
    try:
        for row in rows:
            snap = row.get("snap")
            mid = float(getattr(snap, "mid", 0) or row.get("mid") or 0)
            spread = float(getattr(snap, "spread_pct", 0) or row.get("spread_pct") or 0)
            meta = row.get("rank_meta") or {}
            writer.execute(
                f"""
                INSERT INTO {TABLE}
                (created_at, epoch, symbol, mid, spread_pct, impact_pct, regime,
                 best_setup, best_passed, best_reject, rank_score,
                 measurements_json, signals_json, feature_vector_json)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                """,
                (
                    now,
                    ts,
                    str(row.get("symbol") or ""),
                    mid,
                    spread,
                    float(meta.get("impact_pct") or 0),
                    str(meta.get("regime") or ""),
                    str(row.get("best_setup") or meta.get("best_setup") or ""),
                    1 if (row.get("strategy_passed") or meta.get("strategy_passed")) else 0,
                    str(row.get("soft_reason") or meta.get("soft_reason") or meta.get("hard_block") or ""),
                    float(row.get("rank_score") or 0),
                    json.dumps(meta.get("setup_measurements") or {}, default=str),
                    json.dumps(row.get("all_signals") or [], default=str)[:8000],
                    json.dumps((row.get("rank_meta") or {}).get("feature_vector") or [], default=str)[:4000],
                ),
            )
            written += 1
        if owned:
            writer.commit()
    finally:
        if owned:
            writer.close()
    _ = cost_pct
    return written


def label_due_opportunities(db_path: str, reader: Any, *, now_epoch: float | None = None, cost_pct: float = 0.0006) -> int:
    ensure_opportunity_table(db_path)
    now = float(now_epoch if now_epoch is not None else time.time())
    conn = sqlite3.connect(db_path, timeout=10)
    conn.row_factory = sqlite3.Row
    labeled = 0
    try:
        rows = list(conn.execute(f"SELECT * FROM {TABLE} WHERE labeled=0 ORDER BY id DESC LIMIT 80"))
        for row in rows:
            age = now - float(row["epoch"] or 0)
            mid0 = float(row["mid"] or 0)
            if mid0 <= 0:
                continue
            snap = None
            try:
                snap = reader.read(str(row["symbol"]))
            except Exception:
                snap = None
            if snap is None:
                continue
            mid = float(getattr(snap, "mid", 0) or 0)
            if mid <= 0:
                continue
            gross = (mid - mid0) / mid0
            net = gross - cost_pct
            sets = []
            vals: list[Any] = []
            col = {30: "plus_30s", 60: "plus_60s", 180: "plus_180s", 300: "plus_300s", 600: "plus_600s", 1200: "plus_1200s"}
            for sec, prefix in col.items():
                if row[f"{prefix}_net"] is None and age >= sec:
                    sets.append(f"{prefix}_net=?")
                    vals.append(net)
                    sets.append(f"{prefix}_mfe=?")
                    vals.append(max(0.0, gross))
                    sets.append(f"{prefix}_mae=?")
                    vals.append(min(0.0, gross))
            complete = age >= 1200 and all(row[f"{col[s]}_net"] is not None or age >= s for s in HORIZONS_SEC)
            if complete:
                sets.append("labeled=1")
            if not sets:
                continue
            vals.append(row["id"])
            conn.execute(f"UPDATE {TABLE} SET {', '.join(sets)} WHERE id=?", vals)
            labeled += 1
        conn.commit()
    finally:
        conn.close()
    return labeled
