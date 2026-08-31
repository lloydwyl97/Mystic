"""Persist every SCALP decision-cycle opportunity, including rejects.

Forward labels are filled later from live mids / 1m bars. Never executes.
"""

from __future__ import annotations

import contextlib
import json
import sqlite3
import time
from datetime import datetime, timezone
from typing import Any

TABLE = "scalp_opportunity_snapshots"
HORIZONS_SEC = (30, 60, 180, 300, 600, 1200)
_SIGNAL_KEYS = (
    "setup_name",
    "passed",
    "reject_reason",
    "score",
    "confidence",
    "expected_move_pct",
    "required_target_pct",
    "spread_pct",
    "impact_pct",
    "entry_reason",
)


def compact_signals_json(signals: Any) -> str:
    """Always-valid reduced signal schema. Never character-truncate JSON."""
    rows: list[dict[str, Any]] = []
    if isinstance(signals, list):
        source = signals
    elif isinstance(signals, dict):
        source = [signals]
    else:
        source = []
    for item in source:
        row = item.as_dict() if hasattr(item, "as_dict") else item
        if not isinstance(row, dict):
            continue
        rows.append({key: row.get(key) for key in _SIGNAL_KEYS})
    return json.dumps(rows, separators=(",", ":"), default=str)


def compact_feature_vector_json(vector: Any) -> str:
    """Always-valid numeric vector. Never character-truncate JSON."""
    if not isinstance(vector, list):
        return "[]"
    out: list[float] = []
    for value in vector:
        try:
            out.append(float(value))
        except (TypeError, ValueError):
            out.append(0.0)
    return json.dumps(out, separators=(",", ":"))


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
    labeled INTEGER NOT NULL DEFAULT 0,
    horizon_labels_json TEXT NOT NULL DEFAULT '{{}}',
    predicted_net_ev REAL,
    predicted_prob_positive_net REAL,
    predicted_mfe REAL,
    predicted_mae REAL,
    predicted_horizon TEXT,
    model_version TEXT NOT NULL DEFAULT '',
    peer_micro_json TEXT NOT NULL DEFAULT '{{}}'
)
"""


def ensure_opportunity_table(db_path: str, conn: sqlite3.Connection | None = None) -> None:
    def _apply(c: sqlite3.Connection) -> None:
        c.execute(_CREATE)
        c.execute(f"CREATE INDEX IF NOT EXISTS ix_scalp_opp_epoch ON {TABLE}(epoch, symbol)")
        cols = {str(r[1]) for r in c.execute(f"PRAGMA table_info({TABLE})")}
        if "horizon_labels_json" not in cols:
            c.execute(f"ALTER TABLE {TABLE} ADD COLUMN horizon_labels_json TEXT NOT NULL DEFAULT '{{}}'")
        for col, spec in (
            ("predicted_net_ev", "REAL"),
            ("predicted_prob_positive_net", "REAL"),
            ("predicted_mfe", "REAL"),
            ("predicted_mae", "REAL"),
            ("predicted_horizon", "TEXT"),
            ("model_version", "TEXT NOT NULL DEFAULT ''"),
            ("peer_micro_json", "TEXT NOT NULL DEFAULT '{}'"),
        ):
            if col not in cols:
                c.execute(f"ALTER TABLE {TABLE} ADD COLUMN {col} {spec}")

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
                 measurements_json, signals_json, feature_vector_json,
                 predicted_net_ev, predicted_prob_positive_net, predicted_mfe,
                 predicted_mae, predicted_horizon, model_version, peer_micro_json)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
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
                    compact_signals_json(row.get("all_signals") or []),
                    compact_feature_vector_json((row.get("rank_meta") or {}).get("feature_vector") or []),
                    row.get("expected_net_ev"),
                    row.get("predicted_prob_positive_net"),
                    row.get("expected_mfe"),
                    row.get("expected_mae"),
                    row.get("expected_hold"),
                    str(row.get("forward_net_model_version") or ""),
                    json.dumps(row.get("peer_micro_snapshot") or {}, default=str),
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


def _path_label(mid0: float, epoch: float, horizon_sec: int, bars: list[dict[str, Any]], *, cost_pct: float, target_pct: float = 0.0025) -> dict[str, Any]:
    end_ts = epoch + horizon_sec
    highs: list[float] = []
    lows: list[float] = []
    last_close = mid0
    time_to_target = None
    time_to_adverse = None
    for b in bars:
        ts = float(b.get("ts") or 0)
        if ts < epoch or ts > end_ts:
            continue
        high = float(b.get("high") or 0)
        low = float(b.get("low") or 0)
        close = float(b.get("close") or 0)
        if high > 0:
            highs.append(high)
        if low > 0:
            lows.append(low)
        if close > 0:
            last_close = close
        age = ts - epoch
        if time_to_target is None and mid0 > 0 and high > 0 and (high - mid0) / mid0 >= target_pct:
            time_to_target = age
        if time_to_adverse is None and mid0 > 0 and low > 0 and (low - mid0) / mid0 <= -target_pct:
            time_to_adverse = age
    mfe = max(((h - mid0) / mid0) for h in highs) if highs and mid0 > 0 else 0.0
    mae = min(((lo - mid0) / mid0) for lo in lows) if lows and mid0 > 0 else 0.0
    gross = (last_close - mid0) / mid0 if mid0 > 0 else 0.0
    return {
        "gross": gross,
        "net": gross - cost_pct,
        "mfe": mfe,
        "mae": mae,
        "hit_target": bool(mfe >= target_pct),
        "time_to_target": time_to_target,
        "time_to_adverse": time_to_adverse,
    }


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
            bars: list[dict[str, Any]] = []
            with contextlib.suppress(Exception):
                from backend.services.binance_scalp.strategies.kline_cache import fetch_1m_bars

                bars = fetch_1m_bars(str(row["symbol"]), minutes=40) or []
            existing_labels: dict[str, Any] = {}
            with contextlib.suppress(Exception):
                existing_labels = json.loads(row["horizon_labels_json"] or "{}") if "horizon_labels_json" in row else {}
            gross = (mid - mid0) / mid0
            net = gross - cost_pct
            sets = []
            vals: list[Any] = []
            col = {30: "plus_30s", 60: "plus_60s", 180: "plus_180s", 300: "plus_300s", 600: "plus_600s", 1200: "plus_1200s"}
            labels = dict(existing_labels)
            for sec, prefix in col.items():
                if row[f"{prefix}_net"] is None and age >= sec:
                    path = (
                        _path_label(mid0, float(row["epoch"] or 0), sec, bars, cost_pct=cost_pct)
                        if bars
                        else {
                            "gross": gross,
                            "net": net,
                            "mfe": max(0.0, gross),
                            "mae": min(0.0, gross),
                            "hit_target": gross >= 0.0025,
                            "time_to_target": None,
                            "time_to_adverse": None,
                        }
                    )
                    sets.append(f"{prefix}_net=?")
                    vals.append(path["net"])
                    sets.append(f"{prefix}_mfe=?")
                    vals.append(path["mfe"])
                    sets.append(f"{prefix}_mae=?")
                    vals.append(path["mae"])
                    labels[str(sec)] = path
            if labels != existing_labels:
                sets.append("horizon_labels_json=?")
                vals.append(json.dumps(labels, default=str))
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
