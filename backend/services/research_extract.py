"""Versioned, read-only research extract of the production database.

Copies the tables a causal DAY / SCALP study needs into a separate SQLite file:
candidate rows and their learned state, realized closes, 1m..1d bars, the DAY
145-feature inference vectors, SCALP microstructure snapshots, order-flow bars,
the decision book tape and REST aggTrades. Every copied row is at or before
``now``; nothing is derived or labelled here. The source is opened read-only.

  python -m backend.services.research_extract --src mystic_trading.db --out /tmp/x.db --since 1790890875
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

RESEARCH_EXTRACT_VERSION = "RESEARCH_EXTRACT_V2"
BAR_LOOKBACK_SEC = 10 * 86400
VECTOR_LOOKBACK_SEC = 86400


@dataclass(frozen=True)
class TableSpec:
    table: str
    where: str
    bounds: str
    columns: tuple[str, ...] = ()


def _iso(epoch: float) -> str:
    return datetime.fromtimestamp(float(epoch), tz=timezone.utc).isoformat()


def _ohlcv_ts(epoch: float) -> str:
    return datetime.fromtimestamp(float(epoch), tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S.%f")


INFERENCE_COLUMNS: tuple[str, ...] = (
    "id",
    "decision_id",
    "strategy_id",
    "symbol",
    "ts_utc",
    "prob_buy",
    "prob_hold",
    "confidence",
    "buy_margin",
    "ctx_json",
    "feature_version",
    "feature_dim",
    "features_json",
    "model_artifact",
)


def table_specs() -> tuple[TableSpec, ...]:
    """What is copied and the time column that bounds it.

    ``bounds`` names the window: ``bars`` reaches ``BAR_LOOKBACK_SEC`` before
    ``since`` so indicator windows are complete; ``vectors`` one day before;
    ``window`` is ``since``..``now``; ``all`` is the whole (small) table.
    """
    return (
        TableSpec("feature_ohlcv", "ts >= :bars_ts AND ts <= :now_ts AND interval IN ('1m','5m','15m','1h','4h','1d')", "bars"),
        TableSpec("adaptive_candidate_markouts", "evaluated_at >= :since AND evaluated_at <= :now", "window"),
        TableSpec("adaptive_metric_state", "1=1", "all"),
        TableSpec("adaptive_linear_model", "1=1", "all"),
        TableSpec("paper_trades", "timestamp >= :since_iso AND timestamp <= :now_iso", "window"),
        TableSpec("trade_learning_outcomes", "entry_timestamp >= :since AND entry_timestamp <= :now", "window"),
        TableSpec("ai_position_heartbeats", "ts_utc >= :since_iso AND ts_utc <= :now_iso", "window"),
        TableSpec("continuation_learning_meta", "1=1", "all"),
        TableSpec("continuation_advantage_log", "1=1", "all"),
        TableSpec(
            "ai_inference_log",
            "LOWER(COALESCE(strategy_id,''))='day' AND feature_dim=145 AND ts_utc >= :vectors_iso AND ts_utc <= :now_iso",
            "vectors",
            INFERENCE_COLUMNS,
        ),
        TableSpec("microstructure_feature_snapshots", "ts_utc >= :vectors AND ts_utc <= :now", "vectors"),
        TableSpec("day_order_flow_bars", "bar_open_epoch >= :vectors AND bar_open_epoch <= :now", "vectors"),
        TableSpec("decision_book_tape", "ts_utc >= :vectors_iso AND ts_utc <= :now_iso", "vectors"),
        TableSpec("day_agg_trades", "trade_time_ms >= :vectors_ms AND trade_time_ms <= :now_ms", "vectors"),
        TableSpec("book_queue_chunks", "chunk_start >= :vectors AND chunk_start <= :now", "vectors"),
    )


def window_params(since: float, now: float) -> dict[str, Any]:
    bars = float(since) - BAR_LOOKBACK_SEC
    vectors = float(since) - VECTOR_LOOKBACK_SEC
    return {
        "since": float(since),
        "now": float(now),
        "since_iso": _iso(since),
        "now_iso": _iso(now),
        "bars_ts": _ohlcv_ts(bars),
        "now_ts": _ohlcv_ts(now),
        "vectors": vectors,
        "vectors_iso": _iso(vectors),
        "vectors_ms": int(vectors * 1000),
        "now_ms": int(float(now) * 1000),
    }


def _exists(conn: sqlite3.Connection, schema: str, table: str) -> bool:
    return conn.execute(f"SELECT 1 FROM {schema}.sqlite_master WHERE type='table' AND name=?", (table,)).fetchone() is not None


def _create(conn: sqlite3.Connection, spec: TableSpec) -> None:
    if spec.columns:
        cols = ",".join(spec.columns)
        conn.execute(f"CREATE TABLE ext.{spec.table} AS SELECT {cols} FROM main.{spec.table} WHERE 0")
        return
    ddl = conn.execute("SELECT sql FROM main.sqlite_master WHERE type='table' AND name=?", (spec.table,)).fetchone()[0]
    _head, _, rest = ddl.partition("(")
    conn.execute(f"CREATE TABLE ext.{spec.table} ({rest}")


def _copy_indexes(conn: sqlite3.Connection, table: str) -> int:
    made = 0
    for name, sql in conn.execute("SELECT name, sql FROM main.sqlite_master WHERE type='index' AND tbl_name=? AND sql IS NOT NULL", (table,)).fetchall():
        body = sql.split(" ON ", 1)
        if len(body) != 2:
            continue
        unique = "UNIQUE " if sql.upper().startswith("CREATE UNIQUE") else ""
        try:
            conn.execute(f"CREATE {unique}INDEX IF NOT EXISTS ext.{name} ON {body[1]}")
            made += 1
        except sqlite3.Error:
            continue
    return made


def schema_hash(conn: sqlite3.Connection, schema: str, table: str) -> str:
    cols = [(r[1], r[2]) for r in conn.execute(f"PRAGMA {schema}.table_info({table})")]
    return hashlib.sha256(json.dumps(cols).encode()).hexdigest()[:16]


def build_research_extract(src_db: str | Path, out_db: str | Path, *, since: float, now: float | None = None) -> dict[str, Any]:
    """Copy ``table_specs`` rows inside the window into a fresh ``out_db``."""
    moment = float(now if now is not None else time.time())
    out = Path(out_db)
    if out.exists():
        out.unlink()
    params = window_params(since, moment)
    conn = sqlite3.connect(f"file:{Path(src_db)}?mode=ro", uri=True)
    report: dict[str, Any] = {"version": RESEARCH_EXTRACT_VERSION, "since": float(since), "now": moment, "tables": {}}
    try:
        conn.execute("ATTACH DATABASE ? AS ext", (f"file:{out}?mode=rwc",))
        for spec in table_specs():
            if not _exists(conn, "main", spec.table):
                report["tables"][spec.table] = {"status": "missing"}
                continue
            _create(conn, spec)
            cols = ",".join(spec.columns) if spec.columns else "*"
            conn.execute(f"INSERT INTO ext.{spec.table} SELECT {cols} FROM main.{spec.table} WHERE {spec.where}", params)
            conn.commit()
            rows = int(conn.execute(f"SELECT COUNT(*) FROM ext.{spec.table}").fetchone()[0])
            report["tables"][spec.table] = {"rows": rows, "bounds": spec.bounds, "schema": schema_hash(conn, "ext", spec.table), "indexes": _copy_indexes(conn, spec.table)}
        conn.execute("CREATE TABLE ext.research_extract_meta (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
        meta = {"version": RESEARCH_EXTRACT_VERSION, "since": params["since"], "now": moment, "params": params, "tables": report["tables"], "created_at": _iso(time.time())}
        conn.executemany("INSERT INTO ext.research_extract_meta (key, value) VALUES (?, ?)", [(k, json.dumps(v, default=str)) for k, v in meta.items()])
        conn.commit()
    finally:
        conn.close()
    report["bytes"] = out.stat().st_size
    return report


def extract_meta(db_path: str | Path) -> dict[str, Any]:
    conn = sqlite3.connect(f"file:{Path(db_path)}?mode=ro", uri=True)
    try:
        return {k: json.loads(v) for k, v in conn.execute("SELECT key, value FROM research_extract_meta")}
    finally:
        conn.close()


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Versioned read-only research extract")
    parser.add_argument("--src", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--since", type=float, required=True)
    parser.add_argument("--now", type=float, default=None)
    args = parser.parse_args(argv)
    print(json.dumps(build_research_extract(args.src, args.out, since=args.since, now=args.now), indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "BAR_LOOKBACK_SEC",
    "INFERENCE_COLUMNS",
    "RESEARCH_EXTRACT_VERSION",
    "VECTOR_LOOKBACK_SEC",
    "TableSpec",
    "build_research_extract",
    "extract_meta",
    "schema_hash",
    "table_specs",
    "window_params",
]
