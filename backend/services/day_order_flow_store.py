"""Persistence for tape-derived DAY order flow.

`day_order_flow_bars` holds one row per symbol per 15-minute bar on the same
grid as the DAY decision clock, so it joins to `ai_inference_log` and to the
candidate matrix on symbol and bar time.

Write-only capture. No live model or exit path reads this table; it exists so a
future entry artifact can be trained on real signed volume instead of the
`sign(close - open)` proxy that is correctly zeroed today.

Upserts are keyed on (symbol, bar_open_epoch), so re-running the collector over
an overlapping tape window is safe and refreshes partially-filled bars.
"""

from __future__ import annotations

import logging
import sqlite3
import time
from datetime import datetime, timezone
from typing import Any

from backend.database_schema import DATABASE_PATH

logger = logging.getLogger(__name__)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS day_order_flow_bars (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol TEXT NOT NULL,
    bar_open_epoch INTEGER NOT NULL,
    bar_sec INTEGER NOT NULL DEFAULT 900,
    ts_utc TEXT NOT NULL,
    trade_count INTEGER NOT NULL DEFAULT 0,
    buy_qty REAL NOT NULL DEFAULT 0,
    sell_qty REAL NOT NULL DEFAULT 0,
    buy_notional REAL NOT NULL DEFAULT 0,
    sell_notional REAL NOT NULL DEFAULT 0,
    cvd_qty REAL NOT NULL DEFAULT 0,
    cvd_notional REAL NOT NULL DEFAULT 0,
    imbalance REAL NOT NULL DEFAULT 0,
    notional_imbalance REAL NOT NULL DEFAULT 0,
    vwap REAL NOT NULL DEFAULT 0,
    first_price REAL NOT NULL DEFAULT 0,
    last_price REAL NOT NULL DEFAULT 0,
    first_print_epoch REAL NOT NULL DEFAULT 0,
    last_print_epoch REAL NOT NULL DEFAULT 0,
    coverage_sec REAL NOT NULL DEFAULT 0,
    flow_version INTEGER NOT NULL DEFAULT 1,
    updated_at_utc TEXT NOT NULL,
    UNIQUE(symbol, bar_open_epoch)
);
CREATE INDEX IF NOT EXISTS ix_day_flow_bar ON day_order_flow_bars(bar_open_epoch);
CREATE INDEX IF NOT EXISTS ix_day_flow_sym_bar ON day_order_flow_bars(symbol, bar_open_epoch);
"""

_COLUMNS = (
    "symbol",
    "bar_open_epoch",
    "bar_sec",
    "trade_count",
    "buy_qty",
    "sell_qty",
    "buy_notional",
    "sell_notional",
    "cvd_qty",
    "cvd_notional",
    "imbalance",
    "notional_imbalance",
    "vwap",
    "first_price",
    "last_price",
    "first_print_epoch",
    "last_print_epoch",
    "coverage_sec",
    "flow_version",
)

_tables_ready = False


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def ensure_order_flow_tables(db_path: str = DATABASE_PATH) -> None:
    global _tables_ready
    if _tables_ready:
        return
    try:
        with sqlite3.connect(db_path, timeout=10) as conn:
            conn.executescript(_SCHEMA)
            conn.commit()
        _tables_ready = True
    except sqlite3.Error:
        logger.warning("day_order_flow_bars schema init failed", exc_info=True)


def upsert_bar_rows(rows: list[dict[str, Any]], db_path: str = DATABASE_PATH) -> int:
    """Insert or refresh bar-flow rows. Returns rows written."""
    if not rows:
        return 0
    ensure_order_flow_tables(db_path)
    now = _now_iso()
    placeholders = ",".join("?" for _ in _COLUMNS)
    updates = ",".join(f"{c}=excluded.{c}" for c in _COLUMNS if c not in ("symbol", "bar_open_epoch"))
    sql = (
        f"INSERT INTO day_order_flow_bars ({','.join(_COLUMNS)}, ts_utc, updated_at_utc) "
        f"VALUES ({placeholders},?,?) "
        f"ON CONFLICT(symbol, bar_open_epoch) DO UPDATE SET {updates}, updated_at_utc=excluded.updated_at_utc"
    )
    payload = []
    for r in rows:
        bar_open = int(r.get("bar_open_epoch") or 0)
        if bar_open <= 0 or not r.get("symbol"):
            continue
        bar_iso = datetime.fromtimestamp(bar_open, tz=timezone.utc).isoformat()
        payload.append((*tuple(r.get(c) for c in _COLUMNS), bar_iso, now))
    if not payload:
        return 0
    try:
        with sqlite3.connect(db_path, timeout=20) as conn:
            conn.executemany(sql, payload)
            conn.commit()
        return len(payload)
    except sqlite3.Error:
        logger.warning("day_order_flow_bars upsert failed", exc_info=True)
        return 0


def collect_symbol(
    symbol: str,
    *,
    lookback_sec: float,
    now: float | None = None,
    db_path: str = DATABASE_PATH,
    client: Any | None = None,
) -> int:
    """Read the retained tape for one symbol and persist its bar flow."""
    from backend.services.day_order_flow_tape import bar_flow_rows

    now = float(now if now is not None else time.time())
    rows = bar_flow_rows(
        symbol,
        since_ts=now - float(lookback_sec),
        until_ts=now,
        client=client,
    )
    return upsert_bar_rows(rows, db_path=db_path)


def coverage_summary(db_path: str = DATABASE_PATH) -> list[dict[str, Any]]:
    """Per-symbol row counts and bar range, for verifying capture is alive."""
    ensure_order_flow_tables(db_path)
    try:
        with sqlite3.connect(db_path, timeout=10) as conn:
            cur = conn.execute(
                """
                SELECT symbol, COUNT(*), MIN(bar_open_epoch), MAX(bar_open_epoch),
                       SUM(trade_count), AVG(imbalance),
                       SUM(CASE WHEN trade_count > 0 THEN 1 ELSE 0 END)
                FROM day_order_flow_bars GROUP BY symbol ORDER BY symbol
                """
            ).fetchall()
    except sqlite3.Error:
        logger.warning("day_order_flow_bars coverage query failed", exc_info=True)
        return []
    return [
        {
            "symbol": r[0],
            "bars": int(r[1] or 0),
            "first_bar_epoch": int(r[2] or 0),
            "last_bar_epoch": int(r[3] or 0),
            "prints": int(r[4] or 0),
            "avg_imbalance": float(r[5] or 0.0),
            "bars_with_prints": int(r[6] or 0),
        }
        for r in cur
    ]
