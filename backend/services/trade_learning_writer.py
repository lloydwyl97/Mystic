"""
Mystic unified trade-learning writer.

This module is the SHARED learning-record sink that paper and live both
call when a trade closes. It does not contain any decision logic; it only
writes one normalized row per close into the existing learning tables so
the AI training pipeline can read paper and live with identical schema.

Why this exists:
  * Paper and live MUST learn from the same trade structure.
  * The trading mode MUST be persisted with every outcome so the trainer
    can weight or filter by paper vs live and never confuse historical
    paper-only artifacts with live decisions.
  * Dust outcomes MUST be recorded distinctly from full position closes
    so the model does not learn a fake "sold at X" for dust that live
    Binance.US would have refused.

This writer is intentionally tolerant of partial inputs (e.g., missing
fills after a HUMAN_MANUAL_SELL): it records the missing-data flag instead
of inventing a profit.
"""

from __future__ import annotations

import json
import logging
import sqlite3
import time
from dataclasses import asdict, dataclass, field
from typing import Any

from backend.config.trading_mode import (
    TradingMode,
    TradingModeError,
    resolve_trading_mode,
)
from backend.database_schema import DATABASE_PATH
from backend.utils.sqlite_runtime import connect_rw, run_locked_retry

logger = logging.getLogger(__name__)

TABLE_NAME = "trade_learning_outcomes"

_SCHEMA = f"""
CREATE TABLE IF NOT EXISTS {TABLE_NAME} (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    written_at_utc TEXT NOT NULL,
    mode TEXT NOT NULL,                       -- 'paper' or 'live'
    symbol TEXT NOT NULL,                     -- BTCUSDT / ETHUSDT / SOLUSDT / XRPUSDT
    entry_timestamp REAL,
    exit_timestamp REAL,
    entry_price REAL,
    exit_price REAL,                          -- NULL when unknown (e.g., HUMAN_MANUAL_SELL no-fill)
    quantity REAL,
    fees_paid REAL,
    slippage_cost REAL,
    net_profit_usd REAL,                      -- NULL when unknown
    net_profit_pct REAL,                      -- NULL when unknown
    hold_seconds REAL,
    decision_reason TEXT,
    confidence REAL,
    rank_data_json TEXT,                      -- ranking payload at entry
    indicators_at_entry_json TEXT,
    indicators_while_holding_json TEXT,
    indicators_at_sell_json TEXT,
    timeframes_used_json TEXT,
    cooldown_state_json TEXT,
    manual_sell_flag INTEGER NOT NULL DEFAULT 0,
    close_reason TEXT,                        -- AI_TAKE_PROFIT | HUMAN_MANUAL_SELL | DUST_WRITEOFF | ...
    dust_remaining_qty REAL,                  -- non-zero when dust left over
    dust_remaining_notional_usdt REAL,
    realized_profit_unknown INTEGER NOT NULL DEFAULT 0,
    extra_json TEXT
);

CREATE INDEX IF NOT EXISTS idx_{TABLE_NAME}_symbol_mode
    ON {TABLE_NAME}(symbol, mode, exit_timestamp);
CREATE INDEX IF NOT EXISTS idx_{TABLE_NAME}_close_reason
    ON {TABLE_NAME}(close_reason, mode, exit_timestamp);
"""


def _ensure_table(db_path: str = DATABASE_PATH) -> None:
    def _op() -> None:
        with connect_rw(db_path) as conn:
            conn.execute("BEGIN IMMEDIATE")
            cursor = conn.cursor()
            for stmt in _SCHEMA.strip().split(";\n"):
                stmt = stmt.strip()
                if stmt:
                    cursor.execute(stmt)
            conn.commit()

    run_locked_retry(_op)


def _resolve_mode_safe() -> str:
    try:
        return resolve_trading_mode().value
    except TradingModeError as exc:
        logger.warning("LEARNING_WRITER_MODE_UNRESOLVED %s", exc)
        return "unknown"


@dataclass
class TradeLearningRecord:
    """
    Normalized record for one closed trade. Both paper and live populate
    this same dataclass via ``record_trade_outcome``.
    """

    symbol: str
    entry_timestamp: float | None = None
    exit_timestamp: float | None = None
    entry_price: float | None = None
    exit_price: float | None = None
    quantity: float | None = None
    fees_paid: float | None = None
    slippage_cost: float | None = None
    net_profit_usd: float | None = None
    net_profit_pct: float | None = None
    hold_seconds: float | None = None
    decision_reason: str | None = None
    confidence: float | None = None
    rank_data: dict[str, Any] | None = None
    indicators_at_entry: dict[str, Any] | None = None
    indicators_while_holding: dict[str, Any] | None = None
    indicators_at_sell: dict[str, Any] | None = None
    timeframes_used: list[str] | None = None
    cooldown_state: dict[str, Any] | None = None
    manual_sell_flag: bool = False
    close_reason: str | None = None
    dust_remaining_qty: float = 0.0
    dust_remaining_notional_usdt: float = 0.0
    realized_profit_unknown: bool = False
    extra: dict[str, Any] = field(default_factory=dict)

    def normalize(self) -> TradeLearningRecord:
        if self.exit_timestamp is None:
            self.exit_timestamp = time.time()
        if self.entry_timestamp is not None and self.exit_timestamp is not None:
            if self.hold_seconds is None:
                self.hold_seconds = float(self.exit_timestamp - self.entry_timestamp)
        if self.exit_price is None or self.net_profit_usd is None:
            # If the operator (or live reconcile) handed us no fill info, we
            # MUST flag realized_profit_unknown so the trainer knows not to
            # invent a profit signal.
            self.realized_profit_unknown = True
        return self


def _json(value: Any) -> str | None:
    if value is None:
        return None
    try:
        return json.dumps(value, separators=(",", ":"), default=str)
    except (TypeError, ValueError):
        return json.dumps(str(value))


def record_trade_outcome(
    record: TradeLearningRecord,
    db_path: str = DATABASE_PATH,
    mode_override: TradingMode | None = None,
) -> bool:
    """
    Persist a unified trade-learning row for the trade that just closed.
    Returns True on success, False on database error.

    ``mode_override`` is intended for tests; production code lets the mode
    resolve from ``MYSTIC_TRADING_MODE``.
    """
    try:
        _ensure_table(db_path)
    except Exception as exc:
        logger.warning("LEARNING_WRITER_SCHEMA_FAILED err=%s", exc)
        return False

    record = record.normalize()
    mode_value = mode_override.value if mode_override is not None else _resolve_mode_safe()

    try:

        def _op() -> bool:
            with connect_rw(db_path) as conn:
                conn.execute("BEGIN IMMEDIATE")
                cursor = conn.cursor()
                cursor.execute(
                    f"""
                    INSERT INTO {TABLE_NAME} (
                        written_at_utc, mode, symbol,
                        entry_timestamp, exit_timestamp,
                        entry_price, exit_price, quantity,
                        fees_paid, slippage_cost,
                        net_profit_usd, net_profit_pct, hold_seconds,
                        decision_reason, confidence,
                        rank_data_json,
                        indicators_at_entry_json,
                        indicators_while_holding_json,
                        indicators_at_sell_json,
                        timeframes_used_json,
                        cooldown_state_json,
                        manual_sell_flag, close_reason,
                        dust_remaining_qty, dust_remaining_notional_usdt,
                        realized_profit_unknown, extra_json
                    ) VALUES (
                        datetime('now'), ?, ?,
                        ?, ?,
                        ?, ?, ?,
                        ?, ?,
                        ?, ?, ?,
                        ?, ?,
                        ?,
                        ?,
                        ?,
                        ?,
                        ?,
                        ?,
                        ?, ?,
                        ?, ?,
                        ?, ?
                    )
                    """,
                    (
                        mode_value,
                        record.symbol,
                        record.entry_timestamp,
                        record.exit_timestamp,
                        record.entry_price,
                        record.exit_price,
                        record.quantity,
                        record.fees_paid,
                        record.slippage_cost,
                        record.net_profit_usd,
                        record.net_profit_pct,
                        record.hold_seconds,
                        record.decision_reason,
                        record.confidence,
                        _json(record.rank_data),
                        _json(record.indicators_at_entry),
                        _json(record.indicators_while_holding),
                        _json(record.indicators_at_sell),
                        _json(record.timeframes_used),
                        _json(record.cooldown_state),
                        1 if record.manual_sell_flag else 0,
                        record.close_reason,
                        float(record.dust_remaining_qty or 0.0),
                        float(record.dust_remaining_notional_usdt or 0.0),
                        1 if record.realized_profit_unknown else 0,
                        _json(record.extra),
                    ),
                )
                conn.commit()
            return True

        run_locked_retry(_op)
    except Exception as exc:
        logger.warning(
            "LEARNING_WRITER_INSERT_FAILED symbol=%s mode=%s err=%s",
            record.symbol,
            mode_value,
            exc,
        )
        return False

    logger.info(
        "LEARNING_WRITE_OK mode=%s symbol=%s close_reason=%s manual=%s dust_qty=%s net_pct=%s",
        mode_value,
        record.symbol,
        record.close_reason,
        record.manual_sell_flag,
        record.dust_remaining_qty,
        record.net_profit_pct,
    )
    return True


def read_recent_learning_rows(
    symbol: str | None = None,
    mode: str | None = None,
    limit: int = 100,
    db_path: str = DATABASE_PATH,
) -> list[dict[str, Any]]:
    """Read recent learning rows. Used by trainers / dashboards."""
    try:
        _ensure_table(db_path)
    except Exception as exc:
        logger.warning("LEARNING_READER_SCHEMA_FAILED err=%s", exc)
        return []
    where: list[str] = []
    params: list[Any] = []
    if symbol:
        where.append("symbol = ?")
        params.append(symbol)
    if mode:
        where.append("mode = ?")
        params.append(mode)
    sql = f"SELECT * FROM {TABLE_NAME}"
    if where:
        sql += " WHERE " + " AND ".join(where)
    sql += " ORDER BY id DESC LIMIT ?"
    params.append(int(limit))
    try:
        conn = sqlite3.connect(db_path)
        try:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(sql, params).fetchall()
            return [dict(r) for r in rows]
        finally:
            conn.close()
    except Exception as exc:
        logger.warning("LEARNING_READER_FAILED err=%s", exc)
        return []


__all__ = [
    "TABLE_NAME",
    "TradeLearningRecord",
    "read_recent_learning_rows",
    "record_trade_outcome",
]

# Touch asdict so static analysers do not flag the import as unused (kept
# available for callers that want to log a record in addition to writing).
_ = asdict
