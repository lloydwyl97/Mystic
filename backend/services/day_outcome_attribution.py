"""Post-trade outcome attribution for DAY learning (observation + soft learning feed)."""

from __future__ import annotations

import json
import logging
import sqlite3
from datetime import datetime, timezone
from typing import Any

from backend.database_schema import DATABASE_PATH
from backend.services.day_block_scores import compute_block_scores_from_decision_data
from backend.services.day_feature_health import entry_feature_health_pass

logger = logging.getLogger(__name__)

TABLE = "day_outcome_attribution"

_CREATE = f"""
CREATE TABLE IF NOT EXISTS {TABLE} (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    trade_id TEXT NOT NULL UNIQUE,
    symbol TEXT NOT NULL,
    regime TEXT NOT NULL DEFAULT '',
    setup_thesis TEXT NOT NULL DEFAULT '',
    entry_time TEXT NOT NULL DEFAULT '',
    exit_time TEXT NOT NULL DEFAULT '',
    closed_at_utc TEXT NOT NULL DEFAULT '',
    net_pnl_after_fees REAL,
    exit_reason TEXT NOT NULL DEFAULT '',
    outcome_reason TEXT NOT NULL DEFAULT '',
    entry_145_vector_json TEXT,
    entry_feature_health_json TEXT,
    entry_block_scores_json TEXT,
    setup_score REAL,
    execution_quality_score REAL,
    model_probabilities_json TEXT,
    final_selection_score REAL,
    feature_health_pass INTEGER NOT NULL DEFAULT 0,
    attribution_json TEXT NOT NULL DEFAULT '{{}}',
    ingested_at_utc TEXT NOT NULL,
    created_at TEXT NOT NULL DEFAULT ''
)
"""

_INDEXES = (
    f"CREATE INDEX IF NOT EXISTS ix_day_outcome_attr_symbol ON {TABLE}(symbol, closed_at_utc)",
    f"CREATE INDEX IF NOT EXISTS ix_day_outcome_attr_trade ON {TABLE}(trade_id)",
    f"CREATE INDEX IF NOT EXISTS ix_day_outcome_attr_regime ON {TABLE}(regime, setup_thesis)",
    f"CREATE INDEX IF NOT EXISTS ix_day_outcome_attr_created ON {TABLE}(created_at)",
)

# Idempotent column adds for databases created with the legacy minimal schema.
_MIGRATIONS: list[tuple[str, str]] = [
    ("regime", f"ALTER TABLE {TABLE} ADD COLUMN regime TEXT NOT NULL DEFAULT ''"),
    ("setup_thesis", f"ALTER TABLE {TABLE} ADD COLUMN setup_thesis TEXT NOT NULL DEFAULT ''"),
    ("entry_time", f"ALTER TABLE {TABLE} ADD COLUMN entry_time TEXT NOT NULL DEFAULT ''"),
    ("exit_time", f"ALTER TABLE {TABLE} ADD COLUMN exit_time TEXT NOT NULL DEFAULT ''"),
    ("net_pnl_after_fees", f"ALTER TABLE {TABLE} ADD COLUMN net_pnl_after_fees REAL"),
    ("exit_reason", f"ALTER TABLE {TABLE} ADD COLUMN exit_reason TEXT NOT NULL DEFAULT ''"),
    ("entry_145_vector_json", f"ALTER TABLE {TABLE} ADD COLUMN entry_145_vector_json TEXT"),
    ("entry_feature_health_json", f"ALTER TABLE {TABLE} ADD COLUMN entry_feature_health_json TEXT"),
    ("entry_block_scores_json", f"ALTER TABLE {TABLE} ADD COLUMN entry_block_scores_json TEXT"),
    ("setup_score", f"ALTER TABLE {TABLE} ADD COLUMN setup_score REAL"),
    ("execution_quality_score", f"ALTER TABLE {TABLE} ADD COLUMN execution_quality_score REAL"),
    ("model_probabilities_json", f"ALTER TABLE {TABLE} ADD COLUMN model_probabilities_json TEXT"),
    ("final_selection_score", f"ALTER TABLE {TABLE} ADD COLUMN final_selection_score REAL"),
    ("feature_health_pass", f"ALTER TABLE {TABLE} ADD COLUMN feature_health_pass INTEGER NOT NULL DEFAULT 0"),
    ("created_at", f"ALTER TABLE {TABLE} ADD COLUMN created_at TEXT NOT NULL DEFAULT ''"),
]

OUTCOME_REASONS: tuple[str, ...] = (
    "GOOD_SETUP_GOOD_ENTRY",
    "GOOD_SETUP_BAD_ENTRY",
    "BAD_SETUP",
    "REGIME_SHIFT_AGAINST_TRADE",
    "VOLUME_CONFIRMATION_FAILED",
    "VOLATILITY_EXPANSION_AGAINST_TRADE",
    "EXECUTION_COST_TOO_HIGH",
    "EXIT_TOO_EARLY",
    "EXIT_TOO_LATE",
    "MARKET_WIDE_REVERSAL",
    "FEATURE_HEALTH_WEAK",
    "SETUP_HISTORY_WEAK",
)


def _existing_columns(conn: sqlite3.Connection) -> set[str]:
    try:
        return {row[1] for row in conn.execute(f"PRAGMA table_info({TABLE})")}
    except sqlite3.Error:
        return set()


def ensure_outcome_attribution_table(db_path: str = DATABASE_PATH) -> None:
    with sqlite3.connect(db_path) as conn:
        conn.execute(_CREATE)
        cols = _existing_columns(conn)
        for column, sql in _MIGRATIONS:
            if column not in cols:
                try:
                    conn.execute(sql)
                    cols.add(column)
                except sqlite3.Error as exc:
                    logger.debug("day_outcome_attribution migration skipped (%s): %s", column, exc)
        for stmt in _INDEXES:
            cols_now = _existing_columns(conn)
            if "created_at" in stmt and "created_at" not in cols_now:
                continue
            conn.execute(stmt)
        conn.commit()


def classify_outcome_reason(
    *,
    explainability: dict[str, Any],
    net_profit_pct: float | None,
    close_reason: str,
    hold_seconds: float | None = None,
) -> str:
    ex = explainability or {}
    pnl = float(net_profit_pct or 0.0)
    setup_score = float(ex.get("setup_score") or 0.5)
    exec_q = float(ex.get("execution_quality_score") or 0.5)
    fh = float(ex.get("feature_health_score") or 0.0)
    if not entry_feature_health_pass(ex) or fh < 0.55:
        return "FEATURE_HEALTH_WEAK"
    spread = abs(float(ex.get("entry_spread_pct") or ex.get("signal_spread_pct") or 0.0))
    slip = abs(float(ex.get("entry_slippage_pct") or 0.0))
    if spread > 0.004 or slip > 0.003 or exec_q < 0.35:
        if pnl <= 0:
            return "EXECUTION_COST_TOO_HIGH"
    if setup_score < 0.40:
        return "BAD_SETUP" if pnl <= 0 else "GOOD_SETUP_BAD_ENTRY"
    same_pnl = float(ex.get("same_setup_today_net_pnl") or 0.0)
    if same_pnl < -0.004:
        return "SETUP_HISTORY_WEAK"
    trans = float(ex.get("regime_transition_score") or 0.0)
    if trans > 0.5 and pnl < 0:
        return "REGIME_SHIFT_AGAINST_TRADE"
    vol_block = float(ex.get("volatility_block_score") or 0.5)
    if vol_block > 0.75 and pnl < 0:
        return "VOLATILITY_EXPANSION_AGAINST_TRADE"
    vol_rank = float(ex.get("volume_block_score") or 0.5)
    if vol_rank < 0.35 and pnl < 0:
        return "VOLUME_CONFIRMATION_FAILED"
    cr = str(close_reason or "").upper()
    hs = float(hold_seconds or 0.0)
    if pnl > 0 and hs < 120:
        return "EXIT_TOO_EARLY"
    if pnl < 0 and hs > 3600 * 6:
        return "EXIT_TOO_LATE"
    if "NET_PROFIT" in cr and setup_score >= 0.55 and pnl > 0:
        return "GOOD_SETUP_GOOD_ENTRY"
    if pnl < 0 and setup_score >= 0.55:
        return "GOOD_SETUP_BAD_ENTRY"
    if pnl < 0:
        return "MARKET_WIDE_REVERSAL"
    return "GOOD_SETUP_GOOD_ENTRY"


def build_attribution_payload(
    *,
    trade_id: str,
    symbol: str,
    explainability: dict[str, Any],
    net_profit_usd: float | None,
    net_profit_pct: float | None,
    close_reason: str,
    hold_seconds: float | None,
    entry_features: list[float] | None = None,
) -> dict[str, Any]:
    ex = dict(explainability or {})
    blocks = compute_block_scores_from_decision_data(ex)
    if not ex.get("feature_health_score"):
        ex["feature_health_score"] = blocks.get("feature_health_score", 0.0)
    reason = classify_outcome_reason(
        explainability=ex,
        net_profit_pct=net_profit_pct,
        close_reason=close_reason,
        hold_seconds=hold_seconds,
    )
    fh_pass = entry_feature_health_pass(ex)
    return {
        "trade_id": trade_id,
        "symbol": symbol,
        "regime": str(ex.get("day_route_regime") or ex.get("regime") or ""),
        "setup_thesis": str(ex.get("setup_type") or ex.get("entry_thesis") or ""),
        "entry_time": str(ex.get("entry_timestamp") or ex.get("timestamp") or ""),
        "exit_time": datetime.now(timezone.utc).isoformat(),
        "net_pnl_after_fees": float(net_profit_usd or 0.0),
        "net_pnl_pct": float(net_profit_pct or 0.0),
        "entry_145_vector": list(entry_features[:145]) if entry_features else [],
        "entry_feature_health": {
            "pass": fh_pass,
            "pct": float(ex.get("feature_health_pct") or 0.0),
            "score": float(ex.get("feature_health_score") or blocks.get("feature_health_score", 0.0)),
        },
        "entry_block_scores": blocks,
        "setup_score": float(ex.get("setup_score") or 0.5),
        "model_probabilities": {
            "prob_buy": ex.get("prob_buy"),
            "prob_hold": ex.get("prob_hold"),
            "prob_sell": ex.get("prob_sell"),
            "confidence": ex.get("ai_confidence") or ex.get("confidence"),
            "buy_margin": ex.get("entry_buy_margin") or ex.get("buy_margin"),
        },
        "final_selection_score": float(ex.get("final_selection_score") or 0.0),
        "execution_quality_score": float(ex.get("execution_quality_score") or 0.5),
        "exit_reason": str(close_reason or ""),
        "outcome_reason": reason,
        "feature_health_pass": fh_pass,
        "regime_transition_score": float(ex.get("regime_transition_score") or 0.0),
        "relative_strength_rank": int(float(ex.get("relative_strength_rank") or 0)),
    }


def record_outcome_attribution(
    *,
    trade_id: str,
    symbol: str,
    explainability: dict[str, Any] | None,
    net_profit_usd: float | None,
    net_profit_pct: float | None,
    close_reason: str,
    hold_seconds: float | None = None,
    entry_features: list[float] | None = None,
    db_path: str = DATABASE_PATH,
) -> int | None:
    if not trade_id:
        return None
    try:
        ensure_outcome_attribution_table(db_path)
        payload = build_attribution_payload(
            trade_id=trade_id,
            symbol=symbol,
            explainability=explainability or {},
            net_profit_usd=net_profit_usd,
            net_profit_pct=net_profit_pct,
            close_reason=close_reason,
            hold_seconds=hold_seconds,
            entry_features=entry_features,
        )
        ingested = datetime.now(timezone.utc).isoformat()
        exit_time = str(payload.get("exit_time") or ingested)
        vec_json = json.dumps(payload.get("entry_145_vector") or [], separators=(",", ":"))
        fh_json = json.dumps(payload.get("entry_feature_health") or {}, separators=(",", ":"))
        blocks_json = json.dumps(payload.get("entry_block_scores") or {}, separators=(",", ":"))
        probs_json = json.dumps(payload.get("model_probabilities") or {}, separators=(",", ":"))
        full_json = json.dumps(payload, separators=(",", ":"))
        with sqlite3.connect(db_path) as conn:
            cur = conn.execute(
                f"""
                INSERT INTO {TABLE} (
                    trade_id, symbol, regime, setup_thesis, entry_time, exit_time, closed_at_utc,
                    net_pnl_after_fees, exit_reason, outcome_reason,
                    entry_145_vector_json, entry_feature_health_json, entry_block_scores_json,
                    setup_score, execution_quality_score, model_probabilities_json,
                    final_selection_score, feature_health_pass,
                    attribution_json, ingested_at_utc, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(trade_id) DO UPDATE SET
                    symbol=excluded.symbol,
                    regime=excluded.regime,
                    setup_thesis=excluded.setup_thesis,
                    entry_time=excluded.entry_time,
                    exit_time=excluded.exit_time,
                    closed_at_utc=excluded.closed_at_utc,
                    net_pnl_after_fees=excluded.net_pnl_after_fees,
                    exit_reason=excluded.exit_reason,
                    outcome_reason=excluded.outcome_reason,
                    entry_145_vector_json=excluded.entry_145_vector_json,
                    entry_feature_health_json=excluded.entry_feature_health_json,
                    entry_block_scores_json=excluded.entry_block_scores_json,
                    setup_score=excluded.setup_score,
                    execution_quality_score=excluded.execution_quality_score,
                    model_probabilities_json=excluded.model_probabilities_json,
                    final_selection_score=excluded.final_selection_score,
                    feature_health_pass=excluded.feature_health_pass,
                    attribution_json=excluded.attribution_json,
                    ingested_at_utc=excluded.ingested_at_utc,
                    created_at=excluded.created_at
                """,
                (
                    trade_id,
                    symbol,
                    str(payload.get("regime") or ""),
                    str(payload.get("setup_thesis") or ""),
                    str(payload.get("entry_time") or ""),
                    exit_time,
                    exit_time,
                    float(payload.get("net_pnl_after_fees") or 0.0),
                    str(payload.get("exit_reason") or ""),
                    str(payload.get("outcome_reason") or ""),
                    vec_json,
                    fh_json,
                    blocks_json,
                    float(payload.get("setup_score") or 0.0),
                    float(payload.get("execution_quality_score") or 0.0),
                    probs_json,
                    float(payload.get("final_selection_score") or 0.0),
                    1 if payload.get("feature_health_pass") else 0,
                    full_json,
                    ingested,
                    ingested,
                ),
            )
            conn.commit()
            return int(cur.lastrowid or 0) or None
    except Exception as exc:
        logger.warning("record_outcome_attribution failed trade_id=%s: %s", trade_id, exc)
        return None


def dry_run_insert_and_delete(db_path: str = DATABASE_PATH) -> dict[str, Any]:
    """Insert one synthetic row, read it back, then delete it (schema verification)."""
    ensure_outcome_attribution_table(db_path)
    test_id = "__schema_verify__"
    payload = build_attribution_payload(
        trade_id=test_id,
        symbol="BTCUSDT",
        explainability={
            "day_route_regime": "bull",
            "setup_type": "HTF_TREND_PULLBACK",
            "setup_score": 0.71,
            "execution_quality_score": 0.8,
            "final_selection_score": 0.22,
            "feature_health_score": 0.9,
            "feature_health_pass": True,
        },
        net_profit_usd=1.25,
        net_profit_pct=0.0012,
        close_reason="NET_PROFIT_EXIT",
        hold_seconds=3600.0,
        entry_features=[0.1] * 145,
    )
    row_id = record_outcome_attribution(
        trade_id=test_id,
        symbol="BTCUSDT",
        explainability=payload,
        net_profit_usd=1.25,
        net_profit_pct=0.0012,
        close_reason="NET_PROFIT_EXIT",
        hold_seconds=3600.0,
        entry_features=[0.1] * 145,
        db_path=db_path,
    )
    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        row = conn.execute(
            f"SELECT trade_id, regime, setup_thesis, feature_health_pass, length(entry_145_vector_json) AS vec_len FROM {TABLE} WHERE trade_id=?",
            (test_id,),
        ).fetchone()
        conn.execute(f"DELETE FROM {TABLE} WHERE trade_id=?", (test_id,))
        conn.commit()
    ok = bool(row and row["trade_id"] == test_id and int(row["vec_len"] or 0) > 0)
    return {
        "insert_row_id": row_id,
        "read_back_ok": ok,
        "regime": row["regime"] if row else None,
        "setup_thesis": row["setup_thesis"] if row else None,
        "feature_health_pass": row["feature_health_pass"] if row else None,
        "deleted_test_row": True,
    }


__all__ = [
    "OUTCOME_REASONS",
    "build_attribution_payload",
    "classify_outcome_reason",
    "dry_run_insert_and_delete",
    "ensure_outcome_attribution_table",
    "record_outcome_attribution",
]
