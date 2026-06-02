"""
AI canonical storage — bootstrap of all AI-side SQLite tables.

Tables created here are owned by the AI subsystem only. They are intentionally
kept separate from `paper_trades`, `portfolio_engine_*`, etc. so the AI layer
can be reset / inspected without touching trade ledger data.

Tables:
    ai_context_snapshots        — per-symbol MTF + market context snapshot history
    ai_position_state           — per-symbol open-position AI re-evaluation state
    ai_outcome_training_rows    — closed-trade outcomes mapped back to feature snapshots
    ai_inference_log            — every published AIDecision (audit)
    ai_feature_samples          — self-supervised / telemetry feature rows (must match live dim)
"""

from __future__ import annotations

import json
import logging
import sqlite3
import time
from pathlib import Path
from typing import Any

from backend.database_schema import DATABASE_PATH

logger = logging.getLogger(__name__)


_SCHEMA = [
    """
    CREATE TABLE IF NOT EXISTS ai_context_snapshots (
        id              INTEGER PRIMARY KEY AUTOINCREMENT,
        symbol          TEXT    NOT NULL,
        ts_utc          TEXT    NOT NULL,
        change_24h_pct  REAL,
        volume_24h_usd  REAL,
        relative_volume REAL,
        liquidity_tier  INTEGER,
        spread_pct      REAL,
        depth_imbalance REAL,
        rs_btc          REAL,
        rs_eth          REAL,
        btc_dominance_proxy REAL,
        market_regime   TEXT,
        sentiment_fear_greed REAL,    -- canonical sentiment, alternative.me Fear/Greed Index, [-1,+1]
        mtf_json        TEXT,    -- {tf: {trend, slope, rsi, atr_pct, ema_align}}
        ctx_multiplier  REAL
    )
    """,
    "CREATE INDEX IF NOT EXISTS ix_ai_ctx_symbol_ts ON ai_context_snapshots(symbol, ts_utc)",
    """
    CREATE TABLE IF NOT EXISTS ai_position_state (
        symbol          TEXT    PRIMARY KEY,
        opened_at_utc   TEXT,
        entry_price     REAL,
        last_mark       REAL,
        last_eval_utc   TEXT,
        last_action     TEXT,        -- "HOLD" | "REDUCE" | "EXIT" | "ADD"
        last_confidence REAL,
        unrealized_pct  REAL,
        mtf_alignment   REAL,
        ctx_json        TEXT,        -- snapshot of ai_context applied at last eval
        notes           TEXT
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS ai_position_recommendation (
        id              INTEGER PRIMARY KEY AUTOINCREMENT,
        symbol          TEXT    NOT NULL,
        ts_utc          TEXT    NOT NULL,
        action          TEXT    NOT NULL,    -- "HOLD" | "REDUCE" | "EXIT" | "ADD"
        confidence      REAL,                -- 0..1 strength of recommendation
        unrealized_pct  REAL,
        prob_buy        REAL,
        prob_hold       REAL,
        prob_sell       REAL,
        ctx_multiplier  REAL,
        mtf_alignment   REAL,
        reasons_json    TEXT,                -- ["mtf_flip_negative", "rs_btc_weak", ...]
        ctx_json        TEXT
    )
    """,
    "CREATE INDEX IF NOT EXISTS ix_ai_pos_reco_symbol_ts ON ai_position_recommendation(symbol, ts_utc)",
    """
    CREATE TABLE IF NOT EXISTS ai_outcome_training_rows (
        id              INTEGER PRIMARY KEY AUTOINCREMENT,
        symbol          TEXT NOT NULL,
        opened_at_utc   TEXT NOT NULL,
        closed_at_utc   TEXT NOT NULL,
        hold_seconds    REAL,
        entry_price     REAL,
        exit_price      REAL,
        realized_pct    REAL,
        outcome_label   INTEGER,     -- 1 = trade-worthy win, 0 = not worthy
        outcome_class   TEXT,        -- "WIN" | "LOSS" | "NO_TRACTION" | "ACTIVE_HOLD_DEAD"
        features_json   TEXT,        -- exact model input at entry (124 v1 or 145 v2/v3 primary-clock); join via feature_version
        context_json    TEXT,        -- ai_context snapshot at entry (MTF + market ctx)
        strategy_id     TEXT,
        rank_snapshot_id INTEGER,
        entry_rank      INTEGER,
        peer_ranks_json TEXT,
        score_components_json TEXT,
        selected_net_expected_value REAL,
        actual_net_outcome REAL,
        good_bad_memory_class TEXT,
        gross_pnl_pct REAL,
        estimated_fee_pct REAL,
        estimated_slippage_pct REAL,
        spread_cost_pct REAL,
        net_pnl_pct REAL,
        trade_was_worth_taking INTEGER,
        churn_flag INTEGER,
        exit_scores_json TEXT,
        ai_exit_recommended_action TEXT,
        ai_exit_hold_score REAL,
        ai_exit_take_profit_score REAL,
        ai_exit_cut_score REAL,
        ai_exit_trail_score REAL,
        ai_exit_confidence REAL,
        ai_exit_expected_value REAL,
        ai_exit_reason_json TEXT,
        time_in_trade_sec REAL,
        max_favorable_excursion REAL,
        max_adverse_excursion REAL,
        mfe_giveback_pct REAL,
        exit_quality_label TEXT,
        buy_base_size REAL,
        buy_final_size REAL,
        buy_sizing_multiplier REAL,
        buy_sizing_components_json TEXT,
        buy_cap_reason TEXT,
        buy_drawdown_factor REAL,
        buy_memory_factor REAL,
        buy_ev_factor REAL,
        ingested_at_utc TEXT NOT NULL,
        UNIQUE(symbol, opened_at_utc, closed_at_utc)
    )
    """,
    "CREATE INDEX IF NOT EXISTS ix_ai_outcome_symbol ON ai_outcome_training_rows(symbol)",
    "CREATE INDEX IF NOT EXISTS ix_ai_outcome_closed ON ai_outcome_training_rows(closed_at_utc)",
    """
    CREATE TABLE IF NOT EXISTS ai_inference_log (
        id              INTEGER PRIMARY KEY AUTOINCREMENT,
        decision_id     TEXT,
        strategy_id     TEXT DEFAULT 'day',
        symbol          TEXT NOT NULL,
        ts_utc          TEXT NOT NULL,
        prediction      TEXT,
        argmax_action   TEXT,
        prob_buy        REAL,
        prob_hold       REAL,
        prob_sell       REAL,
        winner_prob_raw REAL,
        confidence      REAL,
        buy_margin      REAL,
        ctx_multiplier  REAL,
        ctx_json        TEXT,
        feature_version INTEGER DEFAULT 1,    -- 1=124, 2/3=145 (v3=primary-clock 5m/15m + context)
        features_json   TEXT,                  -- exact model input vector used (JSON list)
        model_artifact  TEXT,
        label_version   TEXT,
        label_horizon_bars INTEGER
    )
    """,
    "CREATE INDEX IF NOT EXISTS ix_ai_infer_symbol_ts ON ai_inference_log(symbol, ts_utc)",
    """
    CREATE TABLE IF NOT EXISTS ai_feature_samples (
        id                  INTEGER PRIMARY KEY AUTOINCREMENT,
        ts                  REAL    NOT NULL,
        symbol              TEXT    NOT NULL,
        context_key         TEXT    NOT NULL,
        features_json       TEXT    NOT NULL,
        disagreement        REAL    NOT NULL DEFAULT 0.0,
        rare                INTEGER NOT NULL DEFAULT 0,
        model_probs_json    TEXT,
        created_at          DATETIME DEFAULT CURRENT_TIMESTAMP,
        feature_version     INTEGER DEFAULT 2,
        feature_dim         INTEGER DEFAULT 145
    )
    """,
    "CREATE INDEX IF NOT EXISTS ix_ai_feat_samples_sym_ts ON ai_feature_samples(symbol, ts)",
    """
    CREATE TABLE IF NOT EXISTS ai_rank_snapshots (
        id                      INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp               TEXT NOT NULL,
        strategy_id             TEXT NOT NULL,
        selected_symbol         TEXT,
        selected_rank           INTEGER,
        selected_score          REAL,
        selected_net_expected_value REAL,
        leaderboard_json        TEXT,
        score_components_json   TEXT,
        peer_ranks_json         TEXT,
        winner_reason           TEXT,
        rejected_reason_json    TEXT,
        market_regime           TEXT,
        created_at              TEXT NOT NULL DEFAULT (datetime('now'))
    )
    """,
    "CREATE INDEX IF NOT EXISTS ix_ai_rank_snapshots_strategy_ts ON ai_rank_snapshots(strategy_id, timestamp)",
    """
    CREATE TABLE IF NOT EXISTS ai_good_trade_patterns (
        id                  INTEGER PRIMARY KEY AUTOINCREMENT,
        symbol              TEXT NOT NULL,
        strategy_id         TEXT NOT NULL,
        closed_at_utc       TEXT NOT NULL,
        rank_snapshot_id    INTEGER,
        pattern_vector_json TEXT,
        net_outcome_pct     REAL,
        hold_seconds        REAL,
        reason              TEXT,
        created_at          TEXT NOT NULL DEFAULT (datetime('now'))
    )
    """,
    "CREATE INDEX IF NOT EXISTS ix_ai_good_trade_patterns_sym_sid ON ai_good_trade_patterns(symbol, strategy_id)",
    """
    CREATE TABLE IF NOT EXISTS ai_bad_trade_patterns (
        id                  INTEGER PRIMARY KEY AUTOINCREMENT,
        symbol              TEXT NOT NULL,
        strategy_id         TEXT NOT NULL,
        closed_at_utc       TEXT NOT NULL,
        rank_snapshot_id    INTEGER,
        pattern_vector_json TEXT,
        net_outcome_pct     REAL,
        hold_seconds        REAL,
        reason              TEXT,
        created_at          TEXT NOT NULL DEFAULT (datetime('now'))
    )
    """,
    "CREATE INDEX IF NOT EXISTS ix_ai_bad_trade_patterns_sym_sid ON ai_bad_trade_patterns(symbol, strategy_id)",
    """
    CREATE TABLE IF NOT EXISTS ai_symbol_strategy_expectancy (
        symbol              TEXT NOT NULL,
        strategy_id         TEXT NOT NULL,
        expectancy          REAL NOT NULL DEFAULT 0.0,
        total_trades        INTEGER NOT NULL DEFAULT 0,
        good_count          INTEGER NOT NULL DEFAULT 0,
        bad_count           INTEGER NOT NULL DEFAULT 0,
        last_net_outcome    REAL NOT NULL DEFAULT 0.0,
        updated_at          TEXT NOT NULL DEFAULT (datetime('now')),
        PRIMARY KEY (symbol, strategy_id)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS ai_trade_memory_scores (
        id                  INTEGER PRIMARY KEY AUTOINCREMENT,
        symbol              TEXT NOT NULL,
        strategy_id         TEXT NOT NULL,
        memory_score        REAL NOT NULL DEFAULT 0.0,
        good_similarity     REAL NOT NULL DEFAULT 0.0,
        bad_similarity      REAL NOT NULL DEFAULT 0.0,
        updated_at          TEXT NOT NULL DEFAULT (datetime('now'))
    )
    """,
    "CREATE INDEX IF NOT EXISTS ix_ai_trade_memory_scores_sym_sid ON ai_trade_memory_scores(symbol, strategy_id)",
    """
    CREATE TABLE IF NOT EXISTS ai_peer_shadow_outcomes (
        id                              INTEGER PRIMARY KEY AUTOINCREMENT,
        rank_snapshot_id                INTEGER,
        timestamp                       TEXT NOT NULL,
        strategy_id                     TEXT NOT NULL,
        selected_symbol                 TEXT NOT NULL,
        peer_symbol                     TEXT NOT NULL,
        peer_rank_at_entry              INTEGER,
        peer_final_profit_score         REAL,
        peer_net_expected_value         REAL,
        peer_score_components_json      TEXT,
        entry_price                     REAL,
        shadow_exit_price               REAL,
        shadow_exit_time                TEXT,
        shadow_exit_reason              TEXT,
        shadow_gross_pnl_pct            REAL,
        shadow_estimated_fees_pct       REAL,
        shadow_estimated_slippage_pct   REAL,
        shadow_net_pnl_pct              REAL,
        selected_actual_net_pnl_pct     REAL,
        peer_would_have_won             INTEGER DEFAULT 0,
        peer_would_have_lost            INTEGER DEFAULT 0,
        peer_outperformed_selected      INTEGER DEFAULT 0,
        selected_was_correct            INTEGER DEFAULT 0,
        selected_was_wrong              INTEGER DEFAULT 0,
        learning_label                  TEXT,
        created_at                      TEXT NOT NULL DEFAULT (datetime('now'))
    )
    """,
    "CREATE INDEX IF NOT EXISTS ix_ai_peer_shadow_snapshot ON ai_peer_shadow_outcomes(rank_snapshot_id, strategy_id)",
    """
    CREATE TABLE IF NOT EXISTS ai_strategy_score_weights (
        id                  INTEGER PRIMARY KEY AUTOINCREMENT,
        strategy_id         TEXT NOT NULL,
        symbol              TEXT NOT NULL DEFAULT '',
        regime              TEXT NOT NULL DEFAULT '',
        component_name      TEXT NOT NULL,
        weight              REAL NOT NULL,
        previous_weight     REAL NOT NULL DEFAULT 0.0,
        sample_count        INTEGER NOT NULL DEFAULT 0,
        good_count          INTEGER NOT NULL DEFAULT 0,
        bad_count           INTEGER NOT NULL DEFAULT 0,
        net_expectancy      REAL NOT NULL DEFAULT 0.0,
        last_adjusted_at    TEXT,
        created_at          TEXT NOT NULL DEFAULT (datetime('now')),
        updated_at          TEXT NOT NULL DEFAULT (datetime('now')),
        UNIQUE(strategy_id, symbol, regime, component_name)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS ai_strategy_capital_sleeves (
        strategy_id         TEXT PRIMARY KEY,
        allocated_capital   REAL NOT NULL DEFAULT 0.0,
        deployed_capital    REAL NOT NULL DEFAULT 0.0,
        available_capital   REAL NOT NULL DEFAULT 0.0,
        realized_pnl        REAL NOT NULL DEFAULT 0.0,
        unrealized_pnl      REAL NOT NULL DEFAULT 0.0,
        win_count           INTEGER NOT NULL DEFAULT 0,
        loss_count          INTEGER NOT NULL DEFAULT 0,
        net_expectancy      REAL NOT NULL DEFAULT 0.0,
        max_drawdown        REAL NOT NULL DEFAULT 0.0,
        current_drawdown    REAL NOT NULL DEFAULT 0.0,
        allocation_pct      REAL NOT NULL DEFAULT 0.5,
        last_rebalanced_at  TEXT,
        created_at          TEXT NOT NULL DEFAULT (datetime('now')),
        updated_at          TEXT NOT NULL DEFAULT (datetime('now'))
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS ai_model_versions (
        id                      INTEGER PRIMARY KEY AUTOINCREMENT,
        model_id                TEXT NOT NULL,
        strategy_id             TEXT NOT NULL,
        symbol                  TEXT NOT NULL,
        feature_version         INTEGER,
        artifact_hash           TEXT,
        path                    TEXT NOT NULL,
        status                  TEXT NOT NULL, -- candidate | active | archived | rollback
        created_at              TEXT NOT NULL DEFAULT (datetime('now')),
        promoted_at             TEXT,
        retired_at              TEXT,
        validation_metrics_json TEXT,
        live_metrics_json       TEXT,
        promotion_reason        TEXT,
        rollback_reason         TEXT,
        UNIQUE(model_id)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS ai_model_promotion_events (
        id                      INTEGER PRIMARY KEY AUTOINCREMENT,
        strategy_id             TEXT NOT NULL,
        symbol                  TEXT NOT NULL,
        from_model_id           TEXT,
        to_model_id             TEXT,
        event_type              TEXT NOT NULL, -- promote | rollback | reject
        reason                  TEXT,
        metrics_json            TEXT,
        created_at              TEXT NOT NULL DEFAULT (datetime('now'))
    )
    """,
]


# Forward-compatible add-column migrations (idempotent).
_MIGRATIONS: list[tuple[str, str, str]] = [
    # (table, column, sql_to_add)
    ("ai_context_snapshots", "sentiment_fear_greed", "ALTER TABLE ai_context_snapshots ADD COLUMN sentiment_fear_greed REAL"),
    ("ai_inference_log", "feature_version", "ALTER TABLE ai_inference_log ADD COLUMN feature_version INTEGER DEFAULT 1"),
    ("ai_inference_log", "features_json", "ALTER TABLE ai_inference_log ADD COLUMN features_json TEXT"),
    ("ai_inference_log", "strategy_id", "ALTER TABLE ai_inference_log ADD COLUMN strategy_id TEXT DEFAULT 'day'"),
    ("ai_feature_samples", "feature_version", "ALTER TABLE ai_feature_samples ADD COLUMN feature_version INTEGER DEFAULT 2"),
    ("ai_feature_samples", "feature_dim", "ALTER TABLE ai_feature_samples ADD COLUMN feature_dim INTEGER DEFAULT 145"),
    ("ai_outcome_training_rows", "strategy_id", "ALTER TABLE ai_outcome_training_rows ADD COLUMN strategy_id TEXT"),
    ("ai_outcome_training_rows", "rank_snapshot_id", "ALTER TABLE ai_outcome_training_rows ADD COLUMN rank_snapshot_id INTEGER"),
    ("ai_outcome_training_rows", "entry_rank", "ALTER TABLE ai_outcome_training_rows ADD COLUMN entry_rank INTEGER"),
    ("ai_outcome_training_rows", "peer_ranks_json", "ALTER TABLE ai_outcome_training_rows ADD COLUMN peer_ranks_json TEXT"),
    (
        "ai_outcome_training_rows",
        "score_components_json",
        "ALTER TABLE ai_outcome_training_rows ADD COLUMN score_components_json TEXT",
    ),
    (
        "ai_outcome_training_rows",
        "selected_net_expected_value",
        "ALTER TABLE ai_outcome_training_rows ADD COLUMN selected_net_expected_value REAL",
    ),
    ("ai_outcome_training_rows", "actual_net_outcome", "ALTER TABLE ai_outcome_training_rows ADD COLUMN actual_net_outcome REAL"),
    (
        "ai_outcome_training_rows",
        "good_bad_memory_class",
        "ALTER TABLE ai_outcome_training_rows ADD COLUMN good_bad_memory_class TEXT",
    ),
    (
        "ai_outcome_training_rows",
        "gross_pnl_pct",
        "ALTER TABLE ai_outcome_training_rows ADD COLUMN gross_pnl_pct REAL",
    ),
    (
        "ai_outcome_training_rows",
        "estimated_fee_pct",
        "ALTER TABLE ai_outcome_training_rows ADD COLUMN estimated_fee_pct REAL",
    ),
    (
        "ai_outcome_training_rows",
        "estimated_slippage_pct",
        "ALTER TABLE ai_outcome_training_rows ADD COLUMN estimated_slippage_pct REAL",
    ),
    (
        "ai_outcome_training_rows",
        "spread_cost_pct",
        "ALTER TABLE ai_outcome_training_rows ADD COLUMN spread_cost_pct REAL",
    ),
    (
        "ai_outcome_training_rows",
        "net_pnl_pct",
        "ALTER TABLE ai_outcome_training_rows ADD COLUMN net_pnl_pct REAL",
    ),
    (
        "ai_outcome_training_rows",
        "trade_was_worth_taking",
        "ALTER TABLE ai_outcome_training_rows ADD COLUMN trade_was_worth_taking INTEGER",
    ),
    (
        "ai_outcome_training_rows",
        "churn_flag",
        "ALTER TABLE ai_outcome_training_rows ADD COLUMN churn_flag INTEGER",
    ),
    (
        "ai_outcome_training_rows",
        "exit_scores_json",
        "ALTER TABLE ai_outcome_training_rows ADD COLUMN exit_scores_json TEXT",
    ),
    (
        "ai_outcome_training_rows",
        "ai_exit_recommended_action",
        "ALTER TABLE ai_outcome_training_rows ADD COLUMN ai_exit_recommended_action TEXT",
    ),
    ("ai_outcome_training_rows", "ai_exit_hold_score", "ALTER TABLE ai_outcome_training_rows ADD COLUMN ai_exit_hold_score REAL"),
    (
        "ai_outcome_training_rows",
        "ai_exit_take_profit_score",
        "ALTER TABLE ai_outcome_training_rows ADD COLUMN ai_exit_take_profit_score REAL",
    ),
    ("ai_outcome_training_rows", "ai_exit_cut_score", "ALTER TABLE ai_outcome_training_rows ADD COLUMN ai_exit_cut_score REAL"),
    ("ai_outcome_training_rows", "ai_exit_trail_score", "ALTER TABLE ai_outcome_training_rows ADD COLUMN ai_exit_trail_score REAL"),
    ("ai_outcome_training_rows", "ai_exit_confidence", "ALTER TABLE ai_outcome_training_rows ADD COLUMN ai_exit_confidence REAL"),
    (
        "ai_outcome_training_rows",
        "ai_exit_expected_value",
        "ALTER TABLE ai_outcome_training_rows ADD COLUMN ai_exit_expected_value REAL",
    ),
    (
        "ai_outcome_training_rows",
        "ai_exit_reason_json",
        "ALTER TABLE ai_outcome_training_rows ADD COLUMN ai_exit_reason_json TEXT",
    ),
    ("ai_outcome_training_rows", "time_in_trade_sec", "ALTER TABLE ai_outcome_training_rows ADD COLUMN time_in_trade_sec REAL"),
    (
        "ai_outcome_training_rows",
        "max_favorable_excursion",
        "ALTER TABLE ai_outcome_training_rows ADD COLUMN max_favorable_excursion REAL",
    ),
    (
        "ai_outcome_training_rows",
        "max_adverse_excursion",
        "ALTER TABLE ai_outcome_training_rows ADD COLUMN max_adverse_excursion REAL",
    ),
    ("ai_outcome_training_rows", "mfe_giveback_pct", "ALTER TABLE ai_outcome_training_rows ADD COLUMN mfe_giveback_pct REAL"),
    ("ai_outcome_training_rows", "exit_quality_label", "ALTER TABLE ai_outcome_training_rows ADD COLUMN exit_quality_label TEXT"),
    ("ai_outcome_training_rows", "buy_base_size", "ALTER TABLE ai_outcome_training_rows ADD COLUMN buy_base_size REAL"),
    ("ai_outcome_training_rows", "buy_final_size", "ALTER TABLE ai_outcome_training_rows ADD COLUMN buy_final_size REAL"),
    (
        "ai_outcome_training_rows",
        "buy_sizing_multiplier",
        "ALTER TABLE ai_outcome_training_rows ADD COLUMN buy_sizing_multiplier REAL",
    ),
    (
        "ai_outcome_training_rows",
        "buy_sizing_components_json",
        "ALTER TABLE ai_outcome_training_rows ADD COLUMN buy_sizing_components_json TEXT",
    ),
    ("ai_outcome_training_rows", "buy_cap_reason", "ALTER TABLE ai_outcome_training_rows ADD COLUMN buy_cap_reason TEXT"),
    (
        "ai_outcome_training_rows",
        "buy_drawdown_factor",
        "ALTER TABLE ai_outcome_training_rows ADD COLUMN buy_drawdown_factor REAL",
    ),
    (
        "ai_outcome_training_rows",
        "buy_memory_factor",
        "ALTER TABLE ai_outcome_training_rows ADD COLUMN buy_memory_factor REAL",
    ),
    ("ai_outcome_training_rows", "buy_ev_factor", "ALTER TABLE ai_outcome_training_rows ADD COLUMN buy_ev_factor REAL"),
]


def _existing_columns(conn: sqlite3.Connection, table: str) -> set[str]:
    try:
        return {row[1] for row in conn.execute(f"PRAGMA table_info({table})")}
    except sqlite3.Error:
        return set()


def ensure_ai_canonical_tables(db_path: str | Path = DATABASE_PATH) -> None:
    """Idempotently create every AI-canonical SQLite table and apply migrations."""
    path = str(db_path)
    try:
        with sqlite3.connect(path) as conn:
            cur = conn.cursor()
            for stmt in _SCHEMA:
                cur.execute(stmt)
            for table, column, sql in _MIGRATIONS:
                try:
                    cols = _existing_columns(conn, table)
                    if column not in cols:
                        cur.execute(sql)
                except sqlite3.Error as me:
                    logger.debug("ai_canonical migration skipped (%s.%s): %s", table, column, me)
            conn.commit()
        # Step-1 instrumentation: unified audit trail (strategy / artifact / context proof)
        try:
            from backend.services.strategy_runtime_audit import ensure_strategy_runtime_audit_table

            ensure_strategy_runtime_audit_table(path)
        except Exception as e:
            logger.debug("ensure_strategy_runtime_audit_table skipped: %s", e)
    except sqlite3.Error as e:
        logger.warning("ensure_ai_canonical_tables failed at %s: %s", path, e)


def persist_ai_feature_sample_row(
    *,
    symbol: str,
    context_key: str,
    features: list[float],
    feature_version: int = 2,
    disagreement: float = 0.0,
    rare: int = 0,
    model_probs: dict[str, Any] | None = None,
    db_path: str | Path = DATABASE_PATH,
) -> None:
    """
    Persist one training/telemetry feature row. Canonical dim is 145 (v2/v3).
    v3 rows use primary-clock OHLCV (see ``CANONICAL_TELEMETRY_CONTEXT_KEY_V3``).
    """
    ensure_ai_canonical_tables(db_path)
    dim = len(features)
    probs_json = json.dumps(model_probs, separators=(",", ":")) if model_probs is not None else None
    feats_json = json.dumps([float(x) for x in features], separators=(",", ":"))
    path = str(db_path)
    try:
        with sqlite3.connect(path) as conn:
            conn.execute(
                """
                INSERT INTO ai_feature_samples (
                    ts, symbol, context_key, features_json, disagreement, rare,
                    model_probs_json, feature_version, feature_dim
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    time.time(),
                    symbol,
                    context_key,
                    feats_json,
                    float(disagreement),
                    int(rare),
                    probs_json,
                    int(feature_version),
                    int(dim),
                ),
            )
            conn.commit()
    except sqlite3.Error as e:
        logger.debug("persist_ai_feature_sample_row failed: %s", e)


def read_recent_outcome_training_rows(
    symbol: str | None = None,
    strategy_id: str | None = None,
    limit: int = 5000,
    db_path: str | Path = DATABASE_PATH,
) -> list[dict[str, Any]]:
    """Read recent closed-trade outcome rows for AI retraining."""
    ensure_ai_canonical_tables(db_path)
    where: list[str] = []
    params: list[Any] = []
    if symbol:
        where.append("(UPPER(symbol)=UPPER(?) OR UPPER(symbol)=UPPER(?))")
        sym = str(symbol).strip().upper()
        params.extend([sym, sym.replace("/", "")])
    if strategy_id:
        where.append("LOWER(COALESCE(strategy_id, 'day')) = LOWER(?)")
        params.append(str(strategy_id).strip().lower())
    sql = "SELECT * FROM ai_outcome_training_rows"
    if where:
        sql += " WHERE " + " AND ".join(where)
    sql += " ORDER BY id DESC LIMIT ?"
    params.append(int(limit))
    path = str(db_path)
    try:
        with sqlite3.connect(path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(sql, params).fetchall()
            return [dict(r) for r in rows]
    except sqlite3.Error as e:
        logger.warning("read_recent_outcome_training_rows failed: %s", e)
        return []


__all__ = ["ensure_ai_canonical_tables", "persist_ai_feature_sample_row", "read_recent_outcome_training_rows"]
