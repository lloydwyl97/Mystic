"""Tests for adaptive score weight learning writer."""

from __future__ import annotations

import json
import sqlite3

from backend.services.ai_canonical_storage import ensure_ai_canonical_tables
from backend.services.ai_strategy_score_weight_writer import (
    COMPONENT_BOUNDS,
    adaptive_weights_row_count,
    backfill_adaptive_score_weights_from_outcomes,
    extract_scaled_components,
    propagate_adaptive_score_weights_for_close,
)
from backend.services.portfolio_engine import _adaptive_weights_enabled


def test_extract_scaled_components_from_buy_explain():
    explain = {
        "ai_confidence": 0.62,
        "entry_buy_margin": 0.04,
        "signal_ctx_rs_btc": 0.12,
        "trend_score": 0.7,
        "chop_score": 0.4,
        "selected_net_expected_value": 0.011,
        "entry_spread_pct": 0.0004,
        "day_route_regime": "bear",
    }
    comps = extract_scaled_components(explain)
    assert "model_probability" in comps
    assert "buy_margin" in comps
    assert "net_expected_value" in comps
    for name, val in comps.items():
        bmin, bmax = COMPONENT_BOUNDS[name]
        assert bmin <= val <= bmax


def _seed_sol_outcomes_db(tmp_path) -> str:
    db = str(tmp_path / "weights_test.db")
    ensure_ai_canonical_tables(db)
    explain_good = json.dumps(
        {
            "ai_confidence": 0.64,
            "entry_buy_margin": 0.05,
            "signal_ctx_rs_btc": 0.15,
            "trend_score": 0.72,
            "chop_score": 0.25,
            "selected_net_expected_value": 0.012,
            "entry_spread_pct": 0.0003,
            "day_route_regime": "bull",
            "feature_health_pass": True,
        }
    )
    explain_bad = json.dumps(
        {
            "ai_confidence": 0.45,
            "entry_buy_margin": 0.01,
            "signal_ctx_rs_btc": -0.10,
            "trend_score": 0.35,
            "chop_score": 0.65,
            "selected_net_expected_value": 0.002,
            "entry_spread_pct": 0.0008,
            "day_route_regime": "bear",
            "feature_health_pass": True,
        }
    )
    explain_bad_bull = json.dumps(
        {
            "ai_confidence": 0.45,
            "entry_buy_margin": 0.01,
            "signal_ctx_rs_btc": -0.10,
            "trend_score": 0.35,
            "chop_score": 0.65,
            "selected_net_expected_value": 0.002,
            "entry_spread_pct": 0.0008,
            "day_route_regime": "bull",
            "feature_health_pass": True,
        }
    )
    with sqlite3.connect(db) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS paper_trades (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                side TEXT NOT NULL,
                pnl REAL,
                pnl_pct REAL,
                explainability_json TEXT,
                strategy_id TEXT,
                is_synthetic INTEGER DEFAULT 0
            )
            """
        )
        for i, (cls, pnl, sc, reg) in enumerate(
            [
                ("GOOD", 0.008, explain_good, "bull"),
                ("GOOD", 0.006, explain_good, "bull"),
                ("BAD", -0.005, explain_bad_bull, "bull"),
                ("BAD", -0.004, explain_bad, "bear"),
            ],
            start=1,
        ):
            conn.execute(
                """
                INSERT INTO paper_trades (symbol, side, pnl, pnl_pct, explainability_json, strategy_id, is_synthetic)
                VALUES (?, 'BUY', 0.0, 0.0, ?, 'day', 0)
                """,
                ("SOL/USDT", sc),
            )
            conn.execute(
                """
                INSERT INTO paper_trades (symbol, side, pnl, pnl_pct, explainability_json, strategy_id, is_synthetic)
                VALUES (?, 'SELL', ?, ?, ?, 'day', 0)
                """,
                ("SOL/USDT", pnl * 100.0, pnl, sc),
            )
            conn.execute(
                """
                INSERT INTO ai_outcome_training_rows (
                    symbol, opened_at_utc, closed_at_utc, ingested_at_utc, strategy_id,
                    good_bad_memory_class, net_pnl_pct, score_components_json, context_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    "SOL/USDT",
                    f"2026-11-25T0{i}:00:00Z",
                    f"2026-11-25T0{i}:15:00Z",
                    f"2026-11-25T0{i}:16:00Z",
                    "day",
                    cls,
                    pnl,
                    sc,
                    json.dumps({"market_regime": reg}),
                ),
            )
        conn.commit()
    return db


def test_backfill_populates_weights_table(tmp_path):
    db = _seed_sol_outcomes_db(tmp_path)
    before = adaptive_weights_row_count(db)
    stats = backfill_adaptive_score_weights_from_outcomes(db, limit=200)
    after = adaptive_weights_row_count(db)
    assert stats["pairs_processed"] >= 1
    assert after > before
    assert stats["buckets_updated"] >= 1
    assert stats["weight_rows"] >= 1


def test_adaptive_weights_auto_enable_when_rows_exist(tmp_path, monkeypatch):
    monkeypatch.delenv("ADAPTIVE_SCORE_WEIGHT_ENABLED", raising=False)
    db = _seed_sol_outcomes_db(tmp_path)
    backfill_adaptive_score_weights_from_outcomes(db, limit=200)
    assert adaptive_weights_row_count(db) > 0
    assert _adaptive_weights_enabled(db_path=db) is True


def test_propagate_after_close_does_not_raise(tmp_path):
    db = _seed_sol_outcomes_db(tmp_path)
    touched = propagate_adaptive_score_weights_for_close(
        symbol="SOL/USDT",
        strategy_id="day",
        explainability={"day_route_regime": "range", "ai_confidence": 0.51, "entry_buy_margin": 0.03},
        db_path=db,
    )
    assert touched >= 1


def test_propagate_writes_current_regime_bucket_not_unknown(tmp_path):
    """All symbol samples should update the close regime bucket (no unknown fallback)."""
    db = _seed_sol_outcomes_db(tmp_path)
    touched = propagate_adaptive_score_weights_for_close(
        symbol="SOL/USDT",
        strategy_id="day",
        explainability={"day_route_regime": "bull", "ai_confidence": 0.55, "entry_buy_margin": 0.04},
        db_path=db,
    )
    assert touched >= 1
    with sqlite3.connect(db) as conn:
        bull_rows = conn.execute("SELECT COUNT(*) FROM ai_strategy_score_weights WHERE symbol='SOLUSDT' AND LOWER(regime) LIKE 'bull::%'").fetchone()[0]
        unknown_rows = conn.execute("SELECT COUNT(*) FROM ai_strategy_score_weights WHERE symbol='SOLUSDT' AND LOWER(regime) LIKE 'unknown%'").fetchone()[0]
        assert bull_rows >= 1
        assert unknown_rows == 0
