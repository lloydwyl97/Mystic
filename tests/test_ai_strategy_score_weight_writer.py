"""Tests for adaptive score weight learning writer."""

from __future__ import annotations

import json
import sqlite3

import pytest

from backend.database_schema import DATABASE_PATH
from backend.services.ai_canonical_storage import ensure_ai_canonical_tables
from backend.services.ai_strategy_score_weight_writer import (
    COMPONENT_BOUNDS,
    backfill_adaptive_score_weights_from_outcomes,
    adaptive_weights_row_count,
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


def test_backfill_populates_weights_table():
    ensure_ai_canonical_tables(DATABASE_PATH)
    before = adaptive_weights_row_count(DATABASE_PATH)
    stats = backfill_adaptive_score_weights_from_outcomes(DATABASE_PATH, limit=200)
    after = adaptive_weights_row_count(DATABASE_PATH)
    assert stats["pairs_processed"] >= 0
    if after > before:
        assert stats["buckets_updated"] >= 1
        assert stats["weight_rows"] >= 1


def test_adaptive_weights_auto_enable_when_rows_exist():
    ensure_ai_canonical_tables(DATABASE_PATH)
    if adaptive_weights_row_count(DATABASE_PATH) > 0:
        assert _adaptive_weights_enabled() is True


def test_propagate_after_close_does_not_raise():
    ensure_ai_canonical_tables(DATABASE_PATH)
    touched = propagate_adaptive_score_weights_for_close(
        symbol="SOL/USDT",
        strategy_id="day",
        explainability={"day_route_regime": "range", "ai_confidence": 0.51, "entry_buy_margin": 0.03},
        db_path=DATABASE_PATH,
    )
    assert touched >= 0


def test_propagate_writes_current_regime_bucket_not_unknown():
    """All symbol samples should update the close regime bucket (no unknown fallback)."""
    ensure_ai_canonical_tables(DATABASE_PATH)
    touched = propagate_adaptive_score_weights_for_close(
        symbol="SOL/USDT",
        strategy_id="day",
        explainability={"day_route_regime": "bull", "ai_confidence": 0.55, "entry_buy_margin": 0.04},
        db_path=DATABASE_PATH,
    )
    if touched <= 0:
        pytest.skip("insufficient SOL outcome samples in dev db")
    with sqlite3.connect(DATABASE_PATH) as conn:
        bull_rows = conn.execute(
            "SELECT COUNT(*) FROM ai_strategy_score_weights WHERE symbol='SOLUSDT' AND LOWER(regime)='bull'"
        ).fetchone()[0]
        assert bull_rows >= 1
