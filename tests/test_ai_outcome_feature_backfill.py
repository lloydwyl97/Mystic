"""
Regression: AI outcome training rows must be able to recover features_json
from ai_inference_log when the sell-time write is missing them (backfill),
and the OUTCOME_XY_FILTER training pipeline must no longer discard rows for
a deterministically-recoverable feature version.
"""

from __future__ import annotations

import json
import sqlite3
import tempfile
from pathlib import Path

from backend.ai_training_pipeline import _outcome_rows_to_xy_for_strategy
from backend.services.ai_outcome_training_writer import backfill_outcome_features_from_inference


def _make_db(db_path: Path) -> None:
    with sqlite3.connect(str(db_path)) as conn:
        conn.execute(
            """
            CREATE TABLE ai_outcome_training_rows (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT, opened_at_utc TEXT, closed_at_utc TEXT,
                strategy_id TEXT, features_json TEXT, context_json TEXT,
                outcome_label INTEGER, good_bad_memory_class TEXT,
                net_pnl_pct REAL, selected_net_expected_value REAL, rank_snapshot_id TEXT
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE ai_inference_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                decision_id TEXT, symbol TEXT, ts_utc TEXT,
                feature_version INTEGER, feature_dim INTEGER,
                features_json TEXT, strategy_id TEXT
            )
            """
        )
        conn.commit()


def test_backfill_fills_null_features_from_nearest_inference_row():
    with tempfile.TemporaryDirectory() as tmp:
        db_path = Path(tmp) / "backfill.db"
        _make_db(db_path)
        feats = [0.1] * 145
        with sqlite3.connect(str(db_path)) as conn:
            conn.execute(
                "INSERT INTO ai_outcome_training_rows (symbol, opened_at_utc, closed_at_utc, strategy_id, features_json, outcome_label, good_bad_memory_class, net_pnl_pct) "
                "VALUES ('BTC/USDT', '2026-07-01T00:00:00+00:00', '2026-07-01T00:30:00+00:00', 'day', NULL, 1, 'GOOD', 0.01)"
            )
            conn.execute(
                "INSERT INTO ai_inference_log (decision_id, symbol, ts_utc, feature_version, feature_dim, features_json, strategy_id) "
                "VALUES ('d1', 'BTCUSDT', '2026-07-01T00:00:05+00:00', 5, 145, ?, 'day')",
                (json.dumps(feats),),
            )
            conn.commit()

        # Sanity: import module patches DATABASE_PATH default via ensure_ai_canonical_tables which
        # may try to create tables we already made — guard by calling with our db_path explicitly.
        import backend.services.ai_outcome_training_writer as w

        orig_ensure = w.ensure_ai_canonical_tables
        w.ensure_ai_canonical_tables = lambda *_a, **_k: None
        try:
            updated = backfill_outcome_features_from_inference(str(db_path))
        finally:
            w.ensure_ai_canonical_tables = orig_ensure

        assert updated == 1
        with sqlite3.connect(str(db_path)) as conn:
            row = conn.execute("SELECT features_json, context_json FROM ai_outcome_training_rows WHERE symbol='BTC/USDT'").fetchone()
        assert row[0] is not None
        parsed = json.loads(row[0])
        assert len(parsed) == 145
        ctx = json.loads(row[1])
        assert ctx["feature_version"] == 5


def test_outcome_xy_filter_accepts_backfilled_rows():
    """After backfill, OUTCOME_XY_FILTER must count these rows as eligible, not skipped_feature_version."""
    feats = [0.2] * 145
    outcome_rows = [
        {
            "symbol": "ETH/USDT",
            "strategy_id": "day",
            "features_json": json.dumps(feats),
            "context_json": json.dumps({"feature_version": 5, "_live_ai_strategy": "day"}),
            "outcome_label": 1,
            "good_bad_memory_class": "GOOD",
            "net_pnl_pct": 0.01,
            "selected_net_expected_value": None,
            "rank_snapshot_id": None,
        }
    ]
    X, y, syms, w_mult = _outcome_rows_to_xy_for_strategy(outcome_rows, "day", 145)
    assert len(X) == 1
    assert len(w_mult) == 1
    assert syms == ["ETHUSDT"] or syms == ["ETH/USDT"]
