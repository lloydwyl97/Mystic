#!/usr/bin/env python3
"""Verify SCALP outcome attribution schema and learning paths."""

from __future__ import annotations

import json
import sqlite3
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.services.binance_scalp.config import get_scalp_config
from backend.services.scalp_outcome_attribution import dry_run_insert_and_delete, ensure_scalp_outcome_attribution_table
from backend.services.scalp_post_trade_feature_review import ensure_scalp_post_trade_review_table, record_scalp_post_trade_review
from backend.services.scalp_strategy_score_weight_writer import (
    ensure_scalp_strategy_score_weights_table,
    propagate_scalp_adaptive_weights_for_close,
    scalp_weights_row_count,
)

REQUIRED = (
    "trade_id",
    "symbol",
    "scalp_setup",
    "micro_regime",
    "entry_time",
    "exit_time",
    "net_pnl_after_fees",
    "exit_reason",
    "outcome_reason",
    "entry_scalp_vector_json",
    "entry_feature_health_json",
    "entry_block_scores_json",
    "setup_score",
    "execution_quality_score",
    "final_scalp_selection_score",
    "feature_health_pass",
    "created_at",
)


def main() -> int:
    cfg = get_scalp_config()
    ensure_scalp_outcome_attribution_table(cfg.database_path)
    ensure_scalp_strategy_score_weights_table(cfg.database_path)
    ensure_scalp_post_trade_review_table(cfg.database_path)
    with sqlite3.connect(cfg.database_path) as conn:
        cols = {r[1] for r in conn.execute("PRAGMA table_info(scalp_outcome_attribution)")}
    missing = [c for c in REQUIRED if c not in cols]
    dry = dry_run_insert_and_delete(cfg.database_path)
    review = record_scalp_post_trade_review(
        trade_id="__scalp_review_verify__",
        symbol="BTCUSDT",
        closed_at_utc="2026-01-01T00:00:00+00:00",
        intelligence={"scalp_setup": "MICRO_BREAKOUT", "micro_regime": "bull_trend"},
        net_pnl=0.5,
        hold_seconds=60,
        db_path=cfg.database_path,
    )
    with sqlite3.connect(cfg.database_path) as conn:
        conn.execute("DELETE FROM scalp_post_trade_feature_reviews WHERE trade_id='__scalp_review_verify__'")
        conn.commit()
    touched = propagate_scalp_adaptive_weights_for_close(
        symbol="BTCUSDT",
        intelligence={
            "micro_regime": "bull_trend",
            "scalp_setup": "MICRO_BREAKOUT",
            "setup_name": "breakout_momentum",
            "feature_health_pass": True,
            "feature_health_json": '{"pass":true,"health_pct":90}',
            "signal_score": 0.6,
            "mid_change_30s": 0.001,
        },
        net_pnl=0.25,
        db_path=cfg.database_path,
    )
    payload = {
        "database": cfg.database_path,
        "columns_ok": not missing,
        "missing_columns": missing,
        "dry_insert": dry,
        "post_trade_review_row": review,
        "weights_touched": touched,
        "scalp_weights_rows": scalp_weights_row_count(cfg.database_path),
        "pass": not missing and dry.get("read_back_ok"),
    }
    print(json.dumps(payload, indent=2))
    return 0 if payload["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
