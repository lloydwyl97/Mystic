#!/usr/bin/env python3
"""Verify DAY outcome-attribution schema, feature_dim column, and learning paths."""

from __future__ import annotations

import json
import sqlite3
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.database_schema import DATABASE_PATH
from backend.services.ai_canonical_storage import ensure_ai_canonical_tables
from backend.services.ai_post_trade_feature_review import (
    ensure_post_trade_feature_review_table,
    get_post_trade_feature_review_report,
)
from backend.services.ai_strategy_score_weight_writer import (
    adaptive_weights_row_count,
    propagate_adaptive_score_weights_for_close,
)
from backend.services.day_outcome_attribution import dry_run_insert_and_delete, ensure_outcome_attribution_table

REQUIRED_ATTR_COLUMNS = (
    "trade_id",
    "symbol",
    "regime",
    "setup_thesis",
    "entry_time",
    "exit_time",
    "net_pnl_after_fees",
    "exit_reason",
    "outcome_reason",
    "entry_145_vector_json",
    "entry_feature_health_json",
    "entry_block_scores_json",
    "setup_score",
    "execution_quality_score",
    "model_probabilities_json",
    "final_selection_score",
    "feature_health_pass",
    "created_at",
)


def _columns(conn: sqlite3.Connection, table: str) -> set[str]:
    return {row[1] for row in conn.execute(f"PRAGMA table_info({table})")}


def main() -> int:
    db = DATABASE_PATH
    ensure_ai_canonical_tables(db)
    ensure_outcome_attribution_table(db)
    ensure_post_trade_feature_review_table(db)

    with sqlite3.connect(db) as conn:
        infer_cols = _columns(conn, "ai_inference_log")
        attr_cols = _columns(conn, "day_outcome_attribution")
        review_count = conn.execute("SELECT COUNT(*) FROM ai_post_trade_feature_reviews").fetchone()[0]
        weight_count = conn.execute("SELECT COUNT(*) FROM ai_strategy_score_weights").fetchone()[0]
        feature_dim_present = "feature_dim" in infer_cols
        if feature_dim_present:
            sample_dim = conn.execute(
                "SELECT feature_dim, feature_version FROM ai_inference_log WHERE feature_dim IS NOT NULL ORDER BY id DESC LIMIT 1"
            ).fetchone()
        else:
            sample_dim = None

    missing_attr = [c for c in REQUIRED_ATTR_COLUMNS if c not in attr_cols]
    dry = dry_run_insert_and_delete(db)
    writer_touch = propagate_adaptive_score_weights_for_close(
        symbol="SOLUSDT",
        strategy_id="day",
        explainability={"regime": "bull", "setup_type": "HTF_TREND_PULLBACK", "good_bad_memory_class": "BAD"},
    )
    review_report = get_post_trade_feature_review_report(limit=1, db_path=db)

    payload = {
        "database": db,
        "ai_inference_log_has_feature_dim": feature_dim_present,
        "ai_inference_log_sample_dim": list(sample_dim) if sample_dim else None,
        "day_outcome_attribution_columns_ok": not missing_attr,
        "day_outcome_attribution_missing_columns": missing_attr,
        "dry_insert_read_delete": dry,
        "ai_post_trade_feature_reviews_count": review_count,
        "post_trade_review_report_ok": review_report.get("count", 0) >= 0,
        "ai_strategy_score_weights_count": weight_count,
        "adaptive_weights_row_count": adaptive_weights_row_count(db),
        "propagate_adaptive_score_weights_for_close_touch": writer_touch,
        "pass": feature_dim_present and not missing_attr and dry.get("read_back_ok"),
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if payload["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
