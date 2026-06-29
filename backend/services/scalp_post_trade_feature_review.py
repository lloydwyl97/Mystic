"""SCALP post-trade feature review — observation/learning only."""

from __future__ import annotations

import json
import logging
import sqlite3
from datetime import datetime, timezone
from typing import Any

from backend.services.binance_scalp.config import get_scalp_config

logger = logging.getLogger(__name__)

TABLE = "scalp_post_trade_feature_reviews"

_CREATE = f"""
CREATE TABLE IF NOT EXISTS {TABLE} (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    trade_id TEXT NOT NULL UNIQUE,
    symbol TEXT NOT NULL,
    closed_at_utc TEXT NOT NULL,
    review_json TEXT NOT NULL,
    ingested_at_utc TEXT NOT NULL
)
"""


def ensure_scalp_post_trade_review_table(db_path: str | None = None) -> None:
    path = db_path or get_scalp_config().database_path
    with sqlite3.connect(path) as conn:
        conn.execute(_CREATE)
        conn.execute(f"CREATE INDEX IF NOT EXISTS ix_scalp_ptr_symbol ON {TABLE}(symbol, closed_at_utc)")
        conn.commit()


def record_scalp_post_trade_review(
    *,
    trade_id: str,
    symbol: str,
    closed_at_utc: str,
    intelligence: dict[str, Any] | None,
    net_pnl: float,
    hold_seconds: float,
    db_path: str | None = None,
) -> int | None:
    if not trade_id:
        return None
    path = db_path or get_scalp_config().database_path
    try:
        ensure_scalp_post_trade_review_table(path)
        ex = intelligence or {}
        review = {
            "trade_id": trade_id,
            "symbol": symbol,
            "closed_at_utc": closed_at_utc,
            "scalp_setup": ex.get("scalp_setup"),
            "micro_regime": ex.get("micro_regime"),
            "feature_health_score": ex.get("scalp_feature_health_score"),
            "execution_quality_score": ex.get("scalp_execution_quality_score"),
            "outcome_reason": ex.get("outcome_reason"),
            "net_pnl": net_pnl,
            "hold_seconds": hold_seconds,
            "explanation": ex.get("scalp_candidate_explanation_json"),
        }
        ingested = datetime.now(timezone.utc).isoformat()
        with sqlite3.connect(path) as conn:
            cur = conn.execute(
                f"""
                INSERT INTO {TABLE} (trade_id, symbol, closed_at_utc, review_json, ingested_at_utc)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(trade_id) DO UPDATE SET review_json=excluded.review_json, ingested_at_utc=excluded.ingested_at_utc
                """,
                (trade_id, symbol, closed_at_utc, json.dumps(review, separators=(",", ":")), ingested),
            )
            conn.commit()
            return int(cur.lastrowid or 0) or None
    except Exception as exc:
        logger.debug("record_scalp_post_trade_review failed: %s", exc)
        return None


__all__ = ["ensure_scalp_post_trade_review_table", "record_scalp_post_trade_review"]
