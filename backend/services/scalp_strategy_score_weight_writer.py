"""SCALP setup-aware adaptive learning weights — separate from DAY ai_strategy_score_weights."""

from __future__ import annotations

import json
import logging
import sqlite3
import time
from datetime import datetime, timezone
from typing import Any

from backend.services.binance_scalp.config import get_scalp_config
from backend.services.scalp_feature_contract import STRATEGY_TO_SCALP_SETUP, scalp_setup_bucket
from backend.services.scalp_feature_health import entry_feature_health_pass

logger = logging.getLogger(__name__)

TABLE = "scalp_strategy_score_weights"
LEARN_ALPHA = 0.18
MAX_WEIGHT_DELTA = 0.10
MIN_BUCKET_SAMPLES = 3

_COMPONENTS: tuple[str, ...] = (
    "model_probability",
    "buy_margin",
    "micro_momentum",
    "volume_burst",
    "spread_quality",
    "depth_quality",
    "execution_quality",
    "relative_strength",
    "setup_quality",
    "symbol_expectancy",
    "recent_slippage",
    "hold_time_quality",
)

_CREATE = f"""
CREATE TABLE IF NOT EXISTS {TABLE} (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    strategy_id TEXT NOT NULL DEFAULT 'scalp',
    symbol TEXT NOT NULL DEFAULT '',
    regime TEXT NOT NULL DEFAULT '',
    component_name TEXT NOT NULL,
    weight REAL NOT NULL DEFAULT 0.0,
    previous_weight REAL NOT NULL DEFAULT 0.0,
    sample_count INTEGER NOT NULL DEFAULT 0,
    good_count INTEGER NOT NULL DEFAULT 0,
    bad_count INTEGER NOT NULL DEFAULT 0,
    net_expectancy REAL NOT NULL DEFAULT 0.0,
    updated_at TEXT NOT NULL DEFAULT '',
    UNIQUE(strategy_id, symbol, regime, component_name)
)
"""


def ensure_scalp_strategy_score_weights_table(db_path: str | None = None) -> None:
    path = db_path or get_scalp_config().database_path
    with sqlite3.connect(path) as conn:
        conn.execute(_CREATE)
        conn.execute(f"CREATE INDEX IF NOT EXISTS ix_scalp_weights_sym ON {TABLE}(symbol, regime)")
        conn.commit()


def _regime_bucket(intelligence: dict[str, Any]) -> str:
    micro = str(intelligence.get("micro_regime") or "chop")
    setup = str(intelligence.get("scalp_setup") or intelligence.get("setup_name") or "")
    setup = STRATEGY_TO_SCALP_SETUP.get(setup, setup).upper()
    if "::" in setup:
        return setup
    return scalp_setup_bucket(micro, setup)


def extract_scalp_components(intelligence: dict[str, Any]) -> dict[str, float]:
    ex = intelligence or {}
    spread = float(ex.get("spread_pct") or 0.0)
    return {
        "model_probability": float(ex.get("signal_confidence") or ex.get("signal_score") or 0.5) * 100.0,
        "buy_margin": float(ex.get("required_target_pct") or 0.0) * 1000.0,
        "micro_momentum": float(ex.get("mid_change_30s") or 0.0) * 10000.0,
        "volume_burst": float(ex.get("kline_volume_ratio") or 1.0) * 10.0,
        "spread_quality": max(0.0, 1.0 - spread / 0.003) * 100.0,
        "depth_quality": abs(float(ex.get("order_book_imbalance") or 0.0)) * 100.0,
        "execution_quality": float(ex.get("scalp_execution_quality_score") or 0.5) * 100.0,
        "relative_strength": float(ex.get("scalp_rs_rank") or ex.get("relative_strength_rank") or 0.5) * 20.0,
        "setup_quality": float(ex.get("setup_score") or 0.5) * 100.0,
        "symbol_expectancy": float(ex.get("recent_scalp_win_rate") or 0.5) * 100.0,
        "recent_slippage": -abs(float(ex.get("realized_slippage") or ex.get("slippage_estimate") or 0.0)) * 10000.0,
        "hold_time_quality": -abs(float(ex.get("hold_seconds") or 0.0) - 180.0),
    }


def propagate_scalp_adaptive_weights_for_close(
    *,
    symbol: str,
    intelligence: dict[str, Any] | None,
    net_pnl: float,
    db_path: str | None = None,
) -> int:
    path = db_path or get_scalp_config().database_path
    ensure_scalp_strategy_score_weights_table(path)
    ex = dict(intelligence or {})
    is_good = net_pnl > 0
    if is_good and not entry_feature_health_pass(ex):
        return 0
    sym = symbol.upper().replace("/", "")
    bucket = _regime_bucket(ex)
    comps = extract_scalp_components(ex)
    touched = 0
    now = datetime.now(timezone.utc).isoformat()
    with sqlite3.connect(path) as conn:
        for comp, raw in comps.items():
            prev_row = conn.execute(
                f"SELECT weight, sample_count, good_count, bad_count FROM {TABLE} WHERE strategy_id='scalp' AND symbol=? AND regime=? AND component_name=?",
                (sym, bucket, comp),
            ).fetchone()
            prev_w = float(prev_row[0]) if prev_row else raw
            n = int(prev_row[1]) if prev_row else 0
            good = int(prev_row[2]) if prev_row else 0
            bad = int(prev_row[3]) if prev_row else 0
            n += 1
            if is_good:
                good += 1
                delta = LEARN_ALPHA * (raw - prev_w)
            else:
                bad += 1
                delta = -LEARN_ALPHA * abs(raw - prev_w) * 0.5
            delta = max(-MAX_WEIGHT_DELTA, min(MAX_WEIGHT_DELTA, delta))
            new_w = prev_w + delta
            conn.execute(
                f"""
                INSERT INTO {TABLE} (strategy_id, symbol, regime, component_name, weight, previous_weight, sample_count, good_count, bad_count, net_expectancy, updated_at)
                VALUES ('scalp', ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(strategy_id, symbol, regime, component_name) DO UPDATE SET
                    previous_weight=excluded.previous_weight,
                    weight=excluded.weight,
                    sample_count=excluded.sample_count,
                    good_count=excluded.good_count,
                    bad_count=excluded.bad_count,
                    net_expectancy=excluded.net_expectancy,
                    updated_at=excluded.updated_at
                """,
                (sym, bucket, comp, new_w, prev_w, n, good, bad, net_pnl, now),
            )
            touched += 1
        conn.commit()
    return touched


def scalp_adaptive_rank_delta(intelligence: dict[str, Any], symbol: str, db_path: str | None = None) -> float:
    path = db_path or get_scalp_config().database_path
    bucket = _regime_bucket(intelligence or {})
    sym = symbol.upper().replace("/", "")
    try:
        with sqlite3.connect(path) as conn:
            rows = conn.execute(
                f"SELECT component_name, weight FROM {TABLE} WHERE strategy_id='scalp' AND symbol=? AND regime=?",
                (sym, bucket),
            ).fetchall()
        if len(rows) < 2:
            return 0.0
        comps = extract_scalp_components(intelligence or {})
        delta = 0.0
        for comp, weight in rows:
            raw = comps.get(str(comp), 0.0)
            delta += (float(weight) - raw) * 0.0001
        return round(max(-0.05, min(0.05, delta)), 4)
    except Exception:
        return 0.0


def scalp_weights_row_count(db_path: str | None = None) -> int:
    path = db_path or get_scalp_config().database_path
    try:
        with sqlite3.connect(path) as conn:
            return int(conn.execute(f"SELECT COUNT(*) FROM {TABLE}").fetchone()[0] or 0)
    except Exception:
        return 0


__all__ = [
    "ensure_scalp_strategy_score_weights_table",
    "propagate_scalp_adaptive_weights_for_close",
    "scalp_adaptive_rank_delta",
    "scalp_weights_row_count",
]
