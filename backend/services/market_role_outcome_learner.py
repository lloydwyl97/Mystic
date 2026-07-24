"""
Market-Role Outcome Learner — connects closed trade outcomes to context features.

Architecture
============
When a DAY or SCALP trade closes:
  1. The SELL path calls record_trade_outcome() with realized PnL, MFE, MAE,
     hold duration, exit reason, and the entry context snapshot JSON.
  2. Each context feature's direction at entry is compared with the outcome.
  3. Running statistics are updated in `market_role_trade_outcomes` (raw records)
     and `market_role_outcome_stats` (per symbol/strategy/feature aggregates).

Learned adjustment formula
==========================
For each (symbol, strategy, feature):

  correlation = Pearson(feature_value_normed, pnl_pct) over last N outcomes
  learned_adj = clamp(correlation × 0.02, -0.02, +0.02)    if N >= MIN_SAMPLES
              = 0.0                                          if N < MIN_SAMPLES

  Where feature_value_normed is the feature value centered on its neutral point
  (rs=0, momentum=0.5, volatility=0.35, volume_accel=1.0, corr=0.0, beta=1.0).

Total learned adjustment for a symbol = sum of per-feature learned_adj,
clamped to ±0.02 (separate from live_context_adjustment which is capped ±0.06).

Minimum-sample rules
====================
  MIN_OUTCOME_SAMPLES = 10
  CONFIDENCE_FULL_SAMPLES = 50

  confidence = max(0.0, min(1.0, (N - MIN_OUTCOME_SAMPLES) / (CONFIDENCE_FULL_SAMPLES - MIN_OUTCOME_SAMPLES)))
  status     = "insufficient_data" if N < MIN_OUTCOME_SAMPLES else
               "low_confidence"    if confidence < 0.5  else
               "confident"

Feature set tracked
===================
  rs_short_1h, rs_medium_4h, momentum_score, volatility_score,
  volume_accel, btc_correlation, btc_beta, catalyst_score, live_ranking_delta

Safety
======
  - Learned adjustment never blocks a trade.
  - Combined live + learned impact on rank_score is capped at ±0.08 (in market_role_intelligence.py).
  - Old outcomes older than OUTCOME_RETENTION_DAYS are purged on each update.
"""

from __future__ import annotations

import json
import logging
import sqlite3
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

MIN_OUTCOME_SAMPLES: int = 10
CONFIDENCE_FULL_SAMPLES: int = 50
OUTCOME_RETENTION_DAYS: int = 60
MAX_LEARNED_ADJ_PER_FEATURE: float = 0.02
MAX_LEARNED_ADJ_TOTAL: float = 0.02

# Features tracked and their neutral values
_FEATURE_NEUTRALS: dict[str, float] = {
    "rs_short_1h":    0.0,
    "rs_medium_4h":   0.0,
    "momentum_score": 0.5,
    "volatility_score": 0.35,
    "volume_accel":   1.0,
    "btc_correlation": 0.0,
    "btc_beta":       1.0,
    "catalyst_score": 0.0,
    "live_ranking_delta": 0.0,
}

_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS market_role_trade_outcomes (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    trade_id TEXT,
    buy_trade_id TEXT,
    symbol TEXT NOT NULL,
    strategy TEXT NOT NULL,
    realized_pnl_pct REAL,
    hold_seconds INTEGER,
    exit_reason TEXT,
    mfe_pct REAL,
    mae_pct REAL,
    market_regime TEXT,
    rs_short_1h REAL,
    rs_medium_4h REAL,
    momentum_score REAL,
    volatility_score REAL,
    volume_accel REAL,
    btc_correlation REAL,
    btc_beta REAL,
    catalyst_score REAL,
    live_ranking_delta REAL,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS market_role_outcome_stats (
    symbol TEXT NOT NULL,
    strategy TEXT NOT NULL,
    feature_name TEXT NOT NULL,
    sample_count INTEGER NOT NULL DEFAULT 0,
    correlation REAL NOT NULL DEFAULT 0.0,
    learned_adjustment REAL NOT NULL DEFAULT 0.0,
    confidence REAL NOT NULL DEFAULT 0.0,
    confidence_status TEXT NOT NULL DEFAULT 'insufficient_data',
    last_updated TEXT,
    PRIMARY KEY (symbol, strategy, feature_name)
);
"""


def _ensure_schema(db_path: str) -> None:
    try:
        with sqlite3.connect(db_path) as conn:
            conn.executescript(_SCHEMA_SQL)
            conn.commit()
    except Exception as exc:
        logger.warning("market_role_outcome_learner schema init failed: %s", exc)


def _confidence(n: int) -> tuple[float, str]:
    if n < MIN_OUTCOME_SAMPLES:
        return 0.0, "insufficient_data"
    raw = (n - MIN_OUTCOME_SAMPLES) / max(1, CONFIDENCE_FULL_SAMPLES - MIN_OUTCOME_SAMPLES)
    conf = max(0.0, min(1.0, raw))
    status = "confident" if conf >= 0.5 else "low_confidence"
    return round(conf, 4), status


def _pearson(x: list[float], y: list[float]) -> float:
    if len(x) < MIN_OUTCOME_SAMPLES or len(y) < MIN_OUTCOME_SAMPLES:
        return 0.0
    ax, ay = np.array(x, dtype=np.float64), np.array(y, dtype=np.float64)
    std_x, std_y = float(np.std(ax)), float(np.std(ay))
    if std_x < 1e-10 or std_y < 1e-10:
        return 0.0
    corr = float(np.corrcoef(ax, ay)[0, 1])
    return 0.0 if not np.isfinite(corr) else corr


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

@dataclass
class LearningStats:
    """Per-symbol learning statistics returned to callers."""
    symbol: str
    strategy: str
    sample_count: int
    learned_adjustment: float          # total bounded adjustment for ranking
    confidence: float
    confidence_status: str             # "insufficient_data" / "low_confidence" / "confident"
    per_feature: dict[str, dict]       # feature → {corr, adj, count, status}


def record_trade_outcome(
    db_path: str,
    *,
    trade_id: str,
    buy_trade_id: str,
    symbol: str,
    strategy: str,
    realized_pnl_pct: float,
    hold_seconds: int,
    exit_reason: str,
    mfe_pct: float | None,
    mae_pct: float | None,
    market_regime: str,
    context_snapshot_json: str | None,
) -> None:
    """
    Record one closed-trade outcome and update running statistics.
    Called from the SELL path after position is closed.
    """
    _ensure_schema(db_path)

    ctx: dict[str, Any] = {}
    if context_snapshot_json:
        try:
            ctx = json.loads(context_snapshot_json)
        except (json.JSONDecodeError, TypeError):
            pass

    def _f(key: str) -> float | None:
        v = ctx.get(key)
        if v is None:
            return None
        try:
            fv = float(v)
            return fv if np.isfinite(fv) else None
        except (TypeError, ValueError):
            return None

    now_iso = datetime.now(timezone.utc).isoformat()
    cutoff_iso = datetime.fromtimestamp(
        time.time() - OUTCOME_RETENTION_DAYS * 86400, tz=timezone.utc
    ).isoformat()

    try:
        with sqlite3.connect(db_path) as conn:
            conn.execute(
                """
                INSERT INTO market_role_trade_outcomes
                  (trade_id, buy_trade_id, symbol, strategy, realized_pnl_pct,
                   hold_seconds, exit_reason, mfe_pct, mae_pct, market_regime,
                   rs_short_1h, rs_medium_4h, momentum_score, volatility_score,
                   volume_accel, btc_correlation, btc_beta, catalyst_score,
                   live_ranking_delta, created_at)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                """,
                (
                    trade_id, buy_trade_id, symbol.upper(), strategy.lower(),
                    float(realized_pnl_pct), int(hold_seconds), str(exit_reason),
                    mfe_pct, mae_pct, market_regime,
                    _f("rs_short_1h"), _f("rs_medium_4h"),
                    _f("momentum_score"), _f("volatility_score"),
                    _f("volume_accel"), _f("btc_correlation"),
                    _f("btc_beta"), _f("catalyst_score"),
                    _f("live_context_adjustment"),
                    now_iso,
                ),
            )
            # Purge old records
            conn.execute(
                "DELETE FROM market_role_trade_outcomes WHERE created_at < ? AND symbol = ? AND strategy = ?",
                (cutoff_iso, symbol.upper(), strategy.lower()),
            )
            conn.commit()

        # Recompute stats after new record
        _recompute_stats(db_path, symbol.upper(), strategy.lower())

    except Exception as exc:
        logger.warning("record_trade_outcome %s/%s failed: %s", symbol, strategy, exc)


def _recompute_stats(db_path: str, symbol: str, strategy: str) -> None:
    """Recompute per-feature statistics from all outcome records for symbol+strategy."""
    try:
        with sqlite3.connect(db_path) as conn:
            rows = conn.execute(
                """
                SELECT realized_pnl_pct,
                       rs_short_1h, rs_medium_4h, momentum_score, volatility_score,
                       volume_accel, btc_correlation, btc_beta, catalyst_score,
                       live_ranking_delta
                FROM market_role_trade_outcomes
                WHERE symbol = ? AND strategy = ?
                ORDER BY created_at ASC
                """,
                (symbol, strategy),
            ).fetchall()

        if not rows:
            return

        pnl_pcts = [r[0] for r in rows if r[0] is not None]
        n = len(pnl_pcts)
        now_iso = datetime.now(timezone.utc).isoformat()

        feature_cols = [
            "rs_short_1h", "rs_medium_4h", "momentum_score", "volatility_score",
            "volume_accel", "btc_correlation", "btc_beta", "catalyst_score",
            "live_ranking_delta",
        ]
        col_idx = {name: idx + 1 for idx, name in enumerate(feature_cols)}

        with sqlite3.connect(db_path) as conn:
            for feat in feature_cols:
                neutral = _FEATURE_NEUTRALS.get(feat, 0.0)
                col = col_idx[feat]
                # Extract feature values aligned with pnl_pcts (only rows where both are non-null)
                pairs = [
                    (float(r[col]) - neutral, r[0])
                    for r in rows
                    if r[col] is not None and r[0] is not None
                ]
                nf = len(pairs)
                conf, status = _confidence(nf)

                if nf < MIN_OUTCOME_SAMPLES:
                    corr, adj = 0.0, 0.0
                else:
                    feat_vals = [p[0] for p in pairs]
                    pnl_vals = [p[1] for p in pairs]
                    corr = _pearson(feat_vals, pnl_vals)
                    adj = round(
                        max(-MAX_LEARNED_ADJ_PER_FEATURE,
                            min(MAX_LEARNED_ADJ_PER_FEATURE, corr * 0.02)),
                        6,
                    )

                conn.execute(
                    """
                    INSERT INTO market_role_outcome_stats
                      (symbol, strategy, feature_name, sample_count,
                       correlation, learned_adjustment, confidence, confidence_status,
                       last_updated)
                    VALUES (?,?,?,?,?,?,?,?,?)
                    ON CONFLICT(symbol, strategy, feature_name) DO UPDATE SET
                      sample_count = excluded.sample_count,
                      correlation = excluded.correlation,
                      learned_adjustment = excluded.learned_adjustment,
                      confidence = excluded.confidence,
                      confidence_status = excluded.confidence_status,
                      last_updated = excluded.last_updated
                    """,
                    (symbol, strategy, feat, nf, round(corr, 6), adj, conf, status, now_iso),
                )
            conn.commit()

    except Exception as exc:
        logger.warning("_recompute_stats %s/%s failed: %s", symbol, strategy, exc)


def get_learning_stats(
    db_path: str,
    symbol: str,
    strategy: str,
) -> LearningStats:
    """
    Return current learning statistics for (symbol, strategy).
    If no data: returns zero adjustment with insufficient_data status.
    """
    _ensure_schema(db_path)
    sym = symbol.upper().replace("/", "")
    strat = strategy.lower()

    per_feature: dict[str, dict] = {}
    total_adj = 0.0
    max_samples = 0

    try:
        with sqlite3.connect(db_path) as conn:
            rows = conn.execute(
                """
                SELECT feature_name, sample_count, correlation, learned_adjustment,
                       confidence, confidence_status
                FROM market_role_outcome_stats
                WHERE symbol = ? AND strategy = ?
                """,
                (sym, strat),
            ).fetchall()

        for feat, cnt, corr, adj, conf, status in rows:
            per_feature[feat] = {
                "sample_count": cnt,
                "correlation": round(corr, 4),
                "learned_adjustment": round(adj, 6),
                "confidence": round(conf, 4),
                "confidence_status": status,
            }
            if cnt > max_samples:
                max_samples = cnt
            total_adj += adj

    except Exception as exc:
        logger.debug("get_learning_stats %s/%s failed: %s", symbol, strategy, exc)

    total_adj = round(max(-MAX_LEARNED_ADJ_TOTAL, min(MAX_LEARNED_ADJ_TOTAL, total_adj)), 6)
    conf, status = _confidence(max_samples)

    return LearningStats(
        symbol=sym,
        strategy=strat,
        sample_count=max_samples,
        learned_adjustment=total_adj,
        confidence=conf,
        confidence_status=status,
        per_feature=per_feature,
    )


def get_learned_adjustment(db_path: str, symbol: str, strategy: str) -> float:
    """
    Fast accessor: return total bounded learned adjustment for ranking.
    Returns 0.0 on any failure or when samples are insufficient.
    """
    try:
        stats = get_learning_stats(db_path, symbol, strategy)
        return stats.learned_adjustment if stats.sample_count >= MIN_OUTCOME_SAMPLES else 0.0
    except Exception:
        return 0.0


__all__ = [
    "MIN_OUTCOME_SAMPLES",
    "CONFIDENCE_FULL_SAMPLES",
    "LearningStats",
    "get_learned_adjustment",
    "get_learning_stats",
    "record_trade_outcome",
]
