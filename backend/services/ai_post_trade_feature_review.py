"""
Post-trade feature review storage for AI sells — learning/observation only.

Does not change buy/sell rules, thresholds, or execution gates.
"""

from __future__ import annotations

import contextlib
import json
import logging
import math
import sqlite3
from datetime import datetime, timezone
from typing import Any

from backend.config.trading_economics import MIN_NET_PROFIT_TO_SELL
from backend.database_schema import DATABASE_PATH
from backend.services.ai_canonical_storage import ensure_ai_canonical_tables
from backend.services.ai_feature_freshness_diagnostics import (
    IMPORTANCE_ROLLUP,
    build_feature_age_by_block,
)

logger = logging.getLogger(__name__)

TABLE = "ai_post_trade_feature_reviews"

_SCHEMA = f"""
CREATE TABLE IF NOT EXISTS {TABLE} (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    trade_id TEXT NOT NULL UNIQUE,
    symbol TEXT NOT NULL,
    closed_at_utc TEXT NOT NULL,
    review_json TEXT NOT NULL,
    ingested_at_utc TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS ix_post_trade_review_symbol ON {TABLE}(symbol, closed_at_utc);
"""


def ensure_post_trade_feature_review_table(db_path: str = DATABASE_PATH) -> None:
    ensure_ai_canonical_tables(db_path)
    with sqlite3.connect(db_path) as conn:
        for stmt in _SCHEMA.strip().split(";\n"):
            s = stmt.strip()
            if s:
                conn.execute(s)
        conn.commit()


def _block_for_index(idx0: int) -> str:
    from backend.services.ai_decision_contract import AI_FEATURE_DIM_V1
    from backend.services.ai_feature_freshness_diagnostics import FEATURE_BLOCKS

    if idx0 >= AI_FEATURE_DIM_V1:
        return "context_day_full"
    one_based = idx0 + 1
    for block, (lo, hi) in FEATURE_BLOCKS.items():
        if lo <= one_based <= hi:
            return block
    return "unknown"


def _strongest_entry_blocks(features: list[float] | None) -> list[dict[str, Any]]:
    if not features or len(features) < 10:
        return []
    rollup_strength: dict[str, float] = dict.fromkeys(IMPORTANCE_ROLLUP, 0.0)
    for i, raw in enumerate(features[:145]):
        try:
            v = abs(float(raw))
        except (TypeError, ValueError):
            continue
        if not (v > 0 and math.isfinite(v)):
            continue
        block = _block_for_index(i)
        for rollup, members in IMPORTANCE_ROLLUP.items():
            if block in members:
                rollup_strength[rollup] += v
                break
    ranked = sorted(rollup_strength.items(), key=lambda x: x[1], reverse=True)
    return [{"block": k, "strength": round(v, 6)} for k, v in ranked if v > 1e-9][:8]


def _lookup_entry_features(
    db_path: str,
    *,
    decision_id: str,
    symbol: str,
    opened_at_utc: str | None = None,
    window_sec: int = 900,
) -> list[float] | None:
    try:
        with sqlite3.connect(db_path) as conn:
            row = None
            bus = str(symbol or "").replace("/", "").upper()
            if decision_id:
                row = conn.execute(
                    """
                    SELECT features_json FROM ai_inference_log
                    WHERE decision_id=? AND features_json IS NOT NULL
                      AND (
                        UPPER(REPLACE(symbol, '/', '')) = ?
                        OR UPPER(symbol) = ?
                        OR symbol = ?
                      )
                    ORDER BY id DESC LIMIT 1
                    """,
                    (decision_id, bus, bus, symbol),
                ).fetchone()
                if not row:
                    row = conn.execute(
                        """
                        SELECT features_json FROM ai_inference_log
                        WHERE decision_id=? AND features_json IS NOT NULL
                        ORDER BY id DESC LIMIT 1
                        """,
                        (decision_id,),
                    ).fetchone()
            if (not row or not row[0]) and opened_at_utc:
                row = conn.execute(
                    """
                    SELECT features_json FROM ai_inference_log
                    WHERE features_json IS NOT NULL AND length(features_json) > 2
                      AND (
                        UPPER(REPLACE(symbol, '/', '')) = ?
                        OR UPPER(symbol) = ?
                        OR symbol = ?
                      )
                      AND ABS(
                        (julianday(REPLACE(SUBSTR(ts_utc, 1, 19), 'T', ' ')) -
                         julianday(REPLACE(SUBSTR(?, 1, 19), 'T', ' '))) * 86400.0
                      ) <= ?
                    ORDER BY ABS(
                      (julianday(REPLACE(SUBSTR(ts_utc, 1, 19), 'T', ' ')) -
                       julianday(REPLACE(SUBSTR(?, 1, 19), 'T', ' '))) * 86400.0
                    ) ASC
                    LIMIT 1
                    """,
                    (bus, bus, symbol, opened_at_utc, float(window_sec), opened_at_utc),
                ).fetchone()
            if not row or not row[0]:
                return None
            parsed = json.loads(row[0])
            if isinstance(parsed, list):
                return [float(x) for x in parsed[:145]]
    except Exception:
        pass
    return None


def record_post_trade_feature_review(
    *,
    trade_id: str,
    symbol: str,
    closed_at_utc: str,
    explainability: dict[str, Any] | None,
    hold_seconds: float | None,
    net_profit_usd: float | None,
    net_profit_pct: float | None,
    repair_add_count: int = 0,
    db_path: str = DATABASE_PATH,
) -> int | None:
    """Persist one sell review row. Never raises."""
    if not trade_id:
        return None
    try:
        ensure_post_trade_feature_review_table(db_path)
        ex = explainability or {}
        bus = str(ex.get("symbol_canonical_no_slash") or symbol or "").replace("/", "").upper()
        if not bus.endswith("USDT") and bus:
            bus = f"{bus}USDT"
        decision_id = str(ex.get("decision_id") or ex.get("entry_decision_id") or "")
        features = _lookup_entry_features(db_path, decision_id=decision_id, symbol=bus or symbol)
        strongest = _strongest_entry_blocks(features)
        if not strongest:
            strongest = [
                {"block": "signal_regime", "strength": abs(float(ex.get("signal_regime_score") or 0))},
                {"block": "signal_rsi", "strength": abs(float(ex.get("signal_rsi_1m") or 0)) / 100.0},
                {"block": "signal_spread", "strength": abs(float(ex.get("signal_spread_pct") or ex.get("entry_spread_pct") or 0))},
                {"block": "signal_depth", "strength": abs(float(ex.get("signal_ctx_depth_imbalance") or 0))},
            ]
            strongest = sorted(strongest, key=lambda x: float(x.get("strength") or 0), reverse=True)[:8]

        age_report = build_feature_age_by_block(bus or symbol)
        review = {
            "trade_id": trade_id,
            "symbol": bus or symbol,
            "closed_at_utc": closed_at_utc,
            "strongest_entry_feature_blocks": strongest,
            "active_regime": str(ex.get("regime") or ex.get("signal_regime_label") or ex.get("price_structure_regime") or ""),
            "sentiment_freshness": {
                "ages_sec": age_report.get("ages_sec"),
                "freshness_status": age_report.get("freshness_status"),
            },
            "orderbook_spread_pct": float(ex.get("entry_spread_pct") or ex.get("signal_spread_pct") or 0),
            "orderbook_impact_proxy": float(ex.get("entry_slippage_pct") or 0),
            "repair_add_used": bool(repair_add_count > 0),
            "repair_add_count": int(repair_add_count),
            "hold_time_sec": round(float(hold_seconds or 0), 1),
            "executable_net_profit_usd": float(net_profit_usd) if net_profit_usd is not None else None,
            "executable_net_profit_pct": float(net_profit_pct) if net_profit_pct is not None else None,
            "hit_profit_floor": (net_profit_pct is not None and float(net_profit_pct) >= float(MIN_NET_PROFIT_TO_SELL)),
            "model_version": {
                "feature_version": int(ex.get("feature_version") or 0),
                "feature_dim": int(ex.get("feature_dim") or 0),
                "artifact_path": str(ex.get("artifact_path") or ""),
                "artifact_sha256": str(ex.get("artifact_sha256") or ""),
                "model_trained_at": str(ex.get("model_trained_at") or ""),
                "model_accuracy": ex.get("model_accuracy"),
            },
            "ctx_age_sec": float(ex.get("ctx_age_sec") or -1),
            "context_fresh_flag": str(ex.get("context_fresh_flag") or ""),
        }
        ingested = datetime.now(timezone.utc).isoformat()
        with sqlite3.connect(db_path) as conn:
            cur = conn.execute(
                f"""
                INSERT INTO {TABLE} (trade_id, symbol, closed_at_utc, review_json, ingested_at_utc)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(trade_id) DO UPDATE SET
                    review_json=excluded.review_json,
                    ingested_at_utc=excluded.ingested_at_utc
                """,
                (trade_id, bus or symbol, closed_at_utc, json.dumps(review, separators=(",", ":")), ingested),
            )
            conn.commit()
            return int(cur.lastrowid or 0) or None
    except Exception as exc:
        logger.debug("record_post_trade_feature_review failed trade_id=%s: %s", trade_id, exc)
        return None


def get_post_trade_feature_review_report(limit: int = 50, db_path: str = DATABASE_PATH) -> dict[str, Any]:
    ensure_post_trade_feature_review_table(db_path)
    limit = max(1, min(200, int(limit)))
    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            f"SELECT trade_id, symbol, closed_at_utc, review_json FROM {TABLE} ORDER BY id DESC LIMIT ?",
            (limit,),
        ).fetchall()
    reviews: list[dict[str, Any]] = []
    for r in rows:
        try:
            body = json.loads(r["review_json"] or "{}")
        except (json.JSONDecodeError, TypeError):
            body = {}
        reviews.append(
            {
                "trade_id": r["trade_id"],
                "symbol": r["symbol"],
                "closed_at_utc": r["closed_at_utc"],
                **body,
            }
        )
    return {
        "observation_only": True,
        "count": len(reviews),
        "reviews": reviews,
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }


def backfill_post_trade_feature_reviews(
    *,
    limit: int = 10,
    db_path: str = DATABASE_PATH,
) -> dict[str, Any]:
    """
    Backfill recent real AI sells when entry features/audit exist.
    Skips admin clears and rows without decision_id or entry audit data.
    """
    ensure_post_trade_feature_review_table(db_path)
    limit = max(1, min(50, int(limit)))
    inserted = 0
    skipped: list[dict[str, str]] = []
    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        sells = conn.execute(
            """
            SELECT trade_id, symbol, timestamp, pnl, pnl_pct, explainability_json,
                   exit_type, entry_timestamp
            FROM paper_trades
            WHERE side='SELL' AND pnl IS NOT NULL
              AND COALESCE(exit_type,'') NOT IN ('ADMIN_POSITION_CLEAR', 'STALE_PRE_CORRECTION_POSITION_CLEAR')
            ORDER BY timestamp DESC LIMIT ?
            """,
            (limit * 3,),
        ).fetchall()

    for s in sells:
        if inserted >= limit:
            break
        trade_id = str(s["trade_id"] or "")
        if not trade_id:
            skipped.append({"trade_id": "", "reason": "missing_trade_id"})
            continue
        try:
            explain = json.loads(s["explainability_json"] or "{}")
        except (json.JSONDecodeError, TypeError):
            explain = {}
        if not isinstance(explain, dict):
            explain = {}
        decision_id = str(explain.get("decision_id") or explain.get("entry_decision_id") or s["decision_id"] or "")
        if not decision_id:
            skipped.append({"trade_id": trade_id, "reason": "missing_decision_id"})
            continue
        sym = str(s["symbol"] or "")
        bus = str(explain.get("symbol_canonical_no_slash") or sym or "").replace("/", "").upper()
        if not bus.endswith("USDT") and bus:
            bus = f"{bus}USDT"
        features = _lookup_entry_features(db_path, decision_id=decision_id, symbol=bus or sym)
        if not features:
            skipped.append({"trade_id": trade_id, "reason": "missing_entry_features"})
            continue
        entry_ts = s["entry_timestamp"]
        hold_seconds = None
        if entry_ts and s["timestamp"]:
            with contextlib.suppress(Exception):
                et = datetime.fromisoformat(str(entry_ts).replace("Z", "+00:00"))
                ct = datetime.fromisoformat(str(s["timestamp"]).replace("Z", "+00:00"))
                hold_seconds = max(0.0, (ct - et).total_seconds())
        row_id = record_post_trade_feature_review(
            trade_id=trade_id,
            symbol=sym,
            closed_at_utc=str(s["timestamp"] or ""),
            explainability=explain,
            hold_seconds=hold_seconds,
            net_profit_usd=float(s["pnl"]) if s["pnl"] is not None else None,
            net_profit_pct=float(s["pnl_pct"]) if s["pnl_pct"] is not None else None,
            repair_add_count=int(explain.get("repair_add_count") or 0),
            db_path=db_path,
        )
        if row_id:
            inserted += 1
        else:
            skipped.append({"trade_id": trade_id, "reason": "insert_failed"})

    return {
        "requested": limit,
        "inserted": inserted,
        "skipped": skipped[:20],
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }


__all__ = [
    "backfill_post_trade_feature_reviews",
    "ensure_post_trade_feature_review_table",
    "get_post_trade_feature_review_report",
    "record_post_trade_feature_review",
]
