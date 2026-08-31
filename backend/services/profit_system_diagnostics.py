from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from typing import Any

import redis

from backend.config.trading_universe import get_trading_symbols
from backend.database_schema import DATABASE_PATH
from backend.services.ai_decision_contract import REDIS_KEY_AI_CONTEXT


def _load_latest_snapshot(conn: sqlite3.Connection, strategy_id: str) -> dict[str, Any]:
    row = conn.execute(
        """
        SELECT id, timestamp, selected_symbol, selected_score, selected_net_expected_value,
               winner_reason, rejected_reason_json, leaderboard_json
        FROM ai_rank_snapshots
        WHERE strategy_id = ?
        ORDER BY id DESC
        LIMIT 1
        """,
        (strategy_id,),
    ).fetchone()
    if not row:
        return {}
    rejected = {}
    leaderboard = []
    try:
        rejected = json.loads(row[6] or "{}")
    except Exception:
        rejected = {}
    try:
        leaderboard = json.loads(row[7] or "[]")
    except Exception:
        leaderboard = []
    return {
        "snapshot_id": row[0],
        "timestamp": row[1],
        "winner": row[2],
        "final_profit_score": row[3],
        "net_expected_value": row[4],
        "winner_reason": row[5],
        "rejected_reasons": rejected,
        "leaderboard": leaderboard,
    }


def _safe_fetchall(conn: sqlite3.Connection, sql: str, params: tuple[Any, ...] = ()) -> list[tuple[Any, ...]]:
    try:
        return conn.execute(sql, params).fetchall()
    except sqlite3.Error:
        return []


def _parse_ts(ts: str | None) -> datetime | None:
    if not ts:
        return None
    try:
        return datetime.fromisoformat(str(ts).replace("Z", "+00:00"))
    except ValueError:
        return None


def _context_freshness(conn: sqlite3.Connection) -> dict[str, Any]:
    try:
        r = redis.Redis(host="127.0.0.1", port=6379, db=0, decode_responses=True)
    except Exception:
        r = None
    now = datetime.now(timezone.utc)
    rows: list[dict[str, Any]] = []
    for symbol in get_trading_symbols():
        key = REDIS_KEY_AI_CONTEXT.format(symbol=symbol)
        redis_exists = False
        redis_ttl: int | None = None
        redis_ts: str | None = None
        if r is not None:
            try:
                redis_exists = bool(r.exists(key))
                if redis_exists:
                    redis_ttl = int(r.ttl(key))
                    if r.type(key) == "hash":
                        h = r.hgetall(key)
                        redis_ts = h.get("ts_utc") or h.get("timestamp") or h.get("updated_at")
            except Exception:
                redis_exists = False
        db_ts = None
        row = conn.execute("SELECT MAX(ts_utc) FROM ai_context_snapshots WHERE symbol = ?", (symbol,)).fetchone()
        if row:
            db_ts = row[0]
        dt = _parse_ts(db_ts)
        rdt = _parse_ts(redis_ts)
        redis_age = None if rdt is None else max(0.0, (now - rdt).total_seconds())
        rows.append(
            {
                "symbol": symbol,
                "redis_key": key,
                "redis_exists": redis_exists,
                "redis_ttl": redis_ttl,
                "redis_ts_utc": redis_ts,
                "redis_age_sec": redis_age,
                "db_ts_utc": db_ts,
                "db_age_sec": None if dt is None else max(0.0, (now - dt).total_seconds()),
            }
        )
    ages = [r["redis_age_sec"] for r in rows if r.get("redis_age_sec") is not None]
    ages_sorted = sorted(ages)

    def _pct(p: float) -> float | None:
        if not ages_sorted:
            return None
        idx = round((len(ages_sorted) - 1) * p)
        return float(ages_sorted[max(0, min(idx, len(ages_sorted) - 1))])

    summary = {
        "symbols_with_redis_context": sum(1 for r in rows if r.get("redis_exists")),
        "redis_age_sec_min": min(ages) if ages else None,
        "redis_age_sec_max": max(ages) if ages else None,
        "redis_age_sec_spread": (max(ages) - min(ages)) if len(ages) > 1 else (0.0 if ages else None),
        "redis_age_sec_p10": _pct(0.10),
        "redis_age_sec_p50": _pct(0.50),
        "redis_age_sec_p90": _pct(0.90),
        "redis_age_sec_by_symbol": {str(r["symbol"]): r.get("redis_age_sec") for r in rows if r.get("redis_age_sec") is not None},
    }
    return {"rows": rows, "summary": summary}


def _rolling_profit_health(conn: sqlite3.Connection, *, window: int = 200) -> dict[str, Any]:
    w = max(50, min(1000, int(window or 200)))
    rows = _safe_fetchall(
        conn,
        """
        WITH recent AS (
            SELECT
                LOWER(COALESCE(strategy_id, 'day')) AS strategy_id,
                UPPER(REPLACE(COALESCE(symbol,''), '/', '')) AS symbol,
                COALESCE(net_pnl_pct, realized_pct, 0.0) AS net_pnl_pct
            FROM ai_outcome_training_rows
            ORDER BY id DESC
            LIMIT ?
        )
        SELECT
            strategy_id,
            symbol,
            COUNT(*) AS trades,
            SUM(net_pnl_pct) AS sum_net_pct,
            AVG(net_pnl_pct) AS avg_net_pct,
            AVG(CASE WHEN net_pnl_pct > 0 THEN 1.0 ELSE 0.0 END) AS win_rate
        FROM recent
        GROUP BY strategy_id, symbol
        """,
        (w,),
    )
    strategy_rows: dict[str, dict[str, Any]] = {}
    by_pair: list[dict[str, Any]] = []
    for sid, sym, trades, sum_net, avg_net, win_rate in rows:
        pair = {
            "strategy_id": str(sid or "day"),
            "symbol": str(sym or ""),
            "trades": int(trades or 0),
            "sum_net_pct": float(sum_net or 0.0),
            "avg_net_pct": float(avg_net or 0.0),
            "win_rate": float(win_rate or 0.0),
        }
        by_pair.append(pair)
        slot = strategy_rows.setdefault(
            pair["strategy_id"],
            {"trades": 0, "sum_net_pct": 0.0, "wins_weighted": 0.0},
        )
        slot["trades"] += pair["trades"]
        slot["sum_net_pct"] += pair["sum_net_pct"]
        slot["wins_weighted"] += pair["win_rate"] * pair["trades"]
    by_strategy = []
    for sid, agg in strategy_rows.items():
        t = max(1, int(agg["trades"]))
        by_strategy.append(
            {
                "strategy_id": sid,
                "trades": int(agg["trades"]),
                "sum_net_pct": float(agg["sum_net_pct"]),
                "avg_net_pct": float(agg["sum_net_pct"] / t),
                "win_rate": float(agg["wins_weighted"] / t),
            }
        )
    by_strategy.sort(key=lambda r: (r["sum_net_pct"], r["strategy_id"]))
    by_pair.sort(key=lambda r: (r["sum_net_pct"], r["strategy_id"], r["symbol"]))
    return {
        "window_trades": int(w),
        "by_strategy": by_strategy,
        "worst_pairs": [r for r in by_pair if int(r["trades"]) >= 5][:12],
        "best_pairs": [r for r in reversed(by_pair) if int(r["trades"]) >= 5][:12],
    }


def get_profit_system_diagnostics(db_path: str = DATABASE_PATH) -> dict[str, Any]:
    with sqlite3.connect(db_path) as conn:
        day = _load_latest_snapshot(conn, "day")
        safety_rows = _safe_fetchall(
            conn,
            """
            SELECT reject_reason, COUNT(*)
            FROM strategy_runtime_audit
            WHERE reject_reason IS NOT NULL AND reject_reason != ''
            GROUP BY reject_reason
            ORDER BY COUNT(*) DESC
            LIMIT 20
            """,
        )
        no_trade_rows = _safe_fetchall(
            conn,
            """
            SELECT execution_reason, COUNT(*)
            FROM ai_pipeline_decisions
            WHERE stage='EXECUTION' AND execution_result='NOT_EXECUTED' AND execution_reason IS NOT NULL
            GROUP BY execution_reason
            ORDER BY COUNT(*) DESC
            LIMIT 20
            """,
        )
        mem = _safe_fetchall(
            conn,
            """
            SELECT good_bad_memory_class, COUNT(*)
            FROM ai_outcome_training_rows
            GROUP BY good_bad_memory_class
            """,
        )
        peer = _safe_fetchall(
            conn,
            """
            SELECT learning_label, COUNT(*)
            FROM ai_peer_shadow_outcomes
            GROUP BY learning_label
            """,
        )
        sleeves = _safe_fetchall(
            conn,
            """
            SELECT strategy_id, allocated_capital, deployed_capital, available_capital,
                   realized_pnl, unrealized_pnl, current_drawdown, allocation_pct, updated_at
            FROM ai_strategy_capital_sleeves
            ORDER BY strategy_id
            """,
        )
        model_rows = _safe_fetchall(
            conn,
            """
            SELECT strategy_id, symbol, model_id, status, artifact_hash, promoted_at, validation_metrics_json
            FROM ai_model_versions
            WHERE status IN ('active', 'candidate', 'rollback')
            ORDER BY id DESC
            LIMIT 40
            """,
        )
        promo_rows = _safe_fetchall(
            conn,
            """
            SELECT strategy_id, symbol, event_type, reason, created_at
            FROM ai_model_promotion_events
            ORDER BY id DESC
            LIMIT 20
            """,
        )
        exit_cov = _safe_fetchall(
            conn,
            """
            SELECT
              COUNT(*),
              SUM(CASE WHEN COALESCE(ai_exit_recommended_action, '') != '' THEN 1 ELSE 0 END),
              AVG(CASE WHEN strategy_id='day' THEN hold_seconds END),
              AVG(CASE WHEN strategy_id='day' THEN hold_seconds END),
              SUM(CASE WHEN strategy_id='day' AND COALESCE(time_in_trade_sec, hold_seconds) > ? THEN 1 ELSE 0 END),
              SUM(CASE WHEN COALESCE(ai_exit_recommended_action,'')='CUT' AND COALESCE(net_pnl_pct, realized_pct) < 0 AND COALESCE(time_in_trade_sec, hold_seconds) > ? THEN 1 ELSE 0 END),
              SUM(CASE WHEN COALESCE(max_favorable_excursion, 0) > 0 AND COALESCE(mfe_giveback_pct, 0) >= ? THEN 1 ELSE 0 END)
            FROM ai_outcome_training_rows
            """,
            (1800.0, 1800.0, 0.40),
        )
        buy_size_cov = _safe_fetchall(
            conn,
            """
            SELECT
              COUNT(*),
              SUM(CASE WHEN COALESCE(buy_sizing_multiplier, 0) > 0 THEN 1 ELSE 0 END)
            FROM ai_outcome_training_rows
            """,
        )
        peer_count = _safe_fetchall(conn, "SELECT COUNT(*) FROM ai_peer_shadow_outcomes")
        adaptive_count = _safe_fetchall(conn, "SELECT COUNT(*) FROM ai_strategy_score_weights")
        promo_count = _safe_fetchall(conn, "SELECT COUNT(*) FROM ai_model_promotion_events")
        context_block = _context_freshness(conn)
        rolling_profit = _rolling_profit_health(conn, window=200)

    exit_row = exit_cov[0] if exit_cov else (0, 0, None, None, 0, 0, 0)
    buy_row = buy_size_cov[0] if buy_size_cov else (0, 0)
    return {
        "day_leaderboard": day,
        "safety_failures": [{"reason": r[0], "count": int(r[1])} for r in safety_rows],
        "no_trade_reasons": [{"reason": r[0], "count": int(r[1])} for r in no_trade_rows],
        "memory_counts": {str(k or "UNKNOWN"): int(v) for k, v in mem},
        "peer_shadow_summary": {str(k or "UNKNOWN"): int(v) for k, v in peer},
        "exit_telemetry_coverage": {
            "rows_total": int(exit_row[0] or 0),
            "rows_with_ai_exit_recommended_action": int(exit_row[1] or 0),
            "avg_hold_time_sec_window": float(exit_row[2] or 0.0),
            "avg_day_hold_time_sec": float(exit_row[3] or 0.0),
            "day_overstay_count": int(exit_row[4] or 0),
            "late_cut_loser_count": int(exit_row[5] or 0),
            "mfe_giveback_count": int(exit_row[6] or 0),
        },
        "buy_sizing_telemetry_coverage": {
            "rows_total": int(buy_row[0] or 0),
            "rows_with_sizing_multiplier": int(buy_row[1] or 0),
        },
        "peer_shadow_row_count": int((peer_count[0][0] if peer_count else 0) or 0),
        "adaptive_weight_row_count": int((adaptive_count[0][0] if adaptive_count else 0) or 0),
        "model_promotion_event_count": int((promo_count[0][0] if promo_count else 0) or 0),
        "sleeves": [
            {
                "strategy_id": r[0],
                "allocated_capital": float(r[1] or 0.0),
                "deployed_capital": float(r[2] or 0.0),
                "available_capital": float(r[3] or 0.0),
                "realized_pnl": float(r[4] or 0.0),
                "unrealized_pnl": float(r[5] or 0.0),
                "current_drawdown": float(r[6] or 0.0),
                "allocation_pct": float(r[7] or 0.0),
                "updated_at": r[8],
            }
            for r in sleeves
        ],
        "model_versions": [
            {
                "strategy_id": r[0],
                "symbol": r[1],
                "model_id": r[2],
                "status": r[3],
                "artifact_hash": r[4],
                "promoted_at": r[5],
                "validation_metrics_json": r[6],
            }
            for r in model_rows
        ],
        "model_promotion_events": [{"strategy_id": r[0], "symbol": r[1], "event_type": r[2], "reason": r[3], "created_at": r[4]} for r in promo_rows],
        "context_freshness": context_block["rows"],
        "context_freshness_summary": context_block["summary"],
        "rolling_profit_health": rolling_profit,
    }
