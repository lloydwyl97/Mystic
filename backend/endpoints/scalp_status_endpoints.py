"""Read-only scalp status API — isolated from Mystic DAY."""

from __future__ import annotations

import logging
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Query

from backend.services.binance_scalp.config import get_scalp_config
from backend.services.binance_scalp.strategies import STRATEGY_NAMES, enabled_strategies

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/scalp", tags=["scalp"])


def _scalp_db_path() -> Path:
    return Path(get_scalp_config().database_path).resolve()


def _ro_conn() -> sqlite3.Connection:
    path = _scalp_db_path()
    if not path.exists():
        raise FileNotFoundError(f"scalp database not found: {path}")
    return sqlite3.connect(f"file:{path}?mode=ro", uri=True)


def _rows(conn: sqlite3.Connection, sql: str, params: tuple = ()) -> list[dict[str, Any]]:
    conn.row_factory = sqlite3.Row
    cur = conn.execute(sql, params)
    return [dict(r) for r in cur.fetchall()]


def _live_db_path() -> str:
    """Live SCALP_V2 books into the portfolio engine database."""
    from backend.database_schema import DATABASE_PATH

    return DATABASE_PATH


def _live_redis() -> Any:
    try:
        from backend.config.redis_config import get_shared_redis_sync

        return get_shared_redis_sync()
    except Exception:
        return None


@router.get("/status")
def scalp_status(*, warm: int = 0) -> dict:
    """Live SCALP_V2 status — loop heartbeat, latest decisions, open lots, realized P&L.

    ``warm`` is accepted for API compatibility but does not trigger a rebuild.
    """
    _ = warm  # ignored — GET must not cold-build
    try:
        from backend.services.scalp_v2.live_dashboard import live_status

        return live_status(_live_db_path(), _live_redis())
    except Exception as exc:
        logger.exception("scalp_status live read failed: %s", exc)
        return {
            "runner_active": False,
            "engine": "scalp",
            "reason": "SCALP_STATUS_READ_FAILED",
            "pnl_summary": {"engine": "scalp"},
            "overall_decision": "DEGRADED",
            "top_blocker": "STATUS_READ_FAILED",
            "status_error": str(exc)[:240],
            "note": "SCALP status read failed — retry shortly.",
        }


@router.get("/strategies")
def scalp_strategies() -> dict:
    """Enabled/disabled strategy inventory (no market fetch)."""
    config = get_scalp_config()
    return {
        "all": list(STRATEGY_NAMES),
        "enabled": [s.name for s in enabled_strategies(config)],
        "disabled": sorted(config.disabled_strategies),
        "disabled_env": "SCALP_DISABLED_STRATEGIES",
    }


@router.get("/gates/today")
def scalp_gates_today(date: str | None = None) -> dict[str, Any]:
    """SCALP gate counters for today — top blockers by hard_blocked."""
    try:
        from backend.services.scalp_gate_registry import registry_snapshot
        from backend.services.scalp_gate_telemetry import counters_today, ensure_scalp_gate_schema

        cfg = get_scalp_config()
        ensure_scalp_gate_schema(cfg.database_path)
        rows = counters_today(cfg.database_path, date=date)
        snap = registry_snapshot()
        return {
            "success": True,
            "engine": "scalp",
            "date": date or datetime.now(timezone.utc).strftime("%Y-%m-%d"),
            "data": {"gates": rows, "top_blockers": rows[:15]},
            "registry": {
                "decision_policy_version": snap.get("decision_policy_version"),
                "threshold_freeze_active": snap.get("threshold_freeze_active"),
            },
        }
    except Exception as exc:
        return {"success": False, "engine": "scalp", "error": str(exc)[:240]}


@router.get("/gates/registry")
def scalp_gates_registry() -> dict[str, Any]:
    """Versioned SCALP gate registry snapshot."""
    try:
        from backend.services.scalp_gate_registry import registry_snapshot

        return {"success": True, "engine": "scalp", "data": registry_snapshot()}
    except Exception as exc:
        return {"success": False, "engine": "scalp", "error": str(exc)[:240]}


@router.get("/attribution/today")
def scalp_attribution_today(date: str | None = None) -> dict[str, Any]:
    """Executed SCALP PnL + gate opportunity (shadow rejects) for measurement window."""
    try:
        from backend.services.scalp_gate_telemetry import (
            attribution_report,
            ensure_scalp_gate_schema,
            shadow_rejects_summary,
        )

        cfg = get_scalp_config()
        ensure_scalp_gate_schema(cfg.database_path)
        report = attribution_report(cfg.database_path, date=date)
        shadows = shadow_rejects_summary(cfg.database_path, limit=30)
        return {"success": True, "engine": "scalp", "data": {**report, "shadow_summary": shadows}}
    except Exception as exc:
        return {"success": False, "engine": "scalp", "error": str(exc)[:240]}


@router.get("/telemetry")
def scalp_entry_telemetry() -> dict:
    """Latest genuine-pass / reject / post-pass-blocker telemetry + rolling window."""
    try:
        import redis as redis_lib

        from backend.services.binance_scalp.scalp_entry_telemetry import (
            read_entry_telemetry,
            read_rolling_telemetry,
        )

        cfg = get_scalp_config()
        r = redis_lib.Redis(host="127.0.0.1", port=6379, db=0, decode_responses=True)
        payload = read_entry_telemetry(r, prefix=cfg.redis_key_prefix)
        rolling_full = read_rolling_telemetry(r, prefix=cfg.redis_key_prefix)
        if not payload and not rolling_full:
            return {
                "engine": "scalp",
                "available": False,
                "note": "No telemetry yet — wait one paper-runner cycle (~5s) after start.",
            }
        out: dict = {"engine": "scalp", "available": True, "ranking_only": True, "note": "ranking-era telemetry isolated from structural LP"}
        if payload:
            out.update(payload)
        if rolling_full and "rolling" not in out:
            out["rolling"] = {
                "cycles": rolling_full.get("cycles"),
                "pass_rate_overall": rolling_full.get("pass_rate_overall"),
                "pct_cycles_with_pass": rolling_full.get("pct_cycles_with_pass"),
                "pct_cycles_with_eligible": rolling_full.get("pct_cycles_with_eligible"),
                "top_reject_reasons": rolling_full.get("top_reject_reasons"),
                "top_post_pass_blockers": rolling_full.get("top_post_pass_blockers"),
                "strategy_pass_rate": rolling_full.get("strategy_pass_rate"),
                "regime_native_pass_count": rolling_full.get("regime_native_pass_count"),
                "regime_mismatch_pass_count": rolling_full.get("regime_mismatch_pass_count"),
                "genuine_pass_setups": rolling_full.get("genuine_pass_setups"),
                "entry_eligible_count": rolling_full.get("entry_eligible_count"),
                "updated_at_epoch": rolling_full.get("updated_at_epoch"),
            }
        if rolling_full:
            out["rolling_full"] = {
                "cycles": rolling_full.get("cycles"),
                "started_at_epoch": rolling_full.get("started_at_epoch"),
                "strategy_pass_rate": rolling_full.get("strategy_pass_rate"),
                "strategy_eval_counts": rolling_full.get("strategy_eval_counts"),
                "strategy_pass_counts": rolling_full.get("strategy_pass_counts"),
                "recent_cycle_digest": (rolling_full.get("recent_cycle_digest") or [])[-20:],
            }
            # Dashboard cards read top-level pass/eligible — promote rolling when cycle TTL expired.
            for _k in ("genuine_pass_setups", "entry_eligible_count", "regime_native_pass_count", "regime_mismatch_pass_count"):
                if out.get(_k) is None and rolling_full.get(_k) is not None:
                    out[_k] = rolling_full.get(_k)
            if not out.get("reject_reasons") and rolling_full.get("top_reject_reasons"):
                out["reject_reasons"] = rolling_full.get("top_reject_reasons")
        # Eligible map for dashboard symbol table (status router uses the same shape).
        if "per_symbol_entry_eligible" not in out:
            elig_map: dict[str, bool] = {}
            for row in out.get("symbols") or []:
                if isinstance(row, dict) and row.get("symbol") is not None:
                    elig_map[str(row.get("symbol"))] = bool(row.get("entry_eligible"))
            if elig_map:
                out["per_symbol_entry_eligible"] = elig_map
        return out
    except Exception as exc:
        return {"engine": "scalp", "available": False, "error": str(exc)[:240]}


@router.get("/positions")
def scalp_positions() -> dict[str, Any]:
    """Open live SCALP_V2 lots (dust excluded), read-only."""
    from backend.services.scalp_v2.live_dashboard import live_positions

    try:
        return live_positions(_live_db_path())
    except sqlite3.Error as exc:
        return {"engine": "scalp", "open_count": 0, "positions": [], "ledger": None, "note": str(exc)[:240]}


@router.get("/trades")
def scalp_trades(
    limit: int = Query(50, ge=1, le=500),
    days: int | None = Query(None, ge=1, le=365),
) -> dict[str, Any]:
    """Recent live SCALP_V2 fills (read-only)."""
    from backend.services.scalp_v2.live_dashboard import live_trades

    try:
        return live_trades(_live_db_path(), limit=limit, days=days)
    except sqlite3.Error as exc:
        return {"engine": "scalp", "count": 0, "trades": [], "note": str(exc)[:240]}


@router.get("/scoreboard")
def scalp_scoreboard(days: int = Query(7, ge=1, le=90)) -> dict[str, Any]:
    """Daily live SCALP_V2 realized rollup (UTC days, read-only)."""
    from backend.services.scalp_v2.live_dashboard import live_scoreboard

    try:
        return live_scoreboard(_live_db_path(), days=days)
    except sqlite3.Error as exc:
        return {"engine": "scalp", "days": days, "rows": [], "note": str(exc)[:240]}


@router.get("/attribution")
def scalp_attribution(days: int | None = Query(None, ge=1, le=365)) -> dict[str, Any]:
    """Closed live SCALP_V2 PnL attribution by symbol, setup, regime, exit, hold, and cost burden."""
    from backend.services.scalp_v2.live_dashboard import live_attribution

    try:
        return live_attribution(_live_db_path(), days=days)
    except sqlite3.Error as exc:
        return {"engine": "scalp", "error": str(exc)[:200], "rows": []}


@router.get("/learning-summary")
def scalp_learning_summary(limit: int = Query(20, ge=1, le=200)) -> dict[str, Any]:
    """Scalp learning tables — attribution, post-trade reviews, strategy weights."""
    try:
        with _ro_conn() as conn:
            reviews = _rows(
                conn,
                """
                SELECT trade_id, symbol, closed_at_utc, review_json, ingested_at_utc
                FROM scalp_post_trade_feature_reviews
                ORDER BY closed_at_utc DESC
                LIMIT ?
                """,
                (limit,),
            )
            weights = _rows(
                conn,
                """
                SELECT symbol, regime, component_name, weight, sample_count,
                       good_count, bad_count, net_expectancy, updated_at
                FROM scalp_strategy_score_weights
                ORDER BY updated_at DESC, weight DESC
                LIMIT 100
                """,
            )
            attribution = _rows(
                conn,
                """
                SELECT trade_id, symbol, micro_regime, scalp_setup, outcome_reason,
                       net_pnl_after_fees, exit_reason, created_at
                FROM scalp_outcome_attribution
                ORDER BY created_at DESC
                LIMIT ?
                """,
                (limit,),
            )
        from backend.services.scalp_v2.live_dashboard import live_closed_sell_count

        sell_count = live_closed_sell_count(_live_db_path())
        return {
            "engine": "scalp",
            "closed_sells": int(sell_count),
            "first_close_ready": int(sell_count) > 0,
            "outcome_attribution": attribution,
            "post_trade_reviews": reviews,
            "strategy_score_weights": weights,
        }
    except FileNotFoundError as exc:
        return {
            "engine": "scalp",
            "closed_sells": 0,
            "first_close_ready": False,
            "outcome_attribution": [],
            "post_trade_reviews": [],
            "strategy_score_weights": [],
            "note": str(exc),
        }
    except sqlite3.OperationalError as exc:
        return {
            "engine": "scalp",
            "closed_sells": 0,
            "first_close_ready": False,
            "outcome_attribution": [],
            "post_trade_reviews": [],
            "strategy_score_weights": [],
            "note": f"table missing: {exc}",
        }
