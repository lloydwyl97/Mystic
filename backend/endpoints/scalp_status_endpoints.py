"""Read-only scalp status API — isolated from Mystic DAY."""

from __future__ import annotations

import json
import logging
import sqlite3
import time
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


def _scalp_runner_active() -> bool:
    """True iff the scalp paper runner process is actually running."""
    import subprocess

    try:
        res = subprocess.run(
            ["pgrep", "-f", "backend.services.binance_scalp.runner"],
            capture_output=True,
            timeout=5,
            check=False,
        )
        return res.returncode == 0
    except Exception:
        return False


@router.get("/status")
def scalp_status(*, warm: int = 0) -> dict:
    """Read-only scalp engine status — isolated from DAY top-four.

    Fast path only: reads the Redis snapshot published by the paper runner.
    Never rebuilds via REST depth / klines / strategy router / long SQLite.
    ``warm`` is accepted for API compatibility but does not trigger a rebuild.
    """
    _ = warm  # ignored — GET must not cold-build
    active = _scalp_runner_active()
    if not active:
        return {
            "runner_active": False,
            "engine": "scalp",
            "scalp_engaged": False,
            "snapshot_available": False,
            "stale": True,
            "reason": "RUNNER_INACTIVE",
            "operational_summary": {"operational_mode": "runner_dead"},
            "pnl_summary": {"engine": "scalp"},
            "note": "Scalp paper runner is not running. Start with './start_mystic.sh core' or 'scalp'.",
        }

    try:
        from backend.services.binance_scalp.scalp_status_cache import get_cached_scalp_status

        # Read-only Redis snapshot — never rebuilds market state on GET.
        snapshot = get_cached_scalp_status(warm_rounds=0)
        pnl = snapshot.pop("pnl_summary", None) or {"engine": "scalp"}
        return {
            "runner_active": True,
            "engine": "scalp",
            "pnl_summary": pnl,
            **snapshot,
        }
    except Exception as exc:
        logger.exception("scalp_status fast-path failed: %s", exc)
        return {
            "runner_active": True,
            "engine": "scalp",
            "snapshot_available": False,
            "stale": True,
            "reason": "SCALP_STATUS_SNAPSHOT_MISSING",
            "pnl_summary": {"engine": "scalp"},
            "overall_decision": "DEGRADED",
            "top_blocker": "STATUS_READ_FAILED",
            "status_error": str(exc)[:240],
            "note": "Scalp status snapshot unavailable — retry shortly. Other /api/scalp/* endpoints may still work.",
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
        out: dict = {"engine": "scalp", "available": True}
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
    """Open scalp paper positions (read-only) with live lifecycle fields."""
    try:
        with _ro_conn() as conn:
            open_rows = _rows(
                conn,
                """
                SELECT symbol, quantity, entry_price, entry_time, entry_time_epoch,
                       trade_id, status, state, diagnostics_json, last_state_reason,
                       max_favorable_pct, stale_review_count, session_low_bid
                FROM scalp_paper_positions
                WHERE status = 'OPEN'
                ORDER BY entry_time_epoch DESC
                """,
            )
            ledger = _rows(
                conn,
                """
                SELECT principal, cash_balance, positions_value, realized_pnl,
                       unrealized_pnl, total_equity, updated_at
                FROM scalp_paper_ledger WHERE id = 1
                """,
            )
        now = time.time()
        for row in open_rows:
            epoch = float(row.get("entry_time_epoch") or 0)
            row["hold_seconds"] = round(max(0.0, now - epoch), 1) if epoch else None
            diag_raw = row.get("diagnostics_json")
            if diag_raw:
                try:
                    diag = json.loads(diag_raw)
                    row["setup"] = diag.get("setup_name") or diag.get("setup")
                except (json.JSONDecodeError, TypeError):
                    row["setup"] = None
            else:
                row["setup"] = None
        from backend.services.binance_scalp.scalp_position_lifecycle import enrich_open_scalp_positions

        enriched = enrich_open_scalp_positions(open_rows)
        for row in enriched:
            row.pop("diagnostics_json", None)
        return {
            "engine": "scalp",
            "open_count": len(enriched),
            "positions": enriched,
            "ledger": ledger[0] if ledger else None,
        }
    except FileNotFoundError as exc:
        return {"engine": "scalp", "open_count": 0, "positions": [], "ledger": None, "note": str(exc)}


@router.get("/trades")
def scalp_trades(
    limit: int = Query(50, ge=1, le=500),
    days: int | None = Query(None, ge=1, le=365),
) -> dict[str, Any]:
    """Recent scalp paper trades (read-only)."""
    try:
        with _ro_conn() as conn:
            if days is not None:
                rows = _rows(
                    conn,
                    """
                    SELECT trade_id, symbol, side, quantity, price, notional,
                           fee_usd, pnl_usd, pnl_pct, exit_reason, created_at
                    FROM scalp_paper_trades
                    WHERE datetime(created_at) >= datetime('now', ?)
                    ORDER BY created_at DESC
                    LIMIT ?
                    """,
                    (f"-{int(days)} days", limit),
                )
            else:
                rows = _rows(
                    conn,
                    """
                    SELECT trade_id, symbol, side, quantity, price, notional,
                           fee_usd, pnl_usd, pnl_pct, exit_reason, created_at
                    FROM scalp_paper_trades
                    ORDER BY created_at DESC
                    LIMIT ?
                    """,
                    (limit,),
                )
        return {"engine": "scalp", "count": len(rows), "trades": rows}
    except FileNotFoundError as exc:
        return {"engine": "scalp", "count": 0, "trades": [], "note": str(exc)}


@router.get("/scoreboard")
def scalp_scoreboard(days: int = Query(7, ge=1, le=90)) -> dict[str, Any]:
    """Daily scalp scoreboard rollup (read-only)."""
    try:
        with _ro_conn() as conn:
            rows = _rows(
                conn,
                """
                SELECT day, trades, wins, losses, net_pnl, updated_at
                FROM scalp_scoreboard_daily
                ORDER BY day DESC
                LIMIT ?
                """,
                (days,),
            )
        return {"engine": "scalp", "days": days, "rows": rows}
    except FileNotFoundError as exc:
        return {"engine": "scalp", "days": days, "rows": [], "note": str(exc)}


@router.get("/attribution")
def scalp_attribution(days: int | None = Query(None, ge=1, le=365)) -> dict[str, Any]:
    """Closed scalp PnL attribution by symbol, setup, regime, exit, hold, and cost burden."""
    from backend.services.binance_scalp.scalp_attribution_report import build_scalp_attribution_report

    return build_scalp_attribution_report(days=days)


@router.get("/learning-summary")
def scalp_learning_summary(limit: int = Query(20, ge=1, le=200)) -> dict[str, Any]:
    """Scalp learning tables — attribution, post-trade reviews, strategy weights."""
    try:
        with _ro_conn() as conn:
            reviews = _rows(
                conn,
                f"""
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
                f"""
                SELECT trade_id, symbol, micro_regime, scalp_setup, outcome_reason,
                       net_pnl_after_fees, exit_reason, created_at
                FROM scalp_outcome_attribution
                ORDER BY created_at DESC
                LIMIT ?
                """,
                (limit,),
            )
            sell_count = conn.execute("SELECT COUNT(*) FROM scalp_paper_trades WHERE side='SELL'").fetchone()[0]
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
