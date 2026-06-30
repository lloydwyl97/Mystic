"""Read-only scalp status API — isolated from Mystic DAY."""

from __future__ import annotations

import json
import sqlite3
import time
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Query

from backend.services.binance_scalp.config import get_scalp_config
from backend.services.binance_scalp.pnl_summary import build_scalp_pnl_summary
from backend.services.binance_scalp.strategies import STRATEGY_NAMES, enabled_strategies

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

    Default warm=0 returns a cached/fast snapshot (runner scan overlays momentum).
    Pass warm=12 for a full cold momentum warm (~60s, diagnostics only).
    """
    active = _scalp_runner_active()
    pnl = build_scalp_pnl_summary()
    if not active:
        return {
            "runner_active": False,
            "engine": "scalp",
            "scalp_engaged": False,
            "pnl_summary": pnl,
            "note": "Scalp paper runner is not running. Start with './start_mystic.sh core' or 'scalp'.",
        }
    from backend.services.binance_scalp.scalp_status_cache import get_cached_scalp_status

    warm_rounds = max(0, min(int(warm), 12))
    return {
        "runner_active": True,
        "engine": "scalp",
        "pnl_summary": pnl,
        **get_cached_scalp_status(warm_rounds=warm_rounds),
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


@router.get("/positions")
def scalp_positions() -> dict[str, Any]:
    """Open scalp paper positions (read-only)."""
    try:
        with _ro_conn() as conn:
            open_rows = _rows(
                conn,
                """
                SELECT symbol, quantity, entry_price, entry_time, entry_time_epoch,
                       trade_id, status, state, diagnostics_json, last_state_reason
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
            diag_raw = row.pop("diagnostics_json", None)
            if diag_raw:
                try:
                    row["setup"] = json.loads(diag_raw).get("setup_name") or json.loads(diag_raw).get("setup")
                except (json.JSONDecodeError, TypeError):
                    row["setup"] = None
            else:
                row["setup"] = None
        return {
            "engine": "scalp",
            "open_count": len(open_rows),
            "positions": open_rows,
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
            sell_count = conn.execute(
                "SELECT COUNT(*) FROM scalp_paper_trades WHERE side='SELL'"
            ).fetchone()[0]
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
