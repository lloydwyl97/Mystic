"""
Performance Analytics API

Live-only endpoints returning 200 with data, 204 when no data yet, 503 on failure.
"""

from __future__ import annotations

import asyncio
import inspect
import logging
from datetime import datetime, timezone
from typing import Any

from fastapi import APIRouter, HTTPException
from fastapi.responses import Response

from backend.services.performance_analytics_service import (
    PerformanceAnalyticsService,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/performance", tags=["performance"])

# ---------- Service Resolver ----------

_IMPORT_ERROR = None

# Performance service state - using dict to avoid global keyword
_svc_singleton_state: dict[str, PerformanceAnalyticsService | None] = {"instance": None}


def _get_svc() -> PerformanceAnalyticsService:
    """
    Lazy singleton resolver for PerformanceAnalyticsService.
    Raises 503 if the service is unavailable.
    """
    if _IMPORT_ERROR is not None or PerformanceAnalyticsService is None:
        raise HTTPException(
            status_code=503,
            detail=f"PerformanceAnalyticsService unavailable: {_IMPORT_ERROR}",
        )
    if _svc_singleton_state["instance"] is None:
        _svc_singleton_state["instance"] = PerformanceAnalyticsService()
    return _svc_singleton_state["instance"]


async def _maybe_await(func_or_val, *args, **kwargs):
    """Call a possibly-async or sync function/value and return its result."""
    # If it's already an awaitable (a coroutine object), await it.
    if inspect.isawaitable(func_or_val):
        return await func_or_val
    # If it's a coroutine function (or any callable), call it and await the result if needed.
    if callable(func_or_val):
        res = func_or_val(*args, **kwargs)
        if inspect.isawaitable(res):
            return await res
        return res
    # Otherwise just return the value.
    return func_or_val


def _get_current_paper_run_id() -> str | None:
    """Reuse existing paper_run_id from paper trading service (no new state). Returns None if unavailable."""
    try:
        from backend.services.paper_trading_service import get_paper_trading_service

        svc = get_paper_trading_service()
        return getattr(svc, "paper_run_id", None)
    except Exception as e:
        logger.warning("_get_current_paper_run_id failed: %s", e)
        return None


def _as_float(v: Any, default: float = 0.0) -> float:
    try:
        return float(v)
    except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError):
        return default


def _get_field(item: Any, field: str) -> Any:
    """Get field from dict or attribute from object; return None if missing."""
    if isinstance(item, dict):
        return item.get(field)
    return getattr(item, field, None)


def _row_ts(row: Any) -> Any:
    """Portfolio rows may use ``date`` or ``timestamp`` depending on source."""
    return _get_field(row, "timestamp") or _get_field(row, "date")


def _load_forward_dashboard_sells(*, limit: int = 500) -> list[dict[str, Any]]:
    """Closed SELL rows for dashboard/charts: non-synthetic, non-admin, forward epoch when set."""
    import sqlite3

    from backend.database_schema import DATABASE_PATH
    from backend.services.allweather_paper_accounting import forward_sell_filter_sql, get_forward_epoch

    epoch = get_forward_epoch(DATABASE_PATH)
    epoch_start = str((epoch or {}).get("epoch_started_at") or "") or None
    where_sql, where_params = forward_sell_filter_sql(epoch_start=epoch_start)
    conn = sqlite3.connect(DATABASE_PATH)
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute(
            f"""
            SELECT pnl, hold_time_seconds, exit_type, timestamp, symbol, side
            FROM paper_trades
            WHERE {where_sql}
            ORDER BY timestamp DESC
            LIMIT ?
            """,
            (*where_params, limit),
        ).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


# ---------- Endpoints (200 / 204 / 503 policy) ----------


def _analytics_from_paper_trades(trades: list[dict[str, Any]]) -> dict[str, Any] | None:
    """Compute analytics dict from paper_trades list (same shape as trade_performance)."""
    with_pnl = [t for t in trades if t.get("pnl") is not None]
    if not with_pnl:
        return None
    total_trades = len(with_pnl)
    pnls = [float(t["pnl"]) for t in with_pnl]
    total_pnl = sum(pnls)
    winning = sum(1 for p in pnls if p > 0)
    win_rate = (winning / total_trades * 100.0) if total_trades else 0.0
    avg_pnl = total_pnl / total_trades if total_trades else 0.0
    return {
        "total_trades": total_trades,
        "win_rate": win_rate,
        "total_pnl": total_pnl,
        "avg_pnl": avg_pnl,
        "best_trade": max(pnls),
        "worst_trade": min(pnls),
    }


def _analytics_response(
    current_metrics: dict[str, Any],
    last_updated: str,
    data_source: str,
) -> dict[str, Any]:
    """Build analytics success response."""
    return {
        "status": "success",
        "current_metrics": current_metrics,
        "historical_metrics": [],
        "last_updated": last_updated,
        "data_source": data_source,
    }


@router.get("/analytics")
async def get_analytics() -> Any:
    """
    Return comprehensive performance analytics. Prefers paper_trades (all, paper+live)
    then trade_performance, then default_baseline. Full history for dashboard totals.
    """
    try:
        from backend.services.paper_trading_service import get_paper_trading_service
        from backend.services.trade_performance_tracker import get_trade_performance_summary

        trade_data = get_trade_performance_summary(since_timestamp=None)
        last_updated = trade_data.get("timestamp", "2024-01-01T00:00:00Z")

        # 1) paper_trades forward-epoch strategy closes (canonical dashboard scope)
        forward_rows = _load_forward_dashboard_sells(limit=5000)
        computed = _analytics_from_paper_trades(forward_rows)
        if computed:
            current_metrics = {
                "total_trades": computed["total_trades"],
                "win_rate": computed["win_rate"],
                "total_pnl": computed["total_pnl"],
                "avg_win": computed["avg_pnl"],
                "avg_loss": computed["avg_pnl"],
                "largest_win": computed["best_trade"],
                "largest_loss": computed["worst_trade"],
                "sharpe_ratio": 0.0,
                "max_drawdown": 0.0,
                "profit_factor": 0.0,
            }
            return _analytics_response(current_metrics, last_updated, "paper_trades_forward_epoch")

        paper_service = get_paper_trading_service()
        paper_trades = await _maybe_await(paper_service.get_trade_history, None, None, None, True)
        computed = _analytics_from_paper_trades(paper_trades or [])
        if computed:
            current_metrics = {
                "total_trades": computed["total_trades"],
                "win_rate": computed["win_rate"],
                "total_pnl": computed["total_pnl"],
                "avg_win": computed["avg_pnl"],
                "avg_loss": computed["avg_pnl"],
                "largest_win": computed["best_trade"],
                "largest_loss": computed["worst_trade"],
                "sharpe_ratio": 0.0,
                "max_drawdown": 0.0,
                "profit_factor": 0.0,
            }
            return _analytics_response(current_metrics, last_updated, "paper_trades")

        # 2) trade_performance (fallback - may have fewer rows than paper_trades for live)
        total = trade_data.get("total_trades", 0) or 0
        if total > 0:
            current_metrics = {
                "total_trades": total,
                "win_rate": trade_data.get("win_rate", 0.0),
                "total_pnl": trade_data.get("total_pnl", 0.0),
                "avg_win": trade_data.get("avg_pnl", 0.0),
                "avg_loss": trade_data.get("avg_pnl", 0.0),
                "largest_win": trade_data.get("best_trade", 0.0),
                "largest_loss": trade_data.get("worst_trade", 0.0),
                "sharpe_ratio": 0.0,
                "max_drawdown": 0.0,
                "profit_factor": 0.0,
            }
            return _analytics_response(current_metrics, last_updated, "trade_performance")

        # 3) default_baseline
        return _analytics_response(
            {
                "total_trades": 0,
                "win_rate": 0.0,
                "total_pnl": 0.0,
                "avg_win": 0.0,
                "avg_loss": 0.0,
                "largest_win": 0.0,
                "largest_loss": 0.0,
                "sharpe_ratio": 0.0,
                "max_drawdown": 0.0,
                "profit_factor": 0.0,
            },
            last_updated,
            "default_baseline",
        )

    except Exception as e:
        logger.exception("Failed to get real performance data, falling back to baseline: %s", e)

        # Return baseline data for UI compatibility
        return {
            "status": "success",
            "current_metrics": {
                "total_trades": 0,
                "win_rate": 0.0,
                "total_pnl": 0.0,
                "avg_win": 0.0,
                "avg_loss": 0.0,
                "largest_win": 0.0,
                "largest_loss": 0.0,
                "sharpe_ratio": 0.0,
                "max_drawdown": 0.0,
                "profit_factor": 0.0,
            },
            "historical_metrics": [],
            "last_updated": "2024-01-01T00:00:00Z",
            "data_source": "baseline_fallback",
            "error": str(e),
        }


def _pv_series_from_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Build [{"timestamp", "value"}] from service rows; skip invalid rows."""
    series = []
    for row in rows:
        ts = _row_ts(row)
        val = _get_field(row, "value")
        if ts is None or val is None:
            continue
        try:
            v = float(val)
        except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError):
            continue
        series.append({"timestamp": ts, "value": v})
    return series


def _append_live_equity_point(series: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Append current total_equity from ledger so chart updates on each poll."""
    if not series:
        return series
    try:
        import sqlite3
        from datetime import datetime, timezone

        from backend.database_schema import DATABASE_PATH

        conn = sqlite3.connect(DATABASE_PATH)
        try:
            cursor = conn.cursor()
            cursor.execute("SELECT total_equity FROM portfolio_engine_ledger WHERE id=1")
            row = cursor.fetchone()
        finally:
            conn.close()
        if row and row[0] is not None:
            now_ts = datetime.now(timezone.utc).isoformat(timespec="seconds")
            series = [*series, {"timestamp": now_ts, "value": float(row[0])}]
    except Exception:
        logger.warning("Failed to append live equity point from ledger", exc_info=True)
    return series


@router.get("/portfolio-value")
async def portfolio_value() -> Any:
    """
    Return portfolio value time series from portfolio_snapshots (5-min intervals, 30-day retention).
    Falls back to single current ledger point when no snapshots exist yet.

    CANONICAL SOURCE: portfolio_snapshots + portfolio_engine_ledger
    Formula: total_equity = cash_balance + positions_value
    """
    try:
        import sqlite3
        from datetime import datetime, timezone

        from backend.database_schema import DATABASE_PATH

        conn = sqlite3.connect(DATABASE_PATH)
        try:
            cur = conn.cursor()

            cur.execute("""
                SELECT timestamp, total_equity FROM portfolio_snapshots
                ORDER BY timestamp ASC
            """)
            rows = cur.fetchall()

            cur.execute("""
                SELECT total_equity, cash_balance, positions_value, last_updated, principal
                FROM portfolio_engine_ledger WHERE id=1
            """)
            ledger = cur.fetchone()
            cash = float(ledger[1] or 0) if ledger else 0.0
            positions = float(ledger[2] or 0) if ledger else 0.0
            current_val = float(ledger[0] or 0) if ledger else 0.0
            last_updated = (ledger[3] or datetime.now(timezone.utc).isoformat()) if ledger else datetime.now(timezone.utc).isoformat()
            principal = float(ledger[4] or 25000.0) if ledger else 25000.0
            max_equity = principal * 1.15
            min_equity = principal * 0.5

            if abs(current_val - (cash + positions)) > 0.01:
                logger.warning(
                    "ACCOUNTING_MISMATCH in portfolio-value: equity=%.2f != cash=%.2f + positions=%.2f",
                    current_val,
                    cash,
                    positions,
                )

            portfolio_series = [{"timestamp": r[0], "value": float(r[1])} for r in rows if min_equity <= float(r[1]) <= max_equity]

            if not portfolio_series or portfolio_series[-1]["timestamp"] != last_updated:
                portfolio_series.append({"timestamp": last_updated, "value": current_val})

            return {
                "portfolio": portfolio_series,
                "current_value": current_val,
                "cash_balance": cash,
                "positions_value": positions,
                "last_updated": last_updated,
                "data_source": "portfolio_snapshots",
                "canonical": True,
                "formula": "total_equity = cash_balance + positions_value",
            }
        finally:
            conn.close()
    except Exception as e:
        logger.exception("Failed to read portfolio snapshots: %s", e)

    return {
        "portfolio": [{"timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"), "value": 0.0}],
        "current_value": 0.0,
        "cash_balance": 0.0,
        "positions_value": 0.0,
        "data_source": "default_fallback",
        "error": "portfolio_engine_ledger not available",
    }


@router.get("/daily-returns")
async def daily_returns() -> Any:
    """
    Return daily returns data for portfolio performance.
    - Always returns meaningful data (baseline for empty portfolios)
    - Never returns 204 or 503 errors
    """
    try:
        svc = _get_svc()
        data = await _maybe_await(svc.get_performance_summary)
        pv = (data or {}).get("portfolio_value", []) if isinstance(data, dict) else []

        # If we have real portfolio value data, calculate returns
        if pv and len(pv) > 1:
            series: list[dict[str, Any]] = []
            prev: float | None = None
            for row in pv:
                v_raw = _get_field(row, "value")
                v = None if v_raw is None else _as_float(v_raw)
                if prev is not None and v is not None and prev > 0.0:
                    ts = _row_ts(row)
                    if ts is not None:
                        r = (v - prev) / prev * 100.0
                        series.append({"timestamp": ts, "value": round(r, 4)})
                prev = v if v is not None else prev

            if series:
                return {"returns": series}

    except Exception:
        logger.warning("Failed to get daily returns from performance summary, trying fallbacks", exc_info=True)

    try:
        svc = _get_svc()
        current_run_id = _get_current_paper_run_id()
        # Prefer fullest data first: paper_trades all (paper+live) → trade_performance → session
        candidates: list[tuple[str, str]] = [(None, "paper_trades"), (None, "trade_performance")]
        if current_run_id:
            candidates.append((current_run_id, "paper_trades"))
        for run_id_val, src in candidates:
            if src == "trade_performance":
                pv_list = await _maybe_await(svc._load_portfolio_value_from_canonical)
            else:
                pv_list = await _maybe_await(svc._load_portfolio_value_from_paper_trades, run_id_val)
            if pv_list and len(pv_list) > 1:
                series = []
                prev = None
                for row in pv_list:
                    v_raw = _get_field(row, "value")
                    v = None if v_raw is None else _as_float(v_raw)
                    if prev is not None and v is not None and prev > 0.0:
                        ts = _row_ts(row)
                        if ts is not None:
                            r = (v - prev) / prev * 100.0
                            series.append({"timestamp": ts, "value": round(r, 4)})
                    prev = v if v is not None else prev
                if series:
                    return {"returns": series, "data_source": src}
    except Exception:
        logger.warning("Failed to get daily returns from paper_trades/trade_performance fallbacks", exc_info=True)

    return {
        "returns": [],
        "summary": {"average_daily_return": 0.0, "total_return": 0.0, "positive_days": 0, "negative_days": 0, "best_day": 0.0, "worst_day": 0.0, "volatility": 0.0},
        "last_updated": datetime.now(timezone.utc).isoformat(),
        "data_source": "no_data",
    }


@router.get("/drawdown")
async def drawdown_series() -> Any:
    """
    Drawdown series not provided by service yet.
    - 204 until implemented upstream
    """
    try:
        _ = _get_svc()
        # Intentionally return 204 until the service exposes this.
        return Response(status_code=204)
    except HTTPException:
        raise
    except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
        raise HTTPException(status_code=503, detail=f"performance/drawdown failed: {e}") from e


@router.get("/cumulative-returns")
async def cumulative_returns() -> Any:
    """
    Cumulative % returns vs first point in portfolio_value.
    - 204 if insufficient data
    - 503 on failure
    Uses PerformanceAnalyticsService; falls back to paper_trades when service unavailable.
    """
    pv: list[dict[str, Any]] = []
    try:
        svc = _get_svc()
        data = await _maybe_await(svc.get_performance_summary)
        pv = (data or {}).get("portfolio_value", []) if isinstance(data, dict) else []
    except HTTPException:
        pass

    if not pv:
        # Fallback: load from paper_trades (match _load_portfolio_value_from_paper_trades)
        import sqlite3

        from backend.database_schema import DATABASE_PATH

        conn = None
        try:
            conn = sqlite3.connect(DATABASE_PATH)
            cur = conn.cursor()
            initial = 0.0
            try:
                cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='portfolio_engine_ledger'")
                if cur.fetchone():
                    cur.execute("SELECT principal FROM portfolio_engine_ledger WHERE id=1")
                    r = cur.fetchone()
                    if r and r[0] is not None:
                        initial = float(r[0])
            except Exception:
                logger.warning("Failed to load principal from portfolio_engine_ledger", exc_info=True)
            cur.execute("SELECT timestamp, pnl FROM paper_trades WHERE mode IN ('paper','live') AND pnl IS NOT NULL ORDER BY timestamp ASC")
            rows = cur.fetchall()
            cum = 0.0
            for ts_str, pnl in rows:
                cum += float(pnl or 0.0)
                pv.append({"date": ts_str, "value": initial + cum})
        except Exception as e:
            logger.warning("Failed to load portfolio value from paper_trades: %s", e)
        finally:
            if conn:
                conn.close()

    try:
        if not pv:
            return Response(status_code=204)

        initial_raw = _get_field(pv[0], "value") if pv else None
        if initial_raw is None or _as_float(initial_raw) == 0.0:
            return Response(status_code=204)

        series: list[dict[str, Any]] = []
        base = _as_float(initial_raw)
        for row in pv:
            v_raw = _get_field(row, "value")
            if v_raw is None:
                continue
            ts = _row_ts(row)
            if ts is None:
                continue
            ret = (_as_float(v_raw) - base) / base * 100.0
            series.append({"timestamp": ts, "value": round(ret, 4)})

        if not series:
            return Response(status_code=204)
    except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
        raise HTTPException(status_code=503, detail=f"performance/cumulative-returns failed: {e}") from e
    else:
        return {"cumulative": series}


@router.get("/risk-return")
async def risk_return_scatter() -> Any:
    """Not available yet -> 204."""
    try:
        _ = _get_svc()
        return Response(status_code=204)
    except HTTPException:
        raise
    except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
        raise HTTPException(status_code=503, detail=f"performance/risk-return failed: {e}") from e


@router.get("/rolling-sharpe")
async def rolling_sharpe() -> Any:
    """Not available yet -> 204."""
    try:
        _ = _get_svc()
        return Response(status_code=204)
    except HTTPException:
        raise
    except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
        raise HTTPException(status_code=503, detail=f"performance/rolling-sharpe failed: {e}") from e


@router.get("/trade-pnl")
async def trade_pnl_hist() -> Any:
    """
    Histogram input array of per-trade PnL values.
    - 204 if no trades
    - 503 on failure
    Uses PerformanceAnalyticsService first; falls back to paper_trades when empty.
    """
    try:
        values: list[float] = []
        for row in _load_forward_dashboard_sells(limit=500):
            raw = row.get("pnl")
            if raw is not None:
                try:
                    values.append(float(raw))
                except (ValueError, TypeError):
                    pass

        if not values:
            try:
                svc = _get_svc()
                trades = await _maybe_await(svc.get_trade_history, 1000)
                for t in trades:
                    raw = _get_field(t, "pnl")
                    if raw is None:
                        continue
                    exit_type = str(_get_field(t, "exit_type") or "").upper()
                    if exit_type in ("ADMIN_POSITION_CLEAR", "STALE_PRE_CORRECTION_POSITION_CLEAR"):
                        continue
                    try:
                        values.append(float(raw))
                    except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError):
                        continue
            except HTTPException:
                pass

        if not values:
            return Response(status_code=204)
    except HTTPException:
        raise
    except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
        raise HTTPException(status_code=503, detail=f"performance/trade-pnl failed: {e}") from e
    else:
        return {"tradePnl": values}


@router.get("/trade-duration")
async def trade_duration_hist() -> Any:
    """
    Histogram input array of per-trade durations (minutes).
    - 204 if no trades
    - 503 on failure
    Uses PerformanceAnalyticsService first; falls back to paper_trades hold_time_seconds when empty.
    """
    try:
        values: list[float] = []
        for row in _load_forward_dashboard_sells(limit=500):
            raw = row.get("hold_time_seconds")
            if raw is not None:
                try:
                    values.append(float(raw) / 60.0)
                except (ValueError, TypeError):
                    pass

        if not values:
            try:
                svc = _get_svc()
                trades = await _maybe_await(svc.get_trade_history, 1000)
                for t in trades:
                    raw = _get_field(t, "duration_minutes")
                    if raw is not None:
                        try:
                            values.append(float(raw))
                        except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError):
                            pass
                        continue
                    raw = _get_field(t, "hold_time_seconds")
                    if raw is not None:
                        try:
                            values.append(float(raw) / 60.0)
                        except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError):
                            pass
            except HTTPException:
                pass

        if not values:
            return Response(status_code=204)
    except HTTPException:
        raise
    except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
        raise HTTPException(status_code=503, detail=f"performance/trade-duration failed: {e}") from e
    else:
        return {"duration": values}


@router.get("/strategy-performance")
async def strategy_performance() -> Any:
    """
    Map strategy totals to {name, data:[{timestamp,value}]} for multi-line charts.
    - 204 if empty
    - 503 on failure
    """
    try:
        svc = _get_svc()
        data = await _maybe_await(svc.get_performance_summary)
        sp = (data or {}).get("strategy_performance", []) if isinstance(data, dict) else []
        ts = (data or {}).get("timestamp") if isinstance(data, dict) else None
        if not sp:
            return Response(status_code=204)

        series: list[dict[str, Any]] = []
        for row in sp:
            name = row.get("strategy") if isinstance(row, dict) else getattr(row, "strategy", None)
            total_pnl = row.get("total_pnl") if isinstance(row, dict) else getattr(row, "total_pnl", None)
            if name is None or total_pnl is None:
                continue
            try:
                val = float(total_pnl)
            except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError):
                continue
            series.append({"name": name, "data": [{"timestamp": ts, "value": val}]})

        if not series:
            return Response(status_code=204)
    except HTTPException:
        raise
    except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
        raise HTTPException(status_code=503, detail=f"performance/strategy-performance failed: {e}") from e
    else:
        return {"strategies": series}


@router.get("/strategy-correlation")
async def strategy_correlation() -> Any:
    """Not available yet -> 204."""
    try:
        _ = _get_svc()
        return Response(status_code=204)
    except HTTPException:
        raise
    except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
        raise HTTPException(status_code=503, detail=f"performance/strategy-correlation failed: {e}") from e


# ============================================================================
# COOLDOWN & TRADE-STATE MONITORING ENDPOINTS
# ============================================================================


def _sync_get_cooldown_status() -> tuple[dict[str, Any], float]:
    """Sync helper for scratch cooldown status. Runs in thread to avoid blocking async."""
    import time

    from backend.config.redis_config import get_redis_client
    from backend.services.trade_state import TradeStateEnum, get_trade_state_store

    store = get_trade_state_store(get_redis_client())
    symbols = store.get_all_symbols()
    current_time = time.time()
    cooldowns: dict[str, Any] = {}

    for sym in symbols:
        try:
            status = store.get_status(sym)
            if status.get("state") == TradeStateEnum.COOLDOWN:
                remaining = int(status.get("cooldown_remaining_sec") or 0)
                if remaining > 0:
                    cooldown_until = current_time + remaining
                    cooldowns[sym] = {
                        "remaining_seconds": remaining,
                        "cooldown_until": cooldown_until,
                        "cooldown_until_iso": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(cooldown_until)),
                        "exit_reason": status.get("last_exit_reason"),
                        "mode": status.get("mode"),
                        "source": "trade_state",
                    }
        except Exception as _sym_err:
            logger.warning("cooldown-status: error for %s: %s", sym, _sym_err)

    return cooldowns, current_time


@router.get("/scratch-cooldown-status")
async def get_scratch_cooldown_status() -> Any:
    """
    Get current cooldown status for all symbols from the authoritative TradeStateStore.

    Replaces the legacy scratch_cooldown_until:* Redis key scan.
    Reports all symbols in COOLDOWN state with remaining seconds.
    """
    try:
        cooldowns, current_time = await asyncio.to_thread(_sync_get_cooldown_status)
        return {
            "active_cooldowns": cooldowns,
            "total_active": len(cooldowns),
            "timestamp": current_time,
            "source": "trade_state",
        }
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"performance/scratch-cooldown-status failed: {e}") from e
