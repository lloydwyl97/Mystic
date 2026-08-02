"""
Portfolio Engine API Endpoints - Phase 9 Observability

Provides full observability into the Mystic Pro Portfolio Engine:
- Portfolio status (positions, risk, equity)
- Per-coin performance metrics
- Trade explainability (why trades happen)
- Decision history

All endpoints return live data from the portfolio engine.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import os
import sqlite3
from datetime import datetime, timezone
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Request

from backend.services.admin_auth import require_admin_key

from backend.config.trading_universe import TRADING_SYMBOLS
from backend.database_schema import DATABASE_PATH
from backend.services.portfolio_engine import (
    MAX_TOTAL_OPEN_RISK_PCT,
    compute_position_risk_usd,
    get_portfolio_engine,
    initialize_portfolio_engine,
)
from backend.services.portfolio_engine_integration import get_portfolio_integration

_sleeve_cutover_epoch_cache: int | None = None


def _get_sleeve_cutover_epoch() -> int | None:
    """Cached lookup of the sleeve cutover epoch from operational_state."""
    global _sleeve_cutover_epoch_cache
    if _sleeve_cutover_epoch_cache is not None:
        return _sleeve_cutover_epoch_cache
    try:
        import json

        conn = sqlite3.connect(DATABASE_PATH, timeout=2)
        row = conn.execute("SELECT value_json FROM operational_state WHERE key='sleeve_cutover'").fetchone()
        conn.close()
        if row:
            data = json.loads(row[0])
            _sleeve_cutover_epoch_cache = data.get("epoch")
            return _sleeve_cutover_epoch_cache
    except Exception:
        pass
    return None


def _ts_after_cutover(ts: str | None, cutover_epoch: int) -> bool:
    """Return True if a trade timestamp is after the sleeve cutover epoch."""
    if not ts:
        return False
    try:
        dt = datetime.fromisoformat(str(ts).replace("Z", "+00:00"))
        return dt.timestamp() >= cutover_epoch
    except Exception:
        return False


logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/portfolio-engine", tags=["Portfolio Engine"])


def _validate_accounting_identity(cash: float, positions: float, equity: float, realized: float, unrealized: float, total_pnl: float) -> dict[str, Any]:
    """
    INVARIANT CHECK: Validate accounting identities for PAPER mode.

    Must hold:
    1. equity = cash + positions (within 0.01 tolerance)
    2. total_pnl = realized + unrealized (within 0.01 tolerance)

    Returns dict with validation result and any violations found.
    """
    violations = []

    equity_check = abs(equity - (cash + positions))
    if equity_check > 0.01:
        violations.append(f"equity_mismatch: {equity:.2f} != {cash:.2f} + {positions:.2f} (diff={equity_check:.4f})")

    pnl_check = abs(total_pnl - (realized + unrealized))
    if pnl_check > 0.01:
        violations.append(f"pnl_mismatch: {total_pnl:.2f} != {realized:.2f} + {unrealized:.2f} (diff={pnl_check:.4f})")

    if violations:
        logger.error(f"ACCOUNTING_INVARIANT_VIOLATION: {'; '.join(violations)}")

    return {
        "valid": len(violations) == 0,
        "violations": violations,
        "checks": {
            "equity_formula": "total_equity = cash_balance + positions_value",
            "pnl_formula": "total_pnl = realized_pnl + unrealized_pnl",
        },
    }


async def _refresh_engine_from_sqlite_for_live(engine: Any) -> bool:
    """
    Refresh endpoint snapshot from SQLite when portfolio state can be mutated outside
    this process.

    - ``_live_execution_enabled``: live order path may run in a separate worker.
    - ``EXTERNAL_SUPERVISOR_MODE`` (default false): ``start_portfolio_engine_integration.py``
      owns bar/signal execution when set true; then this FastAPI process must read ledger/positions
      from SQLite or status stays stale after BUY_EXECUTED.

    When both are false (typical single-process paper + in-process integration), skip refresh
    on every poll to avoid racing in-process reconcile.
    """
    external_supervisor = os.getenv("EXTERNAL_SUPERVISOR_MODE", "false").lower() == "true"
    if not getattr(engine, "_live_execution_enabled", False) and not external_supervisor:
        return False
    try:
        if hasattr(engine, "_load_ledger_from_sqlite"):
            await engine._load_ledger_from_sqlite()
        if hasattr(engine, "_load_positions_from_sqlite"):
            await engine._load_positions_from_sqlite()
    except Exception as e:
        logger.warning("LIVE_STATUS_REFRESH: SQLite refresh failed: %s", e)
        return False
    return True


def _sqlite_open_positions_count_sync() -> int:
    """Count open rows in portfolio_engine_positions (quantity > 0)."""
    conn = None
    try:
        conn = sqlite3.connect(DATABASE_PATH)
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='portfolio_engine_positions'")
        if not cursor.fetchone():
            return 0
        cursor.execute("SELECT COUNT(*) FROM portfolio_engine_positions WHERE quantity > 0")
        row = cursor.fetchone()
        return int(row[0] or 0) if row else 0
    except Exception as e:
        logger.debug("SQLite open positions count failed: %s", e)
        return 0
    finally:
        if conn:
            conn.close()


async def _ensure_engine_positions_match_sqlite(engine: Any) -> int:
    """
    Ensure in-process engine memory matches SQLite before API responses.

    Under EXTERNAL_SUPERVISOR_MODE the integration worker writes positions to SQLite;
    this FastAPI worker must reload or dashboard/status/positions diverge.
    """
    await _refresh_engine_from_sqlite_for_live(engine)
    sqlite_count = await asyncio.to_thread(_sqlite_open_positions_count_sync)
    engine_count = len(getattr(engine, "open_positions", {}) or {})
    if engine_count != sqlite_count and hasattr(engine, "_load_positions_from_sqlite"):
        try:
            await engine._load_positions_from_sqlite()
            if hasattr(engine, "_recompute_positions_values"):
                mtm = await engine._fetch_mtm_prices_for_open_positions()
                await engine._recompute_positions_values(mtm or None)
        except Exception as e:
            logger.warning("POSITIONS_SQLITE_RESYNC failed: %s", e)
        engine_count = len(getattr(engine, "open_positions", {}) or {})
    if engine_count != sqlite_count:
        logger.warning(
            "POSITIONS_COUNT_MISMATCH: engine=%s sqlite=%s",
            engine_count,
            sqlite_count,
        )
    return sqlite_count


def _read_operator_status_from_sqlite_sync() -> dict[str, Any] | None:
    """
    Read cash_balance, total_equity, positions_value, account_status, trading_paused,
    pause_reason from portfolio_engine_ledger and open_positions_count from
    portfolio_engine_positions. Returns None if no ledger row exists yet.
    """
    conn = None
    try:
        conn = sqlite3.connect(DATABASE_PATH)
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='portfolio_engine_ledger'")
        if not cursor.fetchone():
            return None
        cursor.execute("SELECT cash_balance, total_equity, positions_value, account_status, trading_paused, pause_reason FROM portfolio_engine_ledger WHERE id=1")
        row = cursor.fetchone()
        if not row or row[0] is None:
            return None
        cash_balance, total_equity, positions_value, account_status, trading_paused, pause_reason = row

        cursor.execute("SELECT COUNT(*) FROM portfolio_engine_positions WHERE quantity > 0")
        pos_row = cursor.fetchone()
        open_positions_count = pos_row[0] if pos_row else 0

        return {
            "cash_balance": float(cash_balance or 0),
            "total_equity": float(total_equity or 0),
            "positions_value": float(positions_value or 0),
            "account_status": account_status or "UNKNOWN",
            "trading_paused": bool(trading_paused),
            "pause_reason": pause_reason or "",
            "open_positions_count": int(open_positions_count),
        }
    except Exception as e:
        logger.debug("Operator status SQLite read failed: %s", e)
        return None
    finally:
        if conn:
            conn.close()


async def _read_operator_status_from_sqlite() -> dict[str, Any] | None:
    """Async wrapper for _read_operator_status_from_sqlite_sync."""
    return await asyncio.to_thread(_read_operator_status_from_sqlite_sync)


def _read_positions_for_risk_from_sqlite_sync() -> list[dict[str, Any]]:
    """Read open positions with advisory stop metadata for risk reporting."""
    conn = None
    try:
        conn = sqlite3.connect(DATABASE_PATH)
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='portfolio_engine_positions'")
        if not cursor.fetchone():
            return []
        cursor.execute("SELECT symbol, quantity, entry_price, stop_price FROM portfolio_engine_positions WHERE quantity > 0")
        rows = cursor.fetchall()
        return [
            {
                "symbol": row[0] or "",
                "quantity": float(row[1] or 0),
                "entry_price": float(row[2] or 0),
                "stop_price": float(row[3] or 0),
            }
            for row in rows
        ]
    except Exception as e:
        logger.debug("Positions for risk SQLite read failed: %s", e)
        return []
    finally:
        if conn:
            conn.close()


def _build_position_risk_rows(positions: list[dict[str, Any]], equity: float) -> tuple[list[dict[str, Any]], float]:
    """Return per-position risk rows and total open risk USD."""
    position_risks: list[dict[str, Any]] = []
    total_risk = 0.0
    for pos in positions:
        entry = float(pos.get("entry_price") or 0)
        stop = float(pos.get("stop_price") or 0)
        risk_usd = compute_position_risk_usd(float(pos.get("quantity") or 0), entry, stop)
        stop_distance_pct = ((entry - stop) / entry * 100) if entry > 0 and stop > 0 else 0.0
        total_risk += risk_usd
        position_risks.append(
            {
                "symbol": pos.get("symbol") or "",
                "risk_usd": round(risk_usd, 2),
                "risk_pct": round((risk_usd / equity * 100) if equity > 0 else 0.0, 4),
                "stop_distance_pct": round(stop_distance_pct, 4),
            }
        )
    return position_risks, total_risk


async def _read_positions_for_risk_from_sqlite() -> list[dict[str, Any]]:
    """Async wrapper for _read_positions_for_risk_from_sqlite_sync."""
    return await asyncio.to_thread(_read_positions_for_risk_from_sqlite_sync)


@router.get("/ledger")
async def get_portfolio_ledger() -> dict[str, Any]:
    """
    Get Portfolio Engine ledger - CANONICAL BALANCE SOURCE (BUG-002 Fix)

    This is the authoritative source of truth for all balance/equity data.
    Dashboard and all services must use this endpoint, not PaperTradingService.

    Returns:
    - principal: Initial capital
    - cash_balance: Available cash
    - positions_value: Current value of all open positions
    - realized_pnl: Closed trade profits/losses
    - unrealized_pnl: Open position mark-to-market
    - total_equity: cash + positions_value (INVARIANT: must match)
    - account_status: HEALTHY, OVERALLOCATED, etc.
    - trading_paused: Whether trading is currently paused
    - pause_reason: Reason if paused
    """
    try:
        engine = get_portfolio_engine()

        # Auto-initialize if needed
        from backend.services.portfolio_engine import is_portfolio_engine_initialized

        if not is_portfolio_engine_initialized():
            await initialize_portfolio_engine()

        # Get ledger data from database (canonical source)
        ledger = await engine.get_ledger()

        # BUG #30 FIX: Validate equity invariant
        # INVARIANT: total_equity should equal cash_balance + positions_value
        cash_balance = float(ledger.get("cash_balance", 0))
        positions_value = sum(pos.get("current_value", pos.get("quantity", 0) * pos.get("current_price", 0)) for pos in ledger.get("positions", []))
        total_equity = float(ledger.get("total_equity", 0))
        expected_equity = cash_balance + positions_value

        # Check if invariant holds (allow 0.01 USD tolerance for rounding)
        invariant_check = abs(expected_equity - total_equity) < 0.01

        result = {
            "success": True,
            "ledger": ledger,
            "canonical_source": "portfolio_engine_ledger",
            # BUG #30 FIX: Add invariant validation to response
            "invariant_check": {
                "valid": invariant_check,
                "cash_balance": cash_balance,
                "positions_value": positions_value,
                "total_equity": total_equity,
                "expected_equity": expected_equity,
                "difference": abs(expected_equity - total_equity),
            },
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

        if not invariant_check:
            logger.error(f"EQUITY INVARIANT BROKEN: expected {expected_equity}, got {total_equity}")
            result["warning"] = "Equity invariant violation detected - audit recommended"

        return result
    except Exception as e:
        logger.exception(f"Error getting portfolio ledger: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.get("/performance")
async def get_portfolio_performance() -> dict[str, Any]:
    """
    Get canonical profit/loss metrics from portfolio_engine_ledger.

    CANONICAL SOURCE: portfolio_engine_ledger
    Formulas:
    - total_pnl = realized_pnl + unrealized_pnl
    - equity = cash_balance + positions_value

    Trade stats derived from paper_trades for reference only.
    """
    try:
        import sqlite3

        from backend.database_schema import DATABASE_PATH

        engine = get_portfolio_engine()
        from backend.services.portfolio_engine import is_portfolio_engine_initialized

        if not is_portfolio_engine_initialized():
            await initialize_portfolio_engine()

        await engine._recompute_positions_values()
        ledger = await engine.get_ledger()
        realized_pnl = float(ledger.get("realized_pnl", 0) or 0)
        unrealized_pnl = float(ledger.get("unrealized_pnl", 0) or 0)
        total_equity = float(ledger.get("total_equity", 0) or 0)
        principal = float(ledger.get("principal", 0) or 0)
        cash = float(ledger.get("cash_balance", 0) or 0)
        positions = float(ledger.get("positions_value", 0) or 0)

        def _trades_from_sqlite():
            conn = None
            try:
                conn = sqlite3.connect(DATABASE_PATH)
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT symbol, side, quantity, price, pnl, timestamp,
                           COALESCE(sleeve, 'ACTIVE')
                    FROM paper_trades
                    WHERE UPPER(side) = 'SELL' AND pnl IS NOT NULL
                    ORDER BY timestamp DESC
                    LIMIT 500
                """)
                return cursor.fetchall()
            except Exception as e:
                logger.warning("Failed to fetch trades from SQLite for performance metrics: %s", e)
                return []
            finally:
                if conn:
                    conn.close()

        import asyncio

        loop = asyncio.get_running_loop()
        rows = await loop.run_in_executor(None, _trades_from_sqlite)

        total_trades = len(rows)
        wins = sum(1 for r in rows if r[4] is not None and float(r[4]) > 0)
        losses = sum(1 for r in rows if r[4] is not None and float(r[4]) < 0)
        win_rate = (wins / total_trades * 100) if total_trades > 0 else 0.0

        cutover_epoch = _get_sleeve_cutover_epoch()
        trades_data = [
            {
                "symbol": r[0],
                "side": r[1],
                "quantity": r[2],
                "price": r[3],
                "pnl": float(r[4]) if r[4] is not None else None,
                "timestamp": r[5],
                "sleeve": r[6] if len(r) > 6 else "ACTIVE",
                "sleeve_era": "post" if cutover_epoch and _ts_after_cutover(r[5], cutover_epoch) else "legacy",
            }
            for r in rows[:50]
        ]

        total_pnl = total_equity - principal

        # Validate accounting identity: equity = cash + positions
        if abs(total_equity - (cash + positions)) > 0.01:
            logger.warning(
                f"ACCOUNTING_MISMATCH in /performance: total_equity={total_equity:.2f} != cash={cash:.2f} + positions={positions:.2f}",
            )

        return {
            "success": True,
            "performance": {
                "realized_pnl": realized_pnl,
                "unrealized_pnl": unrealized_pnl,
                "total_pnl": total_pnl,
                "component_pnl_sum": realized_pnl + unrealized_pnl,
                "total_trades": total_trades,
                "winning_trades": wins,
                "losing_trades": losses,
                "win_rate": win_rate,
                "total_equity": total_equity,
                "principal": principal,
                "cash_balance": cash,
                "positions_value": positions,
            },
            "trades": trades_data,
            "canonical_source": "portfolio_engine_ledger",
            "sleeve_cutover_epoch": _get_sleeve_cutover_epoch(),
            "accounting_check": {
                "equity": total_equity,
                "cash_plus_positions": cash + positions,
                "pnl": total_pnl,
                "realized_plus_unrealized": realized_pnl + unrealized_pnl,
                "formula_1": "total_equity = cash_balance + positions_value",
                "formula_2": "total_pnl = total_equity - principal",
                "formula_3": "component_pnl_sum = realized_pnl + unrealized_pnl (open-book MTM; may differ from account return)",
            },
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
    except Exception as e:
        logger.exception(f"Error getting portfolio performance: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.get("/status")
async def get_portfolio_status() -> dict[str, Any]:
    """
    Get full portfolio status including:
    - All open positions with entry price and quantity
    - Total equity and cash balance
    - Position count vs max allowed

    Mystic sells only on confirmed real net profit after costs.

    CANONICAL SOURCE: Adopted from PaperTradingService (not paper_trades)
    """
    try:
        engine = get_portfolio_engine()

        # Check if engine has been initialized from canonical sources
        from backend.services.portfolio_engine import is_portfolio_engine_initialized

        if not is_portfolio_engine_initialized():
            # Auto-initialize from canonical sources
            from backend.services.portfolio_engine import initialize_portfolio_engine

            await initialize_portfolio_engine()

        # Live multi-process: pull ledger/positions from SQLite (see _ensure_engine_positions_match_sqlite).
        # Paper / single-process: skip reload unless external supervisor or live worker is active.
        await _ensure_engine_positions_match_sqlite(engine)
        try:
            await engine._load_ledger_from_sqlite()
            await engine._load_positions_from_sqlite()
        except Exception as e:
            logger.warning("STATUS_SQLITE_RELOAD failed: %s", e)
        await engine._recompute_positions_values(await engine._fetch_mtm_prices_for_open_positions() or None)
        status = engine.get_portfolio_status()
        # Align top-level unrealized with recomputed engine marks (same as ledger MTM persist).
        status["unrealized_pnl"] = engine._unrealized_pnl
        status["positions_value"] = engine._positions_value
        status["total_equity"] = engine._total_equity
        status["account_equity"] = engine._total_equity
        status["operational_equity"] = engine._total_equity
        status["equity_check"] = engine.cash_balance + engine._positions_value
        sqlite_count = await asyncio.to_thread(_sqlite_open_positions_count_sync)
        status["positions"] = status.get("open_positions", [])
        status["sqlite_open_positions_count"] = sqlite_count
        if status.get("positions_count", 0) != sqlite_count:
            logger.warning(
                "STATUS_POSITIONS_MISMATCH: positions_count=%s sqlite=%s",
                status.get("positions_count"),
                sqlite_count,
            )
        engine_status = get_portfolio_integration().get_status()

        # CANONICAL FIX: No Redis overrides in PAPER mode
        # Use portfolio_engine ledger as single source for accounting
        # status already derives from engine which persists to portfolio_engine_ledger

        # Validate operational identity: account_equity == cash + positions_value (canonical ledger)
        cash = status.get("cash_balance", 0.0)
        positions = status.get("positions_value", 0.0)
        operational = float(status.get("account_equity", cash + positions))
        if abs(operational - (cash + positions)) > 0.01:
            logger.warning(
                "OPERATIONAL_EQUITY_MISMATCH: operational=%.2f != cash=%.2f + positions=%.2f",
                operational,
                cash,
                positions,
            )

        # C4: underwater thesis with profit-only hold is normal — not degraded.
        try:
            status["exit_blocked_positions"] = []
        except Exception as e:
            logger.debug("STATUS_DEGRADED_CHECK skipped: %s", e)

        # Ensure we show non-zero equity from adopted data
        return {
            "success": True,
            "data": status,
            "canonical_source": "portfolio_engine_ledger",
            "adopted_equity": status.get("total_equity", engine._total_equity),
            "adopted_cash": status.get("cash_balance", engine.cash_balance),
            "adopted_positions": len(engine.open_positions),
            "dust_pending_positions_current": engine_status["dust_pending_positions_current"],
            "dust_drift_events_total": engine_status["dust_drift_events_total"],
            "dust_reconcile_runs_total": engine_status["dust_reconcile_runs_total"],
            "accounting_check": "total_equity=account_equity=cash_balance+positions_value; performance_equity=principal_based_equity=principal+realized_pnl+unrealized_pnl",
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
    except Exception as e:
        logger.exception(f"Error getting portfolio status: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.get("/metrics")
async def get_engine_metrics() -> dict[str, Any]:
    """P4.1: Expose engine counters for audits (buys_attempted, buys_executed, cooldown_blocks, etc.)."""
    try:
        engine = get_portfolio_engine()
        return {"success": True, "metrics": engine.get_metrics(), "timestamp": datetime.now(timezone.utc).isoformat()}
    except Exception as e:
        logger.exception(f"Error getting engine metrics: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.get("/gates/today")
async def get_day_gates_today(date: str | None = None) -> dict[str, Any]:
    """DAY gate counters for today (or YYYY-MM-DD) — top blockers by hard_blocked."""
    try:
        from backend.services.day_gate_registry import registry_snapshot
        from backend.services.day_gate_telemetry import counters_today, ensure_day_gate_schema

        ensure_day_gate_schema(DATABASE_PATH)
        rows = counters_today(DATABASE_PATH, date=date)
        return {
            "success": True,
            "date": date or datetime.now(timezone.utc).strftime("%Y-%m-%d"),
            "data": {"gates": rows, "top_blockers": rows[:15]},
            "registry": {
                "decision_policy_version": registry_snapshot().get("decision_policy_version"),
                "threshold_freeze_active": registry_snapshot().get("threshold_freeze_active"),
            },
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
    except Exception as e:
        logger.exception("Error getting day gate counters: %s", e)
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.get("/gates/registry")
async def get_day_gate_registry() -> dict[str, Any]:
    """Versioned DAY gate registry snapshot."""
    try:
        from backend.services.day_gate_registry import registry_snapshot

        return {"success": True, "data": registry_snapshot(), "timestamp": datetime.now(timezone.utc).isoformat()}
    except Exception as e:
        logger.exception("Error getting day gate registry: %s", e)
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.get("/attribution/today")
async def get_day_attribution_today(date: str | None = None) -> dict[str, Any]:
    """Executed PnL + gate opportunity (shadow rejects) for DAY measurement window."""
    try:
        from backend.services.day_gate_telemetry import attribution_report, ensure_day_gate_schema, shadow_rejects_summary

        ensure_day_gate_schema(DATABASE_PATH)
        report = attribution_report(DATABASE_PATH, date=date)
        shadows = shadow_rejects_summary(DATABASE_PATH, limit=30)
        return {
            "success": True,
            "data": {**report, "shadow_summary": shadows},
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
    except Exception as e:
        logger.exception("Error getting day attribution: %s", e)
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.get("/positions")
async def get_open_positions() -> dict[str, Any]:
    """
    Get all open positions with the fields Mystic actually uses.

    Mystic sells ONLY when real net profit after costs is confirmed.
    """
    try:
        engine = get_portfolio_engine()
        from backend.services.portfolio_engine import is_portfolio_engine_initialized

        if not is_portfolio_engine_initialized():
            await initialize_portfolio_engine()

        await _ensure_engine_positions_match_sqlite(engine)
        try:
            await engine._load_ledger_from_sqlite()
            await engine._load_positions_from_sqlite()
        except Exception as e:
            logger.warning("POSITIONS_SQLITE_RELOAD failed: %s", e)
        await engine._recompute_positions_values(await engine._fetch_mtm_prices_for_open_positions() or None)
        positions = []

        for symbol, pos in engine.open_positions.items():
            if getattr(pos, "status", "ACTIVE") == "DUST_PENDING":
                continue
            positions.append(engine.build_open_position_api_row(symbol, pos))
        engine.enrich_open_position_rows_from_buy_explain(positions)

        return {
            "success": True,
            "data": {
                "positions": positions,
                "count": len(positions),
                "max_positions": engine.get_max_open_positions(),
                "position_exit_policy": {
                    "automated_sells_triggered_by": "executable_net_profit_after_costs_only",
                    "stop_tp_fields_are_advisory_metadata_only": True,
                },
            },
            "positions": positions,
            "count": len(positions),
            "max_positions": engine.get_max_open_positions(),
            "position_exit_policy": {
                "automated_sells_triggered_by": "executable_net_profit_after_costs_only",
                "stop_tp_fields_are_advisory_metadata_only": True,
            },
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
    except Exception as e:
        logger.exception(f"Error getting positions: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.get("/coin/{symbol}")
async def get_coin_status(symbol: str) -> dict[str, Any]:
    """
    Get per-coin performance metrics:
    - Is paused and pause end time
    - 24h trade count and P&L
    - Rolling 20-trade win rate
    - Expectancy and sizing multiplier
    """
    try:
        symbol = symbol.upper().replace("/", "").replace("-", "").replace("_", "")
        if not symbol.endswith("USDT"):
            symbol = f"{symbol}USDT"

        engine = get_portfolio_engine()
        status = engine.get_coin_status(symbol)

        return {
            "success": True,
            "data": status,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
    except Exception as e:
        logger.exception(f"Error getting coin status for {symbol}: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.get("/coins")
async def get_all_coins_status() -> dict[str, Any]:
    """
    Get performance metrics for all coins in the trading universe.
    """
    try:
        engine = get_portfolio_engine()

        coins = []
        for symbol in TRADING_SYMBOLS:
            ccxt_symbol = f"{symbol[:-4]}/USDT" if symbol.endswith("USDT") else symbol
            status = engine.get_coin_status(ccxt_symbol)
            coins.append(status)

        return {
            "success": True,
            "coins": coins,
            "count": len(coins),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
    except Exception as e:
        logger.exception(f"Error getting all coins status: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.get("/decisions")
async def get_last_decisions(count: int = 10) -> dict[str, Any]:
    """
    Get last N trade decisions with full explainability:
    - Why the trade was entered (scores, features)
    - Why it exited (trigger, R-multiple)
    - Regime state at entry
    - Coin performance snapshot

    When in-memory trade_explanations is empty, falls back to recent paper_trades
    so the panel shows trade activity.
    """
    try:
        if count < 1 or count > 100:
            count = 10

        engine = get_portfolio_engine()
        decisions = engine.get_last_decisions(count)

        return {
            "success": True,
            "decisions": decisions,
            "count": len(decisions),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
    except Exception as e:
        logger.exception(f"Error getting decisions: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.get("/risk")
async def get_risk_metrics() -> dict[str, Any]:
    """
    Get portfolio-level account-state metrics for the dashboard:
    - Account status (HEALTHY, ACCOUNT_OVERALLOCATED, DELEVERAGING)
    - Equity invariant status
    - Open risk to advisory stops (USD and % of equity)

    Dashboard fix: Prefer SQLite (ledger + positions) when available.
    Integration writes there; API must read shared truth, not stale engine state.
    """
    try:
        engine = get_portfolio_engine()
        sqlite_ledger = await _read_operator_status_from_sqlite()
        sqlite_positions = await _read_positions_for_risk_from_sqlite()
        max_risk_pct = MAX_TOTAL_OPEN_RISK_PCT * 100

        if sqlite_ledger is not None:
            equity = sqlite_ledger["total_equity"]
            cash_balance = sqlite_ledger["cash_balance"]
            positions_value = sqlite_ledger["positions_value"]
            account_status = sqlite_ledger["account_status"]
            trading_paused = sqlite_ledger["trading_paused"]
            equity_check = cash_balance + positions_value
            equity_invariant_ok = abs(equity - equity_check) < 1.0
            position_risks, total_risk = _build_position_risk_rows(sqlite_positions, equity)
            risk_pct = (total_risk / equity * 100) if equity > 0 else 0.0
            pause_reason = sqlite_ledger.get("pause_reason", "") if trading_paused else None
        else:
            equity = engine._total_equity
            cash_balance = engine.cash_balance
            positions_value = engine._positions_value
            equity_check = cash_balance + positions_value
            equity_invariant_ok = abs(equity - equity_check) < 1.0
            account_status = engine._account_status.value
            trading_paused = engine._trading_paused
            pause_reason = engine._pause_reason if engine._trading_paused else None
            engine_positions = [
                {
                    "symbol": symbol,
                    "quantity": pos.quantity,
                    "entry_price": pos.entry_price,
                    "stop_price": pos.stop_price,
                }
                for symbol, pos in engine.open_positions.items()
                if getattr(pos, "status", "ACTIVE") != "DUST_PENDING"
            ]
            position_risks, total_risk = _build_position_risk_rows(engine_positions, equity)
            risk_pct = (total_risk / equity * 100) if equity > 0 else 0.0

        # CANONICAL: DB ledger + engine-derived marks — do not override cash/equity with Redis
        # (paper_trading:cash_balance can lag PaperTradingService and falsify risk / overallocated).

        return {
            "success": True,
            "data": {
                "account_status": account_status,
                "trading_paused": trading_paused,
                "pause_reason": pause_reason,
                "equity_invariant_ok": equity_invariant_ok,
                "overallocated": not equity_invariant_ok and positions_value > equity,
                "total_equity": equity,
                "cash_balance": cash_balance,
                "positions_value": positions_value,
                "available_balance": cash_balance,
                "total_open_risk_usd": round(total_risk, 2),
                "total_open_risk_pct": round(risk_pct, 4),
                "max_risk_pct": max_risk_pct,
                "risk_cap_remaining_pct": max(0, max_risk_pct - risk_pct),
                "position_risks": position_risks,
            },
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
    except Exception as e:
        logger.exception(f"Error getting risk metrics: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.get("/invariants")
async def check_invariants() -> dict[str, Any]:
    """
    Check all system invariants and return status.
    Used for monitoring and debugging.

    Dashboard fix: Use SQLite (portfolio_engine_positions) as canonical.
    Integration writes there; API must read shared truth, not stale engine state.
    """
    try:
        engine = get_portfolio_engine()
        sqlite_positions = await _read_positions_for_risk_from_sqlite()
        sqlite_symbols = {p["symbol"] for p in sqlite_positions}
        sqlite_count = len(sqlite_positions)

        # Run invariant check (engine state; may be stale in multi-process)
        is_valid = await engine._validate_invariants("api_check")
        memory_positions = set(engine.open_positions.keys())

        # Position data from SQLite (matches positions table right after buys/sells)
        return {
            "success": True,
            "data": {
                "all_invariants_pass": is_valid,
                "total_violations": engine.invariant_violations,
                "engine_positions": list(memory_positions),
                "canonical_positions": list(sqlite_symbols),
                "positions_match": memory_positions == sqlite_symbols,
                "position_count": sqlite_count,
                "max_positions": 10,
                "under_limit": sqlite_count <= 10,
                "canonical_source": "portfolio_engine_positions",
            },
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
    except Exception as e:
        logger.exception(f"Error checking invariants: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.post("/initialize")
async def initialize_engine(_: None = Depends(require_admin_key)) -> dict[str, Any]:
    """
    Initialize or re-initialize the portfolio engine from database.
    Call this after restarts or to recover from errors.
    """
    try:
        engine = await initialize_portfolio_engine(force=True)

        return {
            "success": True,
            "message": "Portfolio engine initialized from database",
            "data": {
                "open_positions": len(engine.open_positions),
                "cash_balance": engine.cash_balance,
                "principal": engine.principal,
            },
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
    except Exception as e:
        logger.exception(f"Error initializing engine: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.get("/explain/{trade_id}")
async def get_trade_explanation(trade_id: str) -> dict[str, Any]:
    """
    Get full explanation for a specific trade.
    Shows why the trade was entered and exited.
    """
    try:
        engine = get_portfolio_engine()

        if trade_id in engine.trade_explanations:
            explanation = engine.trade_explanations[trade_id].to_dict()
            return {
                "success": True,
                "data": explanation,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
        else:
            return {
                "success": False,
                "error": f"Trade {trade_id} not found in explanations",
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
    except Exception as e:
        logger.exception(f"Error getting trade explanation: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.get("/debug/adoption")
async def get_adoption_debug_info() -> dict[str, Any]:
    """
    DEBUG ENDPOINT: Get adoption snapshot for verification.

    Returns:
    - Portfolio snapshot (cash, equity, principal, risk)
    - Positions snapshot (symbol, qty, entry, stops)
    - DB path and invariant status

    Use this to verify:
    - /api/portfolio-engine/status shows correct non-zero equity
    - /api/portfolio-engine/positions matches /api/paper-trading/status positions
    - Invariants pass immediately on startup
    """
    try:
        engine = get_portfolio_engine()
        debug_info = engine.get_adoption_debug_info()

        # Also get canonical source info for comparison
        canonical_positions = await engine._get_canonical_positions()

        return {
            "success": True,
            "adoption_snapshot": debug_info,
            "canonical_positions": {
                "source": "PaperTradingService.positions",
                "symbols": list(canonical_positions.keys()),
                "quantities": canonical_positions,
            },
            "match_check": {
                "engine_symbols": debug_info["symbols"],
                "canonical_symbols": list(canonical_positions.keys()),
                "match": set(debug_info["symbols"]) == set(canonical_positions.keys()),
            },
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
    except Exception as e:
        logger.exception(f"Error getting adoption debug info: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e


# =============================================================================
# ELITE HARDENING ENDPOINTS (Items 1-7)
# =============================================================================


@router.post("/control")
async def set_kill_switch(mode: str, reason: str = "", _: None = Depends(require_admin_key)) -> dict[str, Any]:
    """
    ITEM 1: Kill switch control endpoint.

    Modes:
    - PAUSE_ALL: Block all BUYs and SELLs (engine keeps running but executes nothing)
    - PAUSE_BUYS: Block BUYs only, allow SELLs (including deleverage and normal exits)
    - RESUME: Normal operation

    Persists across restarts.
    """
    try:
        engine = get_portfolio_engine()
        result = await engine.set_kill_switch(mode, reason)

        return {
            "success": result.get("success", False),
            "data": result,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
    except Exception as e:
        logger.exception(f"Error setting kill switch: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.get("/control")
async def get_kill_switch_status() -> dict[str, Any]:
    """Get current kill switch status"""
    try:
        engine = get_portfolio_engine()
        status = engine.get_kill_switch_status()

        return {
            "success": True,
            "data": status,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
    except Exception as e:
        logger.exception(f"Error getting kill switch status: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.post("/sync-from-binance")
async def force_sync_from_binance(_: None = Depends(require_admin_key)) -> dict[str, Any]:
    """
    Force re-sync from Binance US (LIVE mode only).
    Use when dashboard shows wrong balance/positions after a clear or disconnect.
    """
    try:
        engine = get_portfolio_engine()
        if not getattr(engine, "_live_execution_enabled", False):
            return {
                "success": False,
                "message": "LIVE_EXECUTION disabled - sync from Binance only applies in live mode",
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
        live_svc = getattr(engine, "_live_service", None)
        if not live_svc:
            return {
                "success": False,
                "message": "Live trading service not initialized",
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
        pre_cash = engine.cash_balance
        pre_positions = len(engine.open_positions)
        pre_equity = engine._total_equity

        balance_result = await live_svc.get_balance("binanceus")
        if balance_result.get("status") != "success":
            return {
                "success": False,
                "message": f"Binance balance fetch failed: {balance_result.get('error', 'unknown')}",
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
        total_balances = balance_result.get("balance", {}).get("total", {}) or {}
        free_balances = balance_result.get("balance", {}).get("free", {}) or {}
        exchange_usdt = float(free_balances.get("USDT", 0) or 0)
        await engine.sync_cash_from_exchange(exchange_usdt, "FORCE_SYNC")
        await engine.run_live_reconcile(total_balances, free_balances=free_balances)

        return {
            "success": True,
            "message": "Synced from Binance US",
            "before": {"cash": pre_cash, "positions": pre_positions, "equity": pre_equity},
            "after": {
                "cash": engine.cash_balance,
                "positions": len(engine.open_positions),
                "equity": engine._total_equity,
            },
            "positions": [p.symbol for p in engine.open_positions.values()],
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
    except Exception as e:
        logger.exception("Force Binance sync failed: %s", e)
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.post("/sync")
async def sync_from_paper_trading(_: None = Depends(require_admin_key)) -> dict[str, Any]:
    """
    Force re-sync portfolio engine from Paper Trading Service.

    Use this when positions/balances are out of sync.
    In live mode, prefer POST /sync-from-binance to re-pull from Binance.
    """
    try:
        engine = get_portfolio_engine()

        # LIVE: if Binance-connected, run live reconcile first to import exchange positions
        if getattr(engine, "_live_execution_enabled", False) and getattr(engine, "_live_service", None):
            live_svc = engine._live_service
            balance_result = await live_svc.get_balance("binanceus")
            if balance_result.get("status") == "success":
                total_balances = balance_result.get("balance", {}).get("total", {}) or {}
                free_balances = balance_result.get("balance", {}).get("free", {}) or {}
                exchange_usdt = float(free_balances.get("USDT", 0) or 0)
                await engine.sync_cash_from_exchange(exchange_usdt, "SYNC_LIVE")
                await engine.run_live_reconcile(total_balances, free_balances=free_balances)
                return {
                    "success": True,
                    "message": "Re-synced from Binance US (live mode)",
                    "cash": engine.cash_balance,
                    "positions": len(engine.open_positions),
                    "equity": engine._total_equity,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                }
            # Fall through to paper adopt if Binance fetch failed
        # CRITICAL: Ensure database schema exists before any operations
        engine._ensure_db_schema()

        # Store pre-sync state
        pre_cash = engine.cash_balance
        pre_positions = len(engine.open_positions)
        pre_equity = engine._total_equity

        # Re-adopt from canonical sources
        await engine.adopt_from_portfolios_table()
        await engine.adopt_from_positions_table()

        # Compute risk
        engine._compute_total_open_risk()

        # Validate
        await engine._validate_invariants("manual_sync")

        return {
            "success": True,
            "message": "Re-synced from Paper Trading Service",
            "before": {
                "cash_balance": pre_cash,
                "positions_count": pre_positions,
                "total_equity": pre_equity,
            },
            "after": {
                "cash_balance": engine.cash_balance,
                "positions_count": len(engine.open_positions),
                "total_equity": engine._total_equity,
                "positions_value": engine._positions_value,
            },
            "positions": [{"symbol": p.symbol, "qty": p.quantity, "entry": p.entry_price} for p in engine.open_positions.values()],
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
    except Exception as e:
        logger.exception(f"Error syncing portfolio engine: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.get("/dashboard-canonical")
async def get_dashboard_canonical() -> dict[str, Any]:
    """
    Single coherent snapshot for the dashboard: positions, sleeves, PnL, risk, trades, scoreboard, daily stats.
    All panels should use this path to avoid split-brain between SQLite rows and empty engine memory.
    """
    try:
        engine = get_portfolio_engine()
        from backend.services.portfolio_engine import is_portfolio_engine_initialized

        if not is_portfolio_engine_initialized():
            await initialize_portfolio_engine()

        await _ensure_engine_positions_match_sqlite(engine)
        payload = await engine.build_dashboard_canonical_snapshot()
        from backend.services.market_data_readiness_probe import market_data_dashboard_meta_async

        md_meta = await market_data_dashboard_meta_async()
        md_meta_readiness_fields = md_meta.copy()
        md_meta_readiness_fields["ai_can_act"] = None  # Lite meta only; probe endpoint authorizes counts
        md_meta_readiness_fields["dashboard_note"] = "Call GET /api/portfolio-engine/market-data-readiness for full live Binance checks."

        merged = dict(payload)
        merged["market_data_readiness_dashboard"] = md_meta_readiness_fields

        return {
            "success": True,
            "data": merged,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
    except Exception as e:
        logger.exception("Error building dashboard canonical snapshot: %s", e)
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.get("/market-data-readiness")
async def get_market_data_readiness() -> dict[str, Any]:
    """
    Live Binance.US visibility probe (DAY_TRADE_SYMBOLS only).

    Intended for dashboards and smoke tests; performs multiple REST fetches per symbol.
    """
    try:
        from backend.services.market_data_readiness_probe import probe_market_data_readiness

        data = await probe_market_data_readiness()
        return {
            "success": bool(data.get("success")),
            "data": data,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
    except Exception as e:
        logger.exception("market-data-readiness probe failed: %s", e)
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.get("/execution-protection")
async def get_execution_protection() -> dict[str, Any]:
    """Protected limit execution config + last preflight telemetry."""
    try:
        engine = get_portfolio_engine()
        return {
            "success": True,
            "data": engine.get_execution_protection_state(),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
    except Exception as e:
        logger.exception("Error getting execution protection state: %s", e)
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.get("/trading-economics")
async def get_trading_economics_endpoint() -> dict[str, Any]:
    """Canonical fee/cost model for dashboard (Binance.US Advanced Spot)."""
    try:
        from backend.config.binance_us_fee_schedule import verify_top_four_pairs
        from backend.config.trading_economics import get_trading_economics_display
        from backend.services.fill_fee_audit import bnb_fee_discount_status, config_fee_override_locations

        display = get_trading_economics_display()
        verification = await asyncio.to_thread(verify_top_four_pairs)
        return {
            "success": True,
            "data": {
                **display,
                "bnb_fee_discount": bnb_fee_discount_status(),
                "fee_override_locations": config_fee_override_locations(),
                "binance_us_verification": verification,
            },
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
    except Exception as e:
        logger.exception("Error getting trading economics: %s", e)
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.get("/fill-fee-audit")
async def get_fill_fee_audit(limit: int = 25) -> dict[str, Any]:
    """Recent fill-fee accounting audit rows (expected vs actual on SELL)."""
    try:
        from backend.services.fill_fee_audit import ensure_fill_fee_audit_table

        ensure_fill_fee_audit_table()
        lim = max(1, min(int(limit), 200))

        def _fetch() -> list[dict[str, Any]]:
            conn = sqlite3.connect(DATABASE_PATH, timeout=3)
            conn.row_factory = sqlite3.Row
            try:
                rows = conn.execute(
                    """
                    SELECT id, ts, trade_id, symbol, mode,
                           expected_fee_usd, actual_fee_usd,
                           expected_spread_slippage_usd, realized_slippage_usd,
                           net_pnl_after_actual_fees, fee_delta_usd, slippage_delta_usd,
                           fee_from_exchange, audit_json
                    FROM portfolio_engine_fill_fee_audit
                    ORDER BY id DESC LIMIT ?
                    """,
                    (lim,),
                ).fetchall()
                return [dict(r) for r in rows]
            finally:
                conn.close()

        data = await asyncio.to_thread(_fetch)
        return {"success": True, "data": data, "count": len(data)}
    except Exception as e:
        logger.exception("Error getting fill-fee audit: %s", e)
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.get("/health-pack")
async def get_health_pack(decisions: int = 50, rejects: int = 50, errors: int = 50) -> dict[str, Any]:
    """
    ITEM 2: Comprehensive health pack snapshot.

    Returns:
    - Ledger snapshot (principal/cash/positions_value/equity/pnls/account_status)
    - Open positions (full per-position state)
    - Risk metrics (open risk, cap, invariant status)
    - Last N decisions
    - Last N rejects
    - Uptime + last restart timestamp
    """
    try:
        engine = get_portfolio_engine()
        health_pack = await engine.get_health_pack(decisions, rejects, errors)

        return {
            "success": True,
            "data": health_pack,
        }
    except Exception as e:
        logger.exception(f"Error getting health pack: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.get("/closes")
async def get_position_close_ledger(limit: int = 50) -> dict[str, Any]:
    """Read the canonical position close ledger.

    Returns the most recent N rows from ``position_close_ledger`` (single
    source of truth for post-sell cooldowns). Each row covers exactly one
    closure event:

    * ``close_reason = AI_TP1`` / ``AI_TAKE_PROFIT_FULL`` -> AI net-profit sell.
    * ``close_reason = MANUAL``                          -> operator sell.
    * ``close_reason = HUMAN_MANUAL_SELL``               -> human closed on
      Binance.US; ``realized_profit_unknown=true`` when the fill could not
      be recovered from the exchange.
    * ``close_reason = DUST_WRITEOFF``                   -> dust cleanup.

    ``cooldown_until`` is epoch seconds; new buys on the symbol are blocked
    until that time regardless of close reason.
    """
    try:
        if limit < 1:
            limit = 50
        limit = min(limit, 500)
        engine = get_portfolio_engine()
        rows = engine.get_position_close_ledger(limit=limit)
        return {
            "success": True,
            "data": rows,
            "count": len(rows),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
    except Exception as e:
        logger.exception(f"Error getting position close ledger: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.get("/audit")
async def get_audit_trail(limit: int = 200) -> dict[str, Any]:
    """
    ITEM 3: Get immutable execution audit trail.

    Returns append-only audit records with:
    - Pre/post ledger snapshots
    - Position digests
    - Invariant status
    - Entry/exit reasons
    """
    try:
        if limit < 1:
            limit = 200
        limit = min(limit, 1000)

        engine = get_portfolio_engine()
        records = await engine._get_recent_audit(limit)

        return {
            "success": True,
            "records": records,
            "count": len(records),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
    except Exception as e:
        logger.exception(f"Error getting audit trail: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.get("/audit/export")
async def export_audit(hours: int = 24) -> dict[str, Any]:
    """
    Export audit records for last N hours (for download/reporting).
    """
    try:
        if hours < 1:
            hours = 24
        hours = min(hours, 168)

        engine = get_portfolio_engine()
        records = await engine.get_audit_export(hours)

        return {
            "success": True,
            "hours": hours,
            "records": records,
            "count": len(records),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
    except Exception as e:
        logger.exception(f"Error exporting audit: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.get("/audit/{audit_id}")
async def get_audit_record(audit_id: int) -> dict[str, Any]:
    """Get single audit record by ID with full details"""
    try:
        engine = get_portfolio_engine()
        record = await engine.get_audit_by_id(audit_id)

        if not record:
            raise HTTPException(status_code=404, detail=f"Audit record {audit_id} not found")

        return {
            "success": True,
            "record": record,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Error getting audit record: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.get("/regime")
async def get_regime_status() -> dict[str, Any]:
    """
    ITEM 4: Get current regime guardrail status.

    Shows active guards:
    - VOL_SPIKE: Volatility spike reduction
    - SPREAD_SPIKE: Symbol-level spread blocks
    - DRAWDOWN_GUARD: Equity drawdown protection
    - CHURN_GUARD: Fee/slippage churn protection
    """
    try:
        engine = get_portfolio_engine()
        from backend.services.portfolio_engine import is_portfolio_engine_initialized

        if not is_portfolio_engine_initialized():
            await initialize_portfolio_engine()
        await engine.sync_regime_label_from_redis(force=True)
        status = engine.get_regime_status()

        # Extract regime and confidence from state
        state = status.get("state", {})
        regime = state.get("regime", "unknown")

        # Get confidence from regime state if available
        confidence = getattr(engine._regime_state, "confidence", 0.0) if hasattr(engine._regime_state, "confidence") else 0.0

        return {
            "success": True,
            "data": {
                **status,
                "regime": regime,
                "confidence": confidence,
            },
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
    except Exception as e:
        logger.exception(f"Error getting regime status: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.get("/orders")
async def get_pending_orders() -> dict[str, Any]:
    """
    ITEM 6: Get pending orders (for partial fill simulation).
    """
    try:
        engine = get_portfolio_engine()
        orders = await engine.get_pending_orders()

        return {
            "success": True,
            "orders": orders,
            "count": len(orders),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
    except Exception as e:
        logger.exception(f"Error getting pending orders: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.get("/scoreboard")
async def get_scoreboard(days: int = 7) -> dict[str, Any]:
    """
    ITEM 7: Get validation scoreboard for last N days.

    Metrics:
    - trades, win_rate, expectancy_R
    - profit_factor, max_drawdown
    - fees_paid, slippage_cost, churn_ratio
    - PASS/FAIL status with reasons
    """
    try:
        if days < 1:
            days = 7
        days = min(days, 90)

        engine = get_portfolio_engine()
        scoreboard = await engine.get_scoreboard(days)

        return {
            "success": True,
            "data": scoreboard,
        }
    except Exception as e:
        logger.exception(f"Error getting scoreboard: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.get("/scoreboard/today")
async def get_scoreboard_today() -> dict[str, Any]:
    """Get today's scoreboard only"""
    try:
        engine = get_portfolio_engine()

        # Ensure scoreboard is updated
        await engine.update_scoreboard()

        today = await engine.get_scoreboard_today()

        return {
            "success": True,
            "data": today,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
    except Exception as e:
        logger.exception(f"Error getting today's scoreboard: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.get("/rejects")
async def get_recent_rejects(limit: int = 50) -> dict[str, Any]:
    """
    ITEM 2 (supplement): Get recent rejected trade attempts.

    Shows why trades were blocked with:
    - Filter name (KILL_SWITCH, REGIME_GUARD, etc.)
    - Candidate scores
    - Ledger snapshot at rejection time
    """
    try:
        if limit < 1:
            limit = 50
        limit = min(limit, 500)

        engine = get_portfolio_engine()
        rejects = await engine._get_recent_rejects(limit)

        return {
            "success": True,
            "rejects": rejects,
            "count": len(rejects),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
    except Exception as e:
        logger.exception(f"Error getting rejects: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e


# =============================================================================
# OPERATOR CONSOLE ENDPOINTS (Phase A)
# =============================================================================


@router.get("/execution-mode")
async def get_execution_mode_status() -> dict[str, Any]:
    """
    Get execution mode status (EXECUTION_MODE + LIVE_TRADES_ALLOWED).
    Used by dashboard switch.
    """
    try:
        from backend.services.execution_mode_service import get_execution_status

        status = await get_execution_status()
        return {"success": True, "data": status}
    except Exception as e:
        logger.exception("Error getting execution mode: %s", e)
        raise HTTPException(status_code=500, detail=str(e)) from e


def _write_env_vars(updates: dict) -> None:
    """Read .env, update/append key=value lines, write back."""
    env_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), ".env")
    try:
        with open(env_path) as f:
            lines = f.readlines()
    except FileNotFoundError:
        lines = []
    result = []
    found: set[str] = set()
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("#") or "=" not in stripped:
            result.append(line)
            continue
        key = stripped.split("=", 1)[0].strip()
        if key in updates:
            result.append(f"{key}={updates[key]}\n")
            found.add(key)
        else:
            result.append(line)
    for key, val in updates.items():
        if key not in found:
            result.append(f"{key}={val}\n")
    with open(env_path, "w") as f:
        f.writelines(result)


@router.post("/execution-mode")
async def set_execution_mode(
    request: Request,
    payload: dict[str, Any],
    _: None = Depends(require_admin_key),
) -> dict[str, Any]:
    """
    Set execution mode. Body: { "mode": "paper"|"live", "live_trades_allowed": bool }.

    - mode=paper: write paper flags to .env, return restart_required
    - mode=live: validate API keys, write live flags to .env, return restart_required

    Switching TO live requires ADMIN_TOKEN authentication.
    The upfront check_live_readiness() gate is intentionally removed — it required
    LIVE_EXECUTION=true at process start, creating a chicken-and-egg deadlock when
    starting from a paper boot.  Env flags are validated here directly instead.
    """
    try:
        mode = (payload.get("mode") or "").strip().lower()

        going_live = mode == "live"
        if going_live:
            from backend.middleware.security import AdminAuthMiddleware

            if not AdminAuthMiddleware.verify_admin_token(request):
                raise HTTPException(
                    status_code=401,
                    detail="Unauthorized - ADMIN_TOKEN required to enable live trading",
                    headers={"WWW-Authenticate": "Bearer"},
                )

            # Validate required secrets exist in the environment
            missing = [k for k in ("BINANCE_API_KEY", "BINANCE_SECRET_KEY", "ADMIN_TOKEN") if not os.getenv(k)]
            if missing:
                return {
                    "success": False,
                    "error": f"Missing required env vars: {', '.join(missing)}",
                    "status": "validation_failed",
                }

            _write_env_vars(
                {
                    "TRADING_MODE": "live",
                    "MYSTIC_TRADING_MODE": "live",
                    "LIVE_EXECUTION": "true",
                    "LIVE_TRADES_ALLOWED": "true",
                    "FULL_LIVE_CONFIRMED": "true",
                    "EXECUTION_MODE": "live",
                }
            )
            logger.info("Execution mode: live flags written to .env (restart required)")
            return {
                "success": True,
                "status": "restart_required",
                "message": (
                    "Live flags written to .env. Service must restart to activate real order placement. "
                    "Use /api/system/restart to apply."
                ),
            }

        if mode == "paper":
            _write_env_vars(
                {
                    "TRADING_MODE": "paper",
                    "MYSTIC_TRADING_MODE": "paper",
                    "LIVE_EXECUTION": "false",
                    "LIVE_TRADES_ALLOWED": "false",
                    "FULL_LIVE_CONFIRMED": "false",
                    "EXECUTION_MODE": "paper",
                }
            )
            logger.info("Execution mode: paper flags written to .env (restart required)")
            return {
                "success": True,
                "status": "restart_required",
                "message": (
                    "Paper flags written to .env. Service must restart to deactivate live orders. "
                    "Use /api/system/restart to apply."
                ),
            }

        # Fallback: unknown mode
        return {"success": False, "error": f"Unknown mode '{mode}'. Use 'paper' or 'live'."}

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Error setting execution mode: %s", e)
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.get("/operator-config")
async def get_operator_config_endpoint() -> dict[str, Any]:
    """Dashboard: read max positions, live caps, sizing, kill switch."""
    try:
        from backend.services.operator_config_service import get_operator_config

        data = await get_operator_config()
        return {"success": True, "data": data}
    except Exception as e:
        logger.exception("Error getting operator config: %s", e)
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.get("/live-readiness")
async def get_live_readiness() -> dict[str, Any]:
    """Read-only live readiness probe for operator dashboard (no secrets)."""
    try:
        from backend.services.live_readiness_service import build_live_readiness_report

        data = await build_live_readiness_report()
        return {
            "success": True,
            "data": data,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
    except Exception as e:
        logger.exception("Error building live readiness report: %s", e)
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.get("/trade/{trade_id:path}")
async def get_trade_drilldown(trade_id: str) -> dict[str, Any]:
    """Read-only single-trade drill-down packet."""
    try:
        from backend.services.trade_drilldown_service import build_trade_drilldown

        data = build_trade_drilldown(trade_id)
        if not data.get("found"):
            raise HTTPException(status_code=404, detail=data.get("error") or "trade not found")
        return {
            "success": True,
            "data": data,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Error building trade drilldown for %s: %s", trade_id, e)
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.get("/model-panel")
async def get_model_panel() -> dict[str, Any]:
    """Read-only model/learning visibility per top-4 symbol (no auto-promotion)."""
    try:
        from backend.database_schema import DATABASE_PATH
        from backend.services.ai_market_diagnostics import build_model_freshness_report
        from backend.services.ai_model_behavior_diagnostics import build_model_behavior_report

        behavior = build_model_behavior_report(DATABASE_PATH)
        freshness = build_model_freshness_report(DATABASE_PATH)
        symbols = behavior.get("symbols") if isinstance(behavior, dict) else {}
        promo_events: dict[str, dict[str, Any]] = {}
        try:
            import sqlite3

            conn = sqlite3.connect(DATABASE_PATH)
            try:
                cur = conn.cursor()
                cur.execute(
                    """
                    SELECT symbol, event_type, reason, created_at
                    FROM ai_model_promotion_events
                    ORDER BY id DESC
                    LIMIT 40
                    """
                )
                for sym, evt, reason, ts in cur.fetchall():
                    key = str(sym or "").upper()
                    if key not in promo_events:
                        promo_events[key] = {"last_event_type": evt, "last_event_reason": reason, "last_event_at": ts}
            finally:
                conn.close()
        except Exception:
            pass

        per_symbol = []
        if isinstance(symbols, dict):
            for sym, meta in symbols.items():
                if not isinstance(meta, dict):
                    continue
                pe = promo_events.get(str(sym).upper(), {})
                active_path = meta.get("active_path")
                trained_at = None
                if active_path:
                    try:
                        from pathlib import Path

                        p = Path(str(active_path))
                        if p.exists():
                            trained_at = datetime.fromtimestamp(p.stat().st_mtime, tz=timezone.utc).isoformat()
                    except Exception:
                        trained_at = None
                per_symbol.append(
                    {
                        "symbol": sym,
                        "active_model_path": active_path,
                        "active_model_trained_at": trained_at,
                        "active_accuracy": meta.get("active_artifact_accuracy_stored"),
                        "candidate_accuracy": meta.get("candidate_artifact_accuracy_stored"),
                        "holdout_sample_count": meta.get("holdout_sample_count"),
                        "holdout_low_confidence": meta.get("holdout_low_confidence"),
                        "candidate_always_buy": meta.get("candidate_always_buy"),
                        "candidate_always_hold": meta.get("candidate_always_hold"),
                        "promotion_rejection_reason": pe.get("last_event_reason") or meta.get("holdout_tie_explanation"),
                        "active_holdout_pac": meta.get("active_holdout_pac"),
                        "candidate_holdout_pac": meta.get("candidate_holdout_pac"),
                        "feature_version": meta.get("feature_version"),
                        "feature_dim": meta.get("feature_dim"),
                        "last_promotion_event": pe.get("last_event_at")
                        if pe.get("last_event_type") in ("promote", "promoted")
                        else None,
                        "last_rejection_event": pe.get("last_event_at")
                        if pe.get("last_event_type") in ("reject", "rejected")
                        else None,
                        # Model diversity / calibration provenance (see ai_blended_classifier.py,
                        # ai_training_pipeline.py, ai_feature_importance_diagnostics.py).
                        "blend_status": meta.get("active_blend_status"),
                        "rf_val_acc": meta.get("active_rf_val_acc"),
                        "gbm_val_acc": meta.get("active_gbm_val_acc"),
                        "blend_w_rf": meta.get("active_blend_w_rf"),
                        "blend_w_gbm": meta.get("active_blend_w_gbm"),
                        "confidence_calibrated": meta.get("active_confidence_calibrated"),
                        "feature_importance_weakest": meta.get("active_feature_importance_weakest"),
                    }
                )
        return {
            "success": True,
            "data": {
                "per_symbol": per_symbol,
                "model_freshness": freshness,
                "model_behavior_summary": behavior.get("summary") if isinstance(behavior, dict) else None,
            },
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
    except Exception as e:
        logger.exception("Error building model panel: %s", e)
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.get("/ai-signals-panel")
async def get_ai_signals_panel() -> dict[str, Any]:
    """Read-only per-symbol snapshot of the live DAY ranking/trust signals added
    this session — model disagreement, chart pattern, cross-sectional standing,
    setup/execution nudges, meta-labeling trust — sourced from each symbol's most
    recent ai_candidate_snapshots row. These are decision-cycle signals (recomputed
    every DAY signal loop), distinct from /model-panel's per-artifact training
    diagnostics. Advisory/diagnostic only — never gates a trade."""
    try:
        per_symbol: list[dict[str, Any]] = []
        with sqlite3.connect(DATABASE_PATH, timeout=5) as conn:
            conn.row_factory = sqlite3.Row
            for sym in TRADING_SYMBOLS:
                # ai_candidate_snapshots.symbol is stored in CCXT format ("BTC/USDT"),
                # while TRADING_SYMBOLS is bus format ("BTCUSDT") — same mismatch
                # handled by ai_market_context._to_ccxt elsewhere in the pipeline.
                ccxt_sym = sym if "/" in sym else (f"{sym[:-4]}/USDT" if sym.endswith("USDT") else sym)
                row = conn.execute(
                    """
                    SELECT symbol, ts_utc, decision, confidence, day_route_regime,
                           chart_pattern_label, chart_pattern_score, model_disagreement,
                           cross_sectional_rank_delta, setup_score, execution_rank_delta,
                           meta_trust_multiplier
                    FROM ai_candidate_snapshots
                    WHERE strategy_id = 'day' AND symbol IN (?, ?)
                    ORDER BY epoch_ms DESC
                    LIMIT 1
                    """,
                    (sym, ccxt_sym),
                ).fetchone()
                if row is None:
                    per_symbol.append({"symbol": sym, "available": False})
                    continue
                per_symbol.append(
                    {
                        "symbol": sym,
                        "available": True,
                        "as_of": row["ts_utc"],
                        "decision": row["decision"],
                        "confidence": row["confidence"],
                        "day_route_regime": row["day_route_regime"] or None,
                        "chart_pattern_label": row["chart_pattern_label"] or None,
                        "chart_pattern_score": row["chart_pattern_score"],
                        "model_disagreement": row["model_disagreement"],
                        "cross_sectional_rank_delta": row["cross_sectional_rank_delta"],
                        "setup_score": row["setup_score"],
                        "execution_rank_delta": row["execution_rank_delta"],
                        "meta_trust_multiplier": row["meta_trust_multiplier"],
                    }
                )
        return {
            "success": True,
            "data": {"per_symbol": per_symbol},
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
    except Exception as e:
        logger.exception("Error building AI signals panel: %s", e)
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.post("/operator-config")
async def set_operator_config_endpoint(request: Request, payload: dict[str, Any]) -> dict[str, Any]:
    """Dashboard: update limits (ADMIN_TOKEN required)."""
    try:
        from backend.middleware.security import AdminAuthMiddleware
        from backend.services.operator_config_service import set_operator_config

        if not AdminAuthMiddleware.verify_admin_token(request):
            raise HTTPException(
                status_code=401,
                detail="Unauthorized - ADMIN_TOKEN required to update operator config",
                headers={"WWW-Authenticate": "Bearer"},
            )

        data = await set_operator_config(payload)
        return {"success": True, "data": data}
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Error setting operator config: %s", e)
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.get("/operator-status")
async def get_operator_status() -> dict[str, Any]:
    """
    Single-glance operator status for dashboard.

    Returns compact JSON with:
    - mode: PAPER/LIVE
    - kill_switch: RESUME/PAUSE_BUYS/PAUSE_ALL
    - account_status: HEALTHY/OVERALLOCATED/etc
    - cash_balance, positions_value, total_equity
    - open_positions_count, open_risk_pct
    - last_trade with time_since

    Dashboard fix: cash_balance, total_equity, positions_value, open_positions_count,
    account_status, trading_paused come from SQLite (portfolio_engine_ledger +
    portfolio_engine_positions) when available. Integration writes there; API must
    read shared truth, not stale in-memory engine state.
    """
    try:
        engine = get_portfolio_engine()

        # Auto-initialize if needed
        from backend.services.portfolio_engine import is_portfolio_engine_initialized

        if not is_portfolio_engine_initialized():
            await initialize_portfolio_engine()

        status = await engine.get_operator_status()

        # Read canonical values from SQLite (integration writes here; avoids stale API engine state)
        sqlite_data = await _read_operator_status_from_sqlite()
        if sqlite_data:
            status.update(sqlite_data)

        try:
            from backend.services.day_position_health import load_health

            day_health = await asyncio.to_thread(load_health)
            if day_health:
                status["day_position_health"] = day_health
        except Exception:
            pass

        # PAPER mode: Use SQLite-only path for consistency (no Redis override)
        # LIVE mode: Could add live balance cross-check here if needed
        if status.get("mode") == "PAPER":
            # Remove binance_total_usdt if it exists (paper mode should not show live balance)
            status.pop("binance_total_usdt", None)
            # SQLite is the single source of truth for PAPER mode
            # Do NOT override cash_balance or total_equity from Redis
        elif status.get("mode") == "LIVE":
            # In LIVE mode, optionally cross-check with Binance if needed
            # For now, use SQLite as canonical source
            pass

        return {
            "success": True,
            "data": status,
        }
    except Exception as e:
        logger.exception(f"Error getting operator status: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.get("/daily-performance-snapshot")
async def get_daily_performance_snapshot() -> dict[str, Any]:
    """
    Daily performance snapshot: SELL distribution, exit types, hold-time percentiles, churn rate.
    Tracks whether the churn fix holds over time.
    """
    try:
        from backend.services.daily_performance_snapshot import compute_snapshot

        snapshot = compute_snapshot()
        return {"success": True, "data": snapshot}
    except Exception as e:
        logger.exception("Error getting daily performance snapshot: %s", e)
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.get("/day-health")
async def get_day_position_health() -> dict[str, Any]:
    """
    Trapped-position and idle-capital telemetry (observation only — no auto-sell/rotation).
    """
    try:
        from backend.services.day_position_health import load_health

        payload = await asyncio.to_thread(load_health)
        if payload is None:
            return {"success": True, "data": None, "note": "no_telemetry_yet"}
        return {"success": True, "data": payload}
    except Exception as e:
        logger.exception("Error getting day position health: %s", e)
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.get("/latency")
async def get_latency_metrics() -> dict[str, Any]:
    """
    Get latency and data freshness metrics.

    Returns:
    - market_data_age_seconds per symbol
    - decision_to_execution_ms (rolling avg/p95)
    - health flags: STALE_DATA, HIGH_LATENCY, OK
    """
    try:
        engine = get_portfolio_engine()
        latency = await engine.get_latency_metrics()

        return {
            "success": True,
            "data": latency,
        }
    except Exception as e:
        logger.exception(f"Error getting latency metrics: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.get("/invariants-detail")
async def get_invariants_detail() -> dict[str, Any]:
    """
    Get detailed invariants status with exact numeric values.
    Dashboard fix: Override position_limit and no_stacking with SQLite data.
    """
    try:
        engine = get_portfolio_engine()
        invariants = engine.get_invariants_status()
        sqlite_positions = await _read_positions_for_risk_from_sqlite()
        sqlite_count = len(sqlite_positions)
        sqlite_symbols = [p["symbol"] for p in sqlite_positions]

        # Override position data with SQLite (matches positions table)
        if "position_limit" in invariants:
            pl = invariants["position_limit"]
            mx = int(pl.get("max", 10) or 10)
            invariants["position_limit"] = {
                "ok": sqlite_count <= mx,
                "current": sqlite_count,
                "max": mx,
            }
        if "no_stacking" in invariants:
            invariants["no_stacking"] = {
                "ok": len(sqlite_symbols) == len(set(sqlite_symbols)),
                "symbols": sqlite_symbols,
            }

        return {
            "success": True,
            "data": invariants,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
    except Exception as e:
        logger.exception(f"Error getting invariants detail: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.get("/snapshot")
async def get_snapshot() -> dict[str, Any]:
    """
    Get full snapshot for reporting (ledger + scoreboard + last trades).
    """
    try:
        engine = get_portfolio_engine()
        snapshot = await engine.get_snapshot()

        return {
            "success": True,
            "data": snapshot,
        }
    except Exception as e:
        logger.exception(f"Error getting snapshot: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.get("/calibration")
async def get_calibration_metrics() -> dict[str, Any]:
    """
     Get profitability calibration metrics for 50-100 trade observation window.
    Read-only. Used by calibration_report.py script.
    """
    try:
        import os

        from backend.services.profitability_diagnostics import get_profitability_diagnostics

        diag = get_profitability_diagnostics()
        trades = list(diag._closed_trades)
        n = len(trades)
        if n < 10:
            return {
                "success": True,
                "data": {
                    "trade_count": n,
                    "recommendation": "HOLD",
                    "reason": "Insufficient trades. Collect 50+ then re-run calibration_report.py.",
                },
            }

        wins = [t for t in trades if t.net_pnl_pct > 0]
        losses = [t for t in trades if t.net_pnl_pct < 0]
        wr = len(wins) / n * 100 if n else 0
        avg_win = sum(t.net_pnl_pct for t in wins) / len(wins) if wins else 0.0
        avg_loss_mag = abs(sum(t.net_pnl_pct for t in losses) / len(losses)) if losses else 0.0
        exp = (len(wins) / n * avg_win) - (len(losses) / n * avg_loss_mag) if n else 0.0
        sum_w = sum(t.net_pnl_pct for t in wins)
        sum_l = abs(sum(t.net_pnl_pct for t in losses))
        pf = sum_w / sum_l if sum_l > 0 else (99.99 if sum_w > 0 else 1.0)
        miss = diag.get_missed_counts()
        total_opps = n + sum(miss.values())
        spread_pct = (miss.get("spread", 0) / total_opps * 100) if total_opps > 0 else 0
        low_vol_pct = (miss.get("low_vol", 0) / total_opps * 100) if total_opps > 0 else 0
        fees_avg = sum(t.fees_paid_pct for t in trades) / n
        slip_avg = sum(t.slippage_pct for t in trades) / n
        gross_profit = sum(t.gross_pnl_pct for t in trades if t.gross_pnl_pct > 0)
        cost_as_gross = (sum(t.fees_paid_pct + t.slippage_pct for t in trades) / gross_profit * 100) if gross_profit > 0 else 0
        dd = ((diag._equity_peak - diag._last_equity) / diag._equity_peak * 100) if diag._equity_peak > 0 else 0

        by_sym: dict[str, list] = {}
        for t in trades:
            by_sym.setdefault(t.symbol, []).append(t)
        sym_stats = []
        for s, ts in by_sym.items():
            sym_short = s.split("/")[0]
            ts_last = ts[-100:]
            sym_n = len(ts_last)
            sym_wins = [t for t in ts_last if t.net_pnl_pct > 0]
            sym_losses = [t for t in ts_last if t.net_pnl_pct < 0]
            sym_avg_loss = abs(sum(t.net_pnl_pct for t in sym_losses) / len(sym_losses)) if sym_losses else 0
            sym_exp = (
                ((len(sym_wins) / sym_n * sum(t.net_pnl_pct for t in sym_wins) / len(sym_wins) if sym_wins else 0) - (len(sym_losses) / sym_n * sym_avg_loss if sym_losses else 0)) if sym_n else 0
            )
            sym_stats.append((sym_short, sym_exp, sym_avg_loss))
        sym_stats.sort(key=lambda x: x[1], reverse=True)

        return {
            "success": True,
            "data": {
                "expectancy": round(exp, 3),
                "profit_factor": round(min(pf, 99.99), 2),
                "win_rate": round(wr, 1),
                "instant_stop_rate": round((diag._instant_stop_count / n * 100), 1),
                "spread_skips_pct": round(spread_pct, 1),
                "low_vol_skips_pct": round(low_vol_pct, 1),
                "cooldown_skips": miss.get("cooldown", 0),
                "confidence_override": "ON" if os.getenv("MIN_CONFIDENCE_OVERRIDE") else "OFF",
                "drawdown": round(dd, 1),
                "max_drawdown": round(diag._max_drawdown_pct, 1),
                "top_symbols": [s[0] for s in sym_stats[:3]],
                "worst_symbols": [s[0] for s in sym_stats[-3:]],
                "avg_cost_per_trade_pct": round(fees_avg + slip_avg, 3),
                "cost_as_pct_gross": round(cost_as_gross, 1),
                "trade_count": n,
            },
        }
    except Exception as e:
        logger.exception(f"Error getting calibration: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.get("/profit-system-diagnostics")
async def get_profit_system_diagnostics_endpoint() -> dict[str, Any]:
    """Unified diagnostics for ranking, exits, memory, sleeves and model state."""
    try:
        from backend.services.profit_system_diagnostics import get_profit_system_diagnostics

        return {"success": True, "data": get_profit_system_diagnostics()}
    except Exception as e:
        logger.exception(f"Error building profit-system diagnostics: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.get("/learning-status")
async def get_learning_status(limit: int = 20) -> dict[str, Any]:
    """Single read surface for the trade-learning sink.

    Exposes the unified ``trade_learning_outcomes`` table plus aggregate
    good/bad counts so the dashboard, the engine, and any future trainer
    all read the same numbers.

    No data is invented: if the table is empty the response reports zero
    rows and ``ai_has_enough_data=false`` so the dashboard can show "AI is
    still gathering data" without falling back to fake values.
    """
    try:
        import json as _json

        from backend.services.trade_learning_writer import (
            TABLE_NAME as LEARNING_TABLE,
        )
        from backend.services.trade_learning_writer import (
            read_recent_learning_rows,
        )
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"learning writer unavailable: {e}") from e

    safe_limit = max(1, min(int(limit or 20), 200))
    try:
        rows = read_recent_learning_rows(limit=safe_limit)
    except Exception as e:
        logger.exception("read_recent_learning_rows failed: %s", e)
        rows = []

    good_count = 0
    bad_count = 0
    last_good: dict[str, Any] | None = None
    last_bad: dict[str, Any] | None = None
    sanitized: list[dict[str, Any]] = []
    for row in rows:
        extra_raw = row.get("extra_json") or "{}"
        try:
            extra = _json.loads(extra_raw) if isinstance(extra_raw, str) else dict(extra_raw)
        except Exception:
            extra = {}
        is_good = bool(extra.get("good_trade"))
        is_bad = bool(extra.get("bad_trade"))
        if is_good:
            good_count += 1
            last_good = last_good or {"symbol": row.get("symbol"), "exit_timestamp": row.get("exit_timestamp"), "net_profit_usd": row.get("net_profit_usd")}
        if is_bad:
            bad_count += 1
            last_bad = last_bad or {"symbol": row.get("symbol"), "exit_timestamp": row.get("exit_timestamp"), "close_reason": row.get("close_reason")}
        sanitized.append(
            {
                "id": row.get("id"),
                "symbol": row.get("symbol"),
                "mode": row.get("mode"),
                "close_reason": row.get("close_reason"),
                "entry_timestamp": row.get("entry_timestamp"),
                "exit_timestamp": row.get("exit_timestamp"),
                "exit_price": row.get("exit_price"),
                "net_profit_usd": row.get("net_profit_usd"),
                "net_profit_pct": row.get("net_profit_pct"),
                "manual_sell_flag": bool(row.get("manual_sell_flag")),
                "dust_remaining_qty": row.get("dust_remaining_qty"),
                "realized_profit_unknown": bool(row.get("realized_profit_unknown")),
                "good_trade": is_good,
                "bad_trade": is_bad,
                "lesson": extra.get("lesson"),
                "source": extra.get("source"),
            }
        )

    total_rows = 0
    last_written: str | None = None
    storage_connected = False
    try:
        with sqlite3.connect(DATABASE_PATH, timeout=2) as conn:
            cur = conn.execute(f"SELECT COUNT(*) FROM {LEARNING_TABLE}")
            total_rows = int(cur.fetchone()[0] or 0)
            cur = conn.execute(f"SELECT written_at_utc FROM {LEARNING_TABLE} ORDER BY id DESC LIMIT 1")
            row = cur.fetchone()
            last_written = row[0] if row else None
            storage_connected = True
    except Exception as e:
        logger.warning("learning_status: storage probe failed: %s", e)

    return {
        "success": True,
        "data": {
            "storage_connected": storage_connected,
            "table": LEARNING_TABLE,
            "total_rows": total_rows,
            "last_written_at_utc": last_written,
            "good_trade_count_recent": good_count,
            "bad_trade_count_recent": bad_count,
            "last_good_recent": last_good,
            "last_bad_recent": last_bad,
            "ai_has_enough_data": total_rows >= 5,
            "rows": sanitized,
        },
    }
