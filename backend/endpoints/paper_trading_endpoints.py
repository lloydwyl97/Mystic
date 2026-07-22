"""
Paper Trading Endpoints
FastAPI endpoints for paper trading functionality

MEDIUM #4 FIX: Confidence threshold now centralized via MIN_CONFIDENCE env var
Used consistently in both paper endpoint and portfolio engine
"""

import asyncio
import json
import logging
import os
import sqlite3
import time
from datetime import datetime, timezone
from typing import Any

from fastapi import APIRouter, Depends, HTTPException

from backend.services.admin_auth import require_admin_key
from pydantic import BaseModel

from backend.config.redis_config import get_redis_client, get_shared_redis_async
from backend.database_schema import DATABASE_PATH
from backend.services.paper_trading_service import get_paper_trading_service

_sleeve_cutover_epoch_cache: int | None = None


def _get_sleeve_cutover_epoch() -> int | None:
    global _sleeve_cutover_epoch_cache
    if _sleeve_cutover_epoch_cache is not None:
        return _sleeve_cutover_epoch_cache
    try:
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
    if not ts:
        return False
    try:
        dt = datetime.fromisoformat(str(ts).replace("Z", "+00:00"))
        return dt.timestamp() >= cutover_epoch
    except Exception:
        return False


# Optional imports - services may not be available
try:
    from backend.services.portfolio_engine import (
        get_portfolio_engine,
        is_portfolio_engine_initialized,
    )
except (ImportError, ModuleNotFoundError, AttributeError, ValueError, TypeError, RuntimeError):
    get_portfolio_engine = None  # type: ignore[assignment]

    def is_portfolio_engine_initialized() -> bool:
        return False


try:
    from backend.utils.symbols import to_exchange_symbol
except (ImportError, ModuleNotFoundError, AttributeError, ValueError, TypeError, RuntimeError):

    def to_exchange_symbol(s: str) -> str:
        return s


logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/paper-trading", tags=["Paper Trading"])


def _validate_data_integrity_sync(positions_value: float, principal: float, realized_pnl: float, cash_balance: float) -> tuple[bool, float]:
    """BUG #6 FIX: Sync DB helper for data integrity check. Runs in thread."""
    conn = None
    try:
        conn = sqlite3.connect(DATABASE_PATH)
        cursor = conn.cursor()

        # Get sum of all realized P&L across all runs (only fully closed positions)
        # This matches the cumulative realized_pnl_total in memory
        cursor.execute("""
            SELECT COALESCE(SUM(pnl), 0)
            FROM paper_trades
            WHERE UPPER(side) = 'SELL' AND remaining_position = 0 AND pnl IS NOT NULL
        """)

        sqlite_realized_pnl = float(cursor.fetchone()[0] or 0.0)
        ledger_realized = float(realized_pnl or 0.0)
        books_gap = ledger_realized - sqlite_realized_pnl

        # Use ledger realized (never-decrease / prune-aware), not raw paper Σpnl.
        # Paper prune can leave sqlite_sum << ledger realized without a cash bug.
        # Flat book: cash ≈ principal + realized.
        # Open book: cash ≈ principal + realized − open cost basis.
        if float(positions_value or 0.0) == 0.0:
            expected_cash = float(principal or 0.0) + ledger_realized
            data_integrity_diff = float(cash_balance or 0.0) - expected_cash
            data_integrity_ok = abs(data_integrity_diff) < 0.05
        else:
            cursor.execute(
                """
                SELECT COALESCE(SUM(
                    quantity * entry_price + COALESCE(entry_fee, 0)
                ), 0)
                FROM portfolio_engine_positions
                WHERE quantity > 0
                """
            )
            open_cost = float(cursor.fetchone()[0] or 0.0)
            cursor.execute(
                """
                SELECT COALESCE(total_equity, 0), COALESCE(unrealized_pnl, 0)
                FROM portfolio_engine_ledger WHERE id = 1
                """
            )
            eq_row = cursor.fetchone() or (0.0, 0.0)
            ledger_equity = float(eq_row[0] or 0.0)
            # Ghost MTM (value with no persisted cost) is a real integrity fail.
            if open_cost <= 0 and float(positions_value or 0.0) > 1.0:
                data_integrity_diff = float(positions_value or 0.0)
                data_integrity_ok = False
            else:
                # Open-book identity that must hold: cash + MTM positions = ledger equity.
                composed = float(cash_balance or 0.0) + float(positions_value or 0.0)
                data_integrity_diff = composed - ledger_equity
                data_integrity_ok = abs(data_integrity_diff) < 0.05

        if abs(books_gap) > 1.0:
            logger.info(
                "LEDGER_BOOKS_GAP ledger_realized=%.4f paper_sum=%.4f gap=%.4f (prune/heal expected)",
                ledger_realized,
                sqlite_realized_pnl,
                books_gap,
            )
        if not data_integrity_ok:
            logger.warning(
                "Data integrity check failed: cash=%s principal=%s ledger_realized=%s expected_cash=%s diff=%s",
                cash_balance,
                principal,
                ledger_realized,
                float(principal or 0.0) + ledger_realized,
                data_integrity_diff,
            )

        return data_integrity_ok, data_integrity_diff

    except Exception as e:
        logger.exception(f"Data integrity validation failed: {e}")
        return False, -999.0  # Error indicator
    finally:
        if conn:
            conn.close()


async def _validate_data_integrity_async(positions_value: float, principal: float, realized_pnl: float, cash_balance: float) -> tuple[bool, float]:
    """BUG #6 FIX: Async wrapper that offloads DB work to thread."""
    return await asyncio.to_thread(_validate_data_integrity_sync, positions_value, principal, realized_pnl, cash_balance)


def _get_initial_balance_from_ledger_sync() -> float:
    """Read principal from portfolio_engine_ledger (live data). No hardcoded default. SYNC version."""
    conn = None
    try:
        conn = sqlite3.connect(DATABASE_PATH)
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='portfolio_engine_ledger'")
        if not cursor.fetchone():
            return 0.0
        cursor.execute("SELECT principal FROM portfolio_engine_ledger WHERE id=1")
        row = cursor.fetchone()
        if row and row[0] is not None:
            return float(row[0])
    except Exception as e:
        logger.debug("Ledger principal read failed: %s", e)
    finally:
        if conn:
            conn.close()
    return 0.0


async def _get_initial_balance_from_ledger() -> float:
    """BUG #7 FIX: Read principal from ledger using offloaded thread for async contexts."""
    # If called from async context, offload to thread to keep event loop responsive
    return await asyncio.to_thread(_get_initial_balance_from_ledger_sync)


async def _positions_from_paper_trades() -> list[dict[str, Any]]:
    """
    Fallback: derive open positions from paper_trades when portfolio_engine_positions is empty.
    Uses BUY - SELL net quantity per symbol. Needed when BUYs exist in paper_trades but
    portfolio_engine_positions was cleared (e.g. cleanup) or never synced.
    """

    def _sync_load():
        conn = sqlite3.connect(DATABASE_PATH)
        try:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT symbol,
                       SUM(CASE WHEN side = 'BUY' THEN quantity ELSE 0 END) -
                       SUM(CASE WHEN side = 'SELL' THEN quantity ELSE 0 END) AS net_qty,
                       SUM(CASE WHEN side = 'BUY' THEN quantity * price ELSE 0 END) /
                       NULLIF(SUM(CASE WHEN side = 'BUY' THEN quantity ELSE 0 END), 0) AS avg_entry
                FROM paper_trades
                GROUP BY symbol
                HAVING net_qty > 0
            """)
            rows = cursor.fetchall()
            return rows
        finally:
            conn.close()

    loop = asyncio.get_running_loop()
    rows = await loop.run_in_executor(None, _sync_load)
    if not rows:
        return []

    redis_client = get_shared_redis_async()
    out: list[dict[str, Any]] = []
    for symbol, net_qty, avg_entry in rows:
        norm = to_exchange_symbol(symbol) if symbol else symbol or ""
        base = (norm or str(symbol or "")).replace("USDT", "").replace("/", "").strip()
        cache_key = f"market:{base}"
        current_price = float(avg_entry or 0)
        try:
            if redis_client:
                cached = await redis_client.get(cache_key)
                if cached:
                    data = json.loads(cached) if isinstance(cached, str) else cached
                    current_price = float(data["price"]) if isinstance(data, dict) and "price" in data else float(cached)
                    if current_price <= 0:
                        current_price = float(avg_entry or 0)
        except Exception as ex:
            logger.debug("Could not fetch cached price: %s", ex)
        entry_price = float(avg_entry or 0)
        unrealized_pnl = (current_price - entry_price) * net_qty
        # Use original symbol (ccxt BASE/QUOTE form) for display; norm used only for cache key
        display_symbol = str(symbol or "") if symbol else (norm or "")
        out.append(
            {
                "symbol": display_symbol,
                "quantity": float(net_qty),
                "average_price": entry_price,
                "current_price": current_price,
                "unrealized_pnl": unrealized_pnl,
                "realized_pnl": 0.0,
                "total_pnl": unrealized_pnl,
                "market_value": net_qty * current_price,
                "created_at": datetime.now(timezone.utc).isoformat(),
                "last_updated": datetime.now(timezone.utc).isoformat(),
            }
        )
    return out


async def _positions_from_sqlite() -> list[dict[str, Any]]:
    """
    Return open positions from authoritative SQLite (portfolio_engine_positions).
    Dashboard/API may run in a different process from integration—SQLite is the shared source of truth.
    """

    def _sync_load():
        conn = sqlite3.connect(DATABASE_PATH)
        try:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT symbol, quantity, entry_price, entry_time, last_updated,
                       COALESCE(sleeve, 'ACTIVE')
                FROM portfolio_engine_positions
                WHERE quantity > 0
                ORDER BY symbol
            """)
            return cursor.fetchall()
        finally:
            conn.close()

    loop = asyncio.get_running_loop()
    rows = await loop.run_in_executor(None, _sync_load)
    if not rows:
        return []

    redis_client = get_shared_redis_async()
    out: list[dict[str, Any]] = []
    for row in rows:
        symbol, quantity, entry_price, entry_time, last_updated = row[0], row[1], row[2], row[3], row[4]
        sleeve = row[5] if len(row) > 5 else "ACTIVE"
        base_symbol = (symbol or "").replace("/USDT", "").replace("USDT", "").strip()
        cache_key = f"market:{base_symbol}"
        current_price = float(entry_price or 0)
        try:
            if redis_client:
                cached = await redis_client.get(cache_key)
                if cached:
                    data = json.loads(cached) if isinstance(cached, str) else cached
                    current_price = float(data["price"]) if isinstance(data, dict) and "price" in data else float(cached)
                    if current_price <= 0:
                        current_price = float(entry_price or 0)
        except Exception as e:
            logger.debug("Redis price for %s: %s", symbol, e)
        entry_price_f = float(entry_price or 0)
        unrealized_pnl = (current_price - entry_price_f) * float(quantity or 0)
        out.append(
            {
                "symbol": symbol,
                "quantity": float(quantity or 0),
                "average_price": entry_price_f,
                "current_price": current_price,
                "unrealized_pnl": unrealized_pnl,
                "realized_pnl": 0.0,
                "total_pnl": unrealized_pnl,
                "market_value": float(quantity or 0) * current_price,
                "created_at": datetime.fromtimestamp(float(entry_time or 0), tz=timezone.utc).isoformat() if entry_time else datetime.now(timezone.utc).isoformat(),
                "last_updated": last_updated or datetime.now(timezone.utc).isoformat(),
                "sleeve": sleeve,
            }
        )
    return out


async def _positions_from_portfolio_engine() -> list[dict[str, Any]] | None:
    """Return open positions from portfolio engine in-memory. Used as fallback when SQLite unavailable."""
    if get_portfolio_engine is None:
        return None
    try:
        engine = get_portfolio_engine()
        # LIVE mode: refresh from SQLite (integration may run in external process)
        if getattr(engine, "_live_execution_enabled", False) and hasattr(engine, "_load_positions_from_sqlite"):
            await engine._load_positions_from_sqlite()
        if not engine.open_positions:
            return []
        out: list[dict[str, Any]] = []
        redis_client = get_shared_redis_async()
        for symbol, pos in engine.open_positions.items():
            base_symbol = to_exchange_symbol(symbol).replace("USDT", "")
            cache_key = f"market:{base_symbol}"
            current_price = pos.entry_price
            try:
                cached = await redis_client.get(cache_key)
                if cached:
                    data = json.loads(cached) if isinstance(cached, str) else cached
                    current_price = float(data["price"]) if isinstance(data, dict) and "price" in data else float(cached)
                    if current_price <= 0:
                        current_price = pos.entry_price
            except Exception as e:
                logger.debug("Redis price for %s: %s", symbol, e)
            unrealized_pnl = (current_price - pos.entry_price) * pos.quantity
            entry_dt = datetime.fromtimestamp(pos.entry_time, tz=timezone.utc) if pos.entry_time else datetime.now(timezone.utc)
            out.append(
                {
                    "symbol": symbol,
                    "quantity": pos.quantity,
                    "average_price": pos.entry_price,
                    "current_price": current_price,
                    "unrealized_pnl": unrealized_pnl,
                    "realized_pnl": 0.0,
                    "total_pnl": unrealized_pnl,
                    "market_value": pos.quantity * current_price,
                    "created_at": entry_dt.isoformat(),
                    "last_updated": datetime.now(timezone.utc).isoformat(),
                    "sleeve": getattr(pos, "sleeve", "ACTIVE"),
                }
            )
    except Exception as e:
        logger.debug("Positions from portfolio engine failed: %s", e)
        return None
    else:
        return out


@router.get("/status")
async def get_paper_trading_status() -> dict[str, Any]:
    """Get paper trading status - production data only"""
    try:
        import time

        paper_service = get_paper_trading_service()
        enabled = paper_service.is_enabled()
        running = paper_service._running  # Check if service loop is actually running

        # ================================================================
        # PHASE 4 FIX #5: CHECK CACHE STALENESS AND RETURN FLAG
        # ================================================================
        # Get balance with staleness detection
        # If cache is older than 5 seconds, consider it stale
        # Return stale flag so client knows to refresh
        balance = await paper_service.get_account_balance(use_cache_only=True)  # Dashboard: no exchange weight

        # Add cache age information
        current_time = time.time()
        cache_age_seconds = 0  # Estimate from balance timestamp if available
        balance_timestamp = balance.get("timestamp")
        if balance_timestamp and isinstance(balance_timestamp, str):
            try:
                from datetime import datetime

                ts = datetime.fromisoformat(balance_timestamp)
                cache_age_seconds = current_time - ts.timestamp()
            except Exception as ex:
                logger.debug("balance timestamp parse failed: %s", ex)
                cache_age_seconds = 0

        # Add staleness flag
        is_cache_stale = cache_age_seconds > 5.0  # 5 second threshold
        if "status" not in balance:
            balance["status"] = "success"
        balance["cache_age_seconds"] = cache_age_seconds
        balance["is_cache_stale"] = is_cache_stale
        if is_cache_stale:
            balance["_warning"] = f"Balance cache is {cache_age_seconds:.1f}s old"

        # Signal-to-execution flow is owned exclusively by
        # ``backend.services.portfolio_engine`` and its integration layer.

        # IMPROVED DATA INTEGRITY VALIDATION
        # Now that run_id continuity is fixed, we can do proper reconciliation
        positions_value = balance.get("positions_value", 0.0)
        principal = balance.get("principal", 0.0)
        realized_pnl = balance.get("realized_pnl", 0.0)
        cash_balance = balance.get("cash_balance", 0.0)

        try:
            # BUG #6 FIX: Use async offload for DB work
            data_integrity_ok, data_integrity_diff = await _validate_data_integrity_async(positions_value, principal, realized_pnl, cash_balance)

        except Exception as e:
            logger.exception(f"Data integrity validation failed: {e}")
            data_integrity_ok = False
            data_integrity_diff = -999.0  # Error indicator

        return {
            "status": "success",
            "enabled": enabled,
            "running": running,
            "balance": balance.get("total_balance_usd", balance.get("total_value", 0)),
            "details": balance,
            "paper_run_id": paper_service.paper_run_id,
            "data_integrity_ok": data_integrity_ok,
            "data_integrity_diff": data_integrity_diff,
            "data_source": "sqlite_canonical",
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
    except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
        logger.exception(f"Failed to get paper trading status: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.post("/process-signals")
async def process_signals() -> dict[str, Any]:
    """Retired: Redis buy bridge removed. DAY buys run only via portfolio_engine_integration bar path."""
    raise HTTPException(
        status_code=410,
        detail=("POST /api/paper-trading/process-signals is retired. DAY buys execute only through start_portfolio_engine_integration → process_bar_candidates → execute_buy_fifo."),
    )


@router.get("/portfolio")
async def get_paper_trading_portfolio() -> dict[str, Any]:
    """Get paper trading portfolio data from paper trading service"""
    try:
        paper_service = get_paper_trading_service()
        portfolio_data = await paper_service.get_account_balance(use_cache_only=True)  # Dashboard: no exchange weight

        if portfolio_data.get("error"):
            raise HTTPException(status_code=500, detail=portfolio_data["error"])

        # FIX: Read cash directly from Redis for accurate display
        try:
            redis_client = get_shared_redis_async()
            redis_cash = await redis_client.get("paper_trading:cash_balance")
            if redis_cash:
                paper_cash = float(redis_cash)
                positions_value = portfolio_data.get("positions_value", 0)
                # Override with correct cash from Redis
                portfolio_data["cash_balance"] = paper_cash
                portfolio_data["available_balance"] = paper_cash
                portfolio_data["total_balance"] = paper_cash
                portfolio_data["total_value"] = paper_cash + positions_value
                portfolio_data["total_balance_usd"] = paper_cash + positions_value
        except Exception as e:
            logger.warning(f"Failed to read cash from Redis: {e}")

        return {
            "status": "success",
            "portfolio": portfolio_data,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "source": "paper_trading",
        }
    except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
        logger.exception(f"Error getting paper trading portfolio: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.get("/balance")
async def get_paper_trading_balance() -> dict[str, Any]:
    """Get paper trading account balance"""
    try:
        paper_service = get_paper_trading_service()
        balance_data = await paper_service.get_account_balance()

        if balance_data.get("error"):
            raise HTTPException(status_code=500, detail=balance_data["error"])

        return {
            "status": "success",
            "balance": balance_data,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "source": "paper_trading",
        }
    except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
        logger.exception(f"Error getting paper trading balance: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.get("/positions")
async def get_paper_trading_positions() -> dict[str, Any]:
    """Get open positions from canonical source (SQLite portfolio_engine_positions). Fallback to paper_trades, then paper_trading Redis."""
    try:
        # 1) Prefer SQLite portfolio_engine_positions (authoritative; shared by API + integration processes)
        positions_data = await _positions_from_sqlite()
        source = "portfolio_engine_positions"
        # 2) If empty, try paper_trades (BUY - SELL net) - handles orphaned BUYs
        if len(positions_data) == 0:
            fallback = await _positions_from_paper_trades()
            if fallback:
                positions_data = fallback
                source = "paper_trades"
        # 3) If still empty, try engine in-memory then Redis
        if len(positions_data) == 0:
            engine_pos = await _positions_from_portfolio_engine()
            if engine_pos:
                positions_data = engine_pos
                source = "portfolio_engine"
        if len(positions_data) == 0:
            paper_service = get_paper_trading_service()
            try:
                positions_data = await paper_service.get_positions()
            except AttributeError:
                positions_data = []
            if positions_data:
                source = "paper_trading"

        return {
            "status": "success",
            "positions": positions_data,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "source": source,
        }
    except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
        logger.exception("Error getting paper trading positions: %s", e)
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.get("/performance")
async def get_paper_trading_performance(include_history: bool = True) -> dict[str, Any]:
    """Get trading performance metrics. Default: full history (all runs, paper+live). Set include_history=false for current run only."""
    try:
        paper_service = get_paper_trading_service()
        trade_history = await paper_service.get_trade_history(limit=None, include_history=include_history)

        # Get portfolio for unrealized P&L
        portfolio_response = await get_paper_trading_portfolio()
        portfolio = portfolio_response.get("portfolio", {})
        unrealized_pnl = portfolio.get("unrealized_pnl", 0.0)

        # Initial balance from live ledger (no hardcoded value)
        initial_balance = await _get_initial_balance_from_ledger()  # BUG #7 FIX: Now awaits async offload
        current_balance = portfolio.get("total_balance_usd", 0.0) or 0.0
        if initial_balance == 0 and current_balance:
            initial_balance = current_balance  # No history: treat current as baseline
        total_return = ((current_balance - initial_balance) / initial_balance * 100) if initial_balance > 0 else 0.0

        # Calculate REALIZED performance (completed trade cycles only)
        total_realized_trades = len(trade_history)
        winning_trades = 0
        losing_trades = 0
        realized_pnl = 0.0

        for trade in trade_history:
            pnl = float(trade.get("pnl", 0.0) or 0.0)
            realized_pnl += pnl
            if pnl > 0:
                winning_trades += 1
            elif pnl < 0:
                losing_trades += 1

        realized_win_rate = (winning_trades / total_realized_trades * 100) if total_realized_trades > 0 else 0.0
        avg_trade_pnl = (realized_pnl / total_realized_trades) if total_realized_trades > 0 else 0.0

        # Calculate TOTAL performance (realized + unrealized)
        total_pnl = realized_pnl + unrealized_pnl

        return {
            "status": "success",
            "performance": {
                "total_trades": total_realized_trades,
                "winning_trades": winning_trades,
                "losing_trades": losing_trades,
                "win_rate": realized_win_rate,
                "total_pnl": realized_pnl,
                "avg_trade_pnl": avg_trade_pnl,
                # NEW: Show unrealized and total performance
                "unrealized_pnl": unrealized_pnl,
                "total_pnl_including_unrealized": total_pnl,
                "total_return_pct": total_return,
                "initial_balance": initial_balance,
                "current_balance": current_balance,
            },
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "source": "paper_trading",
        }
    except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
        logger.exception(f"Error getting paper trading performance: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.get("/orders")
async def get_paper_trading_orders() -> dict[str, Any]:
    """Get paper trading orders"""
    try:
        paper_service = get_paper_trading_service()
        orders_data = await paper_service.get_orders()

        return {
            "status": "success",
            "orders": orders_data,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "source": "paper_trading",
        }
    except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
        logger.exception(f"Error getting paper trading orders: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e


class PlaceOrderRequest(BaseModel):
    symbol: str
    side: str
    order_type: str
    quantity: float
    price: float | None = None
    stop_price: float | None = None
    take_profit_price: float | None = None
    # Optional attribution / admission (same pipeline as signals; omitted when unknown)
    decision_id: str | None = None
    sleeve: str | None = None
    spread_pct: float | None = None
    # Optional explicit confidence for legacy paper order model (unused for BUY HTTP path).
    confidence: float | None = None
    # Local harness only: set proof_trade on the request to skip G3/G4 for lifecycle proof
    proof_trade: bool = False
    # Optional rank attribution payload for controlled paper proofs / manual replay.
    rank_snapshot_id: int | None = None
    selected_rank: int | None = None
    selected_score: float | None = None
    selected_net_expected_value: float | None = None
    peer_ranks_json: dict[str, Any] | list[Any] | str | None = None
    score_components_json: dict[str, Any] | str | None = None
    live_ai_strategy: str | None = None
    model_artifact_path: str | None = None
    artifact_sha256: str | None = None
    feature_version: int | None = None
    feature_dim: int | None = None
    # Optional sell reason for controlled proofs (defaults to MANUAL).
    exit_reason: str | None = None


@router.post("/orders")
async def place_paper_order(request: PlaceOrderRequest) -> dict[str, Any]:
    """Retired: no HTTP order placement. DAY trades run only through portfolio_engine integration."""
    side_upper = str(request.side).strip().upper()
    raise HTTPException(
        status_code=410,
        detail=(
            f"POST /api/paper-trading/orders ({side_upper}) is retired. "
            "Mystic does not accept dashboard or HTTP buy/sell orders. "
            "BUY: bar-ranked execute_buy_fifo. SELL: exit monitor execute_sell_fifo."
        ),
    )


@router.delete("/orders/{order_id}")
async def cancel_paper_order(order_id: str) -> dict[str, Any]:
    """Cancel a paper trading order"""
    try:
        paper_service = get_paper_trading_service()
        result = await paper_service.cancel_order(order_id)

        return {
            "status": "success" if (isinstance(result, dict) and result.get("success")) else "error",
            "result": result,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
    except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
        logger.exception(f"Error canceling paper order: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.get("/trades")
async def get_paper_trading_trades(limit: int | None = None, run_id: str | None = None, since: str | None = None, include_history: bool = True) -> dict[str, Any]:
    """Get paper trading trade history from SQLite canonical storage. Default include_history=True so dashboard sees full canonical set."""
    try:
        paper_service = get_paper_trading_service()
        effective_include_history = include_history or (since is not None)
        trades_data = await paper_service.get_trade_history(limit=limit, run_id=run_id, since=since, include_history=effective_include_history)

        cutover_epoch = _get_sleeve_cutover_epoch()
        if cutover_epoch:
            for t in trades_data:
                t["sleeve_era"] = "post" if _ts_after_cutover(t.get("timestamp"), cutover_epoch) else "legacy"

        return {
            "status": "success",
            "trades": trades_data,
            "run_id": run_id or paper_service.paper_run_id,
            "include_history": effective_include_history,
            "total_trades_returned": len(trades_data),
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "source": "sqlite_canonical",
            "sleeve_cutover_epoch": cutover_epoch,
        }
    except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
        logger.exception(f"Error getting paper trading trades: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.post("/reset")
async def reset_paper_account(_: None = Depends(require_admin_key)) -> dict[str, Any]:
    """Reset paper trading account"""
    try:
        paper_service = get_paper_trading_service()
        result = await paper_service.reset_account()

        return {
            "status": "success" if (isinstance(result, dict) and result.get("success")) else "error",
            "result": result,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
    except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
        logger.exception(f"Error resetting paper account: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.post("/enable")
async def enable_paper_trading(_: None = Depends(require_admin_key)) -> dict[str, Any]:
    """Enable paper trading"""
    try:
        paper_service = get_paper_trading_service()
        result = await paper_service.enable_paper_trading()

        return {
            "status": "success" if (isinstance(result, dict) and result.get("success")) else "error",
            "result": result,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
    except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
        logger.exception(f"Error enabling paper trading: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.post("/disable")
async def disable_paper_trading(_: None = Depends(require_admin_key)) -> dict[str, Any]:
    """Disable paper trading"""
    try:
        paper_service = get_paper_trading_service()
        result = await paper_service.disable_paper_trading()

        return {
            "status": "success" if (isinstance(result, dict) and result.get("success")) else "error",
            "result": result,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
    except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
        logger.exception(f"Error disabling paper trading: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.get("/analytics")
async def get_paper_trading_analytics() -> dict[str, Any]:
    """Get paper trading performance analytics (alias for general analytics)"""
    try:
        paper_service = get_paper_trading_service()

        # Calculate stats from trade history (same logic as _update_stats_to_redis)
        trade_history = paper_service.trade_history
        total_trades = len(trade_history)
        winning_trades = sum(1 for t in trade_history if t.get("pnl", 0) > 0)
        losing_trades = sum(1 for t in trade_history if t.get("pnl", 0) < 0)
        total_pnl = sum(t.get("pnl", 0) for t in trade_history)
        win_rate = (winning_trades / total_trades) if total_trades > 0 else 0.0

        # Calculate advanced metrics
        sharpe_ratio = paper_service._calculate_sharpe_ratio()
        max_drawdown_pct, max_drawdown_usd = paper_service._calculate_max_drawdown()

        stats = {
            "total_trades": total_trades,
            "winning_trades": winning_trades,
            "losing_trades": losing_trades,
            "total_pnl": total_pnl,
            "win_rate": win_rate,
            "sharpe_ratio": sharpe_ratio,
            "max_drawdown_pct": max_drawdown_pct,
            "max_drawdown_usd": max_drawdown_usd,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

        return {
            "status": "success",
            "stats": stats,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "source": "paper_trading",
        }
    except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
        logger.exception(f"Error getting paper trading analytics: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.get("/analytics/by-symbol")
async def get_performance_by_symbol() -> dict[str, Any]:
    """Get performance metrics broken down by symbol (live data only)"""
    try:
        paper_service = get_paper_trading_service()
        symbol_performance = paper_service.get_performance_by_symbol()

        # Convert dict to list and sort by total_pnl
        symbols_list = list(symbol_performance.values())
        symbols_list.sort(key=lambda x: x["total_pnl"], reverse=True)

        return {
            "status": "success",
            "symbols": symbols_list,
            "total_symbols": len(symbols_list),
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "source": "paper_trading",
        }
    except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
        logger.exception(f"Error getting per-symbol performance: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e


# NOTE: ``/paper-trading/feedback/stats`` endpoint was retired alongside
# ``paper_trading_feedback_service``. Use ``/api/portfolio-engine/learning-status``
# for the current unified learning sink (paper + live).
