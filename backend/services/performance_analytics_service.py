"""
Performance Analytics Service
Provides comprehensive performance tracking and analytics

Quick Test Checklist:
- No pandas, no numpy; pure Python computations.
- Symbols normalized to BASE/QUOTE via _to_ccxt_symbol; accepts BTCUSDT input but converts to BTC/USDT.
- No binance/binanceus string leaks; exchange id is centralized elsewhere.
- Parameterized ASCII-only logging; no unreachable code.
- No Streamlit, no Docker, no Coinbase, no CoinGecko, no Kraken, no yfinance.
- Python 3.12, backend port 8000, unified dashboard port 8000.
"""

from __future__ import annotations

import asyncio
import logging
import sqlite3
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from math import sqrt
from typing import Any

import redis.asyncio as redis  # type: ignore[import-not-found]

from backend.config.redis_config import get_shared_redis_async
from backend.database_schema import DATABASE_PATH
from backend.services.symbols import _to_ccxt_symbol  # type: ignore[import-not-found]

logger = logging.getLogger(__name__)


@dataclass
class PerformanceMetrics:
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    profit_factor: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float
    max_drawdown_duration: int
    avg_win: float
    avg_loss: float
    largest_win: float
    largest_loss: float
    total_pnl: float
    total_return: float
    annualized_return: float
    volatility: float
    calmar_ratio: float
    recovery_factor: float
    expectancy: float
    kelly_percentage: float
    timestamp: datetime


@dataclass
class TradeRecord:
    trade_id: str
    symbol: str
    side: str
    quantity: float
    entry_price: float
    exit_price: float
    entry_time: datetime
    exit_time: datetime
    duration_minutes: int
    pnl: float
    pnl_percentage: float
    commission: float
    net_pnl: float
    strategy: str
    notes: str = ""


class PerformanceAnalyticsService:
    def __init__(self) -> None:
        self.redis_client: redis.Redis | None = None  # type: ignore[assignment]
        self.trades: list[TradeRecord] = []
        self.performance_history: list[PerformanceMetrics] = []
        self.risk_free_rate = 0.02  # 2% annual
        self.lookback_periods = [1, 7, 30, 90, 365]  # days
        self._trades_loaded_from_db = False
        logger.info("PerformanceAnalyticsService initialized")

    def _ensure_trades_loaded(self) -> None:
        """Load completed trades from paper_trades SQLite. Always re-reads to stay fresh."""
        self.trades.clear()
        try:
            conn = sqlite3.connect(DATABASE_PATH)
            conn.row_factory = sqlite3.Row
            try:
                sells = conn.execute("""
                    SELECT trade_id, symbol, quantity, price, entry_price,
                           pnl, pnl_pct, hold_time_seconds, fees_paid,
                           slippage_cost, exit_type, exit_r_multiple,
                           timestamp, entry_timestamp, strategy
                    FROM paper_trades
                    WHERE side = 'SELL' AND pnl IS NOT NULL
                      AND COALESCE(exit_type, '') NOT IN ('ADMIN_POSITION_CLEAR', 'STALE_PRE_CORRECTION_POSITION_CLEAR')
                    ORDER BY timestamp ASC
                """).fetchall()
                loaded = 0
                for row in sells:
                    try:
                        exit_time = datetime.fromisoformat(row["timestamp"])
                        entry_ts = row["entry_timestamp"]
                        if entry_ts:
                            entry_time = datetime.fromisoformat(entry_ts)
                        else:
                            hold_sec = row["hold_time_seconds"] or 0
                            entry_time = exit_time - timedelta(seconds=hold_sec)
                        duration_min = int((row["hold_time_seconds"] or 0) / 60)
                        commission = float(row["fees_paid"] or 0) + float(row["slippage_cost"] or 0)
                        pnl = float(row["pnl"] or 0)
                        net_pnl = pnl - commission
                        entry_price = float(row["entry_price"] or row["price"] or 0)
                        exit_price = float(row["price"] or 0)
                        pnl_pct = float(row["pnl_pct"] or 0)
                        trade = TradeRecord(
                            trade_id=row["trade_id"] or f"loaded_{loaded}",
                            symbol=self._normalize_symbol(row["symbol"]),
                            side="sell",
                            quantity=float(row["quantity"] or 0),
                            entry_price=entry_price,
                            exit_price=exit_price,
                            entry_time=entry_time,
                            exit_time=exit_time,
                            duration_minutes=duration_min,
                            pnl=pnl,
                            pnl_percentage=pnl_pct,
                            commission=commission,
                            net_pnl=net_pnl,
                            strategy=row["strategy"] or row["exit_type"] or "unknown",
                        )
                        self.trades.append(trade)
                        loaded += 1
                    except Exception:
                        continue
                if loaded:
                    logger.info(f"PERF_ANALYTICS: Loaded {loaded} historical trades from paper_trades")
            finally:
                conn.close()
        except Exception as e:
            logger.warning(f"PERF_ANALYTICS: Failed to load trades from SQLite: {e}")

    async def _ensure_redis(self) -> None:
        if self.redis_client is None and redis is not None:
            try:
                self.redis_client = get_shared_redis_async()
            except Exception as e:
                self.redis_client = None
                logger.exception("Redis connection failed: %s", e)

    def _normalize_symbol(self, symbol: str) -> str:
        s = str(symbol).strip().upper()
        if "/" in s:
            base, quote = s.split("/", 1)
            return _to_ccxt_symbol(f"{base}/{quote}")
        if s.endswith("USDT"):
            base = s[:-4]
            return _to_ccxt_symbol(f"{base}/USDT")
        return _to_ccxt_symbol(f"{s}/USDT")

    async def add_trade(self, trade: TradeRecord) -> None:
        try:
            trade.symbol = self._normalize_symbol(trade.symbol)
        except Exception as e:
            logger.exception("Trade rejected due to invalid symbol '%s': %s", trade.symbol, e)
            return
        else:
            try:
                self.trades.append(trade)
                await self._ensure_redis()
                if self.redis_client is not None:
                    td = {
                        "trade_id": trade.trade_id,
                        "symbol": trade.symbol,
                        "side": trade.side,
                        "quantity": str(trade.quantity),
                        "entry_price": str(trade.entry_price),
                        "exit_price": str(trade.exit_price),
                        "entry_time": trade.entry_time.isoformat(),
                        "exit_time": trade.exit_time.isoformat(),
                        "duration_minutes": str(trade.duration_minutes),
                        "pnl": str(trade.pnl),
                        "pnl_percentage": str(trade.pnl_percentage),
                        "commission": str(trade.commission),
                        "net_pnl": str(trade.net_pnl),
                        "strategy": trade.strategy,
                        "notes": trade.notes,
                    }
                    for k, v in td.items():
                        await self.redis_client.hset(f"trade:{trade.trade_id}", k, v)
                    await self.redis_client.expire(f"trade:{trade.trade_id}", 86400 * 30)
                logger.info(
                    "Trade added: %s %s net_pnl=%.4f",
                    trade.symbol,
                    trade.side,
                    trade.net_pnl,
                )
            except Exception as e:
                logger.exception("Error adding trade: %s", e)

    async def calculate_performance_metrics(self, days: int = 30) -> PerformanceMetrics:
        try:
            self._ensure_trades_loaded()
            cutoff = datetime.now(timezone.utc) - timedelta(days=int(days))
            recent = [t for t in self.trades if t.exit_time >= cutoff]
            if not recent:
                out_empty = self._create_empty_metrics()
                return out_empty

            total_trades = len(recent)
            wins = [t for t in recent if t.net_pnl > 0.0]
            losses = [t for t in recent if t.net_pnl < 0.0]
            winning_trades = len(wins)
            losing_trades = len(losses)
            win_rate = winning_trades / total_trades if total_trades > 0 else 0.0

            total_pnl = sum(t.net_pnl for t in recent)
            gross_profit = sum(t.net_pnl for t in recent if t.net_pnl > 0.0)
            gross_loss = abs(sum(t.net_pnl for t in recent if t.net_pnl < 0.0))
            profit_factor = (gross_profit / gross_loss) if gross_loss > 0.0 else float("inf")

            avg_win = (sum(t.net_pnl for t in wins) / winning_trades) if winning_trades > 0 else 0.0
            avg_loss = (sum(t.net_pnl for t in losses) / losing_trades) if losing_trades > 0 else 0.0
            largest_win = max((t.net_pnl for t in recent), default=0.0)
            largest_loss = min((t.net_pnl for t in recent), default=0.0)

            daily_returns = self._aggregate_daily_returns(recent)  # list of floats (decimal), can be empty
            mean_daily = self._mean(daily_returns) if daily_returns else 0.0
            std_daily = self._stdev(daily_returns) if len(daily_returns) > 1 else 0.0
            volatility = std_daily * sqrt(252.0) if std_daily > 0.0 else 0.0

            rf_daily = self.risk_free_rate / 252.0
            excess_daily = [r - rf_daily for r in daily_returns]
            mean_excess = self._mean(excess_daily) if excess_daily else 0.0
            std_excess = self._stdev(excess_daily) if len(excess_daily) > 1 else 0.0
            sharpe_ratio = (mean_excess / std_excess * sqrt(252.0)) if std_excess > 0.0 else 0.0

            downside = [r - rf_daily for r in daily_returns if (r - rf_daily) < 0.0]
            downside_std = self._stdev(downside) if len(downside) > 1 else 0.0
            sortino_ratio = (mean_excess / downside_std * sqrt(252.0)) if downside_std > 0.0 else 0.0

            equity_curve = self._equity_curve_from_returns(daily_returns)  # list of floats >= 0, starting at 1.0
            max_drawdown, max_dd_duration = self._drawdown_stats(equity_curve)

            annualized_return = mean_daily * 252.0
            calmar_ratio = (annualized_return / max_drawdown) if max_drawdown > 0.0 else 0.0

            recovery_factor = (total_pnl / (max_drawdown * gross_loss)) if (max_drawdown > 0.0 and gross_loss > 0.0) else 0.0
            expectancy = (win_rate * avg_win) - ((1.0 - win_rate) * abs(avg_loss))
            kelly_percentage = self._calculate_kelly_percentage(win_rate, avg_win, abs(avg_loss))

            total_return_pct = 0.0
            if equity_curve:
                total_return_pct = (equity_curve[-1] - 1.0) * 100.0

            return PerformanceMetrics(
                total_trades=total_trades,
                winning_trades=winning_trades,
                losing_trades=losing_trades,
                win_rate=win_rate,
                profit_factor=profit_factor,
                sharpe_ratio=sharpe_ratio,
                sortino_ratio=sortino_ratio,
                max_drawdown=max_drawdown,
                max_drawdown_duration=max_dd_duration,
                avg_win=avg_win,
                avg_loss=avg_loss,
                largest_win=largest_win,
                largest_loss=largest_loss,
                total_pnl=total_pnl,
                total_return=total_return_pct,
                annualized_return=annualized_return,
                volatility=volatility,
                calmar_ratio=calmar_ratio,
                recovery_factor=recovery_factor,
                expectancy=expectancy,
                kelly_percentage=kelly_percentage,
                timestamp=datetime.now(timezone.utc),
            )
        except Exception as e:
            logger.exception("Error calculating performance metrics: %s", e)
            out_empty = self._create_empty_metrics()
            return out_empty
        else:
            return PerformanceMetrics(
                total_trades=total_trades,
                winning_trades=winning_trades,
                losing_trades=losing_trades,
                win_rate=win_rate,
                profit_factor=profit_factor,
                sharpe_ratio=sharpe_ratio,
                sortino_ratio=sortino_ratio,
                max_drawdown=max_drawdown,
                max_drawdown_duration=max_dd_duration,
                avg_win=avg_win,
                avg_loss=avg_loss,
                largest_win=largest_win,
                largest_loss=largest_loss,
                total_pnl=total_pnl,
                total_return=total_return_pct,
                annualized_return=annualized_return,
                volatility=volatility,
                calmar_ratio=calmar_ratio,
                recovery_factor=recovery_factor,
                expectancy=expectancy,
                kelly_percentage=kelly_percentage,
                timestamp=datetime.now(timezone.utc),
            )

    def _create_empty_metrics(self) -> PerformanceMetrics:
        return PerformanceMetrics(
            total_trades=0,
            winning_trades=0,
            losing_trades=0,
            win_rate=0.0,
            profit_factor=0.0,
            sharpe_ratio=0.0,
            sortino_ratio=0.0,
            max_drawdown=0.0,
            max_drawdown_duration=0,
            avg_win=0.0,
            avg_loss=0.0,
            largest_win=0.0,
            largest_loss=0.0,
            total_pnl=0.0,
            total_return=0.0,
            annualized_return=0.0,
            volatility=0.0,
            calmar_ratio=0.0,
            recovery_factor=0.0,
            expectancy=0.0,
            kelly_percentage=0.0,
            timestamp=datetime.now(timezone.utc),
        )

    def _aggregate_daily_returns(self, trades: list[TradeRecord]) -> list[float]:
        buckets: dict[date, list[float]] = {}
        for t in trades:
            d = t.exit_time.astimezone(timezone.utc).date()
            r = float(t.pnl_percentage) / 100.0
            if d not in buckets:
                buckets[d] = []
            buckets[d].append(r)
        days_sorted = sorted(buckets.keys())
        daily_returns: list[float] = []
        for d in days_sorted:
            factor = 1.0
            for r in buckets[d]:
                factor *= 1.0 + r
            daily_returns.append(factor - 1.0)
        return daily_returns

    def _equity_curve_from_returns(self, daily_returns: list[float]) -> list[float]:
        eq: list[float] = []
        cur = 1.0
        for r in daily_returns:
            cur *= 1.0 + r
            eq.append(cur)
        return eq

    def _drawdown_stats(self, equity_curve: list[float]) -> tuple[float, int]:
        if not equity_curve:
            return 0.0, 0
        run_max = 1.0
        max_dd = 0.0
        cur_dd_len = 0
        max_dd_len = 0
        for val in equity_curve:
            run_max = max(run_max, val)
            dd = (val / run_max) - 1.0
            if dd < 0.0:
                cur_dd_len += 1
                max_dd = max(max_dd, abs(dd))
            else:
                max_dd_len = max(max_dd_len, cur_dd_len)
                cur_dd_len = 0
        max_dd_len = max(max_dd_len, cur_dd_len)
        return float(max_dd), int(max_dd_len)

    def _mean(self, vals: list[float]) -> float:
        if not vals:
            return 0.0
        return sum(vals) / float(len(vals))

    def _stdev(self, vals: list[float]) -> float:
        n = len(vals)
        if n < 2:
            return 0.0
        m = self._mean(vals)
        var = sum((x - m) * (x - m) for x in vals) / float(n - 1)
        return sqrt(var)

    def _calculate_kelly_percentage(self, win_rate: float, avg_win: float, avg_loss_abs: float) -> float:
        try:
            if avg_loss_abs <= 0.0:
                return 0.0
            b = (avg_win / avg_loss_abs) if avg_loss_abs > 0.0 else 0.0
            p = max(0.0, min(win_rate, 1.0))
            q = 1.0 - p
            if b <= 0.0:
                return 0.0
            k = (b * p - q) / b
            if k < 0.0:
                return 0.0
            return min(k, 0.25)
        except Exception as e:
            logger.exception("Error calculating Kelly percentage: %s", e)
            return 0.0

    async def get_performance_summary(self) -> dict[str, Any]:
        try:
            self._ensure_trades_loaded()
            period_metrics: dict[str, dict[str, Any]] = {}
            for days in self.lookback_periods:
                m = await self.calculate_performance_metrics(days)
                period_metrics[f"{days}d"] = {
                    "total_trades": m.total_trades,
                    "win_rate": round(m.win_rate * 100.0, 2),
                    "total_pnl": round(m.total_pnl, 2),
                    "total_return": round(m.total_return, 2),
                    "sharpe_ratio": round(m.sharpe_ratio, 3),
                    "max_drawdown": round(m.max_drawdown * 100.0, 2),
                    "profit_factor": round(m.profit_factor, 2),
                }
            current = await self.calculate_performance_metrics(30)
            portfolio_value = await self._calculate_portfolio_value_over_time()
            if not portfolio_value:
                portfolio_value = await self._load_portfolio_value_from_paper_trades()
            if not portfolio_value:
                portfolio_value = await self._load_portfolio_value_from_canonical()
            top_symbols = await self._get_top_performing_symbols()
            strat_perf = await self._get_strategy_performance()
            out_summary = {
                "current_metrics": {
                    "total_trades": current.total_trades,
                    "win_rate": round(current.win_rate * 100.0, 2),
                    "total_pnl": round(current.total_pnl, 2),
                    "total_return": round(current.total_return, 2),
                    "sharpe_ratio": round(current.sharpe_ratio, 3),
                    "sortino_ratio": round(current.sortino_ratio, 3),
                    "max_drawdown": round(current.max_drawdown * 100.0, 2),
                    "profit_factor": round(current.profit_factor, 2),
                    "volatility": round(current.volatility * 100.0, 2),
                    "calmar_ratio": round(current.calmar_ratio, 3),
                    "expectancy": round(current.expectancy, 2),
                    "kelly_percentage": round(current.kelly_percentage * 100.0, 2),
                },
                "period_metrics": period_metrics,
                "portfolio_value": portfolio_value,
                "top_symbols": top_symbols,
                "strategy_performance": strat_perf,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
        except Exception as e:
            logger.exception("Error getting performance summary: %s", e)
            return {"error": str(e)}
        else:
            return out_summary

    async def calculate_performance_metrics_by_trades(self, last_n_trades: int) -> PerformanceMetrics:
        """Calculate performance metrics for the last N trades (most recent first)"""
        try:
            if not self.trades or last_n_trades <= 0:
                return self._create_empty_metrics()

            # Sort trades by exit time (most recent first) and take the last N
            recent_trades = sorted(self.trades, key=lambda t: t.exit_time, reverse=True)[:last_n_trades]

            if not recent_trades:
                return self._create_empty_metrics()

            # Use the same calculation logic as the time-based method
            total_trades = len(recent_trades)
            wins = [t for t in recent_trades if t.net_pnl > 0.0]
            losses = [t for t in recent_trades if t.net_pnl < 0.0]
            winning_trades = len(wins)
            losing_trades = len(losses)
            win_rate = winning_trades / total_trades if total_trades > 0 else 0.0

            total_pnl = sum(t.net_pnl for t in recent_trades)
            gross_profit = sum(t.net_pnl for t in recent_trades if t.net_pnl > 0.0)
            gross_loss = abs(sum(t.net_pnl for t in recent_trades if t.net_pnl < 0.0))
            profit_factor = (gross_profit / gross_loss) if gross_loss > 0.0 else float("inf")

            avg_win = (sum(t.net_pnl for t in wins) / winning_trades) if winning_trades > 0 else 0.0
            avg_loss = (sum(t.net_pnl for t in losses) / losing_trades) if losing_trades > 0 else 0.0
            largest_win = max((t.net_pnl for t in recent_trades), default=0.0)
            largest_loss = min((t.net_pnl for t in recent_trades), default=0.0)

            daily_returns = self._aggregate_daily_returns(recent_trades)
            mean_daily = self._mean(daily_returns) if daily_returns else 0.0
            std_daily = self._stdev(daily_returns) if len(daily_returns) > 1 else 0.0
            volatility = std_daily * sqrt(252.0) if std_daily > 0.0 else 0.0

            total_return = (total_pnl / self.initial_balance) if self.initial_balance > 0 else 0.0
            annualized_return = total_return * (365.0 / 30.0) if total_return != 0 else 0.0  # Rough annualization

            sharpe_ratio = (mean_daily / std_daily) if std_daily > 0.0 else 0.0
            sortino_ratio = self._calculate_sortino_ratio(daily_returns)

            max_drawdown, max_drawdown_duration = self._max_drawdown_from_returns(daily_returns)

            calmar_ratio = (annualized_return / abs(max_drawdown)) if max_drawdown != 0.0 else 0.0

            # Calculate expectancy for last N trades
            expectancy = (win_rate * avg_win) - ((1.0 - win_rate) * abs(avg_loss)) if total_trades > 0 else 0.0

            kelly_percentage = self._calculate_kelly_percentage(win_rate, avg_win, avg_loss)

            return PerformanceMetrics(
                total_trades=total_trades,
                winning_trades=winning_trades,
                losing_trades=losing_trades,
                win_rate=win_rate,
                profit_factor=profit_factor,
                sharpe_ratio=sharpe_ratio,
                sortino_ratio=sortino_ratio,
                max_drawdown=max_drawdown,
                max_drawdown_duration=max_drawdown_duration,
                avg_win=avg_win,
                avg_loss=avg_loss,
                largest_win=largest_win,
                largest_loss=largest_loss,
                total_pnl=total_pnl,
                total_return=total_return,
                annualized_return=annualized_return,
                volatility=volatility,
                calmar_ratio=calmar_ratio,
                expectancy=expectancy,
                kelly_percentage=kelly_percentage,
            )

        except Exception as e:
            logger.exception("Error calculating performance metrics by trades: %s", e)
            return self._create_empty_metrics()

    @staticmethod
    def _get_initial_value_from_ledger_sync(conn: sqlite3.Connection) -> float:
        """Read principal from portfolio_engine_ledger (live data). Fallback: 0 (never assume 2500 for live)."""
        try:
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
        return 0.0

    async def _load_portfolio_value_from_paper_trades(self, run_id: str | None = None) -> list[dict[str, Any]]:
        """Build portfolio value series from paper_trades (live canonical). Optional run_id = current session only."""

        def _sync_load(rid: str | None) -> list[dict[str, Any]]:
            try:
                conn = sqlite3.connect(DATABASE_PATH)
                try:
                    cursor = conn.cursor()
                    initial_value = self._get_initial_value_from_ledger_sync(conn)
                    query = """
                        SELECT timestamp, pnl FROM paper_trades
                        WHERE mode IN ('paper', 'live') AND pnl IS NOT NULL
                    """
                    params: list[Any] = []
                    if rid is not None:
                        query += " AND paper_run_id = ?"
                        params.append(rid)
                    query += " ORDER BY timestamp ASC"
                    cursor.execute(query, params)
                    rows = cursor.fetchall()
                finally:
                    conn.close()
                cumulative = 0.0
                initial_value_local = initial_value
                out: list[dict[str, Any]] = []
                for ts_str, pnl in rows:
                    cumulative += float(pnl or 0.0)
                    cur_val = initial_value_local + cumulative
                    out.append(
                        {
                            "date": ts_str,
                            "value": round(cur_val, 2),
                            "pnl": round(cumulative, 2),
                            "return_pct": round((cur_val - initial_value_local) / initial_value_local * 100.0, 2) if initial_value_local else 0.0,
                        },
                    )
            except Exception as e:
                logger.debug("No portfolio value from paper_trades: %s", e)
                return []
            return out

        # CRITICAL #1 FIX: Replace asyncio.get_event_loop() with asyncio.get_running_loop()
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, lambda: _sync_load(run_id))

    async def _load_portfolio_value_from_canonical(self) -> list[dict[str, Any]]:
        """Read portfolio value series from trade_performance (derived). Use after paper_trades when needed."""

        def _sync_load() -> list[dict[str, Any]]:
            try:
                conn = sqlite3.connect(DATABASE_PATH)
                try:
                    cursor = conn.cursor()
                    initial_value = self._get_initial_value_from_ledger_sync(conn)
                    cursor.execute(
                        "SELECT timestamp, pnl FROM trade_performance ORDER BY timestamp ASC",
                    )
                    rows = cursor.fetchall()
                finally:
                    conn.close()
                cumulative = 0.0
                out: list[dict[str, Any]] = []
                for ts_str, pnl in rows:
                    cumulative += float(pnl or 0.0)
                    cur_val = initial_value + cumulative
                    out.append(
                        {
                            "date": ts_str,
                            "value": round(cur_val, 2),
                            "pnl": round(cumulative, 2),
                            "return_pct": round((cur_val - initial_value) / initial_value * 100.0, 2) if initial_value else 0.0,
                        },
                    )
            except Exception as e:
                logger.debug("No portfolio value from canonical store: %s", e)
                return []
            else:
                return out

        # CRITICAL #1 FIX: Replace asyncio.get_event_loop() with asyncio.get_running_loop()
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, _sync_load)

    async def _calculate_portfolio_value_over_time(self) -> list[dict[str, Any]]:
        try:
            if not self.trades:
                return []

            def _read_initial() -> float:
                conn = sqlite3.connect(DATABASE_PATH)
                try:
                    return self._get_initial_value_from_ledger_sync(conn)
                finally:
                    conn.close()

            # CRITICAL #1 FIX: Replace asyncio.get_event_loop() with asyncio.get_running_loop()
            loop = asyncio.get_running_loop()
            initial_value = await loop.run_in_executor(None, _read_initial)
            sorted_trades = sorted(self.trades, key=lambda t: t.exit_time)
            cumulative = 0.0
            out: list[dict[str, Any]] = []
            for t in sorted_trades:
                cumulative += t.net_pnl
                cur_val = initial_value + cumulative
                out.append(
                    {
                        "date": t.exit_time.isoformat(),
                        "value": round(cur_val, 2),
                        "pnl": round(cumulative, 2),
                        "return_pct": round((cur_val - initial_value) / initial_value * 100.0, 2) if initial_value else 0.0,
                    },
                )
        except Exception as e:
            logger.exception("Error calculating portfolio value: %s", e)
            return []
        else:
            return out

    async def _get_top_performing_symbols(self, limit: int = 5) -> list[dict[str, Any]]:
        try:
            if not self.trades:
                return []
            agg: dict[str, dict[str, float]] = {}
            for t in self.trades:
                sym = self._normalize_symbol(t.symbol)
                if sym not in agg:
                    agg[sym] = {"pnl": 0.0, "count": 0.0}
                agg[sym]["pnl"] += float(t.net_pnl)
                agg[sym]["count"] += 1.0
            ranked = sorted(agg.items(), key=lambda kv: kv[1]["pnl"], reverse=True)
            out: list[dict[str, Any]] = []
            for sym, stats in ranked[: int(limit)]:
                count = int(stats["count"])
                avg_pnl = (stats["pnl"] / count) if count > 0 else 0.0
                out.append(
                    {
                        "symbol": sym,
                        "total_pnl": round(stats["pnl"], 2),
                        "trades_count": count,
                        "avg_pnl": round(avg_pnl, 2),
                    },
                )
        except Exception as e:
            logger.exception("Error getting top performing symbols: %s", e)
            return []
        else:
            return out

    async def _get_strategy_performance(self) -> list[dict[str, Any]]:
        try:
            if not self.trades:
                return []
            bucket: dict[str, dict[str, Any]] = {}
            for t in self.trades:
                k = t.strategy
                if k not in bucket:
                    bucket[k] = {"trades": 0, "wins": 0, "pnl": 0.0}
                bucket[k]["trades"] += 1
                bucket[k]["pnl"] += float(t.net_pnl)
                if t.net_pnl > 0.0:
                    bucket[k]["wins"] += 1
            out: list[dict[str, Any]] = []
            for strat, stats in bucket.items():
                trades_count = int(stats["trades"])
                win_rate = (stats["wins"] / trades_count) if trades_count > 0 else 0.0
                avg_pnl = (stats["pnl"] / trades_count) if trades_count > 0 else 0.0
                out.append(
                    {
                        "strategy": strat,
                        "total_pnl": round(stats["pnl"], 2),
                        "trades_count": trades_count,
                        "win_rate": round(win_rate * 100.0, 2),
                        "avg_pnl": round(avg_pnl, 2),
                    },
                )
            out.sort(key=lambda x: x["total_pnl"], reverse=True)
        except Exception as e:
            logger.exception("Error getting strategy performance: %s", e)
            return []
        else:
            return out

    async def get_trade_history(self, limit: int = 100, symbol: str | None = None) -> list[dict[str, Any]]:
        try:
            trades = list(self.trades)
            if symbol:
                try:
                    sym_norm = self._normalize_symbol(symbol)
                    trades = [t for t in trades if self._normalize_symbol(t.symbol) == sym_norm]
                except (ValueError, TypeError, AttributeError, KeyError, IndexError):
                    trades = []
            trades.sort(key=lambda t: t.exit_time, reverse=True)
            trades = trades[: int(limit)]
            out = []
            for t in trades:
                out.append(
                    {
                        "trade_id": t.trade_id,
                        "symbol": self._normalize_symbol(t.symbol),
                        "side": t.side,
                        "quantity": float(t.quantity),
                        "entry_price": float(t.entry_price),
                        "exit_price": float(t.exit_price),
                        "entry_time": t.entry_time.isoformat(),
                        "exit_time": t.exit_time.isoformat(),
                        "duration_minutes": int(t.duration_minutes),
                        "pnl": round(float(t.pnl), 2),
                        "pnl_percentage": round(float(t.pnl_percentage), 2),
                        "commission": round(float(t.commission), 2),
                        "net_pnl": round(float(t.net_pnl), 2),
                        "strategy": t.strategy,
                        "notes": t.notes,
                    },
                )
        except Exception as e:
            logger.exception("Error getting trade history: %s", e)
            return []
        else:
            return out

    async def generate_performance_report(self) -> dict[str, Any]:
        try:
            summary = await self.get_performance_summary()
            recent_trades = await self.get_trade_history(50)
            insights = await self._generate_insights()

            # Structure data according to PerformanceReport model
            current_time = datetime.now(timezone.utc)

            out_obj = {
                "current_performance": {
                    "total_pnl": summary.get("total_pnl", 0.0),
                    "total_return": summary.get("total_return", 0.0),
                    "win_rate": summary.get("win_rate", 0.0),
                    "total_trades": summary.get("total_trades", 0),
                    "avg_win": summary.get("avg_win", 0.0),
                    "avg_loss": summary.get("avg_loss", 0.0),
                    "capital": 0.0,  # From ledger when available
                    "timestamp": current_time.isoformat(),
                },
                "targets": {
                    "dollar_per_minute": 1.0,
                    "dollar_per_hour": 60.0,
                    "dollar_per_day": 1440.0,
                    "win_rate_target": 0.60,
                    "max_drawdown_limit": 0.20,
                },
                "runtime": {
                    "uptime_seconds": 0,  # Would need to track this
                    "last_restart": current_time.isoformat(),
                    "total_predictions": 0,
                    "active_strategies": 0,
                },
                "milestones": {
                    "achieved": [],
                    "in_progress": ["$0.10/min", "$0.25/min", "$0.50/min"],
                    "remaining": ["$1.00/min"],
                    "next_target": "$0.10/min",
                },
                "streaks": {
                    "current_win_streak": 0,
                    "current_loss_streak": 0,
                    "longest_win_streak": 0,
                    "longest_loss_streak": 0,
                },
                "analytics": {
                    "insights": insights,
                    "risk_metrics": {
                        "sharpe_ratio": summary.get("sharpe_ratio", 0.0),
                        "sortino_ratio": summary.get("sortino_ratio", 0.0),
                        "max_drawdown": summary.get("max_drawdown", 0.0),
                        "recovery_factor": summary.get("recovery_factor", 0.0),
                    },
                    "performance_distribution": {
                        "small_wins": 0,
                        "large_wins": 0,
                        "small_losses": 0,
                        "large_losses": 0,
                    },
                },
                "rolling_performance": {
                    "daily_returns": [],  # Would need historical data
                    "weekly_performance": [],
                    "monthly_performance": [],
                    "recent_trades": recent_trades,
                },
            }
        except Exception as e:
            logger.exception("Error generating performance report: %s", e)
            return {"error": str(e)}
        else:
            return out_obj

    async def _generate_insights(self) -> list[str]:
        try:
            insights: list[str] = []
            m = await self.calculate_performance_metrics(30)
            if m.win_rate > 0.60:
                insights.append(f"Win rate {m.win_rate:.1%} suggests cautiously increasing size.")
            elif m.win_rate < 0.40:
                insights.append(f"Win rate {m.win_rate:.1%} is low; review entries and cut losers faster.")
            if m.profit_factor > 2.0:
                insights.append(f"Profit factor {m.profit_factor:.2f} indicates strong edge.")
            elif m.profit_factor < 1.0:
                insights.append(f"Profit factor {m.profit_factor:.2f} indicates negative edge; adjust strategy.")
            if m.max_drawdown > 0.20:
                insights.append(f"Max drawdown {m.max_drawdown:.1%} is high; tighten risk management.")
            if m.sharpe_ratio > 1.0:
                insights.append(f"Sharpe {m.sharpe_ratio:.2f} shows good risk-adjusted returns.")
            elif m.sharpe_ratio < 0.5:
                insights.append(f"Sharpe {m.sharpe_ratio:.2f} is weak; reduce variance or improve edge.")
            if m.kelly_percentage > 0.10:
                insights.append(f"Kelly suggests {m.kelly_percentage:.1%}; consider modest size increases.")
            elif m.kelly_percentage < 0.05:
                insights.append(f"Kelly suggests {m.kelly_percentage:.1%}; consider reducing size.")
            out_obj = insights
        except Exception as e:
            logger.exception("Error generating insights: %s", e)
            return ["Error generating insights"]
        else:
            return out_obj

    async def get_performance_analytics(self) -> dict[str, Any]:
        """Get performance analytics - alias for generate_performance_report"""
        return await self.generate_performance_report()

    async def update_performance(self, pnl: float, trade_count: int = 1, is_win: bool | None = None) -> dict[str, Any]:
        """Update performance with new trade results"""
        try:
            # Create a trade record for tracking
            trade = TradeRecord(
                trade_id=f"manual_{datetime.now(timezone.utc).timestamp()}",
                symbol="PORTFOLIO",  # Portfolio level trade
                side="BUY",  # Default side
                quantity=1.0,
                entry_price=0.0,
                exit_price=pnl,
                entry_time=datetime.now(timezone.utc),
                exit_time=datetime.now(timezone.utc),
                duration_minutes=0,
                pnl=pnl,
                pnl_percentage=0.0,
                commission=0.0,
                net_pnl=pnl,
                strategy="manual_update",
                notes=f"Manual performance update: ${pnl:.2f}",
            )

            await self.add_trade(trade)

            # Use parameters to avoid unused warnings
            _ = (trade_count, is_win)

            # Return current summary
            summary = await self.get_performance_summary()
        except Exception as e:
            logger.exception("Error updating performance: %s", e)
            return {"error": str(e)}
        else:
            return summary

    def get_goal_progress_visualization(self) -> dict[str, Any]:
        """Get goal progress visualization data for $1/minute target"""
        try:
            # Calculate current metrics (simplified for now)
            current_time = datetime.now(timezone.utc)

            # Structure data according to GoalProgress model
            goal_progress = {
                "current_progress": 0.0,  # Current dollar per minute achieved
                "target": 1.0,  # Target: $1/minute
                "progress_percent": 0.0,  # 0% progress so far
                "milestones_achieved": 0,  # No milestones achieved yet
                "next_milestone_value": 0.1,  # Next milestone: $0.10/min
                "estimated_completion": 999.0,  # Days to completion (placeholder)
                "performance_trend": [
                    {"timestamp": current_time.isoformat(), "value": 0.0, "target": 1.0},
                    # Would include historical trend data in real implementation
                ],
            }

        except Exception as e:
            logger.exception("Error getting goal progress visualization: %s", e)
            return {"error": str(e)}
        else:
            return goal_progress

    def get_performance_history(self, hours: int = 24) -> list[dict[str, Any]]:
        """Get performance history for specified hours"""
        try:
            # For now, return empty history - in a real implementation this would
            # aggregate performance data over time periods
            history = []

            # Calculate time range
            end_time = datetime.now(timezone.utc)
            start_time = end_time - timedelta(hours=hours)

            # Live aggregation is not implemented yet; return empty
            # series so the API never invents synthetic performance data.
            _ = (start_time, end_time)

        except Exception as e:
            logger.exception("Error getting performance history: %s", e)
            return []
        else:
            return history
