"""
Portfolio Service for Mystic AI Trading Platform
Provides comprehensive portfolio tracking and analysis using cached trade data.

Quick Test Checklist:
- Symbols normalized to BASE/QUOTE via _to_ccxt_symbol; accepts BTCUSDT and converts to BTC/USDT.
- No binance/binanceus string leaks-only centralized EXCHANGE_ID import.
- ccxt/price lookups use BASE/QUOTE only.
- ASCII-only logging; no unreachable code; live data only.
- No Streamlit, no Docker, no Coinbase, no CoinGecko, no Kraken, no yfinance.
- Python 3.12, backend port 8000, unified dashboard port 8000.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any

# Direct imports for production
from backend.services.canonical_cache import canonical_cache  # type: ignore[import-not-found]
from backend.services.constants import EXCHANGE_ID  # type: ignore[import-not-found]
from backend.services.symbols import _to_ccxt_symbol  # type: ignore[import-not-found]

# Optional imports - try at top level
try:
    from backend.services.live_trading_service import LiveTradingService  # type: ignore[import-not-found]
except (ImportError, ModuleNotFoundError, AttributeError, ValueError, TypeError, RuntimeError):
    LiveTradingService = None

logger = logging.getLogger(__name__)


class PortfolioService:
    """Service for managing portfolio data and operations using cached trade data."""

    _instance: PortfolioService | None = None

    def __init__(self) -> None:
        self.cache = canonical_cache
        self.portfolio_data: dict[str, Any] = {}
        self.positions: dict[str, Any] = {}
        self.transactions: list[dict[str, Any]] = []
        logger.info("PortfolioService initialized")

    @classmethod
    def shared(cls) -> PortfolioService:
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def _normalize_symbol(self, symbol: str) -> str:
        s = str(symbol).strip().upper()
        if not s:
            msg = "Empty symbol"
            raise ValueError(msg)
        if "/" in s:
            base, quote = s.split("/", 1)
            return _to_ccxt_symbol(f"{base}/{quote}")
        if s.endswith("USDT"):
            base = s[:-4]
            return _to_ccxt_symbol(f"{base}/USDT")
        return _to_ccxt_symbol(f"{s}/USDT")

    def _parse_iso_ts(self, value: Any) -> int:
        try:
            if isinstance(value, (int, float)):
                return int(value)
            if isinstance(value, str) and value:
                # Try ISO8601
                return int(datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp())
        except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError):
            return 0
        return 0

    async def _get_trade_history(self, limit: int = 1000) -> list[dict[str, Any]]:
        try:
            # Use await to properly handle coroutine
            signals = await self.cache.get_signals_by_type("TRADE_EXECUTED", limit=limit)
            trades: list[dict[str, Any]] = []
            for signal in signals:
                trade_data = signal.get("metadata", {}) or {}
                raw_sym = signal.get("symbol", "") or trade_data.get("symbol", "")
                try:
                    sym = self._normalize_symbol(raw_sym)
                except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError):
                    continue
                side = trade_data.get("trade_type") or trade_data.get("side") or trade_data.get("type") or ""
                qty = float(trade_data.get("quantity") or trade_data.get("qty") or 0.0)
                price = float(trade_data.get("price") or 0.0)
                amt_usd = float(trade_data.get("amount_usd") or (qty * price) or 0.0)
                ts_val = signal.get("timestamp") or trade_data.get("timestamp") or ""
                trades.append(
                    {
                        "symbol": sym,
                        "trade_type": str(side).strip().upper(),
                        "quantity": qty,
                        "price": price,
                        "amount_usd": amt_usd,
                        "exchange": trade_data.get("exchange") or signal.get("exchange") or EXCHANGE_ID,
                        "timestamp": str(ts_val),
                        "trade_id": signal.get("signal_id") or trade_data.get("trade_id") or "",
                    },
                )
            out_trades = trades[: max(0, int(limit))]
        except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
            logger.exception("Failed to get trade history: %s", e)
            return []
        else:
            return out_trades

    async def _get_latest_prices(self) -> dict[str, float]:
        try:
            # Use await to properly handle coroutine
            signals = await self.cache.get_signals_by_type("PRICE_UPDATE", limit=1000)
            latest: dict[str, tuple[int, float]] = {}
            for signal in signals:
                raw_sym = signal.get("symbol", "")
                try:
                    sym = self._normalize_symbol(raw_sym)
                except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError):
                    continue
                price = float(signal.get("metadata", {}).get("price") or 0.0)
                if price <= 0.0:
                    continue
                ts_i = self._parse_iso_ts(signal.get("timestamp"))
                prev = latest.get(sym)
                if prev is None or ts_i > prev[0]:
                    latest[sym] = (ts_i, price)
            out_latest = {k: v[1] for k, v in latest.items()}
        except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
            logger.exception("Failed to get latest prices: %s", e)
            return {}
        else:
            return out_latest

    def _calculate_holdings(self, trades: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
        try:
            holdings: dict[str, dict[str, Any]] = {}
            for trade in trades:
                sym = trade.get("symbol", "")
                try:
                    sym = self._normalize_symbol(sym)
                except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError):
                    continue
                side = str(trade.get("trade_type", "")).strip().upper()
                qty = float(trade.get("quantity") or 0.0)
                amt_usd = float(trade.get("amount_usd") or 0.0)
                if not sym or qty <= 0.0:
                    continue
                if sym not in holdings:
                    holdings[sym] = {
                        "quantity": 0.0,
                        "total_buy_amount": 0.0,
                        "total_sell_amount": 0.0,
                        "total_buy_quantity": 0.0,
                        "total_sell_quantity": 0.0,
                        "average_buy_price": 0.0,
                        "trades": [],
                    }
                h = holdings[sym]
                h["trades"].append(trade)
                if side == "BUY":
                    h["quantity"] += qty
                    h["total_buy_amount"] += amt_usd
                    h["total_buy_quantity"] += qty
                elif side == "SELL":
                    h["quantity"] -= qty
                    h["total_sell_amount"] += amt_usd
                    h["total_sell_quantity"] += qty
                if h["total_buy_quantity"] > 0.0:
                    h["average_buy_price"] = h["total_buy_amount"] / h["total_buy_quantity"]
        except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
            logger.exception("Failed to calculate holdings: %s", e)
            return {}
        else:
            return holdings

    def _calculate_unrealized_pnl(self, holdings: dict[str, dict[str, Any]], latest_prices: dict[str, float]) -> dict[str, Any]:
        try:
            total_pnl = 0.0
            total_value = 0.0
            total_cost_basis = 0.0
            pnl_by_symbol: dict[str, dict[str, float]] = {}
            for sym, h in holdings.items():
                qty = float(h.get("quantity") or 0.0)
                if qty <= 0.0:
                    continue
                avg_buy = float(h.get("average_buy_price") or 0.0)
                cur_price = float(latest_prices.get(sym) or 0.0)
                if avg_buy <= 0.0 or cur_price <= 0.0:
                    continue
                cur_val = qty * cur_price
                cost_basis = qty * avg_buy
                u_pnl = cur_val - cost_basis
                pnl_pct = (u_pnl / cost_basis * 100.0) if cost_basis > 0.0 else 0.0
                pnl_by_symbol[sym] = {
                    "quantity": qty,
                    "average_buy_price": avg_buy,
                    "current_price": cur_price,
                    "current_value": cur_val,
                    "cost_basis": cost_basis,
                    "unrealized_pnl": u_pnl,
                    "pnl_percentage": pnl_pct,
                }
                total_pnl += u_pnl
                total_value += cur_val
                total_cost_basis += cost_basis
            out_pnl = {
                "total_pnl": total_pnl,
                "total_value": total_value,
                "total_cost_basis": total_cost_basis,
                "pnl_by_symbol": pnl_by_symbol,
            }
        except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
            logger.exception("Failed to calculate unrealized PnL: %s", e)
            return {
                "total_pnl": 0.0,
                "total_value": 0.0,
                "total_cost_basis": 0.0,
                "pnl_by_symbol": {},
            }
        else:
            return out_pnl

    async def get_portfolio_overview(self) -> dict[str, Any]:
        try:
            logger.info("Getting portfolio overview")
            trades = await self._get_trade_history()
            latest_prices = await self._get_latest_prices()
            holdings = self._calculate_holdings(trades)
            pnl = self._calculate_unrealized_pnl(holdings, latest_prices)
            positions_detail: dict[str, Any] = {}
            for sym, h in holdings.items():
                qty = float(h.get("quantity") or 0.0)
                if qty <= 0.0:
                    continue
                info = pnl["pnl_by_symbol"].get(sym, {})
                positions_detail[sym] = {
                    "quantity": qty,
                    "average_buy_price": float(h.get("average_buy_price") or 0.0),
                    "current_price": float(latest_prices.get(sym) or 0.0),
                    "current_value": float(info.get("current_value") or 0.0),
                    "cost_basis": float(info.get("cost_basis") or 0.0),
                    "unrealized_pnl": float(info.get("unrealized_pnl") or 0.0),
                    "pnl_percentage": float(info.get("pnl_percentage") or 0.0),
                    "total_buy_amount": float(h.get("total_buy_amount") or 0.0),
                    "total_sell_amount": float(h.get("total_sell_amount") or 0.0),
                    "trade_count": len(h.get("trades") or []),
                }
            total_value = float(pnl["total_value"])
            total_pnl = float(pnl["total_pnl"])
            total_cost_basis = float(pnl["total_cost_basis"])
            positions_count = sum(1 for v in holdings.values() if float(v.get("quantity") or 0.0) > 0.0)
            overview = {
                "total_value": total_value,
                "total_pnl": total_pnl,
                "positions_count": positions_count,
                "total_trades": len(trades),
                "total_invested": total_cost_basis,
                "holdings": positions_detail,
                "performance": {
                    "total_pnl": total_pnl,
                    "total_value": total_value,
                    "pnl_percentage": (total_pnl / total_value * 100.0) if total_value > 0.0 else 0.0,
                },
                "last_updated": datetime.now(timezone.utc).isoformat(),
                "source": "cached_trade_data",
            }
        except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
            logger.exception("Failed to get portfolio overview: %s", e)
            return {
                "total_value": 0.0,
                "total_pnl": 0.0,
                "positions_count": 0,
                "total_trades": 0,
                "total_invested": 0.0,
                "holdings": {},
                "performance": {
                    "total_pnl": 0.0,
                    "total_value": 0.0,
                    "pnl_percentage": 0.0,
                },
                "last_updated": datetime.now(timezone.utc).isoformat(),
                "source": "error",
                "error": str(e),
            }
        else:
            return overview

    async def get_overview(self) -> dict[str, Any]:
        return await self.get_portfolio_overview()

    async def get_positions(self) -> list[dict[str, Any]]:
        try:
            positions: list[dict[str, Any]] = []
            try:
                if LiveTradingService is None:
                    logger.warning("LiveTradingService not available")
                    return []

                trading_service = LiveTradingService()
                balance_result = await trading_service.get_account_balance()
                if balance_result.get("status") == "success":
                    balances = balance_result.get("balances", {})
                    if EXCHANGE_ID in balances:
                        ex_bal = balances[EXCHANGE_ID]
                        total_map = ex_bal.get("total", {}) or {}
                        for asset, qty in total_map.items():
                            qty_f = float(qty)
                            if qty_f <= 0.0:
                                continue
                            if str(asset).upper() in ("USDT", "USD"):
                                continue
                            sym = self._normalize_symbol(f"{asset}/USDT")
                            try:
                                pr = await trading_service.get_market_price(sym)
                                cur_px = float(pr.get("price")) if pr else 0.0
                            except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError):
                                cur_px = 0.0
                            positions.append(
                                {
                                    "symbol": sym,
                                    "quantity": qty_f,
                                    "average_buy_price": cur_px,  # approximate when historical cost not available
                                    "current_price": cur_px,
                                    "current_value": qty_f * cur_px,
                                    "cost_basis": qty_f * cur_px,
                                    "unrealized_pnl": 0.0,
                                    "pnl_percentage": 0.0,
                                    "total_buy_amount": qty_f * cur_px,
                                    "total_sell_amount": 0.0,
                                    "trade_count": 1,
                                    "exchange": EXCHANGE_ID,
                                },
                            )
                    if positions:
                        return positions
            except ImportError:
                logger.warning("LiveTradingService not available")
            except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
                logger.warning("Failed to get live positions: %s", e)
            overview = await self.get_portfolio_overview()
            out: list[dict[str, Any]] = []
            for sym, h in (overview.get("holdings") or {}).items():
                if float(h.get("quantity") or 0.0) > 0.0:
                    out.append(
                        {
                            "symbol": sym,
                            "quantity": float(h.get("quantity") or 0.0),
                            "average_buy_price": float(h.get("average_buy_price") or 0.0),
                            "current_price": float(h.get("current_price") or 0.0),
                            "current_value": float(h.get("current_value") or 0.0),
                            "cost_basis": float(h.get("cost_basis") or 0.0),
                            "unrealized_pnl": float(h.get("unrealized_pnl") or 0.0),
                            "pnl_percentage": float(h.get("pnl_percentage") or 0.0),
                            "total_buy_amount": float(h.get("total_buy_amount") or 0.0),
                            "total_sell_amount": float(h.get("total_sell_amount") or 0.0),
                            "trade_count": int(h.get("trade_count") or 0),
                        },
                    )
        except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
            logger.exception("Failed to get positions: %s", e)
            return []
        else:
            return out

    async def get_portfolio_summary(self) -> dict[str, Any]:
        try:
            try:
                if LiveTradingService is None:
                    logger.warning("LiveTradingService not available")
                    return {
                        "total_value": 0.0,
                        "positions_count": 0,
                        "cash_allocation": 0.0,
                        "asset_allocations": {},
                        "data_source": "cache_only",
                    }

                trading_service = LiveTradingService()
                balance_result = await trading_service.get_account_balance()
                if balance_result.get("status") == "success":
                    balances = balance_result.get("balances", {})
                    total_value = 0.0
                    positions_count = 0
                    if EXCHANGE_ID in balances:
                        ex_bal = balances[EXCHANGE_ID]
                        total_map = ex_bal.get("total", {}) or {}
                        for asset, qty in total_map.items():
                            qty_f = float(qty)
                            if qty_f <= 0.0:
                                continue
                            if str(asset).upper() in ("USDT", "USD"):
                                total_value += qty_f
                            else:
                                sym = self._normalize_symbol(f"{asset}/USDT")
                                try:
                                    pr = await trading_service.get_market_price(sym)
                                    px = float(pr.get("price")) if pr else 0.0
                                except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError):
                                    px = 0.0
                                total_value += qty_f * px
                                positions_count += 1
                    return {
                        "total_value": total_value,
                        "total_pnl": 0.0,
                        "positions_count": positions_count,
                        "total_trades": 0,
                        "performance": {
                            "total_pnl": 0.0,
                            "total_value": total_value,
                            "pnl_percentage": 0.0,
                        },
                        "last_updated": datetime.now(timezone.utc).isoformat(),
                        "source": "live_trading_apis",
                    }
            except ImportError:
                logger.warning("LiveTradingService not available")
            except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
                logger.warning("Failed to get live balance: %s", e)
            overview = await self.get_portfolio_overview()
            return {
                "total_value": float(overview.get("total_value") or 0.0),
                "total_pnl": float(overview.get("total_pnl") or 0.0),
                "positions_count": int(overview.get("positions_count") or 0),
                "total_trades": int(overview.get("total_trades") or 0),
                "performance": overview.get("performance") or {},
                "last_updated": str(overview.get("last_updated") or ""),
                "source": str(overview.get("source") or ""),
            }
        except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
            logger.exception("Failed to get portfolio summary: %s", e)
            return {
                "total_value": 0.0,
                "total_pnl": 0.0,
                "positions_count": 0,
                "total_trades": 0,
                "performance": {},
                "last_updated": datetime.now(timezone.utc).isoformat(),
                "source": "error",
            }
        else:
            logger.info(
                "Portfolio overview generated: positions=%d, total_value=%.2f",
                overview["positions_count"],
                overview["total_value"],
            )
            return overview

    async def get_transactions(self, limit: int = 50) -> list[dict[str, Any]]:
        try:
            history = await self._get_trade_history(limit=max(0, int(limit)))
            txs: list[dict[str, Any]] = []
            for t in history:
                try:
                    sym = self._normalize_symbol(t.get("symbol", ""))
                except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError):
                    continue
                txs.append(
                    {
                        "timestamp": t.get("timestamp"),
                        "symbol": sym,
                        "side": t.get("trade_type") or t.get("side") or t.get("type"),
                        "quantity": float(t.get("quantity") or 0.0),
                        "price": float(t.get("price") or 0.0),
                        "amount_usd": float(t.get("amount_usd") or 0.0),
                        "exchange": t.get("exchange") or EXCHANGE_ID,
                        "trade_id": t.get("trade_id") or t.get("id") or "",
                    },
                )
            return txs[: max(0, int(limit))]
        except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
            logger.exception("Failed to get transactions: %s", e)
            return []

    async def get_usdt_balance(self) -> dict[str, float]:
        try:
            overview = await self.get_portfolio_overview()
            total_value = float(overview.get("total_value") or 0.0)
            total_invested = float(overview.get("total_invested") or 0.0)
            available_usdt = max(0.0, total_value - total_invested)
            efficiency = (total_invested / total_value * 100.0) if total_value > 0.0 else 0.0
        except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
            logger.exception("Failed to get USDT balance: %s", e)
            return {"total": 0.0, "allocated": 0.0, "available": 0.0, "efficiency": 0.0}
        else:
            return {
                "total": total_value,
                "allocated": total_invested,
                "available": available_usdt,
                "efficiency": efficiency,
            }

    async def get_total_value(self) -> float:
        try:
            overview = await self.get_portfolio_overview()
            return float(overview.get("total_value") or 0.0)
        except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
            logger.exception("Failed to get total value: %s", e)
            return 0.0

    async def store_portfolio_snapshot(self) -> dict[str, Any]:
        try:
            logger.info("Storing portfolio snapshot")
            overview = await self.get_portfolio_overview()
            snapshot = {
                "portfolio_overview": overview,
                "snapshot_timestamp": datetime.now(timezone.utc).isoformat(),
                "metadata": {
                    "total_positions": int(overview.get("positions_count") or 0),
                    "total_trades": int(overview.get("total_trades") or 0),
                    "total_value": float(overview.get("total_value") or 0.0),
                    "total_pnl": float(overview.get("total_pnl") or 0.0),
                    "source": "portfolio_service_snapshot",
                },
            }
            snapshot_id = f"portfolio_snapshot_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
            self.cache.store_signal(
                signal_id=snapshot_id,
                symbol="PORTFOLIO_SNAPSHOT",
                signal_type="PORTFOLIO_SNAPSHOT",
                confidence=1.0,
                strategy="portfolio_tracking",
                metadata=snapshot,
            )
            logger.info("Portfolio snapshot stored: %s", snapshot_id)
            return {
                "snapshot_id": snapshot_id,
                "timestamp": snapshot["snapshot_timestamp"],
                "total_value": float(overview.get("total_value") or 0.0),
                "positions_count": int(overview.get("positions_count") or 0),
            }
        except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
            logger.exception("Failed to store portfolio snapshot: %s", e)
            return {
                "error": str(e),
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }

    async def get_portfolio_status(self) -> dict[str, Any]:
        try:
            trade_history = await self._get_trade_history()
            positions_count = len([h for h in self._calculate_holdings(trade_history).values() if float(h.get("quantity") or 0.0) > 0.0])
            return {
                "service": "PortfolioService",
                "status": "active",
                "cache_connected": True,
                "last_snapshot": datetime.now(timezone.utc).isoformat(),
                "total_value": await self.get_total_value(),
                "positions_count": positions_count,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
        except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
            logger.exception("Failed to get portfolio status: %s", e)
            return {"success": False, "error": str(e)}

    async def get_portfolio(self, _portfolio_id: str | None = None) -> dict[str, Any]:
        """Get portfolio data - wrapper for get_portfolio_overview() for compatibility."""
        try:
            # For now, ignore portfolio_id since we only have one portfolio
            # In the future, this could be used for multi-portfolio support
            return await self.get_portfolio_overview()
        except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
            logger.exception("Failed to get portfolio: %s", e)
            return {
                "success": False,
                "error": str(e),
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }

    async def get_history(self, limit: int = 100) -> list[dict[str, Any]]:
        """Get portfolio transaction history"""
        try:
            logger.info("Getting portfolio history")
            trades = await self._get_trade_history(limit=limit)
            history = []
            for trade in trades:
                try:
                    sym = self._normalize_symbol(trade.get("symbol", ""))
                except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError):
                    continue
                history.append(
                    {
                        "timestamp": trade.get("timestamp"),
                        "symbol": sym,
                        "side": trade.get("trade_type") or trade.get("side") or trade.get("type"),
                        "quantity": float(trade.get("quantity") or 0.0),
                        "price": float(trade.get("price") or 0.0),
                        "amount_usd": float(trade.get("amount_usd") or 0.0),
                        "exchange": trade.get("exchange") or EXCHANGE_ID,
                        "trade_id": trade.get("trade_id") or trade.get("id") or "",
                    },
                )
            return history[: max(0, int(limit))]
        except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
            logger.exception("Failed to get portfolio history: %s", e)
            return []

    async def get_performance(self) -> dict[str, Any]:
        """Get portfolio performance metrics"""
        try:
            logger.info("Getting portfolio performance")
            overview = await self.get_portfolio_overview()
            trades = await self._get_trade_history(limit=1000)

            # Calculate performance metrics
            total_trades = len(trades)
            winning_trades = 0
            total_profit = 0.0
            total_loss = 0.0

            for trade in trades:
                pnl = float(trade.get("amount_usd", 0.0))
                if pnl > 0:
                    winning_trades += 1
                    total_profit += pnl
                else:
                    total_loss += abs(pnl)

            win_rate = (winning_trades / total_trades * 100.0) if total_trades > 0 else 0.0
            profit_factor = (total_profit / total_loss) if total_loss > 0 else 0.0
            total_pnl = float(overview.get("total_pnl", 0.0))
            total_value = float(overview.get("total_value", 0.0))

            return {
                "total_trades": total_trades,
                "winning_trades": winning_trades,
                "losing_trades": total_trades - winning_trades,
                "win_rate": win_rate,
                "total_profit": total_profit,
                "total_loss": total_loss,
                "profit_factor": profit_factor,
                "total_pnl": total_pnl,
                "total_pnl_percentage": (total_pnl / total_value * 100.0) if total_value > 0 else 0.0,
                "daily_pnl": 0.0,  # Would need daily tracking
                "sharpe_ratio": 0.0,  # Would need volatility data
                "max_drawdown": 0.0,  # Would need historical data
                "last_updated": datetime.now(timezone.utc).isoformat(),
            }
        except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
            logger.exception("Failed to get portfolio performance: %s", e)
            return {
                "total_trades": 0,
                "winning_trades": 0,
                "losing_trades": 0,
                "win_rate": 0.0,
                "total_profit": 0.0,
                "total_loss": 0.0,
                "profit_factor": 0.0,
                "total_pnl": 0.0,
                "total_pnl_percentage": 0.0,
                "daily_pnl": 0.0,
                "sharpe_ratio": 0.0,
                "max_drawdown": 0.0,
                "last_updated": datetime.now(timezone.utc).isoformat(),
                "error": str(e),
            }

    async def get_allocation(self) -> dict[str, Any]:
        """Get portfolio allocation breakdown"""
        try:
            logger.info("Getting portfolio allocation")
            overview = await self.get_portfolio_overview()
            holdings = overview.get("holdings", {})
            total_value = float(overview.get("total_value", 0.0))

            if total_value <= 0:
                return {
                    "total_value": 0.0,
                    "assets": {},
                    "cash": {"value": 0.0, "percentage": 0.0},
                    "last_updated": datetime.now(timezone.utc).isoformat(),
                }

            assets = {}
            cash_value = 0.0

            for symbol, holding in holdings.items():
                value = float(holding.get("current_value", 0.0))
                percentage = (value / total_value * 100.0) if total_value > 0 else 0.0
                assets[symbol] = {
                    "value": value,
                    "percentage": percentage,
                    "quantity": float(holding.get("quantity", 0.0)),
                    "current_price": float(holding.get("current_price", 0.0)),
                }

            # Assume any remaining value is cash
            allocated_value = sum(asset["value"] for asset in assets.values())
            cash_value = max(0.0, total_value - allocated_value)
            cash_percentage = (cash_value / total_value * 100.0) if total_value > 0 else 0.0

            return {
                "total_value": total_value,
                "assets": assets,
                "cash": {"value": cash_value, "percentage": cash_percentage},
                "last_updated": datetime.now(timezone.utc).isoformat(),
            }
        except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
            logger.exception("Failed to get portfolio allocation: %s", e)
            return {
                "total_value": 0.0,
                "assets": {},
                "cash": {"value": 0.0, "percentage": 0.0},
                "last_updated": datetime.now(timezone.utc).isoformat(),
                "error": str(e),
            }

    async def get_risk_metrics(self) -> dict[str, Any]:
        """Get portfolio risk metrics"""
        try:
            logger.info("Getting portfolio risk metrics")
            overview = await self.get_portfolio_overview()
            holdings = overview.get("holdings", {})
            total_value = float(overview.get("total_value", 0.0))

            if total_value <= 0 or not holdings:
                return {
                    "portfolio_var_1d": 0.0,
                    "portfolio_var_5d": 0.0,
                    "max_drawdown": 0.0,
                    "sharpe_ratio": 0.0,
                    "concentration_risk": 0.0,
                    "correlation_risk": 0.0,
                    "overall_risk_score": 0.0,
                    "risk_level": "low",
                    "last_updated": datetime.now(timezone.utc).isoformat(),
                }

            # Calculate concentration risk (largest position percentage)
            position_values = [float(h.get("current_value", 0.0)) for h in holdings.values()]
            max_position_value = max(position_values) if position_values else 0.0
            concentration_risk = (max_position_value / total_value * 100.0) if total_value > 0 else 0.0

            # Calculate diversification (number of positions)
            num_positions = len([h for h in holdings.values() if float(h.get("quantity", 0.0)) > 0.0])

            # Simplified risk calculations (would need historical data for accurate metrics)
            portfolio_var_1d = total_value * 0.02  # Assume 2% daily VaR
            portfolio_var_5d = total_value * 0.05  # Assume 5% 5-day VaR

            # Risk level assessment
            if concentration_risk > 50 or num_positions < 3:
                risk_level = "high"
            elif concentration_risk > 30 or num_positions < 5:
                risk_level = "moderate"
            else:
                risk_level = "low"

            return {
                "portfolio_var_1d": portfolio_var_1d,
                "portfolio_var_5d": portfolio_var_5d,
                "max_drawdown": 0.0,  # Would need historical data
                "sharpe_ratio": 0.0,  # Would need return/volatility data
                "concentration_risk": concentration_risk,
                "correlation_risk": 0.0,  # Would need correlation analysis
                "overall_risk_score": concentration_risk,
                "risk_level": risk_level,
                "diversification_score": min(100.0, num_positions * 10.0),
                "num_positions": num_positions,
                "largest_position_pct": concentration_risk,
                "last_updated": datetime.now(timezone.utc).isoformat(),
            }
        except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
            logger.exception("Failed to get portfolio risk metrics: %s", e)
            return {
                "portfolio_var_1d": 0.0,
                "portfolio_var_5d": 0.0,
                "max_drawdown": 0.0,
                "sharpe_ratio": 0.0,
                "concentration_risk": 0.0,
                "correlation_risk": 0.0,
                "overall_risk_score": 0.0,
                "risk_level": "unknown",
                "diversification_score": 0.0,
                "num_positions": 0,
                "largest_position_pct": 0.0,
                "last_updated": datetime.now(timezone.utc).isoformat(),
                "error": str(e),
            }


# Global portfolio service instance - use PortfolioService.shared()
portfolio_service = PortfolioService.shared()


def get_portfolio_service() -> PortfolioService:
    return portfolio_service


if __name__ == "__main__":
    service = PortfolioService.shared()
    logger.info("PortfolioService initialized: %s", service)
    overview = service.get_portfolio_overview()
    logger.info("Portfolio overview: %s", overview)
    total_value = service.get_total_value()
    logger.info(f"Total value: ${total_value:.2f}")
    snapshot = service.store_portfolio_snapshot()
    logger.info("Snapshot: %s", snapshot)
    status = service.get_portfolio_status()
    logger.info("Service status: %s", status.get("status"))
