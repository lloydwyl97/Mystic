"""
Chart Profit - Profit Tracker Chart

Renders profit charts from trade logs (Python 3.12).
No external data sources. No CoinGecko/Coinbase/etc.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib.dates as mdates
import matplotlib.pyplot as plt

# Optional imports - try at top level
try:
    from backend.utils.cache_guard import CacheGuard
except (ImportError, ModuleNotFoundError, AttributeError, ValueError, TypeError, RuntimeError):
    CacheGuard = None

# Import from single source of truth
try:
    from backend.config.trading_universe import (
        EXCHANGE_ID,
        TRADING_SYMBOLS,
    )
except (ImportError, ModuleNotFoundError, AttributeError, ValueError, TypeError, RuntimeError) as e:
    msg = f"Failed to import trading universe: {e}"
    raise RuntimeError(msg) from e

logger = logging.getLogger("chart_profit")


def _safe_parse_dt(ts: Any) -> datetime:
    if isinstance(ts, datetime):
        return ts
    if isinstance(ts, str):
        try:
            return datetime.fromisoformat(ts.replace("Z", "+00:00"))
        except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError):
            pass
    return datetime.now(timezone.utc)


def _load_json(path: str, default) -> Any:
    try:
        path_obj = Path(path)
        if path_obj.exists():
            with path_obj.open(encoding="utf-8") as f:
                return json.load(f)
    except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
        logger.exception("Error reading %s: %s", path, e)
    return default


def load_trade_data() -> list[dict[str, Any]]:
    trades: list[dict[str, Any]] = []

    active_trades = _load_json("active_trades.json", {})
    for symbol, trade in (active_trades or {}).items():
        try:
            trades.append(
                {
                    "symbol": symbol,
                    "type": "BUY",
                    "amount": float(trade.get("amount", 0) or 0),
                    "price": float(trade.get("buy_price", 0) or 0),
                    "timestamp": trade.get("timestamp"),
                },
            )
        except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
            logger.exception("Bad active trade for %s: %s", symbol, e)

    history = _load_json("trade_history.json", [])
    for t in history or []:
        try:
            trades.append(
                {
                    "symbol": t.get("symbol"),
                    "type": str(t.get("type", "")).upper(),
                    "amount": float(t.get("amount", 0) or 0),
                    "price": float(t.get("price", 0) or 0),
                    "timestamp": t.get("timestamp"),
                },
            )
        except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
            logger.exception("Bad history trade: %s", e)

    trades.sort(key=lambda x: _safe_parse_dt(x.get("timestamp")))
    return trades


def _compute_pnl_series(
    trades: list[dict[str, Any]],
) -> tuple[list[datetime], list[float], list[float]]:
    dates: list[datetime] = []
    values: list[float] = []
    cum: list[float] = []
    running = 0.0

    for t in trades:
        ts = _safe_parse_dt(t.get("timestamp"))
        typ = str(t.get("type", "")).upper()
        amt = float(t.get("amount", 0) or 0)
        px = float(t.get("price", 0) or 0)

        if amt <= 0 or px <= 0 or typ not in ("BUY", "SELL"):
            continue

        val = amt * px
        if typ == "BUY":
            running -= val
        else:
            running += val

        dates.append(ts)
        values.append(val)
        cum.append(running)

    return dates, values, cum


def plot_profits(output_file: str = "profit_chart.png") -> bool:
    try:
        trades = load_trade_data()
        if not trades:
            logger.warning("No trade data available for plotting")
            return False

        dates, values, cumulative_pnl = _compute_pnl_series(trades)
        if not dates:
            logger.warning("No valid trade data for plotting")
            return False

        _fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

        ax1.plot(dates, values, "o-", alpha=0.7, linewidth=1)
        ax1.set_title("Individual Trade Values", fontsize=14, fontweight="bold")
        ax1.set_ylabel("Trade Value (USD)", fontsize=12)
        ax1.grid(True, alpha=0.3)
        ax1.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
        ax1.xaxis.set_major_locator(mdates.DayLocator(interval=1))
        plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45)

        ax2.plot(dates, cumulative_pnl, "-", linewidth=2, label="Cumulative P&L")
        ax2.axhline(y=0, color="r", linestyle="--", alpha=0.5, label="Break-even")
        ax2.set_title("Cumulative Profit & Loss", fontsize=14, fontweight="bold")
        ax2.set_ylabel("P&L (USD)", fontsize=12)
        ax2.set_xlabel("Date", fontsize=12)
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
        ax2.xaxis.set_major_locator(mdates.DayLocator(interval=1))
        plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45)

        plt.tight_layout()
        plt.savefig(output_file, dpi=300, bbox_inches="tight")
        plt.close()
    except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
        logger.exception("Error plotting profits: %s", e)
        return False
    else:
        logger.info("Profit chart saved to %s", output_file)
        return True


def plot_market_performance(output_file: str = "market_performance.png") -> bool:
    """
    Uses cached prices/klines if available (Redis via CacheGuard producers).
    Symbols come from TOP_SYMBOLS env (USDT pairs). Falls back gracefully.
    """
    try:
        # Local import to keep this module standalone

        # Use trading_universe symbols (live data) as default
        symbols_env = os.getenv("TOP_SYMBOLS", ",".join(TRADING_SYMBOLS))
        pairs = [s.strip().upper() for s in symbols_env.split(",") if s.strip()]
        bases = [p[:-4] for p in pairs if p.endswith("USDT")]

        cg = None
        with contextlib.suppress(ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError):
            cg = asyncio_run_cacheguard()

        prices: list[float] = []
        changes: list[float] = []

        if cg:
            for base, pair in zip(bases, pairs, strict=False):
                px = asyncio_get_price(cg, base)
                if px is not None and px > 0:
                    prices.append(px)
                ch = asyncio_get_change_24h(cg, pair)
                if ch is not None:
                    changes.append(ch)

        if not prices:
            logger.warning("No cached market data available")
            return False

        _fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        ax1.hist(prices, bins=20, alpha=0.8, edgecolor="black")
        ax1.set_title("Price Distribution (Cached)", fontsize=14, fontweight="bold")
        ax1.set_xlabel("Price (USD)", fontsize=12)
        ax1.set_ylabel("Count", fontsize=12)
        ax1.grid(True, alpha=0.3)

        if changes:
            ax2.hist(changes, bins=20, alpha=0.8, edgecolor="black")
            ax2.set_title("24h Change Distribution (Cached)", fontsize=14, fontweight="bold")
            ax2.set_xlabel("24h Change (%)", fontsize=12)
            ax2.set_ylabel("Count", fontsize=12)
            ax2.grid(True, alpha=0.3)
            ax2.axvline(x=0, color="red", linestyle="--", alpha=0.7)
        else:
            ax2.text(
                0.5,
                0.5,
                "No 24h change data",
                ha="center",
                va="center",
                transform=ax2.transAxes,
            )
            ax2.set_axis_off()

        plt.tight_layout()
        plt.savefig(output_file, dpi=300, bbox_inches="tight")
        plt.close()
    except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
        logger.exception("Error plotting market performance: %s", e)
        return False
    else:
        logger.info("Market performance chart saved to %s", output_file)
        return True


def asyncio_run_cacheguard():
    if CacheGuard is None:
        msg = "CacheGuard not available"
        raise RuntimeError(msg)

    async def _make():
        return await CacheGuard.create()

    return asyncio.run(_make())


def asyncio_get_price(cg, base: str) -> float | None:
    async def _get():
        return await cg.get_price_cached(base, freshness_sec=300)

    try:
        v = asyncio.run(_get())
        return float(v) if v is not None else None
    except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError):
        return None


def asyncio_get_change_24h(cg, pair: str) -> float | None:
    """
    Tries to compute % change from klines:{PAIR}:1d if present:
    expects list of OHLCV rows; uses last close vs prior close.
    """

    async def _get():
        return await cg.get_klines_cached(pair, "1d", n=2)

    try:
        kl = asyncio.run(_get())
        if isinstance(kl, list) and len(kl) >= 2:
            # Accept [open, high, low, close, ...] or exchange-style arrays
            def close(row):
                try:
                    if isinstance(row, (list, tuple)):
                        # typical: [ts, open, high, low, close, vol] OR [open, high, low, close, ...]
                        if len(row) >= 5 and isinstance(row[4], (int, float)):
                            return float(row[4])
                        if len(row) >= 4 and isinstance(row[3], (int, float)):
                            return float(row[3])
                    if isinstance(row, dict):
                        for k in ("close", "c"):
                            if k in row:
                                return float(row[k])
                except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError):
                    pass
                return None

            c_prev = close(kl[-2])
            c_last = close(kl[-1])
            if c_prev and c_last and c_prev > 0:
                return (c_last - c_prev) / c_prev * 100.0
    except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError):
        pass
    return None


def generate_trading_report() -> dict[str, Any]:
    try:
        trades = load_trade_data()
        total_trades = len([t for t in trades if t.get("type") in ("BUY", "SELL")])
        buy_trades = len([t for t in trades if t.get("type") == "BUY"])
        sell_trades = len([t for t in trades if t.get("type") == "SELL"])

        buy_values = [float(t["amount"]) * float(t["price"]) for t in trades if t.get("type") == "BUY"]
        avg_trade_size = sum(buy_values) / len(buy_values) if buy_values else 0.0

        _, _, cum = _compute_pnl_series(trades)
        realized_cash_pnl = cum[-1] if cum else 0.0

        return {
            "summary": {
                "realized_cash_flow_pnl_usd": round(realized_cash_pnl, 2),
            },
            "active_trades_count": len([t for t in trades if t.get("type") == "BUY"]) - len([t for t in trades if t.get("type") == "SELL"]),
            "total_trades": total_trades,
            "buy_trades": buy_trades,
            "sell_trades": sell_trades,
            "avg_trade_size": round(avg_trade_size, 2),
            "completion_rate": (round(sell_trades / buy_trades * 100.0, 2) if buy_trades > 0 else 0.0),
            "generated_at": datetime.now(timezone.utc).isoformat(),
        }
    except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
        logger.exception("Error generating trading report: %s", e)
        return {"error": str(e), "generated_at": datetime.now(timezone.utc).isoformat()}


if __name__ == "__main__":
    ok1 = plot_profits()
    ok2 = plot_market_performance()
    logger.error(f"profit_chart: {'ok' if ok1 else 'fail'}")
    logger.error(f"market_performance: {'ok' if ok2 else 'fail'}")
