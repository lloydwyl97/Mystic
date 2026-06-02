import asyncio
import json
import logging
import os
import sqlite3
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path
from typing import Any

import numpy as np

from backend.services.canonical_cache import canonical_cache as get_shared_cache
from backend.services.canonical_http_client import get_http_client

logger = logging.getLogger(__name__)

try:
    from backend.config.redis_config import get_redis_client
except (ImportError, ModuleNotFoundError, AttributeError, ValueError, TypeError, RuntimeError):
    get_redis_client = None

# Import from single source of truth
try:
    from backend.config.trading_universe import TRADING_SYMBOLS
except (ImportError, ModuleNotFoundError, AttributeError, ValueError, TypeError, RuntimeError) as e:
    msg = f"Failed to import trading_universe: {e}"
    raise RuntimeError(msg) from e

from backend.exchange_integration import BinanceAPI, OrderRequest
from backend.services.binance_rest_client import BinanceREST

BALANCE_DB = os.getenv("PORTFOLIO_BALANCE_DB", "./data/portfolio_balance.db")
BALANCE_INTERVAL = 1800
REBALANCE_THRESHOLD = 0.05
MAX_ASSETS = 10
RISK_FREE_RATE = 0.02
# Use BINANCEUS_BASE environment variable or default
BINANCEUS_REST_URL = os.getenv("BINANCEUS_BASE", "https://api.binance.us")
# Use TRADING_SYMBOLS from trading_universe (live data)
SYMBOLS = list(TRADING_SYMBOLS)


class BalanceDatabase:
    def __init__(self, db_path: str) -> None:
        self.db_path = db_path
        self.init_database()

    def init_database(self):
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        conn = None
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS portfolio_snapshots (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    total_value REAL NOT NULL,
                    cash_balance REAL NOT NULL,
                    asset_allocations TEXT NOT NULL,
                    risk_metrics TEXT NOT NULL,
                    rebalance_needed BOOLEAN NOT NULL,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            """,
            )
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS rebalance_actions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    action_type TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    quantity REAL NOT NULL,
                    price REAL NOT NULL,
                    reason TEXT NOT NULL,
                    performance_impact REAL,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            """,
            )
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS asset_performance (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    weight REAL NOT NULL,
                    return_1d REAL NOT NULL,
                    return_7d REAL NOT NULL,
                    return_30d REAL NOT NULL,
                    volatility REAL NOT NULL,
                    sharpe_ratio REAL NOT NULL,
                    correlation_btc REAL NOT NULL,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            """,
            )
            conn.commit()
        finally:
            if conn is not None:
                conn.close()

    def save_portfolio_snapshot(self, snapshot: dict):
        conn = None
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO portfolio_snapshots
                (timestamp, total_value, cash_balance, asset_allocations, risk_metrics, rebalance_needed)
                VALUES (?, ?, ?, ?, ?, ?)
            """,
                (
                    snapshot["timestamp"],
                    snapshot["total_value"],
                    snapshot["cash_balance"],
                    json.dumps(snapshot["asset_allocations"]),
                    json.dumps(snapshot["risk_metrics"]),
                    snapshot["rebalance_needed"],
                ),
            )
            conn.commit()
        finally:
            if conn is not None:
                conn.close()

    def save_rebalance_action(self, action: dict):
        conn = None
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO rebalance_actions
                (timestamp, action_type, symbol, quantity, price, reason, performance_impact)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    action["timestamp"],
                    action["action_type"],
                    action["symbol"],
                    action["quantity"],
                    action["price"],
                    action["reason"],
                    action.get("performance_impact", 0.0),
                ),
            )
            conn.commit()
        finally:
            if conn is not None:
                conn.close()

    def save_asset_performance(self, performance: dict):
        conn = None
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO asset_performance
                (timestamp, symbol, weight, return_1d, return_7d, return_30d, volatility, sharpe_ratio, correlation_btc)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    performance["timestamp"],
                    performance["symbol"],
                    performance["weight"],
                    performance["return_1d"],
                    performance["return_7d"],
                    performance["return_30d"],
                    performance["volatility"],
                    performance["sharpe_ratio"],
                    performance["correlation_btc"],
                ),
            )
            conn.commit()
        finally:
            if conn is not None:
                conn.close()


async def _http_get(path, params):
    client = await get_http_client()
    r = await client.get(f"{BINANCEUS_REST_URL}{path}", params=params, timeout=10)
    r.raise_for_status()
    return r.json()


def _redis_client() -> Any:
    """Return shared Redis client. Uses get_redis_client for sync; callers in async paths should use get_shared_redis_async."""
    if get_redis_client is None:
        return None
    try:
        return get_redis_client()
    except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError):
        return None


def get_symbol_filters(symbol: str) -> tuple[Decimal, Decimal, Decimal]:
    """Get symbol filters without calling exchangeInfo API to avoid CloudFront 403s"""
    # Use predefined filters for Top-10 symbols to avoid exchangeInfo calls
    # This prevents CloudFront 403 errors while maintaining functionality

    # Default filters for most symbols
    default_step = "0.00000001"
    default_min_qty = "0.00000001"
    default_min_notional = "10.0"

    # Symbol-specific overrides for Top-10
    symbol_overrides = {
        "BTCUSDT": {"step": "0.00001", "min_qty": "0.00001", "min_notional": "10.0"},
        "ETHUSDT": {"step": "0.00001", "min_qty": "0.00001", "min_notional": "10.0"},
        "ADAUSDT": {"step": "0.01", "min_qty": "0.01", "min_notional": "10.0"},
        "SOLUSDT": {"step": "0.001", "min_qty": "0.001", "min_notional": "10.0"},
        "DOGEUSDT": {"step": "1", "min_qty": "1", "min_notional": "10.0"},
        "XRPUSDT": {"step": "0.01", "min_qty": "0.01", "min_notional": "10.0"},
        "BCHUSDT": {"step": "0.001", "min_qty": "0.001", "min_notional": "10.0"},
        "LTCUSDT": {"step": "0.001", "min_qty": "0.001", "min_notional": "10.0"},
        "AVAXUSDT": {"step": "0.01", "min_qty": "0.01", "min_notional": "10.0"},
        "LINKUSDT": {"step": "0.01", "min_qty": "0.01", "min_notional": "10.0"},
    }

    # Get symbol-specific filters or use defaults
    symbol_upper = symbol.upper()
    filters = symbol_overrides.get(
        symbol_upper,
        {
            "step": default_step,
            "min_qty": default_min_qty,
            "min_notional": default_min_notional,
        },
    )

    return (
        Decimal(filters["step"]),
        Decimal(filters["min_qty"]),
        Decimal(filters["min_notional"]),
    )


def round_step(qty: float, step: Decimal) -> float:
    d = Decimal(str(qty))
    s = Decimal(str(step))
    return float((d // s) * s)


def get_current_price(symbol: str) -> float:
    if symbol not in SYMBOLS:
        msg = "symbol not allowed"
        raise ValueError(msg)

    # Use canonical_cache instead of direct API call
    try:
        shared_cache = get_shared_cache()

        # Try to get price from cache
        prices_data = asyncio.run(shared_cache.get_market_data("prices"))
        if prices_data and symbol in prices_data:
            price_info = prices_data[symbol]
            if isinstance(price_info, dict) and "price" in price_info:
                return float(price_info["price"])
            if isinstance(price_info, (int, float)):
                return float(price_info)

        # Try top10 data
        top10_data = asyncio.run(shared_cache.get_market_data("top10_data"))
        if top10_data and "prices" in top10_data and symbol in top10_data["prices"]:
            return float(top10_data["prices"][symbol])
    except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
        msg = f"Failed to get cached price for {symbol}: {e}"
        raise RuntimeError(msg) from e

    # If we reach here, no price was found
    msg = f"No cached price available for {symbol}"
    raise ValueError(msg)


def get_historical_returns(symbol: str, days: int = 30) -> np.ndarray:
    """Get daily historical returns for the given symbol from shared cache.
    Returns a numpy array of daily returns (fractional), most recent last.
    Falls back to zeros if data unavailable.
    """
    if symbol not in SYMBOLS:
        return np.array([], dtype=float)

    try:
        shared_cache = get_shared_cache()
        # Try several common keys that might contain historical price series
        possible_keys = ["historical_prices", "price_history", "history", "ohlcv", "klines", "candles"]
        prices = None
        for key in possible_keys:
            data = asyncio.run(shared_cache.get_market_data(key))
            if data:
                # Expect dict of symbol -> list of prices (ascending chronological)
                if isinstance(data, dict) and symbol in data and isinstance(data[symbol], (list, tuple)) and len(data[symbol]) > 1:
                    prices = list(data[symbol])
                    break
                # Some caches may store as { "prices": { symbol: [...] } }
                if isinstance(data, dict) and "prices" in data and isinstance(data["prices"], dict) and symbol in data["prices"] and isinstance(data["prices"][symbol], (list, tuple)):
                    prices = list(data["prices"][symbol])
                    break
        if not prices:
            # Try top10_data structure
            top10 = asyncio.run(shared_cache.get_market_data("top10_data"))
            if top10 and isinstance(top10, dict) and "history" in top10 and symbol in top10["history"]:
                prices = list(top10["history"][symbol])

        if not prices or len(prices) < 2:
            # Fallback to empty returns
            return np.zeros(days, dtype=float)

        # Use the last (days + 1) prices to compute 'days' returns
        prices = [float(p) for p in prices]
        prices_arr = np.array(prices, dtype=float)
        if prices_arr.size < 2:
            return np.zeros(days, dtype=float)
        # Compute daily returns: P_t / P_t-1 - 1
        daily_returns = prices_arr[1:] / prices_arr[:-1] - 1.0
        # Take the last 'days' returns
        if daily_returns.size >= days:
            res = daily_returns[-days:]
        else:
            # pad with zeros at the front to reach desired length
            pad = np.zeros(days - daily_returns.size, dtype=float)
            res = np.concatenate([pad, daily_returns])
        return res.astype(float)
    except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError):
        # On any error, return zeros to avoid crashing analyses
        return np.zeros(days, dtype=float)


def calculate_asset_metrics(symbol: str, weight: float) -> dict:
    """Calculate basic performance metrics for a single asset given its weight."""
    days = 30
    rets = get_historical_returns(symbol, days)
    ret_1d = float(rets[-1]) if rets.size >= 1 else 0.0
    if rets.size >= 7:
        try:
            ret_7d = float(np.prod(1.0 + rets[-7:]) - 1.0)
        except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError):
            ret_7d = float(np.sum(rets[-7:]))
    else:
        ret_7d = float(np.prod(1.0 + rets) - 1.0) if rets.size > 0 else 0.0
    if rets.size >= days:
        try:
            ret_30d = float(np.prod(1.0 + rets[-days:]) - 1.0)
        except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError):
            ret_30d = float(np.sum(rets[-days:]))
    else:
        ret_30d = float(np.prod(1.0 + rets) - 1.0) if rets.size > 0 else 0.0
    volatility = float(np.std(rets)) if rets.size > 0 else 0.0
    mean_ret = float(np.mean(rets)) if rets.size > 0 else 0.0
    sharpe = (mean_ret - RISK_FREE_RATE / 365.0) / volatility if volatility > 0 else 0.0

    # Correlation with BTC
    btc_rets = get_historical_returns("BTCUSDT", days)
    correlation = 0.0
    try:
        if rets.size > 1 and btc_rets.size > 1:
            min_len = min(rets.size, btc_rets.size)
            if min_len >= 2:
                correlation = float(np.corrcoef(rets[-min_len:], btc_rets[-min_len:])[0, 1])
    except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError):
        correlation = 0.0

    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "symbol": symbol,
        "weight": float(weight),
        "return_1d": ret_1d,
        "return_7d": ret_7d,
        "return_30d": ret_30d,
        "volatility": volatility,
        "sharpe_ratio": float(sharpe),
        "correlation_btc": float(correlation),
    }


def optimize_portfolio_weights(current_weights: dict, asset_metrics: list) -> dict:
    """Optimize weights to maximize Sharpe ratio with simple heuristic search."""
    symbols = [m["symbol"] for m in asset_metrics]
    n = len(symbols)
    if n == 0:
        return current_weights

    # expected returns (daily) from asset_metrics (use 30d returns converted to daily approx)
    # if return_30d is cumulative over 30 days, convert to daily mean approx
    exp_returns = np.array(
        [(m.get("return_30d", 0.0) / 30.0) if abs(m.get("return_30d", 0.0)) < 1e6 else 0.0 for m in asset_metrics],
        dtype=float,
    )

    # Build return series matrix for covariance estimation
    rets_matrix = []
    for s in symbols:
        r = get_historical_returns(s, 60)  # try 60 for better cov estimation
        if r.size == 0:
            # fallback to constant small returns
            r = np.zeros(60, dtype=float)
        rets_matrix.append(r)

    min_len = min([x.size for x in rets_matrix] or [0])
    if min_len < 2:
        # Not enough data; return normalized current weights for these symbols
        cw = {s: current_weights.get(s, 0.0) for s in symbols}
        total = sum(cw.values())
        if total <= 0:
            # equally weight
            return {s: float(1.0 / n) for s in symbols}
        return {s: float(w / total) for s, w in cw.items()}

    aligned = np.vstack([x[-min_len:] for x in rets_matrix])
    cov = np.cov(aligned)
    bounds_low = np.zeros(n)
    bounds_high = np.ones(n) * 0.3

    def project(w):
        w = np.minimum(np.maximum(w, bounds_low), bounds_high)
        s = np.sum(w)
        return np.ones(n) / n if s == 0 else w / s

    def sharpe(w):
        pr = float(np.dot(w, exp_returns))
        pv = float(np.sqrt(np.dot(w.T, np.dot(cov, w))))
        if pv <= 0:
            return -1e9
        return (pr - RISK_FREE_RATE / 365.0) / pv

    best_w = project(np.array([current_weights.get(s, 1.0 / n) for s in symbols], dtype=float))
    best_s = sharpe(best_w)
    rng = np.random.default_rng()
    for _ in range(500):
        w = rng.random(n)
        w = bounds_low + w * (bounds_high - bounds_low)
        w = project(w)
        s = sharpe(w)
        if s > best_s:
            best_s = s
            best_w = w
    step = 0.05
    for _ in range(200):
        improved = False
        for i in range(n):
            for d in (-step, step):
                w = best_w.copy()
                w[i] = w[i] + d
                w = project(w)
                s = sharpe(w)
                if s > best_s:
                    best_s = s
                    best_w = w
                    improved = True
        if not improved:
            break
    return {sym: float(w) for sym, w in zip(symbols, best_w, strict=False)}


def calculate_portfolio_risk_metrics(weights: dict, asset_metrics: list) -> dict:
    if not weights:
        return {
            "total_volatility": 0.0,
            "portfolio_sharpe": 0.0,
            "max_drawdown_estimate": 0.0,
            "var_95": 0.0,
            "diversification_score": 0.0,
        }
    sym_to_metrics = {m["symbol"]: m for m in asset_metrics}
    syms = [s for s in weights if s in sym_to_metrics]
    w_vec = np.array([weights[s] for s in syms], dtype=float)
    r_vec = np.array([sym_to_metrics[s]["return_30d"] for s in syms], dtype=float)
    rets_matrix = []
    for s in syms:
        r = get_historical_returns(s, 30)
        rets_matrix.append(r if r.size > 0 else np.zeros(30))
    min_len = min([x.size for x in rets_matrix] or [0])
    if min_len < 2:
        total_vol = float(np.sum(w_vec * np.array([sym_to_metrics[s]["volatility"] for s in syms])))
        pr = float(np.sum(w_vec * r_vec))
        vol = max(total_vol, 1e-9)
        ps = (pr - RISK_FREE_RATE / 365.0) / vol
    else:
        aligned = np.vstack([x[-min_len:] for x in rets_matrix])
        cov = np.cov(aligned)
        vol = float(np.sqrt(np.dot(w_vec.T, np.dot(cov, w_vec))))
        pr = float(np.dot(w_vec, r_vec))
        ps = (pr - RISK_FREE_RATE / 365.0) / vol if vol > 0 else 0.0
    mdd = float(vol * 2.5)
    var95 = float(vol * 1.645)
    concentration = float(np.sum(w_vec**2))
    div = float(1.0 - concentration)
    return {
        "total_volatility": float(vol),
        "portfolio_sharpe": float(ps),
        "max_drawdown_estimate": mdd,
        "var_95": var95,
        "diversification_score": div,
    }


def compute_weights_from_account(
    account_info: dict[str, Any],
) -> tuple[dict[str, float], float, float]:
    balances = account_info.get("balances", [])
    price_cache = {}

    def px(sym):
        if sym not in price_cache:
            price_cache[sym] = get_current_price(sym)
        return price_cache[sym]

    holdings = {}
    total_value = 0.0
    for s in SYMBOLS:
        base = s.replace("USDT", "")
        for b in balances:
            if b.get("asset") == base:
                amt = float(b.get("free", "0")) + float(b.get("locked", "0"))
                if amt > 0:
                    v = amt * px(s)
                    holdings[s] = v
                    total_value += v
                break
    usdt_balance = 0.0
    for b in balances:
        if b.get("asset") == "USDT":
            usdt_balance = float(b.get("free", "0")) + float(b.get("locked", "0"))
            total_value += usdt_balance
            break
    if total_value <= 0:
        return {}, 0.0, 0.0
    weights = {s: v / total_value for s, v in holdings.items()}
    if len(weights) > MAX_ASSETS:
        top = sorted(weights.items(), key=lambda x: x[1], reverse=True)[:MAX_ASSETS]
        weights = dict(top)
    return weights, float(total_value), float(usdt_balance)


def build_order_request(symbol: str, side: str, qty: float) -> OrderRequest:
    return OrderRequest(symbol=symbol, side=side, type="MARKET", quantity=qty)


def place_market_order(
    binance_api: BinanceAPI,
    rest_client: BinanceREST,
    symbol: str,
    side: str,
    qty: float,
) -> Any:
    try:
        req = build_order_request(symbol, side, qty)
        return asyncio.get_event_loop().run_until_complete(binance_api.place_order(req))
    except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError):
        return rest_client.order_market(symbol=symbol, side=side, quantity=qty)


def generate_rebalance_actions(
    current_weights: dict[str, float],
    target_weights: dict[str, float],
    total_value: float,
) -> list[dict[str, Any]]:
    actions = []
    for symbol in set(current_weights.keys()) | set(target_weights.keys()):
        if symbol not in SYMBOLS:
            continue
        cw = current_weights.get(symbol, 0.0)
        tw = target_weights.get(symbol, 0.0)
        dev = tw - cw
        if abs(dev) > REBALANCE_THRESHOLD:
            side = "BUY" if dev > 0 else "SELL"
            price = get_current_price(symbol)
            raw_qty = abs(dev) * total_value / max(price, 1e-12)
            step, min_qty, min_notional = get_symbol_filters(symbol)
            q = round_step(raw_qty, step)
            if q * price < float(min_notional):
                continue
            if q < float(min_qty):
                continue
            actions.append(
                {
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "action_type": side,
                    "symbol": symbol,
                    "quantity": float(q),
                    "price": float(price),
                    "reason": f"Rebalance to {tw:.3f}",
                    "performance_impact": 0.0,
                },
            )
    return actions


async def rebalance_once():
    db = BalanceDatabase(BALANCE_DB)
    binance = BinanceAPI()
    rest = BinanceREST()
    account = await binance.get_account_info()
    current_weights, total_value, cash_balance = compute_weights_from_account(account)
    if not current_weights or total_value <= 0:
        msg = "no live portfolio data available"
        raise RuntimeError(msg)
    asset_metrics = []
    for symbol, weight in current_weights.items():
        metrics = calculate_asset_metrics(symbol, weight)
        asset_metrics.append(metrics)
        db.save_asset_performance(metrics)
    target_weights = optimize_portfolio_weights(current_weights, asset_metrics)
    risk_metrics = calculate_portfolio_risk_metrics(target_weights, asset_metrics)
    actions = generate_rebalance_actions(current_weights, target_weights, total_value)
    for a in actions:
        place_market_order(binance, rest, a["symbol"], a["action_type"], a["quantity"])
        await binance.get_account_info()
        await binance.get_simple_positions()
    snapshot = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "total_value": total_value,
        "cash_balance": cash_balance,
        "asset_allocations": {"current": current_weights, "target": target_weights},
        "risk_metrics": risk_metrics,
        "rebalance_needed": len(actions) > 0,
    }
    db.save_portfolio_snapshot(snapshot)
    logger.info("[Balance] Portfolio Analysis Complete:")
    logger.info(f"[Balance] Total Value: ${total_value:,.2f}")
    logger.info(f"[Balance] Cash Balance: ${cash_balance:,.2f}")
    logger.info(f"[Balance] Portfolio Sharpe: {risk_metrics['portfolio_sharpe']:.3f}")
    logger.info(f"[Balance] Volatility: {risk_metrics['total_volatility']:.3f}")
    logger.info(f"[Balance] Diversification: {risk_metrics['diversification_score']:.3f}")
    logger.info("[Balance] Current Allocations:")
    for symbol, weight in current_weights.items():
        logger.info(f"[Balance] {symbol}: {weight:.3f} ({weight * 100:.1f}%)")
    logger.info("[Balance] Target Allocations:")
    for symbol, weight in target_weights.items():
        logger.info(f"[Balance] {symbol}: {weight:.3f} ({weight * 100:.1f}%)")
    if actions:
        logger.info(f"[Balance] Rebalancing Required: {len(actions)} actions")
        for a in actions:
            logger.info(f"[Balance] {a['action_type']} {a['symbol']}: {a['quantity']:.6f} @ {a['price']:.4f} - {a['reason']}")
    else:
        logger.info("[Balance] Portfolio is balanced")


async def run_forever():
    while True:
        try:
            await rebalance_once()
        except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
            logger.exception(f"[Balance] Enhanced balancing error: {e}")
        await asyncio.sleep(BALANCE_INTERVAL)
