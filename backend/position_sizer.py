import asyncio
import json
import logging
import os
import statistics
import urllib.parse
import urllib.request
from typing import Any

from backend.micro_account_manager import get_micro_account_manager
from backend.services.binance_rest_client import BinanceRestClient

# Import from single source of truth
try:
    from backend.config.trading_universe import TRADING_SYMBOLS
except (ImportError, ModuleNotFoundError, AttributeError, ValueError, TypeError, RuntimeError) as e:
    msg = f"Failed to import trading_universe: {e}"
    raise RuntimeError(msg) from e

from backend.services.canonical_cache import canonical_cache as get_shared_cache

logger = logging.getLogger(__name__)

# Use BINANCEUS_BASE environment variable or default
BINANCEUS_REST_URL = os.getenv("BINANCEUS_BASE", "https://api.binance.us")
# Use TRADING_SYMBOLS from trading_universe (live data)
SYMBOLS = list(TRADING_SYMBOLS)

# Position sizing constants
DEFAULT_VOLATILITY = 0.02  # 2% default volatility
DEFAULT_RISK_PER_TRADE = 0.01  # 1% risk per trade
MIN_POSITION_SIZE = 0.01  # Minimum position size (1% of capital)


def _http_get(path: str, params: dict[str, Any]) -> Any:
    qs = urllib.parse.urlencode(params)
    url = f"{BINANCEUS_REST_URL}{path}?{qs}"
    with urllib.request.urlopen(url, timeout=10) as resp:
        data = resp.read().decode("utf-8")
    return json.loads(data)


def get_current_price(symbol: str) -> float:
    if symbol not in SYMBOLS:
        msg = "symbol not allowed"
        raise ValueError(msg)

    try:
        shared_cache = get_shared_cache()

        prices_data = asyncio.run(shared_cache.get_market_data("prices"))
        if prices_data and symbol in prices_data:
            price_info = prices_data[symbol]
            if isinstance(price_info, dict) and "price" in price_info:
                return float(price_info["price"])
            if isinstance(price_info, (int, float)):
                return float(price_info)

        top10_data = asyncio.run(shared_cache.get_market_data("top10_data"))
        if top10_data and "prices" in top10_data and symbol in top10_data["prices"]:
            return float(top10_data["prices"][symbol])
    except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
        msg = f"Failed to get cached price for {symbol}: {e}"
        raise RuntimeError(msg) from e

    # If we reach here, no price was found
    msg = f"No cached price available for {symbol}"
    raise ValueError(msg)


def fetch_klines_closes(symbol: str, interval: str = "1d", limit: int = 30) -> list[float]:
    if symbol not in SYMBOLS:
        msg = "symbol not allowed"
        raise ValueError(msg)

    try:
        shared_cache = get_shared_cache()

        # Try to read some cached market data if available, but proceed to fetch live data regardless.
        try:
            prices_data = asyncio.run(shared_cache.get_market_data("prices"))
            # We don't strictly require cached data to proceed; log if missing.
            if not prices_data or symbol not in prices_data:
                logger.debug(f"No cached price entry for {symbol}; attempting live fetch.")
        except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError):
            logger.debug("Failed to read cached prices; attempting live fetch.")

        try:
            client = BinanceRestClient()
            klines_data = asyncio.run(client.get_klines(symbol, interval, limit=min(limit, 365)))

            result = [float(kline[4]) for kline in klines_data] if klines_data and len(klines_data) >= 1 else []
        except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
            logger.exception(f"Failed to fetch live historical data for {symbol}: {e}")
            return []
        else:
            return result
    except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
        msg = f"Failed to get cached kline data for {symbol}: {e}"
        raise RuntimeError(msg) from e


def compute_volatility_from_closes(closes: list[float]) -> float:
    returns = []
    for i in range(1, len(closes)):
        r = (closes[i] / closes[i - 1]) - 1.0
        returns.append(r)
    if len(returns) < 2:
        return max(abs(returns[0]), 1e-6) if returns else 1e-6
    vol = statistics.stdev(returns)
    return max(vol, 1e-6)


def calculate_position_size(
    capital_usdt: float,
    _strategy_win_rate: float,  # Unused: part of interface, handled by micro_account_manager
    _volatility: float = DEFAULT_VOLATILITY,  # Unused: part of interface, handled by micro_account_manager
    _risk_per_trade: float = DEFAULT_RISK_PER_TRADE,  # Unused: part of interface, handled by micro_account_manager
) -> float:
    mgr = get_micro_account_manager()
    mgr.current_budget = capital_usdt
    scaled_params = mgr.get_scaled_parameters()
    return scaled_params.get("max_position_size", 0.0)


def get_strategy_volatility(_strategy_name: str, symbol: str, lookback_days: int = 30) -> float:
    closes = fetch_klines_closes(symbol, "1d", int(lookback_days))
    return compute_volatility_from_closes(closes)


def size_position_for_strategy(strategy_name: str, symbol: str, capital_usdt: float, win_rate: float) -> float:
    vol = get_strategy_volatility(strategy_name, symbol)
    return calculate_position_size(capital_usdt, win_rate, vol)


class PositionSizer:
    def __init__(self) -> None:
        self.default_risk_per_trade = 0.02
        self.max_position_size = 0.15
        self.min_position_size = MIN_POSITION_SIZE
        self.volatility_lookback_days = 30
        self.kelly_criterion_max = 0.25

    def calculate_kelly_criterion(self, win_rate: float, avg_win: float, avg_loss: float) -> float:
        # Prevent division by zero
        if avg_win == 0:
            return 0.0
        # If avg_loss is zero, the formula simplifies (no downside)
        k = win_rate * avg_win / avg_win if avg_loss == 0 else (win_rate * avg_win - (1 - win_rate) * avg_loss) / avg_win
        return max(0.0, min(k, self.kelly_criterion_max))

    def calculate_position_size(
        self,
        symbol: str,
        capital_usdt: float,
        strategy_win_rate: float,
        risk_per_trade: float = 0.02,
    ) -> float:
        if symbol not in SYMBOLS:
            msg = "symbol not allowed"
            raise ValueError(msg)
        if capital_usdt <= 10000:
            price = get_current_price(symbol)
            mgr = get_micro_account_manager()
            position_data = mgr.calculate_position_size("GENERIC", strategy_win_rate, price)
            return position_data.get("position_value", 0.0)
        vol = get_strategy_volatility("GENERIC", symbol, self.volatility_lookback_days)
        size = calculate_position_size(capital_usdt, strategy_win_rate, vol, risk_per_trade)
        cap_min = capital_usdt * self.min_position_size
        cap_max = capital_usdt * self.max_position_size
        return max(min(size, cap_max), cap_min)
