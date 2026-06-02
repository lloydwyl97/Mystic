import asyncio
import json
import logging
import os
from datetime import datetime, timezone
from typing import Any

from ai_auto_learner import AIAutoLearner
from ai_mode_controller import AITradingController
from backup_utils import snapshot
from chart_generator import plot_performance_over_time
from daily_summary import send_daily_summary
from notifier import send_performance_alert, send_trade_alert
from stagnation_detector import check_performance_plateau, detect_stagnation
from strategy_tagger import get_strategy_confidence, tag_trade

import redis
from backend.config.redis_config import get_shared_redis_sync

# Import from single source of truth
try:
    from backend.config.trading_universe import EXCHANGE_ID, TRADING_SYMBOLS
except (ImportError, ModuleNotFoundError, AttributeError, ValueError, TypeError, RuntimeError) as e:
    msg = f"Failed to import trading_universe: {e}"
    raise RuntimeError(msg) from e

try:
    from backend.modules.market.binance_data_fetcher import _to_ccxt_symbol
except (ImportError, ModuleNotFoundError, AttributeError, ValueError, TypeError, RuntimeError) as e:
    msg = f"Failed to import _to_ccxt_symbol from binance_data_fetcher: {e}"
    raise RuntimeError(msg) from e

from simulation_logger import SimulationLogger

# All Live Data, No Fallback/Hardcoded Data
# Use first symbol from trading_universe (live data)
TRADING_SYMBOL = os.getenv("TRADING_SYMBOL") or (TRADING_SYMBOLS[0] if TRADING_SYMBOLS else None)
if not TRADING_SYMBOL:
    msg = "TRADING_SYMBOL environment variable is required - no fallback/hardcoded symbol"
    raise RuntimeError(msg)
TRADE_INTERVAL_SEC = int(os.getenv("TRADE_INTERVAL_SEC", "60"))
DAILY_TASK_INTERVAL_SEC = int(os.getenv("DAILY_TASK_INTERVAL_SEC", str(24 * 3600)))
TRADE_NOTIONAL_USD = float(os.getenv("TRADE_NOTIONAL_USD", "50"))


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _redis_client() -> redis.Redis:
    client = get_shared_redis_sync()
    if client is None:
        msg = "Shared Redis client unavailable"
        raise RuntimeError(msg)
    return client


def _decode(b: bytes | None) -> str | None:
    if b is None:
        return None
    try:
        return b.decode()
    except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError):
        return None


def _read_price(r: redis.Redis, symbol: str) -> float | None:
    raw = r.hget(f"price:{symbol}", "v")
    if not raw:
        return None
    s = _decode(raw)
    if not s:
        return None
    try:
        return float(s)
    except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError):
        return None


def _read_recent_prices(r: redis.Redis, symbol: str, max_points: int = 50) -> list[float]:
    # Prefer a close series if available; fall back to a rolling prices list
    keys = [f"closes:{symbol}", f"prices:{symbol}"]
    for key in keys:
        vals = r.lrange(key, -max_points, -1)
        out: list[float] = []
        for v in vals:
            s = _decode(v)
            if not s:
                continue
            try:
                out.append(float(s))
            except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError):
                continue
        if out:
            return out
    return []


def _read_signals(r: redis.Redis, symbol: str) -> dict[str, Any]:
    # Optional external signal source; if not present, return empty to avoid mock values
    raw = r.get(f"signals:{symbol}")
    s = _decode(raw) if raw else None
    if not s:
        return {}
    try:
        obj = json.loads(s)
        if not isinstance(obj, dict):
            return {}
        out: dict[str, float] = {}
        for k, v in obj.items():
            # accept ints/floats directly
            if isinstance(v, (int, float)):
                try:
                    out[str(k)] = float(v)
                except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError):
                    continue
            elif isinstance(v, str):
                # try to parse numeric strings, ignore non-numeric
                try:
                    fv = float(v)
                    if str(fv) not in ("nan", "inf", "-inf"):
                        out[str(k)] = fv
                except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError):
                    continue
    except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError):
        return {}
    else:
        return out


class AITradingIntegration:
    def __init__(self) -> None:
        self.controller = AITradingController()
        self.logger = SimulationLogger()
        self.learner = AIAutoLearner()
        self.is_running = False
        logging.basicConfig(level=logging.INFO)
        self.logger_instance = logging.getLogger(__name__)
        self.redis = _redis_client()

    async def start_trading_loop(self):
        self.is_running = True
        self.logger_instance.info(f"[{EXCHANGE_ID}] Starting AI Trading Integration on {TRADING_SYMBOL}")
        while self.is_running:
            try:
                detect_stagnation()
                # evaluate_and_adapt might be synchronous; call directly
                self.learner.evaluate_and_adapt()
                await self._live_trade_cycle(TRADING_SYMBOL)
                try:
                    plot_performance_over_time()
                except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
                    self.logger_instance.warning(f"plot_performance_over_time failed: {e}")
                await asyncio.sleep(TRADE_INTERVAL_SEC)
            except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
                self.logger_instance.exception(f"Error in trading loop: {e}")
                await asyncio.sleep(min(30, TRADE_INTERVAL_SEC))

    async def _live_trade_cycle(self, symbol: str):
        symbol = symbol.replace("/", "").upper()
        current_price = _read_price(self.redis, symbol)
        if current_price is None or current_price <= 0:
            self.logger_instance.warning(f"[{EXCHANGE_ID}] No live price for {symbol}; skipping cycle")
            return

        recent_prices = _read_recent_prices(self.redis, symbol, max_points=60)
        if len(recent_prices) < 5:
            self.logger_instance.warning(f"[{EXCHANGE_ID}] Insufficient recent prices for {symbol}; skipping cycle")
            return

        strategy = tag_trade(current_price, recent_prices)
        external_signals = _read_signals(self.redis, symbol)

        pattern = analyze_trade_pattern(recent_prices)
        confidence = get_strategy_confidence(pattern, external_signals)

        # Edge estimate from recent momentum (live, not mocked)
        returns = []
        for i in range(1, min(6, len(recent_prices))):
            prev = recent_prices[-i - 1]
            cur = recent_prices[-i]
            if prev > 0:
                returns.append((cur - prev) / prev)
        mean_ret = (sum(returns) / len(returns)) if returns else 0.0
        expected_edge_usd = float(TRADE_NOTIONAL_USD * mean_ret)

        if self.controller.should_execute_trade(expected_edge_usd):
            self.logger_instance.info(f"[{EXCHANGE_ID}] Executing trade: {symbol} BUY @ ${current_price:.6f}, edge ${expected_edge_usd:.4f}")
            try:
                send_trade_alert(symbol, "BUY", current_price, expected_edge_usd)
            except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
                self.logger_instance.warning(f"send_trade_alert failed: {e}")
            try:
                if confidence > 0.8:
                    send_performance_alert(expected_edge_usd, 1)
            except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
                self.logger_instance.warning(f"send_performance_alert failed: {e}")
        else:
            try:
                self.logger.log_trade(
                    symbol=symbol,
                    action="BUY",
                    price=current_price,
                    confidence=confidence,
                    simulated_profit=expected_edge_usd,
                    strategy=strategy,
                    mystic_signals=json.dumps(external_signals) if external_signals else "{}",
                )
            except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
                # Fallback logging if SimulationLogger doesn't accept the above signature
                self.logger_instance.debug(f"SimulationLogger.log_trade failed: {e}")
            self.logger_instance.info(f"[{EXCHANGE_ID}] Logged candidate trade: {symbol} BUY @ ${current_price:.6f}")

    def stop_trading(self):
        self.is_running = False
        self.logger_instance.info(f"[{EXCHANGE_ID}] AI Trading Integration stopped at {datetime.now(timezone.utc).isoformat()}")

    async def run_daily_tasks(self):
        while self.is_running:
            try:
                send_daily_summary()
                snapshot()
                check_performance_plateau()
                await asyncio.sleep(DAILY_TASK_INTERVAL_SEC)
            except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
                self.logger_instance.exception(f"Error in daily tasks: {e}")
                await asyncio.sleep(3600)


def analyze_trade_pattern(prices: list[float]) -> dict[str, str]:
    if len(prices) < 3:
        return {"pattern": "insufficient_data"}
    current_price = prices[-1]
    prev_price = prices[-2]
    if prev_price == 0:
        return {"pattern": "insufficient_data"}
    price_change_pct = ((current_price - prev_price) / prev_price) * 100.0
    vol = (max(prices[-10:]) - min(prices[-10:])) / current_price if len(prices) >= 10 and current_price > 0 else 0.0
    return {
        "trend": "up" if price_change_pct > 0 else "down" if price_change_pct < 0 else "sideways",
        "strength": "strong" if abs(price_change_pct) > 2 else "weak",
        "volatility": "high" if vol > 0.1 else "low",
    }
