import asyncio
import contextlib
import json
import logging
import random
import time
from collections import deque
from collections.abc import Sequence
from dataclasses import dataclass
from statistics import pstdev
from typing import Any

import websockets

from backend.services.task_manager import task_manager

try:
    import redis
except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError):
    redis = None

# Import from single source of truth
try:
    from backend.config.trading_universe import TRADING_SYMBOLS
except (ImportError, ModuleNotFoundError, AttributeError, ValueError, TypeError, RuntimeError) as e:
    msg = f"Failed to import trading_universe: {e}"
    raise RuntimeError(msg) from e

logger = logging.getLogger(__name__)

BINANCE_US_STREAM_BASE = "wss://stream.binance.us:9443/stream?streams="
# Use TRADING_SYMBOLS from trading_universe (live data)
SYMBOLS = list(TRADING_SYMBOLS)
MAX_PRICE_POINTS = 500


def tag_trade(price: float, recent_prices: Sequence[float], lookback: int = 50) -> str:
    if not recent_prices:
        return "unknown"
    window = list(recent_prices)[-lookback:]
    if len(window) < 2:
        return "unknown"
    prev_last = window[-2]
    prev_window = window[:-1]
    hi = max(prev_window)
    lo = min(prev_window)
    if price > hi:
        return "breakout"
    if price < lo:
        return "breakdown"
    if price > prev_last:
        return "uptrend"
    if price < prev_last:
        return "downtrend"
    return "consolidation"


def _pct_change(a: float, b: float) -> float:
    if b == 0:
        return 0.0
    return (a - b) / b * 100.0


def _rolling_rsi(values: Sequence[float], period: int = 14) -> float | None:
    if len(values) < period + 1:
        return None
    gains = 0.0
    losses = 0.0
    for i in range(-period, 0):
        change = values[i] - values[i - 1]
        if change >= 0:
            gains += change
        else:
            losses -= change
    avg_gain = gains / period
    avg_loss = losses / period
    if avg_loss == 0:
        return 100.0 if gains > 0 else 50.0
    rs = avg_gain / avg_loss
    return 100.0 - (100.0 / (1.0 + rs))


def _ema(values: Sequence[float], period: int) -> float | None:
    if len(values) < period:
        return None
    k = 2 / (period + 1)
    ema = values[-period]
    for v in values[-period + 1 :]:
        ema = (v - ema) * k + ema
    return float(ema)


def analyze_trade_pattern(prices: Sequence[float], _volumes: Sequence[float] | None = None) -> dict[str, Any]:
    n = len(prices)
    if n < 3:
        return {"pattern": "insufficient_data"}
    current_price = float(prices[-1])
    prev_price = float(prices[-2])
    last_change_pct = _pct_change(current_price, prev_price)
    trend = "up" if last_change_pct > 0 else ("down" if last_change_pct < 0 else "sideways")
    strength = "strong" if abs(last_change_pct) > 2.0 else "weak"
    vol_returns: list[float] = []
    for i in range(max(1, n - 20), n):
        chg = _pct_change(prices[i], prices[i - 1])
        vol_returns.append(chg)
    vol_sigma = pstdev(vol_returns) if len(vol_returns) > 1 else 0.0
    volatility = "high" if vol_sigma > 1.0 else "low"
    rsi14 = _rolling_rsi(prices, 14)
    ema_fast = _ema(prices, 9)
    ema_slow = _ema(prices, 21)
    if ema_fast is not None and ema_slow is not None:
        # Compare EMA difference to threshold (0.0005 = 0.05%)
        threshold = 0.0005
        ma_slope = "flat" if (abs(ema_fast - ema_slow) / ema_slow if ema_slow else threshold > 0.0) else "rising" if ema_fast > ema_slow else "falling"
    else:
        ma_slope = "flat"
    recent_window = prices[-50:] if n >= 50 else prices
    recent_range_pct = ((max(recent_window) - min(recent_window)) / current_price) * 100.0 if current_price else 0.0
    return {
        "trend": trend,
        "strength": strength,
        "volatility": volatility,
        "rsi14": round(rsi14, 2) if rsi14 is not None else None,
        "ma_slope": ma_slope,
        "last_change_pct": round(last_change_pct, 3),
        "recent_range_pct": round(recent_range_pct, 3),
    }


def get_strategy_confidence(pattern: dict[str, Any], mystic_signals: dict[str, float] | None = None) -> float:
    base_confidence = 0.5
    trend = pattern.get("trend")
    strength = pattern.get("strength")
    if trend == "up" and strength == "strong":
        base_confidence += 0.2
    elif trend == "down" and strength == "strong":
        base_confidence -= 0.1
    rsi = pattern.get("rsi14")
    if isinstance(rsi, (int, float)):
        if 45 <= rsi <= 60:
            base_confidence += 0.05
        elif rsi >= 75 or rsi <= 25:
            base_confidence -= 0.05
    if mystic_signals:
        base_confidence += 0.15 * float(mystic_signals.get("tesla_369", 0.0))
        base_confidence += 0.10 * float(mystic_signals.get("faerie_star", 0.0))
        base_confidence += 0.20 * float(mystic_signals.get("mystic_alignment_score", 0.0))
    if pattern.get("volatility") == "high":
        base_confidence -= 0.03
    return max(0.0, min(1.0, round(base_confidence, 4)))


@dataclass
class MysticSignalSource:
    host: str = "127.0.0.1"
    port: int = 6379
    db: int = 0
    _client: Any = None

    def connect(self) -> None:
        if redis is None:
            return
        try:
            # Use shared Redis connection pool to prevent connection exhaustion
            from backend.config.redis_config import get_shared_redis_sync

            self._client = get_shared_redis_sync()
            self._client.ping()
        except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError):
            self._client = None

    def get_latest(self) -> dict[str, float]:
        if not self._client:
            return {}
        try:
            keys = [
                "mystic:signal:tesla_369",
                "mystic:signal:faerie_star",
                "mystic:signal:mystic_alignment_score",
            ]
            out: dict[str, float] = {}
            for k in keys:
                v = self._client.get(k)
                if v is not None:
                    with contextlib.suppress(ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError):
                        out[k.split(":")[-1]] = float(v)
        except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError):
            return {}
        else:
            return out

    def publish(self, channel: str, message: str) -> None:
        if not self._client:
            return
        with contextlib.suppress(ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError):
            self._client.publish(channel, message)


class LiveKlineWatcher:
    def __init__(self, symbols: Sequence[str], max_points: int = MAX_PRICE_POINTS) -> None:
        self.symbols = [s.upper() for s in symbols]
        self.streams = "/".join(f"{s.lower()}@kline_1m" for s in self.symbols)
        self.url = BINANCE_US_STREAM_BASE + self.streams
        self.prices: dict[str, deque[float]] = {s: deque(maxlen=max_points) for s in self.symbols}
        self.volumes: dict[str, deque[float]] = {s: deque(maxlen=max_points) for s in self.symbols}
        self._stop = False
        self.mystic = MysticSignalSource()
        self.mystic.connect()

    async def run(self) -> None:
        backoff = 1.0
        max_backoff = 30.0
        while not self._stop:
            try:
                async with websockets.connect(self.url, ping_interval=20, ping_timeout=20, max_queue=1024) as ws:
                    backoff = 1.0
                    async for raw in ws:
                        try:
                            msg = json.loads(raw)
                            self._handle_message(msg)
                        except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError):
                            pass
            except asyncio.CancelledError:
                raise
            except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError):
                await asyncio.sleep(backoff)
                backoff = min(max_backoff, backoff * 1.7 + random.random())

    def stop(self) -> None:
        self._stop = True

    def _handle_message(self, msg: dict[str, Any]) -> None:
        data = msg.get("data") or msg
        if not data:
            return
        if data.get("e") != "kline":
            return
        k = data.get("k", {})
        if not bool(k.get("x")):
            return
        symbol = str(data.get("s") or k.get("s") or "").upper()
        if symbol not in self.prices:
            return
        try:
            close_price = float(k.get("c"))
        except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError):
            return
        vol = None
        if "q" in k:
            try:
                vol = float(k["q"])
            except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError):
                vol = None
        elif "v" in k:
            try:
                vol = float(k["v"])
            except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError):
                vol = None
        pp = self.prices[symbol]
        vv = self.volumes[symbol]
        pp.append(close_price)
        if vol is not None:
            vv.append(vol)
        trade_tag = tag_trade(close_price, pp)
        pattern = analyze_trade_pattern(pp, vv if vv else None)
        mystic_vals = self.mystic.get_latest()
        confidence = get_strategy_confidence(pattern if isinstance(pattern, dict) else {}, mystic_vals)
        action = "HOLD"
        if isinstance(pattern, dict):
            if pattern.get("trend") == "up" and pattern.get("strength") == "strong" and confidence >= 0.65:
                action = "BUY"
            elif pattern.get("trend") == "down" and pattern.get("strength") == "strong" and confidence <= 0.45:
                action = "SELL"
        payload = {
            "ts": int(time.time() * 1000),
            "symbol": symbol,
            "price": close_price,
            "tag": trade_tag,
            "pattern": pattern,
            "confidence": confidence,
            "action": action,
        }
        line = json.dumps(payload, separators=(",", ":"))
        logger.info(line)
        self.mystic.publish("mystic:stream:trade_insights", line)


async def _amain() -> None:
    watcher = LiveKlineWatcher(SYMBOLS)
    task = await task_manager.create_task(watcher.run(), name="strategy_tagger:run")
    try:
        await task
    except KeyboardInterrupt:
        watcher.stop()
        task.cancel()
        with contextlib.suppress(Exception, asyncio.CancelledError):
            await task


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    with contextlib.suppress(KeyboardInterrupt):
        asyncio.run(_amain())


if __name__ == "__main__":
    main()
