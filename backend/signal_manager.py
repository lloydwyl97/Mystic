"""
Enhanced Signal Manager for Mystic Trading

Manages all trading signals and ensures they are active and properly integrated.
"""

import asyncio
import json
import logging
import math
import os
import random
import statistics
import time
from datetime import datetime, timezone
from typing import Any

# Import from single source of truth
try:
    from backend.config.trading_universe import (
        EXCHANGE_ID,
        TRADING_SYMBOLS,
    )
except (ImportError, ModuleNotFoundError, AttributeError, ValueError, TypeError, RuntimeError) as e:
    msg = f"Failed to import trading_universe: {e}"
    raise RuntimeError(msg) from e

from backend.notification_service import get_notification_service

# Optional imports - try at top level
try:
    from backend.services.canonical_http_client import get_http_client
except (ImportError, ModuleNotFoundError, AttributeError, ValueError, TypeError, RuntimeError):
    get_http_client = None

try:
    from backend.utils.binance_weight_limiter import BinanceWeightLimiter
except (ImportError, ModuleNotFoundError, AttributeError, ValueError, TypeError, RuntimeError):
    BinanceWeightLimiter = None

logger = logging.getLogger(__name__)

# Module-level singletons for rate limiting and caching
_sig_limiter: BinanceWeightLimiter | None = None
_sig_sem = asyncio.Semaphore(2)
_sig_client: Any | None = None
_sig_cache: dict[tuple[str, str, int], tuple[float, Any]] = {}
_sig_init_lock = asyncio.Lock()

# All Live Data, No Fallback/Hardcoded Data
# Use BINANCEUS_BASE environment variable or default
BINANCE_US_BASE = os.getenv("BINANCEUS_BASE", "https://api.binance.us")
# Use TRADING_SYMBOLS from trading_universe (live data)
ALLOWED_SYMBOLS = set(TRADING_SYMBOLS)


def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def _ema(values: list[float], period: int) -> float:
    k = 2 / (period + 1)
    ema = values[0]
    for v in values[1:]:
        ema = (v - ema) * k + ema
    return float(ema)


def _macd(values: list[float]) -> float:
    if len(values) < 26:
        return 0.0
    fast = _ema(values[-26:], 12)
    slow = _ema(values[-26:], 26)
    return fast - slow


def _rsi(values: list[float], period: int = 14) -> float:
    if len(values) <= period:
        return 50.0
    gains = 0.0
    losses = 0.0
    for i in range(-period, 0):
        chg = values[i] - values[i - 1]
        if chg >= 0:
            gains += chg
        else:
            losses -= chg
    if gains == 0 and losses == 0:
        return 50.0
    if losses == 0:
        return 100.0
    rs = gains / period / (losses / period)
    return 100.0 - 100.0 / (1 + rs)


def _bollinger_position(values: list[float], period: int = 20, n_std: float = 2.0) -> float:
    if len(values) < period:
        return 0.5
    window = values[-period:]
    sma = sum(window) / period
    std = statistics.pstdev(window) if len(window) > 1 else 0.0
    lower = sma - n_std * std
    upper = sma + n_std * std
    if upper == lower:
        return 0.5
    pos = (values[-1] - lower) / (upper - lower)
    return float(_clamp(pos, 0.0, 1.0))


def _stochastic(values: list[float], period: int = 14) -> float:
    if len(values) < period:
        return 50.0
    window = values[-period:]
    lo = min(window)
    hi = max(window)
    if hi == lo:
        return 50.0
    return float((values[-1] - lo) / (hi - lo) * 100.0)


def _returns(values: list[float]) -> list[float]:
    out: list[float] = []
    for i in range(1, len(values)):
        prev = values[i - 1]
        curr = values[i]
        if prev == 0:
            out.append(0.0)
        else:
            out.append((curr - prev) / prev)
    return out


def _max_drawdown(values: list[float]) -> float:
    peak = values[0]
    max_dd = 0.0
    for v in values:
        peak = max(peak, v)
        dd = (v - peak) / peak if peak else 0.0
        max_dd = min(max_dd, dd)
    return abs(max_dd)


async def _fetch_klines(symbol: str, interval: str = "1m", limit: int = 200) -> list[list[Any]]:
    global _sig_limiter, _sig_client

    # Check cache first
    now = time.time()
    cache_key = (symbol, interval, int(limit))
    cached = _sig_cache.get(cache_key)
    if cached is not None:
        ts, data = cached
        # TTL rules: 1h => 300s, 15m => 60s, 1m => 10s, otherwise => 30s
        ttl = 300.0 if interval == "1h" else (60.0 if interval == "15m" else (10.0 if interval == "1m" else 30.0))
        if (now - ts) <= ttl:
            return data

    if BinanceWeightLimiter is None:
        msg = "BinanceWeightLimiter not available"
        raise RuntimeError(msg)

    # Initialize limiter and client once
    async with _sig_init_lock:
        if _sig_limiter is None:
            _sig_limiter = await BinanceWeightLimiter.create()
        if _sig_client is None:
            if get_http_client is None:
                msg = "get_http_client not available"
                raise RuntimeError(msg)
            _sig_client = await get_http_client()

    # Rate limit with retries and backoff
    last_exc: Exception | None = None
    for attempt in range(3):
        try:
            await _sig_limiter.consume("/api/v3/klines", weight=1, wait=True, timeout=10.0)
            async with _sig_sem:
                r = await _sig_client.get(
                    f"{BINANCE_US_BASE}/api/v3/klines",
                    params={"symbol": symbol, "interval": interval, "limit": limit},
                    timeout=10,
                )
                if r.status_code == 429:
                    retry_after = r.headers.get("Retry-After")
                    base_sleep = float(retry_after) if retry_after else (2.0**attempt)
                    await asyncio.sleep(base_sleep + random.random() * 0.5)
                    continue
                r.raise_for_status()
                data = r.json()
                # Store in cache
                _sig_cache[cache_key] = (now, data)
                return data
        except Exception as e:
            last_exc = e
            await asyncio.sleep(min(8.0, (2.0**attempt)) + random.random() * 0.5)
            continue

    if last_exc is not None:
        raise last_exc
    msg = "request failed without exception"
    raise RuntimeError(msg)


async def _fetch_ticker24(symbol: str) -> dict[str, Any]:
    if get_http_client is None:
        msg = "get_http_client not available"
        raise RuntimeError(msg)

    client = await get_http_client()
    r = await client.get(f"{BINANCE_US_BASE}/api/v3/ticker/24hr", params={"symbol": symbol}, timeout=10)
    r.raise_for_status()
    return r.json()


async def _fetch_order_book(symbol: str, limit: int = 100) -> dict[str, Any]:
    if get_http_client is None:
        msg = "get_http_client not available"
        raise RuntimeError(msg)

    client = await get_http_client()
    r = await client.get(
        f"{BINANCE_US_BASE}/api/v3/depth",
        params={"symbol": symbol, "limit": limit},
        timeout=10,
    )
    r.raise_for_status()
    return r.json()


async def _indicator_bundle(
    symbol: str,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    kl = await _fetch_klines(symbol, "1m", 200)
    closes = [float(k[4]) for k in kl]
    vols = [float(k[5]) for k in kl]
    price = closes[-1]
    rsi = round(_rsi(closes, 14), 2)
    macd = round(_macd(closes), 4)
    bb_pos = round(_bollinger_position(closes, 20, 2.0), 2)
    stoch = round(_stochastic(closes, 14), 2)
    if len(vols) >= 50:
        avg20 = (sum(vols[-20:]) / 20.0) if sum(vols[-20:]) != 0 else 0.0
        avg50 = (sum(vols[-50:]) / 50.0) if sum(vols[-50:]) != 0 else 0.0
        vol_sma = round((avg20 / avg50) if avg50 > 0 else 1.0, 2)
    else:
        vol_sma = 1.0
    sma20 = sum(closes[-20:]) / 20 if len(closes) >= 20 else price
    price_sma = round(price / sma20 if sma20 else 1.0, 3)
    indicators = {
        "rsi": rsi,
        "macd": macd,
        "bollinger_position": bb_pos,
        "stochastic": stoch,
        "volume_sma": vol_sma,
        "price_sma": price_sma,
    }
    rets = _returns(closes[-120:]) if len(closes) >= 2 else [0.0]
    vol = round(statistics.pstdev(rets) if len(rets) > 1 else 0.0, 4)
    mean = statistics.fmean(rets) if rets else 0.0
    sharpe = round((mean / vol) * math.sqrt(len(rets)) if vol > 0 else 0.0, 2)
    mdd = round(_max_drawdown(closes[-120:]) if len(closes) >= 2 else 0.0, 3)
    sorted_rets = sorted(rets)
    idx = max(0, int(0.05 * len(sorted_rets)) - 1)
    var95 = round(abs(sorted_rets[idx]) if sorted_rets else 0.0, 3)
    risk = {
        "volatility": round(vol, 3),
        "sharpe_ratio": sharpe,
        "max_drawdown": mdd,
        "var_95": var95,
    }
    book = await _fetch_order_book(symbol, 100)
    bids = [(float(p), float(q)) for p, q in book.get("bids", [])]
    asks = [(float(p), float(q)) for p, q in book.get("asks", [])]
    bid_vol_usd = sum(p * q for p, q in bids)
    ask_vol_usd = sum(p * q for p, q in asks)
    bid_ask_ratio = round((bid_vol_usd / ask_vol_usd) if ask_vol_usd else 0.0, 2)
    large_threshold = max(5000.0, price * 50)
    large_orders = sum(1 for p, q in bids + asks if p * q >= large_threshold)
    imbalance = (bid_vol_usd - ask_vol_usd) / (bid_vol_usd + ask_vol_usd) if (bid_vol_usd + ask_vol_usd) else 0.0
    order_flow = {
        "bid_volume": round(bid_vol_usd, 0),
        "ask_volume": round(ask_vol_usd, 0),
        "bid_ask_ratio": bid_ask_ratio,
        "large_orders": float(large_orders),
        "order_imbalance": round(imbalance, 3),
    }
    return indicators, risk, order_flow


def _derive_sentiment(indicators: dict[str, Any]) -> dict[str, Any]:
    rsi = indicators.get("rsi", 50.0)
    macd = indicators.get("macd", 0.0)
    mood_score = 50.0
    if rsi >= 60 and macd > 0:
        mood = "bullish"
        mood_score = 70.0
    elif rsi <= 40 and macd < 0:
        mood = "bearish"
        mood_score = 30.0
    else:
        mood = "neutral"
        mood_score = 50.0
    return {
        "social_score": round(mood_score, 1),
        "news_sentiment": "neutral",
        "fear_greed_index": round(mood_score, 1),
        "market_mood": mood,
    }


def _strategy_recommendations(indicators: dict[str, Any], risk: dict[str, Any]) -> list[dict[str, Any]]:
    rsi = float(indicators.get("rsi") or 50.0)
    macd = float(indicators.get("macd") or 0.0)
    bb = float(indicators.get("bollinger_position") or 0.5)
    stoch = float(indicators.get("stochastic") or 50.0)
    price_sma = float(indicators.get("price_sma") or 1.0)
    vol = float(risk.get("volatility") or 0.0)

    strategies: list[dict[str, Any]] = []

    def add(name: str, conf: float, thresh: float) -> None:
        strategies.append(
            {
                "name": name,
                "confidence": round(_clamp(conf, 0.0, 0.99), 3),
                "recommended": conf >= thresh,
                "position_size": round(0.02 if vol >= 0.01 else 0.05, 3) if conf >= thresh else 0.0,
            },
        )

    mom_conf = 0.5 + (rsi - 50) / 100 + (0.3 if macd > 0 else -0.1) + (0.2 if price_sma > 1.0 else -0.1)
    add("momentum", mom_conf, 0.65)

    mr_conf = 0.6 if rsi <= 30 or rsi >= 70 else 0.4
    mr_conf += 0.2 if 0.0 <= bb <= 0.15 or 0.85 <= bb <= 1.0 else 0.0
    add("mean_reversion", mr_conf, 0.6)

    day_conf = 0.4 + (0.4 if 0.01 <= vol <= 0.03 else 0.0) + (0.1 if 40 <= stoch <= 60 else 0.0)
    add("daying", day_conf, 0.7)

    grid_conf = 0.5 + (0.3 if 0.005 <= vol <= 0.02 else -0.1)
    add("grid_trading", grid_conf, 0.5)

    mm_conf = 0.5 + (-0.2 if vol > 0.03 else 0.2)
    add("market_making", mm_conf, 0.6)

    swing_conf = 0.5 + (0.2 if 0.02 <= vol <= 0.05 else 0.0) + (0.1 if abs(macd) > 0.1 else 0.0)
    add("swing_trading", swing_conf, 0.6)

    return strategies


class SignalManager:
    def __init__(self, redis_client: Any) -> None:
        self.redis_client = redis_client
        self.active_signals: dict[str, Any] = {}
        self.signal_generators: dict[str, Any] = {}
        self.auto_trading_enabled = False
        self.notification_service = get_notification_service(redis_client)
        # previous health state stored to detect changes
        self.previous_health_state: dict[str, Any] | None = None
        # keep a small cache of previous known signals/configs to assist self-healing
        self.previous_signals: dict[str, Any] = {}
        self.previous_auto_trade: dict[str, Any] | None = None

    async def get_auto_trade_status(self) -> dict[str, Any]:
        try:
            config_data = self.redis_client.get("auto_trade_config")
            config = json.loads(config_data) if config_data else {"enabled": False, "strategies": [], "message": "Auto-trading not configured"}
            return {
                "status": "success",
                "auto_trading": config,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
        except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
            logger.exception(f"Error getting auto-trade status: {e!s}")
            raise

    async def self_heal_signals(self) -> dict[str, Any]:
        try:
            logger.info("Starting signal self-healing process...")
            status = await self.get_signal_status()
            signals = status.get("signals", {})
            auto_trading = status.get("auto_trading", {})
            healing_actions: list[str] = []
            needs_healing = False
            for name, cfg in signals.items():
                if not cfg.get("enabled", False):
                    cfg["enabled"] = True
                    cfg["last_update"] = datetime.now(timezone.utc).isoformat()
                    cfg["status"] = "active"
                    healing_actions.append(f"Reactivated signal: {name}")
                    needs_healing = True
            if not auto_trading.get("enabled", False):
                await self.start_auto_trading()
                healing_actions.append("Reactivated auto-trading")
                needs_healing = True
            strategies = status.get("strategies", {})
            for s_name, s_cfg in strategies.items():
                if not s_cfg.get("enabled", False):
                    s_cfg["enabled"] = True
                    healing_actions.append(f"Reactivated strategy: {s_name}")
                    needs_healing = True
            if needs_healing:
                self.redis_client.setex("signal_status", 3600, json.dumps(signals))
                self.redis_client.setex("trading_strategies", 3600, json.dumps(strategies))
                await self.notification_service.send_notification(
                    title="Signal Recovery",
                    message=f"Successfully recovered {len(healing_actions)} components through self-healing",
                    level="info",
                    channels=["in_app", "slack", "email"],
                    data={
                        "actions_taken": healing_actions,
                        "healing_timestamp": datetime.now(timezone.utc).isoformat(),
                    },
                )
            return {
                "status": "success",
                "healing_performed": needs_healing,
                "actions_taken": healing_actions,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
        except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
            logger.exception(f"Error during self-healing: {e!s}")
            await self.notification_service.send_notification(
                title="Self-Healing Failed",
                message=f"Self-healing process failed: {e!s}",
                level="error",
                channels=["in_app", "slack", "email"],
                data={"error": str(e)},
            )
            raise

    async def check_signal_health(self) -> dict[str, Any]:
        try:
            status = await self.get_signal_status()
            signals = status.get("signals", {})
            auto_trading = status.get("auto_trading", {})
            strategies = status.get("strategies", {})
            total_signals = len(signals)
            healthy_signals = sum(1 for s in signals.values() if s.get("enabled", False))
            total_strategies = len(strategies)
            healthy_strategies = sum(1 for s in strategies.values() if s.get("enabled", False))
            auto_trading_healthy = auto_trading.get("enabled", False)
            overall_health = "healthy"
            if healthy_signals < total_signals or healthy_strategies < total_strategies or not auto_trading_healthy:
                overall_health = "degraded"
            if healthy_signals == 0 and healthy_strategies == 0 and not auto_trading_healthy:
                overall_health = "critical"
            current = {
                "overall_health": overall_health,
                "signals": {
                    "total": total_signals,
                    "healthy": healthy_signals,
                    "unhealthy": total_signals - healthy_signals,
                },
                "strategies": {
                    "total": total_strategies,
                    "healthy": healthy_strategies,
                    "unhealthy": total_strategies - healthy_strategies,
                },
                "auto_trading": {"healthy": auto_trading_healthy},
            }
            await self._check_health_changes(current)
            self.previous_health_state = current
            return {
                "status": "success",
                "overall_health": overall_health,
                "signals": current["signals"],
                "strategies": current["strategies"],
                "auto_trading": current["auto_trading"],
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
        except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
            logger.exception(f"Error checking signal health: {e!s}")
            raise

    async def _check_health_changes(self, current_health: dict[str, Any]):
        if self.previous_health_state is None:
            logger.info(f"Initial health state: {current_health['overall_health']}")
            return
        prev = self.previous_health_state
        if prev["overall_health"] == "healthy" and current_health["overall_health"] in [
            "degraded",
            "critical",
        ]:
            await self.notification_service.send_notification(
                title="Signal Health Degradation",
                message=f"Signal health degraded from {prev['overall_health']} to {current_health['overall_health']}",
                level="warning",
                channels=["in_app", "slack", "email"],
                data={
                    "previous_health": prev,
                    "current_health": current_health,
                    "degradation_time": datetime.now(timezone.utc).isoformat(),
                },
            )
        elif current_health["overall_health"] == "critical":
            await self.notification_service.send_notification(
                title="Critical Signal Health",
                message="All signals and auto-trading are down - CRITICAL",
                level="critical",
                channels=["in_app", "slack", "email"],
                data={
                    "current_health": current_health,
                    "critical_time": datetime.now(timezone.utc).isoformat(),
                },
            )
        elif prev["overall_health"] in ["degraded", "critical"] and current_health["overall_health"] == "healthy":
            await self.notification_service.send_notification(
                title="Signal Health Recovery",
                message=f"Signal health recovered from {prev['overall_health']} to {current_health['overall_health']}",
                level="info",
                channels=["in_app", "slack", "email"],
                data={
                    "previous_health": prev,
                    "current_health": current_health,
                    "recovery_time": datetime.now(timezone.utc).isoformat(),
                },
            )


# Signal manager state - using dict to avoid global keyword
_signal_manager_state: dict[str, SignalManager | None] = {"instance": None}


def get_signal_manager(redis_client: Any) -> SignalManager:
    if _signal_manager_state["instance"] is None:
        _signal_manager_state["instance"] = SignalManager(redis_client)
    return _signal_manager_state["instance"]
