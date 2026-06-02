import logging
import os
import time
from typing import Any

from backend.config.redis_config import get_shared_redis_sync

# Import from single source of truth
try:
    from backend.config.trading_universe import EXCHANGE_ID
except (ImportError, ModuleNotFoundError, AttributeError, ValueError, TypeError, RuntimeError) as e:
    msg = f"Failed to import trading_universe: {e}"
    raise RuntimeError(msg) from e

from backend.config.trading_universe import TOP10_COINS

logger = logging.getLogger(__name__)

# Rate limiting configuration - Use EXCHANGE_ID from trading_universe (live data)
RATE_LIMITS: dict[str, dict[str, Any]] = {
    EXCHANGE_ID: {"rpm": 30, "last_call": 0},  # Increased to handle 10 coins
}


def _to_base_symbol(token: str) -> str:
    s = (token or "").strip().upper()
    if not s:
        return ""
    if s.endswith("USDT"):
        return s[:-4]
    if s.endswith("USD"):
        return s[:-3]
    if "/" in s:
        return s.split("/", 1)[0]
    if "-" in s:
        return s.split("-", 1)[0]
    return s


def _load_supported_coins() -> list[str]:
    """
    Load from single source of truth with optional env override.
    Accepts values like "BTCUSDT,ETHUSDT,..." and normalizes to base symbols.
    """
    raw = os.getenv("BINANCE_US_TOP10_SYMBOLS", "").strip()
    if raw:
        out: list[str] = []
        for token in raw.split(","):
            base = _to_base_symbol(token)
            if base and base not in out:
                out.append(base)
        return out
    # Import from single source of truth - NO HARDCODING
    return TOP10_COINS.copy()


SUPPORTED_COINS: list[str] = _load_supported_coins()  # All requests will use <BASE>USDT

FAST_COINS: list[str] = SUPPORTED_COINS

API_SCHEDULE: dict[str, dict[str, Any]] = {
    EXCHANGE_ID: {"coins": FAST_COINS, "delay": 0},
}


def is_supported(coin: str) -> bool:
    return coin.upper() in SUPPORTED_COINS


def throttle(provider: str) -> None:
    """Throttle requests based on provider rate limits."""
    if provider not in RATE_LIMITS:
        return
    current_time = time.time()
    limit_info = RATE_LIMITS[provider]
    min_interval = 60.0 / float(limit_info["rpm"])
    delta = current_time - float(limit_info["last_call"])
    if delta < min_interval:
        time.sleep(min_interval - delta)
    RATE_LIMITS[provider]["last_call"] = time.time()


def fetch_staggered_batch() -> dict[str, dict[str, Any]]:
    """Fetch data using staggered batching to respect rate limits (cache-only)."""
    results: dict[str, dict[str, Any]] = {}
    logger.info("Starting staggered batch fetch for configured coins...")
    logger.info("Source: Binance US cache only")

    for coin in SUPPORTED_COINS:
        try:
            result = fetch_from_binance(coin)
            if result:
                results[coin] = result
                logger.info(f"OK {coin}: ${result['price']} from {result['source']}")
        except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
            logger.exception(f"BINANCE cache read failed for {coin}: {e}")

    logger.info(f"Batch complete: {len(results)}/{len(SUPPORTED_COINS)} coins fetched")
    return results


def fetch_from_binance(symbol: str) -> dict[str, Any] | None:
    """Read price from Redis (populated by WS hydrator) for <BASE>USDT."""
    sym = f"{symbol.upper()}USDT"
    # All Live Data, No Fallback/Hardcoded Data
    try:
        r = get_shared_redis_sync()
        if r is None:
            logger.warning("Shared Redis client unavailable; binance cache fetch skipped")
            return None
        raw = r.hget(f"price:{sym}", "v")
        if not raw:
            return None
        try:
            price = float(raw.decode() if isinstance(raw, (bytes, bytearray)) else raw)
        except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError):
            return None
        if price > 0:
            return {"price": price, "source": "binance_cache"}
    except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
        logger.exception(f"Binance cache read error: {e}")
        return None
    else:
        return None


def fetch_all_supported_coins() -> dict[str, dict[str, Any]]:
    return fetch_staggered_batch()


def test_coin_support(symbol: str) -> dict[str, bool]:
    results: dict[str, bool] = {}
    try:
        result = fetch_from_binance(symbol)
        results[EXCHANGE_ID] = result is not None
    except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
        logger.exception(f"BINANCE test failed for {symbol}: {e}")
        results[EXCHANGE_ID] = False
    return results


def get_coin_support_summary() -> dict[str, dict[str, bool]]:
    summary: dict[str, dict[str, bool]] = {}
    for coin in SUPPORTED_COINS:
        summary[coin] = test_coin_support(coin)
        time.sleep(2)  # be nice
    return summary


def show_coin_summary() -> dict[str, Any]:
    logger.info("=" * 60)
    logger.info("COIN SUMMARY & STAGGERED BATCH ANALYSIS")
    logger.info("=" * 60)
    logger.info(f"\nTOTAL COINS: {len(SUPPORTED_COINS)}")
    logger.info(f"COIN LIST: {', '.join(SUPPORTED_COINS)}")
    logger.info("\nSTAGGERED BATCHING STRATEGY:")
    logger.info(f"   - FAST COINS ({len(FAST_COINS)}): {', '.join(FAST_COINS)}")
    logger.info("\nOPTIMIZED API HITS PER SERVICE:")
    logger.info(f"   - Binance US: {len(SUPPORTED_COINS)} hits (no delay)")
    total_time = 0
    logger.info(f"\nTOTAL TIME (approx): {total_time} seconds ({total_time / 60:.1f} minutes)")
    logger.info("\nRATE LIMIT COMPLIANCE:")
    logger.info(f"   - Binance US: {len(SUPPORTED_COINS)} hits vs {RATE_LIMITS[EXCHANGE_ID]['rpm']}/min")
    logger.info("=" * 60)
    return {
        "total_coins": len(SUPPORTED_COINS),
        "coins": SUPPORTED_COINS,
        "fast_coins": FAST_COINS,
        "hits_per_service": {EXCHANGE_ID: len(SUPPORTED_COINS)},
        "rate_limits": RATE_LIMITS,
        "total_time_seconds": total_time,
    }
