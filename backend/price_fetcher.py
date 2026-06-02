#!/usr/bin/env python3
import asyncio
import inspect
import json

# import aiohttp - moved inside methods to avoid circular imports
import logging
import os
import time
from datetime import datetime, timedelta, timezone
from typing import Any

import httpx

# Import from single source of truth
try:
    from backend.config.trading_universe import EXCHANGE_ID, TRADING_SYMBOLS
except (ImportError, ModuleNotFoundError, AttributeError, ValueError, TypeError, RuntimeError) as e:
    msg = f"Failed to import trading_universe: {e}"
    raise RuntimeError(msg) from e

from backend.events.market_events import PriceUpdate, market_event_bus

logger = logging.getLogger(__name__)


class PriceSignal:
    def __init__(
        self,
        symbol: str,
        price: float,
        change_1m: float = 0.0,
        volume_1m: float = 0.0,
        timestamp: str = "",
        api_source: str = "",
    ) -> None:
        self.symbol = symbol
        self.price = float(price)
        self.change_1m = float(change_1m)
        self.volume_1m = float(volume_1m)
        self.timestamp = timestamp or datetime.now(timezone.utc).isoformat()
        self.api_source = api_source


class PriceFetcher:
    """
    Optimized for Binance US top 10 coins only
    """

    def __init__(self, redis_client: Any = None) -> None:
        self.redis_client = redis_client
        self.client: Any = None
        self.is_running = False

        # Configuration
        self.config = {
            "fetch_interval": 5,  # seconds (per symbol)
            "momentum_fetch_interval": 15,  # seconds (global)
            "cache_ttl": 30,  # seconds
            "max_retries": 3,
            "retry_delay": 1,  # seconds (base for backoff)
            "http_timeout": 10,  # seconds
        }

        # API endpoints - All Live Data, No Fallback/Hardcoded Data
        self.binance_base_url = os.getenv("BINANCEUS_BASE", "https://api.binance.us/api/v3")

        # Track last fetch times for throttling
        self.last_fetch_times: dict[str, float] = {}
        self.last_momentum_fetch = 0.0

        # All Live Data, No Fallback/Hardcoded Data - use trading_universe
        self.binance_coins = list(TRADING_SYMBOLS)

        # Price history for momentum calculations
        # { "BTCUSDT": [ {"price": float, "timestamp": iso}, ... ] }
        self.price_history: dict[str, list[dict[str, Any]]] = {}

        logger.info(f"Price Fetcher initialized with {len(self.binance_coins)} Binance US coins only")

    # ---------------------- Session lifecycle ----------------------

    async def initialize(self):
        """Initialize the price fetcher"""
        if not self.client:
            timeout = httpx.Timeout(self.config["http_timeout"], read=self.config["http_timeout"])
            self.client = httpx.AsyncClient(timeout=timeout)
        logger.info("Price Fetcher initialized")

    async def cleanup(self):
        """Cleanup resources"""
        if self.client:
            try:
                await self.client.aclose()
            except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError):
                pass
            finally:
                self.client = None
        self.is_running = False
        logger.info("Price Fetcher cleaned up")

    # ---------------------- Throttling helpers ----------------------

    def _key_price(self, symbol: str) -> str:
        return f"price:{symbol}"

    def _should_fetch(self, key: str) -> bool:
        """Check if enough time has passed since last fetch for this key."""
        now = time.time()
        last_fetch = self.last_fetch_times.get(key, 0.0)
        return (now - last_fetch) >= float(self.config["fetch_interval"])

    def _should_fetch_momentum(self) -> bool:
        """Check if momentum should be fetched (global cadence)."""
        now = time.time()
        return (now - self.last_momentum_fetch) >= float(self.config["momentum_fetch_interval"])

    def _update_fetch_time(self, key: str):
        """Update fetch time for throttling."""
        if key == "momentum":
            self.last_momentum_fetch = time.time()
        else:
            self.last_fetch_times[key] = time.time()

    # ---------------------- Public API ----------------------

    async def fetch_price(self, symbol: str, exchange: str | None = None) -> PriceSignal | None:
        """Fetch price for a single symbol"""
        try:
            # All Live Data, No Fallback/Hardcoded Data
            if exchange is None:
                exchange = EXCHANGE_ID
            if exchange != EXCHANGE_ID:
                logger.warning(f"Exchange {exchange} not supported - only {EXCHANGE_ID} supported")
                return None

            # per-symbol throttling
            per_symbol_key = self._key_price(symbol)
            if not self._should_fetch(per_symbol_key):
                # Not time yet; return None (callers may rely on cached values)
                return None

            data = await self._fetch_binance_price(symbol)
            if data:
                self._update_fetch_time(per_symbol_key)
                await self._cache_price_signal(data)
                await self._update_price_history(symbol, data.price)

                # Publish price update to event bus
                update = PriceUpdate(
                    symbol=symbol,
                    price=data.price,
                    volume=data.volume_1m,
                    timestamp=data.timestamp,
                )
                await market_event_bus.publish(update)

                return data

        except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
            logger.exception(f"Error fetching price for {symbol} from {exchange}: {e}")

        return None

    async def fetch_momentum_signals(self) -> dict[str, float]:
        """Fetch momentum signals globally (15 second frequency)"""
        if not self._should_fetch_momentum():
            return {}

        try:
            momentum_data: dict[str, float] = {}

            # Calculate 1-minute change for all coins - Binance US only
            for symbol in self.binance_coins:
                change_1m = await self._calculate_1m_change(symbol)
                if change_1m is not None:
                    momentum_data[symbol] = change_1m

            self._update_fetch_time("momentum")
            await self._cache_momentum_data(momentum_data)
        except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
            logger.exception(f"Error fetching momentum signals: {e}")
            return {}
        else:
            return momentum_data

    # ---------------------- Internals ----------------------

    async def _calculate_1m_change(self, symbol: str) -> float | None:
        """Calculate 1-minute price change in % based on local history."""
        try:
            history = self.price_history.get(symbol) or []
            if len(history) < 2:
                return None

            # current price is last
            current_price = float(history[-1]["price"])

            # find price from ~1 minute ago
            cutoff_ts = (datetime.now(timezone.utc) - timedelta(seconds=60)).timestamp()
            old_price: float | None = None

            # Iterate backwards to find the first entry older than 60s
            for entry in reversed(history[:-1]):
                entry_ts = datetime.fromisoformat(entry["timestamp"].replace("Z", "+00:00")).timestamp()
                if entry_ts <= cutoff_ts:
                    old_price = float(entry["price"])
                    break

            if old_price is not None and old_price > 0:
                return ((current_price - old_price) / old_price) * 100.0

        except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
            logger.exception(f"Error calculating 1m change for {symbol}: {e}")

        return None

    async def _fetch_binance_price(self, symbol: str) -> PriceSignal | None:
        """Fetch price from Binance US with simple retry/backoff."""
        # Ensure session exists
        if not self.client:
            await self.initialize()

        url = f"{self.binance_base_url}/ticker/price?symbol={symbol}"
        attempt = 0
        max_retries = int(self.config.get("max_retries", 3))
        while attempt < max_retries:
            try:
                response = await self.client.get(url)
                status = getattr(response, "status_code", None)
                if status == 200:
                    data = response.json()
                    price_val = float(data["price"])
                    return PriceSignal(
                        symbol=symbol,
                        price=price_val,
                        change_1m=0.0,  # computed separately
                        volume_1m=0.0,  # this endpoint doesn't include volume
                        timestamp=datetime.now(timezone.utc).isoformat(),
                        api_source=EXCHANGE_ID,
                    )
                if status in (429, 500, 502, 503, 504):
                    # transient or rate-limited: exponential backoff and retry
                    attempt += 1
                    backoff = float(self.config["retry_delay"]) * (2 ** (attempt - 1))
                    logger.warning(f"Binance price API {status} for {symbol}; retry {attempt}/{max_retries} in {backoff:.1f}s")
                    await asyncio.sleep(backoff)
                else:
                    # non-retryable
                    text = getattr(response, "text", "")
                    logger.error(f"Binance price API error {status} for {symbol}: {text}")
                    return None
            except asyncio.CancelledError:
                raise
            except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
                attempt += 1
                backoff = float(self.config["retry_delay"]) * (2 ** (attempt - 1))
                logger.warning(f"Error fetching Binance price for {symbol}: {e}; retry {attempt}/{max_retries} in {backoff:.1f}s")
                # Exponential backoff for exception retries
                await asyncio.sleep(backoff)

        logger.error(f"Failed to fetch price for {symbol} after {max_retries} attempts")
        return None

    async def _update_price_history(self, symbol: str, price: float):
        """Update price history for momentum calculations. Keep ~10 minutes."""
        bucket = self.price_history.setdefault(symbol, [])
        now_iso = datetime.now(timezone.utc).isoformat()
        bucket.append({"price": float(price), "timestamp": now_iso})

        # Trim by time window (10 minutes)
        ten_min_ago_ts = (datetime.now(timezone.utc) - timedelta(minutes=10)).timestamp()
        trimmed: list[dict[str, Any]] = []
        for entry in bucket:
            try:
                ts = datetime.fromisoformat(entry["timestamp"].replace("Z", "+00:00")).timestamp()
            except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError):
                # If parsing fails, drop the bad entry
                continue
            if ts >= ten_min_ago_ts:
                trimmed.append(entry)
        self.price_history[symbol] = trimmed

    async def _cache_price_signal(self, signal: PriceSignal):
        """Cache price signal"""
        try:
            if not self.redis_client:
                return
            key = f"price:{signal.symbol}"
            data = {
                "symbol": signal.symbol,
                "price": signal.price,
                "change_1m": signal.change_1m,
                "volume_1m": signal.volume_1m,
                "timestamp": signal.timestamp,
                "api_source": signal.api_source,
            }
            result = self.redis_client.setex(key, int(self.config["cache_ttl"]), json.dumps(data))
            if inspect.isawaitable(result):
                await result
        except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
            logger.exception(f"Error caching price signal: {e}")

    async def _cache_momentum_data(self, momentum_data: dict[str, float]):
        """Cache momentum data"""
        try:
            if not self.redis_client:
                return
            result = self.redis_client.setex(
                "momentum_signals",
                int(self.config["cache_ttl"]),
                json.dumps(momentum_data),
            )
            if inspect.isawaitable(result):
                await result
        except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
            logger.exception(f"Error caching momentum data: {e}")

    async def fetch_tier1_data(self) -> dict[str, Any]:
        """Fetch Tier 1 data (Binance US top 10 coins)"""
        results: dict[str, Any] = {
            "prices": {},
            "momentum": {},
            "last_update": int(time.time()),
        }

        # Fetch all prices (per-symbol throttled) - All Live Data, No Fallback/Hardcoded Data
        for symbol in self.binance_coins:
            price_signal = await self.fetch_price(symbol, EXCHANGE_ID)
            if price_signal:
                results["prices"][symbol] = {
                    "price": price_signal.price,
                    "change_1m": price_signal.change_1m,
                    "volume_1m": price_signal.volume_1m,
                    "timestamp": price_signal.timestamp,
                    "api_source": price_signal.api_source,
                }

        # Fetch momentum signals globally
        momentum_data = await self.fetch_momentum_signals()
        results["momentum"] = momentum_data

        # Update momentum data in price signals
        for symbol, momentum in momentum_data.items():
            if symbol in results["prices"]:
                results["prices"][symbol]["change_1m"] = momentum

        # Cache the complete Tier 1 data
        await self._cache_tier1_data(results)

        return results

    async def _cache_tier1_data(self, data: dict[str, Any]):
        """Cache complete Tier 1 data"""
        try:
            if not self.redis_client:
                return
            result = self.redis_client.setex("tier1_signals", int(self.config["cache_ttl"]), json.dumps(data))
            if inspect.isawaitable(result):
                await result
        except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
            logger.exception(f"Error caching Tier 1 data: {e}")

    # ---------------------- Run loop ----------------------

    async def run(self):
        """Main price fetcher loop - OPTIMIZED FOR BINANCE US ONLY"""
        logger.info("Starting Tier 1 Price Fetcher (Binance US only)...")
        self.is_running = True

        try:
            await self.initialize()

            while self.is_running:
                try:
                    tier1_data = await self.fetch_tier1_data()
                    logger.info(f"Fetched {len(tier1_data['prices'])} Binance US prices")
                    # Configurable fetch interval from config
                    await asyncio.sleep(float(self.config["fetch_interval"]))
                except asyncio.CancelledError:
                    raise
                except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
                    logger.exception(f"Error in price fetcher loop: {e}")
                    # Retry delay from config - wait before next attempt
                    await asyncio.sleep(float(self.config["retry_delay"]))
        except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
            logger.exception(f"Fatal error in price fetcher: {e}")
        finally:
            await self.cleanup()

    # ---------------------- Status ----------------------

    def get_status(self) -> dict[str, Any]:
        """Get fetcher status"""
        return {
            "is_running": self.is_running,
            "binance_coins": list(self.binance_coins),
            "total": len(self.binance_coins),
            "last_fetch_times": dict(self.last_fetch_times),
            "last_momentum_fetch": self.last_momentum_fetch,
        }
