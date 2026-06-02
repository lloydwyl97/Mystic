#!/usr/bin/env python3
"""
Mystic Market Data Poller
-------------------------
Handles live market data polling, pre-processing, API throttling,
and redistributing to AI, backend, and alerts via Redis pub/sub.

Py 3.12 • Windows-ready (selector loop policy) • Binance US top-10 only
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import os
import random
import time
from typing import Any

import httpx

# Import from proper package structure
import redis.asyncio as aioredis

# Import shared Redis client to prevent connection leaks
try:
    from backend.config.redis_config import get_shared_redis_async
except (ImportError, ModuleNotFoundError, AttributeError, ValueError, TypeError, RuntimeError):
    get_shared_redis_async = None  # type: ignore[assignment]

# Optional imports - try at top level
try:
    from backend.modules.ai.persistent_cache import (
        get_persistent_cache,
    )  # type: ignore[import-not-found]
except (ImportError, ModuleNotFoundError, AttributeError, ValueError, TypeError, RuntimeError):
    get_persistent_cache = None

# Direct imports for production
from backend.utils.binance_weight_limiter import (
    BinanceWeightLimiter,
    CircuitOpen,
    RateLimited,
)

# Optional CacheGuard import for market_data:last_update
try:
    from backend.utils.cache_guard import CacheGuard
except (ImportError, ModuleNotFoundError, AttributeError, ValueError, TypeError, RuntimeError):
    CacheGuard = None  # type: ignore[assignment,misc]

# Import from single source of truth
try:
    from backend.config.trading_universe import EXCHANGE_ID
except (ImportError, ModuleNotFoundError, AttributeError, ValueError, TypeError, RuntimeError) as e:
    msg = f"Failed to import trading_universe: {e}"
    raise RuntimeError(msg) from e

# Logging - initialize early so it's available for symbol loading
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("market-data-poller")

# Config
REDIS_URL = os.getenv("REDIS_URL")
BINANCE_URL_TEMPLATE = os.getenv(
    "BINANCE_URL_TEMPLATE",
    "https://api.binance.us/api/v3/ticker/price?symbol={symbol}",
)
SYMBOLS_ENV = os.getenv("BINANCE_US_SYMBOLS")

# Parse symbols from environment or use trading_universe as fallback
BINANCE_US_SYMBOLS = None
if SYMBOLS_ENV:
    try:
        BINANCE_US_SYMBOLS = [s.strip() for s in SYMBOLS_ENV.split(",") if s.strip()]
    except Exception:
        BINANCE_US_SYMBOLS = None

# If no symbols from env, try to get from trading_universe (live config)
if not BINANCE_US_SYMBOLS:
    try:
        from backend.config.trading_universe import TRADING_SYMBOLS

        if TRADING_SYMBOLS:
            # Convert trading symbols to Binance format (add USDT if needed)
            BINANCE_US_SYMBOLS = []
            for symbol in TRADING_SYMBOLS:
                if isinstance(symbol, str):
                    # Ensure symbol ends with USDT for Binance
                    symbol_upper = symbol.strip().upper()
                    if not symbol_upper.endswith("USDT"):
                        symbol_upper = f"{symbol_upper}USDT"
                    BINANCE_US_SYMBOLS.append(symbol_upper)
            if BINANCE_US_SYMBOLS:
                logger.info(f"Using {len(BINANCE_US_SYMBOLS)} symbols from trading_universe: {BINANCE_US_SYMBOLS}")
    except (ImportError, ModuleNotFoundError, AttributeError, ValueError, TypeError, RuntimeError) as e:
        logger.warning(f"Failed to load symbols from trading_universe: {e}")
        BINANCE_US_SYMBOLS = None


class MarketDataPoller:
    """
    Live Market Data Poller for fetching and publishing market data.
    """

    def __init__(self):
        self._cg: CacheGuard | None = None  # CacheGuard for market_data:last_update
        self.redis_url = REDIS_URL
        self.symbols = BINANCE_US_SYMBOLS
        self.binance_url_template = BINANCE_URL_TEMPLATE
        self.logger = logger

        # Persistent cache singleton (writer) — optional
        self._pcache: Any | None = None
        try:
            if get_persistent_cache is None:
                self.logger.info("Persistent cache not available")
            else:
                self._pcache = get_persistent_cache()
        except (ImportError, ModuleNotFoundError, AttributeError) as e:
            self.logger.info(f"Persistent cache not available: {e}")
        except Exception as e:
            self.logger.warning(f"Unexpected error initializing persistent cache: {e}")

        # Weight limiter for Binance US — optional
        self._limiter: BinanceWeightLimiter | None = None

    async def connect_redis_with_retry(
        self,
    ) -> aioredis.Redis:
        """Connect to Redis with retry logic."""

        # CRITICAL FIX: Use shared Redis client FIRST (works even without REDIS_URL)
        # This prevents connection leaks (was creating 3850+ connections)
        if get_shared_redis_async is not None:
            try:
                r = get_shared_redis_async()
                await r.ping()
                self.logger.info("Connected to shared Redis pool (prevents connection leaks).")
                return r
            except Exception as e:
                self.logger.warning(f"Shared Redis failed: {e}, falling back to new client")

        # Fallback: create new client (requires REDIS_URL)
        if not self.redis_url:
            msg = "REDIS_URL environment variable not set and shared Redis client failed"
            raise ConnectionError(msg)

        # Fallback: create new client with connection limit and retry logic
        retries = 5
        delay = 3
        last_err = None
        for attempt in range(1, retries + 1):
            try:
                r = aioredis.from_url(
                    self.redis_url,
                    decode_responses=True,
                    protocol=2,  # CRITICAL: Use RESP2 to avoid CLIENT SETINFO issues on Windows Redis
                    max_connections=50,  # Limit connections
                )
                await r.ping()
                self.logger.info("Connected to Redis.")
            except (
                ConnectionError,
                OSError,
                aioredis.ConnectionError,
                aioredis.TimeoutError,
            ) as e:
                last_err = e
                self.logger.warning(f"Redis not ready (attempt {attempt}/{retries}): {e}")
                await asyncio.sleep(delay)
            except Exception as e:
                last_err = e
                self.logger.exception(f"Unexpected Redis connection error (attempt {attempt}/{retries}): {e}")
                await asyncio.sleep(delay)
            else:
                return r
        msg = f"Could not connect to Redis after retries: {last_err!s}"
        raise ConnectionError(msg)

    async def _pcache_set_price(
        self,
        exchange: str,
        symbol: str,
        price: float,
        ts: float,
    ) -> None:
        """Safely write to persistent cache (sync or async)."""
        if not self._pcache:
            return
        try:
            setter = getattr(self._pcache, "set_price", None)
            if setter is None:
                return
            res = setter(exchange, symbol, price, ts)
            if asyncio.iscoroutine(res):
                await res
        except (AttributeError, TypeError, ValueError) as e:
            self.logger.warning(f"Cache write failed for {symbol}: {e}")
        except Exception as e:
            self.logger.exception(f"Unexpected cache write error for {symbol}: {e}")

    async def fetch_json(self, url: str) -> dict[str, Any]:
        """GET JSON with robust error handling."""
        try:
            if httpx is None:
                self.logger.warning("httpx not available")
                return {}
            async with httpx.AsyncClient(timeout=httpx.Timeout(10, read=10)) as client:
                r = await client.get(url)
                if r.status_code != 200:
                    self.logger.warning(f"HTTP {r.status_code} for {url}")
                    return {}
                try:
                    return r.json()
                except (
                    ValueError,
                    TypeError,
                    json.JSONDecodeError,
                ) as je:
                    self.logger.warning(f"JSON decode error for {url}: {je}")
                    return {}
        except (
            httpx.TimeoutException,
            httpx.ConnectError,
            httpx.RequestError,
        ) as e:
            self.logger.warning(f"Network error fetching {url}: {e}")
            return {}
        except Exception as e:
            self.logger.exception(f"Unexpected fetch error for {url}: {e}")
            return {}

    async def _publish_json(
        self,
        r: aioredis.Redis,
        channel: str,
        payload: dict[str, Any],
    ) -> None:
        """Publish a JSON payload, swallowing transient errors."""
        try:
            await r.publish(channel, json.dumps(payload))
        except (
            aioredis.ConnectionError,
            aioredis.TimeoutError,
            OSError,
        ) as e:
            self.logger.warning(f"Redis publish failed ({channel}): {e}")
        except Exception as e:
            self.logger.exception(f"Unexpected Redis publish error ({channel}): {e}")

    async def _init_limiter(self, r: aioredis.Redis) -> None:
        """Initialize the Binance weight limiter if available."""
        try:
            self._limiter = await BinanceWeightLimiter.create()
            # Override default budget if configured
            budget_override = os.getenv("BINANCE_BUDGET_OVERRIDE")
            if budget_override:
                try:
                    budget_val = int(budget_override)
                    await r.set("bwl:budget_override", str(budget_val))
                    self.logger.info(f"Weight limiter initialized with {budget_val}/min budget")
                except (ValueError, TypeError):
                    self.logger.warning("Invalid BINANCE_BUDGET_OVERRIDE value, using default")
            else:
                self.logger.info("Weight limiter initialized with default budget")
        except (ImportError, ModuleNotFoundError, AttributeError) as e:
            self._limiter = None
            self.logger.info(f"Weight limiter not available: {e}")
        except Exception as e:
            self._limiter = None
            self.logger.warning(f"Failed to initialize weight limiter (continuing without it): {e}")

    async def poll_loop(self) -> None:
        """Main polling loop."""
        self.logger.info(f"MarketDataPoller poll_loop starting with symbols: {self.symbols}")
        if not self.symbols:
            self.logger.error("No symbols configured. Set BINANCE_US_SYMBOLS environment variable.")
            return

        self.logger.info("Connecting to Redis...")
        r = await self.connect_redis_with_retry()
        self.logger.info("Initializing weight limiter...")
        await self._init_limiter(r)
        self.logger.info("MarketDataPoller poll_loop initialized, starting main loop")

        # Startup jitter to prevent synchronized bursts after restart
        await asyncio.sleep(3.0 + random.random() * 3.0)

        interval_str = os.getenv("POLL_INTERVAL_SECONDS", "10.0")
        try:
            base_interval = float(interval_str)
        except (ValueError, TypeError):
            base_interval = 10.0
            self.logger.warning("Invalid POLL_INTERVAL_SECONDS, using default: 10.0")

        jitter_str = os.getenv("POLL_JITTER_SECONDS", "0.0")
        try:
            jitter = float(jitter_str)
        except (ValueError, TypeError):
            jitter = 0.0
            self.logger.warning("Invalid POLL_JITTER_SECONDS, using default: 0.0")

        while True:
            try:
                timestamp = time.time()
                market_data: dict[str, float] = {}
                hydrated_symbols = 0

                for symbol in self.symbols:
                    # Respect limiter if present
                    if self._limiter is not None:
                        try:
                            await self._limiter.consume("/api/v3/ticker/price", 1)
                        except (RateLimited, CircuitOpen) as e:
                            self.logger.info(f"Skipped {symbol} due to limiter: {e}")
                            continue
                        except (AttributeError, TypeError, ValueError) as e:
                            # Don't block on limiter internals; proceed anyway
                            self.logger.warning(f"Limiter error (continuing): {e}")
                        except Exception as e:
                            # Don't block on limiter internals; proceed anyway
                            self.logger.exception(f"Unexpected limiter error (continuing): {e}")

                    # Binance US: symbol already in BTCUSDT format
                    url = self.binance_url_template.format(symbol=symbol)
                    data = await self.fetch_json(url)
                    if "price" in data:
                        try:
                            price = float(data["price"])
                        except (ValueError, TypeError) as e:
                            price_raw = data.get("price")
                            self.logger.warning(f"Non-numeric price for {symbol}: {price_raw!r} - {e}")
                            price = None

                        if price is not None:
                            market_data[f"{symbol}_binance"] = price
                            await self._pcache_set_price(EXCHANGE_ID, symbol, price, timestamp)
                            hydrated_symbols += 1

                    # Emit hydration metrics (best-effort)
                    try:
                        await r.set(
                            "metrics:hydrated_symbols",
                            str(hydrated_symbols),
                            ex=30,
                        )
                        # fresh writer heartbeat if full sweep
                        if hydrated_symbols == len(self.symbols):
                            await r.set("metrics:writer_freshness_sec", "0", ex=30)
                    except (
                        aioredis.ConnectionError,
                        aioredis.TimeoutError,
                        OSError,
                    ):
                        # Metrics are best-effort, don't log warnings for
                        # transient Redis issues
                        pass

                    if market_data:
                        payload = {
                            "timestamp": timestamp,
                            "data": market_data,
                        }
                        # Expose latest snapshot for downstream consumers
                        with contextlib.suppress(aioredis.ConnectionError, aioredis.TimeoutError, OSError):
                            await r.set("market_data_latest", json.dumps(payload), ex=30)
                        await self._publish_json(r, "mystic:livefeed", payload)

                        # Update global market data timestamp for stale data detection
                        if hydrated_symbols > 0:  # Only if we actually got prices
                            self.logger.info(f"Updating market data timestamp for {hydrated_symbols} symbols")
                            if self._cg is None and CacheGuard is not None:
                                try:
                                    self._cg = await CacheGuard.create()
                                    self.logger.info("CacheGuard created successfully")
                                except Exception as e:
                                    self.logger.warning(f"CacheGuard creation failed: {e}")

                            if self._cg is not None:
                                try:
                                    await self._cg.mark_market_update("rest_poller")
                                    self.logger.info("Market data timestamp updated")
                                except Exception as e:
                                    self.logger.warning(f"Mark market update failed: {e}")
                            else:
                                self.logger.warning("CacheGuard is None, cannot update timestamp")

                        self.logger.info(f"[Livefeed] Published {len(market_data)} items, hydrated: {hydrated_symbols}/{len(self.symbols)}")
                    else:
                        self.logger.info(f"[Livefeed] No data this cycle (hydrated: {hydrated_symbols}/{len(self.symbols)})")

                # Background cadence: configurable via environment (MOVED OUTSIDE SYMBOL LOOP)
                sleep_time = base_interval + jitter
                self.logger.info(f"Next sweep in {sleep_time:.1f}s")
                await asyncio.sleep(sleep_time)

            except asyncio.CancelledError:
                # BUG #51 FIX: Clean exit on cancellation
                self.logger.info("Market data poller shutting down")
                break
            except Exception as e:
                # BUG #51 FIX: Don't crash on unexpected errors in 24/7 loop
                self.logger.exception(f"Error in poll loop (retrying): {e}")
                await asyncio.sleep(base_interval)

    def start(self) -> None:
        """Start the poller."""
        try:
            asyncio.run(self.poll_loop())
        except KeyboardInterrupt:
            self.logger.info("Shutdown requested, exiting.")
        except Exception as e:
            self.logger.exception(f"Unexpected error in poll loop: {e}")
            raise


# Entry
if __name__ == "__main__":
    # Windows: use Selector policy to avoid edge cases with Proactor +
    # some libs
    if os.name == "nt":
        with contextlib.suppress(AttributeError, OSError):
            # Windows-specific event loop policy - USE PROACTOR (no file descriptor limit)
            import sys

            if sys.platform == "win32":
                asyncio.set_event_loop_policy(
                    asyncio.WindowsProactorEventLoopPolicy()  # type: ignore[misc]
                )

    poller = MarketDataPoller()
    poller.start()
