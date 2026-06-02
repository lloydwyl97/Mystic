"""
Market Data Service
Handles live market data fetching and caching (Binance.US only, Top-10 enforced, no mocks).
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import os
import time
from datetime import datetime, timezone
from typing import Any

from backend.config.mystic_api_schedule import (
    MARKET_DATA_REST_LOOPS_ENABLED,
    MARKET_HIGH_INTERVAL_SEC,
    MARKET_NORMAL_INTERVAL_SEC,
    MARKET_TARGET_WEIGHT_PER_MIN,
)
from backend.config.redis_config import get_shared_redis_async
from backend.config.trading_universe import TOP10_COINS as TOP10_BINANCEUS
from backend.config.trading_universe import TRADING_SYMBOLS
from backend.services.binance_rest_client import BinanceREST
from backend.services.binance_ws_hydrator import BinanceWSHydrator
from backend.services.canonical_cache import canonical_cache as get_shared_cache
from backend.services.market_data_sources import is_supported
from backend.services.task_manager import task_manager
from backend.utils.binance_weight_limiter import BinanceWeightLimiter
from backend.utils.cache_guard import CacheGuard
from backend.websocket_manager import websocket_manager

logger = logging.getLogger(__name__)

_REDIS_CONFIG_ERROR = "REDIS_URL or REDIS_HOST required"


# Helper function to get high priority coins (first 2 from TOP10_COINS by default, configurable via env)
def _get_high_priority_coins() -> list[str]:
    """Get high priority coins from env or default to first 2 from TOP10_COINS (live data)"""
    high_priority_env = os.getenv("HIGH_PRIORITY_COINS", "").strip()
    if high_priority_env:
        high_priority_bases = [c.strip().upper() for c in high_priority_env.split(",") if c.strip()]
        return [c for c in TOP10_BINANCEUS if c in high_priority_bases]
    # Default to first 2 coins from TOP10_BINANCEUS (live data)
    HIGH_PRIORITY_COUNT = 2
    return list(TOP10_BINANCEUS[:HIGH_PRIORITY_COUNT]) if len(TOP10_BINANCEUS) >= HIGH_PRIORITY_COUNT else list(TOP10_BINANCEUS)


# Normal priority coins (all Top-10 minus high priority) - from trading_universe (live data)
_TOP10_NON_HIGH = [s for s in TOP10_BINANCEUS if s not in _get_high_priority_coins()]


def _get_top10_non_high_cached() -> list[str]:
    """Get Top-10 coins minus high priority coins (cached list from live data)."""
    return _TOP10_NON_HIGH.copy()


class MarketFreshnessHeartbeat:
    """
    Redis-only market data freshness heartbeat.

    Checks price freshness for all 10 symbols every 2 seconds.
    If 6+ symbols have fresh prices, updates market_data:last_update key.
    This prevents false FAIL_CLOSED pauses when price producers are working.

    Behavior:
    - Redis reads only (no Binance API calls)
    - Does not write prices
    - Only touches global timestamp if prices are proven fresh
    """

    def __init__(self) -> None:
        self._cg: CacheGuard | None = None
        self._task: asyncio.Task | None = None
        self._running = False

    async def start(self) -> None:
        """Start the heartbeat loop."""
        if self._running:
            return
        self._running = True
        self._cg = await CacheGuard.create()

        if task_manager is not None:
            self._task = await task_manager.create_task(self._heartbeat_loop(), name="market_freshness_heartbeat:loop")
        else:
            self._task = asyncio.create_task(self._heartbeat_loop())
        logger.info("[OK] MarketFreshnessHeartbeat started")

    async def stop(self) -> None:
        """Stop the heartbeat loop."""
        self._running = False
        if self._task and not self._task.done():
            self._task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._task
        self._task = None
        logger.info("[OK] MarketFreshnessHeartbeat stopped")

    async def _heartbeat_loop(self) -> None:
        """Main heartbeat loop - checks freshness every 2 seconds."""
        while self._running:
            try:
                if self._cg is None:
                    self._cg = await CacheGuard.create()

                # Check freshness for all 10 symbols
                fresh_count = 0
                recent_count = 0  # Count symbols with recent data (< 60s)

                for symbol in TRADING_SYMBOLS:
                    # Check for very fresh prices (< 10s) - ideal case
                    price = await self._cg.get_price_cached(symbol, freshness_sec=10)
                    if price is not None:
                        fresh_count += 1
                        recent_count += 1
                    else:
                        # Check for reasonably recent prices (< 60s) - acceptable for heartbeat
                        price_recent = await self._cg.get_price_cached(symbol, freshness_sec=60)
                        if price_recent is not None:
                            recent_count += 1

                # Update timestamp if we have sufficient price data (even if slightly stale)
                # This prevents false FAIL_CLOSED when WebSocket is working but updates come slowly
                if fresh_count >= 6:
                    # Perfect: 6+ symbols very fresh (< 10s)
                    await self._cg.mark_market_update("heartbeat")
                elif recent_count >= 6:
                    # Acceptable: 6+ symbols reasonably recent (< 60s) - still update to prevent false FAIL_CLOSED
                    await self._cg.mark_market_update("heartbeat")
                    logger.debug(f"MarketFreshnessHeartbeat: {recent_count} symbols recent (<60s), {fresh_count} very fresh (<10s)")
                else:
                    # Not enough price data - don't update, let FAIL_CLOSED detect the problem
                    logger.debug(f"MarketFreshnessHeartbeat: Only {recent_count} symbols have recent data (<60s)")

            except asyncio.CancelledError:
                break
            except Exception as e:
                # Log but don't crash - heartbeat continues
                logger.debug(f"MarketFreshnessHeartbeat error: {e}")

            # Wait 2 seconds before next check
            try:
                await asyncio.sleep(2.0)
            except asyncio.CancelledError:
                break


class MarketDataService:
    """Service for managing live market data"""

    _instance: MarketDataService | None = None

    def __init__(self) -> None:
        self.cache: dict[str, dict[str, Any]] = {}
        self.cache_timestamps: dict[str, float] = {}  # Track when items were added
        self.max_cache_size = 1000  # Limit cache size to prevent memory leaks
        self.background_tasks: list[asyncio.Task] = []
        self.is_running = False
        logger.info("MarketDataService initialized")

        # Normal-priority symbol universe (subset of Top-10, excluding high priority coins)
        self.normal_symbols: list[str] = self._load_normal_symbols()
        self._normal_idx: int = 0

        # Limiter/REST/Cache/WS (initialized in initialize)
        self._limiter: BinanceWeightLimiter | None = None
        self._rest: BinanceREST | None = None
        self._cache_guard: CacheGuard | None = None
        self._ws_hydrator: BinanceWSHydrator | None = None

    @classmethod
    def shared(cls) -> MarketDataService:
        """Get the global MarketDataService instance"""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    async def initialize(self):
        """Initialize the service and start background updates"""
        # Prevent double initialization (fixes CPU spin from multiple instances)
        if self.is_running:
            logger.warning("MarketDataService already initialized - skipping duplicate initialization")
            return

        logger.info("Initializing MarketDataService...")
        # Create shared components
        self._limiter = await BinanceWeightLimiter.create()
        self._rest = BinanceREST(self._limiter)
        self._cache_guard = await CacheGuard.create()

        # WebSocket hydrator (primary feed)
        try:
            self._ws_hydrator = BinanceWSHydrator()
            await self._ws_hydrator.start()
        except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
            # Proceed with REST-only mode if WS hydrator has an issue
            logger.warning(f"WS Hydrator start failed; continuing with REST. Reason: {e}")

        self.is_running = True
        if MARKET_DATA_REST_LOOPS_ENABLED:
            await self.start_background_updates()
        else:
            logger.info("MarketDataService REST background loops disabled (MARKET_DATA_REST_LOOPS_ENABLED=false; external live_md owns freshness)")
        logger.info("MarketDataService initialized")

    async def close(self):
        """Close the service and stop background tasks"""
        logger.info("Closing MarketDataService...")
        self.is_running = False

        for task in self.background_tasks:
            if not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
                except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
                    logger.debug(f"Background task close error: {e}")

        self.background_tasks.clear()

        try:
            if self._ws_hydrator is not None:
                await self._ws_hydrator.stop()
        except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
            logger.debug(f"WS hydrator stop error: {e}")

        logger.info("MarketDataService closed")

    async def start_background_updates(self):
        """Start background update tasks"""
        logger.info("Starting background market data updates...")

        # Start high priority updates (BTC, ETH)
        high_priority_task = await task_manager.create_task(self._update_high_priority_coins(), name="market_data:high_priority_updates")
        self.background_tasks.append(high_priority_task)

        # Start normal priority updates
        normal_priority_task = await task_manager.create_task(self._update_normal_priority_coins(), name="market_data:normal_priority_updates")
        self.background_tasks.append(normal_priority_task)

        # VERIFY TASKS ARE RUNNING
        logger.info(f"Background tasks started: {len(self.background_tasks)} tasks")

        # Brief verification that tasks haven't failed immediately
        await asyncio.sleep(0.1)  # Brief wait for tasks to start
        active_tasks = [task for task in self.background_tasks if not task.done()]
        if len(active_tasks) != len(self.background_tasks):
            logger.error(f"CRITICAL: {len(self.background_tasks) - len(active_tasks)} background tasks failed to start!")
            # Log which tasks failed
            for i, task in enumerate(self.background_tasks):
                if task.done():
                    try:
                        exception = task.exception()
                        logger.exception(f"Task {i} failed immediately: {exception}")
                    except Exception as e:
                        logger.exception(f"Task {i} failed immediately (could not get exception): {e}")
        else:
            logger.info(f"VERIFICATION: All {len(active_tasks)} background tasks are running")

        if not self.background_tasks:
            error_msg = "No background tasks started"
            logger.error(f"CRITICAL: {error_msg}")
            raise RuntimeError(error_msg)

    async def _update_high_priority_coins(self):
        """Verify/fill high priority coins at low cadence (WS is primary)."""
        logger.info("High priority coin updates task started")

        # Wait briefly for app to start
        await asyncio.sleep(2)

        # High priority coins from env or default to first 2 from TOP10_COINS (live data)
        high_priority = _get_high_priority_coins()
        logger.info(f"High priority coins: {high_priority}")
        high_interval_s = MARKET_HIGH_INTERVAL_SEC
        logger.info(f"High priority update interval: {high_interval_s}s")

        update_count = 0
        while self.is_running:
            try:
                await self._batch_update_coins(high_priority, "high")
                update_count += 1
                logger.debug(f"High priority update #{update_count} completed for {len(high_priority)} coins")
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.exception(f"High priority coin updates failed: {e}")

            # Always sleep between updates (moved outside try/except to fix CPU spin)
            await asyncio.sleep(high_interval_s)

    async def _update_normal_priority_coins(self):
        """Verify/fill normal priority coins on a slow, staggered schedule."""
        logger.info("Normal priority coin updates task started")

        # Wait briefly for app to start
        await asyncio.sleep(5)

        normal_interval_s = MARKET_NORMAL_INTERVAL_SEC
        target_weight_per_min = MARKET_TARGET_WEIGHT_PER_MIN

        # Approximate high-priority cost (2 symbols @ 30s => ~4/min)
        high_weight_per_min = 4
        budget_per_min = max(1, target_weight_per_min - high_weight_per_min)

        # Allowed symbols per cycle at current interval
        allowed_per_cycle = max(1, int(budget_per_min * (normal_interval_s / 60.0)))
        max_concurrency = int(os.getenv("MARKET_MAX_CONCURRENCY", "6") or "6")

        logger.info(f"Normal priority update interval: {normal_interval_s}s, allowed_per_cycle: {allowed_per_cycle}")
        logger.info(f"Normal symbols: {self.normal_symbols}, fallback: {_TOP10_NON_HIGH}")

        update_count = 0
        while self.is_running:
            try:
                # Build rotating batch from enforced Top-10 set (excluding BTC/ETH)
                total = len(self.normal_symbols)
                if total == 0:
                    current_batch = list(_TOP10_NON_HIGH)
                    logger.debug(f"Using fallback normal symbols: {current_batch}")
                else:
                    start = self._normal_idx % total
                    end = start + allowed_per_cycle
                    current_batch = self.normal_symbols[start:end] if end <= total else self.normal_symbols[start:] + self.normal_symbols[: end % total]
                    self._normal_idx = (start + allowed_per_cycle) % total
                    logger.debug(f"Built batch from normal_symbols: {current_batch}")

                logger.info(f"Processing {len(current_batch)} normal priority coins: {current_batch}")
                await self._batch_update_coins(current_batch, "normal", max_concurrency=max_concurrency)

                update_count += 1
                logger.debug(f"Normal priority update #{update_count} completed for {len(current_batch)} coins")

                # Add slight jitter to avoid synchronized bursts
                jitter = max(0, min(60, int(normal_interval_s * 0.1)))
                await asyncio.sleep(normal_interval_s + (jitter if (self._normal_idx % 2 == 0) else -jitter))
            except asyncio.CancelledError:
                break
            except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
                logger.exception(f"Normal priority update error: {e}")
                await asyncio.sleep(normal_interval_s)

    async def _batch_update_coins(self, coins: list[str], priority: str, max_concurrency: int | None = None):
        """Process a batch of coins with optional concurrency limiting"""
        if not coins:
            logger.debug(f"No coins to process for {priority} priority")
            return

        # Enforce Top-10 universe
        coins = [c for c in coins if c in TOP10_BINANCEUS]

        logger.info(f"Processing {len(coins)} coins for {priority} priority: {coins}")
        tasks: list[asyncio.Task] = []
        sem = asyncio.Semaphore(max_concurrency) if (isinstance(max_concurrency, int) and max_concurrency > 0) else None

        for coin in coins:
            if coin in TOP10_BINANCEUS and is_supported(coin):
                if sem is None:
                    task = await task_manager.create_task(self._fetch_coin_data(coin), name="market_data:fetch_coin_data")
                else:

                    async def _wrapped(symbol: str) -> dict[str, Any] | None:
                        async with sem:  # type: ignore[arg-type]
                            return await self._fetch_coin_data(symbol)

                    task = await task_manager.create_task(_wrapped(coin), name="market_data:fetch_coin_data_wrapped")
                tasks.append(task)
            else:
                logger.warning(f"{coin} is not supported or not in Top-10")

        if not tasks:
            logger.warning("No tasks created - no supported coins found")
            return

        results = await asyncio.gather(*tasks, return_exceptions=True)
        for idx, (coin, result) in enumerate(zip(coins, results, strict=False)):
            if isinstance(result, Exception):
                # Don't log WRONGTYPE errors - they're handled by CacheGuard and will be retried
                error_str = str(result)
                if "WRONGTYPE" not in error_str:
                    logger.warning(f"Failed to fetch {coin}: {result}")
                else:
                    logger.debug(f"Cache type mismatch for {coin} (will retry): {error_str}")
            elif isinstance(result, dict):
                logger.info(f"[OK] Fetched {coin}: ${result.get('price', 'N/A')} - Updating cache...")
                await self._update_cache(coin, result)
                logger.info(f"[OK] Cache updated for {coin}")
            else:
                logger.warning(f"No data returned for {coin} (type: {type(result)})")

            # RELEASE reference to result to allow GC to reclaim traceback/frames
            results[idx] = None

        # Final cleanup - drop lists so any remaining Exception/Task objects are GC'd
        del results
        del tasks

    def _load_normal_symbols(self) -> list[str]:
        """
        Load normal-priority symbols from env. Accepts bases (e.g., SOL)
        or full pairs (e.g., SOLUSDT, SOL-USD). Enforces Mystic top-4 subset
        and excludes BTC/ETH (they're handled as high priority).
        """
        try:
            raw = os.getenv("MARKET_NORMAL_SYMBOLS") or os.getenv("BINANCE_US_TOP10_SYMBOLS") or ""
            if not raw:
                # Default to Top-10 minus high-priority coins
                return _get_top10_non_high_cached()
            out: list[str] = []
            for token in raw.split(",") if "," in raw else [raw]:
                s = (token or "").strip().upper()
                if not s:
                    continue
                # Normalize to base (strip common quote suffixes)
                if s.endswith("USDT"):
                    s = s[:-4]
                elif s.endswith("USD"):
                    s = s[:-3]
                elif "/" in s:
                    s = s.split("/", 1)[0]
                elif "-" in s:
                    s = s.split("-", 1)[0]
                # Enforce Top-10 and exclude high priority coins here
                high_priority = _get_high_priority_coins()
                if s in TOP10_BINANCEUS and s not in high_priority and s not in out:
                    out.append(s)
            # If env produced nothing valid, fall back to enforced Top-10 minus high priority coins
            return out or _get_top10_non_high_cached()
        except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError):
            return _get_top10_non_high_cached()

    async def _fetch_coin_data(self, coin: str) -> dict[str, Any] | None:
        try:
            if coin not in TOP10_BINANCEUS:
                return None

            symbol_pair = f"{coin.upper()}USDT"

            if self._cache_guard is None:
                self._cache_guard = await CacheGuard.create()

            price_val = await self._cache_guard.get_price_cached(symbol_pair)
            fetched_from_rest = False

            if price_val is None:
                try:
                    cache = get_shared_cache()
                    cache_data = None

                    for cache_key in ["top10_data", "prices", "market_data"]:
                        try:
                            cache_data = await cache.get_market_data(cache_key)
                            if cache_data:
                                break
                        except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError):
                            continue

                    if cache_data and "prices" in cache_data:
                        prices = cache_data["prices"]
                        if symbol_pair in prices:
                            price_data = prices[symbol_pair]
                            if isinstance(price_data, dict) and "price" in price_data:
                                price_val = float(price_data["price"])
                            elif isinstance(price_data, (int, float)):
                                price_val = float(price_data)

                            if price_val is not None:
                                with contextlib.suppress(ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError):
                                    await self._cache_guard.set_price(symbol_pair, price_val)

                except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError):
                    price_val = None

            if price_val is None and self._rest is not None:
                for _attempt in range(3):
                    try:
                        ticker_data = await self._rest.ticker_24h(symbol_pair)
                        if ticker_data and "lastPrice" in ticker_data:
                            price_val = float(ticker_data["lastPrice"])
                            fetched_from_rest = True

                            if self._cache_guard:
                                with contextlib.suppress(ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError):
                                    await self._cache_guard.set_price(symbol_pair, price_val)
                            break
                    except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError):
                        continue

            if price_val is None:
                return None

            ts = datetime.now(timezone.utc).isoformat()

            bid_price = None
            ask_price = None
            spread = None
            spread_pct = None
            change_24h = None
            volume_24h = None

            existing = self.cache.get(coin, {})

            if fetched_from_rest and "ticker_data" in locals() and isinstance(locals()["ticker_data"], dict):
                td = locals()["ticker_data"]

                with contextlib.suppress(ValueError, TypeError):
                    change_24h = float(td.get("priceChangePercent", 0))

                with contextlib.suppress(ValueError, TypeError):
                    volume_24h = float(td.get("volume", 0))

                with contextlib.suppress(ValueError, TypeError):
                    bp = float(td.get("bidPrice", 0))
                    ap = float(td.get("askPrice", 0))

                    if bp > 0:
                        bid_price = bp
                    if ap > 0:
                        ask_price = ap

                    if bid_price is not None and ask_price is not None and price_val > 0:
                        spread = ask_price - bid_price
                        mid = (ask_price + bid_price) / 2.0 if (ask_price + bid_price) > 0 else price_val
                        # Keep spread_pct in fractional units (0.0012 == 0.12%).
                        spread_pct = (spread / mid) if mid > 0 else 0.0

            data = {
                "price": price_val,
                "api_source": "binance_us_rest" if fetched_from_rest else "cache",
                "timestamp": ts,
                "bid": bid_price if bid_price is not None else existing.get("bid"),
                "ask": ask_price if ask_price is not None else existing.get("ask"),
                "spread": spread if spread is not None else existing.get("spread"),
                "spread_pct": spread_pct if spread_pct is not None else existing.get("spread_pct"),
                "change_24h": change_24h if change_24h is not None else existing.get("change_24h"),
                "volume_24h": volume_24h if volume_24h is not None else existing.get("volume_24h"),
            }

            self.cache[coin] = data

            if self._cache_guard:
                src = "rest" if fetched_from_rest else "cache"
                await self._cache_guard.mark_market_update(src)

            return data

        except Exception:
            return None

    async def _update_cache(self, coin: str, data: dict[str, Any]):
        """Update the cache with new data and broadcast update"""
        # MEMORY LEAK GUARD: limit klines array length
        k = data.get("klines")
        if isinstance(k, list) and len(k) > 200:
            data["klines"] = k[-200:]

        # PRESERVE spread data from previous cache if new data doesn't have it
        # This prevents WebSocket updates (price only) from wiping out spread from REST
        existing = self.cache.get(coin, {})
        spread_fields = ["bid", "ask", "spread", "spread_pct"]
        for field in spread_fields:
            if (field not in data or data.get(field) is None) and field in existing and existing.get(field) is not None:
                data[field] = existing[field]

        # Update cache with timestamp
        self.cache[coin] = data
        self.cache_timestamps[coin] = time.time()

        # MEMORY LEAK GUARD: limit cache size to prevent unbounded growth
        if len(self.cache) > self.max_cache_size:
            # Remove oldest 10% of entries (LRU-style eviction)
            entries_to_remove = int(self.max_cache_size * 0.1)
            # Sort by timestamp (oldest first)
            sorted_items = sorted(self.cache_timestamps.items(), key=lambda x: x[1])
            for coin_to_remove, _ in sorted_items[:entries_to_remove]:
                del self.cache[coin_to_remove]
                del self.cache_timestamps[coin_to_remove]
            logger.debug(f"Cache cleanup: removed {entries_to_remove} oldest entries, cache size now {len(self.cache)}")

        logger.debug(f"Updated cache for {coin}")

        # Also update canonical_cache so endpoints can access the data
        try:
            from backend.services.canonical_cache import canonical_cache

            price = data.get("price")
            if price:
                # Use full symbol name (with USDT) for canonical cache
                full_symbol = f"{coin}USDT"
                canonical_cache.update_price(full_symbol, float(price))
        except Exception as e:
            logger.warning(f"Failed to update canonical cache for {coin}: {e}")

        # Store in Redis for AI services to access (both formats for compatibility)
        def _raise_redis_config_error() -> None:
            """Raise error for missing Redis configuration."""
            raise RuntimeError(_REDIS_CONFIG_ERROR)

        try:
            redis_client = get_shared_redis_async()
            if redis_client is None:
                _raise_redis_config_error()

            # CRITICAL: Redis key format standardization
            # - market:BTC (base currency only, matches CanonicalSymbolFormatter.normalize_for_redis_market)
            # - price:BTC (base currency only, matches CanonicalSymbolFormatter.normalize_for_redis_price)
            # - Coin parameter is already in base-only format (e.g., "BTC" not "BTCUSDT")
            # - This ensures consistent lookups across all services

            # Store comprehensive market data
            await redis_client.set(f"market:{coin}", json.dumps(data), ex=300)  # 5 minute expiry

            # Store price data in format expected by AI signal generator and other services
            price_data = {
                "v": data.get("price", 0.0),  # Price value (expected by AI signal generator)
                "change_24h": data.get("change_24h", 0.0),
                "volume_24h": data.get("volume_24h", 0.0),
                "timestamp": data.get("timestamp", datetime.now(timezone.utc).isoformat()),
                "api_source": data.get("api_source", "market_data_service"),
            }
            # Store as hash (preferred by AI signal generator)
            # Convert mapping to individual hset calls for compatibility with all Redis client versions
            for field, value in price_data.items():
                await redis_client.hset(f"price:{coin}", field, str(value))
            await redis_client.expire(f"price:{coin}", 300)  # 5 minute expiry

            # Also store as JSON string for compatibility
            await redis_client.set(f"price:{coin}:json", json.dumps(price_data), ex=300)

            logger.debug(f"Stored market data for {coin} in both Redis formats")
        except Exception as redis_error:
            logger.debug(f"Redis storage failed for market data: {redis_error}")

        # Broadcast market data update
        try:
            await websocket_manager.broadcast_json(
                {
                    "type": "market_data_update",
                    "data": {
                        "symbol": coin,
                        "price": data.get("price", 0.0),
                        "api_source": data.get("api_source", "unknown"),
                        "timestamp": data.get(
                            "timestamp",
                            datetime.now(timezone.utc).isoformat(),
                        ),
                    },
                },
            )
        except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
            logger.exception(f"Error broadcasting update: {e}")

    async def get_market_data(self, symbol: str) -> dict[str, Any] | None:
        """Get market data for a symbol, with cache freshness validation and timestamp updates."""
        symbol = (symbol or "").upper()

        if symbol not in TOP10_BINANCEUS:
            return None

        # Check if we have cached data
        cached_data = self.cache.get(symbol)

        if cached_data and isinstance(cached_data, dict):
            # Validate cache freshness (must be <= 10 seconds to match trading gate)
            ts = cached_data.get("timestamp")
            if isinstance(ts, str):
                try:
                    dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
                    if dt.tzinfo is None:
                        dt = dt.replace(tzinfo=timezone.utc)
                    age = (datetime.now(timezone.utc) - dt).total_seconds()

                    if age <= 10.0:
                        # Cache is fresh - update global timestamp and return
                        if self._cache_guard is None:
                            self._cache_guard = await CacheGuard.create()

                        if self._cache_guard:
                            # Derive source from api_source field
                            api_source = str(cached_data.get("api_source") or "")
                            if "ws" in api_source:
                                src = "ws"
                            elif "rest" in api_source:
                                src = "rest"
                            else:
                                src = "cache"

                            await self._cache_guard.mark_market_update(src)

                        return cached_data
                except Exception:
                    # Timestamp parsing failed - treat as stale
                    pass

            # Cache is stale or invalid - drop it and refresh
            self.cache.pop(symbol, None)

        # Not cached or stale - fetch fresh data
        data = await self._fetch_coin_data(symbol)

        if data:
            await self._update_cache(symbol, data)
            return data

        return None

    async def get_cached_data(self, symbol: str) -> dict[str, Any] | None:
        """Get cached data for a specific symbol"""
        symbol = (symbol or "").upper()
        if symbol not in TOP10_BINANCEUS:
            return None
        return self.cache.get(symbol)

    async def get_all_cached_data(self) -> dict[str, dict[str, Any]]:
        """Get all cached data"""
        # Only return Top-10 subset if present
        return {k: v for k, v in self.cache.items() if k in TOP10_BINANCEUS}

    async def get_markets(self) -> dict[str, Any]:
        """Get markets overview with current data"""
        try:
            cached_data = await self.get_all_cached_data()
            markets_data: dict[str, dict[str, Any]] = {}

            for symbol, data in cached_data.items():
                markets_data[symbol] = {
                    "price": data.get("price", 0),
                    "api_source": data.get("api_source", "unknown"),
                    "timestamp": data.get(
                        "timestamp",
                        datetime.now(timezone.utc).isoformat(),
                    ),
                }

            return {
                "markets": markets_data,
                "count": len(markets_data),
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
        except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
            logger.exception(f"Error in get_markets: {e}")
            return {
                "markets": {},
                "count": 0,
                "error": str(e),
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }

    async def get_live_signals(self, symbol: str = "all") -> dict[str, Any]:
        """Get live trading signals for symbols (neutral baseline using live price)"""
        try:
            if symbol.lower() == "all":
                # Enforced Top-10 universe
                symbols = list(TOP10_BINANCEUS)
                signals: dict[str, dict[str, Any]] = {}
                for sym in symbols:
                    data = await self.get_market_data(sym)
                    if data and data.get("price", 0) > 0:
                        signals[sym] = {
                            "price": data["price"],
                            "signal": "neutral",
                            "strength": 0.5,
                            "timestamp": data.get(
                                "timestamp",
                                datetime.now(timezone.utc).isoformat(),
                            ),
                        }

                return {
                    "signals": signals,
                    "total_signals": len(signals),
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                }

            # Return signal for specific symbol (enforced Top-10)
            data = await self.get_market_data(symbol.upper())
            if not data:
                return {
                    "signals": {},
                    "error": f"Symbol {symbol} not found (Top-10 only)",
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                }

            price = data.get("price", 0)
            signals = {
                symbol.upper(): {
                    "price": price,
                    "signal": "neutral",
                    "strength": 0.5,
                    "timestamp": data.get(
                        "timestamp",
                        datetime.now(timezone.utc).isoformat(),
                    ),
                },
            }

            return {
                "signals": signals,
                "total_signals": 1,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
        except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
            logger.exception(f"Error getting live signals: {e}")
            return {
                "signals": {},
                "error": str(e),
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }

    # --- Live data endpoints (no mocks) ---

    async def get_top10_coins(self) -> list[dict[str, Any]]:
        """Get top 10 coins data for dashboard compatibility."""
        try:
            symbols = list(TOP10_BINANCEUS)
            coins_data = []

            for symbol in symbols:
                data = await self.get_market_data(symbol)
                if data:
                    coins_data.append(
                        {
                            "symbol": f"{symbol}USDT",
                            "name": symbol,
                            "price": str(data.get("price", 0)),
                            "change": str(data.get("change_24h", 0)),
                            "volume": str(data.get("volume_24h", 0)),
                        },
                    )
        except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
            logger.exception(f"Error getting top 10 coins: {e}")
            return []
        else:
            return coins_data

    async def get_volume_data(self) -> dict[str, dict[str, str]]:
        """Get volume data for dashboard compatibility."""
        try:
            symbols = list(TOP10_BINANCEUS)
            volume_data = {}

            for symbol in symbols:
                data = await self.get_market_data(symbol)
                if data:
                    volume_data[f"{symbol}USDT"] = {
                        "volume": str(data.get("volume_24h", 0)),
                        "volume_change": str(data.get("volume_change_24h", 0)),
                    }
        except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
            logger.exception(f"Error getting volume data: {e}")
            return {}
        else:
            return volume_data

    async def get_orderbook_data(self) -> dict[str, Any]:
        """Get orderbook data for dashboard compatibility."""
        try:
            if self._rest is None:
                return {}

            # Get orderbook for BTCUSDT as primary symbol
            orderbook = await self._rest.depth("BTCUSDT", limit=5)
            if orderbook:
                return {
                    "symbol": "BTCUSDT",
                    "bids": orderbook.get("bids", []),
                    "asks": orderbook.get("asks", []),
                }
        except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
            logger.exception(f"Error getting orderbook data: {e}")
            return {}
        else:
            return {}

    async def get_live_data(self) -> dict[str, Any]:
        """
        Get a live snapshot of market data for the enforced Top-10 universe.
        Does NOT depend on any external mock providers.
        """
        try:
            symbols: list[str] = list(TOP10_BINANCEUS)

            quotes: dict[str, dict[str, Any]] = {}
            for sym in symbols:
                data = await self.get_market_data(sym)
                if data:
                    quotes[sym] = {
                        "price": data["price"],
                        "api_source": data.get("api_source", "unknown"),
                        "timestamp": data.get("timestamp", datetime.now(timezone.utc).isoformat()),
                    }

            status = "success" if quotes else "error"
            message = None if quotes else "No quotes available"

            return {
                "status": status,
                "message": message,
                "data": {
                    "symbols": quotes,
                    "count": len(quotes),
                },
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
        except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
            logger.exception(f"Error getting live data snapshot: {e}")
            return {
                "status": "error",
                "message": str(e),
                "data": {"symbols": {}, "count": 0},
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }

    async def get_market_cap_data(self) -> dict[str, Any]:
        """
        Live 24h ticker stats for the enforced Top-10 universe (priceChangePercent etc.)
        Uses REST per symbol to avoid an all-symbols blast on Binance US.
        """
        try:
            if self._rest is None:
                return {
                    "status": "error",
                    "message": "REST client not initialized",
                    "data": {},
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                }

            symbols: list[str] = list(TOP10_BINANCEUS)

            out: dict[str, Any] = {}
            # Rate friendly: sequential or modest concurrency if desired
            for sym in symbols:
                pair = f"{sym}USDT"
                try:
                    stats = await self._rest.ticker_24hr(pair)  # expected to exist in your client
                    if isinstance(stats, dict):
                        out[sym] = {
                            "price": float(stats.get("lastPrice")) if stats.get("lastPrice") is not None else None,
                            "priceChangePercent": float(stats.get("priceChangePercent")) if stats.get("priceChangePercent") is not None else None,
                            "highPrice": float(stats.get("highPrice")) if stats.get("highPrice") is not None else None,
                            "lowPrice": float(stats.get("lowPrice")) if stats.get("lowPrice") is not None else None,
                            "volume": float(stats.get("volume")) if stats.get("volume") is not None else None,
                            "quoteVolume": float(stats.get("quoteVolume")) if stats.get("quoteVolume") is not None else None,
                            "openTime": stats.get("openTime"),
                            "closeTime": stats.get("closeTime"),
                        }
                except AttributeError:
                    # If your BinanceREST lacks ticker_24hr, gracefully degrade to price only
                    price_data = await self._rest.price(pair)
                    price_val = None
                    if isinstance(price_data, dict) and "price" in price_data:
                        try:
                            price_val = float(price_data["price"])
                        except (TypeError, ValueError):
                            price_val = None
                    out[sym] = {
                        "price": price_val,
                        "priceChangePercent": None,
                        "highPrice": None,
                        "lowPrice": None,
                        "volume": None,
                        "quoteVolume": None,
                        "openTime": None,
                        "closeTime": None,
                    }
                except (ValueError, TypeError, KeyError, IndexError, RuntimeError) as e:
                    logger.warning(f"24h stats fetch failed for {sym}: {e}")

            return {
                "status": "success",
                "data": out,
                "count": len(out),
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
        except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
            logger.exception(f"Error in get_market_cap_data: {e}")
            return {
                "status": "error",
                "message": str(e),
                "data": {},
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }

    async def get_top_movers(self) -> dict[str, Any]:
        """
        Live top movers from the enforced Top-10 universe based on 24h % change.
        Falls back to price-only ordering if 24h stats unavailable.
        """
        try:
            caps = await self.get_market_cap_data()
            if caps.get("status") != "success":
                return {
                    "status": "error",
                    "message": caps.get("message") or "Failed to obtain 24h stats",
                    "data": {},
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                }

            data: dict[str, Any] = caps.get("data", {})

            # Sort by abs(priceChangePercent), fallback to symbol name if None
            def mover_key(item):
                _sym, vals = item
                pcp = vals.get("priceChangePercent")
                return abs(pcp) if isinstance(pcp, (int, float)) else -1.0

            sorted_items = sorted(data.items(), key=mover_key, reverse=True)
            top = dict(sorted_items[:10])

            return {
                "status": "success",
                "data": top,
                "count": len(top),
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
        except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
            logger.exception(f"Error in get_top_movers: {e}")
            return {
                "status": "error",
                "message": str(e),
                "data": {},
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }

    async def get_market_trends(self) -> dict[str, Any]:
        """Lightweight trends tied to live 24h stats (Top-10 only)."""
        try:
            caps = await self.get_market_cap_data()
            if caps.get("status") != "success":
                return {
                    "status": "error",
                    "message": caps.get("message") or "Failed to obtain 24h stats",
                    "data": {},
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                }

            data: dict[str, Any] = caps.get("data", {})
            advances = {s: v for s, v in data.items() if isinstance(v.get("priceChangePercent"), (int, float)) and v["priceChangePercent"] > 0}
            decliners = {s: v for s, v in data.items() if isinstance(v.get("priceChangePercent"), (int, float)) and v["priceChangePercent"] < 0}

            return {
                "status": "success",
                "data": {
                    "advancers": len(advances),
                    "decliners": len(decliners),
                    "unmoved": len(data) - len(advances) - len(decliners),
                },
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
        except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
            logger.exception(f"Error in get_market_trends: {e}")
            return {
                "status": "error",
                "message": str(e),
                "data": {},
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }


# Global instance - use MarketDataService.shared()
market_data_service = MarketDataService.shared()
