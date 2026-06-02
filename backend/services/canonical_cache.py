"""
Canonical Cache System for Mystic Trading Platform
Single source of truth for all caching operations with unified policies
"""

import asyncio
import contextlib
import json
import logging
import os
import time
from datetime import datetime, timezone
from typing import Any, Optional

import redis.asyncio as aioredis

# Import from single source of truth
try:
    from backend.config.trading_universe import EXCHANGE_ID, TRADING_SYMBOLS
except (ImportError, ModuleNotFoundError, AttributeError, ValueError, TypeError, RuntimeError) as e:
    msg = f"Failed to import trading_universe: {e}"
    raise RuntimeError(msg) from e

# Import feature store for database fallback
try:
    from backend.services.feature_store import get_ohlcv_recent
except (ImportError, ModuleNotFoundError, AttributeError, ValueError, TypeError, RuntimeError):
    get_ohlcv_recent = None  # type: ignore[assignment, misc]

# Import shared Redis client to prevent connection leaks
try:
    from backend.config.redis_config import get_shared_redis_async
except (ImportError, ModuleNotFoundError, AttributeError, ValueError, TypeError, RuntimeError):
    get_shared_redis_async = None  # type: ignore[assignment]

# Lazy imports for optional dependencies
try:
    from backend.config.settings import settings
except (ImportError, ModuleNotFoundError, AttributeError, ValueError, TypeError, RuntimeError):
    settings = None  # type: ignore[assignment]

try:
    from backend.services.binanceus_top10_fetcher import binance_us_top10_fetcher
except (ImportError, ModuleNotFoundError, AttributeError, ValueError, TypeError, RuntimeError):
    binance_us_top10_fetcher = None  # type: ignore[assignment]

try:
    from backend.services.task_manager import task_manager
except (ImportError, ModuleNotFoundError, AttributeError, ValueError, TypeError, RuntimeError):
    task_manager = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)


class CacheEntry:
    """Cache entry with TTL and metadata"""

    def __init__(self, key: str, value: Any, ttl: int = 300) -> None:
        self.key = key
        self.value = value
        self.timestamp = time.time()
        self.ttl = ttl
        self.access_count = 0
        self.last_accessed = self.timestamp

    @property
    def is_expired(self) -> bool:
        return time.time() - self.timestamp > self.ttl

    def touch(self) -> None:
        """Update access time and count"""
        self.access_count += 1
        self.last_accessed = time.time()


class CanonicalCache:
    """
    Single canonical cache system with Redis + in-memory TTL layer.
    Replaces all fragmented cache implementations.
    """

    _instance: Optional["CanonicalCache"] = None
    _lock: asyncio.Lock | None = None  # Lazy init to avoid event loop issues at import

    _initialized: bool = False

    def __init__(self) -> None:
        if self._initialized:
            return
        self.redis_client: aioredis.Redis | None = None
        self.local_cache: dict[str, CacheEntry] = {}
        self.max_local_size = 1000
        self.default_ttl = 300  # 5 minutes
        self.freshness_threshold = 30  # 30 seconds max age for live data

        # Request deduplication
        self.pending_requests: dict[str, asyncio.Future] = {}
        self.request_timeout = 30.0  # 30 second timeout for complex operations
        self.max_pending_requests = 1000  # Maximum pending requests to prevent unbounded growth

        # Circuit breaker
        self.circuit_breaker_threshold = 5
        self.circuit_breaker_timeout = 60
        self.circuit_breaker_failures = 0
        self.circuit_breaker_last_failure = 0
        self.circuit_breaker_open = False

        # Metrics
        self.metrics = {
            "hits": 0,
            "misses": 0,
            "redis_hits": 0,
            "redis_misses": 0,
            "local_hits": 0,
            "local_misses": 0,
            "sets": 0,
            "deletes": 0,
            "errors": 0,
            "deduplicated_requests": 0,
            "circuit_breaker_trips": 0,
        }
        # Track background tasks for proper cleanup
        self._tasks: list[asyncio.Task[Any]] = []
        self.__class__._initialized = True

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    async def initialize(self, redis_url: str | None = None) -> bool:
        """Initialize Redis connection with retry logic"""
        logger.info("[CACHE] initialize() called")
        # All Live Data, No Fallback/Hardcoded Data
        # Get Redis URL from parameter or environment variable
        if not redis_url:
            redis_url = os.getenv("REDIS_URL")
            if not redis_url:
                redis_host = os.getenv("REDIS_HOST")
                if not redis_host:
                    msg = "REDIS_URL or REDIS_HOST environment variable is required - no fallback/hardcoded Redis URL"
                    raise RuntimeError(msg)
                redis_port = os.getenv("REDIS_PORT", "6379")
                redis_db = os.getenv("REDIS_DB", "0")
                redis_url = f"redis://{redis_host}:{redis_port}/{redis_db}"
        logger.info(f"[CACHE] Using redis_url: {redis_url}")
        try:
            # Try multiple times to connect to Redis with increasing backoff
            max_retries = 3
            retry_delay = 1.0  # seconds

            for attempt in range(max_retries):
                logger.info(f"[CACHE] Attempt {attempt + 1} to connect to Redis")
                try:
                    # CRITICAL FIX: Use shared Redis client instead of creating new connections
                    # This prevents connection leaks (was creating 3850+ connections)
                    if get_shared_redis_async is not None:
                        logger.info("[CACHE] Using shared Redis client (prevents connection leaks)...")
                        self.redis_client = get_shared_redis_async()
                    else:
                        # Fallback: create new client (should rarely happen)
                        logger.warning("[CACHE] Shared Redis not available, creating new client...")
                        self.redis_client = aioredis.from_url(
                            redis_url,
                            encoding="utf-8",
                            decode_responses=True,
                            socket_timeout=10,
                            socket_connect_timeout=10,
                            protocol=2,
                            max_connections=50,  # Limit connections
                        )

                    logger.info("[CACHE] Redis client ready, pinging...")
                    await asyncio.wait_for(self.redis_client.ping(), timeout=5.0)
                    logger.info(f"Canonical cache initialized with shared Redis on attempt {attempt + 1}")

                    # Verify connection by setting and getting a test value
                    test_key = "canonical_cache:init_test"
                    test_value = f"test_{int(time.time())}"
                    await self.redis_client.setex(test_key, 60, test_value)
                    retrieved = await self.redis_client.get(test_key)

                    if retrieved == test_value:
                        logger.info("Redis connection verified with test read/write")

                        # Start periodic cleanup task for pending requests
                        # Use asyncio.create_task directly to avoid potential lock issues during startup
                        cleanup_task = asyncio.create_task(self._run_periodic_cleanup())
                        # Store task reference for proper cleanup
                        if not hasattr(self, "_cleanup_tasks"):
                            self._cleanup_tasks = []
                        self._cleanup_tasks.append(cleanup_task)

                        break
                    else:
                        logger.warning(f"Redis test failed: expected {test_value}, got {retrieved}")
                # Continue to retry
                except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError, ConnectionError, OSError, asyncio.TimeoutError) as retry_error:
                    logger.warning(f"Redis connection attempt {attempt + 1} failed: {retry_error}")
                    if self.redis_client:
                        with contextlib.suppress(ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError):
                            await self.redis_client.close()
                        self.redis_client = None

                    if attempt < max_retries - 1:
                        wait_time = retry_delay * (2**attempt)  # Exponential backoff
                        logger.info(f"Retrying Redis connection in {wait_time} seconds...")
                        await asyncio.sleep(wait_time)
                    else:
                        raise  # Re-raise on last attempt

            else:
                # If we get here, all retries failed
                logger.warning(f"Redis unavailable after {max_retries} attempts, using local cache only")
                self.redis_client = None
                return False
        except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
            logger.warning(f"Redis unavailable, using local cache only: {e}")
            self.redis_client = None
            return False
        else:
            return True

    def _prune_tasks(self) -> None:
        """Drop finished background tasks to prevent unbounded growth."""
        if not self._tasks:
            return

        alive: list[asyncio.Task[Any]] = []
        for task in self._tasks:
            if task.done():
                # Touch exception/result so tracebacks can be released
                with contextlib.suppress(Exception):
                    _ = task.exception()
            else:
                alive.append(task)
        self._tasks = alive

    async def get(self, key: str, default: Any = None) -> Any:
        """Get value from cache with TTL enforcement and request deduplication"""
        try:
            # Check circuit breaker
            if self._is_circuit_breaker_open():
                logger.warning(f"Circuit breaker open for cache key {key}")
                return default

            # Check local cache first
            if key in self.local_cache:
                entry = self.local_cache[key]
                if not entry.is_expired:
                    entry.touch()
                    self.metrics["hits"] += 1
                    self.metrics["local_hits"] += 1
                    return entry.value
                # Remove expired entry
                del self.local_cache[key]

            # Check for pending request (deduplication)
            if key in self.pending_requests:
                logger.info(f"Deduplicating request for key {key}")
                self.metrics["deduplicated_requests"] += 1
                try:
                    return await asyncio.wait_for(self.pending_requests[key], timeout=self.request_timeout)
                except asyncio.TimeoutError:
                    logger.warning(f"Pending request timeout for key {key}")
                    del self.pending_requests[key]
                    return default

            # Create new request future
            future = asyncio.Future()
            # Track creation time for timeout cleanup
            future._created_time = time.time()  # type: ignore[attr-defined]

            # Enforce max_pending_requests limit (evict oldest when full)
            if len(self.pending_requests) >= self.max_pending_requests:
                # Find oldest pending request to evict
                oldest_key = min(self.pending_requests.keys(), key=lambda k: getattr(self.pending_requests[k], "_created_time", time.time()))
                evicted_future = self.pending_requests.pop(oldest_key, None)
                if evicted_future and not evicted_future.done():
                    evicted_future.cancel()
                logger.warning(f"Evicted oldest pending request for key {oldest_key} due to limit")

            self.pending_requests[key] = future

            try:
                # Check Redis if available
                if self.redis_client:
                    try:
                        # Try string first (most common format), then hash
                        try:
                            raw_value = await asyncio.wait_for(self.redis_client.get(key), timeout=self.request_timeout)
                        except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError):
                            # Fallback to hash for price/ticker keys
                            if key.startswith(("price:", "ticker:")):
                                raw_value = await asyncio.wait_for(
                                    self.redis_client.hgetall(key),
                                    timeout=self.request_timeout,
                                )
                            else:
                                raw_value = None
                        if raw_value is not None:
                            # Handle hash data differently
                            if isinstance(raw_value, dict):
                                value = raw_value
                            else:
                                try:
                                    value = json.loads(raw_value)
                                except (json.JSONDecodeError, TypeError):
                                    value = raw_value

                            # Store in local cache
                            self.local_cache[key] = CacheEntry(key, value, self.default_ttl)
                            self._cleanup_local_cache()

                            self.metrics["hits"] += 1
                            self.metrics["redis_hits"] += 1
                            future.set_result(value)
                            return value
                    except asyncio.TimeoutError:
                        logger.warning(f"Redis timeout for key {key}")
                        self._record_circuit_breaker_failure()
                    except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
                        logger.warning(f"Redis get error for key {key}: {e}")
                        self._record_circuit_breaker_failure()

                self.metrics["misses"] += 1
                self.metrics["local_misses"] += 1
                future.set_result(default)
                return default

            finally:
                # Clean up pending request
                if key in self.pending_requests:
                    del self.pending_requests[key]

        except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
            logger.exception(f"Cache get error for key {key}: {e}")
            self.metrics["errors"] += 1
            self._record_circuit_breaker_failure()
            return default

    async def set(self, key: str, value: Any, ttl: int | None = None) -> bool:
        """Set value in cache with TTL"""
        try:
            ttl = ttl or self.default_ttl

            # Store in local cache
            self.local_cache[key] = CacheEntry(key, value, ttl)
            self._cleanup_local_cache()

            # Store in Redis if available
            if self.redis_client:
                try:
                    serialized_value = json.dumps(value) if isinstance(value, (dict, list)) else str(value)

                    await self.redis_client.setex(key, ttl, serialized_value)
                except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
                    logger.warning(f"Redis set error for key {key}: {e}")

            self.metrics["sets"] += 1
        except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
            logger.exception(f"Cache set error for key {key}: {e}")
            self.metrics["errors"] += 1
            return False
        else:
            return True

    async def delete(self, key: str) -> bool:
        """Delete value from cache"""
        try:
            # Remove from local cache
            self.local_cache.pop(key, None)

            # Remove from Redis if available
            if self.redis_client:
                try:
                    await self.redis_client.delete(key)
                except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
                    logger.warning(f"Redis delete error for key {key}: {e}")

            self.metrics["deletes"] += 1
        except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
            logger.exception(f"Cache delete error for key {key}: {e}")
            self.metrics["errors"] += 1
            return False
        else:
            return True

    async def get_live_market_data(self) -> dict[str, Any]:
        """Get live market data with freshness enforcement"""
        try:
            data = await self.get("live_market_data")
            if data and self._is_data_fresh(data):
                return {
                    "status": "live",
                    "data": data,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "source": "canonical_cache",
                }
            return {
                "status": "stale",
                "data": data or {},
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "source": "canonical_cache",
            }
        except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
            logger.exception(f"Failed to get live market data: {e}")
            return {
                "status": "error",
                "data": {},
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "error": str(e),
            }

    async def get_latest_price(self, exchange: str, symbol: str) -> float | None:
        """Get latest price for a symbol from cache"""
        try:
            # Try Redis string format first (price:BTCUSDT) - most common
            redis_key = f"price:{symbol}"
            price_data = await self.get(redis_key)
            if price_data and isinstance(price_data, dict):
                # Extract price from JSON data
                price_val = price_data.get("price")
                if price_val:
                    return float(price_val)
            elif isinstance(price_data, (int, float)):
                return float(price_data)

            # Try hash format as fallback
            if isinstance(price_data, dict):
                price_val = price_data.get("v") or price_data.get("price") or price_data.get("last")
                if price_val:
                    return float(price_val)

            # Fallback to canonical cache format
            key = f"price_{exchange}_{symbol}"
            price_data = await self.get(key)
            if price_data and isinstance(price_data, dict):
                return float(price_data.get("price", 0))
            if isinstance(price_data, (int, float)):
                return float(price_data)

            # Try to get price from ticker data as last resort
            try:
                ticker_data = await self.get_latest_ticker_24h(exchange, symbol)
                if ticker_data:
                    # Try different price fields that might be in ticker data
                    for price_field in ["lastPrice", "price", "close", "c"]:
                        if ticker_data.get(price_field):
                            price_val = ticker_data[price_field]
                            if isinstance(price_val, (int, float)) or (isinstance(price_val, str) and price_val.replace(".", "", 1).isdigit()):
                                # Cache this price for future use
                                await self.set_latest_price(exchange, symbol, float(price_val))
                                return float(price_val)
            except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
                logger.debug(f"Failed to extract price from ticker for {symbol}: {e}")
        except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
            logger.warning(f"Failed to get latest price for {symbol}: {e}")
            return None
        else:
            return None

    async def get_price_history(self, exchange: str, symbol: str, limit: int = 100) -> list[dict[str, Any]]:
        """Get price history for a symbol from cache, with database fallback"""
        try:
            key = f"history_{exchange}_{symbol}"
            history_data = await self.get(key)
            if history_data and isinstance(history_data, list):
                return history_data[-limit:] if len(history_data) > limit else history_data

            # Fallback to database if not in cache
            if get_ohlcv_recent is not None:
                try:
                    # Convert symbol format: BTC/USDT -> BTC-USDT for database query
                    db_symbol = symbol.replace("/", "-")
                    ohlcv_data = get_ohlcv_recent(db_symbol, interval="1m", limit=limit)

                    if ohlcv_data and len(ohlcv_data) > 0:
                        # Convert database format to cache format (list of dicts with 'close' key)
                        history_list = [{"close": candle["close"], "timestamp": candle["ts"]} for candle in ohlcv_data]

                        # Cache for future use
                        await self.set(key, history_list)

                        return history_list
                except (ImportError, ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as db_error:
                    logger.debug(f"Database fallback failed for {symbol}: {db_error}")

        except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
            logger.warning(f"Failed to get price history for {symbol}: {e}")
            return []
        else:
            return []

    async def set_latest_price(self, exchange: str, symbol: str, price: float, timestamp: str | None = None) -> bool:
        """Set latest price for a symbol in cache"""
        try:
            key = f"price_{exchange}_{symbol}"
            price_data = {
                "price": price,
                "timestamp": timestamp or datetime.now(timezone.utc).isoformat(),
                "exchange": exchange,
                "symbol": symbol,
            }
            await self.set(key, price_data, ttl=60)  # 1 minute TTL for prices
        except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
            logger.exception(f"Failed to set latest price for {symbol}: {e}")
            return False
        else:
            return True

    async def set_price_history(self, exchange: str, symbol: str, history: list[dict[str, Any]]) -> bool:
        """Set price history for a symbol in cache"""
        try:
            key = f"history_{exchange}_{symbol}"
            await self.set(key, history, ttl=300)  # 5 minute TTL for history
        except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
            logger.exception(f"Failed to set price history for {symbol}: {e}")
            return False
        else:
            return True

    async def get_latest_ticker_24h(self, exchange: str, symbol: str) -> dict[str, Any] | None:
        """Get latest 24h ticker data for a symbol"""
        try:
            # Try Redis hash format first (ticker:BTCUSDT)
            redis_key = f"ticker:{symbol}"
            ticker_data = await self.get(redis_key)
            if ticker_data and isinstance(ticker_data, dict):
                return ticker_data
            # Fallback to canonical cache format
            key = f"ticker_24h_{exchange}_{symbol}"
            ticker_data = await self.get(key)
            if ticker_data and isinstance(ticker_data, dict):
                return ticker_data
        except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
            logger.warning(f"Failed to get 24h ticker for {symbol}: {e}")
            return None
        else:
            return None

    async def set_latest_ticker_24h(self, exchange: str, symbol: str, ticker_data: dict[str, Any]) -> bool:
        """Set latest 24h ticker data for a symbol in cache"""
        try:
            key = f"ticker_24h_{exchange}_{symbol}"
            ticker_data = dict(ticker_data)
            ticker_data["timestamp"] = datetime.now(timezone.utc).isoformat()
            ticker_data["exchange"] = exchange
            ticker_data["symbol"] = symbol
            await self.set(key, ticker_data, ttl=300)  # 5 minute TTL for ticker data
        except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
            logger.exception(f"Failed to set 24h ticker for {symbol}: {e}")
            return False
        else:
            return True

    def update_price(self, symbol: str, price: float) -> bool:
        """Update price for a symbol in cache - compatibility with shared_cache"""
        try:
            # Create a price data object
            price_data = {
                "price": price,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "symbol": symbol,
            }

            # Store in local cache immediately for fast access
            key = f"price:{symbol}"
            self.local_cache[key] = CacheEntry(key, price_data, ttl=60)  # 1 minute TTL for prices

            # Store in Redis asynchronously if available
            if self.redis_client:
                # MEMORY LEAK GUARD: keep internal task list bounded
                self._prune_tasks()
                if task_manager is not None:
                    task = task_manager.create_task_sync(self.redis_client.setex(key, 60, json.dumps(price_data)), name="canonical_cache:redis_setex")
                else:
                    task = asyncio.create_task(self.redis_client.setex(key, 60, json.dumps(price_data)))
                self._tasks.append(task)

            logger.debug(f"Price updated for {symbol}: {price}")
        except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
            logger.exception(f"Error updating price for {symbol}: {e}")
            return False
        else:
            return True

    def update_market_data(self, data: dict[str, Any], cache_type: str = "market_data") -> bool:
        """Update market data in cache - compatibility with shared_cache"""
        try:
            # Store in local cache
            ttl = self.default_ttl
            if cache_type == "top10_data":
                ttl = 30  # 30 seconds for real-time market data
            elif cache_type == "prices":
                ttl = 60  # 1 minute for prices

            # Create a task to store asynchronously
            if task_manager is not None:
                # MEMORY LEAK GUARD: keep internal task list bounded
                self._prune_tasks()
                task = task_manager.create_task_sync(self.set(cache_type, data, ttl=ttl), name="canonical_cache:set_async")
            else:
                task = asyncio.create_task(self.set(cache_type, data, ttl=ttl))
            self._tasks.append(task)

            # If this is top10 data, extract and update individual prices
            if cache_type == "top10_data" and isinstance(data, dict) and "prices" in data:
                for symbol, price_data in data["prices"].items():
                    if isinstance(price_data, (int, float)):
                        self.update_price(symbol, price_data)

            logger.debug(f"Market data updated in cache: {cache_type}")
        except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
            logger.exception(f"Error updating market data in cache: {e}")
            return False
        else:
            return True

    def _is_data_fresh(self, data: dict[str, Any]) -> bool:
        """Check if data is fresh enough for live trading"""
        try:
            if "timestamp" in data:
                data_time = datetime.fromisoformat(data["timestamp"].replace("Z", "+00:00"))
                age = (datetime.now(timezone.utc) - data_time).total_seconds()
                return age <= self.freshness_threshold
        except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError):
            return False
        else:
            return False

    def _cleanup_local_cache(self) -> None:
        """Clean up local cache when it gets too large - VECTORIZED for performance"""
        if len(self.local_cache) > self.max_local_size:
            # VECTORIZED cleanup for performance
            entries = list(self.local_cache.items())
            entries.sort(key=lambda x: x[1].last_accessed)

            remove_count = len(entries) // 5
            # VECTORIZED removal for performance
            for key, _ in entries[:remove_count]:
                self.local_cache.pop(key, None)

    async def _cleanup_pending_requests(self) -> None:
        """Clean up completed or expired pending requests"""
        current_time = time.time()
        keys_to_remove = []

        for key, future in self.pending_requests.items():
            if future.done():
                # Remove completed futures
                keys_to_remove.append(key)
            elif current_time - getattr(future, "_created_time", current_time) > self.request_timeout:
                # Remove expired futures (if we track creation time)
                keys_to_remove.append(key)

        for key in keys_to_remove:
            self.pending_requests.pop(key, None)

        if keys_to_remove:
            logger.debug(f"Cleaned up {len(keys_to_remove)} pending requests")

    async def _run_periodic_cleanup(self) -> None:
        """Run periodic cleanup of pending requests"""
        while True:
            try:
                await self._cleanup_pending_requests()
                await asyncio.sleep(60)  # Run cleanup every 60 seconds
            except asyncio.CancelledError:
                logger.info("Periodic cleanup task cancelled")
                break
            except Exception as e:
                logger.exception(f"Error in periodic cleanup: {e}")
                await asyncio.sleep(10)  # Shorter delay on error

    def _is_circuit_breaker_open(self) -> bool:
        """Check if circuit breaker is open"""
        if not self.circuit_breaker_open:
            return False

        # Check if timeout has passed
        if time.time() - self.circuit_breaker_last_failure > self.circuit_breaker_timeout:
            self.circuit_breaker_open = False
            self.circuit_breaker_failures = 0
            logger.info("Circuit breaker reset - allowing requests")
            return False

        return True

    def _record_circuit_breaker_failure(self) -> None:
        """Record a failure for circuit breaker"""
        self.circuit_breaker_failures += 1
        self.circuit_breaker_last_failure = time.time()

        if self.circuit_breaker_failures >= self.circuit_breaker_threshold:
            self.circuit_breaker_open = True
            self.metrics["circuit_breaker_trips"] += 1
            logger.warning(f"Circuit breaker opened after {self.circuit_breaker_failures} failures")

    async def get_metrics(self) -> dict[str, Any]:
        """Get cache performance metrics"""
        total_requests = self.metrics["hits"] + self.metrics["misses"]
        hit_rate = (self.metrics["hits"] / total_requests * 100) if total_requests > 0 else 0

        return {
            "hit_rate": round(hit_rate, 2),
            "total_requests": total_requests,
            "local_cache_size": len(self.local_cache),
            "redis_available": self.redis_client is not None,
            **self.metrics,
        }

    async def health_check(self) -> dict[str, Any]:
        """Health check for cache system"""
        try:
            redis_healthy = False
            if self.redis_client:
                try:
                    await self.redis_client.ping()
                    redis_healthy = True
                except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError):
                    pass

            # Always return healthy if local cache is available, even if Redis is down
            # This ensures the shared_cache service shows as healthy in health status
            return {
                "status": "healthy",
                "redis_available": redis_healthy,
                "local_cache_size": len(self.local_cache),
                "metrics": await self.get_metrics(),
            }
        except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
            return {"status": "error", "error": str(e)}

    async def warm_cache(self):
        """Warm cache with frequently accessed data and populate with fresh data"""
        try:
            logger.info("Starting cache warming process")

            # Warm market data for top 10 coins - USE CENTRALIZED CONFIG
            symbols = TRADING_SYMBOLS
            warmed_symbols = 0

            # Try to get fresh data from Binance US API
            try:
                if binance_us_top10_fetcher:
                    fresh_data = await binance_us_top10_fetcher.get_top10_24h()
                else:
                    fresh_data = None
                logger.info(f"Retrieved fresh data for {len(fresh_data) if fresh_data else 0} symbols from Binance US")

                # Populate cache with fresh data - handle list return type
                if fresh_data:
                    # The fresh_data is a list of ticker dictionaries, not a dict
                    if isinstance(fresh_data, list):
                        for ticker in fresh_data:
                            if not isinstance(ticker, dict):
                                continue

                            symbol = ticker.get("symbol")
                            if symbol:
                                # Store ticker data
                                await self.set_latest_ticker_24h(EXCHANGE_ID, symbol, ticker)
                                logger.debug(f"Set fresh ticker for {symbol}")

                                # Try to extract price from ticker data
                                if "lastPrice" in ticker:
                                    price = float(ticker["lastPrice"])
                                    await self.set_latest_price(EXCHANGE_ID, symbol, price)
                                    logger.debug(f"Set fresh price for {symbol}: {price}")
                    # Handle dict format if that's what's returned
                    elif isinstance(fresh_data, dict):
                        # Handle price data if available
                        if "prices" in fresh_data:
                            for symbol, price in fresh_data["prices"].items():
                                if isinstance(price, (int, float)) or (isinstance(price, str) and price.replace(".", "", 1).isdigit()):
                                    await self.set_latest_price(EXCHANGE_ID, symbol, float(price))
                                    logger.debug(f"Set fresh price for {symbol}: {price}")

                        # Handle ticker data if available
                        if "stats" in fresh_data and isinstance(fresh_data["stats"], list):
                            for ticker in fresh_data["stats"]:
                                symbol = ticker.get("symbol")
                                if symbol:
                                    await self.set_latest_ticker_24h(EXCHANGE_ID, symbol, ticker)
                                    logger.debug(f"Set fresh ticker for {symbol}")

                    logger.info("Populated cache with fresh Binance US data")
            except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as fresh_error:
                logger.warning(f"Failed to get fresh data from Binance US: {fresh_error}")

            # Warm cache for all symbols
            for symbol in symbols:
                try:
                    # Pre-load price data
                    price = await self.get_latest_price(EXCHANGE_ID, symbol)
                    # Pre-load ticker data
                    ticker = await self.get_latest_ticker_24h(EXCHANGE_ID, symbol)

                    if price or ticker:
                        warmed_symbols += 1
                        logger.debug(f"Warmed cache for {symbol} with price: {price}, ticker: {bool(ticker)}")
                except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
                    logger.debug(f"Failed to warm cache for {symbol}: {e}")

            # Warm AI signals
            try:
                await self.get("ai_signals")
                await self.get("ai_predictions")
                logger.debug("Warmed AI signals cache")
            except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
                logger.debug(f"Failed to warm AI signals cache: {e}")

            logger.info(f"Cache warming completed - warmed {warmed_symbols}/{len(symbols)} symbols")
            self._last_warmed = datetime.now(timezone.utc).isoformat()
        except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
            logger.exception(f"Cache warming failed: {e}")
            return False
        else:
            return warmed_symbols > 0

    async def get_cache_stats(self) -> dict[str, Any]:
        """Get cache statistics for monitoring"""
        try:
            total_entries = len(self.local_cache)
            expired_entries = sum(1 for entry in self.local_cache.values() if entry.is_expired)
            active_entries = total_entries - expired_entries

            # Calculate hit rate
            total_accesses = sum(entry.access_count for entry in self.local_cache.values())
            cache_hits = sum(entry.access_count for entry in self.local_cache.values() if not entry.is_expired)
            hit_rate = (cache_hits / total_accesses * 100) if total_accesses > 0 else 0

            return {
                "total_entries": total_entries,
                "active_entries": active_entries,
                "expired_entries": expired_entries,
                "total_accesses": total_accesses,
                "cache_hits": cache_hits,
                "hit_rate_percent": round(hit_rate, 2),
                "memory_usage_mb": round(
                    sum(len(str(entry.value)) for entry in self.local_cache.values()) / 1024 / 1024,
                    2,
                ),
                "redis_connected": self.redis_client is not None,
                "last_warmed": getattr(self, "_last_warmed", None),
            }
        except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
            logger.exception(f"Failed to get cache stats: {e}")
            return {"error": str(e)}

    async def get_volume_data(self, symbol: str | None = None) -> dict[str, Any]:
        """Get volume data for a symbol or all symbols"""
        if symbol:
            # Get volume data for specific symbol
            try:
                ticker = await self.get_latest_ticker_24h(EXCHANGE_ID, symbol)
                if ticker and isinstance(ticker, dict):
                    return {
                        "symbol": symbol,
                        "volume_24h": ticker.get("volume", 0),
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                        "source": "live",
                    }
            except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError):
                pass
            # All Live Data, No Fallback/Hardcoded Data - raise error if no live data available
            # Validate outside try to avoid TRY301
            msg = f"No live volume data available for {symbol}"
            raise RuntimeError(msg)

        try:
            # Get volume data for all top 10 symbols - USE CENTRALIZED CONFIG
            symbols = settings.trading_symbols if settings and hasattr(settings, "trading_symbols") else TRADING_SYMBOLS
            volume_data = {}

            for sym in symbols:
                ticker = await self.get_latest_ticker_24h(EXCHANGE_ID, sym)
                if ticker and isinstance(ticker, dict):
                    volume_data[sym] = ticker.get("volume", 0)
                else:
                    volume_data[sym] = 0

            return {
                "volume_data": volume_data,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "source": "live",
            }
        except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
            logger.warning(f"Failed to get volume data: {e}")
            return {
                "volume_data": {},
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "error": str(e),
            }

    async def get_indicator_data(self, symbol: str, indicator: str = "rsi") -> dict[str, Any]:
        """Get technical indicator data for a symbol"""
        try:
            key = f"indicator_{symbol}_{indicator}"
            data = await self.get(key)
            if data and self._is_data_fresh(data):
                return data
            return {}
        except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
            logger.warning(f"Failed to get indicator data for {symbol}: {e}")
            return None
        else:
            return None

    async def get_market_data(self, data_type: str = "live") -> dict[str, Any]:
        """Get market data with optional data type parameter for compatibility"""
        if data_type == "live":
            return await self.get_live_market_data()
        if data_type == "top10_data":
            return await self.get_top10_data()
        if data_type == "prices":
            # Return price data from cache
            price_data = await self.get("price_data")
            if price_data and self._is_data_fresh(price_data):
                return {
                    "status": "live",
                    "data": price_data,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "source": "canonical_cache",
                }
            return {
                "status": "stale",
                "data": price_data or {},
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "source": "canonical_cache",
            }
        # Default to live market data
        return await self.get_live_market_data()

    async def update_top10_data(self, data: dict[str, Any]) -> bool:
        """Update top 10 market data in cache"""
        try:
            await self.set("top10_market_data", data, ttl=30)  # 30 second TTL for live data
        except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
            logger.exception(f"Failed to update top10 data: {e}")
            return False
        else:
            return True

    async def get_top10_data(self) -> dict[str, Any]:
        """Get top 10 market data from cache"""
        try:
            data = await self.get("top10_market_data")
            if data and self._is_data_fresh(data):
                return {
                    "status": "live",
                    "data": data,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "source": "canonical_cache",
                }
            return {
                "status": "stale",
                "data": data or {},
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "source": "canonical_cache",
            }
        except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
            logger.exception(f"Failed to get top10 data: {e}")
            return {
                "status": "error",
                "data": {},
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "error": str(e),
            }

    async def get_signals_by_type(self, signal_type: str = "all", limit: int | None = None) -> list[dict[str, Any]]:
        """Get signals by type from cache with optional limit"""
        try:
            if signal_type == "all":
                # Get all signals
                signals = await self.get("ai_signals")
                if signals and isinstance(signals, list):
                    if limit is not None:
                        return signals[:limit]
                    return signals
                return []
            # Get signals by specific type
            key = f"signals_{signal_type}"
            signals = await self.get(key)
            if signals and isinstance(signals, list):
                if limit is not None:
                    return signals[:limit]
                return signals
        except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
            logger.warning(f"Failed to get signals by type {signal_type}: {e}")
            return []
        else:
            return []

    async def close(self):
        """Close cache connections"""
        # Cancel any outstanding background tasks
        if self._tasks:
            for task in self._tasks:
                if not task.done():
                    task.cancel()
                    with contextlib.suppress(asyncio.CancelledError):
                        await task
            self._tasks.clear()

        if self.redis_client:
            await self.redis_client.close()
            self.redis_client = None


# Global canonical cache instance
canonical_cache = CanonicalCache()


# Convenience functions for backward compatibility
async def get_cache() -> CanonicalCache:
    """Get the canonical cache instance"""
    return canonical_cache


async def get_live_market_data() -> dict[str, Any]:
    """Get live market data from canonical cache"""
    return await canonical_cache.get_live_market_data()


async def set_market_data(data: dict[str, Any], ttl: int = 30) -> bool:
    """Set market data in canonical cache"""
    return await canonical_cache.set("live_market_data", data, ttl)


async def initialize_cache(redis_url: str | None = None) -> bool:
    """Initialize the canonical cache"""
    return await canonical_cache.initialize(redis_url)


async def close_cache():
    """Close the canonical cache"""
    await canonical_cache.close()
