"""
backend/services/live_market_data.py

Live Market Data Service (Binance.US only, Top-10 enforced)
- Works with or without API keys (public endpoints if no keys)
- Async-friendly: ccxt sync calls are offloaded via asyncio.to_thread
- Exposes:
    - watchlist_ccxt  -> ["BTC/USDT", ...]  (for ccxt methods / other services)
    - watchlist_pairs -> ["BTCUSDT", ...]   (Binance pair format)
- Provides:
    start(), stop(), get_live_data(), get_ticker(), get_ohlcv(),
    get_order_book(), get_candlestick_data(), get_historical_data(), get_market_overview()
"""

from __future__ import annotations

import asyncio
import logging
import os
import random
import time
from datetime import datetime, timezone
from typing import Any

import ccxt
import httpx
from dotenv import load_dotenv

try:
    from backend.services.canonical_http_client import canonical_http_client
except ImportError:
    canonical_http_client = None

from backend.config.mystic_api_schedule import (
    AI_OHLCV_TARGET_WEIGHT_PER_MIN,
    AI_TICKER_TARGET_WEIGHT_PER_MIN,
    LIVE_OHLCV_INTERVAL_SEC,
    LIVE_TICKER_INTERVAL_SEC,
)
from backend.config.trading_universe import EXCHANGE_ID
from backend.config.trading_universe import TRADING_SYMBOLS as TOP10_BINANCEUS_DEFAULT
from backend.services.task_manager import task_manager
from backend.utils.binance_weight_limiter import (
    LIMITER_CONSUME_TIMEOUT_CRITICAL,
    LIMITER_CONSUME_TIMEOUT_LOOP,
    OHLCV_STALE_FALLBACK_MAX_AGE_SEC,
    BinanceWeightLimiter,
    CircuitOpenError,
    RateLimitedError,
)

# Optional imports - try at top level
try:
    from backend.utils.symbols import to_ccxt_symbol, to_display_symbol, to_exchange_symbol
except (ImportError, ModuleNotFoundError, AttributeError, ValueError, TypeError, RuntimeError):
    to_ccxt_symbol = None
    to_display_symbol = None
    to_exchange_symbol = None

logger = logging.getLogger(__name__)

CCXT_AVAILABLE = True
load_dotenv()

logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(level=logging.INFO)


def _to_ccxt_symbol(s: str) -> str:
    """Normalize to ccxt format (BTC/USDT)."""
    if to_ccxt_symbol is None:
        msg = "to_ccxt_symbol not available"
        raise RuntimeError(msg)

    return to_ccxt_symbol(s)


def _to_binance_pair(s: str) -> str:
    """Normalize to Binance pair without slash, default quote USDT."""
    if to_ccxt_symbol is None or to_exchange_symbol is None:
        msg = "symbol conversion functions not available"
        raise RuntimeError(msg)

    return to_exchange_symbol(to_ccxt_symbol(s))


def _safe_float(v: Any, default: float = 0.0) -> float:
    if v in (None, False, True):
        return default
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


class LiveMarketDataService:
    """Binance.US-only live market data service."""

    def __init__(self) -> None:
        api_key = os.getenv("BINANCE_US_API_KEY") or os.getenv("BINANCE_API_KEY") or ""
        secret = os.getenv("BINANCE_US_SECRET_KEY") or os.getenv("BINANCE_SECRET") or ""

        # Use CCXT for Binance US API calls (recommended for trading platforms)
        self.base_url = "https://api.binance.us/api/v3"
        self.api_key = api_key
        self.secret = secret
        self.authenticated = bool(api_key and secret)

        # Initialize CCXT Binance US client
        if CCXT_AVAILABLE:
            self.binance = ccxt.binanceus(
                {
                    "apiKey": api_key if api_key else None,
                    "secret": secret if secret else None,
                    "enableRateLimit": True,
                    "options": {"defaultType": "spot"},
                },
            )
            logger.info("CCXT Binance US client initialized")
        else:
            self.binance = None
            logger.error("CCXT not available - market data service will not work!")

        if self.authenticated:
            logger.info("Binance US API keys configured (using CCXT for API calls)")
        else:
            logger.info("No API keys configured, using public endpoints only")

        # Build watchlist from env (BINANCE_US_TOP10_SYMBOLS) or fallback to default Top-10
        raw = os.getenv("BINANCE_US_TOP10_SYMBOLS", "")
        items = [x for x in (p.strip() for p in raw.split(",")) if x] if raw.strip() else TOP10_BINANCEUS_DEFAULT

        # Normalize both formats
        self.watchlist_pairs: list[str] = [_to_binance_pair(s) for s in items]
        self.watchlist_ccxt: list[str] = [_to_ccxt_symbol(s) for s in items]

        # Add watchlist_human for app_factory compatibility (display format: BTC-USD)
        if to_display_symbol is None:
            msg = "to_display_symbol not available"
            raise RuntimeError(msg)

        self.watchlist_human: list[str] = [to_display_symbol(ccxt_sym) for ccxt_sym in self.watchlist_ccxt]

        # Loop cadence / budgeting — see backend/config/mystic_api_schedule.py
        self.ticker_interval = LIVE_TICKER_INTERVAL_SEC
        self.ohlcv_interval = LIVE_OHLCV_INTERVAL_SEC
        self.ticker_weight_per_min = AI_TICKER_TARGET_WEIGHT_PER_MIN
        self.ohlcv_weight_per_min = AI_OHLCV_TARGET_WEIGHT_PER_MIN
        self.ticker_max_conc = int(os.getenv("AI_TICKER_MAX_CONCURRENCY", "8"))
        self.ohlcv_max_conc = int(os.getenv("AI_OHLCV_MAX_CONCURRENCY", "2"))

        # Internal state
        self._running = False
        self._tasks: list[asyncio.Task] = []
        self._lock = asyncio.Lock()
        self._ticker_cache: dict[str, dict] = {}  # key: "BTC/USDT"
        self._ohlcv_cache: dict[str, list] = {}  # key: "BTC/USDT" (1m loop cache)
        self._ohlcv_tf_cache: dict[tuple[str, str, int], tuple[float, list]] = {}
        self._ticker_idx = 0
        self._ohlcv_idx = 0
        self._limiter: BinanceWeightLimiter | None = None
        self._limiter_lock = asyncio.Lock()
        self._cache_guard: Any | None = None

    async def _persist_latest_1m_candle(self, ccxt_symbol: str, ohlcv: list) -> bool:
        """Write latest 1m bar to feature_ohlcv (replaces standalone live_data_collector)."""
        if not ohlcv:
            return False
        last = ohlcv[-1]
        if not isinstance(last, (list, tuple)) or len(last) < 6:
            return False
        db_symbol = str(ccxt_symbol).replace("/", "-")
        candle = {
            "open": float(last[1]),
            "high": float(last[2]),
            "low": float(last[3]),
            "close": float(last[4]),
            "volume": float(last[5]),
        }
        try:
            from backend.services.feature_store import insert_ohlcv

            await asyncio.to_thread(insert_ohlcv, db_symbol, "1m", candle)
            return True
        except Exception as exc:
            logger.debug("feature_ohlcv persist failed %s: %s", db_symbol, exc)
            return False

    async def _mark_market_heartbeat(self) -> None:
        if self._cache_guard is None:
            return
        try:
            await self._cache_guard.mark_market_update("live_market_data")
        except Exception as exc:
            logger.debug("market_data:last_update write failed: %s", exc)

    # ---------------- lifecycle ----------------

    async def start(self) -> None:
        if self._running:
            return
        self._running = True
        try:
            # Skip load_markets for Binance.US as it doesn't support margin endpoints
            # and causes 404 errors on /sapi/v1/margin/allAssets
            logger.info("Skipping load_markets for Binance.US (margin endpoints not supported)")
        except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
            logger.warning(f"load_markets failed (continuing): {e}")

        try:
            from backend.utils.cache_guard import CacheGuard

            self._cache_guard = await CacheGuard.create()
        except Exception as exc:
            logger.warning("CacheGuard init failed (heartbeat disabled): %s", exc)
            self._cache_guard = None

        self._tasks = [
            await task_manager.create_task(self._ticker_loop(), name="live_market_data:ticker_loop"),
            await task_manager.create_task(self._ohlcv_loop(), name="live_market_data:ohlcv_loop"),
        ]
        logger.info("LiveMarketDataService started (Binance.US only)")

    async def _get_limiter(self) -> BinanceWeightLimiter:
        """Return shared Binance weight limiter for market data REST calls."""
        if self._limiter is not None:
            return self._limiter

        async with self._limiter_lock:
            if self._limiter is None:
                self._limiter = await BinanceWeightLimiter.create()
        return self._limiter

    def _ohlcv_cache_key(self, ccxt_symbol: str, timeframe: str, limit: int) -> tuple[str, str, int]:
        return (_to_ccxt_symbol(ccxt_symbol), str(timeframe), int(limit))

    def _stale_ohlcv_fallback(self, key: tuple[str, str, int]) -> list | None:
        cached = self._ohlcv_tf_cache.get(key)
        if not cached:
            return None
        fetched_at, rows = cached
        age = time.time() - fetched_at
        if age > OHLCV_STALE_FALLBACK_MAX_AGE_SEC or not rows:
            return None
        logger.info(
            "OHLCV_STALE_FALLBACK %s %s limit=%s age_sec=%.1f rows=%d",
            key[0],
            key[1],
            key[2],
            age,
            len(rows),
        )
        return list(rows)

    def _store_ohlcv_cache(self, key: tuple[str, str, int], rows: list) -> None:
        if rows:
            self._ohlcv_tf_cache[key] = (time.time(), list(rows))

    async def stop(self) -> None:
        self._running = False
        for t in self._tasks:
            if not t.done():
                t.cancel()
        self._tasks.clear()
        logger.info("LiveMarketDataService stopped")

    # ---------------- loops ----------------

    async def _ticker_loop(self) -> None:
        # Startup jitter to prevent synchronized bursts after restart
        await asyncio.sleep(2.0 + random.random() * 2.0)
        while self._running:
            t0 = time.time()
            try:
                # Check if CCXT client is available
                if not self.binance:
                    logger.error("CCXT binance client not initialized, cannot fetch market data")
                    await asyncio.sleep(self.ticker_interval)
                    continue

                symbols = self.watchlist_ccxt
                total = len(symbols)
                if total == 0:
                    await asyncio.sleep(self.ticker_interval)
                    continue

                allowed = max(1, int(self.ticker_weight_per_min * (self.ticker_interval / 60.0)))
                start = self._ticker_idx % total
                end = start + allowed
                batch = symbols[start:end] if end <= total else (symbols[start:] + symbols[: (end % total)])
                self._ticker_idx = (start + allowed) % total
                limiter = await self._get_limiter()

                sem = asyncio.Semaphore(max(1, self.ticker_max_conc))

                async def _fetch(sym: str, semaphore=sem, limiter=limiter) -> None:
                    try:
                        async with semaphore:
                            # Use direct Binance US API call instead of CCXT to avoid margin endpoint issues
                            # Convert symbol format (BTC/USDT -> BTCUSDT)
                            api_symbol = sym.replace("/", "")

                            url = "https://api.binance.us/api/v3/ticker/24hr"
                            params = {"symbol": api_symbol}

                            await limiter.consume(
                                "/api/v3/ticker/24hr",
                                weight=1,
                                wait=True,
                                timeout=LIMITER_CONSUME_TIMEOUT_LOOP,
                            )
                            async with httpx.AsyncClient() as client:
                                response = await client.get(url, params=params, timeout=10.0)
                                response.raise_for_status()
                                data = response.json()

                            # Convert to CCXT format
                            ticker_data = {
                                "symbol": sym,
                                "last": float(data.get("lastPrice", 0)),
                                "bid": float(data.get("bidPrice", 0)),
                                "ask": float(data.get("askPrice", 0)),
                                "high": float(data.get("highPrice", 0)),
                                "low": float(data.get("lowPrice", 0)),
                                "volume": float(data.get("volume", 0)),
                                "quoteVolume": float(data.get("quoteVolume", 0)),
                                "change": float(data.get("priceChange", 0)),
                                "percentage": float(data.get("priceChangePercent", 0)),
                                "timestamp": int(data.get("closeTime", 0)),
                            }

                            async with self._lock:
                                self._ticker_cache[sym] = ticker_data
                            logger.debug(f"Fetched ticker for {sym}: ${ticker_data.get('last', 0)}")
                    except (RateLimitedError, CircuitOpenError) as e:
                        logger.warning("Rate limited fetching ticker for %s: %s", sym, e)
                    except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
                        # Handle API key errors gracefully
                        if "Invalid Api-Key ID" in str(e) or "API-key format invalid" in str(e):
                            logger.warning(f"[WARNING] Ticker fetch disabled for {sym}: API key issue - {e}")
                        else:
                            logger.debug(f"ticker fetch failed {sym}: {e}")

                tasks = [await task_manager.create_task(_fetch(s), name="live_market_data:ticker_fetch") for s in batch]
                await asyncio.gather(*tasks)
            finally:
                elapsed = max(0.0, time.time() - t0)
                await asyncio.sleep(max(0.0, self.ticker_interval - elapsed))

    async def _ohlcv_loop(self) -> None:
        while self._running:
            t0 = time.time()
            try:
                # Check if CCXT client is available
                if not self.binance:
                    logger.error("CCXT binance client not initialized, cannot fetch OHLCV data")
                    await asyncio.sleep(self.ohlcv_interval)
                    continue

                symbols = self.watchlist_ccxt
                total = len(symbols)
                if total == 0:
                    await asyncio.sleep(self.ohlcv_interval)
                    continue

                allowed = max(1, int(self.ohlcv_weight_per_min * (self.ohlcv_interval / 60.0)))
                start = self._ohlcv_idx % total
                end = start + allowed
                batch = symbols[start:end] if end <= total else (symbols[start:] + symbols[: (end % total)])
                self._ohlcv_idx = (start + allowed) % total
                limiter = await self._get_limiter()

                sem = asyncio.Semaphore(max(1, self.ohlcv_max_conc))
                persisted_any = False

                async def _fetch(sym: str, semaphore=sem, limiter=limiter) -> None:
                    nonlocal persisted_any
                    try:
                        async with semaphore:
                            # Use direct Binance US API call instead of CCXT to avoid margin endpoint issues
                            # Convert symbol format (BTC/USDT -> BTCUSDT)
                            api_symbol = sym.replace("/", "")

                            url = "https://api.binance.us/api/v3/klines"
                            params = {
                                "symbol": api_symbol,
                                "interval": "1m",
                                "limit": 300,
                            }

                            await limiter.consume(
                                "/api/v3/klines",
                                weight=1,
                                wait=True,
                                timeout=LIMITER_CONSUME_TIMEOUT_LOOP,
                            )
                            async with httpx.AsyncClient() as client:
                                response = await client.get(url, params=params, timeout=10.0)
                                response.raise_for_status()
                                klines_data = response.json()

                            # Convert to CCXT format
                            ohlcv = []
                            for kline in klines_data:
                                ohlcv.append(
                                    [
                                        int(kline[0]),  # timestamp
                                        float(kline[1]),  # open
                                        float(kline[2]),  # high
                                        float(kline[3]),  # low
                                        float(kline[4]),  # close
                                        float(kline[5]),  # volume
                                    ],
                                )

                            async with self._lock:
                                self._ohlcv_cache[sym] = ohlcv
                            if await self._persist_latest_1m_candle(sym, ohlcv):
                                persisted_any = True
                            logger.debug(f"Fetched {len(ohlcv)} candles for {sym}")
                    except (RateLimitedError, CircuitOpenError) as e:
                        logger.warning("Rate limited fetching klines for %s: %s", sym, e)
                    except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
                        # Handle API key errors gracefully
                        if "Invalid Api-Key ID" in str(e) or "API-key format invalid" in str(e):
                            logger.warning(f"[WARNING] OHLCV fetch disabled for {sym}: API key issue - {e}")
                        else:
                            logger.debug(f"ohlcv fetch failed {sym}: {e}")

                tasks = [await task_manager.create_task(_fetch(s), name="live_market_data:ohlcv_fetch") for s in batch]
                await asyncio.gather(*tasks)
                if persisted_any:
                    await self._mark_market_heartbeat()
            finally:
                elapsed = max(0.0, time.time() - t0)
                await asyncio.sleep(max(0.0, self.ohlcv_interval - elapsed))

    # ---------------- getters ----------------

    async def get_ticker(self, ccxt_symbol: str) -> dict | None:
        """
        Return a normalized ticker for a ccxt symbol (e.g., 'BTC/USDT').
        Output has convenience fields expected by downstream code (insert_tick):
            { price, bid, ask, volume_24h, change_24h, high, low, source }
        """
        # Normalize symbol to proper CCXT format first
        from backend.utils.symbols import normalize_symbol

        s = normalize_symbol(ccxt_symbol)
        t: dict | None = None
        try:
            limiter = await self._get_limiter()
            # Get from cache first (populated by ticker loop)
            async with self._lock:
                t = self._ticker_cache.get(s)

            if not t:
                # If not in cache, try direct API call
                api_symbol = s.replace("/", "")
                url = "https://api.binance.us/api/v3/ticker/24hr"
                params = {"symbol": api_symbol}

                await limiter.consume(
                    "/api/v3/ticker/24hr",
                    weight=1,
                    wait=True,
                    timeout=LIMITER_CONSUME_TIMEOUT_LOOP,
                )
                async with httpx.AsyncClient() as client:
                    response = await client.get(url, params=params, timeout=10.0)
                    response.raise_for_status()
                    data = response.json()

                # Convert to CCXT format
                t = {
                    "symbol": s,
                    "last": float(data.get("lastPrice", 0)),
                    "bid": float(data.get("bidPrice", 0)),
                    "ask": float(data.get("askPrice", 0)),
                    "high": float(data.get("highPrice", 0)),
                    "low": float(data.get("lowPrice", 0)),
                    "volume": float(data.get("volume", 0)),
                    "quoteVolume": float(data.get("quoteVolume", 0)),
                    "change": float(data.get("priceChange", 0)),
                    "percentage": float(data.get("priceChangePercent", 0)),
                    "timestamp": int(data.get("closeTime", 0)),
                }

                # Cache it for future use
                async with self._lock:
                    self._ticker_cache[s] = t

        except (RateLimitedError, CircuitOpenError) as e:
            async with self._lock:
                t_stale = self._ticker_cache.get(s)
            if t_stale:
                logger.debug("Rate limited getting ticker for %s, returning cached: %s", s, e)
                t = t_stale
            else:
                logger.warning("Rate limited getting ticker for %s: %s", s, e)
                return None
        except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
            logger.debug(f"Failed to get ticker for {s}: {e}")
            return None

        if not t:
            return None

        last = _safe_float(t.get("last"))
        bid = _safe_float(t.get("bid"))
        ask = _safe_float(t.get("ask"))
        base_vol = _safe_float(t.get("baseVolume"))
        quote_vol = _safe_float(t.get("quoteVolume"))
        pct = _safe_float(t.get("percentage"))
        high = _safe_float(t.get("high"))
        low = _safe_float(t.get("low"))

        return {
            "symbol": s,
            "price": last,
            "bid": bid,
            "ask": ask,
            "volume_24h": quote_vol if quote_vol > 0 else base_vol,
            "change_24h": pct,
            "baseVolume": base_vol,
            "percentage": pct,
            "high": high,
            "low": low,
            "source": EXCHANGE_ID,
        }

    async def get_ohlcv(
        self,
        ccxt_symbol: str,
        timeframe: str = "1m",
        limit: int = 300,
        *,
        end_time_ms: int | None = None,
    ) -> list | None:
        """
        Return raw OHLCV (list of lists) for a ccxt symbol and timeframe.
        Compatible with backtester & feature ingestor callers:
            await service.get_ohlcv("BTC-USDT", "1m", limit=1)
        """
        s = _to_ccxt_symbol(ccxt_symbol)
        cache_key = self._ohlcv_cache_key(s, timeframe, limit)
        # Use httpx for direct Binance API calls instead of ccxt
        try:
            if canonical_http_client is None:
                logger.warning("canonical_http_client not available")
                return self._stale_ohlcv_fallback(cache_key)

            # Convert symbol format and make direct API call
            symbol = ccxt_symbol.replace("/", "").replace("-", "")
            url = f"{self.base_url}/klines"
            params: dict[str, Any] = {"symbol": symbol, "interval": timeframe, "limit": limit}
            if end_time_ms is not None:
                params["endTime"] = max(0, int(end_time_ms))

            limiter = await self._get_limiter()
            await limiter.consume(
                "/api/v3/klines",
                weight=1,
                wait=True,
                timeout=LIMITER_CONSUME_TIMEOUT_CRITICAL,
            )
            data = await canonical_http_client.get_json(url, params=params)

            # Convert to ccxt format
            ohlcv = []
            for kline in data:
                ohlcv.append(
                    [
                        int(kline[0]),  # timestamp
                        float(kline[1]),  # open
                        float(kline[2]),  # high
                        float(kline[3]),  # low
                        float(kline[4]),  # close
                        float(kline[5]),  # volume
                    ],
                )
        except (RateLimitedError, CircuitOpenError) as e:
            stale = self._stale_ohlcv_fallback(cache_key)
            if stale is not None:
                return stale
            logger.warning("Rate limited getting ohlcv for %s: %s", s, e)
            return None
        except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
            # Handle API key errors gracefully
            if "Invalid Api-Key ID" in str(e) or "API-key format invalid" in str(e):
                logger.warning(f"[WARNING] OHLCV data unavailable for {s}: API key issue - {e}")
                logger.info("[INFO] Note: OHLCV data requires valid Binance US API credentials")
            else:
                logger.exception(f"get_ohlcv failed {s} {timeframe} limit={limit}: {e}")
            return None
        else:
            self._store_ohlcv_cache(cache_key, ohlcv)
            return ohlcv

    # ---------------- used by other services / UI ----------------

    async def get_live_data(self) -> dict[str, Any]:
        """
        Return a compact live bundle for the first 10 watchlist symbols:
        [{symbol, price, change_24h, volume, high_24h, low_24h, source}, ...]
        """
        out: list[dict[str, Any]] = []
        async with self._lock:
            for sym in self.watchlist_ccxt[:10]:
                t = self._ticker_cache.get(sym)
                if not t:
                    continue
                out.append(
                    {
                        "symbol": sym,
                        "price": _safe_float(t.get("last")),
                        "change_24h": _safe_float(t.get("percentage")),
                        "volume": _safe_float(t.get("baseVolume")),
                        "high_24h": _safe_float(t.get("high")),
                        "low_24h": _safe_float(t.get("low")),
                        "source": EXCHANGE_ID,
                    },
                )
        return {
            "status": "success",
            "data": out,
            "timestamp": datetime.now(timezone.utc).isoformat() + "Z",
            "source": EXCHANGE_ID,
        }

    async def get_order_book(self, symbol: str, limit: int = 10) -> dict[str, Any]:
        s = _to_ccxt_symbol(symbol)
        try:
            # Use public-only client for orderbook (no API keys needed)
            public_client = ccxt.binanceus(
                {
                    "enableRateLimit": True,
                    "options": {"defaultType": "spot"},
                },
            )
            ob = await asyncio.to_thread(public_client.fetch_order_book, s, limit)
            return {
                "bids": ob.get("bids", [])[:limit],
                "asks": ob.get("asks", [])[:limit],
            }
        except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
            logger.exception(f"order_book failed {s}: {e}")
            return {"bids": [], "asks": []}

    async def get_historical_data(self, symbol: str, timeframe: str = "1m", limit: int = 300) -> dict[str, Any]:
        s = _to_ccxt_symbol(symbol)
        try:
            ohlcv = await asyncio.to_thread(self.binance.fetch_ohlcv, s, timeframe, limit=limit)
            return {
                "status": "success",
                "data": {
                    "timestamps": [c[0] for c in ohlcv],
                    "opens": [c[1] for c in ohlcv],
                    "highs": [c[2] for c in ohlcv],
                    "lows": [c[3] for c in ohlcv],
                    "closes": [c[4] for c in ohlcv],
                    "volumes": [c[5] for c in ohlcv],
                },
                "symbol": s,
                "timeframe": timeframe,
                "source": EXCHANGE_ID,
            }
        except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
            # Handle API key errors gracefully
            if "Invalid Api-Key ID" in str(e) or "API-key format invalid" in str(e):
                logger.warning(f"[WARNING] Historical data unavailable for {s}: API key issue - {e}")
                logger.info("[INFO] Note: Historical data requires valid Binance US API credentials")
                return {
                    "status": "error",
                    "message": "API key required for historical data",
                    "symbol": s,
                }
            logger.exception(f"historical failed {s} {timeframe}: {e}")
            return {"status": "error", "message": str(e), "symbol": s}

    async def get_candlestick_data(self, symbol: str, interval: str = "1m") -> list[dict[str, Any]]:
        tf = {
            "1m": "1m",
            "3m": "3m",
            "5m": "5m",
            "15m": "15m",
            "30m": "30m",
            "1h": "1h",
            "2h": "2h",
            "4h": "4h",
            "6h": "6h",
            "12h": "12h",
            "1d": "1d",
            "1w": "1w",
        }.get(interval, "1m")
        res = await self.get_historical_data(symbol, tf, 300)
        if res.get("status") != "success":
            return []
        d = res["data"]
        return [
            {
                "timestamp": d["timestamps"][i],
                "open": d["opens"][i],
                "high": d["highs"][i],
                "low": d["lows"][i],
                "close": d["closes"][i],
                "volume": d["volumes"][i],
            }
            for i in range(len(d["timestamps"]))
        ]

    async def get_market_overview(self) -> dict[str, Any]:
        """Lightweight overview wrapper."""
        try:
            live = await self.get_live_data()
            return {
                "status": "success",
                "crypto": {row["symbol"]: row for row in live.get("data", [])},
                "timestamp": datetime.now(timezone.utc).isoformat() + "Z",
                "source": EXCHANGE_ID,
            }
        except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
            logger.exception(f"market_overview failed: {e}")
            return {"status": "error", "message": str(e)}

    async def get_market_summary(self) -> dict[str, Any]:
        """
        Get market summary for the 10 allowlisted symbols.
        Returns: { "symbols": [...], "prices": {...}, "stats_24h": {...}, "last_update": <ts> }
        """
        try:
            symbols = []
            prices = {}
            stats_24h = {}

            # Get data from cache first (most recent)
            async with self._lock:
                ticker_cache = dict(self._ticker_cache)

            # Get latest prices and stats for allowlisted symbols
            for ccxt_symbol in self.watchlist_ccxt:
                try:
                    # Try to get from cache first
                    ticker_data = ticker_cache.get(ccxt_symbol)
                    if not ticker_data:
                        # If not in cache, fetch live
                        ticker_data = await asyncio.to_thread(self.binance.fetch_ticker, ccxt_symbol)

                    if ticker_data:
                        # Convert to display symbol (BTCUSDT -> BTC)
                        display_symbol = ccxt_symbol.replace("/USDT", "").replace("/USD", "")

                        symbols.append(display_symbol)
                        prices[display_symbol] = float(ticker_data.get("last", 0) or 0)

                        # Get 24h stats
                        change_24h = float(ticker_data.get("percentage", 0) or 0)
                        volume_24h = float(ticker_data.get("quoteVolume", 0) or 0)

                        stats_24h[display_symbol] = {
                            "change_24h": change_24h,
                            "volume_24h": volume_24h,
                            "high_24h": float(ticker_data.get("high", 0) or 0),
                            "low_24h": float(ticker_data.get("low", 0) or 0),
                        }
                except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
                    logger.debug(f"Failed to get data for {ccxt_symbol}: {e}")
                    continue

            return {
                "symbols": symbols,
                "prices": prices,
                "stats_24h": stats_24h,
                "last_update": time.time(),
            }
        except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
            logger.exception(f"get_market_summary failed: {e}")
            return {
                "symbols": [],
                "prices": {},
                "stats_24h": {},
                "last_update": time.time(),
            }


# Global singleton expected by other modules
live_market_data_service = LiveMarketDataService()
