from __future__ import annotations

import asyncio
import contextlib
import hashlib
import hmac
import os
import time
from typing import Any

import httpx
import redis.asyncio as redis

from backend.config.redis_config import get_shared_redis_async
from backend.metrics import (
    rest_errors_total,
    rest_latency_seconds,
    rest_requests_total,
    rest_retries_total,
)
from backend.services.canonical_http_client import get_http_client
from backend.services.circuit_breaker_service import get_api_breaker
from backend.utils.binance_credentials import get_binance_us_api_key, get_binance_us_secret_key
from backend.utils.binance_weight_limiter import (
    ENDPOINT_WEIGHTS,
    BinanceWeightLimiter,
    CircuitOpen,
)

# Import from single source of truth
try:
    from backend.config.trading_universe import TRADING_SYMBOLS
except (ImportError, ModuleNotFoundError, AttributeError, ValueError, TypeError, RuntimeError) as e:
    msg = f"Failed to import trading_universe: {e}"
    raise RuntimeError(msg) from e

# Use TRADING_SYMBOLS from trading_universe (live data)
BINANCE_US_TOP_10 = list(TRADING_SYMBOLS)

# Retry backoff constants
API_RETRY_BACKOFF_MULTIPLIER_THROTTLE = 2.0  # Multiplier for throttle/ban responses
API_RETRY_BACKOFF_MULTIPLIER_ERROR = 0.5  # Multiplier for network errors
API_RETRY_JITTER_FACTOR = 0.4  # Random jitter factor (±40%)

"""
Binance.US REST client (LIVE ONLY)
- Enforces Binance.US Top-10 symbols everywhere.
- Live trading only.
- Applies request-weight limiter + backoff.
- Provides common method-name aliases used elsewhere (get_ticker_price, ticker_price, get_price, price).
"""

BINANCEUS_BASE = os.getenv("BINANCEUS_BASE", "https://api.binance.us")
# Canonical: BINANCE_US_*; BINANCEUS_* kept as legacy fallback inside helper.
API_KEY = get_binance_us_api_key()
API_SECRET = get_binance_us_secret_key()

DEFAULT_TIMEOUT = float(os.getenv("BINANCE_HTTP_TIMEOUT", "20"))  # Increased for personal network
MAX_RETRIES = int(os.getenv("BINANCE_HTTP_MAX_RETRIES", "3"))

_ALLOWED_INTERVALS: set[str] = {"1m", "5m", "15m", "1h", "4h", "1d"}


def _sign(params: dict[str, Any]) -> str:
    q = "&".join(f"{k}={v}" for k, v in sorted(params.items()))
    return hmac.new(API_SECRET.encode(), q.encode(), hashlib.sha256).hexdigest()


def _jitter(base: float, spread: float = 0.4) -> float:
    # Remove random jitter - use deterministic spread based on market volatility
    return base * (1.0 + spread * 0.1)  # Use small deterministic spread


def _normalize_symbol(symbol: str) -> str:
    """
    Normalize user input to raw 'BTCUSDT' form and enforce Top-10.
    Accepts 'BTCUSDT', 'BTC/USDT', 'BTC-USD' (mapped to USDT).
    """
    s = symbol.strip().upper()
    if "/" in s:
        base, _quote = s.split("/", 1)
        s = f"{base}USDT"
    elif "-" in s:
        base, _quote = s.split("-", 1)
        s = f"{base}USDT"
    # already raw like BTCUSDT
    if s not in BINANCE_US_TOP_10:
        msg = f"Symbol {s} not allowed. Must be one of: {sorted(BINANCE_US_TOP_10)}"
        raise ValueError(msg)
    return s


def _validate_interval(interval: str) -> str:
    i = interval.strip()
    if i not in _ALLOWED_INTERVALS:
        msg = f"Interval '{i}' not allowed. Choose from: {_ALLOWED_INTERVALS}"
        raise ValueError(msg)
    return i


class BinanceREST:
    def __init__(self, limiter: BinanceWeightLimiter) -> None:
        self.limiter = limiter
        self._redis: redis.Redis | None = None

    async def _get_redis(self) -> redis.Redis | None:
        if self._redis is None:
            try:
                self._redis = get_shared_redis_async()
            except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError):
                self._redis = None
        return self._redis

    async def _request(
        self,
        method: str,
        path: str,
        params: dict[str, Any] | None = None,
        signed: bool = False,
    ) -> Any | None:
        params = params or {}
        headers = {"X-MBX-APIKEY": API_KEY} if API_KEY else {}
        await self._get_redis()

        if signed:
            params["timestamp"] = int(time.time() * 1000)
            params["signature"] = _sign(params)

        weight = ENDPOINT_WEIGHTS.get(path, 1)

        start_req = time.time()
        client = await get_http_client()

        # Get circuit breaker for this endpoint
        breaker = get_api_breaker(f"binance_{path.replace('/', '_')}")

        async def _make_api_call():
            return await client.request(
                method,
                f"{BINANCEUS_BASE}{path}",
                params=params,
                headers=headers,
                timeout=DEFAULT_TIMEOUT,
            )

        for attempt in range(1, MAX_RETRIES + 1):
            await self.limiter.consume(path, weight, wait=True, timeout=5.0)
            try:
                resp = await breaker.call(_make_api_call)
                if resp.status_code in (418, 429):
                    # Throttle or ban-ish responses; trip circuit and backoff
                    try:
                        data = resp.json()
                        msg = data.get("msg", "")
                    except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError):
                        msg = resp.text
                    if "-1003" in msg or "Way too much request weight" in msg or "banned" in msg.lower():
                        await self.limiter.open_circuit()
                        # Longer backoff for throttle responses
                        await asyncio.sleep(_jitter(API_RETRY_BACKOFF_MULTIPLIER_THROTTLE * attempt))
                        with contextlib.suppress(ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError):
                            rest_retries_total.labels(path=path).inc()
                        try:
                            if self._redis:
                                await self._redis.incr(f"rest:retries:{path}")
                                await self._redis.incr(f"rest:throttle:{path}")
                        except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError):
                            pass
                        continue
                resp.raise_for_status()
                # Try JSON first
                try:
                    data = resp.json()
                    try:
                        rest_requests_total.labels(path=path, status="ok").inc()
                        rest_latency_seconds.labels(path=path).observe(max(0.0, time.time() - start_req))
                    except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError):
                        pass
                    try:
                        if self._redis:
                            await self._redis.incr(f"rest:ok:{path}")
                    except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError):
                        pass
                except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError):
                    # Non-JSON OK response
                    try:
                        rest_requests_total.labels(path=path, status="ok_nojson").inc()
                        rest_latency_seconds.labels(path=path).observe(max(0.0, time.time() - start_req))
                    except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError):
                        pass
                    try:
                        if self._redis:
                            await self._redis.incr(f"rest:ok_nojson:{path}")
                    except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError):
                        pass
                    return None
                else:
                    return data
            except (httpx.TimeoutException, httpx.HTTPError) as e:
                try:
                    rest_errors_total.labels(path=path, type=e.__class__.__name__).inc()
                    rest_retries_total.labels(path=path).inc()
                except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError):
                    pass
                # Shorter backoff for network errors
                await asyncio.sleep(_jitter(API_RETRY_BACKOFF_MULTIPLIER_ERROR * attempt))
                try:
                    if self._redis:
                        await self._redis.incr(f"rest:error:{path}:{e.__class__.__name__}")
                        await self._redis.incr(f"rest:retries:{path}")
                except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError):
                    pass
                continue
            except CircuitOpen:
                # Circuit is open - wait before retry
                await asyncio.sleep(_jitter(API_RETRY_BACKOFF_MULTIPLIER_THROTTLE * attempt))
                with contextlib.suppress(ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError):
                    rest_errors_total.labels(path=path, type="CircuitOpen").inc()
                try:
                    if self._redis:
                        await self._redis.incr(f"rest:circuit_open:{path}")
                except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError):
                    pass
                return None

        # Exhausted attempts
        try:
            rest_requests_total.labels(path=path, status="exhausted").inc()
            rest_latency_seconds.labels(path=path).observe(max(0.0, time.time() - start_req))
        except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError):
            pass
        try:
            if self._redis:
                await self._redis.incr(f"rest:exhausted:{path}")
        except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError):
            pass
        return None

    # --- Public endpoints (LIVE only, Top-10 enforced) ---

    async def price(self, symbol: str) -> dict[str, Any] | None:
        raw = _normalize_symbol(symbol)
        return await self._request("GET", "/api/v3/ticker/price", params={"symbol": raw})

    # Aliases for compatibility (RESTAdapter tries these names)
    async def get_ticker_price(self, symbol: str) -> dict[str, Any] | None:
        return await self.price(symbol)

    async def ticker_price(self, symbol: str) -> dict[str, Any] | None:
        return await self.price(symbol)

    async def get_price(self, symbol: str) -> dict[str, Any] | None:
        return await self.price(symbol)

    async def ticker_24h(self, symbol: str) -> dict[str, Any] | None:
        raw = _normalize_symbol(symbol)
        return await self._request("GET", "/api/v3/ticker/24hr", params={"symbol": raw})

    async def get_server_time(self) -> dict[str, Any] | None:
        """Get Binance server time (no symbol required, no Top-10 enforcement)"""
        return await self._request("GET", "/api/v3/time")

    async def get_exchange_info(self) -> dict[str, Any] | None:
        """Get exchange information (no symbol required, no Top-10 enforcement)"""
        return await self._request("GET", "/api/v3/exchangeInfo")

    async def klines(self, symbol: str, interval: str = "1m", limit: int = 100) -> list | None:
        raw = _normalize_symbol(symbol)
        i = _validate_interval(interval)
        if limit <= 0:
            msg = "limit must be positive"
            raise ValueError(msg)
        return await self._request(
            "GET",
            "/api/v3/klines",
            params={"symbol": raw, "interval": i, "limit": int(limit)},
        )

    # --- Private endpoints (LIVE only; requires API key/secret) ---

    async def order_market(
        self,
        symbol: str,
        side: str,
        quantity: str | float | None = None,
        quoteOrderQty: str | float | None = None,
        recvWindow: int | None = None,
    ) -> dict[str, Any] | None:
        if not API_KEY or not API_SECRET:
            msg = "API credentials not configured for signed endpoints"
            raise RuntimeError(msg)
        raw = _normalize_symbol(symbol)
        s = side.upper().strip()
        if s not in {"BUY", "SELL"}:
            msg = "side must be 'BUY' or 'SELL'"
            raise ValueError(msg)
        if quantity is None and quoteOrderQty is None:
            msg = "Provide either 'quantity' (base units) or 'quoteOrderQty' (quote USDT)"
            raise ValueError(msg)

        params: dict[str, Any] = {"symbol": raw, "side": s, "type": "MARKET"}
        if quantity is not None:
            params["quantity"] = str(quantity)
        if quoteOrderQty is not None:
            params["quoteOrderQty"] = str(quoteOrderQty)
        if recvWindow is not None:
            params["recvWindow"] = int(recvWindow)
        return await self._request("POST", "/api/v3/order", params=params, signed=True)

    async def depth(self, symbol: str, limit: int = 100) -> dict[str, Any] | None:
        """Get order book depth for a symbol"""
        raw = _normalize_symbol(symbol)
        if limit <= 0:
            msg = "limit must be positive"
            raise ValueError(msg)
        return await self._request("GET", "/api/v3/depth", params={"symbol": raw, "limit": int(limit)})

    # --- Cleanup ---

    async def aclose(self) -> None:
        try:
            # Centralized client is managed globally; no-op here
            return
        except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError):
            pass
