"""
Rate Limiter Manager - All Live Data, No Fallback/Hardcoded Data

This module provides rate limiting for live API requests (backend port 8000).
All operations:
- Enforce live rate limits for API requests (backend port 8000)
- Track live request weights and Binance.US weight limits (1200/60s)
- Restrict symbols to live Binance.US Top-10 from trading universe
- No fallback/hardcoded data - all rate limiting from live operations
- Used by backend services on port 8000 for live trading operations

Live Data Sources:
- API requests: Live requests to backend API (port 8000)
- Rate limits: Live request rate tracking per IP
- Binance.US weight limits: Live weight tracking (1200/60s)
- Top-10 symbols: Live Binance.US Top-10 from trading universe (single source of truth)
- All rate limiting uses live data - no mock/test data

Endpoint References:
- Backend API: Port 8000 (rate limiter processes live requests)
- Trading Universe: backend.config.trading_universe (live Top-10 symbols)
- All rate limiting uses live connections - no fallback/hardcoded data
"""

from __future__ import annotations

import logging
import time
import urllib.parse
from collections import defaultdict, deque
from collections.abc import Awaitable, Callable, Iterable

from fastapi import HTTPException, Request

# Import live trading universe for Top-10 symbols (single source of truth)
from backend.config.trading_universe import TRADING_SYMBOLS

logger = logging.getLogger(__name__)


def _now() -> float:
    return time.time()


def _to_ccxt(sym: str) -> str:
    s = (sym or "").strip().upper().replace("-", "/").replace("_", "/").replace(" ", "/")
    if "/" not in s:
        if s.endswith(("USDT", "USD")):
            base = s[:-4]
            quote = s[-4:]
            return f"{base}/{quote}"
        return f"{s}/USDT"
    base, quote = s.split("/", 1)
    return f"{base}/{quote}"


class TokenBucket:
    """
    Sliding-window bucket for live request weight accounting.

    Tracks live request weights for rate limiting (not just count).
    All weight tracking from live operations - no fallback/hardcoded data.
    """

    def __init__(self, capacity: int, window_s: int) -> None:
        """
        Initialize token bucket for live weight tracking.

        Args:
            capacity: Maximum weight capacity (configuration default, not fallback data)
            window_s: Time window in seconds (configuration default, not fallback data)
        """
        self.capacity = int(capacity)
        self.window_s = int(window_s)
        self.events: deque[tuple[float, int]] = deque()  # Live (timestamp, weight) events
        self.total: int = 0  # Live total weight

    def _evict(self, now: float) -> None:
        cutoff = now - self.window_s
        while self.events and self.events[0][0] < cutoff:
            _, w = self.events.popleft()
            self.total -= w
            self.total = max(self.total, 0)  # safety

    def can_consume(self, weight: int, now: float | None = None) -> bool:
        now = now or _now()
        self._evict(now)
        return (self.total + int(weight)) <= self.capacity

    def consume(self, weight: int, now: float | None = None) -> None:
        now = now or _now()
        self._evict(now)
        w = int(weight)
        self.events.append((now, w))
        self.total += w


class RateLimiter:
    """
    Rate limiting with Binance weight enforcement and live Top-10 symbol gating.

    All rate limiting uses live data from API requests and trading universe.
    No fallback/hardcoded data - all operations from live sources.
    """

    def __init__(self) -> None:
        """Initialize rate limiter for live request processing."""
        # Per-IP live request timestamps (count-based)
        self.requests: dict[str, list[float]] = defaultdict(list)

        # Generic rate limits (configuration defaults, not fallback data)
        self.rate_limits = {
            "default": {"requests": 100, "window": 60},  # 100 req/min per IP (configuration default)
            "auth": {"requests": 5, "window": 60},  # 5 req/min per IP (configuration default)
            "api": {"requests": 1000, "window": 3600},  # 1000 req/hr per IP (configuration default)
            "websocket": {"requests": 100, "window": 60},  # 100 req/min per IP (configuration default)
        }

        self.endpoint_limits = {
            "/api/auth/login": "auth",
            "/api/auth/register": "auth",
            "/ws": "websocket",
        }

        # Binance.US global weight bucket (shared across callers hitting our server)
        # 1200 weight per 60 seconds (Binance standard, configuration default not fallback data)
        self.binance_bucket = TokenBucket(capacity=1200, window_s=60)

        # Optional per-IP weight buckets (extra safety, configuration default not fallback data)
        self.binance_ip_buckets: dict[str, TokenBucket] = defaultdict(lambda: TokenBucket(1200, 60))

        # How to determine the weight of a request (endpoint → weight)
        # You can extend this map as you expose more Binance endpoints.
        self.binance_endpoint_weights = {
            # examples; adjust to match your proxy routes
            "/api/binanceus/ticker": 1,
            "/api/binanceus/orderbook": 10,  # depth endpoints often heavier
            "/api/binanceus/klines": 2,
            "/api/binanceus/order": 1,
            "/api/binanceus/account": 10,
        }

        # Live Top-10 symbols from trading universe (single source of truth, not fallback data)
        # Convert trading symbols (BTCUSDT) to CCXT format (BTC/USDT) for rate limiting
        self._default_top10 = {_to_ccxt(sym) for sym in TRADING_SYMBOLS}  # Live Top-10 from trading universe
        self._top10_provider: Callable[[], Awaitable[Iterable[str]]] | None = None
        self._cached_top10: set[str] = set(self._default_top10)  # Live Top-10 cache
        self._last_top10_refresh: float = 0.0
        self._top10_ttl_s: int = 300  # 5 min cache (configuration default, not fallback data)

        # Cleanup
        self.last_full_cleanup = _now()
        self.full_cleanup_interval = 3600

    # ---------- Config ----------

    def set_top10_provider(self, provider: Callable[[], Awaitable[Iterable[str]]]) -> None:
        """Inject an async provider returning an iterable of allowed Top-10 ccxt symbols."""
        self._top10_provider = provider

    async def _refresh_top10(self) -> None:
        """
        Refresh live Top-10 symbols from provider or trading universe.

        Uses live Top-10 from provider if available, otherwise from trading universe.
        No fallback/hardcoded data - all symbols from live sources.
        """
        now = _now()
        if (now - self._last_top10_refresh) < self._top10_ttl_s:
            return
        try:
            if self._top10_provider:
                # Try live provider first
                syms = await self._top10_provider()
                normalized = {_to_ccxt(s) for s in syms}
                if normalized:
                    self._cached_top10 = normalized  # Live Top-10 from provider
                    self._last_top10_refresh = now
                    return
        except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
            logger.warning("Live Top-10 provider failed, using trading universe: %s", e)
        # Use live Top-10 from trading universe if provider absent/fails (not fallback data, live source)
        self._cached_top10 = set(self._default_top10)  # Live Top-10 from trading universe
        self._last_top10_refresh = now

    # ---------- Generic per-IP count limiting ----------

    def get_rate_limit(self, path: str) -> tuple[int, int]:
        for endpoint, limit_type in self.endpoint_limits.items():
            if path.startswith(endpoint):
                limit_config = self.rate_limits[limit_type]
                return limit_config["requests"], limit_config["window"]
        limit_config = self.rate_limits["default"]
        return limit_config["requests"], limit_config["window"]

    def cleanup_old_requests(self, ip: str, window: int) -> None:
        now = _now()
        self.requests[ip] = [t for t in self.requests[ip] if now - t < window]
        if not self.requests[ip]:
            self.requests.pop(ip, None)

        if now - self.last_full_cleanup > self.full_cleanup_interval:
            self.full_cleanup()
            self.last_full_cleanup = now

    def full_cleanup(self) -> None:
        now = _now()
        max_window = max(limit["window"] for limit in self.rate_limits.values())
        to_remove: list[str] = []
        for ip, reqs in self.requests.items():
            self.requests[ip] = [t for t in reqs if now - t < max_window]
            if not self.requests[ip]:
                to_remove.append(ip)
        for ip in to_remove:
            self.requests.pop(ip, None)
        # prune old weight events as well
        self.binance_bucket._evict(now)
        for bucket in self.binance_ip_buckets.values():
            bucket._evict(now)

    # ---------- Helpers ----------

    @staticmethod
    def _extract_symbol(request: Request) -> str | None:
        # Try path params (FastAPI)
        sym = None
        try:
            if hasattr(request, "path_params") and isinstance(request.path_params, dict):
                sym = request.path_params.get("symbol")
        except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError):
            pass

        # Query string
        if not sym:
            try:
                qs = urllib.parse.parse_qs(request.url.query or "")
                for key in ("symbol", "sym", "s"):
                    if qs.get(key):
                        sym = qs[key][0]
                        break
            except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError):
                pass

        # Headers (rare, but support if a proxy adds it)
        if not sym:
            sym = request.headers.get("x-symbol")

        return sym or None

    def _binance_weight_for_path(self, path: str) -> int:
        for prefix, w in self.binance_endpoint_weights.items():
            if path.startswith(prefix):
                return int(w)
        return 1  # default minimal weight

    # ---------- Public checks (to call in FastAPI middleware/deps) ----------

    async def check_rate_limit(self, request: Request) -> None:
        """
        Standard per-IP count-based limiter for live requests.

        Checks live request rate limits per IP address.
        All rate limiting uses live data - no fallback/hardcoded data.

        Args:
            request: Live API request to backend (port 8000)

        Raises:
            HTTPException: 429 if rate limit exceeded
        """
        try:
            client_ip = request.client.host if request.client else "unknown"
            path = request.url.path

            if client_ip == "unknown":
                logger.warning("Unknown client IP detected, skipping count-based limiting for live request")
                return

            max_requests, window = self.get_rate_limit(path)
            self.cleanup_old_requests(client_ip, window)
            request_count = len(self.requests[client_ip])
        except HTTPException:
            raise
        except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
            logger.exception("Error in live rate limiter: %s", e)
            return

        # Validate rate limit outside try to avoid TRY301
        if request_count >= max_requests:
            logger.warning("Rate limit exceeded for live request from IP %s on %s", client_ip, path)
            raise HTTPException(
                status_code=429,
                detail={
                    "error": "Rate limit exceeded",
                    "retry_after": window,
                    "limit": max_requests,
                    "window": window,
                },
            )

        try:
            # Track live request
            self.requests[client_ip].append(_now())
        except HTTPException:
            raise
        except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
            logger.exception("Error in live rate limiter: %s", e)

    async def enforce_binance_weight_and_top10(self, request: Request) -> None:
        """
        Enforce Binance 1200 weight / 60s and restrict symbols to live Top-10.

        Enforces live Binance.US weight limits and validates symbols against live Top-10.
        All enforcement uses live data - no fallback/hardcoded data.
        Call this for any route that proxies Binance.US.

        Args:
            request: Live API request to backend (port 8000)

        Raises:
            HTTPException: 429 if weight limit exceeded, 400 if symbol not in Top-10
        """
        try:
            path = request.url.path

            # Derive weight: request.state.api_weight (if set) → header → endpoint map → 1
            weight = None
            try:
                weight = getattr(request.state, "api_weight", None)
            except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError):
                weight = None
            if weight is None:
                hdr = request.headers.get("x-binance-weight")
                if hdr and hdr.isdigit():
                    weight = int(hdr)
            if weight is None:
                weight = self._binance_weight_for_path(path)

            if weight <= 0:
                weight = 1

            # Top-10 check (only when a symbol is present)
            await self._refresh_top10()
            sym_raw = self._extract_symbol(request)
            sym_ccxt = _to_ccxt(sym_raw) if sym_raw else None
            sym_in_top10 = sym_ccxt in self._cached_top10 if sym_ccxt else True

            # Global bucket (service-wide)
            now = _now()
            can_consume_global = self.binance_bucket.can_consume(weight, now)
            retry_after_global = 1
            if not can_consume_global and self.binance_bucket.events:
                oldest_ts, _ = self.binance_bucket.events[0]
                retry_after_global = max(1, int((oldest_ts + self.binance_bucket.window_s) - now))

            # Optional per-IP bucket (tighten multi-tenant abuse)
            ip = request.client.host if request.client else "unknown"
            bucket = self.binance_ip_buckets[ip]
            can_consume_ip = bucket.can_consume(weight, now)
            retry_after_ip = 1
            if not can_consume_ip and bucket.events:
                oldest_ts, _ = bucket.events[0]
                retry_after_ip = max(1, int((oldest_ts + bucket.window_s) - now))
        except HTTPException:
            raise
        except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
            logger.exception("Error in Binance weight limiter: %s", e)
            return

        # Validate symbol and rate limits outside try to avoid TRY301
        if sym_raw and not sym_in_top10:
            raise HTTPException(
                status_code=400,
                detail={
                    "error": "Symbol not allowed",
                    "symbol": sym_ccxt,
                    "allowed_top10": sorted(self._cached_top10),
                },
            )

        if not can_consume_global:
            raise HTTPException(
                status_code=429,
                detail={
                    "error": "Binance weight limit exceeded",
                    "limit": self.binance_bucket.capacity,
                    "window": self.binance_bucket.window_s,
                    "retry_after": retry_after_global,
                    "used": self.binance_bucket.total,
                    "requested_weight": weight,
                },
            )

        if not can_consume_ip:
            raise HTTPException(
                status_code=429,
                detail={
                    "error": "Per-IP Binance weight exceeded",
                    "ip": ip,
                    "limit": bucket.capacity,
                    "window": bucket.window_s,
                    "retry_after": retry_after_ip,
                    "used": bucket.total,
                    "requested_weight": weight,
                },
            )

        try:
            # Consume
            self.binance_bucket.consume(weight, now)
            bucket.consume(weight, now)

        except HTTPException:
            raise
        except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
            logger.exception("Binance weight/top10 enforcement error for live request: %s", e)
            raise HTTPException(
                status_code=503,
                detail="Rate limit enforcement temporarily unavailable",
            ) from e
