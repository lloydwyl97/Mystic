"""
Live Trading Service
Connects to the real Binance.US trading API for live order execution.
"""

import asyncio
import hashlib
import hmac
import logging
import os
import time
from datetime import datetime, timezone
from typing import Any

import ccxt
from dotenv import load_dotenv

from backend.utils.binance_weight_limiter import (
    BinanceWeightLimiter,
    CircuitOpenError,
    RateLimitedError,
)
from backend.utils.exceptions import APIError

# Optional imports - try at top level
try:
    from backend.services.canonical_http_client import canonical_http_client
except (ImportError, ModuleNotFoundError, AttributeError, ValueError, TypeError, RuntimeError):
    canonical_http_client = None

# Load environment variables
load_dotenv()

# Force IPv4 only for all connections (Binance US requirement)
try:
    import socket as _socket

    _orig_getaddrinfo = getattr(_socket, "getaddrinfo", None)

    def _ipv4_getaddrinfo(host, port, family=0, sock_type=0, proto=0, flags=0):
        try:
            return _orig_getaddrinfo(host, port, _socket.AF_INET, sock_type, proto, flags)
        except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError):
            return _orig_getaddrinfo(host, port, family, sock_type, proto, flags)

    if callable(_orig_getaddrinfo):
        _socket.getaddrinfo = _ipv4_getaddrinfo
except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError):
    pass

# Configure logging (guard to avoid duplicate handlers)
if not logging.getLogger().handlers:
    logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import from single source of truth
try:
    from backend.config.trading_universe import EXCHANGE_ID
except (ImportError, ModuleNotFoundError, AttributeError, ValueError, TypeError, RuntimeError) as e:
    msg = f"Failed to import trading_universe: {e}"
    raise RuntimeError(msg) from e


def _to_binance_pair(sym: str) -> str:
    """
    Normalize incoming symbols to Binance.US API form (e.g. ``BTCUSDT``).

    Accepts strings with separators (``/``, ``-``, ``_``, space), bare base
    symbols, and ``...USD`` quote forms; returns the canonical ``...USDT``
    Binance.US pair. Strings already in API form are returned unchanged.
    """
    s = (sym or "").strip().upper()
    if not s:
        return s
    s = s.replace("/", "").replace("-", "").replace("_", "").replace(" ", "")
    if s.endswith("USDT"):
        return s
    if s.endswith("USD"):
        return s[:-3] + "USDT"
    # bare base -> assume USDT quote
    return s + "USDT"


class LiveTradingService:
    """Service for live trading operations with real APIs (Binance.US only)."""

    def __init__(self) -> None:
        # Initialize Binance.US (spot) - LAZY AUTHENTICATION
        # Don't make any API calls on startup - only authenticate when first trade is needed
        self.binance: ccxt.binanceus | None = None
        self.authenticated = False
        self._limiter: BinanceWeightLimiter | None = None
        self._limiter_lock = asyncio.Lock()
        self._initialized = False

        # Prefer Binance US keys; fall back to generic if not set
        # Strip any trailing \r or whitespace from Windows line endings in .env
        raw_api_key = os.getenv("BINANCE_US_API_KEY") or os.getenv("BINANCE_API_KEY") or ""
        raw_secret = os.getenv("BINANCE_US_SECRET_KEY") or os.getenv("BINANCE_SECRET") or ""
        self.binance_api_key = raw_api_key.strip().rstrip("\r") if raw_api_key else None
        self.binance_secret = raw_secret.strip().rstrip("\r") if raw_secret else None

        # RATE LIMIT FIX: Add caching for balance and orders (reduces API weight by 80%)
        self._balance_cache: dict[str, Any] | None = None
        self._balance_cache_time: float = 0.0
        self._balance_cache_ttl: float = 30.0  # Cache for 30 seconds

        self._orders_cache: dict[str, Any] | None = None
        self._orders_cache_time: float = 0.0
        self._orders_cache_ttl: float = 10.0  # Cache for 10 seconds

        # Check for API keys but DON'T authenticate yet
        if not self.binance_api_key or not self.binance_secret:
            logger.warning(f"[WARNING] Live trading disabled: Missing API keys - API_KEY: {'SET' if self.binance_api_key else 'MISSING'}, SECRET: {'SET' if self.binance_secret else 'MISSING'}")
            return

        # Just log that we're ready - DON'T connect yet
        logger.info(f"[INFO] Live trading service ready (lazy init) - API key: {self.binance_api_key[:8]}...")

    async def _ensure_initialized(self) -> bool:
        """Lazy initialization - only connect to Binance when actually needed (async version)."""
        if self._initialized:
            return self.authenticated

        if not self.binance_api_key or not self.binance_secret:
            return False

        self._initialized = True
        logger.info("[INFO] Connecting to Binance.US (first use)...")

        try:
            self.binance = ccxt.binanceus(
                {
                    "apiKey": self.binance_api_key,
                    "secret": self.binance_secret,
                    "enableRateLimit": True,
                    "options": {
                        "defaultType": "spot",
                        "adjustForTimeDifference": True,
                        "fetchOHLCV": "public",
                        "fetchTicker": "public",
                        "fetchTickers": "public",
                    },
                },
            )

            # Test authentication with actual API call
            try:
                if hasattr(self.binance, "fetch_balance"):
                    # ================================================================
                    # PHASE 3 FIX #3: ADD MISSING AWAIT ON CCXT CALL
                    # ================================================================
                    # fetch_balance() is synchronous but blocking - wrap with asyncio.to_thread
                    test_balance = await asyncio.to_thread(self.binance.fetch_balance)
                    if test_balance:
                        self.authenticated = True
                        logger.info("[OK] Live trading service authenticated successfully")
                    else:
                        logger.warning("[WARNING] Could not fetch account balance")
                        self.binance = None
                else:
                    self.binance.check_required_credentials()
                    self.authenticated = True
                    logger.info("[OK] Live trading service credentials valid")

            except Exception as auth_e:
                error_str = str(auth_e)
                if "Invalid Api-Key ID" in error_str or "-2008" in error_str or "API-key format invalid" in error_str:
                    logger.warning(f"[WARNING] Live trading disabled: Invalid API keys - {auth_e}")
                    logger.info("[INFO] Please check your Binance US API key configuration")
                    self.binance = None
                elif "Signature" in error_str or "Permission denied" in error_str:
                    logger.warning(f"[WARNING] Live trading disabled: API key lacks trading permissions - {auth_e}")
                    logger.info("[INFO] Ensure your Binance US API key has 'Enable Spot & Margin Trading' permission")
                    self.binance = None
                else:
                    logger.exception(f"[ERROR] Live trading authentication failed: {auth_e}")
                    self.binance = None

        except Exception as e:
            logger.exception(f"[ERROR] Error initializing Binance.US client: {e}")
            logger.info("[INFO] Live trading will remain disabled")
            self.binance = None

        return self.authenticated
        # BUG #L4 FIX: Removed unreachable dead code block (was after return statement)
        # The cache attributes are already initialized in __init__ (lines 118-120)

    async def _get_limiter(self) -> BinanceWeightLimiter:
        """Return shared Binance weight limiter (Redis-backed)."""
        if self._limiter is not None:
            return self._limiter

        async with self._limiter_lock:
            if self._limiter is None:
                self._limiter = await BinanceWeightLimiter.create()
        return self._limiter

    async def get_account_balance(self) -> dict[str, Any]:
        """Get account balance from Binance.US (if configured) - with 30s cache."""
        try:
            # RATE LIMIT FIX: Return cached balance if fresh (< 30s old)
            # This reduces balance API calls by 80-90% (weight=10 per call)
            now = time.time()
            if self._balance_cache is not None and (now - self._balance_cache_time) < self._balance_cache_ttl:
                logger.debug(f"Returning cached balance (age: {now - self._balance_cache_time:.1f}s)")
                return self._balance_cache

            balances: dict[str, Any] = {}
            await self._ensure_initialized()
            if self.binance:
                try:
                    # Use direct API call to avoid margin endpoints
                    if canonical_http_client is None:
                        logger.warning("canonical_http_client not available")
                        return balances

                    # Prepare request for Binance.US account endpoint
                    api_key = self.binance_api_key
                    api_secret = self.binance_secret

                    if api_key and api_secret:
                        # Build request parameters
                        timestamp = int(time.time() * 1000)
                        query_string = f"timestamp={timestamp}&recvWindow=10000"

                        # Create signature
                        signature = hmac.new(
                            api_secret.encode("utf-8"),
                            query_string.encode("utf-8"),
                            hashlib.sha256,
                        ).hexdigest()

                        # Build headers and URL
                        headers = {"X-MBX-APIKEY": api_key}

                        url = f"https://api.binance.us/api/v3/account?{query_string}&signature={signature}"

                        # Make request
                        response = await canonical_http_client.make_request("GET", url, headers=headers)

                        # Check for HTTP errors
                        if response.status_code != 200:
                            logger.error(f"Binance.US API error: {response.status_code} - {response.text}")
                            msg = f"API error {response.status_code}: {response.text}"
                            raise APIError(msg)

                        account_data = response.json()

                        # Process balances
                        total_balances = {}
                        free_balances = {}
                        used_balances = {}

                        for balance in account_data.get("balances", []):
                            asset = balance.get("asset")
                            free = float(balance.get("free", 0))
                            locked = float(balance.get("locked", 0))
                            total = free + locked

                            if total > 0:
                                total_balances[asset] = total
                                free_balances[asset] = free
                                used_balances[asset] = locked

                        balances[EXCHANGE_ID] = {
                            "total": total_balances,
                            "free": free_balances,
                            "used": used_balances,
                            "timestamp": datetime.now(timezone.utc).isoformat(),
                        }

                        logger.info(f"Successfully fetched Binance.US balance: {len(total_balances)} assets")
                    else:
                        # No API keys available
                        logger.warning("Cannot fetch account balance: API keys not configured")
                        balances[EXCHANGE_ID] = {
                            "total": {},
                            "free": {},
                            "used": {},
                            "timestamp": datetime.now(timezone.utc).isoformat(),
                            "error": "API keys not configured",
                        }
                except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
                    logger.exception(f"Error fetching Binance.US balance: {e}")
                    # Return empty balance structure on error
                    balances[EXCHANGE_ID] = {
                        "total": {},
                        "free": {},
                        "used": {},
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                        "error": str(e),
                    }

            # RATE LIMIT FIX: Cache the successful response
            result = {
                "status": "success",
                "balances": balances,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "source": "live_trading_apis",
            }
            self._balance_cache = result
            self._balance_cache_time = time.time()
        except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
            logger.exception(f"Error fetching account balance: {e}")
            return {
                "status": "error",
                "message": str(e),
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
        else:
            return result

    async def get_balance(self, exchange: str = "binanceus", force_refresh: bool = False) -> dict[str, Any]:
        """
        Get balance in the shape expected by portfolio_engine execute_sell_fifo
        (execute_sell_fifo calls _entry_ensure_constraints):
        {"status": "success", "balance": {"total": {...}, "free": {...}, "used": {...}}}.
        Callers must use "free" for sell quantity cap (Binance uses free for order execution).
        Delegates to get_account_balance() and reshapes; uses same cache.
        force_refresh: if True, bypass cache for fresh balance (e.g. reconciliation).
        """
        if force_refresh:
            self._balance_cache = None
        result = await self.get_account_balance()
        if result.get("status") != "success":
            return {
                "status": result.get("status", "error"),
                "balance": {"total": {}, "free": {}, "used": {}},
            }
        normalized = (exchange or "").strip().lower().replace(" ", "").replace("_", "")
        exchange_key = EXCHANGE_ID if normalized == "binanceus" else (exchange or EXCHANGE_ID)
        by_exchange = result.get("balances", {})
        data = by_exchange.get(exchange_key) or by_exchange.get(EXCHANGE_ID) or {}
        total = data.get("total", {})
        free = data.get("free", {})
        used = data.get("used", {})
        return {
            "status": "success",
            "balance": {"total": dict(total), "free": dict(free), "used": dict(used)},
        }

    async def get_symbol_constraints(
        self,
        exchange: str,
        symbols: list[str] | None = None,
        force_refresh: bool = False,
    ) -> dict[str, dict[str, float]]:
        """
        Return per-symbol trading constraints (min_qty, qty_step, min_notional, tick_size)
        from CCXT markets. Only exchange="binanceus" supported.
        Phase 2: CCXT-first; uses market.info.filters then CCXT normalized fields.
        """
        out: dict[str, dict[str, float]] = {}
        if (exchange or "").strip().lower() not in ("binanceus", "binance"):
            logger.warning("get_symbol_constraints: only exchange=binanceus supported, got %r", exchange)
            return out
        if not symbols:
            return out
        await self._ensure_initialized()
        if not self.binance:
            logger.warning("get_symbol_constraints: Binance client not available")
            return out
        try:
            await asyncio.to_thread(self.binance.load_markets, force_refresh)
        except Exception as e:
            logger.exception("get_symbol_constraints: load_markets failed: %s", e)
            return out
        markets = getattr(self.binance, "markets", None) or {}
        for sym in symbols:
            ccxt_symbol = sym if "/" in sym else (sym.replace("USDT", "/USDT") if sym.endswith("USDT") else sym)
            if not ccxt_symbol:
                continue
            market = markets.get(ccxt_symbol)
            if not market:
                logger.debug("get_symbol_constraints: no market for %r", ccxt_symbol)
                out[ccxt_symbol] = {"qty_step": 0.0, "min_qty": 0.0, "min_notional": 0.0, "tick_size": 0.0}
                continue
            info = market.get("info") or {}
            filters_list = info.get("filters")
            qty_step = 0.0
            min_qty = 0.0
            min_notional = 0.0
            tick_size = 0.0
            if filters_list:
                filters_by_type: dict[str, Any] = {}
                for f in filters_list:
                    if isinstance(f, dict) and f.get("filterType"):
                        filters_by_type[f["filterType"]] = f
                lot = filters_by_type.get("LOT_SIZE") or filters_by_type.get("MARKET_LOT_SIZE")
                if lot:
                    try:
                        min_qty = float(lot.get("minQty", 0))
                        qty_step = float(lot.get("stepSize", 0))
                    except (TypeError, ValueError):
                        logger.debug("get_symbol_constraints: missing/invalid LOT_SIZE for %s", ccxt_symbol)
                notional = filters_by_type.get("MIN_NOTIONAL") or filters_by_type.get("NOTIONAL")
                if notional:
                    try:
                        min_notional = float(notional.get("minNotional", notional.get("notionalMin", 0)))
                    except (TypeError, ValueError):
                        logger.debug("get_symbol_constraints: missing/invalid MIN_NOTIONAL for %s", ccxt_symbol)
                price_f = filters_by_type.get("PRICE_FILTER")
                if price_f:
                    try:
                        tick_size = float(price_f.get("tickSize", 0))
                    except (TypeError, ValueError):
                        logger.debug("get_symbol_constraints: missing/invalid PRICE_FILTER for %s", ccxt_symbol)
            if not filters_list or (qty_step == 0.0 and min_qty == 0.0):
                limits = market.get("limits") or {}
                prec = market.get("precision") or {}
                if min_qty == 0.0:
                    am = limits.get("amount") or {}
                    if isinstance(am, dict) and am.get("min") is not None:
                        min_qty = float(am["min"])
                    else:
                        logger.debug("get_symbol_constraints: missing min_qty for %s", ccxt_symbol)
                if qty_step == 0.0 and prec.get("amount") is not None:
                    try:
                        p = prec["amount"]
                        qty_step = 10.0 ** (-int(p)) if p is not None else 0.0
                    except (TypeError, ValueError):
                        pass
                if min_notional == 0.0:
                    cost = limits.get("cost") or {}
                    if isinstance(cost, dict) and cost.get("min") is not None:
                        min_notional = float(cost["min"])
                    else:
                        logger.debug("get_symbol_constraints: missing min_notional for %s", ccxt_symbol)
            if qty_step == 0.0 or min_qty == 0.0:
                logger.debug("get_symbol_constraints: %s qty_step=%s min_qty=%s (missing flagged)", ccxt_symbol, qty_step, min_qty)
            out[ccxt_symbol] = {"qty_step": qty_step, "min_qty": min_qty, "min_notional": min_notional, "tick_size": tick_size}
        return out

    async def get_open_orders(self) -> dict[str, Any]:
        """Get open spot orders from Binance.US (ALL symbols in single call - saves 45 weight) - with 10s cache."""
        try:
            # RATE LIMIT FIX: Return cached orders if fresh (< 10s old)
            # Orders don't change that often, safe to cache briefly
            now = time.time()
            if self._orders_cache is not None and (now - self._orders_cache_time) < self._orders_cache_ttl:
                logger.debug(f"Returning cached orders (age: {now - self._orders_cache_time:.1f}s)")
                return self._orders_cache

            orders: dict[str, Any] = {}
            # Lazy init - only connect when actually needed
            await self._ensure_initialized()
            if self.binance:
                try:
                    limiter = await self._get_limiter()
                    # CRITICAL FIX: Fetch ALL open orders in ONE API call (5 weight total)
                    # Instead of looping 10 symbols x 5 weight = 50 weight
                    # This reduces API weight by 90% (50 -> 5)
                    await limiter.consume("/api/v3/openOrders", weight=5, wait=True, timeout=5.0)
                    # Fetch all open orders without symbol filter (gets all pairs)
                    binance_orders = await asyncio.to_thread(self.binance.fetch_open_orders)

                    orders[EXCHANGE_ID] = [
                        {
                            "id": o.get("id"),
                            "symbol": o.get("symbol"),
                            "type": o.get("type"),
                            "side": o.get("side"),
                            "amount": o.get("amount"),
                            "price": o.get("price"),
                            "status": o.get("status"),
                            "timestamp": o.get("timestamp"),
                        }
                        for o in binance_orders
                        if isinstance(o, dict)
                    ]
                    logger.debug(f"Fetched {len(binance_orders)} open orders with single API call (5 weight)")
                except (RateLimitedError, CircuitOpenError) as e:
                    logger.warning(f"Rate limited fetching open orders: {e}")
                    orders[EXCHANGE_ID] = []
                except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
                    logger.exception(f"Error fetching Binance.US orders: {e}")
                    orders[EXCHANGE_ID] = []

            # RATE LIMIT FIX: Cache the successful response
            result = {
                "status": "success",
                "orders": orders,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "source": "live_trading_apis",
            }
            self._orders_cache = result
            self._orders_cache_time = time.time()
        except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
            logger.exception(f"Error fetching open orders: {e}")
            return {
                "status": "error",
                "message": str(e),
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
        else:
            return result

    async def get_trade_history(self, symbol: str | None = None, limit: int = 100) -> dict[str, Any]:
        """Get trade history from Binance.US. Symbol optional; falls back to all."""
        try:
            trades: dict[str, Any] = {}
            await self._ensure_initialized()
            if self.binance:
                try:
                    # Ensure markets are loaded before fetching trades
                    if not self.binance.markets:
                        await asyncio.to_thread(self.binance.load_markets)

                    binance_symbol = _to_binance_pair(symbol) if symbol else None
                    if binance_symbol:
                        binance_trades = await asyncio.to_thread(self.binance.fetch_my_trades, binance_symbol, limit=limit)
                    else:
                        # When no symbol specified, return empty list (fetching all trades has issues)
                        binance_trades = []
                    if isinstance(binance_trades, list):
                        trades[EXCHANGE_ID] = [
                            {
                                "id": t.get("id"),
                                "symbol": t.get("symbol"),
                                "side": t.get("side"),
                                "amount": t.get("amount"),
                                "price": t.get("price"),
                                "cost": t.get("cost"),
                                "fee": t.get("fee"),
                                "timestamp": t.get("timestamp"),
                            }
                            for t in binance_trades
                            if isinstance(t, dict) and t.get("id")
                        ]
                    else:
                        trades[EXCHANGE_ID] = []
                except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
                    logger.exception(f"Error fetching Binance.US trades: {e}")
                    trades[EXCHANGE_ID] = []
            return {
                "status": "success",
                "trades": trades,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "source": "live_trading_apis",
            }
        except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
            logger.exception(f"Error fetching trade history: {e}")
            return {
                "status": "error",
                "message": str(e),
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }

    async def place_order(
        self,
        exchange: str,
        symbol: str,
        order_type: str,
        side: str,
        amount: float,
        price: float | None = None,
        client_order_id: str | None = None,
        time_in_force: str | None = None,
    ) -> dict[str, Any]:
        """Place a new spot order on Binance.US. Accepts exchange EXCHANGE_ID (or 'binance' for back-compat)."""
        try:
            # Lazy init - connect to Binance only when placing first trade
            await self._ensure_initialized()
            if exchange.lower() in {EXCHANGE_ID.lower(), "binance", "binanceus"} and self.binance:
                pair = _to_binance_pair(symbol)
                params: dict[str, Any] = {}
                if client_order_id:
                    params["clientOrderId"] = client_order_id
                if time_in_force and str(order_type).lower() == "limit":
                    params["timeInForce"] = time_in_force
                order = await asyncio.to_thread(
                    self.binance.create_order,
                    symbol=pair,
                    type=order_type,
                    side=side,
                    amount=amount,
                    price=price,
                    params=params,
                )
                return {
                    "status": "success",
                    "order": {
                        "id": order.get("id"),
                        "symbol": order.get("symbol"),
                        "type": order.get("type"),
                        "side": order.get("side"),
                        "amount": order.get("amount"),
                        "price": order.get("price"),
                        "filled": order.get("filled"),
                        "average": order.get("average"),
                        "cost": order.get("cost"),
                        "clientOrderId": order.get("clientOrderId"),
                        "status": order.get("status"),
                        "timestamp": order.get("timestamp"),
                    },
                    "exchange": EXCHANGE_ID,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                }
            return {
                "status": "error",
                "message": f"Exchange {exchange} not available or not configured",
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
        except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
            logger.exception(f"Error placing order: {e}")
            return {
                "status": "error",
                "message": str(e),
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
        except Exception as e:
            err_str = str(e)
            FILTER_ERR_HINTS = ("LOT_SIZE", "MIN_NOTIONAL", "PRICE_FILTER", "-1013")
            if any(h in err_str for h in FILTER_ERR_HINTS):
                try:
                    from backend.services.portfolio_engine import get_portfolio_engine

                    engine = get_portfolio_engine()
                    sym = symbol if "/" in symbol else f"{symbol[:-4]}/{symbol[-4:]}"
                    await engine._force_refresh_symbol_constraints(sym)
                except Exception as erf:
                    logger.warning("CONSTRAINTS_FORCE_REFRESH_FAILED: %s %s", symbol, erf)
            logger.exception(f"Error placing order: {e}")
            return {
                "status": "error",
                "message": str(e),
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }

    async def fetch_order(self, exchange: str, order_id: str, symbol: str) -> dict[str, Any]:
        """Fetch order status from exchange. For PARTIALLY_FILLED verification."""
        try:
            await self._ensure_initialized()
            if exchange.lower() in {EXCHANGE_ID.lower(), "binance", "binanceus"} and self.binance:
                pair = _to_binance_pair(symbol)
                order = await asyncio.to_thread(self.binance.fetch_order, id=order_id, symbol=pair)
                return {
                    "status": "success",
                    "order": {
                        "id": order.get("id"),
                        "symbol": order.get("symbol"),
                        "status": order.get("status"),
                        "filled": order.get("filled"),
                        "average": order.get("average"),
                        "amount": order.get("amount"),
                        "cost": order.get("cost"),
                    },
                }
            return {"status": "error", "message": "Exchange not available"}
        except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
            logger.exception("fetch_order %s: %s", order_id, e)
            return {"status": "error", "message": str(e)}

    async def cancel_order(self, exchange: str, order_id: str, symbol: str) -> dict[str, Any]:
        """Cancel an existing spot order on Binance.US."""
        try:
            await self._ensure_initialized()
            if exchange.lower() in {EXCHANGE_ID.lower(), "binance", "binanceus"} and self.binance:
                pair = _to_binance_pair(symbol)
                await asyncio.to_thread(self.binance.cancel_order, id=order_id, symbol=pair)
                return {
                    "status": "success",
                    "message": f"Order {order_id} cancelled successfully",
                    "exchange": EXCHANGE_ID,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                }
            return {
                "status": "error",
                "message": f"Exchange {exchange} not available or not configured",
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
        except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
            logger.exception(f"Error cancelling order: {e}")
            return {
                "status": "error",
                "message": str(e),
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }

    async def get_positions(self) -> dict[str, Any]:
        """Spot trading => no leveraged positions. Return empty list per exchange."""
        try:
            positions: dict[str, Any] = {}
            if self.binance:
                positions[EXCHANGE_ID] = []
            return {
                "status": "success",
                "positions": positions,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "source": "live_trading_apis",
                "note": "Spot trading - no positions, use balances instead",
            }
        except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
            logger.exception(f"Error fetching positions: {e}")
            return {
                "status": "error",
                "message": str(e),
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }

    async def get_market_price(self, symbol: str) -> dict[str, Any] | None:
        """Get current market price for a symbol using direct API call (Binance.US)."""
        try:
            if canonical_http_client is None:
                logger.warning("canonical_http_client not available")
                return None

            # Convert symbol to Binance format
            pair = _to_binance_pair(symbol)

            # Use public endpoint - no authentication required
            url = f"https://api.binance.us/api/v3/ticker/price?symbol={pair}"

            # Make request
            response = await canonical_http_client.make_request("GET", url)

            # Check for HTTP errors
            if response.status_code != 200:
                logger.error(f"Binance.US price API error: {response.status_code} - {response.text}")
                return None

            data = response.json()

            def safe_float(v: Any, default: float = 0.0) -> float:
                if v is None or v is False or v is True:
                    return default
                try:
                    return float(v)
                except (ValueError, TypeError):
                    return default

            return {
                "price": safe_float(data.get("price")),
                "symbol": symbol,
                "source": EXCHANGE_ID,
            }

        except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
            logger.exception(f"Error getting market price for {symbol}: {e}")
            return None

    async def get_account_balances(self) -> dict[str, Any]:
        """Get account balances (plural) - alias for get_account_balance"""
        return await self.get_account_balance()


# Global instance
trading_service = LiveTradingService()
