import logging
import time
from datetime import datetime, timezone
from typing import Any

# Direct imports for production
import ccxt

CCXT_AVAILABLE = True

# Initialize logger (no basicConfig here)
logger = logging.getLogger(__name__)

# Import from single source of truth
try:
    from backend.config.trading_universe import EXCHANGE_ID, TRADING_SYMBOLS
except (ImportError, ModuleNotFoundError, AttributeError, ValueError, TypeError, RuntimeError) as e:
    msg = f"Failed to import trading_universe: {e}"
    raise RuntimeError(msg) from e

# Configuration constants
REQUEST_TIMEOUT = 10  # seconds
MAX_RETRIES = 3
RETRY_DELAY = 1  # seconds

# All Live Data, No Fallback/Hardcoded Data
TOP10_BINANCEUS = list(TRADING_SYMBOLS)

# Exchange instance state - using dict to avoid global keyword
_exchange_instance_state: dict[str, Any | None] = {"instance": None}
_health_issues = []


def _to_ccxt_symbol(symbol: str) -> str:
    """Convert concatenated symbol to ccxt format (e.g., BTCUSDT -> BTC/USDT)"""
    if not symbol:
        msg = "Symbol cannot be empty"
        raise ValueError(msg)

    # Remove any existing slashes and convert to uppercase
    clean_symbol = symbol.replace("/", "").upper()

    # Validate against Top-10 allowlist
    if clean_symbol not in TOP10_BINANCEUS:
        msg = f"Symbol {clean_symbol} not in Binance US Top-10 allowlist"
        raise ValueError(msg)

    # Convert to ccxt format (add slash before USDT)
    if clean_symbol.endswith("USDT"):
        base = clean_symbol[:-4]  # Remove USDT
        return f"{base}/USDT"
    msg = f"Unsupported symbol format: {clean_symbol}"
    raise ValueError(msg)


def _from_ccxt_symbol(ccxt_symbol: str) -> str:
    """Convert ccxt symbol format to concatenated format (e.g., BTC/USDT -> BTCUSDT)"""
    if not ccxt_symbol or "/" not in ccxt_symbol:
        msg = f"Invalid ccxt symbol format: {ccxt_symbol}"
        raise ValueError(msg)

    return ccxt_symbol.replace("/", "")


def _get_exchange() -> Any | None:
    """Lazy initialization of exchange client with proper configuration"""
    if _exchange_instance_state["instance"] is not None:
        return _exchange_instance_state["instance"]

    if not CCXT_AVAILABLE:
        _health_issues.append("ccxt module not available")
        return None

    try:
        _exchange_instance_state["instance"] = ccxt.binanceus(  # type: ignore[assignment]
            {
                "options": {
                    "defaultType": "spot",
                    "adjustForTimeDifference": True,
                },
                "timeout": REQUEST_TIMEOUT * 1000,  # ccxt expects milliseconds
                "enableRateLimit": True,  # Enable built-in rate limiting
                "rateLimit": 1200,  # Binance US rate limit
            },
        )

        # Test connection
        _exchange_instance_state["instance"].load_markets()
        logger.info(f"Exchange client initialized successfully: {EXCHANGE_ID}")
        return _exchange_instance_state["instance"]

    except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
        error_msg = f"Failed to initialize exchange client: {e}"
        logger.exception(error_msg)
        _health_issues.append(error_msg)
        return None


def _make_request_with_retry(func, *args, **kwargs) -> Any | None:
    """Make request with retry logic and proper error handling"""
    for attempt in range(MAX_RETRIES):
        try:
            return func(*args, **kwargs)
        except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
            if attempt == MAX_RETRIES - 1:
                logger.exception(f"Request failed after {MAX_RETRIES} attempts: {e}")
                return None
            logger.warning(f"Request attempt {attempt + 1} failed: {e}, retrying...")
            time.sleep(RETRY_DELAY * (attempt + 1))  # Exponential backoff

    return None


def fetch_prices(symbol: str = "BTCUSDT") -> dict[str, Any]:
    """Fetch prices from Binance US with structured response"""
    start_time = time.time()
    timestamp = datetime.now(timezone.utc).isoformat()

    try:
        # Validate and convert symbol format
        ccxt_symbol = _to_ccxt_symbol(symbol)

        # Get exchange instance
        exchange = _get_exchange()
        if not exchange:
            return {
                "success": False,
                "error": "Exchange not available",
                "symbol": symbol,
                "ccxt_symbol": None,
                "exchange_id": EXCHANGE_ID,
                "timestamp": timestamp,
                "latency_ms": int((time.time() - start_time) * 1000),
                "health_issues": _health_issues.copy(),
            }

        # Fetch ticker with retry logic
        ticker_data = _make_request_with_retry(exchange.fetch_ticker, ccxt_symbol)

        if not ticker_data:
            return {
                "success": False,
                "error": "Failed to fetch ticker data",
                "symbol": symbol,
                "ccxt_symbol": ccxt_symbol,
                "exchange_id": EXCHANGE_ID,
                "timestamp": timestamp,
                "latency_ms": int((time.time() - start_time) * 1000),
                "health_issues": _health_issues.copy(),
            }

        # Extract price and metadata
        price = ticker_data.get("last")
        if price is None:
            return {
                "success": False,
                "error": "No price data in ticker response",
                "symbol": symbol,
                "ccxt_symbol": ccxt_symbol,
                "exchange_id": EXCHANGE_ID,
                "timestamp": timestamp,
                "latency_ms": int((time.time() - start_time) * 1000),
                "ticker_data": ticker_data,
            }

        latency_ms = int((time.time() - start_time) * 1000)

        logger.debug(f"Fetched price for {symbol}: ${price:.2f} (latency: {latency_ms}ms)")

        return {
            "success": True,
            "symbol": symbol,
            "ccxt_symbol": ccxt_symbol,
            "exchange_id": EXCHANGE_ID,
            "price": float(price),
            "timestamp": timestamp,
            "latency_ms": latency_ms,
            "ticker_data": {
                "bid": ticker_data.get("bid"),
                "ask": ticker_data.get("ask"),
                "high": ticker_data.get("high"),
                "low": ticker_data.get("low"),
                "volume": ticker_data.get("baseVolume"),
                "quote_volume": ticker_data.get("quoteVolume"),
                "change": ticker_data.get("change"),
                "percentage": ticker_data.get("percentage"),
                "timestamp": ticker_data.get("timestamp"),
            },
        }

    except ValueError as e:
        # Symbol validation error
        logger.exception(f"Symbol validation error for {symbol}: {e}")
        return {
            "success": False,
            "error": str(e),
            "symbol": symbol,
            "ccxt_symbol": None,
            "exchange_id": EXCHANGE_ID,
            "timestamp": timestamp,
            "latency_ms": int((time.time() - start_time) * 1000),
            "health_issues": _health_issues.copy(),
        }
    except (TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
        logger.exception(f"Error fetching price for {symbol}: {e}")
        return {
            "success": False,
            "error": str(e),
            "symbol": symbol,
            "ccxt_symbol": None,
            "exchange_id": EXCHANGE_ID,
            "timestamp": timestamp,
            "latency_ms": int((time.time() - start_time) * 1000),
            "health_issues": _health_issues.copy(),
        }


def check_arbitrage(symbols: list[str] | None = None) -> dict[str, Any]:
    """Check for arbitrage opportunities across Binance US Top-10 symbols"""
    timestamp = datetime.now(timezone.utc).isoformat()

    try:
        # Use provided symbols or default to first 5 from trading_universe (live data)
        if symbols is None:
            symbols = list(TRADING_SYMBOLS[:5]) if len(TRADING_SYMBOLS) >= 5 else list(TRADING_SYMBOLS)

        # Validate all symbols against allowlist
        validated_symbols = []
        for symbol in symbols:
            try:
                _to_ccxt_symbol(symbol)  # This validates against allowlist
                validated_symbols.append(symbol)
            except ValueError as e:
                logger.warning(f"Skipping invalid symbol {symbol}: {e}")

        if not validated_symbols:
            return {
                "success": False,
                "error": "No valid symbols provided",
                "timestamp": timestamp,
                "exchange_id": EXCHANGE_ID,
                "health_issues": _health_issues.copy(),
            }

        # Fetch prices for all symbols
        prices = {}
        failed_symbols = []

        for symbol in validated_symbols:
            result = fetch_prices(symbol)
            if result["success"]:
                prices[symbol] = result
            else:
                failed_symbols.append(symbol)
                logger.warning(f"Failed to fetch price for {symbol}: {result.get('error', 'Unknown error')}")

        if not prices:
            return {
                "success": False,
                "error": "Failed to fetch prices for any symbols",
                "timestamp": timestamp,
                "exchange_id": EXCHANGE_ID,
                "failed_symbols": failed_symbols,
                "health_issues": _health_issues.copy(),
            }

        # Calculate price statistics for arbitrage analysis
        price_values = [data["price"] for data in prices.values()]
        min_price = min(price_values)
        max_price = max(price_values)
        avg_price = sum(price_values) / len(price_values)

        # Find symbols with min/max prices
        min_symbol = next(symbol for symbol, data in prices.items() if data["price"] == min_price)
        max_symbol = next(symbol for symbol, data in prices.items() if data["price"] == max_price)

        # Calculate spread (potential arbitrage opportunity)
        spread_percentage = ((max_price - min_price) / min_price) * 100 if min_price > 0 else 0

        # Determine if there's a significant arbitrage opportunity
        # Threshold: > 0.1% spread (adjustable based on trading costs)
        arbitrage_threshold = 0.1
        has_arbitrage = spread_percentage > arbitrage_threshold

        result = {
            "success": True,
            "timestamp": timestamp,
            "exchange_id": EXCHANGE_ID,
            "symbols_checked": validated_symbols,
            "prices_fetched": len(prices),
            "failed_symbols": failed_symbols,
            "price_data": prices,
            "arbitrage_analysis": {
                "has_opportunity": has_arbitrage,
                "spread_percentage": round(spread_percentage, 4),
                "spread_threshold": arbitrage_threshold,
                "min_price": {
                    "symbol": min_symbol,
                    "price": min_price,
                    "ccxt_symbol": prices[min_symbol]["ccxt_symbol"],
                },
                "max_price": {
                    "symbol": max_symbol,
                    "price": max_price,
                    "ccxt_symbol": prices[max_symbol]["ccxt_symbol"],
                },
                "avg_price": round(avg_price, 4),
                "price_range": round(max_price - min_price, 4),
            },
            "health_issues": _health_issues.copy(),
        }

        if has_arbitrage:
            logger.info(f"[ARBITRAGE] Opportunity detected: {spread_percentage:.2f}% spread between {min_symbol} and {max_symbol}")
        else:
            logger.debug(f"[ARBITRAGE] No significant opportunity: {spread_percentage:.2f}% spread (threshold: {arbitrage_threshold}%)")
    except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
        logger.exception(f"Arbitrage check failed: {e}")
        return {
            "success": False,
            "error": str(e),
            "timestamp": timestamp,
            "exchange_id": EXCHANGE_ID,
            "health_issues": _health_issues.copy(),
        }
    else:
        return result


def get_bot_health() -> dict[str, Any]:
    """Get arbitrage bot health status"""
    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "exchange_id": EXCHANGE_ID,
        "ccxt_available": CCXT_AVAILABLE,
        "exchange_initialized": _exchange_instance_state["instance"] is not None,
        "health_issues": _health_issues.copy(),
        "status": "healthy" if CCXT_AVAILABLE and _exchange_instance_state["instance"] is not None and not _health_issues else "degraded",
        "supported_symbols": TOP10_BINANCEUS.copy(),
        "config": {
            "request_timeout": REQUEST_TIMEOUT,
            "max_retries": MAX_RETRIES,
            "retry_delay": RETRY_DELAY,
            "rate_limiting_enabled": True,
        },
    }
