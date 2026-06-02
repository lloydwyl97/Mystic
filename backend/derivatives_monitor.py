import logging
import os
from typing import Any

import httpx

logger = logging.getLogger(__name__)

# Import from single source of truth
try:
    from backend.config.trading_universe import EXCHANGE_ID
except (ImportError, ModuleNotFoundError, AttributeError, ValueError, TypeError, RuntimeError) as e:
    msg = f"Failed to import EXCHANGE_ID from trading_universe: {e}"
    raise RuntimeError(msg) from e


def _http_get(
    url: str,
    params: dict[str, Any] | None = None,
    headers: dict[str, str] | None = None,
) -> dict[str, Any]:
    try:
        resp = httpx.get(url, params=params or {}, headers=headers or {}, timeout=10.0)
        resp.raise_for_status()
        data = resp.json()
        if not isinstance(data, dict):
            logger.warning(f"Unexpected JSON response type: {type(data)}")
            result = {}
        else:
            result = data
    except httpx.HTTPStatusError as e:
        # Handle 451 "Unavailable For Legal Reasons" (Binance Futures blocked in US)
        if e.response.status_code == 451:
            logger.debug(f"API blocked in your region (HTTP 451): {url}")
            return {}
        # Log other HTTP errors as warnings, not exceptions
        logger.warning(f"HTTP {e.response.status_code} error for {url}")
        return {}
    except httpx.RequestError as e:
        logger.warning(f"HTTP request failed for {url}: {e}")
        return {}
    except ValueError as e:
        logger.warning(f"JSON decode failed for {url}: {e}")
        return {}
    else:
        return result


def fetch_binance_open_interest(symbol: str) -> dict[str, Any]:
    """Fetch open interest data from Binance Futures API.

    Returns aggregated derivatives positioning data showing market sentiment
    and potential volatility based on futures/perpetuals open interest.

    Note: Binance US does not support futures trading - this will return empty
    for Binance US users (HTTP 451).
    """
    # Check if using Binance US (no futures support)
    exchange_id = os.getenv("EXCHANGE_ID", "binanceus")
    if exchange_id in {"binanceus", "binance_us"}:
        logger.debug("Open interest data not available on Binance US (futures not supported)")
        return {}

    api_key = os.getenv("BINANCE_API_KEY", "")
    if not api_key:
        logger.debug("BINANCE_API_KEY not set; cannot fetch open interest data")
        return {}

    # Normalize symbol (remove USDT and add USDT back for futures)
    normalized = symbol.upper().replace("/", "").replace("-", "")
    if not normalized.endswith("USDT"):
        normalized += "USDT"

    url = "https://fapi.binance.com/fapi/v1/openInterest"
    params = {
        "symbol": normalized,
    }
    headers = {"X-MBX-APIKEY": api_key, "User-Agent": "mystic-trading/1.0"}

    data = _http_get(url, params=params, headers=headers)

    if not data or not isinstance(data, dict):
        return {}

    try:
        open_interest = float(data.get("openInterest", 0))
        timestamp = int(data.get("time", 0))
    except (ValueError, TypeError, KeyError):
        return {}

    if open_interest <= 0:
        return {}

    # Get additional market data for context
    market_data = _fetch_futures_ticker_data(normalized)

    # Calculate positioning metrics
    positioning_signals = _analyze_positioning(open_interest, market_data, symbol)

    return {
        "symbol": symbol,
        "open_interest": open_interest,
        "positioning_signals": positioning_signals,
        "timestamp": timestamp,
    }


def _fetch_futures_ticker_data(symbol: str) -> dict[str, Any]:
    """Fetch 24hr ticker data for additional context."""
    api_key = os.getenv("BINANCE_API_KEY", "")
    if not api_key:
        return {}

    url = "https://fapi.binance.com/fapi/v1/ticker/24hr"
    params = {"symbol": symbol}
    headers = {"X-MBX-APIKEY": api_key, "User-Agent": "mystic-trading/1.0"}

    data = _http_get(url, params=params, headers=headers)
    return data if isinstance(data, dict) else {}


def _analyze_positioning(open_interest: float, market_data: dict[str, Any], _symbol: str) -> dict[str, Any]:
    """Analyze open interest and market data to derive positioning signals."""

    # Get market data values
    try:
        price_change_pct = float(market_data.get("priceChangePercent", 0))
        volume = float(market_data.get("volume", 0))
        count_top_bid = float(market_data.get("countTopBid", 0))
        count_top_ask = float(market_data.get("countTopAsk", 0))
    except (ValueError, TypeError):
        price_change_pct = 0.0
        volume = 0.0
        count_top_bid = 0.0
        count_top_ask = 0.0

    # Calculate positioning metrics
    bid_ask_imbalance = count_top_bid / max(count_top_ask + count_top_bid, 1)

    # Open interest relative to volume (liquidity positioning)
    oi_volume_ratio = open_interest / max(volume, 1)

    # Price momentum vs positioning - GRADUAL SCALING (was binary 1.0/-1.0)
    # Price momentum component: scale from -1 to +1 based on price change (+-5% = full signal)
    price_momentum = max(-1.0, min(1.0, price_change_pct / 5.0))

    # Imbalance component: 0.5 is neutral, scale from -1 to +1
    imbalance_signal = (bid_ask_imbalance - 0.5) * 2.0  # 0.0->-1.0, 0.5->0.0, 1.0->+1.0

    # Combined alignment: average of price momentum and order flow
    if (price_momentum > 0 and imbalance_signal > 0) or (price_momentum < 0 and imbalance_signal < 0):
        # Aligned - bullish or bearish with conviction
        momentum_alignment = (price_momentum + imbalance_signal) / 2.0
    else:
        # Misaligned - reduce confidence, stay closer to neutral
        momentum_alignment = (price_momentum + imbalance_signal) / 4.0

    # Volatility expectation based on open interest changes
    # High OI often precedes volatility
    volatility_expectation = min(oi_volume_ratio / 1000.0, 1.0)  # Normalize

    # Market positioning bias
    if bid_ask_imbalance > 0.6:
        positioning_bias = "bullish"
        bias_strength = min(bid_ask_imbalance, 1.0)
    elif bid_ask_imbalance < 0.4:
        positioning_bias = "bearish"
        bias_strength = min(1.0 - bid_ask_imbalance, 1.0)
    else:
        positioning_bias = "neutral"
        bias_strength = 0.5

    # Extreme positioning warnings
    extreme_positioning = oi_volume_ratio > 2000.0  # Very high OI relative to spot volume

    return {
        "open_interest_volume_ratio": oi_volume_ratio,
        "bid_ask_imbalance": bid_ask_imbalance,
        "momentum_alignment": momentum_alignment,
        "volatility_expectation": volatility_expectation,
        "positioning_bias": positioning_bias,
        "bias_strength": bias_strength,
        "extreme_positioning": extreme_positioning,
        "price_change_pct": price_change_pct,
    }


def derivatives_signal_check(symbol: str) -> dict[str, Any]:
    """Get derivatives positioning signals for decision making."""
    data = fetch_binance_open_interest(symbol)

    if not data:
        return {
            "oi_volume_ratio": 0.0,
            "positioning_volatility": 0.0,
            "momentum_alignment": 0.0,
            "positioning_bias": "neutral",
            "bias_strength": 0.0,
            "extreme_positioning": False,
        }

    signals = data.get("positioning_signals", {})

    # Transform for decision making
    oi_ratio = float(signals.get("open_interest_volume_ratio", 0.0))
    volatility = float(signals.get("volatility_expectation", 0.0))
    alignment = float(signals.get("momentum_alignment", 0.0))
    bias = signals.get("positioning_bias", "neutral")
    strength = float(signals.get("bias_strength", 0.0))
    extreme = bool(signals.get("extreme_positioning", False))

    # Normalize OI ratio (cap at reasonable levels)
    normalized_oi_ratio = min(oi_ratio / 1000.0, 1.0)

    # Calculate positioning risk (high OI + extreme positioning = higher risk)
    positioning_risk = normalized_oi_ratio
    if extreme:
        positioning_risk *= 1.5
    positioning_risk = min(positioning_risk, 1.0)

    # Bias confidence adjustment
    bias_confidence = strength if bias != "neutral" else 0.0

    logger.info(f"Derivatives signals for {symbol}: OI_ratio={normalized_oi_ratio:.3f}, volatility={volatility:.3f}, alignment={alignment:.2f}, bias={bias}({bias_confidence:.2f}), extreme={extreme}")

    return {
        "oi_volume_ratio": normalized_oi_ratio,
        "positioning_volatility": volatility,
        "momentum_alignment": alignment,
        "positioning_bias": bias,
        "bias_strength": bias_confidence,
        "extreme_positioning": extreme,
        "positioning_risk": positioning_risk,
    }
