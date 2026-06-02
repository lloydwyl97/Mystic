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
) -> dict[str, Any] | list[Any]:
    try:
        resp = httpx.get(url, params=params or {}, headers=headers or {}, timeout=10.0)
        resp.raise_for_status()
        data = resp.json()
        # Accept both dict and list responses (Binance liquidations returns list)
        if isinstance(data, (dict, list)):
            return data
        else:
            logger.warning(f"Unexpected JSON response type: {type(data)}")
            return {}
    except httpx.HTTPStatusError as e:
        # Handle 451 "Unavailable For Legal Reasons" (Binance Futures blocked in some regions)
        if e.response.status_code == 451:
            logger.debug(f"API blocked in your region (HTTP 451): {url}")
            return {}
        # Log other HTTP errors as warnings, not exceptions (reduce noise)
        logger.warning(f"HTTP {e.response.status_code} error for {url}")
        return {}
    except httpx.RequestError as e:
        logger.warning(f"HTTP request failed for {url}: {e}")
        return {}
    except ValueError as e:
        logger.warning(f"JSON decode failed for {url}: {e}")
        return {}


def fetch_binance_liquidation_data(symbol: str) -> dict[str, Any]:
    """Fetch recent liquidation data from Binance Futures API for a symbol.

    Returns aggregated liquidation heatmap data showing price levels with
    concentrated liquidation activity.

    Note: Binance US does not support futures trading - this will return empty
    for Binance US users (HTTP 451).
    """
    # Check if using Binance US (no futures support)
    exchange_id = os.getenv("EXCHANGE_ID", "binanceus")
    if exchange_id == "binanceus":
        logger.debug("Liquidation data not available on Binance US (futures not supported)")
        return {}

    api_key = os.getenv("BINANCE_API_KEY", "")
    if not api_key:
        logger.debug("BINANCE_API_KEY not set; cannot fetch liquidation data")
        return {}

    # Normalize symbol (remove USDT and add USDT back for futures)
    normalized = symbol.upper().replace("/", "").replace("-", "")
    if not normalized.endswith("USDT"):
        normalized += "USDT"

    url = "https://fapi.binance.com/fapi/v1/forceOrders"
    params = {
        "symbol": normalized,
        "limit": 100,  # Get recent liquidations
    }
    headers = {"X-MBX-APIKEY": api_key, "User-Agent": "mystic-trading/1.0"}

    data = _http_get(url, params=params, headers=headers)

    if not data or not isinstance(data, list):
        return {}

    # Aggregate liquidations by price levels to create heatmap
    long_liqs = []
    short_liqs = []
    total_long_volume = 0.0
    total_short_volume = 0.0

    for liq in data:
        try:
            price = float(liq.get("avgPrice", 0))
            qty = float(liq.get("origQty", 0))
            side = liq.get("side", "")

            if price <= 0 or qty <= 0:
                continue

            if side == "BUY":  # Long position liquidation
                long_liqs.append({"price": price, "volume": qty})
                total_long_volume += qty
            elif side == "SELL":  # Short position liquidation
                short_liqs.append({"price": price, "volume": qty})
                total_short_volume += qty
        except (ValueError, TypeError, KeyError):
            continue

    # Calculate concentration metrics
    long_concentration = _calculate_price_concentration(long_liqs)
    short_concentration = _calculate_price_concentration(short_liqs)

    return {
        "symbol": symbol,
        "long_liquidations": len(long_liqs),
        "short_liquidations": len(short_liqs),
        "total_long_volume": total_long_volume,
        "total_short_volume": total_short_volume,
        "long_concentration_zones": long_concentration,
        "short_concentration_zones": short_concentration,
        "liquidation_imbalance": (total_long_volume - total_short_volume) / max(total_long_volume + total_short_volume, 1.0),
    }


def _calculate_price_concentration(liquidations: list[dict[str, float]], bins: int = 10) -> list[dict[str, float]]:
    """Calculate price concentration zones for heatmap visualization."""
    if not liquidations:
        return []

    # Sort by price
    sorted_liqs = sorted(liquidations, key=lambda x: x["price"])

    # Create price bins
    bins = min(bins, len(sorted_liqs))

    min_price = sorted_liqs[0]["price"]
    max_price = sorted_liqs[-1]["price"]
    price_range = max_price - min_price

    if price_range <= 0:
        return [{"price": min_price, "concentration": sum(liq["volume"] for liq in sorted_liqs)}]

    bin_size = price_range / bins
    concentration_zones = []

    for i in range(bins):
        bin_min = min_price + (i * bin_size)
        bin_max = min_price + ((i + 1) * bin_size)

        bin_volume = sum(liq["volume"] for liq in sorted_liqs if bin_min <= liq["price"] < bin_max)

        if bin_volume > 0:
            concentration_zones.append(
                {
                    "price_min": bin_min,
                    "price_max": bin_max,
                    "concentration": bin_volume,
                }
            )

    return concentration_zones


def liquidation_signal_check(symbol: str) -> dict[str, Any]:
    """Get liquidation heatmap signals for decision making."""
    data = fetch_binance_liquidation_data(symbol)

    if not data:
        return {
            "liquidation_risk": 0.0,
            "long_liq_pressure": 0.0,
            "short_liq_pressure": 0.0,
            "price_support_level": 0.0,
            "price_resistance_level": 0.0,
        }

    # Calculate risk metrics
    long_liq_count = data.get("long_liquidations", 0)
    short_liq_count = data.get("short_liquidations", 0)
    total_liqs = long_liq_count + short_liq_count

    if total_liqs == 0:
        return {
            "liquidation_risk": 0.0,
            "long_liq_pressure": 0.0,
            "short_liq_pressure": 0.0,
            "price_support_level": 0.0,
            "price_resistance_level": 0.0,
        }

    # Risk increases with more liquidations
    liquidation_risk = min(total_liqs / 50.0, 1.0)  # Normalize to 0-1

    # Pressure metrics (normalized by total)
    long_liq_pressure = long_liq_count / total_liqs
    short_liq_pressure = short_liq_count / total_liqs

    # Find key price levels from concentration zones
    long_zones = data.get("long_concentration_zones", [])
    short_zones = data.get("short_concentration_zones", [])

    # Support level: highest concentration of long liquidations (potential support)
    support_level = 0.0
    if long_zones:
        support_zone = max(long_zones, key=lambda x: x["concentration"])
        support_level = (support_zone["price_min"] + support_zone["price_max"]) / 2

    # Resistance level: highest concentration of short liquidations (potential resistance)
    resistance_level = 0.0
    if short_zones:
        resistance_zone = max(short_zones, key=lambda x: x["concentration"])
        resistance_level = (resistance_zone["price_min"] + resistance_zone["price_max"]) / 2

    logger.info(
        f"Liquidation signals for {symbol}: risk={liquidation_risk:.3f}, "
        f"long_pressure={long_liq_pressure:.3f}, short_pressure={short_liq_pressure:.3f}, "
        f"support={support_level:.4f}, resistance={resistance_level:.4f}"
    )

    return {
        "liquidation_risk": liquidation_risk,
        "long_liq_pressure": long_liq_pressure,
        "short_liq_pressure": short_liq_pressure,
        "price_support_level": support_level,
        "price_resistance_level": resistance_level,
    }
