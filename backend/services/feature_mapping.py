"""
Feature Mapping for 124-Feature AI Model
Provides a consistent mapping between feature names and positions in the 124-feature vector
"""

import logging
from typing import Any

logger = logging.getLogger(__name__)

# Complete mapping of all 124 features by index (1-based)
FEATURE_MAPPING = {
    # Basic price features (10)
    "price": 1,
    "high": 2,
    "low": 3,
    "open": 4,
    "volume": 5,
    "change_24h": 6,
    "change_7d": 7,
    "change_30d": 8,
    "price_range": 9,
    "typical_price": 10,
    # Technical indicators (24)
    "ma_5": 11,
    "ma_10": 12,
    "ma_20": 13,
    "ma_50": 14,
    "ma_100": 15,
    "ma_200": 16,
    "ema_12": 17,
    "ema_26": 18,
    "ema_50": 19,
    "rsi": 20,
    "rsi_14": 21,
    "stoch_k": 22,
    "stoch_d": 23,
    "williams_r": 24,
    "cci": 25,
    "macd": 26,
    "macd_signal": 27,
    "macd_histogram": 28,
    "bb_upper": 29,
    "bb_middle": 30,
    "bb_lower": 31,
    "bb_position": 32,
    "bb_width": 33,
    "obv": 34,
    "ad_line": 35,
    "cmf": 36,
    "mfi": 37,
    # Volatility indicators (10)
    "volatility": 38,
    "atr": 39,
    "natr": 40,
    "keltner_upper": 41,
    "keltner_lower": 42,
    "donchian_upper": 43,
    "donchian_lower": 44,
    "parabolic_sar": 45,
    "volatility_ratio": 46,
    "price_volatility": 47,
    # Momentum indicators (15)
    "roc": 48,
    "momentum": 49,
    "ppo": 50,
    "trix": 51,
    "ultimate_oscillator": 52,
    "awesome_oscillator": 53,
    "balance_of_power": 54,
    "ease_of_movement": 55,
    "mass_index": 56,
    "vortex_vi_plus": 57,
    "vortex_vi_minus": 58,
    "kst": 59,
    "tsi": 60,
    "aroon_up": 61,
    "aroon_down": 62,
    # Trend indicators (10)
    "adx": 63,
    "di_plus": 64,
    "di_minus": 65,
    "aroon_oscillator": 66,
    "ichimoku_tenkan": 67,
    "ichimoku_kijun": 68,
    "ichimoku_senkou_a": 69,
    "ichimoku_senkou_b": 70,
    "psar": 71,
    "trend_strength": 72,
    # Volume profile (8)
    "volume_ma_5": 73,
    "volume_ma_10": 74,
    "volume_ma_20": 75,
    "volume_ratio": 76,
    "volume_price_trend": 77,
    "negative_volume_index": 78,
    "positive_volume_index": 79,
    "volume_weighted_price": 80,
    # Market sentiment (10)
    "fear_greed_index": 81,
    "social_sentiment": 82,
    "news_sentiment": 83,
    "put_call_ratio": 84,
    "vix": 85,
    "market_cap": 86,
    "supply": 87,
    "circulating_supply": 88,
    "max_supply": 89,
    "market_dominance": 90,
    # Time-based features (10)
    "hour": 91,
    "day_of_week": 92,
    "day_of_month": 93,
    "month": 94,
    "iso_weekday": 95,
    "day_of_year": 96,
    "hour_12h": 97,
    "minute": 98,
    "second": 99,
    "seconds_since_midnight": 100,
    # Advanced technical analysis (8)
    "fibonacci_retracement_23.6": 101,
    "fibonacci_retracement_38.2": 102,
    "fibonacci_retracement_61.8": 103,
    "pivot_point": 104,
    "resistance_1": 105,
    "resistance_2": 106,
    "support_1": 107,
    "support_2": 108,
    # Advanced volume analysis (8)
    "volume_profile_poc": 109,
    "volume_profile_vah": 110,
    "volume_profile_val": 111,
    "vwap": 112,
    "twap": 113,
    "volume_imbalance": 114,
    "volume_delta": 115,
    "order_flow": 116,
    # Advanced market microstructure (8)
    "bid_ask_spread": 117,
    "order_book_imbalance": 118,
    "market_depth": 119,
    "liquidity_score": 120,
    "price_impact": 121,
    "market_efficiency": 122,
    "volatility_smile": 123,
    "price_skewness": 124,
}

# Mapping of sentiment feature names to 124-feature positions
SENTIMENT_FEATURE_MAPPING = {
    "sentiment_score": "fear_greed_index",  # Position 81
    "sentiment_positive_ratio": "social_sentiment",  # Position 82
    "sentiment_negative_ratio": "news_sentiment",  # Position 83
    "sentiment_confidence": "put_call_ratio",  # Position 84
    "sentiment_post_volume": "vix",  # Position 85
}


def get_feature_index(feature_name: str) -> int:
    """Get the 1-based index of a feature in the 124-feature vector"""
    return FEATURE_MAPPING.get(feature_name, -1)


def get_feature_name(index: int) -> str:
    """Get the name of a feature from its 1-based index"""
    for name, idx in FEATURE_MAPPING.items():
        if idx == index:
            return name
    return f"unknown_feature_{index}"


def map_sentiment_feature(sentiment_feature: str) -> str:
    """Map a sentiment feature name to its corresponding 124-feature name"""
    return SENTIMENT_FEATURE_MAPPING.get(sentiment_feature, sentiment_feature)


def create_empty_feature_vector() -> list[float]:
    """Create an empty 124-feature vector with zeros"""
    return [0.0] * 124


def update_feature_vector(feature_vector: list[float], feature_name: str, value: float) -> list[float]:
    """Update a feature vector with a named feature value"""
    index = get_feature_index(feature_name)
    if index > 0 and index <= len(feature_vector):
        feature_vector[index - 1] = value  # Convert to 0-based index
    return feature_vector


def dict_to_feature_vector(features_dict: dict[str, Any]) -> list[float]:
    """Convert a dictionary of features to a 124-feature vector"""
    feature_vector = create_empty_feature_vector()

    for name, value in features_dict.items():
        try:
            # Convert sentiment features to their 124-feature equivalents
            mapped_name = map_sentiment_feature(name)
            # Update the feature vector
            feature_vector = update_feature_vector(feature_vector, mapped_name, float(value))
        except (ValueError, TypeError):
            pass

    return feature_vector


# ============================================================================
# SIGNAL QUALITY METADATA
# Added 2025-10-29 to track which signals have real live data vs fallbacks
# ============================================================================

FEATURE_QUALITY: dict[str, str] = {
    # Features 1-10: Basic Price Features
    "price": "LIVE",  # Real-time from exchange
    "high": "LIVE",  # Real-time from exchange
    "low": "LIVE",  # Real-time from exchange
    "open": "LIVE",  # Real-time from exchange
    "volume": "LIVE",  # Real-time from exchange
    "change_24h": "LIVE",  # Calculated from exchange data
    "change_7d": "LIVE",  # Calculated from exchange data
    "change_30d": "LIVE",  # Calculated from exchange data
    "price_range": "CALCULATED",  # Derived from high/low
    "typical_price": "CALCULATED",  # Derived from high/low/close
    # Features 11-37: Technical Indicators
    "ma_5": "CALCULATED",
    "ma_10": "CALCULATED",
    "ma_20": "CALCULATED",
    "ma_50": "CALCULATED",
    "ma_100": "CALCULATED",
    "ma_200": "CALCULATED",
    "ema_12": "CALCULATED",
    "ema_26": "CALCULATED",
    "ema_50": "CALCULATED",
    "rsi": "CALCULATED",
    "rsi_14": "CALCULATED",
    "stoch_k": "CALCULATED",
    "stoch_d": "CALCULATED",
    "williams_r": "CALCULATED",
    "cci": "CALCULATED",
    "macd": "CALCULATED",
    "macd_signal": "CALCULATED",
    "macd_histogram": "CALCULATED",
    "bb_upper": "CALCULATED",
    "bb_middle": "CALCULATED",
    "bb_lower": "CALCULATED",
    "bb_position": "CALCULATED",
    "bb_width": "CALCULATED",
    "obv": "CALCULATED",
    "ad_line": "CALCULATED",
    "cmf": "CALCULATED",
    "mfi": "CALCULATED",
    # Features 38-47: Volatility Indicators
    "volatility": "CALCULATED",
    "atr": "CALCULATED",
    "natr": "CALCULATED",
    "keltner_upper": "CALCULATED",
    "keltner_lower": "CALCULATED",
    "donchian_upper": "CALCULATED",
    "donchian_lower": "CALCULATED",
    "parabolic_sar": "CALCULATED",
    "volatility_ratio": "CALCULATED",
    "price_volatility": "CALCULATED",
    # Features 48-62: Momentum Indicators
    "roc": "CALCULATED",
    "momentum": "CALCULATED",
    "ppo": "CALCULATED",
    "trix": "CALCULATED",
    "ultimate_oscillator": "CALCULATED",
    "awesome_oscillator": "CALCULATED",
    "balance_of_power": "CALCULATED",
    "ease_of_movement": "CALCULATED",
    "mass_index": "CALCULATED",
    "vortex_vi_plus": "CALCULATED",
    "vortex_vi_minus": "CALCULATED",
    "kst": "CALCULATED",
    "tsi": "CALCULATED",
    "aroon_up": "CALCULATED",
    "aroon_down": "CALCULATED",
    # Features 63-72: Trend Indicators
    "adx": "CALCULATED",
    "di_plus": "CALCULATED",
    "di_minus": "CALCULATED",
    "aroon_oscillator": "CALCULATED",
    "ichimoku_tenkan": "CALCULATED",
    "ichimoku_kijun": "CALCULATED",
    "ichimoku_senkou_a": "CALCULATED",
    "ichimoku_senkou_b": "CALCULATED",
    "psar": "CALCULATED",
    "trend_strength": "CALCULATED",
    # Features 73-80: Volume Profile
    "volume_ma_5": "CALCULATED",
    "volume_ma_10": "CALCULATED",
    "volume_ma_20": "CALCULATED",
    "volume_ratio": "CALCULATED",
    "volume_price_trend": "CALCULATED",
    "negative_volume_index": "CALCULATED",
    "positive_volume_index": "CALCULATED",
    "volume_weighted_price": "CALCULATED",
    # Features 81-90: Market Sentiment - UPGRADED: Now using live APIs
    "fear_greed_index": "LIVE",  # ✅ UPGRADED: alternative.me API (free, no key)
    "social_sentiment": "LIVE",  # ✅ UPGRADED: Live sentiment from market_sentiment_agent
    "news_sentiment": "LIVE",  # ✅ UPGRADED: Live news via real_time_news_analyzer
    "put_call_ratio": "UNSUPPORTED_FOR_SPOT",  # N/A for crypto spot (options data)
    "vix": "CALCULATED",  # ✅ UPGRADED: Derived from BTC volatility
    "market_cap": "LIVE",  # ✅ UPGRADED: From CoinGecko/Binance
    "supply": "LIVE",  # ✅ UPGRADED: From exchange info
    "circulating_supply": "LIVE",  # ✅ UPGRADED: From exchange info
    "max_supply": "LIVE",  # ✅ UPGRADED: From exchange info
    "market_dominance": "CALCULATED",  # ✅ UPGRADED: BTC dominance calculated
    # Features 91-100: Time-Based Features
    "hour": "LIVE",
    "day_of_week": "LIVE",
    "day_of_month": "LIVE",
    "month": "LIVE",
    "iso_weekday": "LIVE",
    "day_of_year": "LIVE",
    "hour_12h": "LIVE",
    "minute": "LIVE",
    "second": "LIVE",
    "seconds_since_midnight": "LIVE",
    # Features 101-108: Advanced Technical Analysis
    "fibonacci_retracement_23.6": "CALCULATED",
    "fibonacci_retracement_38.2": "CALCULATED",
    "fibonacci_retracement_61.8": "CALCULATED",
    "pivot_point": "CALCULATED",
    "resistance_1": "CALCULATED",
    "resistance_2": "CALCULATED",
    "support_1": "CALCULATED",
    "support_2": "CALCULATED",
    # Features 109-116: Advanced Volume Analysis
    "volume_profile_poc": "CALCULATED",  # ✅ Binance historical data
    "volume_profile_vah": "CALCULATED",  # ✅ Binance historical data
    "volume_profile_val": "CALCULATED",  # ✅ Binance historical data
    "vwap": "CALCULATED",
    "twap": "CALCULATED",
    "volume_imbalance": "CALCULATED_PROXY",
    "volume_delta": "CALCULATED_PROXY",
    "order_flow": "CALCULATED_PROXY",
    # Features 117-124: Market Microstructure
    "bid_ask_spread": "LIVE",  # ✅ Binance WebSocket order book
    "order_book_imbalance": "LIVE",  # ✅ Binance WebSocket order book
    "market_depth": "LIVE",  # ✅ Binance WebSocket order book
    "liquidity_score": "CALCULATED",  # ✅ Derived from order book
    "price_impact": "CALCULATED",  # ✅ Derived from order book
    "market_efficiency": "CALCULATED",  # ✅ Derived from order book
    "volatility_smile": "UNSUPPORTED_FOR_SPOT",
    "price_skewness": "CALCULATED",
}

QUALITY_CATEGORIES: dict[str, str] = {
    "LIVE": "Real-time data from exchange or system",
    "CALCULATED": "Calculated from live price/volume data",
    "FALLBACK": "Uses default/fallback value (zero or price)",
    "MISSING_API": "Requires external API not yet integrated",
    "MISSING_DATA": "Requires data source not yet fetched",
}


def get_signal_quality_summary() -> dict[str, int]:
    """
    Get summary of signal data quality across all 124 features.

    Returns:
        Dict mapping quality category to count of features

    Example:
        >>> get_signal_quality_summary()
        {'LIVE': 18, 'CALCULATED': 62, 'FALLBACK': 20, 'MISSING_API': 3, 'MISSING_DATA': 21}
    """
    counts: dict[str, int] = {}
    for quality in FEATURE_QUALITY.values():
        counts[quality] = counts.get(quality, 0) + 1
    return counts


def get_real_data_percentage() -> float:
    """
    Calculate percentage of features with real live data.

    Returns:
        Percentage of features that have LIVE or CALCULATED data (0.0 to 1.0)
    """
    summary = get_signal_quality_summary()
    real_data_count = summary.get("LIVE", 0) + summary.get("CALCULATED", 0)
    return real_data_count / 124.0


def get_feature_quality(feature_name: str) -> str:
    """
    Get the data quality category for a specific feature.

    Args:
        feature_name: Name of the feature

    Returns:
        Quality category string (LIVE, CALCULATED, FALLBACK, MISSING_API, MISSING_DATA)
    """
    return FEATURE_QUALITY.get(feature_name, "UNKNOWN")
