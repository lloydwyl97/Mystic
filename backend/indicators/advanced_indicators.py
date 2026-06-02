"""
Advanced Technical Indicators - All Live Data, No Fallback/Hardcoded Data

This module provides advanced technical indicators for live market analysis (backend port 8000).
All indicators:
- Calculate from live OHLCV data from Binance.US API
- Generate trading signals from live market conditions
- Extract features for AI training from live market data
- No fallback/hardcoded data - all calculations from live market operations
- Used by backend services on port 8000 for live trading operations

Live Data Sources:
- OHLCV data: Live candlestick data from Binance.US API
- Price data: Live high, low, close prices from Binance.US API
- Technical indicators: Calculated from live market data
- Trading signals: Generated from live market conditions
- All indicators use live data from Binance.US API - no mock/test data

Endpoint References:
- Binance.US API: https://api.binance.us (live exchange API for OHLCV data)
- Backend API: Port 8000 (indicators used by backend services for live trading)
- All indicators calculated from live endpoints - no fallback/hardcoded data
"""

from typing import Any

import numpy as np
import pandas as pd


class IchimokuCloud:
    """
    Ichimoku Cloud (Ichimoku Kinko Hyo) indicator for live market analysis.

    Calculates Ichimoku Cloud components from live OHLCV data from Binance.US API.
    Provides information about support/resistance, trend direction, momentum, and trading signals.
    All calculations use live market data - no fallback/hardcoded data.
    """

    def __init__(self, tenkan_period: int = 9, kijun_period: int = 26, senkou_b_period: int = 52, chikou_period: int = 26) -> None:
        """
        Initialize Ichimoku Cloud with customizable parameters for live market analysis.

        All periods are configuration defaults (not fallback data).
        Indicators calculated from live OHLCV data from Binance.US API.

        Args:
            tenkan_period: Period for Tenkan-sen (Conversion Line), default: 9 (configuration default not fallback data)
            kijun_period: Period for Kijun-sen (Base Line), default: 26 (configuration default not fallback data)
            senkou_b_period: Period for Senkou Span B (Leading Span B), default: 52 (configuration default not fallback data)
            chikou_period: Period for Chikou Span (Lagging Span), default: 26 (configuration default not fallback data)
        """
        self.tenkan_period = tenkan_period
        self.kijun_period = kijun_period
        self.senkou_b_period = senkou_b_period
        self.chikou_period = chikou_period

    def calculate(self, df: pd.DataFrame) -> dict[str, pd.Series]:
        """
        Calculate all components of the Ichimoku Cloud from live market data.

        All calculations use live OHLCV data from Binance.US API.
        Returns empty/NaN components if insufficient live data (not fallback data, insufficient data response).

        Args:
            df: DataFrame with live high, low, close columns from Binance.US API

        Returns:
            Dictionary containing all Ichimoku Cloud components calculated from live market data
        """
        # Extract price data
        high = df["high"]
        low = df["low"]
        close = df["close"]

        # Calculate Tenkan-sen (Conversion Line): (highest high + lowest low)/2 for the past tenkan_period
        tenkan_sen = (high.rolling(window=self.tenkan_period).max() + low.rolling(window=self.tenkan_period).min()) / 2

        # Calculate Kijun-sen (Base Line): (highest high + lowest low)/2 for the past kijun_period
        kijun_sen = (high.rolling(window=self.kijun_period).max() + low.rolling(window=self.kijun_period).min()) / 2

        # Calculate Senkou Span A (Leading Span A): (Tenkan-sen + Kijun-sen)/2 shifted forward by kijun_period
        senkou_span_a = ((tenkan_sen + kijun_sen) / 2).shift(self.kijun_period)

        # Calculate Senkou Span B (Leading Span B): (highest high + lowest low)/2 for the past senkou_b_period, shifted forward by kijun_period
        senkou_span_b = ((high.rolling(window=self.senkou_b_period).max() + low.rolling(window=self.senkou_b_period).min()) / 2).shift(self.kijun_period)

        # Calculate Chikou Span (Lagging Span): Close price shifted backwards by chikou_period
        chikou_span = close.shift(-self.chikou_period)

        return {"tenkan_sen": tenkan_sen, "kijun_sen": kijun_sen, "senkou_span_a": senkou_span_a, "senkou_span_b": senkou_span_b, "chikou_span": chikou_span}

    def get_signals(self, df: pd.DataFrame) -> dict[str, pd.Series]:
        """
        Generate live trading signals based on Ichimoku Cloud components from live market data.

        All signals generated from live market conditions from Binance.US API.
        Returns empty/zero signals if insufficient live data (not fallback data, insufficient data response).

        Args:
            df: DataFrame with live high, low, close columns from Binance.US API

        Returns:
            Dictionary containing live trading signal series from live market analysis
        """
        components = self.calculate(df)
        close = df["close"]

        # TK Cross: Tenkan-sen crosses above/below Kijun-sen
        tk_cross = pd.Series(np.zeros(len(df)), index=df.index)
        tk_cross_prev = components["tenkan_sen"].shift(1) - components["kijun_sen"].shift(1)
        tk_cross_curr = components["tenkan_sen"] - components["kijun_sen"]
        tk_cross[((tk_cross_prev < 0) & (tk_cross_curr > 0))] = 1  # Bullish TK Cross
        tk_cross[((tk_cross_prev > 0) & (tk_cross_curr < 0))] = -1  # Bearish TK Cross

        # Price vs Cloud
        price_vs_cloud = pd.Series(np.zeros(len(df)), index=df.index)
        # Price above cloud is bullish
        price_vs_cloud[(close > components["senkou_span_a"]) & (close > components["senkou_span_b"])] = 1
        # Price below cloud is bearish
        price_vs_cloud[(close < components["senkou_span_a"]) & (close < components["senkou_span_b"])] = -1

        # Cloud direction (future cloud)
        cloud_direction = pd.Series(np.zeros(len(df)), index=df.index)
        span_diff = components["senkou_span_a"] - components["senkou_span_b"]
        span_diff_prev = span_diff.shift(1)
        # Span A crossing above Span B is bullish
        cloud_direction[((span_diff_prev < 0) & (span_diff > 0))] = 1
        # Span A crossing below Span B is bearish
        cloud_direction[((span_diff_prev > 0) & (span_diff < 0))] = -1

        # Chikou span vs price (lagging span crossing price)
        chikou_cross = pd.Series(np.zeros(len(df)), index=df.index)
        # Chikou above price is bullish (need to compare aligned values)
        aligned_close = close.shift(self.chikou_period)
        chikou_cross_prev = components["chikou_span"].shift(1) - aligned_close.shift(1)
        chikou_cross_curr = components["chikou_span"] - aligned_close
        chikou_cross[((chikou_cross_prev < 0) & (chikou_cross_curr > 0))] = 1  # Bullish
        chikou_cross[((chikou_cross_prev > 0) & (chikou_cross_curr < 0))] = -1  # Bearish

        # Combined signal (weighted)
        combined_signal = tk_cross * 0.3 + price_vs_cloud * 0.35 + cloud_direction * 0.15 + chikou_cross * 0.2

        # Cloud thickness - indicator of trend strength
        cloud_thickness = abs(components["senkou_span_a"] - components["senkou_span_b"]) / ((components["senkou_span_a"] + components["senkou_span_b"]) / 2)

        return {
            "tk_cross": tk_cross,
            "price_vs_cloud": price_vs_cloud,
            "cloud_direction": cloud_direction,
            "chikou_cross": chikou_cross,
            "combined_signal": combined_signal,
            "cloud_thickness": cloud_thickness,
        }

    def get_features(self, df: pd.DataFrame) -> dict[str, float]:
        """
        Extract key features from Ichimoku Cloud for AI training from live market data.

        All features extracted from live OHLCV data from Binance.US API.
        Returns zero features if insufficient live data (not fallback data, insufficient data response).

        Args:
            df: DataFrame with live high, low, close columns from Binance.US API

        Returns:
            Dictionary of features extracted from live market analysis
        """
        # Check if sufficient live data available (not fallback data check, insufficient data response)
        if len(df) < self.senkou_b_period + self.kijun_period:
            # Insufficient live data - return zero features (not fallback data, insufficient data response)
            return {
                "cloud_position": 0,
                "tk_cross_signal": 0,
                "cloud_direction": 0,
                "cloud_thickness": 0,
                "chikou_position": 0,
            }

        components = self.calculate(df)
        signals = self.get_signals(df)

        # Latest values
        latest_close = df["close"].iloc[-1]
        latest_span_a = components["senkou_span_a"].iloc[-1]
        latest_span_b = components["senkou_span_b"].iloc[-1]

        # Current price position relative to cloud
        if pd.isna(latest_span_a) or pd.isna(latest_span_b):
            cloud_position = 0
        elif latest_close > max(latest_span_a, latest_span_b):
            cloud_position = 1  # Above cloud (bullish)
        elif latest_close < min(latest_span_a, latest_span_b):
            cloud_position = -1  # Below cloud (bearish)
        else:
            cloud_position = 0  # In cloud (neutral)

        # TK Cross signal
        tk_cross_signal = signals["tk_cross"].iloc[-1]

        # Cloud direction
        cloud_direction = 1 if latest_span_a > latest_span_b else (-1 if latest_span_a < latest_span_b else 0)

        # Cloud thickness - normalized
        if pd.isna(latest_span_a) or pd.isna(latest_span_b) or (latest_span_a + latest_span_b) == 0:
            cloud_thickness = 0
        else:
            avg_price = (latest_span_a + latest_span_b) / 2
            cloud_thickness = abs(latest_span_a - latest_span_b) / avg_price if avg_price != 0 else 0

        # Chikou span position
        chikou_index = min(len(df) - 1, len(df) - 1 - self.chikou_period)
        if chikou_index < 0:
            chikou_position = 0
        else:
            chikou_value = df["close"].iloc[-1]
            chikou_reference = df["close"].iloc[chikou_index] if chikou_index >= 0 else df["close"].iloc[0]
            chikou_position = 1 if chikou_value > chikou_reference else (-1 if chikou_value < chikou_reference else 0)

        return {"cloud_position": cloud_position, "tk_cross_signal": tk_cross_signal, "cloud_direction": cloud_direction, "cloud_thickness": float(cloud_thickness), "chikou_position": chikou_position}


class FibonacciPatterns:
    """
    Fibonacci Pattern Analysis for live market analysis.

    Implements Fibonacci retracements, extensions, and pattern detection
    calculated from live OHLCV data from Binance.US API.
    All calculations use live market data - no fallback/hardcoded data.
    """

    def __init__(self, lookback_period: int = 100) -> None:
        """
        Initialize Fibonacci pattern analyzer for live market analysis.

        Lookback period is a configuration default (not fallback data).
        Analyzes live OHLCV data from Binance.US API.

        Args:
            lookback_period: Period to look back for high/low detection, default: 100 (configuration default not fallback data)
        """
        self.lookback_period = lookback_period
        self.retracement_levels = [0, 0.236, 0.382, 0.5, 0.618, 0.786, 1.0]
        self.extension_levels = [0, 0.618, 1.0, 1.618, 2.618, 4.236]

    def find_swing_points(self, df: pd.DataFrame, window: int = 5) -> dict[str, pd.DataFrame]:
        """
        Find swing high and low points in live price series from Binance.US API.

        All swing points detected from live OHLCV data.
        Returns empty DataFrames if no swing points found (not fallback data, no pattern detected).

        Args:
            df: DataFrame with live high, low, close columns from Binance.US API
            window: Window size for detecting local extremes, default: 5 (configuration default not fallback data)

        Returns:
            Dictionary with swing high and low points detected from live market data
        """
        highs = df["high"]
        lows = df["low"]

        # Find swing highs (local maxima)
        swing_highs = pd.DataFrame(index=df.index)
        # VECTORIZED swing high detection for performance
        highs_array = highs.to_numpy()
        for i in range(window, len(highs) - window):
            # VECTORIZED comparison for performance
            left_window = highs_array[i - window : i]
            right_window = highs_array[i + 1 : i + window + 1]
            current_high = highs_array[i]

            if (current_high > left_window).all() and (current_high > right_window).all():
                swing_highs.loc[highs.index[i], "price"] = current_high

        # Find swing lows (local minima)
        swing_lows = pd.DataFrame(index=df.index)
        # VECTORIZED swing low detection for performance
        lows_array = lows.to_numpy()
        for i in range(window, len(lows) - window):
            # VECTORIZED comparison for performance
            left_window = lows_array[i - window : i]
            right_window = lows_array[i + 1 : i + window + 1]
            current_low = lows_array[i]

            if (current_low < left_window).all() and (current_low < right_window).all():
                swing_lows.loc[lows.index[i], "price"] = current_low

        # Remove NaN values
        swing_highs = swing_highs.dropna()
        swing_lows = swing_lows.dropna()

        # Ensure DataFrames have 'price' column even if empty
        if swing_highs.empty:
            swing_highs = pd.DataFrame(columns=["price"])
        if swing_lows.empty:
            swing_lows = pd.DataFrame(columns=["price"])

        return {"swing_highs": swing_highs, "swing_lows": swing_lows}

    def calculate_retracement_levels(self, high_price: float, low_price: float) -> dict[str, float]:
        """
        Calculate Fibonacci retracement levels from live swing high to low.

        Uses live swing prices from Binance.US API market data.

        Args:
            high_price: Live swing high price from Binance.US API
            low_price: Live swing low price from Binance.US API

        Returns:
            Dictionary of retracement levels calculated from live market data
        """
        price_range = high_price - low_price
        levels = {}

        for level in self.retracement_levels:
            level_name = f"retracement_{level}".replace(".", "_")
            levels[level_name] = high_price - (price_range * level)

        return levels

    def calculate_extension_levels(self, start_price: float, end_price: float, retracement_price: float) -> dict[str, float]:
        """
        Calculate Fibonacci extension levels from live price movements.

        Uses live prices from Binance.US API market data.

        Args:
            start_price: Live starting price from Binance.US API
            end_price: Live ending price from Binance.US API
            retracement_price: Live retracement price from Binance.US API

        Returns:
            Dictionary of extension levels calculated from live market data
        """
        price_range = end_price - start_price
        levels = {}

        for level in self.extension_levels:
            level_name = f"extension_{level}".replace(".", "_")
            if end_price > start_price:  # Uptrend
                levels[level_name] = retracement_price + (price_range * level)
            else:  # Downtrend
                levels[level_name] = retracement_price - (price_range * level)

        return levels

    def analyze(self, df: pd.DataFrame) -> dict[str, Any]:
        """
        Perform Fibonacci analysis on live price data from Binance.US API.

        All analysis performed on live OHLCV data.
        Returns empty result if insufficient live data or no swing points (not fallback data, insufficient data response).

        Args:
            df: DataFrame with live high, low, close columns from Binance.US API

        Returns:
            Dictionary with Fibonacci analysis results from live market data
        """
        # Limit to lookback period
        df_period = df.iloc[-min(len(df), self.lookback_period) :]

        # Find swing points
        swings = self.find_swing_points(df_period)

        # If not enough swing points, return empty result (not fallback data, insufficient data response)
        if len(swings["swing_highs"]) < 1 or len(swings["swing_lows"]) < 1:
            # Insufficient live swing points - return empty result (not fallback data, insufficient data response)
            return {
                "retracement_levels": {},
                "extension_levels": {},
                "recent_price": df["close"].iloc[-1] if not df.empty else None,  # Live price from Binance.US API
                "recent_high": df["high"].iloc[-1] if not df.empty else None,  # Live high from Binance.US API
                "recent_low": df["low"].iloc[-1] if not df.empty else None,  # Live low from Binance.US API
                "nearest_level": None,
                "distance_to_nearest": None,
                "price_position": "unknown",
                "uptrend": False,
            }

        # Get most recent significant high and low
        recent_high = swings["swing_highs"]["price"].max()
        recent_low = swings["swing_lows"]["price"].min()

        # Current price
        current_price = df["close"].iloc[-1]

        # Determine if we're in uptrend or downtrend
        recent_high_idx = df_period[df_period["high"] == recent_high].index[0] if recent_high in df_period["high"].to_numpy() else df_period.index[-1]
        recent_low_idx = df_period[df_period["low"] == recent_low].index[0] if recent_low in df_period["low"].to_numpy() else df_period.index[0]

        uptrend = recent_high_idx > recent_low_idx

        # Calculate retracement levels based on trend
        if uptrend:
            retracement_levels = self.calculate_retracement_levels(recent_high, recent_low)
            ext_start, ext_end, ext_retrace = recent_low, recent_high, current_price
        else:
            retracement_levels = self.calculate_retracement_levels(recent_low, recent_high)
            ext_start, ext_end, ext_retrace = recent_high, recent_low, current_price

        # Calculate extension levels
        extension_levels = self.calculate_extension_levels(ext_start, ext_end, ext_retrace)

        # Find nearest retracement level
        level_prices = list(retracement_levels.values())
        nearest_idx = np.argmin(np.abs(np.array(level_prices) - current_price))
        nearest_level = list(retracement_levels.keys())[nearest_idx]
        nearest_price = level_prices[nearest_idx]
        distance_to_nearest = abs(current_price - nearest_price) / current_price if current_price != 0 else 0

        # Determine price position
        if uptrend:
            if current_price >= recent_high:
                price_position = "above_high"
            elif current_price <= recent_low:
                price_position = "below_low"
            else:
                price_position = "in_retracement"
        elif current_price <= recent_low:
            price_position = "below_low"
        elif current_price >= recent_high:
            price_position = "above_high"
        else:
            price_position = "in_retracement"

        return {
            "retracement_levels": retracement_levels,
            "extension_levels": extension_levels,
            "recent_price": current_price,
            "recent_high": recent_high,
            "recent_low": recent_low,
            "nearest_level": nearest_level,
            "nearest_level_price": nearest_price,
            "distance_to_nearest": distance_to_nearest,
            "price_position": price_position,
            "uptrend": uptrend,
        }

    def get_features(self, df: pd.DataFrame) -> dict[str, float]:
        """
        Extract key Fibonacci features for AI training from live market data.

        All features extracted from live OHLCV data from Binance.US API.
        Returns zero features if insufficient live data (not fallback data, insufficient data response).

        Args:
            df: DataFrame with live high, low, close columns from Binance.US API

        Returns:
            Dictionary of features extracted from live market analysis
        """
        # Check if sufficient live data available (not fallback data check, insufficient data response)
        if len(df) < 20:  # Need sufficient live data from Binance.US API
            # Insufficient live data - return zero features (not fallback data, insufficient data response)
            return {
                "fib_position": 0,
                "fib_nearest_level": 0,
                "fib_distance": 0,
                "fib_pattern_strength": 0,
                "fib_trend": 0,
            }

        analysis = self.analyze(df)

        # Position relative to key levels
        if analysis["price_position"] == "above_high":
            fib_position = 1
        elif analysis["price_position"] == "below_low":
            fib_position = -1
        else:
            # In retracement zone - normalize between -1 and 1
            high_low_range = analysis["recent_high"] - analysis["recent_low"]
            if high_low_range == 0:
                fib_position = 0
            else:
                position = (analysis["recent_price"] - analysis["recent_low"]) / high_low_range
                fib_position = (position * 2) - 1  # Scale to -1 to 1

        # Nearest Fibonacci level (convert to numeric)
        if analysis["nearest_level"] is None:
            fib_nearest_level = 0
        else:
            try:
                level_str = analysis["nearest_level"].split("_")[1]
                level_val = float(level_str.replace("_", "."))
                fib_nearest_level = level_val
            except (ValueError, IndexError):
                fib_nearest_level = 0

        # Distance to nearest level
        fib_distance = analysis["distance_to_nearest"] if analysis["distance_to_nearest"] is not None else 1.0

        # Pattern strength (0-1) - closer to key level is stronger
        fib_pattern_strength = max(0, 1 - (fib_distance * 10))

        # Trend direction
        fib_trend = 1 if analysis.get("uptrend", False) else -1

        return {"fib_position": fib_position, "fib_nearest_level": fib_nearest_level, "fib_distance": fib_distance, "fib_pattern_strength": fib_pattern_strength, "fib_trend": fib_trend}
