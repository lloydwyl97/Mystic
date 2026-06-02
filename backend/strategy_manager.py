"""
Enhanced Strategy Manager
Multi-timeframe analysis and market regime adaptation for trading strategies
"""

import logging
from datetime import datetime, timezone
from typing import Any, ClassVar

import numpy as np
import pandas as pd

from backend.services.binance_rest_client import BinanceREST, BinanceWeightLimiter

logger = logging.getLogger(__name__)


class MarketRegime:
    """Market regime classification"""

    BULL = "bull"
    BEAR = "bear"
    SIDEWAYS = "sideways"
    HIGH_VOLATILITY = "high_volatility"
    LOW_VOLATILITY = "low_volatility"

    @staticmethod
    def classify_regime(df: pd.DataFrame, symbol: str) -> str:  # noqa: ARG004
        """Classify current market regime based on price action"""
        try:
            if df.empty or len(df) < 50:
                return MarketRegime.SIDEWAYS

            close = df["close"].to_numpy()

            # Calculate trend strength (50-day vs 200-day MA)
            ma50 = pd.Series(close).rolling(50).mean().iloc[-1]
            ma200 = pd.Series(close).rolling(200).mean().iloc[-1]

            trend_ratio = 0 if ma200 == 0 else (ma50 - ma200) / ma200

            # Calculate volatility (20-day standard deviation of returns)
            returns = np.diff(np.log(close))
            volatility = np.std(returns[-20:]) * np.sqrt(252)  # Annualized

            # Classify regime
            if volatility > 0.8:  # Very high volatility
                return MarketRegime.HIGH_VOLATILITY
            elif volatility < 0.2:  # Very low volatility
                return MarketRegime.LOW_VOLATILITY
            elif trend_ratio > 0.05:  # Strong uptrend
                return MarketRegime.BULL
            elif trend_ratio < -0.05:  # Strong downtrend
                return MarketRegime.BEAR
            else:
                return MarketRegime.SIDEWAYS

        except Exception as ex:
            logger.debug("Regime detection failed: %s", ex)
            return MarketRegime.SIDEWAYS


class MultiTimeframeAnalyzer:
    """Multi-timeframe analysis for enhanced signal confirmation"""

    TIMEFRAMES: ClassVar[list[str]] = ["5m", "15m", "1h", "4h", "1d"]

    def __init__(self):
        # Initialize with proper limiter
        self.limiter = None
        self.client = None

    async def _ensure_client(self):
        """Ensure client is initialized with limiter"""
        if self.client is None:
            if self.limiter is None:
                self.limiter = await BinanceWeightLimiter.create()
            self.client = BinanceREST(self.limiter)

    async def analyze_multi_tf(self, symbol: str, base_timeframe: str = "1h") -> dict[str, Any]:
        """Analyze multiple timeframes for signal confirmation"""
        try:
            await self._ensure_client()
            results = {}

            # Analyze each timeframe
            for tf in self.TIMEFRAMES:
                try:
                    # Get data for this timeframe
                    klines = await self.client.get_klines(symbol, tf, limit=100)
                    if not klines:
                        continue

                    # Convert to DataFrame
                    df = pd.DataFrame(
                        klines,
                        columns=[
                            "timestamp",
                            "open",
                            "high",
                            "low",
                            "close",
                            "volume",
                            "close_time",
                            "quote_asset_volume",
                            "number_of_trades",
                            "taker_buy_base_asset_volume",
                            "taker_buy_quote_asset_volume",
                            "ignore",
                        ],
                    )
                    df = df[["timestamp", "open", "high", "low", "close", "volume"]].copy()
                    df = df.astype({"open": float, "high": float, "low": float, "close": float, "volume": float})

                    # Calculate technical indicators
                    df = self._add_technical_indicators(df)

                    # Determine trend and strength
                    trend, strength = self._calculate_trend_strength(df)

                    results[tf] = {
                        "trend": trend,
                        "strength": strength,
                        "last_price": df["close"].iloc[-1],
                        "volume": df["volume"].iloc[-1],
                        "indicators": {
                            "rsi": df["rsi"].iloc[-1] if "rsi" in df.columns else None,
                            "macd": df["macd"].iloc[-1] if "macd" in df.columns else None,
                            "bb_position": df["bb_position"].iloc[-1] if "bb_position" in df.columns else None,
                        },
                    }

                except Exception as ex:
                    logger.debug("Multi-timeframe %s failed: %s", tf, ex)
                    continue

            # Generate multi-timeframe signal
            signal = self._generate_multi_tf_signal(results, base_timeframe)

            return {"timeframes": results, "multi_tf_signal": signal, "confidence": self._calculate_signal_confidence(results, signal), "timestamp": datetime.now(timezone.utc).isoformat()}

        except Exception as e:
            logger.exception(f"Error in multi-timeframe analysis for {symbol}: {e}")
            return {"error": str(e)}

    def _add_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add technical indicators to DataFrame"""
        try:
            close = df["close"]

            # RSI
            delta = close.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            df["rsi"] = 100 - (100 / (1 + rs))

            # MACD
            exp1 = close.ewm(span=12, adjust=False).mean()
            exp2 = close.ewm(span=26, adjust=False).mean()
            df["macd"] = exp1 - exp2
            df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()

            # Bollinger Bands
            sma = close.rolling(window=20).mean()
            std = close.rolling(window=20).std()
            df["bb_upper"] = sma + (std * 2)
            df["bb_lower"] = sma - (std * 2)
            df["bb_position"] = (close - df["bb_lower"]) / (df["bb_upper"] - df["bb_lower"])

        except Exception as ex:
            logger.debug("_add_technical_indicators failed: %s", ex)
            return df
        else:
            return df

    def _calculate_trend_strength(self, df: pd.DataFrame) -> tuple[str, float]:
        """Calculate trend direction and strength"""
        try:
            if len(df) < 20:
                return "neutral", 0.0

            close = df["close"].to_numpy()
            recent_prices = close[-20:]

            # Linear regression slope
            x = np.arange(len(recent_prices))
            slope, _ = np.polyfit(x, recent_prices, 1)

            # Normalize slope by average price
            avg_price = np.mean(recent_prices)
            normalized_slope = slope / avg_price if avg_price != 0 else 0

            # Determine trend direction
            if normalized_slope > 0.001:
                trend = "bullish"
            elif normalized_slope < -0.001:
                trend = "bearish"
            else:
                trend = "neutral"

            # Calculate strength (absolute slope normalized)
            strength = min(abs(normalized_slope) * 1000, 1.0)

        except Exception as ex:
            logger.debug("_calculate_trend_strength failed: %s", ex)
            return "neutral", 0.0
        else:
            return trend, strength

    def _generate_multi_tf_signal(self, timeframe_results: dict[str, Any], base_timeframe: str) -> str:
        """Generate multi-timeframe signal based on trend alignment"""
        try:
            if not timeframe_results:
                return "HOLD"

            # Weight timeframes by importance (shorter timeframes have higher weight)
            weights = {"5m": 0.1, "15m": 0.15, "1h": 0.25, "4h": 0.3, "1d": 0.2}

            bullish_score = 0
            bearish_score = 0
            total_weight = 0

            for tf, data in timeframe_results.items():
                weight = weights.get(tf, 0.1)
                trend = data.get("trend", "neutral")
                strength = data.get("strength", 0)

                if trend == "bullish":
                    bullish_score += weight * strength
                elif trend == "bearish":
                    bearish_score += weight * strength

                total_weight += weight

            # Normalize scores
            if total_weight > 0:
                bullish_score /= total_weight
                bearish_score /= total_weight

            # Generate signal
            if bullish_score > bearish_score and bullish_score > 0.3:
                return "BUY"
            elif bearish_score > bullish_score and bearish_score > 0.3:
                return "SELL"
            else:
                return "HOLD"

        except Exception as ex:
            logger.debug("_generate_multi_tf_signal failed: %s", ex)
            return "HOLD"

    def _calculate_signal_confidence(self, timeframe_results: dict[str, Any], signal: str) -> float:
        """Calculate confidence level for the multi-timeframe signal"""
        try:
            if signal == "HOLD" or not timeframe_results:
                return 0.5

            confirming_timeframes = 0
            total_timeframes = len(timeframe_results)

            expected_trend = "bullish" if signal == "BUY" else "bearish"

            for tf_data in timeframe_results.values():
                if tf_data.get("trend") == expected_trend:
                    confirming_timeframes += 1

            # Confidence based on agreement across timeframes
            agreement_ratio = confirming_timeframes / total_timeframes

            # Base confidence on agreement level
            if agreement_ratio >= 0.8:
                confidence = 0.9
            elif agreement_ratio >= 0.6:
                confidence = 0.7
            elif agreement_ratio >= 0.4:
                confidence = 0.6
            else:
                confidence = 0.4

        except Exception as ex:
            logger.debug("_calculate_signal_confidence failed: %s", ex)
            return 0.5
        else:
            return confidence


class AdaptiveStrategyManager:
    """Adaptive strategy manager that selects strategies based on market regime"""

    STRATEGY_MAPPINGS: ClassVar[dict[str, dict[str, Any]]] = {
        MarketRegime.BULL: {"primary": "momentum_following", "secondary": "breakout_trading", "risk_multiplier": 1.2, "timeframe_preference": "1h"},
        MarketRegime.BEAR: {"primary": "mean_reversion", "secondary": "short_bias", "risk_multiplier": 0.8, "timeframe_preference": "4h"},
        MarketRegime.SIDEWAYS: {"primary": "range_trading", "secondary": "daying", "risk_multiplier": 0.9, "timeframe_preference": "15m"},
        MarketRegime.HIGH_VOLATILITY: {
            "primary": "volatility_breakout",
            "secondary": "straddle_options",  # Would be implemented if options available
            "risk_multiplier": 0.7,
            "timeframe_preference": "5m",
        },
        MarketRegime.LOW_VOLATILITY: {"primary": "trend_following", "secondary": "carry_trading", "risk_multiplier": 1.1, "timeframe_preference": "1d"},
    }

    def __init__(self):
        self.multi_tf_analyzer = MultiTimeframeAnalyzer()
        self.regime_cache = {}
        self.strategy_performance = {}

    async def get_adaptive_strategy(self, symbol: str) -> dict[str, Any]:
        """Get adaptive strategy recommendation based on current market conditions"""
        try:
            # Get market regime
            regime = await self._get_market_regime(symbol)

            # Get multi-timeframe analysis
            mtf_analysis = await self.multi_tf_analyzer.analyze_multi_tf(symbol)

            # Select strategy based on regime
            strategy_config = self.STRATEGY_MAPPINGS.get(regime, self.STRATEGY_MAPPINGS[MarketRegime.SIDEWAYS])

            # Adjust based on multi-timeframe signals
            adjusted_config = self._adjust_for_multi_tf(strategy_config, mtf_analysis)

            # Get performance metrics for this strategy
            performance = self._get_strategy_performance(adjusted_config["primary"])

            return {
                "symbol": symbol,
                "market_regime": regime,
                "recommended_strategy": adjusted_config["primary"],
                "backup_strategy": adjusted_config["secondary"],
                "risk_multiplier": adjusted_config["risk_multiplier"],
                "preferred_timeframe": adjusted_config["timeframe_preference"],
                "multi_tf_signal": mtf_analysis.get("multi_tf_signal", "HOLD"),
                "signal_confidence": mtf_analysis.get("confidence", 0.5),
                "strategy_performance": performance,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }

        except Exception as e:
            logger.exception(f"Error getting adaptive strategy for {symbol}: {e}")
            return {"symbol": symbol, "error": str(e), "fallback_strategy": "conservative_hold", "risk_multiplier": 0.5, "timestamp": datetime.now(timezone.utc).isoformat()}

    async def _get_market_regime(self, symbol: str) -> str:
        """Get current market regime for symbol"""
        try:
            # Check cache first (valid for 1 hour)
            cache_key = f"{symbol}_regime"
            if cache_key in self.regime_cache:
                cached_time, regime = self.regime_cache[cache_key]
                if (datetime.now(timezone.utc) - cached_time).seconds < 3600:
                    return regime

            # Fetch data and classify regime
            limiter = await BinanceWeightLimiter.create()
            client = BinanceREST(limiter)
            klines = await client.get_klines(symbol, "1h", limit=200)

            if not klines:
                return MarketRegime.SIDEWAYS

            # Convert to DataFrame
            df = pd.DataFrame(
                klines,
                columns=[
                    "timestamp",
                    "open",
                    "high",
                    "low",
                    "close",
                    "volume",
                    "close_time",
                    "quote_asset_volume",
                    "number_of_trades",
                    "taker_buy_base_asset_volume",
                    "taker_buy_quote_asset_volume",
                    "ignore",
                ],
            )
            df = df[["timestamp", "open", "high", "low", "close", "volume"]].copy()
            df = df.astype({"open": float, "high": float, "low": float, "close": float, "volume": float})

            regime = MarketRegime.classify_regime(df, symbol)

            # Cache result
            self.regime_cache[cache_key] = (datetime.now(timezone.utc), regime)

        except Exception as ex:
            logger.debug("get_current_regime failed: %s", ex)
            return MarketRegime.SIDEWAYS
        else:
            return regime

    def _adjust_for_multi_tf(self, strategy_config: dict[str, Any], mtf_analysis: dict[str, Any]) -> dict[str, Any]:
        """Adjust strategy based on multi-timeframe analysis"""
        try:
            from backend.services.confidence_normalizer import ConfidenceNormalizer

            signal = mtf_analysis.get("multi_tf_signal", "HOLD")
            raw_conf = mtf_analysis.get("confidence", 0.5)
            confidence = ConfidenceNormalizer.normalize(float(raw_conf) if raw_conf is not None else 0.5)

            adjusted_config = strategy_config.copy()

            # Adjust risk based on signal strength
            if signal in ["BUY", "SELL"] and confidence > 0.7:
                adjusted_config["risk_multiplier"] *= 1.1  # Increase risk for strong signals
            elif confidence < 0.4:
                adjusted_config["risk_multiplier"] *= 0.8  # Decrease risk for weak signals

            # Adjust timeframe preference based on signal
            if signal != "HOLD" and confidence > 0.8:
                # Strong signals work better on shorter timeframes
                timeframe_hierarchy = {"5m": 0, "15m": 1, "1h": 2, "4h": 3, "1d": 4}
                current_idx = timeframe_hierarchy.get(adjusted_config["timeframe_preference"], 2)

                if current_idx > 0:  # Can go shorter
                    timeframes = list(timeframe_hierarchy.keys())
                    adjusted_config["timeframe_preference"] = timeframes[current_idx - 1]

        except Exception as ex:
            logger.debug("_adjust_for_multi_tf failed: %s", ex)
            return strategy_config
        else:
            return adjusted_config

    def _get_strategy_performance(self, strategy_name: str) -> dict[str, Any]:
        """Get performance metrics for a specific strategy"""
        # This would be populated from actual trading results
        # For now, return default metrics
        return {"win_rate": 0.55, "avg_return": 0.02, "max_drawdown": 0.08, "sharpe_ratio": 1.2, "total_trades": 150, "last_updated": datetime.now(timezone.utc).isoformat()}


# Global instance
adaptive_strategy_manager = AdaptiveStrategyManager()
