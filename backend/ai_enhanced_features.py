"""
Enhanced AI Features for Mystic Trading Platform
Advanced AI capabilities with modern LLM integration (no external hard deps required).
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Any

from backend.services.confidence_normalizer import ConfidenceNormalizer
from backend.utils.exceptions import AIError

# Core libs
try:
    import numpy as np  # type: ignore[import-untyped]
except (ImportError, ModuleNotFoundError, AttributeError):  # pragma: no cover
    np = None  # type: ignore[assignment]

try:
    import pandas as pd  # type: ignore[import-untyped]
except (ImportError, ModuleNotFoundError, AttributeError):  # pragma: no cover
    pd = None  # type: ignore[assignment]

# Optional AI/ML libs
_HAS_LANGCHAIN = False
_HAS_SENTENCE_TRANSFORMERS = False
_HAS_TRANSFORMERS = False
_HAS_TORCH = False

try:
    from langchain.chains import LLMChain  # type: ignore[import-not-found]
    from langchain.prompts import PromptTemplate  # type: ignore[import-not-found]
    from langchain_community.llms import OpenAI  # type: ignore[import-not-found]

    _HAS_LANGCHAIN = True
except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError):
    pass

# SentenceTransformer disabled - conflicts with torch 2.5.1
# See requirements.txt for details
_HAS_SENTENCE_TRANSFORMERS = False
# try:
#     from sentence_transformers import SentenceTransformer
#     _HAS_SENTENCE_TRANSFORMERS = True
# except ImportError:
#     pass

try:
    from transformers import pipeline  # type: ignore[import-not-found]

    _HAS_TRANSFORMERS = True
except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError):
    pass

try:
    import torch  # type: ignore[import-untyped]

    _HAS_TORCH = True
except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError):
    pass

logger = logging.getLogger(__name__)


@dataclass
class MarketSentiment:
    symbol: str
    sentiment_score: float  # -1 to 1
    confidence: float
    sources: list[str]
    timestamp: datetime
    news_count: int
    social_volume: int
    fear_greed_index: float


@dataclass
class AIPrediction:
    symbol: str
    prediction_type: str  # "price_direction" | "volatility" | "volume"
    predicted_value: float
    confidence: float
    timeframe: str
    model_version: str
    features_used: list[str]
    timestamp: datetime


@dataclass
class StrategyRecommendation:
    symbol: str
    action: str  # "buy" | "sell" | "hold"
    confidence: float
    reasoning: str
    risk_level: str
    expected_return: float
    time_horizon: str
    stop_loss: float | None
    take_profit: float | None
    timestamp: datetime


class EnhancedSentimentAnalyzer:
    def __init__(self) -> None:
        self.sentiment_model = None
        self.embedding_model = None
        self.news_sources = [
            "reuters",
            "bloomberg",
            "cnbc",
            "marketwatch",
            "yahoo_finance",
            "seeking_alpha",
            "reddit",
            "twitter",
        ]

        try:
            if _HAS_TRANSFORMERS:
                device = 0 if (_HAS_TORCH and torch.cuda.is_available()) else -1  # type: ignore[attr-defined]
                self.sentiment_model = pipeline("sentiment-analysis", model="ProsusAI/finbert", device=device)  # type: ignore[call-arg]
            # Sentence transformers disabled - conflicts with torch 2.5.1
            # if _HAS_SENTENCE_TRANSFORMERS:
            #     self.embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
        except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
            logger.warning("AI model init failed: %s", e)

    def _raise_sentiment_error(self, error: Exception) -> None:
        """Helper function to raise sentiment analysis errors (abstracts raise for TRY301)"""
        logger.critical(f"Sentiment model failed - NO FALLBACK IN PRODUCTION: {error}")
        msg = f"Sentiment analysis failed - production requires working sentiment model: {error}"
        raise RuntimeError(msg) from error

    async def analyze_market_sentiment(
        self,
        symbol: str,
        news_data: list[dict[str, Any]],
        social_data: list[dict[str, Any]],
    ) -> MarketSentiment:
        if not self.sentiment_model:
            logger.critical("Sentiment model not loaded - NO FALLBACK IN PRODUCTION")
            msg = "Sentiment model unavailable - production requires loaded models"
            raise RuntimeError(msg)

        try:
            news_sentiments: list[float] = []
            for news in news_data[:50]:
                text = f"{news.get('title', '')} {news.get('content', '')}".strip()
                if not text:
                    continue
                try:
                    result = self.sentiment_model(text)
                    label = str(result[0].get("label", "NEUTRAL")).upper()
                    score = float(result[0].get("score", 0.0))
                    news_sentiments.append((1.0 if "POS" in label else -1.0 if "NEG" in label else 0.0) * score)
                except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
                    self._raise_sentiment_error(e)

            social_sentiments: list[float] = []
            for post in social_data[:100]:
                text = str(post.get("text", "")).strip()
                if not text:
                    continue
                try:
                    result = self.sentiment_model(text)
                    label = str(result[0].get("label", "NEUTRAL")).upper()
                    score = float(result[0].get("score", 0.0))
                    social_sentiments.append((1.0 if "POS" in label else -1.0 if "NEG" in label else 0.0) * score)
                except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
                    self._raise_sentiment_error(e)

            news_weight = 0.7
            social_weight = 0.3

            avg_news = float(np.mean(news_sentiments)) if (np is not None and news_sentiments) else 0.0  # type: ignore[arg-type]
            avg_social = float(np.mean(social_sentiments)) if (np is not None and social_sentiments) else 0.0  # type: ignore[arg-type]
            overall = news_weight * avg_news + social_weight * avg_social

            volume_factor = min(1.0, (len(news_sentiments) + len(social_sentiments)) / 100.0)
            confidence = ConfidenceNormalizer.normalize(float(volume_factor))

            fear_greed = self._calculate_fear_greed_index(overall, confidence)

            return MarketSentiment(
                symbol=symbol,
                sentiment_score=overall,
                confidence=confidence,
                sources=self.news_sources,
                timestamp=datetime.now(timezone.utc),
                news_count=len(news_data),
                social_volume=len(social_data),
                fear_greed_index=fear_greed,
            )
        except RuntimeError as e:
            if "production requires" in str(e) or "NO FALLBACK" in str(e):
                raise
            logger.exception("Sentiment analysis failed for %s: %s", symbol, e)
            return MarketSentiment(
                symbol=symbol,
                sentiment_score=0.0,
                confidence=0.0,
                sources=[],
                timestamp=datetime.now(timezone.utc),
                news_count=0,
                social_volume=0,
                fear_greed_index=50.0,
            )
        except (ValueError, TypeError, AttributeError, KeyError, IndexError) as e:
            logger.exception("Sentiment analysis failed for %s: %s", symbol, e)
            return MarketSentiment(
                symbol=symbol,
                sentiment_score=0.0,
                confidence=0.0,
                sources=[],
                timestamp=datetime.now(timezone.utc),
                news_count=0,
                social_volume=0,
                fear_greed_index=50.0,
            )

    def _calculate_fear_greed_index(self, sentiment: float, confidence: float) -> float:
        fg = 50.0 + (sentiment * 50.0 * confidence)
        return float(max(0.0, min(100.0, fg)))


class AdvancedPredictor:
    def __init__(self) -> None:
        self.models: dict[str, Any] = {}
        self.feature_importance: dict[str, float] = {}

    async def predict_price_direction(
        self,
        symbol: str,
        historical_data: Any,
        sentiment_data: MarketSentiment,
        technical_indicators: dict[str, float],
    ) -> AIPrediction:
        try:
            features = self._prepare_features(historical_data, sentiment_data, technical_indicators)

            preds: list[float] = []
            weights: list[float] = []

            preds.append(self._lstm_prediction(historical_data))
            weights.append(0.3)

            preds.append(self._transformer_prediction(features))
            weights.append(0.3)

            preds.append(self._sentiment_prediction(sentiment_data))
            weights.append(0.2)

            preds.append(self._technical_prediction(technical_indicators))
            weights.append(0.2)

            if preds:
                if np is not None:
                    try:
                        ensemble = float(np.average(preds, weights=weights))  # type: ignore[arg-type]
                    except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError):
                        # fallback to simple average
                        ensemble = float(sum(preds) / len(preds))
                else:
                    ensemble = float(sum(preds) / len(preds))
                conf = self._confidence_from_agreement(preds, weights)
            else:
                ensemble = 0.0
                conf = 0.0

            return AIPrediction(
                symbol=symbol,
                prediction_type="price_direction",
                predicted_value=ensemble,
                confidence=conf,
                timeframe="1h",
                model_version="ensemble_v2",
                features_used=list(features.keys()),
                timestamp=datetime.now(timezone.utc),
            )
        except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
            logger.exception("Prediction failed for %s: %s", symbol, e)
            return AIPrediction(
                symbol=symbol,
                prediction_type="price_direction",
                predicted_value=0.0,
                confidence=0.0,
                timeframe="1h",
                model_version="error",
                features_used=[],
                timestamp=datetime.now(timezone.utc),
            )

    def _prepare_features(
        self,
        historical_data: Any,
        sentiment_data: MarketSentiment,
        technical_indicators: dict[str, float],
    ) -> dict[str, float]:
        features: dict[str, float] = {}
        # Historical features
        try:
            if pd is not None and isinstance(historical_data, pd.DataFrame) and not historical_data.empty:
                df = historical_data
                # Safe access to close column
                if "close" in df.columns:
                    closes = df["close"].dropna().astype(float)
                    if len(closes) > 0:
                        last = float(closes.iloc[-1])
                        mean5 = float(closes.tail(5).mean()) if len(closes) >= 1 else last
                        returns = (last - mean5) / (mean5 if mean5 != 0 else 1.0)
                        features["hist_recent_return"] = returns
                        features["hist_mean5"] = mean5
                # volume
                if "volume" in df.columns:
                    vols = df["volume"].dropna().astype(float)
                    if len(vols) > 0:
                        features["avg_volume_5"] = float(vols.tail(5).mean())
        except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError):
            logger.debug("Failed to extract historical features", exc_info=True)

        # Sentiment features
        try:
            features["sentiment_score"] = float(getattr(sentiment_data, "sentiment_score", 0.0) or 0.0)
            raw_sc = float(getattr(sentiment_data, "confidence", 0.0) or 0.0)
            features["sentiment_confidence"] = ConfidenceNormalizer.normalize(raw_sc)
        except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError):
            features["sentiment_score"] = 0.0
            features["sentiment_confidence"] = 0.0

        # Technical indicators
        try:
            for k, v in (technical_indicators or {}).items():
                try:
                    features[f"ti_{k}"] = float(v)
                except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError):
                    continue
        except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError):
            pass

        return features

    def _lstm_prediction(self, historical_data: Any) -> float:
        # Placeholder: simple momentum-like heuristic instead of real LSTM
        try:
            if pd is not None and isinstance(historical_data, pd.DataFrame) and not historical_data.empty and "close" in historical_data.columns:
                closes = historical_data["close"].dropna().astype(float)
                if len(closes) >= 2:
                    last = float(closes.iloc[-1])
                    prev = float(closes.iloc[-2])
                    ret = (last - prev) / (prev if prev != 0 else 1.0)
                    # Clamp to reasonable range [-1, 1]
                    return max(-1.0, min(1.0, float(ret)))
        except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError):
            logger.debug("LSTM prediction fallback used", exc_info=True)
        return 0.0

    def _transformer_prediction(self, features: dict[str, float]) -> float:
        # Placeholder: weighted sum of features normalized
        try:
            if not features:
                return 0.0
            total = 0.0
            count = 0
            for idx, (_k, v) in enumerate(features.items()):
                weight = 1.0 / (idx + 1)
                try:
                    total += float(v) * weight
                    count += abs(weight)
                except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError):
                    continue
            if count == 0:
                return 0.0
            val = total / count
            # Normalize to [-1,1] roughly
            return max(-1.0, min(1.0, float(val)))
        except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError):
            return 0.0

    def _sentiment_prediction(self, sentiment_data: MarketSentiment) -> float:
        try:
            return float(getattr(sentiment_data, "sentiment_score", 0.0) or 0.0)
        except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError):
            return 0.0

    def _technical_prediction(self, technical_indicators: dict[str, float]) -> float:
        try:
            vals = []
            for v in (technical_indicators or {}).values():
                try:
                    vals.append(float(v))
                except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError):
                    continue
            if not vals:
                return 0.0
            avg = sum(vals) / len(vals)
            return max(-1.0, min(1.0, float(avg)))
        except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError):
            return 0.0

    def _confidence_from_agreement(self, preds: list[float], weights: list[float]) -> float:
        try:
            if not preds:
                return 0.0
            # Compute weighted mean
            if np is not None:
                try:
                    mean = float(np.average(preds, weights=weights))  # type: ignore[arg-type]
                    variance = float(np.average([(p - mean) ** 2 for p in preds], weights=weights))  # type: ignore[arg-type]
                    std = float(np.sqrt(variance))
                except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError):
                    mean = float(sum(preds) / len(preds))
                    # sample std
                    var = sum((p - mean) ** 2 for p in preds) / len(preds)
                    std = float(var**0.5)
            else:
                mean = float(sum(preds) / len(preds))
                var = sum((p - mean) ** 2 for p in preds) / len(preds)
                std = float(var**0.5)
            # Normalize std to [0,1] relative to scale of mean/magnitude
            denom = max(1.0, max(abs(p) for p in preds) * 2.0)
            score = 1.0 - min(1.0, std / denom)
            return float(max(0.0, min(1.0, score)))
        except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError):
            return 0.0


class AIStrategyGenerator:
    def __init__(self) -> None:
        self.llm_chain: Any | None = None
        self.strategy_templates = self._load_strategy_templates()

        if _HAS_LANGCHAIN:
            try:
                prompt_template = PromptTemplate(
                    input_variables=[
                        "market_data",
                        "sentiment",
                        "technical_indicators",
                        "risk_profile",
                    ],
                    template=(
                        "Based on the following market data, generate a trading strategy:\n\n"
                        "Market Data: {market_data}\n"
                        "Sentiment: {sentiment}\n"
                        "Technical Indicators: {technical_indicators}\n"
                        "Risk Profile: {risk_profile}\n\n"
                        "Generate a detailed trading strategy with:\n"
                        "1. Entry conditions\n"
                        "2. Exit conditions\n"
                        "3. Risk management rules\n"
                        "4. Position sizing\n"
                        "5. Expected outcomes\n\n"
                        "Strategy:\n"
                    ),
                )
                self.llm_chain = LLMChain(llm=OpenAI(temperature=0.7), prompt=prompt_template)  # type: ignore[call-arg]
            except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
                logger.warning("LLM chain init failed, using fallback strategies: %s", e)

    def _load_strategy_templates(self) -> dict[str, str]:
        return {
            "momentum": ("Strategy: Momentum Breakout\nEntry: Break above resistance with rising volume\nExit: Stop at support or TP at 2:1 R/R\nRisk: 2% per trade\n"),
            "mean_reversion": ("Strategy: Mean Reversion\nEntry: Oversold (RSI < 30) or overbought (RSI > 70)\nExit: Revert to mean or opposite extreme\nRisk: 1.5% per trade\n"),
            "trend_following": ("Strategy: Trend Following\nEntry: Above MAs with momentum confirmation\nExit: Trend reversal or trailing stop\nRisk: 2.5% per trade\n"),
        }

    async def generate_strategy(
        self,
        symbol: str,
        market_data: dict[str, Any],
        sentiment_data: MarketSentiment,
        technical_indicators: dict[str, float],
        risk_profile: str = "moderate",
    ) -> StrategyRecommendation:
        try:
            market_summary = self._summarize_market_data(market_data)
            sentiment_summary = f"Sentiment={sentiment_data.sentiment_score:.2f}, FG={sentiment_data.fear_greed_index:.1f}"
            technical_summary = self._summarize_technical_indicators(technical_indicators)

            strategy_text: str
            if self.llm_chain and hasattr(self.llm_chain, "arun"):
                try:
                    strategy_text = await self.llm_chain.arun(  # type: ignore[func-returns-value]
                        market_data=market_summary,
                        sentiment=sentiment_summary,
                        technical_indicators=technical_summary,
                        risk_profile=risk_profile,
                    )
                except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
                    # Return error instead of fallback strategy
                    msg = f"LLM generation failed: {e}"
                    raise AIError(msg, original_exception=e) from e
            else:
                # Return error instead of fallback strategy
                msg = "LLM service not available"
                raise AIError(msg)

            return self._parse_strategy_to_recommendation(symbol, strategy_text, sentiment_data, technical_indicators)
        except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
            logger.critical("Strategy generation failed for %s: %s - NO FALLBACK IN PRODUCTION", symbol, e)
            msg = f"Strategy generation failed for {symbol} - production requires working strategy system: {e}"
            raise RuntimeError(msg) from e

    def _summarize_market_data(self, market_data: dict[str, Any]) -> str:
        price = float(market_data.get("current_price", 0.0))
        vol = float(market_data.get("volume", 0.0))
        ch = float(market_data.get("change_24h", 0.0))
        return f"Price={price:.2f}, Volume={vol:.0f}, 24hChange={ch:.2%}"

    def _summarize_technical_indicators(self, indicators: dict[str, float]) -> str:
        parts: list[str] = []
        for k, v in (indicators or {}).items():
            try:
                parts.append(f"{k}={float(v):.2f}")
            except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError):
                continue
        return ", ".join(parts)

    def _parse_strategy_to_recommendation(
        self,
        symbol: str,
        strategy_text: str,
        sentiment_data: MarketSentiment,
        _indicators: dict[str, float],
    ) -> StrategyRecommendation:
        text_l = strategy_text.lower()
        if ("buy" in text_l) or ("long" in text_l) or ("momentum" in text_l) or ("trend" in text_l):
            action = "buy"
            confidence = min(0.8, float(sentiment_data.confidence) + 0.2)
        elif ("sell" in text_l) or ("short" in text_l):
            action = "sell"
            confidence = min(0.8, float(sentiment_data.confidence) + 0.2)
        else:
            action = "hold"
            confidence = 0.5

        return StrategyRecommendation(
            symbol=symbol,
            action=action,
            confidence=ConfidenceNormalizer.normalize(float(confidence)),
            reasoning=strategy_text.strip(),
            risk_level="moderate",
            expected_return=0.02 if action != "hold" else 0.0,
            time_horizon="1d",
            stop_loss=None,
            take_profit=None,
            timestamp=datetime.now(timezone.utc),
        )


class EnhancedAITrading:
    def __init__(self) -> None:
        self.sentiment_analyzer = EnhancedSentimentAnalyzer()
        self.predictor = AdvancedPredictor()
        self.strategy_generator = AIStrategyGenerator()
        self.active_predictions: dict[str, AIPrediction] = {}
        self.active_recommendations: dict[str, StrategyRecommendation] = {}

    async def analyze_symbol(
        self,
        symbol: str,
        market_data: dict[str, Any],
        news_data: list[dict[str, Any]],
        social_data: list[dict[str, Any]],
        technical_indicators: dict[str, float],
    ) -> dict[str, Any]:
        try:
            historical_df = None if pd is None else pd.DataFrame(market_data.get("historical", []))

            sentiment = await self.sentiment_analyzer.analyze_market_sentiment(symbol, news_data, social_data)

            hist_arg = historical_df if (pd is not None and isinstance(historical_df, pd.DataFrame)) else None

            prediction = await self.predictor.predict_price_direction(
                symbol,
                hist_arg,
                sentiment,
                technical_indicators,
            )

            recommendation = await self.strategy_generator.generate_strategy(symbol, market_data, sentiment, technical_indicators)

            self.active_predictions[symbol] = prediction
            self.active_recommendations[symbol] = recommendation

            return {
                "symbol": symbol,
                "sentiment": asdict(sentiment),
                "prediction": asdict(prediction),
                "recommendation": asdict(recommendation),
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
        except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
            logger.exception("AI analysis failed for %s: %s", symbol, e)
            return {
                "symbol": symbol,
                "error": str(e),
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }

    async def get_portfolio_insights(self, portfolio_symbols: list[str]) -> dict[str, Any]:
        insights = {
            "overall_sentiment": 0.0,
            "risk_assessment": "moderate",
            "recommendations": [],
            "high_confidence_signals": [],
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

        vals: list[float] = []
        for sym in portfolio_symbols:
            pred = self.active_predictions.get(sym)
            if pred is not None:
                vals.append(float(pred.predicted_value))

        if vals:
            if np is not None:
                try:
                    insights["overall_sentiment"] = float(np.mean(vals))  # type: ignore[arg-type]
                except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError):
                    insights["overall_sentiment"] = float(sum(vals) / len(vals))
            else:
                insights["overall_sentiment"] = float(sum(vals) / len(vals))

        if float(insights["overall_sentiment"]) < -0.3:
            insights["risk_assessment"] = "high"
        elif float(insights["overall_sentiment"]) > 0.3:
            insights["risk_assessment"] = "low"

        for sym, rec in self.active_recommendations.items():
            norm_conf = ConfidenceNormalizer.normalize(float(rec.confidence))
            if norm_conf > 0.7:
                insights["high_confidence_signals"].append(
                    {
                        "symbol": sym,
                        "action": rec.action,
                        "confidence": norm_conf,
                        "reasoning": rec.reasoning,
                    }
                )

        return insights
