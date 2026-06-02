"""
Multi-Modal AI Learning System
Integrates learning from price data, sentiment, news, and social media

Quick test checklist:
- ccxt calls only receive BASE/QUOTE (N/A in this module; verify where used).
- No exchange string leaks: EXCHANGE_ID imported from trading_universe.
- No unreachable code after returns.
- Logging has no weird characters.
- No mock data anywhere: all features are derived from real inputs.
"""

import json
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any

import numpy as np
import torch
from textblob import TextBlob
from torch import nn

import redis

# Import from single source of truth
try:
    from backend.config.trading_universe import EXCHANGE_ID
except (ImportError, ModuleNotFoundError, AttributeError, ValueError, TypeError, RuntimeError) as e:
    msg = f"Failed to import EXCHANGE_ID from trading_universe: {e}"
    raise RuntimeError(msg) from e

logger = logging.getLogger(__name__)


@dataclass
class MultiModalData:
    technical_features: np.ndarray
    sentiment_features: np.ndarray
    news_features: np.ndarray
    social_features: np.ndarray
    timestamp: str
    symbol: str


class SentimentLearner:
    def __init__(self) -> None:
        self.sentiment_history: dict[str, list[dict[str, Any]]] = {}
        self.sentiment_weights = {"news": 0.4, "social": 0.3, "technical": 0.3}

    async def process_sentiment_data(self, symbol: str, data: dict[str, Any]) -> np.ndarray:
        try:
            features = []

            news_sentiment = await self.analyze_news_sentiment(data.get("news", []))
            features.extend(
                [
                    news_sentiment.get("compound", 0.0),
                    news_sentiment.get("positive", 0.0),
                    news_sentiment.get("negative", 0.0),
                    news_sentiment.get("neutral", 0.0),
                    news_sentiment.get("volume", 0.0),
                ],
            )

            social_sentiment = await self.analyze_social_sentiment(data.get("social", []))
            features.extend(
                [
                    social_sentiment.get("compound", 0.0),
                    social_sentiment.get("positive", 0.0),
                    social_sentiment.get("negative", 0.0),
                    social_sentiment.get("engagement", 0.0),
                    social_sentiment.get("volume", 0.0),
                ],
            )

            sentiment_momentum = await self.calculate_sentiment_momentum(symbol, features[:5])
            features.extend(sentiment_momentum)

            return np.array(features, dtype=np.float32)
        except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
            logger.exception(f"Error processing sentiment data: {e}")
            return np.zeros(12, dtype=np.float32)

    async def analyze_news_sentiment(self, news_articles: list[dict]) -> dict[str, float]:
        try:
            if not news_articles:
                return {
                    "compound": 0.0,
                    "positive": 0.0,
                    "negative": 0.0,
                    "neutral": 0.0,
                    "volume": 0.0,
                }

            sentiments = []
            total_weight = 0.0

            for article in news_articles:
                text = f"{article.get('title', '')} {article.get('description', '')}".strip()
                if not text:
                    continue

                blob = TextBlob(text)
                sentiment = float(blob.sentiment.polarity)

                weight = self.calculate_news_weight(article)
                sentiments.append(sentiment * weight)
                total_weight += weight

            if not sentiments or total_weight <= 0:
                return {
                    "compound": 0.0,
                    "positive": 0.0,
                    "negative": 0.0,
                    "neutral": 0.0,
                    "volume": 0.0,
                }

            avg_sentiment = float(sum(sentiments) / total_weight)

            positive = max(0.0, avg_sentiment)
            negative = abs(min(0.0, avg_sentiment))
            neutral = max(0.0, 1.0 - positive - negative)

            return {
                "compound": avg_sentiment,
                "positive": positive,
                "negative": negative,
                "neutral": neutral,
                "volume": len(news_articles) / 100.0,
            }
        except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
            logger.exception(f"Error analyzing news sentiment: {e}")
            return {
                "compound": 0.0,
                "positive": 0.0,
                "negative": 0.0,
                "neutral": 0.0,
                "volume": 0.0,
            }

    def _parse_ts_aware(self, ts: str | None) -> datetime:
        if not ts:
            return datetime.now(timezone.utc)
        try:
            dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt.astimezone(timezone.utc)
        except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError):
            return datetime.now(timezone.utc)

    def calculate_news_weight(self, article: dict) -> float:
        try:
            weight = 1.0
            source = str(article.get("source", "")).lower()
            credible_sources = ["reuters", "bloomberg", "cnbc", "marketwatch", "wsj"]
            if any(s in source for s in credible_sources):
                weight *= 1.5

            pub_time = article.get("publishedAt") or article.get("published_at")
            pub_dt = self._parse_ts_aware(pub_time if isinstance(pub_time, str) else None)
            hours_old = max(0.0, (datetime.now(timezone.utc) - pub_dt).total_seconds() / 3600.0)
            recency_weight = max(0.1, 1.0 - (hours_old / 24.0))
            weight *= recency_weight

            return float(weight)
        except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError):
            return 1.0

    async def analyze_social_sentiment(self, social_posts: list[dict]) -> dict[str, float]:
        try:
            if not social_posts:
                return {
                    "compound": 0.0,
                    "positive": 0.0,
                    "negative": 0.0,
                    "engagement": 0.0,
                    "volume": 0.0,
                }

            num = 0
            sum_weighted = 0.0
            sum_weights = 0.0
            total_engagement = 0.0

            for post in social_posts:
                text = str(post.get("text", "")).strip()
                if not text:
                    continue

                blob = TextBlob(text)
                sentiment = float(blob.sentiment.polarity)

                likes = float(post.get("likes", 0) or 0)
                retweets = float(post.get("retweets", 0) or 0)
                replies = float(post.get("replies", 0) or 0)
                engagement = max(0.0, likes + retweets + replies)
                weight = max(1.0, float(np.log(engagement + 1.0)))

                sum_weighted += sentiment * weight
                sum_weights += weight
                total_engagement += engagement
                num += 1

            if num == 0 or sum_weights == 0:
                return {
                    "compound": 0.0,
                    "positive": 0.0,
                    "negative": 0.0,
                    "engagement": 0.0,
                    "volume": 0.0,
                }

            avg_sentiment = float(sum_weighted / sum_weights)

            return {
                "compound": avg_sentiment,
                "positive": max(0.0, avg_sentiment),
                "negative": abs(min(0.0, avg_sentiment)),
                "engagement": min(total_engagement / 1000.0, 10.0),
                "volume": num / 100.0,
            }
        except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
            logger.exception(f"Error analyzing social sentiment: {e}")
            return {
                "compound": 0.0,
                "positive": 0.0,
                "negative": 0.0,
                "engagement": 0.0,
                "volume": 0.0,
            }

    async def calculate_sentiment_momentum(self, symbol: str, current_sentiment: list[float]) -> list[float]:
        """
        Calculate momentum and acceleration based on stored sentiment history.
        Returns [momentum, acceleration]
        """
        try:
            if symbol not in self.sentiment_history:
                self.sentiment_history[symbol] = []

            history = self.sentiment_history[symbol]
            history.append(
                {
                    "sentiment": list(current_sentiment),
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                }
            )

            # prune older than 24 hours
            cutoff_time = datetime.now(timezone.utc) - timedelta(hours=24)
            pruned = []
            for h in history:
                ts = h.get("timestamp")
                try:
                    dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
                    if dt.tzinfo is None:
                        dt = dt.replace(tzinfo=timezone.utc)
                    if dt > cutoff_time:
                        pruned.append(h)
                except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError):
                    # if parsing fails, keep the item conservatively
                    pruned.append(h)
            self.sentiment_history[symbol] = pruned
            history = pruned

            if len(history) < 2:
                # Not enough history to compute momentum
                setattr(self, f"_last_momentum_{symbol}", 0.0)
                return [0.0, 0.0]

            # Use the first element of sentiment vector as representative (compound)
            recent_vals = [float(h["sentiment"][0]) for h in history[-5:]]
            older_vals = [float(h["sentiment"][0]) for h in history[:-5]] if len(history) > 5 else []

            momentum = float(np.mean(recent_vals) - recent_vals[0]) if not older_vals else float(np.mean(recent_vals) - np.mean(older_vals))

            last_m = getattr(self, f"_last_momentum_{symbol}", 0.0)
            acceleration = float(momentum - last_m)

            setattr(self, f"_last_momentum_{symbol}", momentum)
        except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
            logger.exception(f"Error calculating sentiment momentum: {e}")
            return [0.0, 0.0]
        else:
            return [momentum, acceleration]


class FusionNetwork(nn.Module):
    def __init__(self, hidden_size: int = 64):
        super().__init__()
        # We'll accept variable sized inputs by concatenating flattened vectors
        self.hidden_size = hidden_size
        self.fc1 = nn.Linear(512, hidden_size)  # will adapt at runtime if needed
        self.fc2 = nn.Linear(hidden_size, 1)
        self.activation = nn.ReLU()
        # We'll handle potential size mismatch in forward

    def forward(self, tech: torch.Tensor, sent: torch.Tensor, news: torch.Tensor, social: torch.Tensor) -> torch.Tensor:
        # Flatten last dim and concatenate along feature axis
        def flatten(x: torch.Tensor) -> torch.Tensor:
            return x.view(x.size(0), -1)

        t = flatten(tech)
        s = flatten(sent)
        n = flatten(news)
        so = flatten(social)
        x = torch.cat([t, s, n, so], dim=1)

        # If fc1 weight shape doesn't match input, recreate small MLP on the fly
        if self.fc1.in_features != x.size(1):
            # recreate layers to match input dim while preserving device and dtype
            device = x.device
            dtype = x.dtype
            self.fc1 = nn.Linear(x.size(1), self.hidden_size).to(device=device, dtype=dtype)
            self.fc2 = nn.Linear(self.hidden_size, 1).to(device=device, dtype=dtype)

        x = self.activation(self.fc1(x))
        x = self.fc2(x)
        # Keep output as unbounded float; downstream logic will interpret magnitude
        return x.squeeze(1)  # shape (batch,)


class MultiModalAILearner:
    def __init__(self, redis_client: redis.Redis):
        self.redis_client = redis_client
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.sentiment_learner = SentimentLearner()
        self.learning_history: list[dict[str, Any]] = []
        self.max_history = 10000
        self.fusion_network = FusionNetwork()
        self.fusion_network.to(self.device)
        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.fusion_network.parameters(), lr=1e-3)

    async def compute_news_features(self, news_data: list[dict]) -> np.ndarray:
        try:
            if not news_data:
                return np.zeros(10, dtype=np.float32)

            sent = await self.sentiment_learner.analyze_news_sentiment(news_data)

            credible_sources = ["reuters", "bloomberg", "cnbc", "marketwatch", "wsj"]
            now = datetime.now(timezone.utc)
            rec_hours = []
            lengths = []
            credible = 0

            for article in news_data:
                pub = article.get("publishedAt") or article.get("published_at")
                dt = self.sentiment_learner._parse_ts_aware(pub if isinstance(pub, str) else None)
                rec_hours.append(max(0.0, (now - dt).total_seconds() / 3600.0))
                text = f"{article.get('title', '')} {article.get('description', '')}"
                lengths.append(float(len(text)))
                src = str(article.get("source", "")).lower()
                if any(s in src for s in credible_sources):
                    credible += 1

            n = float(len(news_data))
            avg_h = float(np.mean(rec_hours)) if rec_hours else 24.0
            min_h = float(np.min(rec_hours)) if rec_hours else 24.0
            max_h = float(np.max(rec_hours)) if rec_hours else 24.0
            avg_len = float(np.mean(lengths)) if lengths else 0.0

            def nh(x: float) -> float:
                return max(0.0, 1.0 - min(x, 24.0) / 24.0)

            features = [
                float(sent.get("compound", 0.0)),
                float(sent.get("positive", 0.0)),
                float(sent.get("negative", 0.0)),
                float(sent.get("neutral", 0.0)),
                float(sent.get("volume", 0.0)),
                float(credible / n) if n > 0 else 0.0,
                nh(avg_h),
                nh(min_h),
                nh(max_h),
                float(min(avg_len / 1000.0, 1.0)),
            ]
            return np.array(features, dtype=np.float32)
        except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
            logger.exception(f"Error computing news features: {e}")
            return np.zeros(10, dtype=np.float32)

    async def _compute_social_features(self, social_data: list[dict]) -> np.ndarray:
        try:
            if not social_data:
                return np.zeros(8, dtype=np.float32)

            sent = await self.sentiment_learner.analyze_social_sentiment(social_data)

            now = datetime.now(timezone.utc)
            rec_hours = []
            text_lengths = []
            likes_list = []
            retweets_list = []

            for p in social_data:
                ts = p.get("timestamp") or p.get("created_at") or p.get("time")
                dt = self.sentiment_learner._parse_ts_aware(ts if isinstance(ts, str) else None)
                rec_hours.append(max(0.0, (now - dt).total_seconds() / 3600.0))
                text_lengths.append(float(len(str(p.get("text", "")))))
                likes_list.append(float(p.get("likes", 0) or 0))
                retweets_list.append(float(p.get("retweets", 0) or 0))

            def nh(x: float) -> float:
                return max(0.0, 1.0 - min(x, 24.0) / 24.0)

            avg_len = float(np.mean(text_lengths)) if text_lengths else 0.0
            avg_recency = float(np.mean(rec_hours)) if rec_hours else 24.0
            avg_likes = float(np.mean(likes_list)) if likes_list else 0.0

            features = [
                float(sent.get("compound", 0.0)),
                float(sent.get("positive", 0.0)),
                float(sent.get("negative", 0.0)),
                float(sent.get("engagement", 0.0)),
                float(sent.get("volume", 0.0)),
                float(min(avg_len / 500.0, 1.0)),
                nh(avg_recency),
                float(min(avg_likes / 1000.0, 1.0)),
            ]
            return np.array(features, dtype=np.float32)
        except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
            logger.exception(f"Error computing social features: {e}")
            return np.zeros(8, dtype=np.float32)

    async def update_fusion_network(self, multimodal_data: MultiModalData, reward: float):
        try:
            self.fusion_network.train()

            tech_tensor = torch.as_tensor(multimodal_data.technical_features, dtype=torch.float32).unsqueeze(0).to(self.device)
            sent_tensor = torch.as_tensor(multimodal_data.sentiment_features, dtype=torch.float32).unsqueeze(0).to(self.device)
            news_tensor = torch.as_tensor(multimodal_data.news_features, dtype=torch.float32).unsqueeze(0).to(self.device)
            social_tensor = torch.as_tensor(multimodal_data.social_features, dtype=torch.float32).unsqueeze(0).to(self.device)
            reward_tensor = torch.as_tensor([reward], dtype=torch.float32).to(self.device)

            self.optimizer.zero_grad()
            prediction = self.fusion_network(tech_tensor, sent_tensor, news_tensor, social_tensor)
            # prediction shape is (batch,), make sure we match reward_tensor
            pred = prediction.unsqueeze(0) if prediction.dim() == 0 else prediction
            loss = self.criterion(pred.squeeze(), reward_tensor)
            loss.backward()
            self.optimizer.step()

            logger.debug(f"Fusion network updated - Loss: {loss.item():.6f}, Reward: {reward:.4f}")
        except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
            logger.exception(f"Error updating fusion network: {e}")

    async def make_multimodal_prediction(self, multimodal_data: MultiModalData) -> dict[str, Any]:
        try:
            self.fusion_network.eval()
            with torch.no_grad():
                tech_tensor = torch.as_tensor(multimodal_data.technical_features, dtype=torch.float32).unsqueeze(0).to(self.device)
                sent_tensor = torch.as_tensor(multimodal_data.sentiment_features, dtype=torch.float32).unsqueeze(0).to(self.device)
                news_tensor = torch.as_tensor(multimodal_data.news_features, dtype=torch.float32).unsqueeze(0).to(self.device)
                social_tensor = torch.as_tensor(multimodal_data.social_features, dtype=torch.float32).unsqueeze(0).to(self.device)

                prediction = self.fusion_network(tech_tensor, sent_tensor, news_tensor, social_tensor)
                # ensure scalar
                prediction_value = float(prediction.detach().cpu().numpy().ravel()[0]) if prediction.dim() > 0 else float(prediction.item())

                if prediction_value > 0.1:
                    signal = "BUY"
                    confidence = min(abs(prediction_value), 1.0)
                elif prediction_value < -0.1:
                    signal = "SELL"
                    confidence = min(abs(prediction_value), 1.0)
                else:
                    signal = "HOLD"
                    confidence = max(0.0, 1.0 - abs(prediction_value))

                try:
                    from backend.services.confidence_normalizer import ConfidenceNormalizer

                    conf_val = ConfidenceNormalizer.normalize(float(confidence))
                except Exception as ex:
                    logger.debug("ConfidenceNormalizer unavailable: %s", ex)
                    conf_val = float(confidence)
                return {
                    "signal": signal,
                    "confidence": conf_val,
                    "prediction_value": prediction_value,
                    "technical_weight": 0.4,
                    "sentiment_weight": 0.3,
                    "news_weight": 0.2,
                    "social_weight": 0.1,
                }
        except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
            logger.exception(f"Error making multimodal prediction: {e}")
            return {"signal": "HOLD", "confidence": 0.0, "prediction_value": 0.0}

    async def store_multimodal_learning(self, multimodal_data: MultiModalData, trade_result: dict | None):
        try:
            learning_record: dict[str, Any] = {
                "symbol": multimodal_data.symbol,
                "timestamp": multimodal_data.timestamp,
                "technical_features_count": int(np.size(multimodal_data.technical_features)),
                "sentiment_features_count": int(np.size(multimodal_data.sentiment_features)),
                "news_features_count": int(np.size(multimodal_data.news_features)),
                "social_features_count": int(np.size(multimodal_data.social_features)),
                "has_trade_result": trade_result is not None,
            }

            if trade_result:
                learning_record["trade_pnl"] = float(trade_result.get("pnl", 0.0) or 0.0)
                raw_tc = float(trade_result.get("confidence_score", 0.0) or 0.0)
                try:
                    from backend.services.confidence_normalizer import ConfidenceNormalizer

                    learning_record["trade_confidence"] = ConfidenceNormalizer.normalize(raw_tc)
                except Exception as ex:
                    logger.debug("ConfidenceNormalizer unavailable: %s", ex)
                    learning_record["trade_confidence"] = raw_tc

            key = f"multimodal_learning:{datetime.now(tz=timezone.utc).strftime('%Y%m%d')}"
            try:
                self.redis_client.lpush(key, json.dumps(learning_record))
                self.redis_client.expire(key, 86400 * 7)
            except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
                logger.debug(f"Redis unavailable or error storing learning record: {e}")

            self.learning_history.append(learning_record)
            if len(self.learning_history) > self.max_history:
                self.learning_history = self.learning_history[-self.max_history :]
        except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
            logger.exception(f"Error storing multimodal learning data: {e}")

    async def get_learning_statistics(self) -> dict[str, Any]:
        try:
            return {
                "total_learning_samples": len(self.learning_history),
                "symbols_learned": len({record["symbol"] for record in self.learning_history}),
                "samples_with_trades": sum(1 for record in self.learning_history if record["has_trade_result"]),
                "avg_technical_features": float(np.mean([record["technical_features_count"] for record in self.learning_history])) if self.learning_history else 0.0,
                "avg_sentiment_features": float(np.mean([record["sentiment_features_count"] for record in self.learning_history])) if self.learning_history else 0.0,
                "fusion_network_parameters": int(sum(p.numel() for p in self.fusion_network.parameters())),
                "device": str(self.device),
                "last_update": self.learning_history[-1]["timestamp"] if self.learning_history else None,
            }
        except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
            logger.exception(f"Error getting learning statistics: {e}")
            return {"error": str(e)}


# Multimodal learner state - using dict to avoid global keyword
_multimodal_learner_state: dict[str, MultiModalAILearner | None] = {"instance": None}


def get_multimodal_learner(redis_client: redis.Redis) -> MultiModalAILearner:
    if _multimodal_learner_state["instance"] is None:
        _multimodal_learner_state["instance"] = MultiModalAILearner(redis_client)
    return _multimodal_learner_state["instance"]
