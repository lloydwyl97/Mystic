"""
Enhanced AI Learning System
Real-time learning with advanced reward engineering and multi-modal integration
"""

from __future__ import annotations

import json
import logging
from collections import deque
from datetime import datetime, timezone
from typing import Any

import numpy as np

try:
    import torch
    from torch import nn, optim

    TORCH_AVAILABLE = True
except ImportError:
    torch = None
    nn = None
    optim = None
    TORCH_AVAILABLE = False

from sklearn.preprocessing import StandardScaler

try:
    import redis

    REDIS_AVAILABLE = True
except ImportError:
    redis = None
    REDIS_AVAILABLE = False

logger = logging.getLogger(__name__)

FEATURE_DIM = 32  # fixed-length feature vector to keep training shapes consistent


class AdvancedRewardCalculator:
    def __init__(self) -> None:
        self.volatility_window = 24
        self.consistency_window = 10
        self.market_impact_weight = 0.3
        self.risk_penalty_weight = 2.0

    def calculate_comprehensive_reward(self, trade_result: dict[str, Any]) -> float:
        try:
            position_size = max(float(trade_result.get("position_size", 1.0)), 0.01)
            profit_reward = float(trade_result.get("pnl", 0.0)) / position_size

            max_drawdown = float(trade_result.get("max_drawdown", 0.0))
            risk_penalty = -abs(max_drawdown) * self.risk_penalty_weight

            hold_time_hours = max(float(trade_result.get("hold_time_hours", 0.1)), 0.1)
            time_bonus = min(1.0 / hold_time_hours, 2.0) if profit_reward > 0 else -abs(profit_reward) * hold_time_hours * 0.1

            volatility_adjustment = self.get_volatility_adjustment(trade_result)
            consistency_bonus = self.calculate_consistency_bonus(trade_result)
            execution_quality = self.calculate_execution_quality(trade_result)

            total_reward = profit_reward + risk_penalty + time_bonus + volatility_adjustment + consistency_bonus + execution_quality
            return float(np.tanh(total_reward))
        except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
            logger.exception(f"Error calculating reward: {e}")
            return 0.0

    def get_volatility_adjustment(self, trade_result: dict[str, Any]) -> float:
        try:
            market_volatility = float(trade_result.get("market_volatility", 0.0) or 0.0)
            pnl = float(trade_result.get("pnl", 0.0))
            if pnl > 0:
                return market_volatility * self.market_impact_weight
            return market_volatility * 0.5 * self.market_impact_weight
        except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError):
            return 0.0

    def calculate_consistency_bonus(self, trade_result: dict[str, Any]) -> float:
        try:
            recent_trades = trade_result.get("recent_trades_pnl", [])
            if len(recent_trades) < 3:
                return 0.0

            recent_pnl = np.array(recent_trades[-self.consistency_window :], dtype=float)
            if recent_pnl.size == 0:
                return 0.0

            positive_ratio = float(np.sum(recent_pnl > 0) / len(recent_pnl))
            volatility_penalty = float(np.std(recent_pnl) * -0.1)
            return float(positive_ratio * 0.2 + volatility_penalty)
        except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError):
            return 0.0

    def calculate_execution_quality(self, trade_result: dict[str, Any]) -> float:
        try:
            expected_price = float(trade_result.get("expected_price", 0.0))
            actual_price = float(trade_result.get("actual_price", 0.0))
            if expected_price > 0:
                slippage = abs(actual_price - expected_price) / expected_price
                slippage_penalty = -float(slippage) * 0.5
            else:
                slippage_penalty = 0.0
            timing_score = float(trade_result.get("timing_score", 0.0))
            return float(slippage_penalty + (timing_score * 0.1))
        except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError):
            return 0.0


class RealTimeLearner:
    def __init__(self, redis_client: redis.Redis) -> None:
        self.redis_client = redis_client
        self.reward_calculator = AdvancedRewardCalculator()
        self.learning_buffer: deque[dict[str, Any]] = deque(maxlen=1000)
        self.model_cache: dict[str, nn.Module] = {}
        self.learning_rate = 0.001
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.performance_history: deque[dict[str, Any]] = deque(maxlen=100)
        self.last_model_update = datetime.now(timezone.utc)

        self.min_learning_rate = 0.0001
        self.max_learning_rate = 0.01
        self.performance_window = 20

        logger.info("RealTimeLearner initialized")

    async def learn_from_trade(self, trade_result: dict[str, Any]) -> dict[str, Any]:
        try:
            reward = self.reward_calculator.calculate_comprehensive_reward(trade_result)
            features = await self.extract_trade_features(trade_result)

            learning_sample = {
                "features": features,
                "reward": float(reward),
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "trade_id": str(trade_result.get("trade_id", "unknown")),
            }
            self.learning_buffer.append(learning_sample)

            self.performance_history.append(
                {
                    "reward": float(reward),
                    "pnl": float(trade_result.get("pnl", 0.0)),
                    "timestamp": learning_sample["timestamp"],
                }
            )

            update_result = await self.update_models_realtime(learning_sample)
            await self.adapt_learning_parameters()
            await self.store_learning_data(learning_sample)

            logger.info(
                "Real-time learning completed - Reward: %.4f, Features: %d, Update: %s",
                reward,
                len(features),
                update_result.get("status", "unknown"),
            )

            return {
                "status": "success",
                "reward": float(reward),
                "features_count": len(features),
                "model_update": update_result,
                "learning_rate": float(self.learning_rate),
                "buffer_size": len(self.learning_buffer),
            }
        except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
            logger.exception(f"Error in real-time learning: {e}")
            return {"status": "error", "error": str(e)}

    async def extract_trade_features(self, trade_result: dict[str, Any]) -> np.ndarray:
        try:
            feats: list[float] = []

            feats.extend(
                [
                    float(trade_result.get("position_size", 0.0)),
                    float(trade_result.get("hold_time_hours", 0.0)),
                    float(trade_result.get("pnl", 0.0)),
                    float(trade_result.get("max_drawdown", 0.0)),
                    float(trade_result.get("confidence_score", 0.0)),
                ],
            )

            feats.extend(
                [
                    float(trade_result.get("market_volatility", 0.0) or 0.0),
                    float(trade_result.get("market_trend", 0.0)),
                    float(trade_result.get("volume_ratio", 1.0)),
                    float(trade_result.get("rsi", 50.0)) / 100.0,
                    float(trade_result.get("macd_signal", 0.0)),
                ],
            )

            strategy_name = str(trade_result.get("strategy_name", "unknown"))
            strategy_features = await self.get_strategy_features(strategy_name)
            feats.extend(strategy_features)

            timestamp = str(trade_result.get("timestamp", datetime.now(timezone.utc).isoformat()))
            feats.extend(self.extract_time_features(timestamp))

            feats.extend(self.get_recent_performance_features())

            arr = np.array(feats, dtype=np.float32)
            arr = np.nan_to_num(arr, nan=0.0, posinf=1.0, neginf=-1.0)

            if arr.size < FEATURE_DIM:
                arr = np.pad(
                    arr,
                    (0, FEATURE_DIM - arr.size),
                    mode="constant",
                    constant_values=0.0,
                )
            elif arr.size > FEATURE_DIM:
                arr = arr[:FEATURE_DIM]

            return arr.astype(np.float32, copy=False)
        except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
            logger.exception(f"Error extracting features: {e}")
            return np.zeros(FEATURE_DIM, dtype=np.float32)

    async def get_strategy_features(self, strategy_name: str) -> list[float]:
        try:
            strategy_key = f"strategy_performance:{strategy_name}"
            strategy_data = self.redis_client.get(strategy_key)
            if strategy_data:
                strategy_text = strategy_data.decode("utf-8") if isinstance(strategy_data, (bytes, bytearray)) else str(strategy_data)
                data = json.loads(strategy_text)
                return [
                    float(data.get("win_rate", 0.5)),
                    float(data.get("avg_profit", 0.0)),
                    float(data.get("max_drawdown", 0.0)),
                    float(data.get("sharpe_ratio", 0.0)),
                    float(data.get("total_trades", 0.0)) / 1000.0,
                ]
            else:
                result = [0.5, 0.0, 0.0, 0.0, 0.0]
        except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError):
            return [0.5, 0.0, 0.0, 0.0, 0.0]
        else:
            return result

    def extract_time_features(self, timestamp_str: str) -> list[float]:
        try:
            dt = datetime.fromisoformat(timestamp_str.replace("Z", "+00:00"))
            hour_sin = float(np.sin(2 * np.pi * dt.hour / 24))
            hour_cos = float(np.cos(2 * np.pi * dt.hour / 24))
            day_sin = float(np.sin(2 * np.pi * dt.weekday() / 7))
            day_cos = float(np.cos(2 * np.pi * dt.weekday() / 7))
            is_market_hours = 1.0 if 9 <= dt.hour <= 16 else 0.0
            is_weekend = 1.0 if dt.weekday() >= 5 else 0.0
            return [
                hour_sin,
                hour_cos,
                day_sin,
                day_cos,
                float(is_market_hours),
                float(is_weekend),
            ]
        except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError):
            return [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

    def get_recent_performance_features(self) -> list[float]:
        try:
            if len(self.performance_history) < 5:
                return [0.0, 0.0, 0.0, 0.0]
            recent = list(self.performance_history)[-10:]
            recent_rewards = [float(p["reward"]) for p in recent]
            recent_pnl = [float(p["pnl"]) for p in recent]
            mean_r = float(np.mean(recent_rewards))
            std_r = float(np.std(recent_rewards))
            mean_p = float(np.mean(recent_pnl))
            win_rate = float(len([r for r in recent_rewards if r > 0]) / len(recent_rewards))
        except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError):
            return [0.0, 0.0, 0.0, 0.0]
        else:
            return [mean_r, std_r, mean_p, win_rate]

    async def update_models_realtime(self, _learning_sample: dict[str, Any]) -> dict[str, Any]:
        try:
            if len(self.learning_buffer) < 10:
                return {
                    "status": "insufficient_data",
                    "buffer_size": len(self.learning_buffer),
                }

            recent_samples = list(self.learning_buffer)[-32:]
            features_batch = np.stack([np.asarray(s["features"], dtype=np.float32) for s in recent_samples])
            rewards_batch = np.asarray([float(s["reward"]) for s in recent_samples], dtype=np.float32)

            scaler = StandardScaler()
            features_normalized = scaler.fit_transform(features_batch)

            model_update_count = 0
            for model_name in (
                "strategy_selector",
                "risk_predictor",
                "timing_optimizer",
            ):
                model = self.model_cache.get(model_name)
                if model is not None:
                    try:
                        await self.update_neural_model(model, features_normalized, rewards_batch)
                        model_update_count += 1
                    except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
                        logger.warning("Failed to update %s: %s", model_name, e)

            if model_update_count > 0:
                self.last_model_update = datetime.now(timezone.utc)

            return {
                "status": "success",
                "models_updated": model_update_count,
                "samples_used": len(recent_samples),
                "learning_rate": float(self.learning_rate),
            }
        except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
            logger.exception(f"Error updating models: {e}")
            return {"status": "error", "error": str(e)}

    async def update_neural_model(self, model: nn.Module, features: np.ndarray, rewards: np.ndarray) -> None:
        try:
            model.train()
            X_tensor = torch.as_tensor(features, dtype=torch.float32, device=self.device)
            y_tensor = torch.as_tensor(rewards, dtype=torch.float32, device=self.device)

            optimizer = optim.Adam(model.parameters(), lr=self.learning_rate)
            criterion = nn.MSELoss()

            optimizer.zero_grad()
            preds = model(X_tensor)
            loss = criterion(preds.squeeze(), y_tensor)
            loss.backward()
            optimizer.step()
            self.last_model_update = datetime.now(timezone.utc)
            logger.debug("Model updated - Loss: %.6f", float(loss.item()))
        except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
            logger.exception(f"Error updating neural model: {e}")
            raise

    async def adapt_learning_parameters(self) -> None:
        try:
            if len(self.performance_history) < self.performance_window:
                return
            recent = list(self.performance_history)[-self.performance_window :]
            rewards = np.asarray([float(p["reward"]) for p in recent], dtype=np.float32)
            trend = float(np.mean(rewards))
            vol = float(np.std(rewards))

            if trend < -0.1:
                self.learning_rate = min(self.learning_rate * 1.1, self.max_learning_rate)
            elif trend > 0.1:
                self.learning_rate = max(self.learning_rate * 0.95, self.min_learning_rate)

            if vol > 0.3:
                self.learning_rate = max(self.learning_rate * 0.9, self.min_learning_rate)

            logger.debug(
                "Learning rate adapted: %.6f (trend=%.4f, vol=%.4f)",
                self.learning_rate,
                trend,
                vol,
            )
        except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
            logger.exception(f"Error adapting learning parameters: {e}")

    async def store_learning_data(self, learning_sample: dict[str, Any]) -> None:
        try:
            learning_key = f"realtime_learning:{datetime.now(timezone.utc).strftime('%Y%m%d')}"
            learning_data = {
                "timestamp": learning_sample["timestamp"],
                "reward": float(learning_sample["reward"]),
                "features_count": len(learning_sample["features"]),
                "trade_id": learning_sample["trade_id"],
            }

            self.redis_client.lpush(learning_key, json.dumps(learning_data))
            self.redis_client.expire(learning_key, 86400 * 7)

            stats_key = "realtime_learning_stats"
            current_stats = self.redis_client.get(stats_key)
            if current_stats:
                stats_text = current_stats.decode("utf-8") if isinstance(current_stats, (bytes, bytearray)) else str(current_stats)
                stats = json.loads(stats_text)
            else:
                stats = {"total_samples": 0, "avg_reward": 0.0, "last_update": None}

            stats["total_samples"] = int(stats.get("total_samples", 0)) + 1
            prev_total = max(stats["total_samples"] - 1, 0)
            prev_avg = float(stats.get("avg_reward", 0.0))
            stats["avg_reward"] = float((prev_avg * prev_total + float(learning_sample["reward"])) / stats["total_samples"])
            stats["last_update"] = learning_sample["timestamp"]

            self.redis_client.set(stats_key, json.dumps(stats))
        except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
            logger.exception(f"Error storing learning data: {e}")

    async def get_learning_stats(self) -> dict[str, Any]:
        try:
            stats_key = "realtime_learning_stats"
            stats_data = self.redis_client.get(stats_key)
            if stats_data:
                stats_text = stats_data.decode("utf-8") if isinstance(stats_data, (bytes, bytearray)) else str(stats_data)
                stats = json.loads(stats_text)
            else:
                stats = {"total_samples": 0, "avg_reward": 0.0, "last_update": None}

            stats["current_session"] = {
                "buffer_size": len(self.learning_buffer),
                "learning_rate": float(self.learning_rate),
                "performance_samples": len(self.performance_history),
                "last_model_update": self.last_model_update.isoformat(),
            }
        except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
            logger.exception(f"Error getting learning stats: {e}")
            return {"error": str(e)}
        else:
            return stats


# Enhanced learner state - using dict to avoid global keyword
_enhanced_learner_state: dict[str, RealTimeLearner | None] = {"instance": None}


def get_enhanced_learner(redis_client: redis.Redis) -> RealTimeLearner:
    if _enhanced_learner_state["instance"] is None:
        _enhanced_learner_state["instance"] = RealTimeLearner(redis_client)
    return _enhanced_learner_state["instance"]
