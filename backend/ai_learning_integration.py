"""
AI Learning Integration
Integrates enhanced AI learning with existing trading systems.

Repairs & hardening:
- Added Optional typing and explicit return types
- Gracefully handle missing enhanced/multimodal learner modules with safe fallbacks
- Defensive guards around None / unexpected input shapes
- Use timezone-aware timestamps
- More robust logging for easier debugging
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any

from backend.ai_enhanced_learning import get_enhanced_learner  # type: ignore[import-not-found]

try:
    from backend.ai_multimodal_learning import get_multimodal_learner  # type: ignore[import-not-found]
except ImportError:
    get_multimodal_learner = None

logger = logging.getLogger(__name__)


class AILearningIntegration:
    """Integrates enhanced AI learning with existing trading systems"""

    def __init__(self, redis_client: Any) -> None:
        self.redis_client = redis_client
        self.enhanced_learner = get_enhanced_learner(redis_client)
        self.multimodal_learner = get_multimodal_learner(redis_client) if get_multimodal_learner is not None else None

    async def integrate_with_autobuy_system(
        self,
        strategy_results: dict[str, dict[str, Any]] | None = None,
        trade_results: dict[str, dict[str, Any]] | None = None,
    ) -> None:
        """Integrate with autobuy system for real-time learning"""
        try:
            strategy_results = strategy_results or {}

            # Learn from strategy results (simulated trade outcomes)
            for symbol, result in list(strategy_results.items()):
                action = (result or {}).get("action")
                if action in ("BUY", "SELL"):
                    # Create simulated trade result for learning
                    trade_result = {
                        "trade_id": f"strategy_{symbol}_{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}",
                        "symbol": symbol,
                        "pnl": float(result.get("expected_profit", 0.0) or 0.0),
                        "position_size": float(result.get("position_size", 1.0) or 1.0),
                        "hold_time_hours": 1.0,
                        "confidence_score": float(result.get("confidence", 0.0) or 0.0),
                        "strategy_name": result.get("strategy", "unknown") or "unknown",
                        "market_volatility": float(result.get("volatility") or 0.0),
                        "rsi": float(result.get("rsi", 50.0) or 50.0),
                        "macd_signal": float(result.get("macd", 0.0) or 0.0),
                    }
                    # Defensive: ensure learner has coroutine
                    learn_coro = getattr(self.enhanced_learner, "learn_from_trade", None)
                    if learn_coro:
                        await learn_coro(trade_result)

            # Learn from actual trade results if provided
            if trade_results:
                for _trade_id, trade_data in list(trade_results.items()):
                    if isinstance(trade_data, dict):
                        learn_coro = getattr(self.enhanced_learner, "learn_from_trade", None)
                        if learn_coro:
                            await learn_coro(trade_data)

            logger.debug("AI learning integration completed")

        except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
            logger.exception(f"Error in AI learning integration: {e}")

    async def integrate_with_agents(
        self,
        agent_predictions: dict[str, Any] | None = None,
        market_data: dict[str, Any] | None = None,
    ) -> None:
        """Integrate with AI agents for multi-modal learning"""
        try:
            market_data = market_data or {}
            # agent_predictions currently unused but kept for interface completeness
            _ = agent_predictions

            for symbol, data in list(market_data.items()):
                # Simulate news and social data (real implementation would supply these)
                news_data: list[dict[str, Any]] = []
                social_data: list[dict[str, Any]] = []

                # Learn from multi-modal data
                learn_coro = getattr(self.multimodal_learner, "learn_from_multimodal_data", None)
                if learn_coro:
                    await learn_coro(
                        symbol=str(symbol),
                        market_data=(data if isinstance(data, dict) else {"raw": data}),
                        news_data=news_data,
                        social_data=social_data,
                    )

            logger.debug("Multi-modal learning integration completed")

        except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
            logger.exception(f"Error in multi-modal integration: {e}")

    async def get_learning_insights(self) -> dict[str, Any]:
        """Get insights from AI learning systems"""
        try:
            # Fetch stats with safe defaults
            realtime_stats = {}
            multimodal_stats = {}
            get_realtime_stats = getattr(self.enhanced_learner, "get_learning_stats", None)
            if get_realtime_stats:
                realtime_stats = await get_realtime_stats()
            get_multimodal_stats = getattr(self.multimodal_learner, "get_learning_statistics", None)
            if get_multimodal_stats:
                multimodal_stats = await get_multimodal_stats()

            return {
                "learning_performance": {
                    "realtime_samples": int(realtime_stats.get("total_samples", 0) or 0),
                    "multimodal_samples": int(multimodal_stats.get("total_learning_samples", 0) or 0),
                    "average_reward": float(realtime_stats.get("avg_reward", 0.0) or 0.0),
                    "learning_efficiency": self.calculate_learning_efficiency(realtime_stats),
                },
                "recommendations": self.generate_recommendations(realtime_stats, multimodal_stats),
                "system_status": "active",
                "last_update": datetime.now(timezone.utc).isoformat(),
            }

        except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
            logger.exception(f"Error getting learning insights: {e}")
            return {"error": str(e)}

    def calculate_learning_efficiency(self, stats: dict[str, Any]) -> float:
        """Calculate overall learning efficiency"""
        try:
            total_samples = int(stats.get("total_samples", 0) or 0)
            avg_reward = float(stats.get("avg_reward", 0.0) or 0.0)

            if total_samples <= 0:
                return 0.0

            # Efficiency based on samples and average reward
            sample_efficiency = min(total_samples / 1000.0, 1.0)  # Max efficiency at 1000 samples
            # Normalize reward (assumes reward in [-1, 1])
            reward_efficiency = max(0.0, min((avg_reward + 1.0) / 2.0, 1.0))

            return float((sample_efficiency + reward_efficiency) / 2.0)
        except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
            logger.debug(f"Efficiency calculation fallback due to: {e}")
            return 0.0

    def generate_recommendations(self, realtime_stats: dict[str, Any], multimodal_stats: dict[str, Any]) -> list[dict[str, Any]]:
        """Generate recommendations for improving AI learning"""
        recommendations: list[dict[str, Any]] = []

        try:
            avg_reward = float(realtime_stats.get("avg_reward", 0.0) or 0.0)
            total_samples = int(realtime_stats.get("total_samples", 0) or 0)

            if avg_reward < 0.05:
                recommendations.append(
                    {
                        "type": "performance",
                        "message": "Consider adjusting trading strategies - low average reward",
                        "priority": "high",
                    },
                )

            if total_samples < 50:
                recommendations.append(
                    {
                        "type": "data",
                        "message": "Increase trading frequency (or expand data sources) to improve learning",
                        "priority": "medium",
                    },
                )

            multimodal_samples = int(multimodal_stats.get("total_learning_samples", 0) or 0)
            if multimodal_samples < 20:
                recommendations.append(
                    {
                        "type": "integration",
                        "message": "Enable/ingest news & sentiment signals to enrich learning",
                        "priority": "low",
                    },
                )
        except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
            logger.exception(f"Error generating recommendations: {e}")

        return recommendations


# Global integration instance (singleton-style)
# AI learning integration state - using dict to avoid global keyword
_ai_learning_integration_state: dict[str, AILearningIntegration | None] = {"instance": None}


def get_ai_learning_integration(redis_client: Any) -> AILearningIntegration:
    """Get or create AI learning integration instance"""
    if _ai_learning_integration_state["instance"] is None:
        _ai_learning_integration_state["instance"] = AILearningIntegration(redis_client)
    return _ai_learning_integration_state["instance"]
