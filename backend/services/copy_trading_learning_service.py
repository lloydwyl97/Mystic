"""
Copy Trading Learning Service
AI learns from successful social traders by analyzing their strategies and outcomes
Identifies profitable patterns and incorporates them into AI decision-making
"""

import asyncio
import logging
from datetime import datetime, timezone
from typing import Any

try:
    from backend.services.data_collector import DataCollector
except (ImportError, ModuleNotFoundError, AttributeError, ValueError, TypeError, RuntimeError):
    DataCollector = None  # type: ignore[assignment,misc]

try:
    from backend.services.social_trading_service import get_social_trading_service
except (ImportError, ModuleNotFoundError, AttributeError, ValueError, TypeError, RuntimeError):
    get_social_trading_service = None  # type: ignore[assignment]

from backend.services.task_manager import task_manager

logger = logging.getLogger(__name__)


class CopyTradingLearningService:
    """
    Learns from top social traders to improve AI predictions

    Strategy:
    1. Monitor top performers on social trading platform
    2. Analyze their trade patterns and decisions
    3. Extract features from successful trades
    4. Identify common patterns in winning trades
    5. Feed these patterns to AI as training samples
    6. AI learns what makes trades successful
    7. Apply learned patterns to own trading decisions
    """

    def __init__(self) -> None:
        self.enabled = True
        self.min_trader_win_rate = 60.0  # Only learn from traders with 60%+ win rate
        self.min_trader_trades = 50  # Trader must have 50+ trades for statistical significance

        # Tracked traders and their performance
        self.tracked_traders: dict[int, dict[str, Any]] = {}
        self.learned_patterns: list[dict[str, Any]] = []

        # Learning stats
        self.learning_stats = {
            "traders_tracked": 0,
            "trades_analyzed": 0,
            "patterns_learned": 0,
            "training_samples_created": 0,
            "models_updated": 0,
            "last_learning_time": None,
        }

        logger.info("Copy Trading Learning Service initialized")

    async def start(self) -> None:
        """Start the copy trading learning service"""
        logger.info("Copy Trading Learning Service started - learning from top social traders")

        # Start background learning loop
        task = await task_manager.create_task(self._learning_loop(), name="copy_trading_learning_service:learning_loop")
        # Store task reference if class has task tracking
        if hasattr(self, "_tasks"):
            self._tasks.append(task)
        elif not hasattr(self, "_tasks"):
            self._tasks: list[asyncio.Task[Any]] = []
            self._tasks.append(task)

    async def _learning_loop(self) -> None:
        """Background loop to continuously learn from social traders"""
        while self.enabled:
            try:
                await self._analyze_top_traders()
                await asyncio.sleep(300)  # Run every 5 minutes
            except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
                logger.exception(f"Error in copy trading learning loop: {e}")
                await asyncio.sleep(60)
            except (asyncio.CancelledError, KeyboardInterrupt):
                logger.info("Copy trading learning loop cancelled")
                break

    async def _analyze_top_traders(self) -> None:
        """Analyze top traders and learn from their patterns"""
        try:
            if not get_social_trading_service:
                logger.debug("Social trading service not available")
                return

            social_service = get_social_trading_service()
            if not social_service:
                logger.debug("Social trading service not available")
                return

            # Get top performers
            leaderboard = await social_service.get_leaderboard(_period="monthly", category="pnl", limit=20)

            # Filter for high-quality traders
            quality_traders = [trader for trader in leaderboard if trader.get("win_rate", 0) >= self.min_trader_win_rate and trader.get("total_trades", 0) >= self.min_trader_trades]

            logger.info(f"Found {len(quality_traders)} quality traders to learn from")

            # Analyze each trader's patterns
            for trader in quality_traders:
                await self._learn_from_trader(trader)

        except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
            logger.exception(f"Failed to analyze top traders: {e}")

    async def _learn_from_trader(self, trader: dict[str, Any]) -> None:
        """
        Analyze a single trader's patterns and extract learning
        """
        trader_id = trader.get("user_id") or trader.get("trader_id")
        if not trader_id:
            return

        # Track this trader
        self.tracked_traders[trader_id] = {
            "win_rate": trader.get("win_rate", 0),
            "total_trades": trader.get("total_trades", 0),
            "pnl": trader.get("pnl", 0),
            "tracked_since": datetime.now(timezone.utc).isoformat(),
        }

        self.learning_stats["traders_tracked"] = len(self.tracked_traders)

        # Analyze their trading patterns
        patterns = await self._extract_trader_patterns(trader)

        if patterns:
            # Store patterns for AI learning
            for pattern in patterns:
                self.learned_patterns.append(pattern)
                self.learning_stats["patterns_learned"] += 1

            # Convert patterns to training samples
            training_samples = await self._patterns_to_training_samples(patterns)

            if training_samples:
                # Feed to AI learner
                await self._feed_to_ai_learner(training_samples)

    async def _extract_trader_patterns(self, trader: dict[str, Any]) -> list[dict[str, Any]]:
        """
        Extract common patterns from a successful trader's decisions

        Patterns to identify:
        - Entry timing (technical indicator values when they buy)
        - Exit timing (when they take profit/cut losses)
        - Position sizing
        - Symbol preferences
        - Time of day patterns
        - Market condition preferences
        """
        patterns = []

        # Pattern 1: Entry conditions
        # When does this trader typically enter trades?
        entry_pattern = {
            "type": "entry_timing",
            "trader_id": trader.get("user_id") or trader.get("trader_id"),
            "win_rate": trader.get("win_rate", 0),
            "characteristics": {
                "preferred_indicators": [],  # Would extract from trade history
                "typical_confidence": 0.7,  # Would calculate from history
                "market_conditions": "trending",  # Would analyze from history
            },
            "weight": trader.get("win_rate", 0) / 100.0,  # Weight by win rate
        }
        patterns.append(entry_pattern)

        # Pattern 2: Risk management
        # How does this trader manage risk?
        risk_pattern = {
            "type": "risk_management",
            "trader_id": trader.get("user_id") or trader.get("trader_id"),
            "characteristics": {
                "avg_position_size": 0.05,  # Would calculate from history
                "risk_pct": 0.02,  # Would extract from trades
                "target_pct": 0.05,  # Would extract from trades
            },
            "weight": trader.get("win_rate", 0) / 100.0,
        }
        patterns.append(risk_pattern)

        return patterns

    async def _patterns_to_training_samples(self, patterns: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Convert learned patterns into training samples for AI"""
        training_samples = []

        for pattern in patterns:
            if pattern["type"] == "entry_timing":
                # Create positive training sample from successful entry pattern
                sample = {
                    "pattern_type": "entry",
                    "weight": pattern["weight"],
                    "features": pattern["characteristics"],
                    "label": 1,  # Successful pattern
                    "source": "copy_trading_learning",
                }
                training_samples.append(sample)

        self.learning_stats["training_samples_created"] += len(training_samples)

        return training_samples

    async def _feed_to_ai_learner(self, training_samples: list[dict[str, Any]]) -> None:
        """Feed learned patterns to AI learner"""
        try:
            if not DataCollector:
                logger.debug("DataCollector not available")
                return

            data_collector = DataCollector()

            for sample in training_samples:
                await data_collector.collect_sample(sample)

            logger.info(f"Fed {len(training_samples)} copy trading patterns to AI learner")

        except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
            logger.exception(f"Failed to feed patterns to AI: {e}")

    async def get_top_traders_to_copy(self, limit: int = 5) -> list[dict[str, Any]]:
        """
        Get recommended traders to copy based on learned patterns
        """
        try:
            if not get_social_trading_service:
                return []

            social_service = get_social_trading_service()
            if not social_service:
                return []

            leaderboard = await social_service.get_leaderboard(_period="monthly", category="pnl", limit=50)

            # Filter and rank
            recommendations = [trader for trader in leaderboard if trader.get("win_rate", 0) >= self.min_trader_win_rate and trader.get("total_trades", 0) >= self.min_trader_trades]

            # Sort by win rate and return top N
            recommendations.sort(key=lambda t: t.get("win_rate", 0), reverse=True)

            return recommendations[:limit]

        except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
            logger.exception(f"Failed to get top traders: {e}")
            return []

    async def get_learning_stats(self) -> dict[str, Any]:
        """Get copy trading learning statistics"""
        return {
            "enabled": self.enabled,
            "learning_stats": self.learning_stats,
            "tracked_traders_count": len(self.tracked_traders),
            "learned_patterns_count": len(self.learned_patterns),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }


# Singleton instance
copy_trading_learning_service = CopyTradingLearningService()
