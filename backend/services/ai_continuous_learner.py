#!/usr/bin/env python3
"""
Simple AI Continuous Learning System

Tracks AI performance metrics and stores learning data in Redis.
"""

import asyncio
import json
import logging
import time
from datetime import datetime, timezone
from typing import Any

logger = logging.getLogger(__name__)

# Lazy import to avoid circular dependencies
_redis_client = None

# Import here to avoid circular dependencies
try:
    from backend.config.redis_config import get_redis_client
except ImportError:
    get_redis_client = None

try:
    from backend.services.live_strategy_contracts import REDIS_ML_SIGNAL_SCAN_PATTERN
except ImportError:
    REDIS_ML_SIGNAL_SCAN_PATTERN = "ai_signal:*:*"


def get_redis():
    global _redis_client
    if _redis_client is None:
        try:
            if get_redis_client is not None:
                _redis_client = get_redis_client()
        except Exception as e:
            logger.warning(f"Could not get Redis client: {e}")
            return None
    return _redis_client


class LearningMetrics:
    """Tracks AI learning performance"""

    def __init__(self):
        self.total_predictions = 0
        self.correct_predictions = 0
        self.learning_sessions = 0
        self.last_learning_time = 0
        self.accuracy_history = []
        self.confidence_trends = []

    def record_prediction(self, prediction: str, confidence: float, actual: str):
        """Record a prediction and its outcome"""
        self.total_predictions += 1
        if prediction == actual:
            self.correct_predictions += 1

        accuracy = self.correct_predictions / self.total_predictions if self.total_predictions > 0 else 0.0
        self.accuracy_history.append(accuracy)
        self.confidence_trends.append(confidence)

        # Keep only last 100 entries
        if len(self.accuracy_history) > 100:
            self.accuracy_history = self.accuracy_history[-100:]
            self.confidence_trends = self.confidence_trends[-100:]

    def get_accuracy(self) -> float:
        """Get current accuracy"""
        if self.total_predictions == 0:
            return 0.0
        return self.correct_predictions / self.total_predictions

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for Redis storage"""
        return {
            "total_predictions": self.total_predictions,
            "correct_predictions": self.correct_predictions,
            "accuracy": self.get_accuracy(),
            "learning_sessions": self.learning_sessions,
            "last_learning_time": self.last_learning_time,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }


class ContinuousLearner:
    """Simple AI continuous learning system"""

    def __init__(self):
        self.metrics = LearningMetrics()
        self.running = False
        self.learning_interval = 300  # 5 minutes
        self._feedback_tasks = set()  # Track feedback storage tasks
        logger.info("ContinuousLearner initialized")

    async def start(self):
        """Start the continuous learning system"""
        if self.running:
            return

        self.running = True
        logger.info("AI Continuous Learner starting...")
        self._learning_task = asyncio.create_task(self.run_continuous_learning())

    async def stop(self):
        """Stop the continuous learning system and cleanup tasks"""
        if not self.running:
            return

        self.running = False
        logger.info("Stopping AI Continuous Learner...")

        # Cancel learning task
        if hasattr(self, "_learning_task") and self._learning_task:
            self._learning_task.cancel()
            with asyncio.suppress(asyncio.CancelledError):
                await self._learning_task

        # Cancel all feedback tasks
        if self._feedback_tasks:
            logger.info(f"Cancelling {len(self._feedback_tasks)} feedback tasks...")
            for task in self._feedback_tasks:
                if not task.done():
                    task.cancel()
            # Wait for feedback tasks to complete
            if self._feedback_tasks:
                await asyncio.gather(*self._feedback_tasks, return_exceptions=True)

        logger.info("AI Continuous Learner stopped")

    async def run_continuous_learning(self):
        """Main learning loop"""
        logger.info("AI Continuous Learning loop started")

        while self.running:
            try:
                await self.perform_learning_cycle()
                await asyncio.sleep(self.learning_interval)
            except Exception as e:
                logger.exception(f"Error in learning cycle: {e}")
                await asyncio.sleep(60)  # Wait 1 minute on error

    async def perform_learning_cycle(self):
        """Perform one learning cycle"""
        try:
            # Update learning metrics
            self.metrics.learning_sessions += 1
            self.metrics.last_learning_time = time.time()

            # Store learning data in Redis
            redis = get_redis()
            if redis:
                learning_data = self.metrics.to_dict()
                await redis.set("ai_learning:metrics", json.dumps(learning_data))
                await redis.set("ai_learning:last_session", str(self.metrics.learning_sessions))

                logger.info(f"AI Learning cycle {self.metrics.learning_sessions} completed - Accuracy: {self.metrics.get_accuracy():.3f}")

            # Analyze recent AI signals for learning
            await self.analyze_recent_signals()

        except Exception as e:
            logger.exception(f"Error in learning cycle: {e}")

    async def analyze_recent_signals(self):
        """Analyze recent AI signals for learning"""
        try:
            redis = get_redis()
            if not redis:
                return

            # Get all AI signals
            signal_keys = []
            async for key in redis.scan_iter(match=REDIS_ML_SIGNAL_SCAN_PATTERN):
                signal_keys.append(key)

            if signal_keys:
                logger.info(f"Analyzing {len(signal_keys)} AI signals for learning")

                # Store analysis count
                await redis.set("ai_learning:signal_analysis_count", str(len(signal_keys)))

        except Exception as e:
            logger.exception(f"Error analyzing signals: {e}")

    def collect_feedback(self, prediction: str, confidence: float, actual: str):
        """Collect feedback on AI predictions"""
        self.metrics.record_prediction(prediction, confidence, actual)

        # Store feedback in Redis - track the task for proper cleanup
        feedback_task = asyncio.create_task(self._store_feedback_async(prediction, confidence, actual))
        self._feedback_tasks.add(feedback_task)
        # Remove task from tracking when it completes
        feedback_task.add_done_callback(self._feedback_tasks.discard)

    async def _store_feedback_async(self, prediction: str, confidence: float, actual: str):
        """Store feedback asynchronously"""
        try:
            redis = get_redis()
            if redis:
                feedback_data = {"prediction": prediction, "confidence": confidence, "actual": actual, "timestamp": datetime.now(timezone.utc).isoformat()}
                await redis.lpush("ai_learning:feedback", json.dumps(feedback_data))
                # Keep only last 100 feedback entries
                await redis.ltrim("ai_learning:feedback", 0, 99)
        except Exception as e:
            logger.exception(f"Error storing feedback: {e}")

    def get_performance_metrics(self) -> dict[str, Any]:
        """Get current performance metrics"""
        return self.metrics.to_dict()
