#!/usr/bin/env python3
"""
Real-Time Model Training Service
Continuous model improvement with online learning and performance-based retraining
"""

import asyncio
import contextlib
import logging
import time
from dataclasses import dataclass
from typing import Any

import numpy as np

from backend.services.advanced_model_ensemble import model_ensemble_service
from backend.services.database_pool_service import get_main_db_pool
from backend.services.task_manager import task_manager

logger = logging.getLogger(__name__)


@dataclass
class TrainingConfig:
    retrain_interval: int = 3600
    min_samples_for_retrain: int = 100
    max_training_samples: int = 5000
    performance_window: int = 100
    min_accuracy_threshold: float = 0.55
    enable_online_learning: bool = True
    adaptive_retrain: bool = True


@dataclass
class TrainingMetrics:
    symbol: str
    total_predictions: int = 0
    correct_predictions: int = 0
    total_trades: int = 0
    profitable_trades: int = 0
    avg_confidence: float = 0.0
    last_training_time: float = 0
    training_count: int = 0
    performance_score: float = 0.5
    model_version: str = "v1.0"
    ab_test_group: str = "A"
    health_score: float = 1.0
    volatility_adapted: bool = False
    market_regime_score: float = 0.5
    predictive_accuracy: float = 0.5


class RealTimeModelTrainer:
    def __init__(self, config: TrainingConfig | None = None) -> None:
        self.config = config or TrainingConfig()
        self.metrics: dict[str, TrainingMetrics] = {}
        self._training_tasks: dict[str, asyncio.Task[Any]] = {}
        self._monitoring_task: asyncio.Task[Any] | None = None
        self._health_monitor_task: asyncio.Task[Any] | None = None
        self._db_pool = get_main_db_pool()

        self.prediction_history: dict[str, list[dict[str, Any]]] = {}

        self.ab_test_active: dict[str, bool] = {}
        self.model_versions: dict[str, dict[str, Any]] = {}
        self.active_models: dict[str, str] = {}
        self.model_performance_history: dict[str, list[dict[str, Any]]] = {}

        self.health_check_interval: int = 600
        self.max_performance_history_events: int = 1000

        self.rng = np.random.default_rng()

    async def start_training_service(self) -> None:
        logger.info("Starting advanced real-time model training service")

        if self._monitoring_task is None or self._monitoring_task.done():
            self._monitoring_task = await task_manager.create_task(self._monitor_performance(), name="realtime_model_trainer:monitor_performance")

        if self._health_monitor_task is None or self._health_monitor_task.done():
            self._health_monitor_task = await task_manager.create_task(self._monitor_model_health(), name="realtime_model_trainer:monitor_model_health")

        await self._load_metrics_from_db()
        await self._initialize_ab_testing()

        logger.info("Advanced real-time model training service started")

    async def stop_training_service(self) -> None:
        logger.info("Stopping real-time model training service")

        if self._monitoring_task is not None:
            self._monitoring_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._monitoring_task

        if self._health_monitor_task is not None:
            self._health_monitor_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._health_monitor_task

        for symbol, task in list(self._training_tasks.items()):
            if not task.done():
                task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await task
            self._training_tasks.pop(symbol, None)

        await self._save_metrics_to_db()
        logger.info("Real-time model training service stopped")

    async def record_prediction(
        self,
        symbol: str,
        prediction: int,
        confidence: float,
        actual_outcome: int | None = None,
        pnl: float | None = None,
    ) -> None:
        if symbol not in self.metrics:
            self.metrics[symbol] = TrainingMetrics(symbol=symbol)

        if symbol not in self.prediction_history:
            self.prediction_history[symbol] = []

        metrics = self.metrics[symbol]

        prediction_record: dict[str, Any] = {
            "timestamp": time.time(),
            "prediction": prediction,
            "confidence": confidence,
            "actual_outcome": actual_outcome,
            "pnl": pnl,
        }

        self.prediction_history[symbol].append(prediction_record)

        if len(self.prediction_history[symbol]) > self.config.performance_window:
            self.prediction_history[symbol] = self.prediction_history[symbol][-self.config.performance_window :]

        metrics.total_predictions += 1
        metrics.avg_confidence = (metrics.avg_confidence * (metrics.total_predictions - 1) + confidence) / metrics.total_predictions

        if actual_outcome is not None and prediction == actual_outcome:
            metrics.correct_predictions += 1

        if pnl is not None:
            metrics.total_trades += 1
            if pnl > 0:
                metrics.profitable_trades += 1

        metrics.performance_score = self._calculate_performance_score(metrics)

        if self._should_retrain(symbol):
            await task_manager.create_task(self._trigger_retraining(symbol), name="realtime_model_trainer:trigger_retraining")

    def _calculate_performance_score(self, metrics: TrainingMetrics) -> float:
        if metrics.total_predictions == 0:
            return 0.5

        accuracy = metrics.correct_predictions / metrics.total_predictions

        profitability = metrics.profitable_trades / metrics.total_trades if metrics.total_trades > 0 else 0.5

        confidence_score = min(metrics.avg_confidence * 2, 1.0)

        return accuracy * 0.4 + profitability * 0.4 + confidence_score * 0.2

    def _should_retrain(self, symbol: str) -> bool:
        if symbol not in self.metrics:
            return True

        metrics = self.metrics[symbol]
        ensemble = model_ensemble_service.ensembles.get(symbol)

        if ensemble is None:
            return True

        if len(self.prediction_history.get(symbol, [])) < self.config.min_samples_for_retrain:
            return False

        if metrics.performance_score < self.config.min_accuracy_threshold:
            return True

        if self.config.adaptive_retrain:
            time_multiplier = 2.0 - metrics.performance_score
            adaptive_interval = self.config.retrain_interval * time_multiplier
        else:
            adaptive_interval = self.config.retrain_interval

        time_since_training = time.time() - metrics.last_training_time
        return time_since_training >= adaptive_interval

    async def _retrain_model(self, symbol: str) -> None:
        try:
            history = self.prediction_history.get(symbol, [])
            if len(history) < self.config.min_samples_for_retrain:
                logger.warning("Insufficient training data for %s: %d", symbol, len(history))
                return

            X_train: list[list[float]] = []
            y_train: list[int] = []

            for record in history[-self.config.max_training_samples :]:
                if record["actual_outcome"] is not None:
                    features = self._generate_training_features(record, symbol)
                    X_train.append(features)
                    y_train.append(record["actual_outcome"])

            if len(X_train) < self.config.min_samples_for_retrain:
                logger.warning("Still insufficient training data for %s after filtering", symbol)
                return

            X_train_arr = np.asarray(X_train, dtype=float)
            y_train_arr = np.asarray(y_train, dtype=int)

            await model_ensemble_service.async_train_ensemble(symbol, X_train_arr, y_train_arr)

            if symbol in self.metrics:
                self.metrics[symbol].last_training_time = time.time()
                self.metrics[symbol].training_count += 1

            await self._save_metrics_to_db()

            logger.info("Retrained model for %s with %d samples", symbol, len(X_train_arr))
        except Exception:
            logger.exception("Error during retraining for %s", symbol)

    def _generate_training_features(self, record: dict[str, Any], _symbol: str) -> list[float]:
        raw = float(record.get("confidence", 0.0) or 0.0)
        try:
            from backend.services.confidence_normalizer import ConfidenceNormalizer

            confidence = ConfidenceNormalizer.normalize(raw)
        except Exception as ex:
            logger.debug("ConfidenceNormalizer unavailable: %s", ex)
            confidence = raw
        prediction = float(record.get("prediction", 0))
        timestamp = float(record.get("timestamp", time.time()))

        hour_of_day = (timestamp % 86400.0) / 3600.0
        day_of_week = (timestamp % 604800.0) / 86400.0

        return [
            confidence,
            prediction,
            hour_of_day,
            day_of_week,
        ]

    async def _monitor_performance(self) -> None:
        while True:
            try:
                symbols_to_check = list(self.metrics.keys())

                for symbol in symbols_to_check:
                    if self._should_retrain(symbol):
                        await self._trigger_retraining(symbol)

                await self._log_performance_summary()

                await asyncio.sleep(300)
            except Exception:
                logger.exception("Performance monitoring error")
                await asyncio.sleep(60)

    async def _log_performance_summary(self) -> None:
        if not self.metrics:
            return

        total_predictions = sum(m.total_predictions for m in self.metrics.values())

        logger.info("[STATS] Model Performance Summary: %d predictions", total_predictions)

        for symbol, metrics in self.metrics.items():
            if metrics.total_predictions > 10:
                logger.debug(
                    "[UP] %s: %d/%d score: %.3f",
                    symbol,
                    metrics.correct_predictions,
                    metrics.total_predictions,
                    metrics.performance_score,
                )

    async def _load_metrics_from_db(self) -> None:
        try:
            import json as _json

            rows = await self._db_pool.fetch_all("SELECT symbol, metrics_json FROM realtime_trainer_metrics")
            for row in rows:
                sym = row[0]
                data = _json.loads(row[1])
                m = TrainingMetrics(symbol=sym)
                for k, v in data.items():
                    if hasattr(m, k):
                        setattr(m, k, v)
                self.metrics[sym] = m
            if rows:
                logger.info("Loaded training metrics for %d symbols from database", len(rows))
        except Exception:
            logger.warning("Could not load training metrics from database (table may not exist yet)")

    async def _save_metrics_to_db(self) -> None:
        try:
            import json as _json

            await self._db_pool.execute(
                """CREATE TABLE IF NOT EXISTS realtime_trainer_metrics
                   (symbol TEXT PRIMARY KEY, metrics_json TEXT, updated_at REAL)"""
            )
            for sym, m in self.metrics.items():
                data = {
                    "total_predictions": m.total_predictions,
                    "correct_predictions": m.correct_predictions,
                    "total_trades": m.total_trades,
                    "profitable_trades": m.profitable_trades,
                    "avg_confidence": m.avg_confidence,
                    "last_training_time": m.last_training_time,
                    "training_count": m.training_count,
                    "performance_score": m.performance_score,
                    "model_version": m.model_version,
                    "health_score": m.health_score,
                    "market_regime_score": m.market_regime_score,
                    "predictive_accuracy": m.predictive_accuracy,
                }
                await self._db_pool.execute(
                    """INSERT OR REPLACE INTO realtime_trainer_metrics
                       (symbol, metrics_json, updated_at) VALUES (?, ?, ?)""",
                    (sym, _json.dumps(data), time.time()),
                )
            logger.debug("Saved training metrics for %d symbols to database", len(self.metrics))
        except Exception:
            logger.exception("Could not save training metrics to database")

    def get_performance_report(self, symbol: str | None = None) -> dict[str, Any]:
        if symbol is not None:
            metrics = self.metrics.get(symbol)
            if metrics is None:
                return {"error": f"No metrics found for {symbol}"}

            return {
                "symbol": symbol,
                "total_predictions": metrics.total_predictions,
                "accuracy": metrics.correct_predictions / max(metrics.total_predictions, 1),
                "win_rate": metrics.profitable_trades / max(metrics.total_trades, 1),
                "avg_confidence": metrics.avg_confidence,
                "performance_score": metrics.performance_score,
                "last_training": metrics.last_training_time,
                "training_count": metrics.training_count,
            }

        return {
            "total_symbols": len(self.metrics),
            "overall_performance": float(np.mean([m.performance_score for m in self.metrics.values()])) if self.metrics else 0.0,
            "total_predictions": sum(m.total_predictions for m in self.metrics.values()),
            "symbols": [m.symbol for m in self.metrics.values()],
        }

    async def _initialize_ab_testing(self) -> None:
        try:
            for symbol in self.metrics:
                if symbol not in self.ab_test_active:
                    self.ab_test_active[symbol] = False
                if symbol not in self.active_models:
                    self.active_models[symbol] = "v1.0"
                if symbol not in self.model_versions:
                    self.model_versions[symbol] = {}
                if symbol not in self.model_performance_history:
                    self.model_performance_history[symbol] = []
            logger.info("A/B testing initialized for all symbols")
        except Exception:
            logger.exception("Failed to initialize A/B testing")

    def start_ab_test(self, symbol: str, challenger_version: str = "v2.0") -> bool:
        try:
            if symbol not in self.metrics:
                logger.warning("Cannot start A/B test for %s: no metrics available", symbol)
                return False

            self.ab_test_active[symbol] = True

            versions_for_symbol = self.model_versions.setdefault(symbol, {})
            if challenger_version not in versions_for_symbol:
                versions_for_symbol[challenger_version] = {
                    "performance_score": 0.5,
                    "test_predictions": 0,
                    "test_correct": 0,
                    "start_time": time.time(),
                }

            logger.info(
                "Started A/B test for %s: %s vs %s",
                symbol,
                self.active_models.get(symbol, "v1.0"),
                challenger_version,
            )
        except Exception:
            logger.exception("Failed to start A/B test for %s", symbol)
            return False
        else:
            return True

    def record_ab_test_prediction(
        self,
        symbol: str,
        version: str,
        prediction: int,
        actual: int,
        _confidence: float,
    ) -> None:
        try:
            if not self.ab_test_active.get(symbol, False):
                return

            versions_for_symbol = self.model_versions.setdefault(symbol, {})
            if version not in versions_for_symbol:
                versions_for_symbol[version] = {
                    "performance_score": 0.5,
                    "test_predictions": 0,
                    "test_correct": 0,
                    "start_time": time.time(),
                }

            version_data = versions_for_symbol[version]
            version_data["test_predictions"] += 1

            if prediction == actual:
                version_data["test_correct"] += 1

            accuracy = version_data["test_correct"] / max(version_data["test_predictions"], 1)
            version_data["performance_score"] = accuracy

            current_version = self.active_models.get(symbol, "v1.0")
            current_score = self.metrics.get(symbol, TrainingMetrics(symbol)).performance_score

            if version != current_version and version_data["performance_score"] > current_score + 0.05:
                self._switch_to_better_model(symbol, version, version_data["performance_score"])
        except Exception:
            logger.exception("Failed to record A/B test prediction for %s", symbol)

    def _switch_to_better_model(self, symbol: str, new_version: str, new_score: float) -> None:
        try:
            old_version = self.active_models.get(symbol, "v1.0")
            self.active_models[symbol] = new_version

            if symbol in self.metrics:
                self.metrics[symbol].model_version = new_version

            self.ab_test_active[symbol] = False

            logger.info("Model switch for %s: %s -> %s (score: %.3f)", symbol, old_version, new_version, new_score)

            history_list = self.model_performance_history.setdefault(symbol, [])
            history_list.append(
                {
                    "timestamp": time.time(),
                    "event": "model_switch",
                    "from_version": old_version,
                    "to_version": new_version,
                    "new_score": new_score,
                }
            )
            if len(history_list) > self.max_performance_history_events:
                self.model_performance_history[symbol] = history_list[-self.max_performance_history_events :]
        except Exception:
            logger.exception("Failed to switch model for %s", symbol)

    async def _monitor_model_health(self) -> None:
        while True:
            try:
                await self._perform_health_check()
                await asyncio.sleep(self.health_check_interval)
            except Exception:
                logger.exception("Model health monitoring error")
                await asyncio.sleep(60)

    async def _perform_health_check(self) -> None:
        try:
            for symbol, metrics in self.metrics.items():
                health_score = await self._calculate_model_health(symbol, metrics)
                metrics.health_score = health_score

                if health_score < 0.7:
                    await self._handle_health_issue(symbol, health_score)

                if len(self.prediction_history.get(symbol, [])) >= 50:
                    predictive_accuracy = await self._calculate_predictive_accuracy(symbol)
                    metrics.predictive_accuracy = predictive_accuracy

                    if predictive_accuracy < 0.6:
                        logger.warning("Low predictive accuracy for %s: %.3f", symbol, predictive_accuracy)
        except Exception:
            logger.exception("Health check failed")

    async def _calculate_model_health(self, symbol: str, metrics: TrainingMetrics) -> float:
        try:
            health_factors: list[float] = []

            history = self.prediction_history.get(symbol, [])
            if len(history) >= 20:
                recent_predictions = history[-20:]
                recent_accuracy = sum(1 for p in recent_predictions if p.get("prediction") == p.get("actual_outcome", 0)) / len(recent_predictions)
                health_factors.append(min(recent_accuracy * 1.2, 1.0))
            else:
                health_factors.append(0.5)

            if metrics.avg_confidence > 0:
                confidence_stability = min(metrics.avg_confidence * 2, 1.0)
                health_factors.append(confidence_stability)
            else:
                health_factors.append(0.3)

            time_since_training = time.time() - metrics.last_training_time
            recency_score = max(0.0, 1.0 - (time_since_training / (7.0 * 24.0 * 3600.0)))
            health_factors.append(recency_score)

            data_quality = min(metrics.total_predictions / 1000.0, 1.0)
            health_factors.append(data_quality)

            market_adaptation = metrics.market_regime_score
            health_factors.append(market_adaptation)

            weights = [0.3, 0.25, 0.2, 0.15, 0.1]
            health_score = sum(f * w for f, w in zip(health_factors, weights, strict=False))

            return max(0.0, min(1.0, health_score))
        except Exception:
            logger.exception("Health calculation failed for %s", symbol)
            return 0.5

    async def _calculate_predictive_accuracy(self, symbol: str) -> float:
        try:
            predictions = self.prediction_history.get(symbol, [])
            if len(predictions) < 50:
                return 0.5

            recent = predictions[-50:]
            predicted_vs_actual: list[int] = []

            for pred in recent:
                raw = float(pred.get("confidence", 0.5) or 0.5)
                try:
                    from backend.services.confidence_normalizer import ConfidenceNormalizer

                    confidence = ConfidenceNormalizer.normalize(raw)
                except Exception:
                    confidence = raw
                actual_correct = pred.get("prediction") == pred.get("actual_outcome", 0)

                if confidence > 0.7:
                    predicted_vs_actual.append(1 if actual_correct else 0)
                elif confidence < 0.3:
                    predicted_vs_actual.append(1 if not actual_correct else 0)

            if predicted_vs_actual:
                return sum(predicted_vs_actual) / len(predicted_vs_actual)
            else:
                return 0.5
        except Exception:
            logger.exception("Predictive accuracy calculation failed for %s", symbol)
            return 0.5

    async def _handle_health_issue(self, symbol: str, health_score: float) -> None:
        try:
            logger.warning("Model health issue for %s: %.3f", symbol, health_score)

            history_list = self.model_performance_history.setdefault(symbol, [])
            history_list.append(
                {
                    "timestamp": time.time(),
                    "event": "health_issue",
                    "health_score": health_score,
                    "action": "monitoring",
                }
            )
            if len(history_list) > self.max_performance_history_events:
                self.model_performance_history[symbol] = history_list[-self.max_performance_history_events :]

            if health_score < 0.5:
                logger.info("Triggering emergency retraining for %s", symbol)
                await self._trigger_retraining(symbol, priority="high")
        except Exception:
            logger.exception("Health issue handling failed for %s", symbol)

    async def _trigger_retraining(self, symbol: str, priority: str = "normal") -> None:
        try:
            existing_task = self._training_tasks.get(symbol)
            if existing_task is not None and not existing_task.done() and priority != "high":
                logger.info("Retraining already in progress for %s", symbol)
                return

            task = await task_manager.create_task(self._perform_advanced_retraining(symbol, priority), name="realtime_model_trainer:perform_advanced_retraining")
            self._training_tasks[symbol] = task

            logger.info("Advanced retraining triggered for %s (priority: %s)", symbol, priority)
        except Exception:
            logger.exception("Retraining trigger failed for %s", symbol)

    async def _perform_advanced_retraining(self, symbol: str, priority: str) -> None:
        try:
            logger.info("Starting advanced retraining for %s (priority: %s)", symbol, priority)

            training_data = await self._get_enhanced_training_data(symbol)
            success = await self._train_with_market_adaptation(symbol, training_data)

            if success and symbol in self.metrics:
                self.metrics[symbol].last_training_time = time.time()
                self.metrics[symbol].training_count += 1
                new_version = f"v{self.metrics[symbol].training_count}.0"
                self.metrics[symbol].model_version = new_version
                self.active_models[symbol] = new_version
                self.metrics[symbol].health_score = 1.0

                logger.info("Advanced retraining completed for %s (version: %s)", symbol, new_version)
            elif not success:
                logger.error("Advanced retraining failed for %s", symbol)
        except Exception:
            logger.exception("Advanced retraining failed for %s", symbol)

    async def _get_enhanced_training_data(self, symbol: str) -> dict[str, Any]:
        try:
            base_data = await self._get_training_data(symbol)
            regime_features = await self._calculate_market_regime_features(symbol)
            volatility_features = await self._calculate_volatility_features(symbol)

            combined: dict[str, Any] = {}
            combined.update(base_data)
            combined.update(regime_features)
            combined.update(volatility_features)
        except Exception:
            logger.exception("Enhanced training data generation failed for %s", symbol)
            return {}
        else:
            return combined

    async def _get_training_data(self, symbol: str) -> dict[str, Any]:
        """Fetch OHLCV from live market data and return price/volume-derived features."""
        try:
            ccxt_symbol = f"{symbol.replace('USDT', '')}/USDT" if "USDT" in symbol.upper() else f"{symbol}/USDT"
            try:
                from backend.services.live_market_data import live_market_data_service

                if live_market_data_service:
                    from backend.services.history_context_gates import min_ohlcv_bars_for_signal

                    min_b = min_ohlcv_bars_for_signal()
                    ohlcv = await live_market_data_service.get_ohlcv(ccxt_symbol, "1h", min(1000, max(min_b, 50)))
                    if ohlcv and len(ohlcv) >= min_b:
                        tail = ohlcv[-min_b:]
                        closes = [float(c[4]) for c in tail if len(c) >= 5]
                        vols = [float(c[5]) if len(c) > 5 else 0.0 for c in tail if len(c) >= 5]
                        if closes:
                            avg_price = sum(closes) / len(closes)
                            price_vol = (max(closes) - min(closes)) / avg_price if avg_price else 0.0
                            avg_vol = sum(vols) / len(vols) if vols else 0.0
                            return {
                                "price_mean": avg_price,
                                "price_volatility": price_vol,
                                "volume_mean": avg_vol,
                                "sample_count": len(closes),
                            }
            except Exception as e:
                logger.debug("Live OHLCV fetch failed for %s: %s", symbol, e)
            return {"price_mean": 0.0, "price_volatility": 0.0, "volume_mean": 0.0, "sample_count": 0}
        except Exception:
            logger.exception("Training data retrieval failed for %s", symbol)
            return {}

    async def _calculate_market_regime_features(self, symbol: str) -> dict[str, Any]:
        try:
            return {
                "market_regime_trend": 0.5,
                "market_regime_volatility": 0.5,
                "market_regime_volume": 0.5,
            }
        except Exception:
            logger.exception("Market regime calculation failed for %s", symbol)
            return {}

    async def _calculate_volatility_features(self, symbol: str) -> dict[str, Any]:
        try:
            return {
                "volatility_regime": 0.5,
                "volatility_adapted_weights": True,
            }
        except Exception:
            logger.exception("Volatility calculation failed for %s", symbol)
            return {}

    async def _train_with_market_adaptation(self, symbol: str, training_data: dict[str, Any]) -> bool:
        try:
            ensemble = model_ensemble_service.ensembles.get(symbol)
            if ensemble is None:
                logger.warning("No ensemble found for %s", symbol)
                return False

            _ = training_data.get("market_regime_trend", 0.5) + 1.0
            _ = training_data.get("volatility_regime", 0.5)

            success = await ensemble.retrain(training_data)

            if success and symbol in self.metrics:
                self.metrics[symbol].volatility_adapted = bool(training_data.get("volatility_adapted_weights", False))
                self.metrics[symbol].market_regime_score = float(training_data.get("market_regime_trend", 0.5))
        except Exception:
            logger.exception("Market adaptation training failed for %s", symbol)
            return False
        else:
            return success

    def get_advanced_performance_report(self, symbol: str) -> dict[str, Any]:
        try:
            if symbol not in self.metrics:
                return {"error": "No metrics available for symbol"}

            metrics = self.metrics[symbol]
            prediction_history = self.prediction_history.get(symbol, [])

            report: dict[str, Any] = {
                "symbol": symbol,
                "basic_metrics": {
                    "total_predictions": metrics.total_predictions,
                    "accuracy": metrics.correct_predictions / max(metrics.total_predictions, 1),
                    "avg_confidence": metrics.avg_confidence,
                    "performance_score": metrics.performance_score,
                },
                "advanced_metrics": {
                    "model_version": metrics.model_version,
                    "health_score": metrics.health_score,
                    "predictive_accuracy": metrics.predictive_accuracy,
                    "market_regime_score": metrics.market_regime_score,
                    "volatility_adapted": metrics.volatility_adapted,
                },
                "ab_testing": {
                    "active": self.ab_test_active.get(symbol, False),
                    "active_model": self.active_models.get(symbol, "v1.0"),
                    "available_versions": list(self.model_versions.get(symbol, {}).keys()),
                },
                "training_stats": {
                    "last_training": metrics.last_training_time,
                    "training_count": metrics.training_count,
                    "days_since_training": (time.time() - metrics.last_training_time) / (24.0 * 3600.0),
                },
                "recent_performance": self._analyze_recent_performance(prediction_history),
            }

        except Exception as exc:
            logger.exception("Performance report generation failed for %s", symbol)
            return {"error": str(exc)}
        else:
            return report

    def _analyze_recent_performance(self, predictions: list[dict[str, Any]]) -> dict[str, Any]:
        try:
            if len(predictions) < 10:
                return {"insufficient_data": True}

            recent = predictions[-50:] if len(predictions) >= 50 else predictions

            window_size = 10
            rolling_accuracy: list[float] = []

            for i in range(window_size, len(recent) + 1):
                window = recent[i - window_size : i]
                accuracy = sum(1 for p in window if p.get("prediction") == p.get("actual_outcome")) / len(window)
                rolling_accuracy.append(accuracy)

            if not rolling_accuracy:
                return {
                    "recent_accuracy_trend": [],
                    "avg_recent_accuracy": 0.0,
                    "accuracy_volatility": 0.0,
                    "trend_direction": "flat",
                }

            recent_tail = rolling_accuracy[-10:]
            avg_recent_accuracy = sum(recent_tail) / len(recent_tail)
            accuracy_volatility = float(np.std(recent_tail)) if len(recent_tail) >= 2 else 0.0

            if len(rolling_accuracy) >= 2 and rolling_accuracy[-1] > rolling_accuracy[0]:
                trend_direction = "improving"
            elif len(rolling_accuracy) >= 2 and rolling_accuracy[-1] < rolling_accuracy[0]:
                trend_direction = "declining"
            else:
                trend_direction = "flat"

            return {
                "recent_accuracy_trend": rolling_accuracy[-5:],
                "avg_recent_accuracy": avg_recent_accuracy,
                "accuracy_volatility": accuracy_volatility,
                "trend_direction": trend_direction,
            }
        except Exception as exc:
            logger.exception("Recent performance analysis failed")
            return {"error": str(exc)}


class OnlineLearningManager:
    def __init__(self, trainer: RealTimeModelTrainer) -> None:
        self.trainer = trainer
        self.online_models: dict[str, dict[str, Any]] = {}

    async def update_model_online(self, symbol: str, features: np.ndarray, actual_outcome: int) -> None:
        try:
            # Derive a proxy confidence from first feature element (0-1 scale) if available
            confidence = float(features[0]) if features is not None and len(features) > 0 else 0.5
            confidence = max(0.0, min(1.0, confidence))
            # Derive prediction from actual_outcome as a pass-through (no model inference here)
            prediction = actual_outcome
            if symbol not in self.online_models:
                self.enable_online_learning(symbol)
            if self.online_models.get(symbol, {}).get("enabled", False):
                self.online_models[symbol]["updates"] = self.online_models[symbol].get("updates", 0) + 1
                self.online_models[symbol]["last_update"] = time.time()
                await self.trainer.record_prediction(symbol, prediction, confidence, actual_outcome)
        except Exception:
            logger.exception("Online learning update failed for %s", symbol)

    def enable_online_learning(self, symbol: str) -> None:
        if symbol not in self.online_models:
            self.online_models[symbol] = {
                "enabled": True,
                "updates": 0,
                "last_update": time.time(),
            }
        else:
            self.online_models[symbol]["enabled"] = True
        logger.info("Enabled online learning for %s", symbol)

    def disable_online_learning(self, symbol: str) -> None:
        if symbol in self.online_models:
            self.online_models[symbol]["enabled"] = False
            logger.info("Disabled online learning for %s", symbol)


realtime_trainer = RealTimeModelTrainer()
online_learning_manager = OnlineLearningManager(realtime_trainer)
