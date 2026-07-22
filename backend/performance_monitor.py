import asyncio
import contextlib
import json
import logging
import threading
import time
from collections import defaultdict, deque
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

# Direct imports for production
import psutil
from sqlalchemy import text

# Redis service for storing metrics
try:
    from backend.services.redis_service import get_redis_service

    REDIS_AVAILABLE = True
except ImportError:
    get_redis_service = None
    REDIS_AVAILABLE = False

# Database imports for connectivity checks
try:
    from backend.database_init import SessionLocal
except ImportError:
    SessionLocal = None

# External API imports
try:
    from backend.services.binance_rest_client import BinanceREST
    from backend.utils.binance_weight_limiter import BinanceWeightLimiter
except ImportError:
    BinanceREST = None
    BinanceWeightLimiter = None

from backend.services.task_manager import task_manager

logger = logging.getLogger(__name__)

DB_QUERY_THRESHOLD = 1.0
CACHE_HIT_RATE_THRESHOLD = 0.8
MEMORY_USAGE_THRESHOLD = 0.9
CPU_USAGE_THRESHOLD = 0.8


@dataclass
class PerformanceMetric:
    name: str
    value: float
    timestamp: float
    threshold: float
    status: str


class PerformanceAlert:
    def __init__(self) -> None:
        self.alerts: deque = deque(maxlen=100)
        self.alert_counts = defaultdict(int)

    def add_alert(self, metric_name: str, value: float, threshold: float, severity: str) -> None:
        alert = {
            "metric": metric_name,
            "value": value,
            "threshold": threshold,
            "severity": severity,
            "timestamp": time.time(),
        }
        self.alerts.append(alert)
        self.alert_counts[metric_name] += 1
        logger.warning(f"Performance alert: {metric_name}={value:.4f} threshold={threshold:.4f} severity={severity}")

    def get_recent_alerts(self, minutes: int = 10) -> list[dict[str, Any]]:
        cutoff_time = time.time() - (minutes * 60)
        return [alert for alert in self.alerts if alert["timestamp"] > cutoff_time]

    def get_alert_stats(self) -> dict[str, Any]:
        return {
            "total_alerts": len(self.alerts),
            "alert_counts": dict(self.alert_counts),
            "recent_alerts": len(self.get_recent_alerts()),
        }


class PerformanceMonitor:
    def __init__(self) -> None:
        self.metrics: dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        self.alerts = PerformanceAlert()
        self.running = False
        self.monitoring_task: asyncio.Task | None = None
        self.lock = threading.Lock()
        self.db_queries: dict[str, list[dict[str, Any]]] = defaultdict(list)
        self.cache_operations: dict[str, dict[str, int]] = defaultdict(dict)
        self.model_operations: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))

    async def start(self) -> None:
        self.running = True
        self.monitoring_task = await task_manager.create_task(self._monitoring_loop(), name="performance_monitor:monitoring_loop")
        logger.info("Performance monitor started")

    # ENHANCED MONITORING METHODS

    async def get_comprehensive_performance_report(self) -> dict[str, Any]:
        """
        Generate comprehensive performance report with all metrics and insights
        """
        try:
            report = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "system_health": await self._get_system_health_status(),
                "trading_performance": await self._get_trading_performance_metrics(),
                "ai_performance": await self._get_ai_performance_metrics(),
                "risk_metrics": await self._get_risk_metrics(),
                "alerts": self.alerts.get_recent_alerts(60),  # Last hour
                "recommendations": await self._generate_performance_recommendations(),
            }

            # Cache report in Redis if available
            await self._cache_performance_report(report)

        except Exception as e:
            logger.exception(f"Error generating performance report: {e}")
            return {"error": str(e), "timestamp": datetime.now(timezone.utc).isoformat()}
        else:
            return report

    async def _get_system_health_status(self) -> dict[str, Any]:
        """Get comprehensive system health status"""
        try:
            health = {
                "cpu_usage": self._get_latest_metric_value("cpu_usage"),
                "memory_usage": self._get_latest_metric_value("memory_usage"),
                "disk_usage": self._get_latest_metric_value("disk_usage"),
                "db_query_performance": self._get_latest_metric_value("db_query_time"),
                "cache_hit_rate": self._get_latest_metric_value("cache_hit_rate"),
                "redis_connectivity": await self._check_redis_connectivity(),
                "database_connectivity": await self._check_database_connectivity(),
                "external_api_status": await self._check_external_api_status(),
                "overall_status": "unknown",
            }

            # Determine overall status
            critical_metrics = ["cpu_usage", "memory_usage", "db_query_performance"]
            warning_count = 0
            critical_count = 0

            for metric in critical_metrics:
                if health.get(metric, 0) > 0.8:  # 80% threshold
                    critical_count += 1
                elif health.get(metric, 0) > 0.6:  # 60% threshold
                    warning_count += 1

            if critical_count > 0:
                health["overall_status"] = "critical"
            elif warning_count > 0:
                health["overall_status"] = "warning"
            elif all(v is not None for v in health.values() if isinstance(v, (int, float))):
                health["overall_status"] = "healthy"
            else:
                health["overall_status"] = "degraded"

        except Exception as e:
            logger.exception(f"Error getting system health: {e}")
            return {"error": str(e)}
        else:
            return health

    async def _get_trading_performance_metrics(self) -> dict[str, Any]:
        """Get trading performance metrics from Redis/database"""
        try:
            # Try to get from Redis cache first
            trading_stats = await self._get_redis_trading_stats()

            return {
                "total_trades": trading_stats.get("total_trades", 0),
                "win_rate": trading_stats.get("win_rate", 0.0),
                "total_pnl": trading_stats.get("total_pnl", 0.0),
                "sharpe_ratio": trading_stats.get("sharpe_ratio", 0.0),
                "max_drawdown": trading_stats.get("max_drawdown_pct", 0.0),
                "active_positions": trading_stats.get("active_positions", 0),
                "portfolio_value": trading_stats.get("portfolio_value", 0.0),
                "trading_volume_24h": trading_stats.get("trading_volume_24h", 0.0),
            }

        except Exception as e:
            logger.exception(f"Error getting trading metrics: {e}")
            return {"error": str(e)}

    async def _get_ai_performance_metrics(self) -> dict[str, Any]:
        """Get AI model performance metrics"""
        try:
            ai_stats = await self._get_redis_ai_stats()

            return {
                "model_accuracy": ai_stats.get("current_accuracy", 0.0),
                "total_predictions": ai_stats.get("total_predictions", 0),
                "correct_predictions": ai_stats.get("correct_predictions", 0),
                "model_loaded": ai_stats.get("model_loaded", False),
                "last_training_time": ai_stats.get("last_training_time"),
                "feature_count": ai_stats.get("feature_count", 0),
                "prediction_confidence_avg": ai_stats.get("avg_confidence", 0.0),
                "model_version": ai_stats.get("model_version", "unknown"),
            }

        except Exception as e:
            logger.exception(f"Error getting AI metrics: {e}")
            return {"error": str(e)}

    async def _get_risk_metrics(self) -> dict[str, Any]:
        """Get portfolio risk metrics"""
        try:
            risk_stats = await self._get_redis_risk_stats()

            return {
                "portfolio_volatility": risk_stats.get("volatility", 0.0),
                "value_at_risk_95": risk_stats.get("var_95", 0.0),
                "expected_shortfall_95": risk_stats.get("cvar_95", 0.0),
                "max_drawdown": risk_stats.get("max_drawdown", 0.0),
                "correlation_average": risk_stats.get("correlation_avg", 0.0),
                "largest_position_pct": risk_stats.get("max_weight", 0.0),
                "risk_adjusted_return": risk_stats.get("sharpe_ratio", 0.0),
            }

        except Exception as e:
            logger.exception(f"Error getting risk metrics: {e}")
            return {"error": str(e)}

    async def _generate_performance_recommendations(self) -> list[dict[str, Any]]:
        """Generate performance improvement recommendations"""
        recommendations = []

        try:
            # Analyze system health
            health = await self._get_system_health_status()

            if health.get("cpu_usage", 0) > 0.8:
                recommendations.append(
                    {
                        "type": "system_optimization",
                        "priority": "high",
                        "message": "High CPU usage detected. Consider optimizing database queries or increasing server capacity.",
                        "metric": "cpu_usage",
                        "value": health["cpu_usage"],
                        "threshold": 0.8,
                    }
                )

            if health.get("memory_usage", 0) > 0.85:
                recommendations.append(
                    {
                        "type": "memory_optimization",
                        "priority": "high",
                        "message": "High memory usage. Implement memory pooling or increase RAM.",
                        "metric": "memory_usage",
                        "value": health["memory_usage"],
                        "threshold": 0.85,
                    }
                )

            # Analyze trading performance
            trading = await self._get_trading_performance_metrics()

            if trading.get("win_rate", 0) < 0.3:
                recommendations.append(
                    {
                        "type": "strategy_optimization",
                        "priority": "high",
                        "message": "Low win rate detected. Review AI model performance and risk management.",
                        "metric": "win_rate",
                        "value": trading["win_rate"],
                        "threshold": 0.3,
                    }
                )

            if trading.get("max_drawdown", 0) > 0.1:
                recommendations.append(
                    {
                        "type": "risk_management",
                        "priority": "critical",
                        "message": "High drawdown detected. Implement stricter risk controls.",
                        "metric": "max_drawdown",
                        "value": trading["max_drawdown"],
                        "threshold": 0.1,
                    }
                )

            # Analyze AI performance
            ai = await self._get_ai_performance_metrics()

            if ai.get("model_accuracy", 0) < 0.65:
                recommendations.append(
                    {
                        "type": "ai_training",
                        "priority": "high",
                        "message": "AI accuracy below threshold. Retrain models with recent data.",
                        "metric": "model_accuracy",
                        "value": ai["model_accuracy"],
                        "threshold": 0.65,
                    }
                )

            if not ai.get("model_loaded", True):
                recommendations.append(
                    {"type": "system_maintenance", "priority": "critical", "message": "AI model not loaded. Check model files and loading process.", "metric": "model_loaded", "value": False}
                )

        except Exception as e:
            logger.exception(f"Error generating recommendations: {e}")
            recommendations.append({"type": "system_error", "priority": "high", "message": f"Error generating recommendations: {e}", "error": str(e)})

        return recommendations

    def _get_latest_metric_value(self, metric_name: str) -> float | None:
        """Get the latest value for a metric"""
        try:
            if self.metrics.get(metric_name):
                return self.metrics[metric_name][-1].value
            else:
                return None
        except Exception as ex:
            logger.debug("_get_latest_metric_value failed: %s", ex)
            return None

    async def _check_redis_connectivity(self) -> bool:
        """Check Redis connectivity"""
        if not REDIS_AVAILABLE:
            return False

        try:
            redis_service = get_redis_service()
            return await redis_service.ping()
        except Exception as ex:
            logger.debug("_check_redis_connectivity failed: %s", ex)
            return False

    async def _check_database_connectivity(self) -> bool:
        """Check database connectivity"""
        try:
            if SessionLocal is None:
                return False

            with SessionLocal() as session:
                session.execute(text("SELECT 1"))
        except Exception as ex:
            logger.debug("_check_database_connectivity failed: %s", ex)
            return False
        else:
            return True

    async def _check_external_api_status(self) -> dict[str, bool]:
        """Check external API connectivity"""
        try:
            if BinanceREST is None or BinanceWeightLimiter is None:
                return {"binance_api": False}

            limiter = await BinanceWeightLimiter.create()
            client = BinanceREST(limiter)
            # Try a lightweight API call
            response = await client.get_server_time()
            if not response or "serverTime" not in response:
                return {"binance_api": False}
        except Exception as ex:
            logger.debug("_check_external_api_status failed: %s", ex)
            return {"binance_api": False}
        else:
            return {"binance_api": True}

    async def _get_redis_trading_stats(self) -> dict[str, Any]:
        """Get trading statistics from the shared async Redis client."""
        if not REDIS_AVAILABLE:
            return {}
        try:
            r = get_redis_service()
            if r is None:
                return {}
            stats = {}
            paper_stats_raw = await r.get("paper_trading:stats")
            if paper_stats_raw:
                try:
                    paper_stats = json.loads(paper_stats_raw)
                    stats.update(
                        {
                            "total_trades": paper_stats.get("total_trades", 0),
                            "win_rate": paper_stats.get("win_rate", 0.0),
                            "total_pnl": paper_stats.get("total_pnl", 0.0),
                            "sharpe_ratio": paper_stats.get("sharpe_ratio", 0.0),
                            "max_drawdown_pct": paper_stats.get("max_drawdown_pct", 0.0),
                        }
                    )
                except Exception as ex:
                    logger.debug("paper_stats parse failed: %s", ex)
            position_keys = []
            try:
                async for key in r.scan_iter(match="paper:position:*", count=100):
                    position_keys.append(key)
            except Exception as scan_err:
                logger.debug("SCAN failed for paper:position: %s", scan_err)
                position_keys = []
            stats["active_positions"] = len(position_keys) if position_keys else 0
            portfolio_value = 0.0
            for key in position_keys or []:
                try:
                    pos_data = await r.hgetall(key)
                    if pos_data:
                        quantity = float(pos_data.get("quantity", 0))
                        current_price = float(pos_data.get("current_price", 0))
                        portfolio_value += quantity * current_price
                except Exception as ex:
                    logger.debug("position parse failed: %s", ex)
            stats["portfolio_value"] = portfolio_value
            return stats
        except Exception as e:
            logger.warning(f"Error getting Redis trading stats: {e}")
            return {}

    async def _get_redis_ai_stats(self) -> dict[str, Any]:
        """Get AI performance statistics from the shared async Redis client."""
        if not REDIS_AVAILABLE:
            return {}
        try:
            r = get_redis_service()
            if r is None:
                return {}
            stats = {}
            orchestrator_raw = await r.get("orchestrator:status")
            if orchestrator_raw:
                try:
                    orchestrator_data = json.loads(orchestrator_raw)
                    stats.update(
                        {
                            "current_accuracy": orchestrator_data.get("current_accuracy", 0.0),
                            "total_predictions": 0,
                            "correct_predictions": 0,
                            "model_loaded": orchestrator_data.get("is_running", False),
                        }
                    )
                except Exception as ex:
                    logger.debug("orchestrator_data parse failed: %s", ex)
            decision_keys = []
            try:
                count = 0
                async for key in r.scan_iter(match="ai_decision:*", count=100):
                    decision_keys.append(key)
                    count += 1
                    if count >= 10:
                        break
            except Exception as scan_err:
                logger.debug("SCAN failed for ai_decision: %s", scan_err)
                decision_keys = []
            if decision_keys:
                confidences = []
                for key in decision_keys:
                    try:
                        decision_data = await r.hgetall(key)
                        if decision_data and "confidence" in decision_data:
                            raw = float(decision_data["confidence"])
                            try:
                                from backend.services.confidence_normalizer import ConfidenceNormalizer

                                confidences.append(ConfidenceNormalizer.normalize(raw))
                            except Exception as ex:
                                logger.debug("ConfidenceNormalizer failed: %s", ex)
                                confidences.append(raw)
                    except Exception as ex:
                        logger.debug("decision_data parse failed: %s", ex)
                if confidences:
                    stats["avg_confidence"] = sum(confidences) / len(confidences)
            return stats
        except Exception as e:
            logger.warning(f"Error getting Redis AI stats: {e}")
            return {}

    async def _get_redis_risk_stats(self) -> dict[str, Any]:
        """Get risk statistics from the shared async Redis client."""
        if not REDIS_AVAILABLE:
            return {}
        try:
            r = get_redis_service()
            if r is None:
                return {}
            stats = {}
            risk_raw = await r.get("risk_data")
            if risk_raw:
                try:
                    risk_data = json.loads(risk_raw)
                    portfolio_risk = risk_data.get("portfolio_risk", {})
                    return {
                        "volatility": portfolio_risk.get("volatility", 0.0),
                        "var_95": portfolio_risk.get("var_95", 0.0),
                        "cvar_95": portfolio_risk.get("cvar_95", 0.0),
                        "max_drawdown": portfolio_risk.get("max_drawdown", 0.0),
                        "correlation_avg": portfolio_risk.get("correlation_avg", 0.0),
                        "max_weight": risk_data.get("position_limits", {}).get("max_weight", 0.0),
                        "sharpe_ratio": 0.0,
                    }
                except Exception as ex:
                    logger.debug("risk_data parse failed: %s", ex)
            return stats
        except Exception as e:
            logger.warning(f"Error getting Redis risk stats: {e}")
            return {}

    async def _cache_performance_report(self, report: dict[str, Any]) -> None:
        """Cache performance report in Redis"""
        if not REDIS_AVAILABLE:
            return

        try:
            r = get_redis_service()

            # Cache for 5 minutes
            await r.set("performance_report", json.dumps(report), ex=300)

        except Exception as e:
            logger.warning(f"Error caching performance report: {e}")

    async def stop(self) -> None:
        self.running = False
        if self.monitoring_task:
            self.monitoring_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self.monitoring_task
        logger.info("Performance monitor stopped")

    async def _monitoring_loop(self) -> None:
        while self.running:
            try:
                await self._collect_system_metrics()
                await self._check_performance_thresholds()
                await self._generate_recommendations()
                await asyncio.sleep(30)
            except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
                logger.exception(f"Performance monitoring error: {e}")
                await asyncio.sleep(60)

    async def _collect_system_metrics(self) -> None:
        if psutil is None:
            return
        try:
            cpu_percent = psutil.cpu_percent(interval=1)
            # cpu_percent returns a value between 0 and 100, normalize to 0.0-1.0
            self._add_metric("cpu_usage", cpu_percent / 100.0, CPU_USAGE_THRESHOLD)
            memory = psutil.virtual_memory()
            self._add_metric("memory_usage", memory.percent / 100.0, MEMORY_USAGE_THRESHOLD)
            disk = psutil.disk_usage("/")
            self._add_metric("disk_usage", disk.percent / 100.0, 0.9)
        except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
            logger.exception(f"Error collecting system metrics: {e}")

    def _add_metric(self, name: str, value: float, threshold: float) -> None:
        status = "normal"
        if value > threshold:
            status = "critical"
        elif value > threshold * 0.8:
            status = "warning"
        metric = PerformanceMetric(
            name=name,
            value=value,
            timestamp=time.time(),
            threshold=threshold,
            status=status,
        )
        with self.lock:
            self.metrics[name].append(metric)

    async def _check_performance_thresholds(self) -> None:
        with self.lock:
            for metric_name, metrics in self.metrics.items():
                if not metrics:
                    continue
                latest = metrics[-1]
                if latest.status == "critical":
                    self.alerts.add_alert(metric_name, latest.value, latest.threshold, "critical")
                elif latest.status == "warning":
                    self.alerts.add_alert(metric_name, latest.value, latest.threshold, "warning")

    async def _generate_recommendations(self) -> None:
        recommendations: list[dict[str, Any]] = []
        avg_query_time = self._get_average_metric("db_query_time")
        if avg_query_time is not None and avg_query_time > DB_QUERY_THRESHOLD:
            recommendations.append(
                {
                    "type": "database",
                    "issue": "Slow database queries",
                    "recommendation": "Review query plans and add indexes where appropriate",
                    "severity": "high",
                },
            )
        cache_hit_rate = self._get_average_metric("cache_hit_rate")
        if cache_hit_rate is not None and cache_hit_rate < CACHE_HIT_RATE_THRESHOLD:
            recommendations.append(
                {
                    "type": "cache",
                    "issue": "Low cache hit rate",
                    "recommendation": "Increase TTL or adjust key strategy to improve locality",
                    "severity": "medium",
                },
            )
        cpu_usage = self._get_average_metric("cpu_usage")
        if cpu_usage is not None and cpu_usage > CPU_USAGE_THRESHOLD:
            recommendations.append(
                {
                    "type": "system",
                    "issue": "High CPU usage",
                    "recommendation": "Profile hot paths and consider concurrency limits or scaling",
                    "severity": "high",
                },
            )
        memory_usage = self._get_average_metric("memory_usage")
        if memory_usage is not None and memory_usage > MEMORY_USAGE_THRESHOLD:
            recommendations.append(
                {
                    "type": "system",
                    "issue": "High memory usage",
                    "recommendation": "Investigate object retention and optimize data structures",
                    "severity": "critical",
                },
            )
        if recommendations:
            self._store_recommendations(recommendations)
            logger.info(f"Generated {len(recommendations)} performance recommendations")

    def _get_average_metric(self, metric_name: str, window_minutes: int = 5) -> float | None:
        if metric_name not in self.metrics:
            return None
        cutoff_time = time.time() - (window_minutes * 60)
        values = [m.value for m in self.metrics[metric_name] if m.timestamp > cutoff_time]
        return (sum(values) / len(values)) if values else None

    def _get_latest_metric(self, metric_name: str) -> PerformanceMetric | None:
        if metric_name not in self.metrics or not self.metrics[metric_name]:
            return None
        return self.metrics[metric_name][-1]

    def _store_recommendations(self, recommendations: list[dict[str, Any]]) -> None:
        for rec in recommendations:
            logger.info(f"Performance recommendation: {rec['issue']} - {rec['recommendation']}")

    def track_db_query(self, query: str, execution_time: float) -> None:
        self._add_metric("db_query_time", execution_time, DB_QUERY_THRESHOLD)
        self.db_queries[query].append({"execution_time": execution_time, "timestamp": time.time()})
        if execution_time > DB_QUERY_THRESHOLD:
            logger.warning(f"Slow database query: {execution_time:.2f}s query={query[:120]}")

    def track_cache_operation(self, operation: str, hit: bool, response_time: float) -> None:
        if operation not in self.cache_operations:
            self.cache_operations[operation] = {"hits": 0, "misses": 0}
        if hit:
            self.cache_operations[operation]["hits"] += 1
        else:
            self.cache_operations[operation]["misses"] += 1
        total = self.cache_operations[operation]["hits"] + self.cache_operations[operation]["misses"]
        if total > 0:
            hit_rate = self.cache_operations[operation]["hits"] / total
            self._add_metric("cache_hit_rate", hit_rate, CACHE_HIT_RATE_THRESHOLD)
        self._add_metric("cache_response_time", response_time, 0.1)

    def track_model_operation(self, model_name: str, operation: str, duration: float) -> None:
        metric_name = f"model_{operation}_{model_name}"
        self._add_metric(metric_name, duration, 5.0)
        self.model_operations[model_name][operation].append({"duration": duration, "timestamp": time.time()})

    def get_performance_summary(self) -> dict[str, Any]:
        with self.lock:
            summary: dict[str, Any] = {
                "system_metrics": {},
                "database_metrics": {},
                "cache_metrics": {},
                "model_metrics": {},
                "alerts": self.alerts.get_alert_stats(),
                "recommendations": [],
            }
            for metric_name in ["cpu_usage", "memory_usage", "disk_usage"]:
                latest = self._get_latest_metric(metric_name)
                if latest:
                    summary["system_metrics"][metric_name] = {
                        "value": latest.value,
                        "status": latest.status,
                        "timestamp": latest.timestamp,
                    }
            avg_query_time = self._get_average_metric("db_query_time")
            if avg_query_time is not None:
                summary["database_metrics"]["avg_query_time"] = avg_query_time
                summary["database_metrics"]["total_queries"] = sum(len(v) for v in self.db_queries.values())
            cache_hit_rate = self._get_average_metric("cache_hit_rate")
            if cache_hit_rate is not None:
                summary["cache_metrics"]["hit_rate"] = cache_hit_rate
                summary["cache_metrics"]["total_operations"] = sum(ops.get("hits", 0) + ops.get("misses", 0) for ops in self.cache_operations.values())
            for model_name, operations in self.model_operations.items():
                summary["model_metrics"][model_name] = {}
                for operation, times in operations.items():
                    if times:
                        avg_duration = sum(t["duration"] for t in times) / len(times)
                        summary["model_metrics"][model_name][operation] = {
                            "avg_duration": avg_duration,
                            "total_operations": len(times),
                        }
            return summary

    def get_detailed_metrics(self, metric_name: str, minutes: int = 60) -> list[dict[str, Any]]:
        if metric_name not in self.metrics:
            return []
        cutoff_time = time.time() - (minutes * 60)
        return [{"value": m.value, "timestamp": m.timestamp, "status": m.status} for m in self.metrics[metric_name] if m.timestamp > cutoff_time]

    def clear_old_metrics(self, days: int = 7) -> None:
        cutoff_time = time.time() - (days * 24 * 60 * 60)
        with self.lock:
            for metric_name in list(self.metrics.keys()):
                self.metrics[metric_name] = deque(
                    (m for m in self.metrics[metric_name] if m.timestamp > cutoff_time),
                    maxlen=1000,
                )
        logger.info(f"Cleared metrics older than {days} days")


performance_monitor = PerformanceMonitor()
