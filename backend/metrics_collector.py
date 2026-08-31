"""
Metrics Collector for Mystic Trading

Collects and exposes comprehensive metrics for Prometheus monitoring.
Includes signal health, auto-trading status, test results, and system performance.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import time
from datetime import datetime, timezone
from typing import Any

import redis
from prometheus_client import (
    CONTENT_TYPE_LATEST,
    Counter,
    Gauge,
    Histogram,
    Info,
    generate_latest,
)

try:
    from backend.services.websocket_manager import websocket_manager  # type: ignore[import-not-found]
except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError):
    websocket_manager = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)


class MetricsCollector:
    def __init__(self, redis_client: redis.Redis) -> None:
        self.redis_client = redis_client

        self.signal_health_gauge = Gauge(
            "mystic_signal_health_status",
            "Overall signal health status (0=critical, 1=degraded, 2=healthy)",
            ["component"],
        )
        self.active_signals_gauge = Gauge(
            "mystic_active_signals_total",
            "Number of active signals",
            ["signal_type"],
        )
        self.active_strategies_gauge = Gauge(
            "mystic_active_strategies_total",
            "Number of active trading strategies",
            ["strategy_type"],
        )
        self.auto_trading_status_gauge = Gauge(
            "mystic_auto_trading_status",
            "Auto-trading status (0=disabled, 1=enabled)",
        )

        self.test_runs_total = Counter("mystic_test_runs_total", "Total number of test runs", ["status"])
        self.test_success_rate_gauge = Gauge("mystic_test_success_rate", "Test success rate percentage")
        self.test_duration_histogram = Histogram(
            "mystic_test_duration_seconds",
            "Test run duration in seconds",
            buckets=[10, 30, 60, 120, 300, 600],
        )
        self.self_healing_triggered_total = Counter(
            "mystic_self_healing_triggered_total",
            "Number of times self-healing was triggered",
            ["reason"],
        )

        self.notifications_sent_total = Counter(
            "mystic_notifications_sent_total",
            "Number of notifications sent",
            ["channel", "level"],
        )
        self.notification_delivery_duration = Histogram(
            "mystic_notification_delivery_seconds",
            "Notification delivery duration in seconds",
            ["channel"],
            buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0],
        )

        self.api_requests_total = Counter(
            "mystic_api_requests_total",
            "Total number of API requests",
            ["method", "endpoint", "status"],
        )
        self.api_request_duration = Histogram(
            "mystic_api_request_duration_seconds",
            "API request duration in seconds",
            ["method", "endpoint"],
            buckets=[0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0],
        )

        self.websocket_connections_gauge = Gauge(
            "mystic_websocket_connections_active",
            "Number of active WebSocket connections",
        )
        self.websocket_messages_sent_total = Counter(
            "mystic_websocket_messages_sent_total",
            "Total number of WebSocket messages sent",
        )

        self.system_uptime_gauge = Gauge("mystic_system_uptime_seconds", "System uptime in seconds")
        self.service_health_gauge = Gauge(
            "mystic_service_health_status",
            "Service health status (0=unhealthy, 1=healthy)",
            ["service"],
        )

        self.live_signals_generated_total = Counter(
            "mystic_live_signals_generated_total",
            "Total number of live signals generated",
            ["symbol", "signal_type"],
        )
        self.signal_confidence_histogram = Histogram(
            "mystic_signal_confidence",
            "Signal confidence levels",
            ["signal_type"],
            buckets=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
        )

        self.redis_operations_total = Counter(
            "mystic_redis_operations_total",
            "Total number of Redis operations",
            ["operation", "status"],
        )
        self.redis_operation_duration = Histogram(
            "mystic_redis_operation_duration_seconds",
            "Redis operation duration in seconds",
            ["operation"],
            buckets=[0.001, 0.005, 0.01, 0.05, 0.1, 0.5],
        )

        self.app_info = Info("mystic_app", "Application information")
        self.app_info.info(
            {
                "version": "1.0.0",
                "name": "Mystic Trading Bot",
                "description": "Advanced trading bot with automated signals and self-healing",
            }
        )

        self.start_time = time.time()

    def update_signal_health_metrics(self, health_data: dict[str, Any]) -> None:
        try:
            overall_health = health_data.get("overall_health", "unknown")
            health_value = {"healthy": 2, "degraded": 1, "critical": 0}.get(overall_health, 0)
            self.signal_health_gauge.labels(component="overall").set(health_value)

            signals = health_data.get("signals", {}) or {}
            self.active_signals_gauge.labels(signal_type="total").set(float(signals.get("total", 0)))
            self.active_signals_gauge.labels(signal_type="healthy").set(float(signals.get("healthy", 0)))
            self.active_signals_gauge.labels(signal_type="unhealthy").set(float(signals.get("unhealthy", 0)))

            strategies = health_data.get("strategies", {}) or {}
            self.active_strategies_gauge.labels(strategy_type="total").set(float(strategies.get("total", 0)))
            self.active_strategies_gauge.labels(strategy_type="healthy").set(float(strategies.get("healthy", 0)))
            self.active_strategies_gauge.labels(strategy_type="unhealthy").set(float(strategies.get("unhealthy", 0)))

            auto_trading = health_data.get("auto_trading", {}) or {}
            auto_trading_status = 1 if auto_trading.get("healthy", False) else 0
            self.auto_trading_status_gauge.set(float(auto_trading_status))
        except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
            logger.exception(f"Error updating signal health metrics: {e!s}")

    def update_test_metrics(self, test_run: dict[str, Any]) -> None:
        try:
            status = "success" if int(test_run.get("failed_tests", 0)) == 0 else "failure"
            self.test_runs_total.labels(status=status).inc()
            success_rate = float(test_run.get("success_rate", 0.0) or 0.0)
            self.test_success_rate_gauge.set(success_rate)

            start_str = test_run.get("start_time", "")
            end_str = test_run.get("end_time", "")
            if start_str and end_str:
                try:
                    start_time = datetime.fromisoformat(str(start_str).replace("Z", "+00:00"))
                    end_time = datetime.fromisoformat(str(end_str).replace("Z", "+00:00"))
                except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError):
                    # fallback to numeric timestamps (seconds since epoch)
                    try:
                        start_time = datetime.fromtimestamp(float(start_str), tz=timezone.utc)
                        end_time = datetime.fromtimestamp(float(end_str), tz=timezone.utc)
                    except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError):
                        start_time = None
                        end_time = None

                if start_time and end_time:
                    duration = max(0.0, (end_time - start_time).total_seconds())
                    self.test_duration_histogram.observe(duration)

            if test_run.get("triggered_healing", False):
                self.self_healing_triggered_total.labels(reason="test_failure").inc()
        except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
            logger.exception(f"Error updating test metrics: {e!s}")

    def update_notification_metrics(self, notification_result: dict[str, Any]) -> None:
        try:
            channels = notification_result.get("channels", {}) or {}
            level = notification_result.get("level", "unknown")
            # channels may be a dict mapping channel->result or channel->bool
            if isinstance(channels, dict):
                items = channels.items()
            else:
                # if channels provided as list or other, try to iterate
                try:
                    items = list(channels)
                except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError):
                    items = []

            for channel, result in items:
                # normalize result
                success = False
                duration = None
                if isinstance(result, dict):
                    success = bool(result.get("success", False))
                    duration = result.get("duration")
                else:
                    # treat non-dict truthy value as success
                    success = bool(result)
                if success:
                    self.notifications_sent_total.labels(channel=channel, level=level).inc()
                    if duration is not None:
                        with contextlib.suppress(ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError):
                            self.notification_delivery_duration.labels(channel=channel).observe(float(duration))
        except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
            logger.exception(f"Error updating notification metrics: {e!s}")

    def update_api_metrics(self, method: str, endpoint: str, status: int, duration: float) -> None:
        try:
            status_category = f"{int(status) // 100}xx"
            self.api_requests_total.labels(method=method, endpoint=endpoint, status=status_category).inc()
            self.api_request_duration.labels(method=method, endpoint=endpoint).observe(float(duration))
        except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
            logger.exception(f"Error updating API metrics: {e!s}")

    def update_websocket_metrics(self, connections: int, messages_sent: int = 0) -> None:
        try:
            self.websocket_connections_gauge.set(float(connections))
            if messages_sent and messages_sent > 0:
                # Counter.inc accepts a positive numeric amount
                self.websocket_messages_sent_total.inc(float(messages_sent))
        except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
            logger.exception(f"Error updating WebSocket metrics: {e!s}")

    def update_system_metrics(self) -> None:
        try:
            uptime = time.time() - self.start_time
            self.system_uptime_gauge.set(float(uptime))

            if not self.redis_client:
                return

            raw = None
            try:
                raw = self.redis_client.get("service_health")
            except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError):
                raw = None

            if raw:
                try:
                    health_str = raw.decode("utf-8") if isinstance(raw, (bytes, bytearray)) else str(raw)
                    services = json.loads(health_str)
                    if isinstance(services, dict):
                        for service, status in services.items():
                            try:
                                # status expected to be a dict like {"status": "healthy"}
                                health_value = (1 if str(status.get("status", "")).lower() == "healthy" else 0) if isinstance(status, dict) else (1 if str(status).lower() == "healthy" else 0)
                                self.service_health_gauge.labels(service=service).set(float(health_value))
                            except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError):
                                # skip malformed entries
                                continue
                except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError):
                    # ignore parsing errors
                    pass
        except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
            logger.exception(f"Error updating system metrics: {e!s}")

    def update_trading_metrics(self, signal_data: dict[str, Any]) -> None:
        try:
            from backend.services.confidence_normalizer import ConfidenceNormalizer

            symbol = str(signal_data.get("symbol", "unknown"))
            signal_type = str(signal_data.get("signal_type", "unknown"))
            raw = float(signal_data.get("confidence", 0.0) or 0.0)
            confidence = ConfidenceNormalizer.normalize(raw)

            self.live_signals_generated_total.labels(symbol=symbol, signal_type=signal_type).inc()
            self.signal_confidence_histogram.labels(signal_type=signal_type).observe(confidence)

            if websocket_manager is not None:
                try:
                    loop = asyncio.get_running_loop()
                    coro = websocket_manager.broadcast_json(
                        {
                            "type": "metrics_update",
                            "data": {
                                "symbol": symbol,
                                "signal_type": signal_type,
                                "confidence": confidence,
                                "timestamp": time.time(),
                            },
                        }
                    )
                    # ensure coro is awaitable before scheduling
                    if asyncio.iscoroutine(coro):
                        task = loop.create_task(coro)
                        # Store task reference if MetricsCollector has task tracking
                        if hasattr(self, "_tasks"):
                            self._tasks.append(task)
                        elif not hasattr(self, "_tasks"):
                            self._tasks: list[asyncio.Task[Any]] = []
                            self._tasks.append(task)
                except RuntimeError:
                    # no running loop; skip broadcasting
                    pass
                except (ValueError, TypeError, AttributeError, KeyError, IndexError):
                    # any other error from websocket manager should not break metrics
                    pass
        except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
            logger.exception(f"Error updating trading metrics: {e!s}")

    def update_redis_metrics(self, operation: str, status: str, duration: float) -> None:
        try:
            self.redis_operations_total.labels(operation=operation, status=status).inc()
            self.redis_operation_duration.labels(operation=operation).observe(float(duration))
        except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
            logger.exception(f"Error updating Redis metrics: {e!s}")

    def record_self_healing_event(self, reason: str, actions_taken: int) -> None:
        try:
            _ = actions_taken
            self.self_healing_triggered_total.labels(reason=reason).inc()
            logger.info(f"Self-healing triggered: {reason}")
        except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
            logger.exception(f"Error recording self-healing event: {e!s}")

    def get_metrics(self) -> str:
        try:
            output = generate_latest()
            if isinstance(output, (bytes, bytearray)):
                return output.decode("utf-8")
            return str(output)
        except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
            logger.exception(f"Error generating metrics: {e!s}")
            return ""

    def get_metrics_content_type(self) -> str:
        return CONTENT_TYPE_LATEST


# Metrics collector state - using dict to avoid global keyword
_metrics_collector_state: dict[str, MetricsCollector | None] = {"instance": None}


def get_metrics_collector(redis_client: redis.Redis) -> MetricsCollector:
    if _metrics_collector_state["instance"] is None:
        _metrics_collector_state["instance"] = MetricsCollector(redis_client)
    return _metrics_collector_state["instance"]
