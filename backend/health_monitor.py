"""
Health Monitoring Service for Mystic Trading

Monitors the health of all system components and performs self-healing when needed.
"""

import asyncio
import contextlib
import logging
from datetime import datetime, timezone
from typing import Any

from backend.services.task_manager import task_manager
from backend.services.websocket_manager import websocket_manager

logger = logging.getLogger(__name__)

# Health monitor timing constant
HEALTH_MONITOR_CHECK_INTERVAL = 60.0  # Check health every minute


class HealthMonitor:
    """Monitors the health of all system components and performs self-healing when needed."""

    def __init__(
        self,
        signal_manager: Any,
        auto_trading_manager: Any,
        notification_service: Any,
        metrics_collector: Any | None = None,
    ) -> None:
        self.signal_manager = signal_manager
        self.auto_trading_manager = auto_trading_manager
        self.notification_service = notification_service
        self.metrics_collector = metrics_collector
        self.is_running = False
        self.monitor_task: asyncio.Task | None = None
        self.service_health: dict[str, dict[str, Any]] = {}
        self.system_metrics: dict[str, dict[str, Any]] = {}
        # Track background tasks for proper cleanup
        self._tasks: list[asyncio.Task[Any]] = []

    async def start_monitoring(self) -> None:
        if self.is_running:
            logger.warning("Health monitoring is already running")
            return
        self.is_running = True
        self.monitor_task = await task_manager.create_task(self._monitor_system_health(), name="health_monitor:monitor_system_health")
        logger.info("System health monitoring task started")

    async def stop_monitoring(self) -> None:
        if not self.is_running:
            logger.warning("Health monitoring is not running")
            return
        self.is_running = False
        if self.monitor_task:
            self.monitor_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self.monitor_task
        logger.info("System health monitoring task stopped")

    async def _monitor_system_health(self) -> None:
        logger.info("Starting system health monitoring background task...")
        while self.is_running:
            try:
                try:
                    signal_health = await self.signal_manager.check_signal_health()
                    auto_trading_health = await self.auto_trading_manager.check_health()

                    overall_health = signal_health.get("overall_health", "unknown")
                    if not auto_trading_health.get("healthy", False) and overall_health == "healthy":
                        overall_health = "degraded"

                    if overall_health in ["degraded", "critical"]:
                        logger.warning(
                            "System health is %s, triggering self-healing...",
                            overall_health,
                        )

                        signal_healing_result = await self.signal_manager.self_heal_signals()
                        auto_trading_healing_result = await self.auto_trading_manager.self_heal()

                        if signal_healing_result.get("healing_performed") or auto_trading_healing_result.get("status") == "healed":
                            logger.info(
                                "Self-healing completed: Signals=%s, Auto-trading=%s",
                                signal_healing_result.get("actions_taken", []),
                                auto_trading_healing_result.get("message", "No action"),
                            )
                            try:
                                await self.notification_service.send_notification(
                                    title="System Self-Healing Performed",
                                    message=(
                                        f"Automatic healing actions were taken: "
                                        f"Signals: {signal_healing_result.get('actions_taken', [])}, "
                                        f"Auto-trading: {auto_trading_healing_result.get('message', 'No action')}"
                                    ),
                                    level="info",
                                    channels=["in_app"],
                                )
                            except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as notify_error:
                                logger.exception(
                                    "Failed to send healing notification: %s",
                                    notify_error,
                                )
                        else:
                            logger.info("No healing actions needed")

                    if overall_health != "healthy":
                        logger.info(
                            "Current system health: %s (Signals: %s, Auto-trading: %s)",
                            overall_health,
                            signal_health.get("overall_health"),
                            auto_trading_health.get("healthy"),
                        )
                except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as health_check_error:
                    logger.exception("Error checking system health: %s", health_check_error)
            except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
                logger.exception("Error in system health monitoring: %s", e)
                try:
                    await self.notification_service.send_notification(
                        title="Health Monitoring Error",
                        message=f"Background health monitoring failed: {e!s}",
                        level="error",
                        channels=["in_app"],
                    )
                except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as notify_error:
                    logger.exception("Failed to send notification: %s", notify_error)

            # Regular health check interval
            await asyncio.sleep(HEALTH_MONITOR_CHECK_INTERVAL)

    async def check_health(self) -> dict[str, Any]:
        try:
            signal_health = await self.signal_manager.check_signal_health()
            auto_trading_health = await self.auto_trading_manager.check_health()
            combined_health = {
                "status": "success",
                "overall_health": signal_health.get("overall_health", "unknown"),
                "signals": signal_health.get("signals", {}),
                "strategies": signal_health.get("strategies", {}),
                "auto_trading": auto_trading_health,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
            if not auto_trading_health.get("healthy", False) and combined_health["overall_health"] == "healthy":
                combined_health["overall_health"] = "degraded"
        except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
            logger.exception("Error getting system health: %s", e)
            return {
                "status": "error",
                "message": f"Error getting system health: {e!s}",
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
        else:
            return combined_health

    async def perform_self_healing(self) -> dict[str, Any]:
        try:
            signal_result = await self.signal_manager.self_heal_signals()
            auto_trading_result = await self.auto_trading_manager.self_heal()
            combined_result = {
                "status": "success",
                "signals": signal_result,
                "auto_trading": auto_trading_result,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
            if signal_result.get("healing_performed", False) and self.metrics_collector:
                actions_taken = len(signal_result.get("actions_taken", []))
                try:
                    self.metrics_collector.record_self_healing_event("manual_trigger", actions_taken)
                except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
                    logger.exception("Failed to record self-healing event: %s", e)
        except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
            logger.exception("Error performing self-healing: %s", e)
            return {
                "status": "error",
                "message": f"Error performing self-healing: {e!s}",
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
        else:
            return combined_result

    def _safe_create_task(self, coro: Any) -> None:
        try:
            loop = asyncio.get_running_loop()
            task = loop.create_task(coro)
            self._tasks.append(task)
        except RuntimeError:
            try:
                loop = asyncio.new_event_loop()
                loop.run_until_complete(coro)
                loop.close()
            except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
                logger.exception("Failed to dispatch async task: %s", e)

    def update_service_health(
        self,
        service_name: str,
        status: str,
        details: dict[str, Any] | None = None,
    ) -> None:
        try:
            self.service_health[service_name] = {
                "status": status,
                "last_check": datetime.now(timezone.utc).isoformat(),
                "details": details or {},
            }
            self._safe_create_task(
                websocket_manager.broadcast_json(
                    {
                        "type": "health_update",
                        "data": {
                            "service": service_name,
                            "status": status,
                            "last_check": self.service_health[service_name]["last_check"],
                            "details": details or {},
                        },
                    }
                )
            )
            logger.info("Health updated for %s: %s", service_name, status)
        except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
            logger.exception("Error updating health for %s: %s", service_name, e)

    def update_system_metrics(self, metrics: dict[str, Any]) -> None:
        try:
            for k, v in metrics.items():
                if isinstance(v, dict):
                    self.system_metrics[k] = v
                else:
                    self.system_metrics[k] = {"value": v}
            self.system_metrics["last_update"] = {"value": datetime.now(timezone.utc).isoformat()}
            self._safe_create_task(websocket_manager.broadcast_json({"type": "system_metrics", "data": self.system_metrics}))
        except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
            logger.exception("Error updating system metrics: %s", e)


# Health monitor state - using dict to avoid global keyword
_health_monitor_state: dict[str, HealthMonitor | None] = {"instance": None}


def get_health_monitor(
    signal_manager: Any,
    auto_trading_manager: Any,
    notification_service: Any,
    metrics_collector: Any | None = None,
) -> HealthMonitor:
    if _health_monitor_state["instance"] is None:
        _health_monitor_state["instance"] = HealthMonitor(
            signal_manager,
            auto_trading_manager,
            notification_service,
            metrics_collector,
        )
    return _health_monitor_state["instance"]
