"""
Enhanced Signal Manager Manager for Mystic Trading

Manages all trading signals and ensures they are active and properly integrated.
"""

import json
import logging
import time
from datetime import datetime, timezone
from typing import Any

# Import from single source of truth
try:
    from backend.config.trading_universe import TRADING_SYMBOLS
except (ImportError, ModuleNotFoundError, AttributeError, ValueError, TypeError, RuntimeError) as e:
    msg = f"Failed to import trading_universe: {e}"
    raise RuntimeError(msg) from e

from backend.notification_service import get_notification_service

logger = logging.getLogger(__name__)

# Use TRADING_SYMBOLS from trading_universe (live data)
ALLOWED_SYMBOLS = set(TRADING_SYMBOLS)


class SignalManager:
    """Core Signal Manager handling signal lifecycle and health"""

    def __init__(self, redis_client: Any) -> None:
        self.redis_client = redis_client
        self.notification_service = get_notification_service()
        self._health_check_interval = 30
        self._last_health_check = 0
        self.previous_health_state: dict[str, Any] | None = None
        self.auto_trading_enabled = False

        # Try to load existing signal types from redis, else initialize defaults
        try:
            raw = self.redis_client.get("signal_status")
            if raw:
                self.signal_types = json.loads(raw)
            else:
                now = datetime.now(timezone.utc).isoformat()
                # initialize a default set based on allowed symbols (small sample)
                self.signal_types = {
                    symbol: {
                        "enabled": True,
                        "status": "active",
                        "priority": "medium",
                        "update_interval": 30,
                        "last_update": now,
                    }
                    for symbol in ALLOWED_SYMBOLS
                }
                # Persist initial state
                self.redis_client.setex("signal_status", 3600, json.dumps(self.signal_types))
        except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError):
            # Fallback to in-memory defaults if redis access fails
            now = datetime.now(timezone.utc).isoformat()
            self.signal_types = {
                symbol: {
                    "enabled": True,
                    "status": "active",
                    "priority": "medium",
                    "update_interval": 30,
                    "last_update": now,
                }
                for symbol in ALLOWED_SYMBOLS
            }

    async def activate_all_signals(self) -> dict[str, Any]:
        try:
            for symbol in ALLOWED_SYMBOLS:
                self.signal_types[symbol] = {
                    "enabled": True,
                    "status": "active",
                    "priority": "medium",
                    "update_interval": 30,
                    "last_update": datetime.now(timezone.utc).isoformat(),
                }
            try:
                self.redis_client.setex("signal_status", 3600, json.dumps(self.signal_types))
            except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError):
                # redis may be unavailable; continue with in-memory state
                logger.warning("Could not persist signal_status to redis")
            await self.notification_service.send_notification(
                "Signals activated",
                f"Activated {len(self.signal_types)} signals",
                "info",
            )
            return {
                "status": "success",
                "message": f"Activated {len(self.signal_types)} signals",
                "signals": self.signal_types,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
        except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
            logger.exception(f"Error activating all signals: {e!s}")
            raise

    async def get_signal_status(self) -> dict[str, Any]:
        try:
            raw = None
            try:
                raw = self.redis_client.get("signal_status")
            except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError):
                raw = None
            signals = json.loads(raw) if raw else self.signal_types
            return {
                "status": "success",
                "signals": signals,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
        except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
            logger.exception(f"Error getting signal status: {e!s}")
            raise

    async def generate_live_signal(self, symbol: str) -> dict[str, Any]:
        try:
            symbol = (symbol or "").upper()
            if symbol not in ALLOWED_SYMBOLS:
                return {"status": "error", "message": f"Symbol {symbol} not supported"}
            # Simple synthetic live signal generator
            now = datetime.now(timezone.utc)
            ts = now.isoformat()
            # derive deterministic pseudo-price from timestamp to avoid random import
            price = round((time.time() % 10000) + 0.01, 2)
            # strength between 0 and 1 based on seconds % 100
            strength = (int(time.time()) % 100) / 100.0
            signal = {
                "symbol": symbol,
                "strength": round(strength, 3),
                "price": price,
                "price_change_pct": 0.0,
                "timestamp": ts,
                "source": "synthetic_ticker",
            }
        except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
            logger.exception(f"Error generating live signal: {e!s}")
            raise
        else:
            return {"status": "success", "signal": signal, "timestamp": ts}

    async def start_auto_trading(self) -> dict[str, Any]:
        try:
            logger.info("Starting auto-trading...")
            signal_data = None
            try:
                signal_data = self.redis_client.get("signal_status")
            except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError):
                signal_data = None
            if not signal_data and not self.signal_types:
                return {
                    "status": "error",
                    "message": "Cannot start auto-trading: signals not active",
                }
            self.auto_trading_enabled = True
            try:
                self.redis_client.setex(
                    "auto_trading_enabled",
                    3600,
                    json.dumps(
                        {
                            "enabled": True,
                            "started_at": datetime.now(timezone.utc).isoformat(),
                            "status": "running",
                        },
                    ),
                )
            except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError):
                logger.warning("Could not persist auto_trading_enabled to redis")
            await self.notification_service.send_notification(
                "Auto-trading started",
                "Automated trading has been activated",
                "info",
            )
            logger.info("Auto-trading started successfully")
            return {
                "status": "success",
                "message": "Auto-trading started",
                "auto_trading_enabled": True,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
        except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
            logger.exception(f"Error starting auto-trading: {e!s}")
            raise

    async def stop_auto_trading(self) -> dict[str, Any]:
        try:
            logger.info("Stopping auto-trading...")
            self.auto_trading_enabled = False
            try:
                self.redis_client.setex(
                    "auto_trading_enabled",
                    3600,
                    json.dumps(
                        {
                            "enabled": False,
                            "stopped_at": datetime.now(timezone.utc).isoformat(),
                            "status": "stopped",
                        },
                    ),
                )
            except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError):
                logger.warning("Could not persist auto_trading_enabled to redis")
            await self.notification_service.send_notification(
                "Auto-trading stopped",
                "Automated trading has been deactivated",
                "warning",
            )
            logger.info("Auto-trading stopped successfully")
            return {
                "status": "success",
                "message": "Auto-trading stopped",
                "auto_trading_enabled": False,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
        except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
            logger.exception(f"Error stopping auto-trading: {e!s}")
            raise

    async def get_auto_trade_status(self) -> dict[str, Any]:
        try:
            auto_trade_data = None
            try:
                auto_trade_data = self.redis_client.get("auto_trading_enabled")
            except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError):
                auto_trade_data = None
            auto_trading = json.loads(auto_trade_data) if auto_trade_data else {"enabled": self.auto_trading_enabled}
            return {
                "status": "success",
                "auto_trading": auto_trading,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
        except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
            logger.exception(f"Error getting auto-trade status: {e!s}")
            raise

    async def self_heal_signals(self) -> dict[str, Any]:
        try:
            logger.info("Starting signal self-healing...")
            healed_signals: list[str] = []
            failed_signals: list[dict[str, str]] = []
            for signal_type, config in list(self.signal_types.items()):
                try:
                    if not config.get("enabled", False):
                        config["enabled"] = True
                        config["status"] = "healed"
                        config["last_update"] = datetime.now(timezone.utc).isoformat()
                        healed_signals.append(signal_type)
                        logger.info(f"Healed signal: {signal_type}")
                except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
                    failed_signals.append({"signal": signal_type, "error": str(e)})
                    logger.exception(f"Failed to heal signal {signal_type}: {e!s}")
            try:
                self.redis_client.setex("signal_status", 3600, json.dumps(self.signal_types))
            except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError):
                logger.warning("Could not persist healed signal_status to redis")
            if healed_signals:
                await self.notification_service.send_notification(
                    "Signals healed",
                    f"Successfully healed {len(healed_signals)} signals",
                    "success",
                )
            logger.info(f"Self-healing completed. Healed: {len(healed_signals)}, Failed: {len(failed_signals)}")
            return {
                "status": "success",
                "healed_signals": healed_signals,
                "failed_signals": failed_signals,
                "total_healed": len(healed_signals),
                "total_failed": len(failed_signals),
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
        except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
            logger.exception(f"Error during self-healing: {e!s}")
            raise

    async def check_signal_health(self) -> dict[str, Any]:
        try:
            current_time = time.time()
            if current_time - self._last_health_check < self._health_check_interval:
                return {
                    "status": "success",
                    "message": "Health check rate limited",
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                }
            self._last_health_check = current_time
            raw = None
            try:
                raw = self.redis_client.get("signal_status")
            except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError):
                raw = None
            signals = json.loads(raw) if raw else self.signal_types
            signal_health: dict[str, Any] = {}
            healthy_signals = 0
            unhealthy_signals = 0
            for signal_type, config in signals.items():
                is_healthy = config.get("enabled", False) and config.get("status") == "active"
                signal_health[signal_type] = {
                    "healthy": is_healthy,
                    "enabled": config.get("enabled", False),
                    "status": config.get("status", "unknown"),
                    "last_update": config.get("last_update"),
                    "priority": config.get("priority", "medium"),
                }
                if is_healthy:
                    healthy_signals += 1
                else:
                    unhealthy_signals += 1
            total_signals = len(signals)
            health_percentage = (healthy_signals / total_signals) * 100 if total_signals > 0 else 0
            if health_percentage >= 90:
                overall_health = "healthy"
            elif health_percentage >= 70:
                overall_health = "warning"
            else:
                overall_health = "critical"
            current_health = {
                "overall_health": overall_health,
                "health_percentage": health_percentage,
                "healthy_signals": healthy_signals,
                "unhealthy_signals": unhealthy_signals,
            }
            await self._check_health_changes(current_health)
            auto_status = await self.get_auto_trade_status()
            return {
                "status": "success",
                "overall_health": overall_health,
                "health_percentage": health_percentage,
                "healthy_signals": healthy_signals,
                "unhealthy_signals": unhealthy_signals,
                "signal_health": signal_health,
                "auto_trading": auto_status.get("auto_trading", {}),
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
        except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
            logger.exception(f"Error checking signal health: {e!s}")
            raise

    async def _check_health_changes(self, current_health: dict[str, Any]):
        if self.previous_health_state is None:
            self.previous_health_state = current_health
            return
        prev_health = self.previous_health_state.get("overall_health")
        curr_health = current_health.get("overall_health")
        if prev_health != curr_health:
            message = f"Signal health changed from {prev_health} to {curr_health}"
            notification_type = "warning" if curr_health in ["warning", "critical"] else "info"
            try:
                await self.notification_service.send_notification("Signal Health Change", message, notification_type)
            except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError):
                logger.warning("Failed to send health change notification")
        self.previous_health_state = current_health


class SignalManagerManager:
    """Manager for the Signal Manager system"""

    def __init__(self, redis_client: Any) -> None:
        self.redis_client = redis_client
        self.signal_manager = SignalManager(redis_client)
        self._health_check_interval = 30
        self._last_health_check = 0

    async def activate_all_signals(self) -> dict[str, Any]:
        return await self.signal_manager.activate_all_signals()

    async def get_signal_status(self) -> dict[str, Any]:
        return await self.signal_manager.get_signal_status()

    async def generate_live_signal(self, symbol: str) -> dict[str, Any]:
        return await self.signal_manager.generate_live_signal(symbol)

    async def start_auto_trading(self) -> dict[str, Any]:
        return await self.signal_manager.start_auto_trading()

    async def stop_auto_trading(self) -> dict[str, Any]:
        return await self.signal_manager.stop_auto_trading()

    async def get_auto_trade_status(self) -> dict[str, Any]:
        return await self.signal_manager.get_auto_trade_status()

    async def self_heal_signals(self) -> dict[str, Any]:
        return await self.signal_manager.self_heal_signals()

    async def check_signal_health(self) -> dict[str, Any]:
        return await self.signal_manager.check_signal_health()

    async def get_signal_performance_metrics(self) -> dict[str, Any]:
        try:
            signal_status = await self.get_signal_status()
            signals = signal_status.get("signals", {})

            metrics: dict[str, Any] = {
                "total_signals": len(signals),
                "active_signals": 0,
                "inactive_signals": 0,
                "high_priority_signals": 0,
                "medium_priority_signals": 0,
                "low_priority_signals": 0,
                "critical_priority_signals": 0,
                "average_update_interval": 0,
                "signal_types": {},
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }

            total_interval = 0
            interval_count = 0

            for signal_type, config in signals.items():
                if config.get("enabled", False):
                    metrics["active_signals"] += 1
                else:
                    metrics["inactive_signals"] += 1

                priority = config.get("priority", "medium")
                if priority == "high":
                    metrics["high_priority_signals"] += 1
                elif priority == "medium":
                    metrics["medium_priority_signals"] += 1
                elif priority == "low":
                    metrics["low_priority_signals"] += 1
                elif priority == "critical":
                    metrics["critical_priority_signals"] += 1

                update_interval = config.get("update_interval", 0)
                if update_interval > 0:
                    total_interval += update_interval
                    interval_count += 1

                metrics["signal_types"][signal_type] = {
                    "enabled": config.get("enabled", False),
                    "priority": priority,
                    "update_interval": update_interval,
                    "last_update": config.get("last_update"),
                    "status": config.get("status", "unknown"),
                }

            if interval_count > 0:
                metrics["average_update_interval"] = total_interval / interval_count
        except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
            logger.exception(f"Error getting signal performance metrics: {e!s}")
            raise
        else:
            return metrics

    async def get_trading_strategy_metrics(self) -> dict[str, Any]:
        try:
            signal_status = await self.get_signal_status()
            strategies = signal_status.get("strategies", {})

            metrics: dict[str, Any] = {
                "total_strategies": len(strategies),
                "enabled_strategies": 0,
                "disabled_strategies": 0,
                "average_confidence": 0.0,
                "strategy_details": {},
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }

            total_confidence = 0.0
            confidence_count = 0

            for strategy_name, config in strategies.items():
                if config.get("enabled", False):
                    metrics["enabled_strategies"] += 1
                else:
                    metrics["disabled_strategies"] += 1

                confidence = config.get("min_confidence", 0.0)
                if confidence > 0:
                    total_confidence += confidence
                    confidence_count += 1

                metrics["strategy_details"][strategy_name] = {
                    "enabled": config.get("enabled", False),
                    "min_confidence": confidence,
                    "description": self._get_strategy_description(strategy_name),
                }

            if confidence_count > 0:
                metrics["average_confidence"] = total_confidence / confidence_count
        except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
            logger.exception(f"Error getting trading strategy metrics: {e!s}")
            raise
        else:
            return metrics

    def _get_strategy_description(self, strategy_name: str) -> str:
        descriptions = {
            "daying": "High-frequency trading with small profits",
            "swing_trading": "Medium-term position holding",
            "arbitrage": "Exploiting temporary price discrepancies",
            "momentum": "Following price momentum trends",
            "mean_reversion": "Trading price reversals to the mean",
            "grid_trading": "Automated grid-based trading",
            "statistical_arbitrage": "Statistical arbitrage opportunities",
            "market_making": "Providing liquidity to the market",
            "high_frequency_trading": "Ultra-fast algorithmic trading",
            "options_trading": "Options and derivatives trading",
            "futures_trading": "Futures contract trading",
        }
        return descriptions.get(strategy_name, "Unknown strategy")

    async def get_system_health_summary(self) -> dict[str, Any]:
        try:
            health_data = await self.check_signal_health()
            performance_metrics = await self.get_signal_performance_metrics()
            strategy_metrics = await self.get_trading_strategy_metrics()
            auto_trade_status = await self.get_auto_trade_status()

            return {
                "overall_health": health_data.get("overall_health", "unknown"),
                "signal_health": health_data.get("signal_health", {}),
                "performance_metrics": performance_metrics,
                "strategy_metrics": strategy_metrics,
                "auto_trading": auto_trade_status.get("auto_trading", {}),
                "recommendations": self._generate_health_recommendations(health_data, performance_metrics, strategy_metrics),
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }

        except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
            logger.exception(f"Error getting system health summary: {e!s}")
            raise

    def _generate_health_recommendations(
        self,
        health_data: dict[str, Any],
        performance_metrics: dict[str, Any],
        strategy_metrics: dict[str, Any],
    ) -> list[str]:
        recommendations: list[str] = []

        overall_health = health_data.get("overall_health", "unknown")
        if overall_health != "healthy":
            recommendations.append(f"Signal system health is {overall_health}. Consider running self-healing.")

        inactive_signals = performance_metrics.get("inactive_signals", 0)
        if inactive_signals > 0:
            recommendations.append(f"{inactive_signals} signals are inactive. Review and reactivate if needed.")

        auto_trading = health_data.get("auto_trading", {})
        if not auto_trading.get("enabled", False):
            recommendations.append("Auto-trading is disabled. Enable for automated trading.")

        avg_confidence = strategy_metrics.get("average_confidence", 0.0)
        if avg_confidence < 0.6:
            recommendations.append("Average strategy confidence is low. Review strategy parameters.")

        avg_interval = performance_metrics.get("average_update_interval", 0)
        if avg_interval > 60:
            recommendations.append("Average update interval is high; consider reducing update intervals.")

        # Additional basic recommendations
        if health_data.get("healthy_signals", 0) < 1:
            recommendations.append("No healthy signals detected. Run self-healing and inspect sources.")

        return recommendations


def get_signal_manager_manager(redis_client: Any) -> SignalManagerManager:
    return SignalManagerManager(redis_client)
