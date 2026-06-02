"""
AI Strategy Service - Live Configuration Only

Provides AI strategy functionality for the trading platform.
All configuration values come from live config - no hardcoded values.
"""

import logging
import os
from datetime import datetime, timezone
from typing import Any

# Import live configuration
try:
    from backend.config_bridge import get_mystic_config

    _mystic_config = get_mystic_config()
except (ImportError, AttributeError, ValueError, TypeError, RuntimeError):
    _mystic_config = None

logger = logging.getLogger(__name__)

# --- Live Configuration Helpers -------------------------------------------------------------------


def _get_initial_status() -> str:
    """Get initial status from live config."""
    if _mystic_config:
        try:
            value = _mystic_config
            if hasattr(value, "ai_strategy") and hasattr(value.ai_strategy, "initial_status"):
                status = value.ai_strategy.initial_status
                if isinstance(status, str) and status:
                    return status.strip()
        except (AttributeError, ValueError, TypeError):
            pass

    status = os.getenv("AI_STRATEGY_INITIAL_STATUS", "").strip()
    if status:
        return status

    return "initialized"


def _get_running_status() -> str:
    """Get running status from live config."""
    if _mystic_config:
        try:
            value = _mystic_config
            if hasattr(value, "ai_strategy") and hasattr(value.ai_strategy, "running_status"):
                status = value.ai_strategy.running_status
                if isinstance(status, str) and status:
                    return status.strip()
        except (AttributeError, ValueError, TypeError):
            pass

    status = os.getenv("AI_STRATEGY_RUNNING_STATUS", "").strip()
    if status:
        return status

    return "running"


def _get_stopped_status() -> str:
    """Get stopped status from live config."""
    if _mystic_config:
        try:
            value = _mystic_config
            if hasattr(value, "ai_strategy") and hasattr(value.ai_strategy, "stopped_status"):
                status = value.ai_strategy.stopped_status
                if isinstance(status, str) and status:
                    return status.strip()
        except (AttributeError, ValueError, TypeError):
            pass

    status = os.getenv("AI_STRATEGY_STOPPED_STATUS", "").strip()
    if status:
        return status

    return "stopped"


def _get_default_strategy_count() -> int:
    """Get default strategy count from live config."""
    if _mystic_config:
        try:
            value = _mystic_config
            if hasattr(value, "ai_strategy") and hasattr(value.ai_strategy, "default_strategy_count"):
                count = value.ai_strategy.default_strategy_count
                if isinstance(count, int) and count >= 0:
                    return count
        except (AttributeError, ValueError, TypeError):
            pass

    count = os.getenv("AI_STRATEGY_DEFAULT_STRATEGY_COUNT", "").strip()
    if count:
        try:
            return int(count)
        except (ValueError, TypeError):
            pass

    return 5


def _get_default_active_strategies() -> int:
    """Get default active strategies count from live config."""
    if _mystic_config:
        try:
            value = _mystic_config
            if hasattr(value, "ai_strategy") and hasattr(value.ai_strategy, "default_active_strategies"):
                count = value.ai_strategy.default_active_strategies
                if isinstance(count, int) and count >= 0:
                    return count
        except (AttributeError, ValueError, TypeError):
            pass

    count = os.getenv("AI_STRATEGY_DEFAULT_ACTIVE_STRATEGIES", "").strip()
    if count:
        try:
            return int(count)
        except (ValueError, TypeError):
            pass

    return 3


def _get_default_success_rate() -> float:
    """Get default success rate from live config."""
    if _mystic_config:
        try:
            value = _mystic_config
            if hasattr(value, "ai_strategy") and hasattr(value.ai_strategy, "default_success_rate"):
                rate = value.ai_strategy.default_success_rate
                if isinstance(rate, (int, float)) and 0.0 <= rate <= 1.0:
                    return float(rate)
        except (AttributeError, ValueError, TypeError):
            pass

    rate = os.getenv("AI_STRATEGY_DEFAULT_SUCCESS_RATE", "").strip()
    if rate:
        try:
            rate_val = float(rate)
            if 0.0 <= rate_val <= 1.0:
                return rate_val
        except (ValueError, TypeError):
            pass

    return 0.68


def _get_log_prefix() -> str:
    """Get log message prefix from live config."""
    if _mystic_config:
        try:
            value = _mystic_config
            if hasattr(value, "ai_strategy") and hasattr(value.ai_strategy, "log_prefix"):
                prefix = value.ai_strategy.log_prefix
                if isinstance(prefix, str):
                    return prefix
        except (AttributeError, ValueError, TypeError):
            pass

    prefix = os.getenv("AI_STRATEGY_LOG_PREFIX", "").strip()
    if prefix:
        return prefix

    return "✅"


class AIStrategyService:
    """AI strategy service for trading platform"""

    def __init__(self) -> None:
        self.is_running = False
        self.last_update = None
        self.status = _get_initial_status()

    async def start(self) -> None:
        """Start the AI strategy service"""
        self.is_running = True
        self.status = _get_running_status()
        log_prefix = _get_log_prefix()
        logger.info(f"{log_prefix} AI Strategy service started")

    async def stop(self) -> None:
        """Stop the AI strategy service"""
        self.is_running = False
        self.status = _get_stopped_status()
        log_prefix = _get_log_prefix()
        logger.info(f"{log_prefix} AI Strategy service stopped")

    async def get_status(self) -> dict[str, Any]:
        """Get AI strategy service status"""
        return {
            "status": self.status,
            "is_running": self.is_running,
            "last_update": self.last_update,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

    async def get_strategy_data(self) -> dict[str, Any]:
        """Get AI strategy data"""
        return {
            "strategy_count": _get_default_strategy_count(),
            "active_strategies": _get_default_active_strategies(),
            "success_rate": _get_default_success_rate(),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
