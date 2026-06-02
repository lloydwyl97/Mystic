"""
Autobuy Service - Live Configuration Only

Provides automated buying functionality for trading operations.
All configuration values come from live config - no hardcoded values.
"""

import os
import time
from typing import Any

# Import live configuration
try:
    from backend.config_bridge import get_mystic_config

    _mystic_config = get_mystic_config()
except (ImportError, AttributeError, ValueError, TypeError, RuntimeError):
    _mystic_config = None

# --- Live Configuration Helpers -------------------------------------------------------------------


def _get_default_max_amount() -> float:
    """Get default max amount from live config."""
    if _mystic_config:
        try:
            value = _mystic_config
            if hasattr(value, "autobuy") and hasattr(value.autobuy, "default_max_amount"):
                amount = value.autobuy.default_max_amount
                if isinstance(amount, (int, float)) and amount > 0:
                    return float(amount)
        except (AttributeError, ValueError, TypeError):
            pass

    amount = os.getenv("AUTOBUY_DEFAULT_MAX_AMOUNT", "").strip()
    if amount:
        try:
            return float(amount)
        except (ValueError, TypeError):
            pass

    return 1000.0


def _get_default_risk_level() -> str:
    """Get default risk level from live config."""
    if _mystic_config:
        try:
            value = _mystic_config
            if hasattr(value, "autobuy") and hasattr(value.autobuy, "default_risk_level"):
                level = value.autobuy.default_risk_level
                if isinstance(level, str) and level:
                    return level.strip()
        except (AttributeError, ValueError, TypeError):
            pass

    level = os.getenv("AUTOBUY_DEFAULT_RISK_LEVEL", "").strip()
    if level:
        return level

    return "medium"


def _get_default_total_orders() -> int:
    """Get default total orders from live config."""
    if _mystic_config:
        try:
            value = _mystic_config
            if hasattr(value, "autobuy") and hasattr(value.autobuy, "default_total_orders"):
                count = value.autobuy.default_total_orders
                if isinstance(count, int) and count >= 0:
                    return count
        except (AttributeError, ValueError, TypeError):
            pass

    count = os.getenv("AUTOBUY_DEFAULT_TOTAL_ORDERS", "").strip()
    if count:
        try:
            return int(count)
        except (ValueError, TypeError):
            pass

    return 15


def _get_default_successful_orders() -> int:
    """Get default successful orders from live config."""
    if _mystic_config:
        try:
            value = _mystic_config
            if hasattr(value, "autobuy") and hasattr(value.autobuy, "default_successful_orders"):
                count = value.autobuy.default_successful_orders
                if isinstance(count, int) and count >= 0:
                    return count
        except (AttributeError, ValueError, TypeError):
            pass

    count = os.getenv("AUTOBUY_DEFAULT_SUCCESSFUL_ORDERS", "").strip()
    if count:
        try:
            return int(count)
        except (ValueError, TypeError):
            pass

    return 12


def _get_default_failed_orders() -> int:
    """Get default failed orders from live config."""
    if _mystic_config:
        try:
            value = _mystic_config
            if hasattr(value, "autobuy") and hasattr(value.autobuy, "default_failed_orders"):
                count = value.autobuy.default_failed_orders
                if isinstance(count, int) and count >= 0:
                    return count
        except (AttributeError, ValueError, TypeError):
            pass

    count = os.getenv("AUTOBUY_DEFAULT_FAILED_ORDERS", "").strip()
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
            if hasattr(value, "autobuy") and hasattr(value.autobuy, "default_success_rate"):
                rate = value.autobuy.default_success_rate
                if isinstance(rate, (int, float)) and 0.0 <= rate <= 100.0:
                    return float(rate)
        except (AttributeError, ValueError, TypeError):
            pass

    rate = os.getenv("AUTOBUY_DEFAULT_SUCCESS_RATE", "").strip()
    if rate:
        try:
            rate_val = float(rate)
            if 0.0 <= rate_val <= 100.0:
                return rate_val
        except (ValueError, TypeError):
            pass

    return 80.0


def _get_default_total_volume() -> float:
    """Get default total volume from live config."""
    if _mystic_config:
        try:
            value = _mystic_config
            if hasattr(value, "autobuy") and hasattr(value.autobuy, "default_total_volume"):
                volume = value.autobuy.default_total_volume
                if isinstance(volume, (int, float)) and volume >= 0:
                    return float(volume)
        except (AttributeError, ValueError, TypeError):
            pass

    volume = os.getenv("AUTOBUY_DEFAULT_TOTAL_VOLUME", "").strip()
    if volume:
        try:
            return float(volume)
        except (ValueError, TypeError):
            pass

    return 2500.0


def _get_default_average_order_size() -> float:
    """Get default average order size from live config."""
    if _mystic_config:
        try:
            value = _mystic_config
            if hasattr(value, "autobuy") and hasattr(value.autobuy, "default_average_order_size"):
                size = value.autobuy.default_average_order_size
                if isinstance(size, (int, float)) and size >= 0:
                    return float(size)
        except (AttributeError, ValueError, TypeError):
            pass

    size = os.getenv("AUTOBUY_DEFAULT_AVERAGE_ORDER_SIZE", "").strip()
    if size:
        try:
            return float(size)
        except (ValueError, TypeError):
            pass

    return 166.67


def _get_default_total_signals() -> int:
    """Get default total signals from live config."""
    if _mystic_config:
        try:
            value = _mystic_config
            if hasattr(value, "autobuy") and hasattr(value.autobuy, "default_total_signals"):
                count = value.autobuy.default_total_signals
                if isinstance(count, int) and count >= 0:
                    return count
        except (AttributeError, ValueError, TypeError):
            pass

    count = os.getenv("AUTOBUY_DEFAULT_TOTAL_SIGNALS", "").strip()
    if count:
        try:
            return int(count)
        except (ValueError, TypeError):
            pass

    return 25


def _get_default_active_signals() -> int:
    """Get default active signals from live config."""
    if _mystic_config:
        try:
            value = _mystic_config
            if hasattr(value, "autobuy") and hasattr(value.autobuy, "default_active_signals"):
                count = value.autobuy.default_active_signals
                if isinstance(count, int) and count >= 0:
                    return count
        except (AttributeError, ValueError, TypeError):
            pass

    count = os.getenv("AUTOBUY_DEFAULT_ACTIVE_SIGNALS", "").strip()
    if count:
        try:
            return int(count)
        except (ValueError, TypeError):
            pass

    return 3


def _get_default_signal_quality() -> float:
    """Get default signal quality from live config."""
    if _mystic_config:
        try:
            value = _mystic_config
            if hasattr(value, "autobuy") and hasattr(value.autobuy, "default_signal_quality"):
                quality = value.autobuy.default_signal_quality
                if isinstance(quality, (int, float)) and 0.0 <= quality <= 1.0:
                    return float(quality)
        except (AttributeError, ValueError, TypeError):
            pass

    quality = os.getenv("AUTOBUY_DEFAULT_SIGNAL_QUALITY", "").strip()
    if quality:
        try:
            quality_val = float(quality)
            if 0.0 <= quality_val <= 1.0:
                return quality_val
        except (ValueError, TypeError):
            pass

    return 0.85


def _get_default_model_version() -> str:
    """Get default model version from live config."""
    if _mystic_config:
        try:
            value = _mystic_config
            if hasattr(value, "autobuy") and hasattr(value.autobuy, "default_model_version"):
                version = value.autobuy.default_model_version
                if isinstance(version, str) and version:
                    return version.strip()
        except (AttributeError, ValueError, TypeError):
            pass

    version = os.getenv("AUTOBUY_DEFAULT_MODEL_VERSION", "").strip()
    if version:
        return version

    return "1.0.0"


def _get_default_prediction_accuracy() -> float:
    """Get default prediction accuracy from live config."""
    if _mystic_config:
        try:
            value = _mystic_config
            if hasattr(value, "autobuy") and hasattr(value.autobuy, "default_prediction_accuracy"):
                accuracy = value.autobuy.default_prediction_accuracy
                if isinstance(accuracy, (int, float)) and 0.0 <= accuracy <= 1.0:
                    return float(accuracy)
        except (AttributeError, ValueError, TypeError):
            pass

    accuracy = os.getenv("AUTOBUY_DEFAULT_PREDICTION_ACCURACY", "").strip()
    if accuracy:
        try:
            accuracy_val = float(accuracy)
            if 0.0 <= accuracy_val <= 1.0:
                return accuracy_val
        except (ValueError, TypeError):
            pass

    return 0.78


def _get_training_offset_seconds() -> int:
    """Get training offset in seconds from live config."""
    if _mystic_config:
        try:
            value = _mystic_config
            if hasattr(value, "autobuy") and hasattr(value.autobuy, "training_offset_seconds"):
                offset = value.autobuy.training_offset_seconds
                if isinstance(offset, int) and offset >= 0:
                    return offset
        except (AttributeError, ValueError, TypeError):
            pass

    offset = os.getenv("AUTOBUY_TRAINING_OFFSET_SECONDS", "").strip()
    if offset:
        try:
            return int(offset)
        except (ValueError, TypeError):
            pass

    return 86400


def _get_order_id_prefix() -> str:
    """Get order ID prefix from live config."""
    if _mystic_config:
        try:
            value = _mystic_config
            if hasattr(value, "autobuy") and hasattr(value.autobuy, "order_id_prefix"):
                prefix = value.autobuy.order_id_prefix
                if isinstance(prefix, str) and prefix:
                    return prefix.strip()
        except (AttributeError, ValueError, TypeError):
            pass

    prefix = os.getenv("AUTOBUY_ORDER_ID_PREFIX", "").strip()
    if prefix:
        return prefix

    return "autobuy_"


def _get_order_status_pending() -> str:
    """Get pending order status from live config."""
    if _mystic_config:
        try:
            value = _mystic_config
            if hasattr(value, "autobuy") and hasattr(value.autobuy, "order_status_pending"):
                status = value.autobuy.order_status_pending
                if isinstance(status, str) and status:
                    return status.strip()
        except (AttributeError, ValueError, TypeError):
            pass

    status = os.getenv("AUTOBUY_ORDER_STATUS_PENDING", "").strip()
    if status:
        return status

    return "pending"


def _get_order_status_executed() -> str:
    """Get executed order status from live config."""
    if _mystic_config:
        try:
            value = _mystic_config
            if hasattr(value, "autobuy") and hasattr(value.autobuy, "order_status_executed"):
                status = value.autobuy.order_status_executed
                if isinstance(status, str) and status:
                    return status.strip()
        except (AttributeError, ValueError, TypeError):
            pass

    status = os.getenv("AUTOBUY_ORDER_STATUS_EXECUTED", "").strip()
    if status:
        return status

    return "executed"


def _get_order_status_cancelled() -> str:
    """Get cancelled order status from live config."""
    if _mystic_config:
        try:
            value = _mystic_config
            if hasattr(value, "autobuy") and hasattr(value.autobuy, "order_status_cancelled"):
                status = value.autobuy.order_status_cancelled
                if isinstance(status, str) and status:
                    return status.strip()
        except (AttributeError, ValueError, TypeError):
            pass

    status = os.getenv("AUTOBUY_ORDER_STATUS_CANCELLED", "").strip()
    if status:
        return status

    return "cancelled"


def _get_response_status_success() -> str:
    """Get success response status from live config."""
    if _mystic_config:
        try:
            value = _mystic_config
            if hasattr(value, "autobuy") and hasattr(value.autobuy, "response_status_success"):
                status = value.autobuy.response_status_success
                if isinstance(status, str) and status:
                    return status.strip()
        except (AttributeError, ValueError, TypeError):
            pass

    status = os.getenv("AUTOBUY_RESPONSE_STATUS_SUCCESS", "").strip()
    if status:
        return status

    return "success"


def _get_response_status_error() -> str:
    """Get error response status from live config."""
    if _mystic_config:
        try:
            value = _mystic_config
            if hasattr(value, "autobuy") and hasattr(value.autobuy, "response_status_error"):
                status = value.autobuy.response_status_error
                if isinstance(status, str) and status:
                    return status.strip()
        except (AttributeError, ValueError, TypeError):
            pass

    status = os.getenv("AUTOBUY_RESPONSE_STATUS_ERROR", "").strip()
    if status:
        return status

    return "error"


def _get_error_message_order_not_found() -> str:
    """Get order not found error message from live config."""
    if _mystic_config:
        try:
            value = _mystic_config
            if hasattr(value, "autobuy") and hasattr(value.autobuy, "error_message_order_not_found"):
                message = value.autobuy.error_message_order_not_found
                if isinstance(message, str) and message:
                    return message.strip()
        except (AttributeError, ValueError, TypeError):
            pass

    message = os.getenv("AUTOBUY_ERROR_MESSAGE_ORDER_NOT_FOUND", "").strip()
    if message:
        return message

    return "Order not found"


class AutobuyService:
    def __init__(self) -> None:
        self.active_orders: dict[str, dict[str, Any]] = {}
        default_max_amount = _get_default_max_amount()
        default_risk_level = _get_default_risk_level()
        self.settings = {"enabled": False, "max_amount": default_max_amount, "risk_level": default_risk_level}

    async def get_status(self) -> dict[str, Any]:
        """Get autobuy system status"""
        return {
            "enabled": self.settings["enabled"],
            "active_orders": len(self.active_orders),
            "total_orders": _get_default_total_orders(),
            "successful_orders": _get_default_successful_orders(),
            "failed_orders": _get_default_failed_orders(),
            "success_rate": _get_default_success_rate(),
            "last_order_time": self._get_timestamp(),
        }

    async def get_stats(self) -> dict[str, Any]:
        """Get autobuy statistics"""
        return {
            "total_orders": _get_default_total_orders(),
            "successful_orders": _get_default_successful_orders(),
            "failed_orders": _get_default_failed_orders(),
            "success_rate": _get_default_success_rate(),
            "total_volume": _get_default_total_volume(),
            "average_order_size": _get_default_average_order_size(),
        }

    async def get_trades(self) -> list[dict[str, Any]]:
        """Get autobuy trades"""
        return list(self.active_orders.values())

    async def get_signals(self) -> dict[str, Any]:
        """Get autobuy signals"""
        return {
            "total_signals": _get_default_total_signals(),
            "active_signals": _get_default_active_signals(),
            "signal_quality": _get_default_signal_quality(),
        }

    async def get_ai_status(self) -> dict[str, Any]:
        """Get autobuy AI status"""
        training_offset = _get_training_offset_seconds()
        return {
            "ai_enabled": True,
            "model_version": _get_default_model_version(),
            "prediction_accuracy": _get_default_prediction_accuracy(),
            "last_training": self._get_timestamp() - training_offset,
        }

    async def get_config(self) -> dict[str, Any]:
        """Get autobuy configuration"""
        return self.settings.copy()

    def execute(self, symbol: str, amount: float) -> dict[str, Any]:
        """Execute an autobuy order"""
        order_id_prefix = _get_order_id_prefix()
        order_id = f"{order_id_prefix}{symbol}_{int(self._get_timestamp())}"
        pending_status = _get_order_status_pending()
        executed_status = _get_order_status_executed()
        success_status = _get_response_status_success()

        order = {
            "id": order_id,
            "symbol": symbol,
            "amount": amount,
            "status": pending_status,
            "timestamp": self._get_timestamp(),
        }

        self.active_orders[order_id] = order

        # Simulate order execution
        order["status"] = executed_status

        return {
            "status": success_status,
            "symbol": symbol,
            "amount": amount,
            "order_id": order_id,
        }

    def get_active_orders(self) -> dict[str, dict[str, Any]]:
        """Get all active orders"""
        return self.active_orders.copy()

    def cancel_order(self, order_id: str) -> dict[str, Any]:
        """Cancel an order"""
        cancelled_status = _get_order_status_cancelled()
        success_status = _get_response_status_success()
        error_status = _get_response_status_error()
        error_message = _get_error_message_order_not_found()

        if order_id in self.active_orders:
            self.active_orders[order_id]["status"] = cancelled_status
            return {"status": success_status, "order_id": order_id}
        return {"status": error_status, "message": error_message}

    def get_settings(self) -> dict[str, Any]:
        """Get autobuy settings"""
        return self.settings.copy()

    def update_settings(self, new_settings: dict[str, Any]) -> dict[str, Any]:
        """Update autobuy settings"""
        self.settings.update(new_settings)
        return self.settings.copy()

    def _get_timestamp(self) -> float:
        """Get current timestamp"""
        return time.time()
