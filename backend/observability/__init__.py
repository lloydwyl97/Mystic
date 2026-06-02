"""
Observability module for Mystic Trading Platform
Provides metrics, monitoring, and observability capabilities.
"""

from .metrics import metrics_router
from .middleware import ObservabilityMiddleware
from .mystic_metrics import (
    MYSTIC_SIGNAL_GENERATED_TOTAL,
    MYSTIC_SIGNAL_LATENCY_SECONDS,
)
from .order_metrics import (
    ACTIVE_ORDERS_GAUGE,
    ADV_ORDERS_CREATE_LATENCY_SECONDS,
    ORDERS_CANCEL_LATENCY_SECONDS,
    ORDERS_CANCELLED_TOTAL,
    ORDERS_CREATE_LATENCY_SECONDS,
    ORDERS_CREATED_TOTAL,
    ORDERS_ERRORS_TOTAL,
    ORDERS_GET_LATENCY_SECONDS,
)

__all__ = [
    "ACTIVE_ORDERS_GAUGE",
    "ADV_ORDERS_CREATE_LATENCY_SECONDS",
    "MYSTIC_SIGNAL_GENERATED_TOTAL",
    "MYSTIC_SIGNAL_LATENCY_SECONDS",
    "ORDERS_CANCELLED_TOTAL",
    "ORDERS_CANCEL_LATENCY_SECONDS",
    "ORDERS_CREATED_TOTAL",
    "ORDERS_CREATE_LATENCY_SECONDS",
    "ORDERS_ERRORS_TOTAL",
    "ORDERS_GET_LATENCY_SECONDS",
    "ObservabilityMiddleware",
    "metrics_router",
]
