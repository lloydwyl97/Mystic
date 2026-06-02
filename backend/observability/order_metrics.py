"""
Order Metrics for Prometheus Monitoring
Tracks order-related operations and performance
"""

from prometheus_client import Counter, Gauge, Histogram

# Order creation metrics
ORDERS_CREATED_TOTAL = Counter(
    "orders_created_total",
    "Total number of orders created",
    ["symbol", "side", "type"],
)

# Order cancellation metrics
ORDERS_CANCELLED_TOTAL = Counter(
    "orders_cancelled_total",
    "Total number of orders cancelled",
    ["symbol"],
)

# Order error metrics
ORDERS_ERRORS_TOTAL = Counter(
    "orders_errors_total",
    "Total number of order errors",
    ["operation", "error_type"],
)

# Active orders gauge
ACTIVE_ORDERS_GAUGE = Gauge(
    "active_orders",
    "Current number of active orders",
)

# Order operation latency
ORDERS_CREATE_LATENCY_SECONDS = Histogram(
    "orders_create_latency_seconds",
    "Time to create an order",
    buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0),
)

ORDERS_GET_LATENCY_SECONDS = Histogram(
    "orders_get_latency_seconds",
    "Time to fetch orders",
    buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0),
)

ORDERS_CANCEL_LATENCY_SECONDS = Histogram(
    "orders_cancel_latency_seconds",
    "Time to cancel an order",
    buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0),
)

# Advanced order metrics
ADV_ORDERS_CREATE_LATENCY_SECONDS = Histogram(
    "adv_orders_create_latency_seconds",
    "Time to create an advanced order",
    buckets=(0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0),
)
