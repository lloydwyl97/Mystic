from __future__ import annotations

# Direct imports for production
from prometheus_client import Counter, Histogram

MYSTIC_SIGNAL_GENERATED_TOTAL = Counter(
    "mystic_signal_generated_total",
    "Total mystic signals generated",
    ["symbol", "signal_type"],
)

MYSTIC_SIGNAL_LATENCY_SECONDS = Histogram(
    "mystic_signal_latency_seconds",
    "Latency for generating mystic comprehensive signals",
    ["symbol", "signal_type"],
    buckets=(0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2, 5),
)
