"""
Explicit DAY trading timeframe contract (top-4 Binance.US, 24/7).

Single source of truth for which CCXT intervals the AI must have before it may
BUY, while HOLDing, or before it may SELL (net-profit sell still requires full
round-trip economics; missing context => no-action).

Month context is **not** an exchange-native CCXT interval; it is derived only
from sufficient **1d** closes (see ``day_active_market_bundle.month_context_four_from_daily``).
"""

from __future__ import annotations

import os
from typing import Final

# Ordered list — stable forever for vector packing / telemetry.
DAY_ACTIVE_TIMEFRAMES: Final[tuple[str, ...]] = (
    "1m",
    "5m",
    "15m",
    "30m",
    "1h",
    "4h",
    "8h",
    "12h",
    "1d",
    "1w",
)

# Minimum closed bars required per TF before DAY AI may act (env overrides supported).
_DEF_MIN: dict[str, int] = {
    "1m": 250,
    "5m": 80,
    "15m": 80,
    "30m": 60,
    "1h": 120,
    "4h": 80,
    "8h": 60,
    "12h": 60,
    "1d": 60,
    "1w": 24,
}

# Minimum **daily** bars to derive month context (≈1 calendar month of trading days).
DAY_MONTH_CONTEXT_MIN_1D_BARS: Final[int] = max(22, int(os.getenv("DAY_MONTH_CONTEXT_MIN_1D_BARS", "26")))

# Used for build_feature_vector_124 primary series (named indicators need depth).
DAY_FEATURE_BUILDER_MIN_1M_BARS: Final[int] = max(200, int(os.getenv("DAY_FEATURE_BUILDER_MIN_1M_BARS", "250")))


def min_bars_for_day_tf(tf: str) -> int:
    t = (tf or "").strip().lower()
    alt = t.upper().replace(" ", "")
    try:
        override = os.getenv(f"DAY_TF_MIN_BARS_{alt}", "")
        if override.strip():
            return max(5, int(override))
    except (ValueError, TypeError):
        pass
    return max(5, _DEF_MIN.get(t, 40))


def fetch_limit_for_day_tf(tf: str) -> int:
    """Upper bound for REST klines fetch per TF (never invent rows beyond exchange)."""
    t = (tf or "").strip().lower()
    caps: dict[str, int] = {
        "1m": 1000,
        "5m": 500,
        "15m": 400,
        "30m": 400,
        "1h": 1000,
        "4h": 500,
        "8h": 400,
        "12h": 400,
        "1d": 400,
        "1w": 120,
    }
    try:
        cap_env = os.getenv(f"DAY_TF_FETCH_{t.upper()}", "")
        if cap_env.strip():
            return min(1000, max(50, int(cap_env)))
    except (ValueError, TypeError):
        pass
    return min(1000, max(80, caps.get(t, 300)))


__all__ = [
    "DAY_ACTIVE_TIMEFRAMES",
    "DAY_FEATURE_BUILDER_MIN_1M_BARS",
    "DAY_MONTH_CONTEXT_MIN_1D_BARS",
    "fetch_limit_for_day_tf",
    "min_bars_for_day_tf",
]
