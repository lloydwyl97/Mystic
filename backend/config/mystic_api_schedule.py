"""
Coordinated Binance.US API + AI cadence schedule.

Design: ONE writer (signal gen) force-refreshes the 10-TF DAY bundle; all other
consumers read Redis cache. Stagger offsets prevent aligned bursts. Target combined
weight ~150-350/min vs BINANCEUS_WEIGHT_PER_MIN budget (default 1100).

Timeframe note: 1m/5m/15m/30m/1h/4h/8h/12h/1d/1w are fetched together per bundle
(10 klines calls), not on independent per-TF timers.
"""

from __future__ import annotations

import os
from typing import Final

# --- Global weight budget ---
BINANCEUS_WEIGHT_PER_MIN: Final[int] = int(os.getenv("BINANCEUS_WEIGHT_PER_MIN", "1100"))
BINANCE_WEIGHT_WARN_TOTAL: Final[int] = int(os.getenv("BINANCE_WEIGHT_WARN_TOTAL", "900"))

# --- DAY bundle (10 TFs) — signal gen is primary writer ---
DAY_AI_SIGNAL_LOOP_SEC: Final[int] = int(os.getenv("DAY_AI_SIGNAL_LOOP_SEC", "120"))
DAY_BUNDLE_CACHE_TTL_SEC: Final[int] = int(os.getenv("DAY_BUNDLE_CACHE_TTL_SEC", "110"))

# Stagger startup (seconds) — one-time phase offset per role
DAY_BUNDLE_STAGGER_SIGNAL_SEC: Final[int] = int(os.getenv("DAY_BUNDLE_STAGGER_SIGNAL_SEC", "0"))
DAY_BUNDLE_STAGGER_PORTFOLIO_SEC: Final[int] = int(os.getenv("DAY_BUNDLE_STAGGER_PORTFOLIO_SEC", "20"))
DAY_BUNDLE_STAGGER_AI_CONTEXT_SEC: Final[int] = int(os.getenv("DAY_BUNDLE_STAGGER_AI_CONTEXT_SEC", "25"))
DAY_BUNDLE_STAGGER_READINESS_SEC: Final[int] = int(os.getenv("DAY_BUNDLE_STAGGER_READINESS_SEC", "40"))
DAY_BUNDLE_STAGGER_LEARNING_SEC: Final[int] = int(os.getenv("DAY_BUNDLE_STAGGER_LEARNING_SEC", "50"))

# --- Limiter wait timeouts ---
BINANCE_LIMITER_CONSUME_TIMEOUT_CRITICAL_SEC: Final[float] = float(os.getenv("BINANCE_LIMITER_CONSUME_TIMEOUT_CRITICAL_SEC", "12"))
BINANCE_LIMITER_CONSUME_TIMEOUT_SEC: Final[float] = float(os.getenv("BINANCE_LIMITER_CONSUME_TIMEOUT_SEC", "8"))
BINANCE_LIMITER_CONSUME_TIMEOUT_LOOP_SEC: Final[float] = float(os.getenv("BINANCE_LIMITER_CONSUME_TIMEOUT_LOOP_SEC", "5"))
BINANCE_OHLCV_STALE_FALLBACK_MAX_AGE_SEC: Final[float] = float(os.getenv("BINANCE_OHLCV_STALE_FALLBACK_MAX_AGE_SEC", "150"))

# --- Live market data loops (external process or embedded uvicorn) ---
LIVE_TICKER_INTERVAL_SEC: Final[int] = int(os.getenv("LIVE_TICKER_INTERVAL", "12"))
LIVE_OHLCV_INTERVAL_SEC: Final[int] = int(os.getenv("LIVE_OHLCV_INTERVAL", "30"))
AI_TICKER_TARGET_WEIGHT_PER_MIN: Final[int] = int(os.getenv("AI_TICKER_TARGET_WEIGHT_PER_MIN", "80"))
AI_OHLCV_TARGET_WEIGHT_PER_MIN: Final[int] = int(os.getenv("AI_OHLCV_TARGET_WEIGHT_PER_MIN", "30"))

# --- MarketDataService REST loops (uvicorn; disable when external live_md owns freshness) ---
MARKET_HIGH_INTERVAL_SEC: Final[int] = int(os.getenv("MARKET_HIGH_INTERVAL_S", "60"))
MARKET_NORMAL_INTERVAL_SEC: Final[int] = int(os.getenv("MARKET_NORMAL_INTERVAL_S", "120"))
MARKET_TARGET_WEIGHT_PER_MIN: Final[int] = int(os.getenv("MARKET_TARGET_WEIGHT_PER_MIN", "120"))
MARKET_DATA_REST_LOOPS_ENABLED: Final[bool] = os.getenv("MARKET_DATA_REST_LOOPS_ENABLED", "false").lower() in (
    "1",
    "true",
    "yes",
    "on",
)

# --- Portfolio / prices ---
EXIT_MONITOR_INTERVAL_SEC: Final[int] = int(os.getenv("EXIT_MONITOR_INTERVAL_SEC", "45"))
PRICE_PUBLISHER_INTERVAL_SEC: Final[int] = int(os.getenv("PRICE_PUBLISHER_INTERVAL_SEC", "15"))
SIGNAL_CONSUMER_INTERVAL_SEC: Final[int] = int(os.getenv("SIGNAL_CONSUMER_INTERVAL_SEC", "2"))
LEDGER_MTM_PERSIST_INTERVAL_SEC: Final[float] = float(os.getenv("LEDGER_MTM_PERSIST_INTERVAL_SEC", "20"))
BAR_INTERVAL_SEC: Final[int] = int(os.getenv("BAR_INTERVAL_SEC", "60"))
SELL_CHECK_INTERVAL_SEC: Final[int] = int(os.getenv("SELL_CHECK_INTERVAL", "45"))

# --- AI context (reads bundle cache; publishes ai_context:*) ---
AI_CONTEXT_LOOP_SEC: Final[int] = int(os.getenv("AI_CONTEXT_LOOP_SEC", "60"))

# --- AI learning (reads bundle cache; asof only on new 4h anchors) ---
AI_COLLECTION_INTERVAL_SEC: Final[int] = int(os.getenv("AI_COLLECTION_INTERVAL", "60"))
AI_COLLECTION_INTERVAL_IDLE_SEC: Final[int] = int(os.getenv("AI_COLLECTION_INTERVAL_IDLE", "120"))
AI_LEARNING_FREQUENCY_SEC: Final[int] = int(os.getenv("AI_LEARNING_FREQUENCY", "900"))

# --- Diagnostics ---
MARKET_READINESS_CACHE_SEC: Final[int] = int(os.getenv("MARKET_READINESS_CACHE_SEC", "60"))

# --- 1m collector (SQLite feature store) ---
COLLECTOR_INTERVAL_SEC: Final[int] = int(os.getenv("COLLECTOR_INTERVAL_SEC", "30"))

# --- Signal / context freshness contract (aligned) ---
MAX_SIGNAL_AGE_SEC: Final[int] = int(os.getenv("MAX_SIGNAL_AGE_SEC", "300"))
SIGNAL_CONTENT_STALE_ALERT_SEC: Final[int] = int(os.getenv("SIGNAL_CONTENT_STALE_ALERT_SEC", "300"))
CTX_FRESH_MAX_AGE_SEC: Final[int] = int(os.getenv("CTX_FRESH_MAX_AGE_SEC", str(MAX_SIGNAL_AGE_SEC)))
SELL_MARK_MAX_AGE_SECONDS: Final[float] = float(os.getenv("SELL_MARK_MAX_AGE_SECONDS", str(max(20, PRICE_PUBLISHER_INTERVAL_SEC + 5))))

# --- Volume profile / ancillary ---
VOLUME_PROFILE_UPDATE_INTERVAL_SEC: Final[int] = int(os.getenv("VOLUME_PROFILE_UPDATE_INTERVAL", "900"))
