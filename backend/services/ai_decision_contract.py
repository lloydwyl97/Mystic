"""
AI Decision Contract — single source of truth for the live AI decision pipeline.

This module defines the canonical contract for everything AI in Mystic:
    * what inputs are produced
    * what models are authoritative vs telemetry-only
    * how raw probabilities become a (BUY/HOLD/SELL, confidence, buy_margin) decision
    * how multi-timeframe + market context modify confidence
    * what artifacts are read/written and on what cadence
    * what Redis keys are written and their TTLs

NOTHING in this module talks to the network. It is intentionally pure and
importable from anywhere (training, inference, position tracker, outcome bridge,
docs/tests). All other AI files MUST import their constants from here so we
never get parallel hidden contracts again.

Author note: this file replaces ad-hoc "telemetry" comments scattered across
ai_signal_generator.py with a single declarative table.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from enum import Enum
from typing import Final

from backend.config.day_active_timeframes import DAY_ACTIVE_TIMEFRAMES

# ---------------------------------------------------------------------------
# Feature contract version (bumps when the canonical feature vector changes)
# ---------------------------------------------------------------------------

FEATURE_VERSION_CURRENT: Final[int] = 5

# v1: 124-dim technical block only (legacy).
# v2: 124 technical + 21 context (145).
# v3: day — primary-clock OHLCV + v2-shaped context (145).
# v4: day — stacked HTF 31-blocks + legacy CONTEXT_DIMS_V2 (145).
# v5: day — 124 named indicators from **native 1m** + 21-dim CONTEXT_DIMS_DAY_FULL (slopesx10 TF + month + macro tail); still 145.
FEATURE_VERSION_DAY_HTF: Final[int] = 5
FEATURE_VERSION_DAY_FULL_MTF: Final[int] = 5
AI_FEATURE_DIM_V1: Final[int] = 124
AI_FEATURE_DIM_V2: Final[int] = 145

# Canonical, ordered context dimensions. The order MUST be stable forever
# (artifacts trained today must work tomorrow). Append-only.
CONTEXT_DIMS_V2: Final[tuple[str, ...]] = (
    # MTF EMA-alignment (5 dims) — value in [0..1]
    "mtf_5m_trend",
    "mtf_15m_trend",
    "mtf_1h_trend",
    "mtf_4h_trend",
    "mtf_1d_trend",
    # MTF % slope over last lookback (5 dims) — value in ~[-0.2..0.2]
    "mtf_5m_slope",
    "mtf_15m_slope",
    "mtf_1h_slope",
    "mtf_4h_slope",
    "mtf_1d_slope",
    # 24h market context (4 dims)
    "ctx_change_24h_pct",
    "ctx_volume_24h_log",  # log1p(volume_usd / 1e6) for scale stability
    "ctx_relative_volume",
    "ctx_liquidity_tier_norm",  # 0/0.33/0.66/1
    # Microstructure (2 dims)
    "ctx_spread_pct",
    "ctx_depth_imbalance",  # [-1, +1]
    # Cross-asset RS (3 dims)
    "ctx_rs_btc",  # scaled to [-1, +1]
    "ctx_rs_eth",  # scaled to [-1, +1]
    "ctx_btc_dominance_proxy",  # [0, 1]
    # Regime + sentiment (2 dims)
    "ctx_regime_signed",  # +1 trending_up / 0 chop / -1 trending_down
    "ctx_sentiment_fear_greed",  # alternative.me fear/greed scaled to [-1, +1]
)
assert len(CONTEXT_DIMS_V2) == (AI_FEATURE_DIM_V2 - AI_FEATURE_DIM_V1)

# v5 DAY context slice (still length 21; semantic layout differs from CONTEXT_DIMS_V2).
CONTEXT_DIMS_DAY_FULL: Final[tuple[str, ...]] = (
    *[f"slope_pct_{tf}" for tf in DAY_ACTIVE_TIMEFRAMES],
    "month_log_ret_window",
    "month_realized_vol_window",
    "mean_ema_align_all_tf",
    "ctx_change_24h_pct",
    "ctx_volume_24h_log",
    "ctx_relative_volume",
    "ctx_spread_pct",
    "ctx_depth_imbalance",
    "ctx_rs_mean_btc_eth",
    "ctx_btc_dominance_proxy",
    "ctx_regime_sentiment_blend",
)
assert len(CONTEXT_DIMS_DAY_FULL) == (AI_FEATURE_DIM_V2 - AI_FEATURE_DIM_V1)


def feature_dim_for_version(v: int) -> int:
    if v == 1:
        return AI_FEATURE_DIM_V1
    if v in (2, 3, 4, 5):
        return AI_FEATURE_DIM_V2
    raise ValueError(f"unknown FEATURE_VERSION {v}")


# ---------------------------------------------------------------------------
# Open-position AI action contract
# ---------------------------------------------------------------------------


class AIPositionAction(str, Enum):
    HOLD = "HOLD"
    REDUCE = "REDUCE"
    EXIT = "EXIT"
    ADD = "ADD"


# ---------------------------------------------------------------------------
# Cadences (seconds) — re-export from mystic_api_schedule (single source of truth)
# ---------------------------------------------------------------------------

from backend.config.mystic_api_schedule import (
    AI_COLLECTION_INTERVAL_SEC,
    AI_CONTEXT_LOOP_SEC,
    AI_LEARNING_FREQUENCY_SEC,
    DAY_AI_SIGNAL_LOOP_SEC,
)

AI_RETRAIN_FREQUENCY_SEC: Final[int] = AI_LEARNING_FREQUENCY_SEC
AI_SIGNAL_LOOP_SEC: Final[int] = DAY_AI_SIGNAL_LOOP_SEC
AI_MODEL_RELOAD_CHECK_SEC: Final[int] = int(os.getenv("AI_MODEL_RELOAD_CHECK_SEC", "300"))  # 5 min mtime check
# Inactive legacy advisory services (not started; modules not shipped):
AI_POSITION_TRACKER_LOOP_SEC: Final[int] = 20
AI_OUTCOME_BRIDGE_LOOP_SEC: Final[int] = 60


# ---------------------------------------------------------------------------
# Redis key contract (only these keys are written by the canonical AI path)
# ---------------------------------------------------------------------------

# Historical doc-only shape. Live ML MUST use ``live_strategy_contracts.redis_ai_signal_key(strategy_id, bus)``.
# Never publish ``ai_signal:{BUS}`` without strategy segment — portfolio rejects it as non-canonical.
NON_CANONICAL_LEGACY_AI_SIGNAL_BUS_ONLY_TEMPLATE: Final[str] = "ai_signal:{symbol}"
REDIS_KEY_AI_CONTEXT: Final[str] = "ai_context:{symbol}"
REDIS_KEY_AI_POSITION: Final[str] = "ai_position:{symbol}"
REDIS_KEY_AI_POSITION_RECO: Final[str] = "ai_position_reco:{symbol}"
REDIS_KEY_AI_OUTCOME_PULSE: Final[str] = "ai_outcome:last_ingest_ts"
REDIS_KEY_AI_LEARNING_STATS: Final[str] = "ai_learning_stats"
REDIS_KEY_AI_SENTIMENT: Final[str] = "ai_sentiment:fear_greed"

REDIS_TTL_AI_CONTEXT_SEC: Final[int] = int(os.getenv("REDIS_TTL_AI_CONTEXT_SEC", "1800"))  # full 10-symbol pass can exceed cadence
# Entry/signal context age gate: env CTX_FRESH_MAX_AGE_SEC, default 900s (see strategy_runtime_audit.get_ctx_fresh_max_age_sec).
REDIS_TTL_AI_POSITION_SEC: Final[int] = 300  # 15x tracker loop


# ---------------------------------------------------------------------------
# Multi-timeframe contract
# ---------------------------------------------------------------------------

MTF_TIMEFRAMES: Final[tuple[str, ...]] = ("1m", "5m", "15m", "1h", "4h", "1d")
MTF_BARS_PER_TF: Final[dict[str, int]] = {
    "1m": 240,
    "5m": 240,
    "15m": 192,
    "1h": 168,
    "4h": 90,
    "1d": 60,
}


# ---------------------------------------------------------------------------
# Market context contract — every key here is REAL and used downstream
# ---------------------------------------------------------------------------

MARKET_CONTEXT_FIELDS: Final[tuple[str, ...]] = (
    "ctx_change_24h_pct",
    "ctx_volume_24h_usd",
    "ctx_relative_volume",
    "ctx_liquidity_tier",
    "ctx_spread_pct",
    "ctx_depth_imbalance",
    "ctx_rs_btc",
    "ctx_rs_eth",
    "ctx_btc_dominance_proxy",
    "ctx_market_regime",
    "ctx_sentiment_fear_greed",
)


# Fear/Greed refresh cadence — index updates ~daily; default 10 min poll.
def _fear_greed_fetch_interval_sec() -> int:
    for key in ("FEAR_GREED_FETCH_INTERVAL_SEC", "AI_SENTIMENT_LOOP_SEC"):
        raw = os.getenv(key)
        if raw is None or not str(raw).strip():
            continue
        try:
            return max(60, int(str(raw).split()[0]))
        except (TypeError, ValueError):
            continue
    return 600


AI_SENTIMENT_LOOP_SEC: Final[int] = _fear_greed_fetch_interval_sec()


# Fundamental / sentiment slots in the **first 124 dims** (FEATURE_MAPPING 81-90).
# Wired by ``backend.services.ai_feature_fundamentals`` into the sentiment dict consumed by
# ``build_feature_vector_124`` — same inputs are used for v2 (145 = 124 base + 21 context).
LEGACY_SENTIMENT_FEATURE_NAMES: Final[tuple[str, ...]] = (
    "fear_greed_index",
    "social_sentiment",
    "news_sentiment",
    "put_call_ratio",
    "vix",
    "market_cap",
    "supply",
    "circulating_supply",
    "max_supply",
    "market_dominance",
)

# Vector indices 80-89 (0-based) == FEATURE_MAPPING 81-90 (1-based). Same block as LEGACY_SENTIMENT_*.
SENTIMENT_BLOCK_ZERO_BASE_START: Final[int] = 80
SENTIMENT_BLOCK_ZERO_BASE_END: Final[int] = 89  # inclusive

# True live external feeds (not synthesized). When unset, slots stay 0 — not “fake neutral”.
CANONICAL_NEWS_SENTIMENT_ENV: Final[str] = "NEWS_API_KEY"
CANONICAL_SOCIAL_SENTIMENT_ENV: Final[str] = "REDDIT_CLIENT_ID"  # also requires REDDIT_CLIENT_SECRET

# Declarative: these names depend on external APIs for non-zero live values (see ai_feature_fundamentals).
EXTERNAL_API_SENTIMENT_FEATURES: Final[frozenset[str]] = frozenset({"social_sentiment", "news_sentiment"})

# Tier-2 inputs (still live external data, not fabricated constants):
# - news_sentiment: NewsAPI ``market_wide`` query when symbol-specific returns no articles (see news_sentiment.py ``tier``).
# - social_sentiment: Reddit global ``/search`` when subreddit hot scan finds no matching titles.
# Slot 88 max_supply: when CoinGecko reports 0 / missing cap, effective value = circulating_supply (minimum cap).


# ---------------------------------------------------------------------------
# Confidence multiplier weights
# ---------------------------------------------------------------------------
# These multipliers are the one and only AI-side adjustment applied AFTER
# normalize() to get the published "winner_probability". They are NOT a gate.
# They are part of the AI decision (model probability adjusted by context).
# ---------------------------------------------------------------------------

# NOTE: With v2 these context dims are inputs to the model itself, so the
# downstream multiplier is now a small "sanity nudge", not the primary
# context-to-decision channel. Reduced caps reflect that.
CTX_MTF_ALIGN_WEIGHT: Final[float] = 0.05  # +/- 5% from MTF agreement
CTX_RS_WEIGHT: Final[float] = 0.025  # +/- 2.5% from BTC/ETH RS
CTX_DEPTH_WEIGHT: Final[float] = 0.02  # +/- 2% from orderbook imbalance
CTX_REGIME_WEIGHT: Final[float] = 0.025  # +/- 2.5% from market regime
CTX_TOTAL_CAP: Final[float] = 0.10  # absolute cap on combined adjustment


# ---------------------------------------------------------------------------
# Active vs non-active model registry (the truth)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ModelRegistryEntry:
    """One row in the canonical AI model registry."""

    name: str
    artifact_filename: str  # filename in models/active/, formatted with {symbol}
    role: str  # "DIRECTIONAL" | "TELEMETRY_ONLY" | "DEAD_CODE"
    note: str = ""


AI_MODEL_REGISTRY: Final[tuple[ModelRegistryEntry, ...]] = (
    ModelRegistryEntry(
        name="per_coin_rf",
        artifact_filename="{symbol}_direction.pkl",
        role="DIRECTIONAL",
        note=(
            "Authoritative BUY/HOLD/SELL; per (strategy, symbol) artifact; trained on "
            "feature_version>=2 145-dim contract; v3 requires primary-clock OHLCV "
            "(5m day / 15m day) + ai_context, 1m execution support only."
        ),
    ),
    ModelRegistryEntry(
        name="fear_greed_predictor",
        artifact_filename="(no artifact — alternative.me API)",
        role="DIRECTIONAL_INPUT",
        note=("Real sentiment signal from alternative.me Fear/Greed Index. Wired into the canonical AI input as ctx_sentiment_fear_greed."),
    ),
)


def directional_models() -> list[ModelRegistryEntry]:
    return [m for m in AI_MODEL_REGISTRY if m.role == "DIRECTIONAL"]


def telemetry_models() -> list[ModelRegistryEntry]:
    return [m for m in AI_MODEL_REGISTRY if m.role == "TELEMETRY_ONLY"]


def dead_models() -> list[ModelRegistryEntry]:
    return [m for m in AI_MODEL_REGISTRY if m.role == "DEAD_CODE"]


# ---------------------------------------------------------------------------
# Decision payload schema (what every published ai_signal:<strategy>:<symbol> contains)
# ---------------------------------------------------------------------------


@dataclass
class AIDecision:
    """Canonical AI decision record — exactly what gets published per symbol."""

    symbol: str
    decision_id: str
    timestamp: float

    # Raw model output (per-coin RF)
    prob_buy: float
    prob_hold: float
    prob_sell: float
    argmax_action: str  # "BUY" | "HOLD" | "SELL" before any context

    # Post-normalize, pre-context confidence
    winner_probability_raw: float

    # Final (post-context-multiplier) values that downstream consumers use
    prediction: str  # "BUY" | "HOLD" | "SELL"
    confidence: float  # final winner_probability (post-context)
    buy_margin: float  # telemetry only

    # Context applied (so we can audit any decision later)
    ctx_applied: dict[str, float] = field(default_factory=dict)
    ctx_multiplier: float = 1.0

    # Model artifact actually used
    model_artifact_path: str = ""
    label_version: str = ""
    label_horizon_bars: int = 0


__all__ = [
    "AI_COLLECTION_INTERVAL_SEC",
    "AI_CONTEXT_LOOP_SEC",
    "AI_FEATURE_DIM_V1",
    "AI_FEATURE_DIM_V2",
    "AI_MODEL_REGISTRY",
    "AI_MODEL_RELOAD_CHECK_SEC",
    "AI_OUTCOME_BRIDGE_LOOP_SEC",
    "AI_POSITION_TRACKER_LOOP_SEC",
    "AI_RETRAIN_FREQUENCY_SEC",
    "AI_SENTIMENT_LOOP_SEC",
    "AI_SIGNAL_LOOP_SEC",
    "CONTEXT_DIMS_DAY_FULL",
    "CONTEXT_DIMS_V2",
    "CTX_DEPTH_WEIGHT",
    "CTX_MTF_ALIGN_WEIGHT",
    "CTX_REGIME_WEIGHT",
    "CTX_RS_WEIGHT",
    "CTX_TOTAL_CAP",
    "FEATURE_VERSION_CURRENT",
    "FEATURE_VERSION_DAY_FULL_MTF",
    "FEATURE_VERSION_DAY_HTF",
    "MARKET_CONTEXT_FIELDS",
    "MTF_BARS_PER_TF",
    "MTF_TIMEFRAMES",
    "NON_CANONICAL_LEGACY_AI_SIGNAL_BUS_ONLY_TEMPLATE",
    "REDIS_KEY_AI_CONTEXT",
    "REDIS_KEY_AI_LEARNING_STATS",
    "REDIS_KEY_AI_OUTCOME_PULSE",
    "REDIS_KEY_AI_POSITION",
    "REDIS_KEY_AI_POSITION_RECO",
    "REDIS_KEY_AI_SENTIMENT",
    "REDIS_TTL_AI_CONTEXT_SEC",
    "REDIS_TTL_AI_POSITION_SEC",
    "AIDecision",
    "AIPositionAction",
    "ModelRegistryEntry",
    "dead_models",
    "directional_models",
    "feature_dim_for_version",
    "telemetry_models",
]
