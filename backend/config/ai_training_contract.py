"""
Canonical training geometry vs live inference — single reference for audits.

- **day v4 (HTF)**: real **1h / 4h / 1d / 1w** stacks (see ``ai_day_htf_features``) + context;
  label grid and horizons follow ``day_label_grid_seconds`` / ``label_horizon_primary_bars_for_strategy``.

**1m** remains execution / microstructure ingest where applicable.

Label horizons: import from ``trade_worthiness_timing`` and ``training_label_economics`` (do not duplicate).
"""

from __future__ import annotations

from backend.config.trade_worthiness_timing import (
    TRAINING_LABEL_BAR_SECONDS,
    label_horizon_bars_for_strategy,
    label_horizon_bars_for_symbol,
    primary_label_bar_seconds_for_strategy,
)
from backend.services.ai_decision_contract import (
    AI_FEATURE_DIM_V1,
    AI_FEATURE_DIM_V2,
    FEATURE_VERSION_CURRENT,
)

# Persistence: self-supervised SQLite rows must match FEATURE_VERSION_CURRENT dims.
CANONICAL_TELEMETRY_CONTEXT_KEY_V2 = "canonical_v2_collect_training_data"
CANONICAL_TELEMETRY_CONTEXT_KEY_V3 = "canonical_v3_collect_training_data"
CANONICAL_TELEMETRY_CONTEXT_KEY_DAY_HTF = "canonical_day_htf_collect_training_data"

__all__ = [
    "AI_FEATURE_DIM_V1",
    "AI_FEATURE_DIM_V2",
    "CANONICAL_TELEMETRY_CONTEXT_KEY_DAY_HTF",
    "CANONICAL_TELEMETRY_CONTEXT_KEY_V2",
    "CANONICAL_TELEMETRY_CONTEXT_KEY_V3",
    "FEATURE_VERSION_CURRENT",
    "TRAINING_LABEL_BAR_SECONDS",
    "label_horizon_bars_for_strategy",
    "label_horizon_bars_for_symbol",
    "primary_label_bar_seconds_for_strategy",
]
