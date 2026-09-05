"""Research contract for clock-consistent DAY path features and labels.

Offline only. Not imported by live ranking, path-EV authority, sizing, or exits.
Does not train or promote a model. Does not change day_path_net_v1.
"""

from __future__ import annotations

from typing import Any

from backend.services.day_4h_entry_features import COINS, HOLD_SYMBOL
from backend.services.day_forward_lock import FORWARD_LOCK_START
from backend.services.day_model_readiness import ACCEPTANCE_STANDARD, MIN_CHRONOLOGICAL_BLOCKS, MIN_EVENTS_PER_FEATURE
from backend.services.day_path_input_validity import MAX_GAP_SEC, MAX_LAST_BAR_AGE_SEC

SCHEMA_VERSION = "day_path_clock_v2"
SOURCE_INTERVAL = "1m"
SOURCE_INTERVAL_SEC = 60
MISSING_POLICY = "unavailable_none_no_zero_impute"
AGGREGATION = "last_close_at_or_before_cutoff"
EXECUTABLE_PRICE_METHOD = "last_1m_close_at_or_before_horizon"
PRIMARY_TARGET = "expected_executable_net_bps"

CLOCK_LOOKBACKS_SEC: dict[str, int] = {
    "ret_5m": 5 * 60,
    "ret_15m": 15 * 60,
    "ret_30m": 30 * 60,
    "ret_1h": 60 * 60,
    "realized_vol_10m": 10 * 60,
    "drawdown_30m": 30 * 60,
    "rebound_30m": 30 * 60,
    "rel_volume_15m": 15 * 60,
    "btc_rel_ret_5m": 5 * 60,
}

MIN_OBSERVATIONS: dict[str, int] = {
    "ret_5m": 2,
    "ret_15m": 2,
    "ret_30m": 2,
    "ret_1h": 2,
    "realized_vol_10m": 5,
    "drawdown_30m": 3,
    "rebound_30m": 3,
    "rel_volume_15m": 3,
    "btc_rel_ret_5m": 2,
}

CLOCK_LABEL_HORIZONS_SEC: dict[str, int] = {
    "15m": 15 * 60,
    "30m": 30 * 60,
    "1h": 60 * 60,
    "2h": 2 * 60 * 60,
    "3h": 3 * 60 * 60,
    "4h": 4 * 60 * 60,
}

FEATURE_SPECS: dict[str, dict[str, Any]] = {
    name: {
        "name": name,
        "source_interval": SOURCE_INTERVAL,
        "clock_lookback_seconds": seconds,
        "minimum_observations": MIN_OBSERVATIONS[name],
        "maximum_allowed_age_seconds": MAX_LAST_BAR_AGE_SEC,
        "maximum_permitted_data_gap_seconds": MAX_GAP_SEC,
        "aggregation_method": AGGREGATION,
        "missing_data_policy": MISSING_POLICY,
    }
    for name, seconds in CLOCK_LOOKBACKS_SEC.items()
}


def feature_contract() -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "source_interval": SOURCE_INTERVAL,
        "source_interval_seconds": SOURCE_INTERVAL_SEC,
        "maximum_allowed_age_seconds": MAX_LAST_BAR_AGE_SEC,
        "maximum_permitted_data_gap_seconds": MAX_GAP_SEC,
        "aggregation_method": AGGREGATION,
        "missing_data_policy": MISSING_POLICY,
        "features": FEATURE_SPECS,
        "passthrough_context": (
            "p_buy",
            "legacy_path_ev",
            "final_rank_score",
            "production_4h_break_true_at_decision",
            "distance_to_4h_break_bps",
            "4h_range_position",
            "4h_alignment_state",
            "spread_bps",
            "expected_slippage_bps",
            "estimated_all_in_cost_bps",
        ),
        "ambiguous_names_forbidden": ("ret_20", "ret_5", "realized_vol_10"),
    }


def clock_challenger_export_schema() -> dict[str, Any]:
    """Predeclared small successor inputs. Not trained. Does not replace readiness schema."""
    return {
        "schema_version": SCHEMA_VERSION,
        "inputs": [
            "symbol",
            "p_buy",
            "legacy_path_ev",
            "final_rank_score",
            "ret_5m",
            "ret_15m",
            "ret_30m",
            "realized_vol_10m",
            "btc_rel_ret_5m",
            "production_4h_break_true_at_decision",
            "distance_to_4h_break_bps",
            "4h_range_position",
            "spread_bps",
            "estimated_all_in_cost_bps",
            "rel_volume_15m",
        ],
        "targets": [PRIMARY_TARGET],
        "hold_value_bps": 0.0,
        "actions": [*COINS, HOLD_SYMBOL],
        "train": False,
        "live_gate": False,
        "readiness_required_mature_trade_labels": 14 * MIN_EVENTS_PER_FEATURE,
        "readiness_required_chronological_blocks": MIN_CHRONOLOGICAL_BLOCKS,
    }


def future_acceptance_bar() -> dict[str, Any]:
    return {
        "criteria": list(ACCEPTANCE_STANDARD),
        "champion": "production entry behaviour at f942fea (unchanged); live path-EV is day_path_net_v1 on valid dense inputs only",
        "hold_value_bps": 0.0,
        "actions": [*COINS, HOLD_SYMBOL],
        "target": PRIMARY_TARGET,
        "lock_cutoff": FORWARD_LOCK_START,
        "disqualifiers": [
            "hindsight or MFE-credit labels as the training target",
            "thresholds tuned on the locked test period",
            "permanent HOLD or trade-opinion permission behaviour",
            "any artifact that merely loses less than the champion",
            "feeding clock-resampled features through day_path_net_v1 coefficients",
            "oracle or same-horizon leakage",
        ],
    }


def future_decision_contract() -> dict[str, Any]:
    return {
        "estimate": PRIMARY_TARGET,
        "actions": [*COINS, HOLD_SYMBOL],
        "hold_value_bps": 0.0,
        "activated": False,
        "live_authority": "day_path_net_v1_valid_dense_only",
    }
