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


PLANNED_EXPERIMENT_ID = "M_clock_v2_planned_20260905"
PLANNED_RESULT = "PLANNED_NOT_RUN"


def planned_challenger_specification() -> dict[str, Any]:
    """Frozen first-challenger spec. Recorded before readiness. Not fitted."""
    schema = clock_challenger_export_schema()
    return {
        "experiment_id": PLANNED_EXPERIMENT_ID,
        "result": PLANNED_RESULT,
        "promoted": False,
        "train": False,
        "live_gate": False,
        "feature_set": SCHEMA_VERSION,
        "inputs": list(schema["inputs"]),
        "target": PRIMARY_TARGET,
        "actions": list(schema["actions"]),
        "hold_value_bps": 0.0,
        "model_class": "small_regularized_not_selected",
        "acceptance": future_acceptance_bar(),
        "training_procedure": planned_training_procedure(),
        "notes": "Frozen before lock inspection. Do not redesign after seeing the lock.",
    }


def planned_training_procedure() -> dict[str, Any]:
    """Future training recipe only. Execution is a later explicit task."""
    return {
        "executed": False,
        "folds": "expanding_chronological",
        "purge_overlapping_outcome_intervals": True,
        "embargo_seconds_min": max(CLOCK_LABEL_HORIZONS_SEC.values()),
        "regularization": "strong",
        "model_family": "small_only",
        "calibration": "training_only",
        "hold_alternative_bps": 0.0,
        "portfolio_replay": "real_constraints",
        "locked_test": "used_once",
        "hyperparameter_search_on_lock": False,
        "grid_search": False,
    }


REQUIRED_CLOCK_V2_FIELDS: tuple[str, ...] = (
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
)

# Generic day_model_readiness G_forward_span uses the 4H-entry challenger
# (day_forward_lock.challenger_export_schema): 14 listed inputs including
# categorical symbol x 10 events = 140 authoritative selected-trade labels.
# That 140 is not a clock-v2 group count and is not mutated here.
GENERIC_4H_CHALLENGER_INPUT_COUNT = 14
GENERIC_SELECTED_TRADE_LABEL_REQUIREMENT = GENERIC_4H_CHALLENGER_INPUT_COUNT * MIN_EVENTS_PER_FEATURE

CLOCK_V2_LISTED_INPUT_COUNT = len(REQUIRED_CLOCK_V2_FIELDS)  # 15
CLOCK_V2_CATEGORICAL_FEATURES = ("symbol",)
CLOCK_V2_NUMERIC_FEATURE_COUNT = CLOCK_V2_LISTED_INPUT_COUNT - len(CLOCK_V2_CATEGORICAL_FEATURES)  # 14
CLOCK_V2_CATEGORICAL_PARAMETER_COUNT = 4  # BTC/ETH/SOL/XRP dummy; HOLD is the omitted level

PLANNED_EXPERIMENT_ID_V2 = "M_clock_v2_planned_v2_20260905"


def clock_v2_statistical_contract() -> dict[str, Any]:
    """Document the 14-vs-15 split. Does not overwrite the generic 140 gate."""
    return {
        "generic_4h_challenger_inputs": list(challenger_inputs_4h()),
        "generic_numeric_or_listed_count": GENERIC_4H_CHALLENGER_INPUT_COUNT,
        "generic_categorical_included_in_count": True,
        "generic_events_per_listed_input": MIN_EVENTS_PER_FEATURE,
        "generic_required_authoritative_selected_trades": GENERIC_SELECTED_TRADE_LABEL_REQUIREMENT,
        "generic_observation_unit": "authoritative_selected_trade_label",
        "clock_v2_listed_inputs": list(REQUIRED_CLOCK_V2_FIELDS),
        "clock_v2_listed_input_count": CLOCK_V2_LISTED_INPUT_COUNT,
        "clock_v2_numeric_features": [n for n in REQUIRED_CLOCK_V2_FIELDS if n != "symbol"],
        "clock_v2_numeric_feature_count": CLOCK_V2_NUMERIC_FEATURE_COUNT,
        "clock_v2_categorical_features": list(CLOCK_V2_CATEGORICAL_FEATURES),
        "clock_v2_categorical_parameter_count": CLOCK_V2_CATEGORICAL_PARAMETER_COUNT,
        "clock_v2_effective_fitted_parameters_note": (
            "symbol is one categorical with 4 coin levels plus HOLD as the reference action; "
            "the 14 remaining fields are numeric. The original planned arm reused 14x10=140 "
            "from the generic 4H schema instead of 15x10. That reuse is frozen, not silently "
            "rewritten to 150."
        ),
        "why_140_is_correct_for_generic_gate": ("G_forward_span multiplies len(challenger_export_schema()['inputs']) by 10. Those 14 inputs are the 4H-entry schema, not clock-v2."),
        "why_140_is_not_a_clock_v2_ranker_population": (
            "140 selected-trade labels are fills chosen by the old policy. A BTC/ETH/SOL/XRP/HOLD ranker needs complete decision groups, not selected rows alone."
        ),
    }


def challenger_inputs_4h() -> tuple[str, ...]:
    from backend.services.day_forward_lock import challenger_export_schema

    return tuple(challenger_export_schema()["inputs"])


def planned_challenger_specification_v2() -> dict[str, Any]:
    """New readiness/experiment contract. Does not mutate M_clock_v2_planned_20260905."""
    schema = clock_challenger_export_schema()
    return {
        "experiment_id": PLANNED_EXPERIMENT_ID_V2,
        "result": PLANNED_RESULT,
        "promoted": False,
        "train": False,
        "live_gate": False,
        "feature_set": f"{SCHEMA_VERSION}_capture_1",
        "inputs": list(schema["inputs"]),
        "target": PRIMARY_TARGET,
        "actions": list(schema["actions"]),
        "hold_value_bps": 0.0,
        "model_class": "small_regularized_not_selected",
        "acceptance": future_acceptance_bar(),
        "training_procedure": {
            **planned_training_procedure(),
            "folds": "expanding_chronological",
            "purge_overlapping_4h_labels": True,
            "embargo_seconds_min": max(4 * 3600, *CLOCK_LABEL_HORIZONS_SEC.values()),
            "calendar_day_blocks_are_not_independent_folds": True,
        },
        "group_definition": "one DAY ranking timestamp with BTC/ETH/SOL/XRP/HOLD actions",
        "missing_data_rules": "NULL + reason; never zero-impute; ineligible coins keep eligibility=false",
        "readiness_requirements": clock_v2_v2_readiness_requirements(),
        "statistical_contract": clock_v2_statistical_contract(),
        "notes": ("Corrected clock-v2 readiness: complete groups, not selected-trade-only. Original M_clock_v2_planned_20260905 remains PLANNED_NOT_RUN and unmodified."),
    }


def clock_v2_v2_readiness_requirements() -> dict[str, Any]:
    return {
        "min_feature_complete_groups": CLOCK_V2_NUMERIC_FEATURE_COUNT * MIN_EVENTS_PER_FEATURE,
        "min_fully_comparable_labeled_groups": CLOCK_V2_NUMERIC_FEATURE_COUNT * MIN_EVENTS_PER_FEATURE,
        "min_authoritative_selected_trade_labels": GENERIC_SELECTED_TRADE_LABEL_REQUIREMENT,
        "selected_trades_alone_insufficient": True,
        "min_chronological_blocks_bookkeeping": MIN_CHRONOLOGICAL_BLOCKS,
        "block_definition": "UTC calendar day with >=1 decision group (bookkeeping only, not 5 independent folds)",
        "validation_folds": "expanding chronological; purge overlapping 4h labels; embargo >= 4h",
        "observation_unit": "complete_decision_group",
        "numeric_features_counted": CLOCK_V2_NUMERIC_FEATURE_COUNT,
        "categorical_parameters_counted": CLOCK_V2_CATEGORICAL_PARAMETER_COUNT,
        "events_per_numeric_feature": MIN_EVENTS_PER_FEATURE,
        "why_multiplier_uses_14": ("14 is the numeric clock-v2 field count after excluding categorical symbol. It matches the generic 140 number but the unit is complete groups, not fills."),
    }
