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


# V3: grouped-ranker contract. Does not mutate v1 or v2.
# Observation unit is one decision timestamp. Four coins in one group are
# one correlated observation, not four independent events.
CLOCK_V2_INTERCEPT_COUNT = 1
CLOCK_V2_EFFECTIVE_FITTED_PARAMETERS = CLOCK_V2_INTERCEPT_COUNT + CLOCK_V2_NUMERIC_FEATURE_COUNT + CLOCK_V2_CATEGORICAL_PARAMETER_COUNT  # 19
CLOCK_V2_COST_CALIBRATION_PARAMETERS = 5  # 4 symbol spreads + 1 slippage; commission is known
EVENTS_PER_PARAMETER = MIN_EVENTS_PER_FEATURE
PLANNED_EXPERIMENT_ID_V3 = "M_clock_v2_planned_v3_20260905"
PLANNED_EXPERIMENT_ID_V4 = "M_clock_v2_planned_v4_20260904"

# Horizon freeze evidence (pre-existing, outcome-blind):
# 1. scripts/train_day_path_net.py defines DAY_HORIZONS_MIN = (60, 120, 180) where
#    180 is 180 1m-bars = 180 minutes = 3 clock hours. This is the design ceiling
#    for DAY path prediction and predates clock-v2 research.
# 2. models/day_path_net_v1.json stores primary_horizon_min=180, confirming the
#    v1 artifact's horizon interpretation as 180 minutes.
# 3. Verified: future = raw[i: i + max_h] uses 1m OHLCV rows, so 180 rows == 180 min.
# 4. The 3h horizon is NOT selected from P&L inspection of locked outcomes;
#    it is the inherited design ceiling from the pre-research training pipeline.
PRIMARY_TARGET_HORIZON_SEC = 3 * 60 * 60  # 10800 seconds = 3 hours
PRIMARY_TARGET_HORIZON_NAME = "3h"
TARGET_HORIZON_STATUS = "PRIMARY_TARGET_HORIZON_3H"

# Forming-candle contract:
# Clock-v2 features use LAST CLOSED 1m candle only.
# The forming (in-progress) bar for the current minute is excluded.
# This is enforced by the kline pipeline: only x=True (closed) bars are written.
# "last_close_at_or_before_cutoff" is the aggregation policy (see AGGREGATION).
FORMING_CANDLE_ALLOWED = False


def clock_v2_v3_parameter_contract() -> dict[str, Any]:
    """Outcome-blind parameter accounting for a grouped action ranker."""
    return {
        "decision_group_unit": "one DAY ranking timestamp; BTC/ETH/SOL/XRP/HOLD share one information set",
        "candidates_in_a_group_are_not_independent_events": True,
        "intra_group_correlation": "same bar, shared BTC reference, shared quote time, shared capital/slot state",
        "categorical_encoding": "symbol dummies for BTC/ETH/SOL/XRP; HOLD is the omitted reference action",
        "numeric_feature_count": CLOCK_V2_NUMERIC_FEATURE_COUNT,
        "categorical_parameter_count": CLOCK_V2_CATEGORICAL_PARAMETER_COUNT,
        "intercept_count": CLOCK_V2_INTERCEPT_COUNT,
        "effective_fitted_parameters": CLOCK_V2_EFFECTIVE_FITTED_PARAMETERS,
        "events_per_parameter": EVENTS_PER_PARAMETER,
        "why_not_4x": "four candidates at one timestamp are one grouped observation",
        "chronological_dependence": "overlapping clock-horizon labels make nearby groups dependent until purged",
        "label_overlap": "purge window equals the undeclared max clock horizon once a horizon is frozen",
        "roles": {
            "A_model_fitting_sample_support": "fully comparable independent decision groups",
            "B_counterfactual_label_support": "same groups, same-horizon labels for every eligible action plus HOLD=0",
            "C_executable_cost_calibration": "authoritative fills used only to check spread/slippage realism",
            "D_production_lifecycle_validation": "real exits, separate from the ranking target",
            "E_final_live_policy_validation": "untouched lock used once after development",
        },
        "generic_140_selected_fills": {
            "justified_for": "D/E of a selected-trade 4H-entry predictor (generic day_model_readiness)",
            "justified_as_clock_v2_fitting_sample": False,
            "kept_unaltered_in_generic_gate": True,
        },
    }


def clock_v2_v3_readiness_requirements() -> dict[str, Any]:
    return {
        "version": "v3",
        "observation_unit": "independent_decision_group_after_purge",
        "min_fully_comparable_independent_groups": CLOCK_V2_EFFECTIVE_FITTED_PARAMETERS * EVENTS_PER_PARAMETER,
        "min_authoritative_fills_execution_calibration": CLOCK_V2_COST_CALIBRATION_PARAMETERS * EVENTS_PER_PARAMETER,
        "generic_140_selected_fills_not_the_fitting_population": True,
        "selected_trades_alone_insufficient": True,
        "do_not_count_four_candidates_as_four_events": True,
        "min_chronological_blocks_bookkeeping": MIN_CHRONOLOGICAL_BLOCKS,
        "block_definition": "UTC calendar day with >=1 decision group (bookkeeping only, not independent folds)",
        "validation_folds": "expanding chronological; grouped timestamp atomic; purge overlapping horizons; embargo >= max target horizon",
        # V3 was frozen when the horizon was not yet decided; value is immutable.
        "target_horizon_status": "TARGET_HORIZON_NOT_FROZEN",
        "train_blocked_until_horizon_frozen": True,
        "numeric_features_counted": CLOCK_V2_NUMERIC_FEATURE_COUNT,
        "categorical_parameters_counted": CLOCK_V2_CATEGORICAL_PARAMETER_COUNT,
        "effective_fitted_parameters": CLOCK_V2_EFFECTIVE_FITTED_PARAMETERS,
        "cost_calibration_parameters": CLOCK_V2_COST_CALIBRATION_PARAMETERS,
        "events_per_parameter": EVENTS_PER_PARAMETER,
        "parameter_contract": clock_v2_v3_parameter_contract(),
        "why_training_n_uses_19x10": (
            "19 = 1 intercept + 14 numeric clock-v2 fields + 4 symbol dummies. "
            "10 events per parameter is the conventional linear-support rule. "
            "The event is a purged decision group, not a selected fill and not a candidate row."
        ),
        "why_fill_n_uses_5x10": ("Executable-cost calibration has 5 free parameters (4 symbol spreads + slippage). Commission is a known exchange constant. This is not a lowered 140."),
        "missing_data_policy": MISSING_POLICY,
        "locked_test_rule": "feature capture allowed; outcomes uninspected until a later explicit task; lock used once",
    }


def comparable_label_contract() -> dict[str, Any]:
    return {
        "primary_target": PRIMARY_TARGET,
        "target_horizon_status": TARGET_HORIZON_STATUS,
        "horizon_outputs_are_not_the_same_target": True,
        "available_clock_horizons_sec": dict(CLOCK_LABEL_HORIZONS_SEC),
        "combination_rule": None,
        "same_for_every_eligible_coin": {
            "horizon": "undeclared_until_frozen",
            "price_methodology": EXECUTABLE_PRICE_METHOD,
            "commission_methodology": "expected_exchange_commission_rt",
            "spread_methodology": "decision_time_quote_spread_bps",
            "slippage_methodology": "expected_slippage_rt",
        },
        "HOLD_bps": 0.0,
        "production_lifecycle_is_separate_validation": True,
        "do_not_mix_production_exit_with_clock_markout_in_ranking_target": True,
    }


def planned_challenger_specification_v3() -> dict[str, Any]:
    """Corrected grouped-ranker readiness. Does not mutate v1 or v2."""
    schema = clock_challenger_export_schema()
    req = clock_v2_v3_readiness_requirements()
    return {
        "experiment_id": PLANNED_EXPERIMENT_ID_V3,
        "result": PLANNED_RESULT,
        "promoted": False,
        "train": False,
        "live_gate": False,
        "feature_set": f"{SCHEMA_VERSION}_capture_1",
        "feature_schema": list(schema["inputs"]),
        "categorical_encoding": "symbol dummies; HOLD omitted reference",
        "inputs": list(schema["inputs"]),
        "target": PRIMARY_TARGET,
        # V3 was frozen before the horizon decision; status is immutable here.
        "target_horizon_status": "TARGET_HORIZON_NOT_FROZEN",
        "target_contract": comparable_label_contract(),
        "actions": list(schema["actions"]),
        "hold_value_bps": 0.0,
        "model_class": "small_regularized_not_selected",
        "decision_group_unit": "one ranking timestamp",
        "minimum_trainable_comparable_groups": req["min_fully_comparable_independent_groups"],
        "minimum_authoritative_real_fill_support": req["min_authoritative_fills_execution_calibration"],
        "minimum_chronological_span": "5 UTC calendar-day blocks (bookkeeping) plus expanding purged folds",
        "fold_construction": "expanding chronological",
        "purge_window": "max target horizon once frozen; blocked while TARGET_HORIZON_NOT_FROZEN",
        "embargo": ">= max target horizon once frozen",
        "missingness_policy": MISSING_POLICY,
        "cost_calibration_method": "authoritative fills vs decision-time quote spread and named slippage; never substitute estimated_all_in_cost_bps for spread",
        "locked_test_rule": req["locked_test_rule"],
        "acceptance": {
            **future_acceptance_bar(),
            "primary_objective": "positive expected executable net after genuine costs",
            "profit_factor_above_one": True,
            "beats_current_champion": True,
            "majority_chronological_folds": True,
            "beats_hold_aware_baseline": True,
            "positive_untouched_lock": True,
            "drawdown_not_materially_worse": True,
            "robust_under_conservative_spread_slippage": True,
            "no_leakage": True,
            "win_rate_is_diagnostic_not_objective": True,
        },
        "training_procedure": {
            **planned_training_procedure(),
            "executed": False,
            "folds": "expanding_chronological",
            "grouped_decision_timestamp_atomic": True,
            "no_same_group_in_train_and_validation": True,
            "purge_overlapping_label_horizons": True,
            "embargo_seconds_min": None,
            "embargo_pending_horizon_freeze": True,
            "training_only_transforms": True,
            "lock_inspection_during_development": False,
            "untouched_lock_exactly_once": True,
        },
        "readiness_requirements": req,
        "statistical_contract": clock_v2_v3_parameter_contract(),
        "notes": (
            "V3 freezes grouped-ranker support before outcome research. Does not mutate M_clock_v2_planned_20260905 or M_clock_v2_planned_v2_20260905. TARGET_HORIZON_NOT_FROZEN blocks training."
        ),
    }


def clock_v2_v4_readiness_requirements() -> dict[str, Any]:
    """V4 readiness: v3 grouped-ranker numbers unchanged; horizon now frozen at 3h."""
    v3 = clock_v2_v3_readiness_requirements()
    return {
        **v3,
        "target_horizon_status": TARGET_HORIZON_STATUS,
        "primary_target_horizon_sec": PRIMARY_TARGET_HORIZON_SEC,
        "primary_target_horizon_name": PRIMARY_TARGET_HORIZON_NAME,
        "embargo_seconds_min": max(PRIMARY_TARGET_HORIZON_SEC, *CLOCK_LABEL_HORIZONS_SEC.values()),
        "purge_window_seconds": PRIMARY_TARGET_HORIZON_SEC,
        "horizon_freeze_evidence": (
            "scripts/train_day_path_net.py defines DAY_HORIZONS_MIN=(60,120,180) with 180 "
            "as design ceiling; models/day_path_net_v1.json stores primary_horizon_min=180; "
            "180 rows of 1m OHLCV == 180 minutes == 3 clock hours. "
            "Horizon NOT selected from P&L inspection of locked outcomes."
        ),
        "horizon_not_chosen_by_pnl": True,
    }


def planned_challenger_specification_v4() -> dict[str, Any]:
    """V4: same as v3 but target horizon is frozen at 3h. Does not mutate v1/v2/v3."""
    req = clock_v2_v4_readiness_requirements()
    schema = clock_challenger_export_schema()
    return {
        "experiment_id": PLANNED_EXPERIMENT_ID_V4,
        "result": PLANNED_RESULT,
        "promoted": False,
        "train": False,
        "live_gate": False,
        "feature_set": f"{SCHEMA_VERSION}_capture_1",
        "inputs": list(schema["inputs"]),
        "target": PRIMARY_TARGET,
        "primary_target_horizon_sec": PRIMARY_TARGET_HORIZON_SEC,
        "primary_target_horizon_name": PRIMARY_TARGET_HORIZON_NAME,
        "target_horizon_status": TARGET_HORIZON_STATUS,
        "executable_price_method": EXECUTABLE_PRICE_METHOD,
        "commission_methodology": "estimated_all_in_cost_bps from decision_book_tape spread at entry",
        "spread_methodology": "recomputed from live bid/ask at decision time; stored in candidate artifact",
        "slippage_methodology": "named expected_slippage_bps from entry telemetry",
        "hold_value_bps": 0.0,
        "actions": list(schema["actions"]),
        "forming_candle_allowed": FORMING_CANDLE_ALLOWED,
        "diagnostic_horizons": [k for k in CLOCK_LABEL_HORIZONS_SEC if k != PRIMARY_TARGET_HORIZON_NAME],
        "diagnostic_horizons_not_alternate_targets": True,
        "model_class": "small_regularized_not_selected",
        "acceptance": {
            **future_acceptance_bar(),
            "primary_objective": "positive expected executable net after genuine costs",
            "profit_factor_above_one": True,
            "beats_current_champion": True,
            "majority_chronological_folds": True,
            "beats_hold_aware_baseline": True,
            "positive_untouched_lock": True,
            "drawdown_not_materially_worse": True,
            "robust_under_conservative_spread_slippage": True,
            "no_leakage": True,
        },
        "training_procedure": {
            **planned_training_procedure(),
            "executed": False,
            "folds": "expanding_chronological",
            "grouped_decision_timestamp_atomic": True,
            "no_same_group_in_train_and_validation": True,
            "purge_overlapping_label_horizons": True,
            "embargo_seconds_min": req["embargo_seconds_min"],
            "training_only_transforms": True,
            "lock_inspection_during_development": False,
            "untouched_lock_exactly_once": True,
        },
        "readiness_requirements": req,
        "statistical_contract": clock_v2_v3_parameter_contract(),
        "notes": (
            "V4 freezes the primary target horizon at 3h based on pre-research design evidence "
            "(DAY_HORIZONS_MIN max = 180 min). Does not mutate v1/v2/v3. "
            "Diagnostic horizons (15m/30m/1h/2h/4h) are NOT selectable after performance inspection."
        ),
    }


# ---------------------------------------------------------------------------
# V5: corrected action semantics. Does not mutate v1/v2/v3/v4.
#
# V4 listed `final_rank_score` as a model input. That field is only defined for
# symbols the legacy 15m signal consumer admitted as buy-intent candidates: it is
# the output of the bandit / adaptive-weight / symbol-trust / thesis / haircut
# chain applied to a constructed BuyCandidate. Coins with no candidate have no
# such score, and capture-v1 silently backfilled raw path_ev in its place.
#
# Option A (reconstruct an all-action shadow legacy rank) was rejected: the chain
# consumes candidate-only inputs (buy_margin, setup_type_canonical, confidence,
# chop/regime penalties) that do not exist for a coin whose signal side was
# `hold`, so any all-action value would require inventing signal state. That is
# not deterministically reproducible and fails point-in-time recovery.
#
# Option B is taken: `final_rank_score` is removed from the v5 model schema
# because it is not a well-defined all-action feature. Its provenance is still
# CAPTURED and audited (legacy_final_rank_score / _valid / _reason) so the
# fabrication is visible; it is simply not a model input.
#
# Per the correction rule, dropping a parameter must not lower the support bar:
# required groups = max(previous v4 requirement, new parameter count * 10).
# ---------------------------------------------------------------------------

PLANNED_EXPERIMENT_ID_V5 = "M_clock_v2_planned_v5_20260905"
FINAL_RANK_TREATMENT_V5 = "REMOVED_NOT_WELL_DEFINED_ALL_ACTION_FEATURE"
FINAL_RANK_PROVENANCE_CAPTURED = True

REQUIRED_CLOCK_V2_FIELDS_V5: tuple[str, ...] = tuple(n for n in REQUIRED_CLOCK_V2_FIELDS if n != "final_rank_score")
CLOCK_V2_V5_LISTED_INPUT_COUNT = len(REQUIRED_CLOCK_V2_FIELDS_V5)  # 14
CLOCK_V2_V5_NUMERIC_FEATURE_COUNT = CLOCK_V2_V5_LISTED_INPUT_COUNT - len(CLOCK_V2_CATEGORICAL_FEATURES)  # 13
CLOCK_V2_V5_EFFECTIVE_FITTED_PARAMETERS = CLOCK_V2_INTERCEPT_COUNT + CLOCK_V2_V5_NUMERIC_FEATURE_COUNT + CLOCK_V2_CATEGORICAL_PARAMETER_COUNT  # 18

# max(190, 18*10) == 190. The correction cannot make training happen sooner.
CLOCK_V2_V5_REQUIRED_GROUPS = max(
    CLOCK_V2_EFFECTIVE_FITTED_PARAMETERS * EVENTS_PER_PARAMETER,
    CLOCK_V2_V5_EFFECTIVE_FITTED_PARAMETERS * EVENTS_PER_PARAMETER,
)
CLOCK_V2_V5_REQUIRED_CALIBRATION_FILLS = CLOCK_V2_COST_CALIBRATION_PARAMETERS * EVENTS_PER_PARAMETER  # 50
CLOCK_V2_V5_EMBARGO_SEC = max(4 * 3600, PRIMARY_TARGET_HORIZON_SEC)


def clock_v2_v5_feature_schema() -> dict[str, Any]:
    """All-action feature schema. Every listed input must be definable for every
    production-available action, or it does not belong in the schema."""
    return {
        "inputs": list(REQUIRED_CLOCK_V2_FIELDS_V5),
        "removed_from_v4": ["final_rank_score"],
        "removal_reason": (
            "legacy final_rank_score is only defined for legacy scored candidates; it is not an "
            "all-action feature and capture-v1 substituted raw path_ev for absent candidates"
        ),
        "all_action_inputs_verified": {
            "p_buy": "per-symbol ML signal published for all four coins regardless of side; "
            "captured as production_p_buy with shadow_candidate_p_buy fallback and explicit provenance",
            "legacy_path_ev": "scored independently for all four coins by score_four_coins",
            "clock_features": "computed from 1m klines per symbol, independent of candidacy",
            "structure_and_cost": "per-symbol 4H structure and decision-time quote",
        },
        "categorical_encoding": "symbol dummies for BTC/ETH/SOL/XRP; HOLD is the omitted reference action",
        "targets": [PRIMARY_TARGET],
        "target_horizon_sec": PRIMARY_TARGET_HORIZON_SEC,
        "hold_value_bps": 0.0,
        "actions": [*COINS, HOLD_SYMBOL],
        "train": False,
        "live_gate": False,
    }


def clock_v2_v5_readiness_requirements() -> dict[str, Any]:
    """Frozen immutable v5 readiness contract."""
    v4 = clock_v2_v4_readiness_requirements()
    return {
        **v4,
        "version": "v5",
        "action_contract_version": "day_clock_v2_action_contract_v1",
        "partition_contract_version": "day_clock_v2_partition_v1",
        "observation_unit": "independent_v5_DEVELOPMENT_decision_group_after_purge",
        "counted_partition": "DEVELOPMENT",
        "excluded_partitions": ["PRE_MODEL_QUARANTINE", "FINAL_TEST"],
        "feature_complete_definition": (
            "every production-available modeled action (action_available=true) plus HOLD has the "
            "required v5 feature state; a legacy-unscored action does not disappear"
        ),
        "fully_comparable_definition": (
            "all production-available actions share the 3h horizon, executable-price method, "
            "commission method, spread method and slippage method, and all have valid labels; HOLD=0"
        ),
        "listed_inputs": list(REQUIRED_CLOCK_V2_FIELDS_V5),
        "listed_input_count": CLOCK_V2_V5_LISTED_INPUT_COUNT,
        "numeric_features_counted": CLOCK_V2_V5_NUMERIC_FEATURE_COUNT,
        "effective_fitted_parameters": CLOCK_V2_V5_EFFECTIVE_FITTED_PARAMETERS,
        "previous_v4_required_groups": CLOCK_V2_EFFECTIVE_FITTED_PARAMETERS * EVENTS_PER_PARAMETER,
        "min_feature_complete_groups": CLOCK_V2_V5_REQUIRED_GROUPS,
        "min_fully_comparable_independent_groups": CLOCK_V2_V5_REQUIRED_GROUPS,
        "min_authoritative_fills_execution_calibration": CLOCK_V2_V5_REQUIRED_CALIBRATION_FILLS,
        "support_floor_rule": "required_groups = max(previous_v4_requirement, new_parameter_count * 10)",
        "support_floor_not_lowered": True,
        "embargo_seconds_min": CLOCK_V2_V5_EMBARGO_SEC,
        "purge_window_seconds": PRIMARY_TARGET_HORIZON_SEC,
        "target": PRIMARY_TARGET,
        "target_horizon_name": PRIMARY_TARGET_HORIZON_NAME,
        "hold_target_bps": 0.0,
        "require_zero_future_data_violations": True,
        "require_accounting_pass": True,
        "require_no_lock_inspection": True,
        "require_final_test_contract_before_promotion": True,
        "final_test_status_required_before_promotion": "DECLARED_FUTURE_WINDOW",
        "train_on_readiness_pass": False,
        "auto_train_forbidden": True,
        "final_rank_treatment": FINAL_RANK_TREATMENT_V5,
        "final_rank_provenance_still_captured": FINAL_RANK_PROVENANCE_CAPTURED,
    }


def planned_challenger_specification_v5() -> dict[str, Any]:
    """V5: corrected action semantics, own partition, final_rank_score removed."""
    req = clock_v2_v5_readiness_requirements()
    schema = clock_v2_v5_feature_schema()
    return {
        "experiment_id": PLANNED_EXPERIMENT_ID_V5,
        "parent_contracts": [
            PLANNED_EXPERIMENT_ID,
            PLANNED_EXPERIMENT_ID_V2,
            PLANNED_EXPERIMENT_ID_V3,
            PLANNED_EXPERIMENT_ID_V4,
        ],
        "result": PLANNED_RESULT,
        "promoted": False,
        "train": False,
        "live_gate": False,
        "feature_set": f"{SCHEMA_VERSION}_capture_2_action_corrected",
        "inputs": list(schema["inputs"]),
        "feature_schema": schema,
        "removed_inputs": list(schema["removed_from_v4"]),
        "final_rank_treatment": FINAL_RANK_TREATMENT_V5,
        "target": PRIMARY_TARGET,
        "primary_target_horizon_sec": PRIMARY_TARGET_HORIZON_SEC,
        "primary_target_horizon_name": PRIMARY_TARGET_HORIZON_NAME,
        "target_horizon_status": TARGET_HORIZON_STATUS,
        "executable_price_method": EXECUTABLE_PRICE_METHOD,
        "commission_methodology": "expected_exchange_commission_rt, identical for every action",
        "spread_methodology": "decision_time_quote_spread_bps, identical for every action",
        "slippage_methodology": "expected_slippage_rt, identical for every action",
        "hold_value_bps": 0.0,
        "actions": list(schema["actions"]),
        "forming_candle_allowed": FORMING_CANDLE_ALLOWED,
        "model_class": "small_regularized_not_selected",
        "effective_fitted_parameters": CLOCK_V2_V5_EFFECTIVE_FITTED_PARAMETERS,
        "minimum_trainable_comparable_groups": CLOCK_V2_V5_REQUIRED_GROUPS,
        "minimum_authoritative_real_fill_support": CLOCK_V2_V5_REQUIRED_CALIBRATION_FILLS,
        "acceptance": {
            **future_acceptance_bar(),
            "primary_objective": "positive expected executable net after genuine costs",
            "profit_factor_above_one": True,
            "beats_current_champion": True,
            "majority_chronological_folds": True,
            "beats_hold_aware_baseline": True,
            "drawdown_not_materially_worse": True,
            "no_leakage": True,
            "future_final_test_required_before_promotion": True,
        },
        "training_procedure": {
            **planned_training_procedure(),
            "executed": False,
            "folds": "expanding_chronological",
            "grouped_decision_timestamp_atomic": True,
            "no_same_group_in_train_and_validation": True,
            "purge_overlapping_label_horizons": True,
            "embargo_seconds_min": CLOCK_V2_V5_EMBARGO_SEC,
            "training_partition": "DEVELOPMENT",
            "quarantine_excluded": True,
            "lock_inspection_during_development": False,
            "generic_4h_lock_untouched": True,
        },
        "readiness_requirements": req,
        "statistical_contract": clock_v2_v3_parameter_contract(),
        "notes": (
            "V5 corrects action semantics: action_available is path_input_valid plus real hard "
            "gates, not legacy candidate-list membership. Uses the clock-v2 partition contract "
            "instead of the open-ended 4H lock, which stays sealed and uninspected. "
            "final_rank_score removed as not all-action definable; support floor held at "
            f"{CLOCK_V2_V5_REQUIRED_GROUPS} groups. Does not mutate v1/v2/v3/v4. PLANNED_NOT_RUN."
        ),
    }
