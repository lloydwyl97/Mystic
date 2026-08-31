"""New SCALP book: structural economics, not prediction ranking.

The 1-120s / 3-20m / maker-on-ranker theses are retired. This module is the
only authorized SCALP entry policy.

Paper LP rests at touch and fills only on through-price. Cross-venue / mid
dislocation is not arb and is not executable. Live is not armed.
The prediction-book circuit breaker does not apply to this thesis.
"""

from __future__ import annotations

from typing import Any

THESIS_STRUCTURAL = "structural"
THESIS_LEGACY_PREDICTION = "legacy_prediction"
PREDICTION_RETIRED = "SCALP_PREDICTION_THESIS_RETIRED"
STRUCTURAL_RANKING_BLOCKED = "SCALP_STRUCTURAL_NOT_EXECUTABLE"


def normalize_thesis(raw: Any) -> str:
    thesis = str(raw or THESIS_STRUCTURAL).strip().lower()
    if thesis == THESIS_LEGACY_PREDICTION:
        return THESIS_LEGACY_PREDICTION
    return THESIS_STRUCTURAL


def prediction_entries_permitted(config: Any) -> bool:
    """Legacy VWAP/pullback/momentum/imbalance ranking may buy only if both flags are on."""
    thesis = normalize_thesis(getattr(config, "scalp_thesis", THESIS_STRUCTURAL))
    return thesis == THESIS_LEGACY_PREDICTION and bool(getattr(config, "legacy_prediction_entries", False))


def ranking_eval_permitted(config: Any) -> bool:
    return prediction_entries_permitted(config)


def prediction_circuit_breaker_applies(config: Any) -> bool:
    """Consec-loss breaker belongs to the retired ranking book only."""
    return prediction_entries_permitted(config)


def structural_lp_executable(config: Any) -> bool:
    if normalize_thesis(getattr(config, "scalp_thesis", THESIS_STRUCTURAL)) != THESIS_STRUCTURAL:
        return False
    if bool(getattr(config, "scalp_live", False)):
        return False
    return bool(getattr(config, "scalp_paper_enabled", False))


def new_entry_block_reason(config: Any) -> str | None:
    """Blocks ranking `_try_entry` only. Structural LP uses its own tick path."""
    if prediction_entries_permitted(config):
        return None
    if normalize_thesis(getattr(config, "scalp_thesis", THESIS_STRUCTURAL)) == THESIS_STRUCTURAL:
        return STRUCTURAL_RANKING_BLOCKED
    return PREDICTION_RETIRED


def status_fields(config: Any) -> dict[str, Any]:
    thesis = normalize_thesis(getattr(config, "scalp_thesis", THESIS_STRUCTURAL))
    return {
        "scalp_thesis": thesis,
        "legacy_prediction_entries": bool(getattr(config, "legacy_prediction_entries", False)),
        "prediction_entries_permitted": prediction_entries_permitted(config),
        "ranking_eval_permitted": ranking_eval_permitted(config),
        "prediction_circuit_breaker_applies": prediction_circuit_breaker_applies(config),
        "structural_entries_executable": structural_lp_executable(config),
        "structural_arb_executable": False,
        "structural_fill_model": "through_price_only",
        "new_entry_block_reason": new_entry_block_reason(config),
    }


# Compat alias used by older tests/imports
STRUCTURAL_NOT_EXECUTABLE = STRUCTURAL_RANKING_BLOCKED
