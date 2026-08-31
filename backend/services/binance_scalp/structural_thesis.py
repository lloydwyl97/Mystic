"""New SCALP book: structural economics, not prediction ranking.

The 1-120s / 3-20m / maker-on-ranker theses are retired. This module is the
only authorized SCALP entry policy.

Executable structural modes (liquidity provision, locked cross-market arb)
are not armed. Existing tape cannot honestly prove at-touch maker fills.
Do not treat mid dislocation as arb.
"""

from __future__ import annotations

from typing import Any

THESIS_STRUCTURAL = "structural"
THESIS_LEGACY_PREDICTION = "legacy_prediction"
PREDICTION_RETIRED = "SCALP_PREDICTION_THESIS_RETIRED"
STRUCTURAL_NOT_EXECUTABLE = "SCALP_STRUCTURAL_NOT_EXECUTABLE"


def normalize_thesis(raw: Any) -> str:
    thesis = str(raw or THESIS_STRUCTURAL).strip().lower()
    if thesis == THESIS_LEGACY_PREDICTION:
        return THESIS_LEGACY_PREDICTION
    return THESIS_STRUCTURAL


def prediction_entries_permitted(config: Any) -> bool:
    """Legacy VWAP/pullback/momentum/imbalance ranking may buy only if both flags are on."""
    thesis = normalize_thesis(getattr(config, "scalp_thesis", THESIS_STRUCTURAL))
    return thesis == THESIS_LEGACY_PREDICTION and bool(getattr(config, "legacy_prediction_entries", False))


def new_entry_block_reason(config: Any) -> str | None:
    if prediction_entries_permitted(config):
        return None
    if normalize_thesis(getattr(config, "scalp_thesis", THESIS_STRUCTURAL)) == THESIS_STRUCTURAL:
        return STRUCTURAL_NOT_EXECUTABLE
    return PREDICTION_RETIRED


def status_fields(config: Any) -> dict[str, Any]:
    thesis = normalize_thesis(getattr(config, "scalp_thesis", THESIS_STRUCTURAL))
    return {
        "scalp_thesis": thesis,
        "legacy_prediction_entries": bool(getattr(config, "legacy_prediction_entries", False)),
        "prediction_entries_permitted": prediction_entries_permitted(config),
        "structural_entries_executable": False,
        "new_entry_block_reason": new_entry_block_reason(config),
    }
