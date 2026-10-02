"""Sources a SCALP candidate's raw expected directional move may come from.

STRATEGY_CLAIM is the setup's own structural projection (distance to the
reclaim/bounce/breakout target the strategy confirmed). It is the only
directional estimate SCALP produces at decision time:

- ATR magnitude is volatility, not direction (ATR_ESTIMATE is forensic only:
  historical rows recorded before this contract).
- The microstructure EV is a fixed-weight heuristic already net of half-spread;
  micro features enter the edge as the bounded micro residual, not as raw.
- Momentum changes are realized past moves, not a forecast.

NONE means no directional claim exists: NO_EXECUTABLE_EDGE_ESTIMATE.
"""

from __future__ import annotations

STRATEGY_CLAIM = "STRATEGY_CLAIM"
NO_RAW_MOVE_SOURCE = "NONE"
ATR_ESTIMATE = "ATR_ESTIMATE"

VALID_RAW_MOVE_SOURCES: frozenset[str] = frozenset({STRATEGY_CLAIM})

_LEGACY = {"strategy": STRATEGY_CLAIM, "atr_estimate": ATR_ESTIMATE, "unavailable": NO_RAW_MOVE_SOURCE, "": NO_RAW_MOVE_SOURCE}


def normalize_raw_move_source(source: str | None) -> str:
    """Strict enum for stored or live source labels (legacy spellings mapped)."""
    s = str(source or "").strip()
    if s.upper() in {STRATEGY_CLAIM, NO_RAW_MOVE_SOURCE, ATR_ESTIMATE}:
        return s.upper()
    return _LEGACY.get(s.lower(), NO_RAW_MOVE_SOURCE)


def is_directional(source: str | None) -> bool:
    return normalize_raw_move_source(source) in VALID_RAW_MOVE_SOURCES


__all__ = [
    "ATR_ESTIMATE",
    "NO_RAW_MOVE_SOURCE",
    "STRATEGY_CLAIM",
    "VALID_RAW_MOVE_SOURCES",
    "is_directional",
    "normalize_raw_move_source",
]
