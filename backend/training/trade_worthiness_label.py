"""
Binary trade-worthiness label for per-coin RF training (close-only path).

Positive = forward path shows enough favorable excursion to clear execution
economics and is not a dead / no-traction loser at the early check.
"""

from __future__ import annotations

import os
from collections.abc import Sequence

import numpy as np

from backend.config.training_label_economics import (
    NO_TRACTION_CHECK_FRAC,
    NO_TRACTION_MIN_MFE_PCT,
    ambiguous_max_mfe_pct_for_training,
    required_edge_pct_for_training,
)


def trade_worthiness_binary_label(
    closes: np.ndarray | Sequence[float],
    entry_index: int,
    horizon_bars: int,
    *,
    required_edge_pct: float | None = None,
    traction_check_frac: float = NO_TRACTION_CHECK_FRAC,
    traction_min_mfe_pct: float = NO_TRACTION_MIN_MFE_PCT,
    ambiguous_max_mfe_pct: float | None = None,
) -> int | None:
    """
    Returns
        1 — worth taking: clears edge after friction; not a no-traction loser.
        0 — not worth taking.
        None — skip row (ambiguous flat path; max excursion below ambiguous floor).

    `closes` must be the price series used in training (feature column 0), oldest-first.
    `horizon_bars` is a **primary-clock** bar count (5m day / 15m day) for v3; see
    ``label_horizon_bars_for_strategy`` in ``trade_worthiness_timing``.
    """
    if horizon_bars < 2:
        return None
    arr = np.asarray(closes, dtype=np.float64)
    if entry_index < 0 or entry_index + horizon_bars >= len(arr):
        return None
    entry = float(arr[entry_index])
    if entry <= 0 or not np.isfinite(entry):
        return None

    req = float(required_edge_pct) if required_edge_pct is not None else required_edge_pct_for_training()
    if not np.isfinite(req) or req <= 0:
        return None

    amb = float(ambiguous_max_mfe_pct) if ambiguous_max_mfe_pct is not None else ambiguous_max_mfe_pct_for_training()

    fwd = arr[entry_index + 1 : entry_index + 1 + horizon_bars]
    if fwd.size < horizon_bars:
        return None

    max_mfe = float(np.max(fwd) / entry - 1.0)
    if not np.isfinite(max_mfe):
        return None
    if max_mfe < amb:
        return None

    tb_frac = horizon_bars * traction_check_frac
    tb_rounded = round(tb_frac)
    tb = max(1, min(tb_rounded, horizon_bars))
    window_tb = fwd[:tb]
    mfe_at_tb = float(np.max(window_tb) / entry - 1.0)
    close_tb = float(fwd[tb - 1])
    underwater = close_tb < entry
    no_traction_loser = (mfe_at_tb < traction_min_mfe_pct) and underwater

    if no_traction_loser:
        return 0
    if max_mfe >= req:
        return 1
    return 0
