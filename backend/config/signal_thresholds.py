"""
`MIN_CONFIDENCE` / `MIN_CONFIDENCE_BUY` — legacy **winner-probability** floor.

BUY admission for ML `ai_signal:*` uses **buy_margin** and
`BUY_MARGIN_THRESHOLD_CORE` / `BUY_MARGIN_THRESHOLD_ACTIVE` (see `buy_admission.py`).

`MIN_CONFIDENCE` remains referenced for:
- non-ML / legacy paths that only expose a scalar confidence
- churn / quality heuristics in `portfolio_engine` that tighten on winner probability
- historical scripts and APIs using `ConfidenceNormalizer.is_above_threshold`
"""

from __future__ import annotations

import os


def min_confidence_buy() -> float:
    """Minimum normalized confidence [0,1] for a BUY to be emitted or executed.
    ML uses buy_margin as primary edge; this floor is for legacy/scalar paths.
    Lowered to 0.58 to allow the model to generate real trades and learnable outcomes
    when it sees edge (current outputs often ~0.58-0.65 on "buy" symbols), instead of sitting dead.
    """
    return float(os.getenv("MIN_CONFIDENCE_OVERRIDE") or os.getenv("MIN_CONFIDENCE", "0.58"))


# Resolved once at import (same pattern as engine constants)
MIN_CONFIDENCE_BUY = min_confidence_buy()
