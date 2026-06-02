import logging
import math
import os
from collections.abc import Iterable, Mapping
from typing import Any

logger = logging.getLogger(__name__)


def _to_float(value: Any, default: float) -> float:
    try:
        f = float(value)
        if math.isnan(f):  # NaN check
            return default
    except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError):
        return default
    else:
        return f


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def run_trial(accuser: str, defendant: str, facts: Mapping[str, Any], witnesses: Iterable[Any]) -> str:
    """
    Evaluate a simple guilt score using provided facts and witnesses.
    Logging only (no prints). Thresholds can be tuned via env:
      - TRIAL_GUILT_THRESHOLD (default 0.7)
      - TRIAL_WITNESS_WEIGHT  (default 0.1 per witness)
      - TRIAL_WITNESS_CAP     (default 0.5, max total witness contribution)
    """
    threshold = _clamp(_to_float(os.getenv("TRIAL_GUILT_THRESHOLD", "0.7"), 0.7), 0.0, 1.0)
    per_witness = _clamp(_to_float(os.getenv("TRIAL_WITNESS_WEIGHT", "0.1"), 0.1), 0.0, 1.0)
    witness_cap = _clamp(_to_float(os.getenv("TRIAL_WITNESS_CAP", "0.5"), 0.5), 0.0, 1.0)

    # Ensure facts behaves like a mapping with .get
    _facts = {} if facts is None or not hasattr(facts, "get") else facts

    evidence_score = _clamp(_to_float(_facts.get("evidence", 0.0), 0.0), 0.0, 1.0)

    # Robust witness count handling: accept None, int (as count), iterable, or generator
    if witnesses is None:
        witness_count = 0
    elif isinstance(witnesses, int):
        witness_count = max(0, witnesses)
    # Avoid treating strings/bytes as sequences of witnesses
    elif isinstance(witnesses, (str, bytes)):
        witness_count = 1 if witnesses else 0
    else:
        # Try to get length without exhausting generator if possible
        try:
            witness_count = len(witnesses)  # type: ignore[arg-type]
        except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError):
            try:
                witness_count = len(list(witnesses))
            except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError):
                witness_count = 0

    witness_contrib = min(witness_count * per_witness, witness_cap)

    guilt_score = _clamp(evidence_score + witness_contrib, 0.0, 1.0)
    outcome = "Guilty" if guilt_score > threshold else "Innocent"

    logger.info(
        "Trial evaluated | accuser=%s defendant=%s evidence=%.3f witnesses=%d per_witness=%.3f witness_cap=%.3f threshold=%.3f guilt_score=%.3f outcome=%s",
        accuser,
        defendant,
        evidence_score,
        witness_count,
        per_witness,
        witness_cap,
        threshold,
        guilt_score,
        outcome,
    )

    return outcome
