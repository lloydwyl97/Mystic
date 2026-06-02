"""
BUY admission for 3-class models: sleeve-aware thresholds on buy_margin.

buy_margin = P_buy - max(P_hold, P_sell)

MIN_CONFIDENCE / MIN_CONFIDENCE_BUY in signal_thresholds remains for:
- legacy non-ML paths where buy_margin is absent
- sizing/churn heuristics that reference winner probability (see portfolio_engine docs)
"""

from __future__ import annotations

import math
import os
from typing import Any


def buy_margin_threshold_core() -> float:
    """Stricter floor for CORE sleeve (higher required separation)."""
    return float(os.getenv("BUY_MARGIN_THRESHOLD_CORE", "0.08"))


def buy_margin_threshold_active() -> float:
    """Looser floor for ACTIVE sleeve."""
    return float(os.getenv("BUY_MARGIN_THRESHOLD_ACTIVE", "0.05"))


def compute_buy_margin(probs: dict[str, float] | dict[str, Any]) -> float:
    """Canonical BUY separation score: P_buy - max(P_hold, P_sell)."""
    try:
        b = float(probs.get("buy", 0.0) or 0.0)
        h = float(probs.get("hold", 0.0) or 0.0)
        s = float(probs.get("sell", 0.0) or 0.0)
    except (TypeError, ValueError):
        return 0.0
    return b - max(h, s)


def parse_float_field(raw: Any) -> float | None:
    if raw is None:
        return None
    if isinstance(raw, bytes):
        raw = raw.decode()
    try:
        v = float(raw)
        if not math.isfinite(v):
            return None
        return v
    except (TypeError, ValueError):
        return None


def resolve_buy_margin_from_payload(decision_data: dict[str, Any]) -> float | None:
    """
    Read buy_margin from Redis/hash, or recompute from prob_buy/prob_hold/prob_sell if present.
    """
    bm = parse_float_field(decision_data.get("buy_margin"))
    if bm is not None:
        return bm
    pb = parse_float_field(decision_data.get("prob_buy"))
    ph = parse_float_field(decision_data.get("prob_hold"))
    ps = parse_float_field(decision_data.get("prob_sell"))
    if pb is not None and ph is not None and ps is not None:
        return compute_buy_margin({"buy": pb, "hold": ph, "sell": ps})
    return None
