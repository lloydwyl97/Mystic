"""
Shared "near-pass" scalp opportunity helpers.

Moved out of scripts/watch_scalp_entry_opportunity.py: the live status API
(backend/services/binance_scalp/status_snapshot.py) is core backend truth
consumed by the dashboard, and must not depend on an operator CLI script
living under scripts/. scripts/ remains a launch/operator entry point that
imports these helpers, not the other way around.
"""

from __future__ import annotations

import time
from typing import Any

from backend.services.binance_scalp.economics import ScalpEconomics
from backend.services.binance_scalp.market_reader import ScalpMarketReader
from backend.services.binance_scalp.momentum_tracker import MomentumTracker
from backend.services.binance_scalp.protected_preflight import (
    DEPTH_INSUFFICIENT,
    PRICE_IMPACT_TOO_HIGH,
    SPREAD_TOO_WIDE,
)

NEAR_PASS_THRESHOLD = 0.0005  # 0.05% from passing
HQ_NEAR_PASS_DISALLOWED = frozenset({SPREAD_TOO_WIDE, DEPTH_INSUFFICIENT, PRICE_IMPACT_TOO_HIGH})


def _distance_to_pass(
    *,
    projected_gross: float,
    required_gross: float,
    projected_surplus: float,
    min_surplus: float,
) -> dict[str, float]:
    dist_gross = max(0.0, required_gross - projected_gross)
    dist_surplus = max(0.0, min_surplus - projected_surplus)
    combined = max(dist_gross, dist_surplus)
    return {
        "distance_gross_pct": dist_gross,
        "distance_surplus_pct": dist_surplus,
        "distance_to_pass_pct": combined,
    }


def is_high_quality_near_pass(row: dict[str, Any], econ: ScalpEconomics) -> bool:
    """Conservative pre-arm candidate — does not bypass engine entry gate."""
    if row.get("error") or row.get("preflight_pass"):
        return False
    reject = row.get("reject_reason") or ""
    if reject in HQ_NEAR_PASS_DISALLOWED:
        return False

    def _pct(key: str, default: float = 999.0) -> float:
        val = row.get(key)
        return float(val) if val is not None else default

    spread = _pct("spread_pct")
    buy_i = _pct("buy_impact_pct")
    sell_i = _pct("sell_impact_pct")
    if spread > econ.spread_cap_pct:
        return False
    if buy_i > econ.impact_cap_pct or sell_i > econ.impact_cap_pct:
        return False
    if not econ.is_fee_model_verified():
        return False
    if not row.get("momentum_confirmed"):
        return False
    if not row.get("breakout_confirmed"):
        return False
    if float(row.get("projected_surplus") or -1.0) < 0.0:
        return False
    dist = float((row.get("distance_to_pass") or {}).get("distance_to_pass_pct") or 999.0)
    return not dist > NEAR_PASS_THRESHOLD


def build_arm_event(row: dict[str, Any], *, arm_reason: str) -> dict[str, Any]:
    dist = row.get("distance_to_pass") or {}
    return {
        "event": "HIGH_QUALITY_NEAR_PASS_ARMED" if arm_reason == "HIGH_QUALITY_NEAR_PASS" else arm_reason,
        "arm_reason": arm_reason,
        "arm_symbol": row.get("symbol"),
        "arm_distance_to_pass": dist.get("distance_to_pass_pct"),
        "arm_projected_gross": row.get("projected_gross"),
        "arm_required_gross": row.get("required_gross"),
        "arm_surplus": row.get("projected_surplus"),
        "arm_ts": row.get("ts"),
        "row": row,
    }


def warm_momentum(
    reader: ScalpMarketReader,
    tracker: MomentumTracker,
    symbols: tuple[str, ...],
    *,
    rounds: int = 8,
    interval_sec: float = 5.0,
) -> None:
    for _ in range(rounds):
        now = time.time()
        for sym in symbols:
            snap = reader.read(sym)
            if snap:
                tracker.record(sym, now, snap.best_bid, snap.mid)
        time.sleep(interval_sec)


__all__ = [
    "HQ_NEAR_PASS_DISALLOWED",
    "NEAR_PASS_THRESHOLD",
    "_distance_to_pass",
    "build_arm_event",
    "is_high_quality_near_pass",
    "warm_momentum",
]
