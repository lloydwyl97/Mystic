"""DAY V2 structure runner: setup objectives and the winner ratchet.

DAY is a day trader. A winner is never sold at a small fixed target; it is
held until the move proves itself (1x the 1h ATR of favourable excursion),
then protected by a stop that trails the high-water mark by a 1h-ATR distance
and tightens once the setup's structural objective has been reached.

The stop is a pure non-decreasing function of the high-water mark, so it can
only move in the profitable direction and survives restarts through the
persisted ``highest_price``.

Loss protection (catastrophic ATR stop, structural invalidation) lives in
``live_exit_evaluator`` and is unchanged by this contract.
"""

from __future__ import annotations

import math
from typing import Any

DAY_EXIT_CONTRACT_RUNNER: str = "DAY_V2_STRUCTURE_RUNNER_V1"

RUNNER_ACTIVATION_ATR_1H: float = 1.0
RUNNER_TRAIL_ATR_1H: float = 1.5
RUNNER_TIGHT_TRAIL_ATR_1H: float = 0.75

# Median 1h/15m ATR ratio across BTC/ETH/SOL/XRP over 180 days (2.3-2.5).
# Used only for positions opened before the 1h ATR was stamped at entry.
LEGACY_ATR_1H_PER_ATR_15M: float = 2.5

_OBJECTIVE_ATR_FLOOR: dict[str, float] = {
    "HTF_TREND_PULLBACK": 2.0,
    "BREAKOUT_CONTINUATION": 2.0,
    "RANGE_BOUNCE": 1.5,
    "VWAP_REVERSION": 1.5,
    "EXHAUSTION_MR": 1.5,
}
_DEFAULT_OBJECTIVE_ATR_FLOOR: float = 1.5


def atr_from_bars(bars: list[dict[str, Any]], n: int = 14) -> float:
    """Simple-mean true range over the last ``n`` closed bars; 0.0 when too short."""
    if len(bars) < n + 1:
        return 0.0
    trs = []
    for i in range(len(bars) - n, len(bars)):
        hi = float(bars[i]["high"])
        lo = float(bars[i]["low"])
        prev_c = float(bars[i - 1]["close"])
        trs.append(max(hi - lo, abs(hi - prev_c), abs(lo - prev_c)))
    value = sum(trs) / n
    return value if math.isfinite(value) and value > 0 else 0.0


def structural_objective(
    setup: str,
    bars_15m: list[dict[str, Any]],
    bars_1h: list[dict[str, Any]],
    bars_4h: list[dict[str, Any]],
) -> float:
    """Setup-specific structural level the move is expected to reach.

    HTF_TREND_PULLBACK: prior 4h swing high (last 6 closed 4h bars).
    BREAKOUT_CONTINUATION: measured move (the broken range of the 20 closed bars
        before the signal bar, projected above its high).
    RANGE_BOUNCE: opposing range boundary (20-bar 15m high).
    VWAP_REVERSION / EXHAUSTION_MR: 1h mean (last 20 closed 1h closes).
    """
    try:
        if setup == "HTF_TREND_PULLBACK":
            return max(float(b["high"]) for b in bars_4h[-6:]) if bars_4h else 0.0
        if setup == "BREAKOUT_CONTINUATION":
            prior = bars_15m[-21:-1]
            if len(prior) < 20:
                return 0.0
            prior_high = max(float(b["high"]) for b in prior)
            prior_low = min(float(b["low"]) for b in prior)
            return prior_high + (prior_high - prior_low)
        w15 = bars_15m[-20:]
        if not w15:
            return 0.0
        high20 = max(float(b["high"]) for b in w15)
        if setup == "RANGE_BOUNCE":
            return high20
        w1h = bars_1h[-20:]
        return sum(float(b["close"]) for b in w1h) / len(w1h) if w1h else 0.0
    except (KeyError, TypeError, ValueError):
        return 0.0


def objective_level(
    setup: str,
    entry_price: float,
    atr_1h: float,
    structural: float,
    *,
    atr_mult: float = 1.0,
    structural_emphasis: float = 1.0,
) -> float:
    """Objective = max(structural level, entry + k x 1h ATR). Never a fixed percentage.

    ``atr_mult`` and ``structural_emphasis`` are bounded adaptive calibrations.
    Defaults of 1 leave the structure-runner contract unchanged.
    """
    k = _OBJECTIVE_ATR_FLOOR.get(str(setup or ""), _DEFAULT_OBJECTIVE_ATR_FLOOR)
    k *= max(0.75, min(1.35, float(atr_mult or 1.0)))
    floor = entry_price + k * max(0.0, atr_1h)
    level = float(structural or 0.0)
    if level > entry_price > 0:
        emphasis = max(0.85, min(1.25, float(structural_emphasis or 1.0)))
        level = entry_price + (level - entry_price) * emphasis
    return max(level, floor)


def move_potential(
    setup: str,
    ref_price: float,
    atr_1h: float,
    structural: float,
    *,
    atr_mult: float = 1.0,
    structural_emphasis: float = 1.0,
) -> float:
    """Expected move to the objective in 1h-ATR units. Ranking telemetry only, never a gate."""
    if ref_price <= 0 or atr_1h <= 0:
        return 0.0
    level = objective_level(
        setup,
        ref_price,
        atr_1h,
        structural,
        atr_mult=atr_mult,
        structural_emphasis=structural_emphasis,
    )
    return (level - ref_price) / atr_1h


def runner_stop(
    *,
    entry_price: float,
    highest_price: float,
    atr_1h: float,
    objective: float,
    estimated_roundtrip_cost: float,
    activation_mult: float = 1.0,
    trail_mult: float = 1.0,
    tighten_mult: float = 1.0,
) -> dict[str, Any]:
    """Current ratchet state. ``stop`` is 0.0 until the trail itself locks profit.

    A favourable move that arms the activation distance is not enough. The
    computed trail (high-water minus the adaptive ATR trail) must sit strictly
    above break-even after executable round-trip cost. A trail that is still
    at or below that level is not replaced with break-even; the position stays
    under structural protection. Once live, the stop is the computed trail and
    only moves up with the high-water mark or a tighter post-objective trail.
    """
    hwm = max(float(highest_price or 0.0), float(entry_price or 0.0))
    activation = RUNNER_ACTIVATION_ATR_1H * max(0.80, min(1.25, float(activation_mult or 1.0)))
    meaningful = atr_1h > 0 and entry_price > 0 and (hwm - entry_price) >= activation * atr_1h
    objective_reached = objective > 0 and hwm >= objective
    if not meaningful:
        return {"activated": False, "objective_reached": objective_reached, "stop": 0.0, "trail_atr_1h": 0.0}
    if objective_reached:
        trail = RUNNER_TIGHT_TRAIL_ATR_1H * max(0.75, min(1.15, float(tighten_mult or 1.0)))
    else:
        trail = RUNNER_TRAIL_ATR_1H * max(0.80, min(1.20, float(trail_mult or 1.0)))
    computed_trail = hwm - trail * atr_1h
    breakeven_after_cost = entry_price * (1.0 + max(0.0, estimated_roundtrip_cost))
    if computed_trail <= breakeven_after_cost:
        return {"activated": False, "objective_reached": objective_reached, "stop": 0.0, "trail_atr_1h": 0.0}
    return {
        "activated": True,
        "objective_reached": objective_reached,
        "stop": computed_trail,
        "trail_atr_1h": trail,
    }
