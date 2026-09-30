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
    BREAKOUT_CONTINUATION: measured move (20-bar 15m range projected above its high).
    RANGE_BOUNCE: opposing range boundary (20-bar 15m high).
    VWAP_REVERSION / EXHAUSTION_MR: 1h mean (last 20 closed 1h closes).
    """
    try:
        if setup == "HTF_TREND_PULLBACK":
            return max(float(b["high"]) for b in bars_4h[-6:]) if bars_4h else 0.0
        w15 = bars_15m[-20:]
        if not w15:
            return 0.0
        high20 = max(float(b["high"]) for b in w15)
        low20 = min(float(b["low"]) for b in w15)
        if setup == "BREAKOUT_CONTINUATION":
            return high20 + (high20 - low20)
        if setup == "RANGE_BOUNCE":
            return high20
        w1h = bars_1h[-20:]
        return sum(float(b["close"]) for b in w1h) / len(w1h) if w1h else 0.0
    except (KeyError, TypeError, ValueError):
        return 0.0


def objective_level(setup: str, entry_price: float, atr_1h: float, structural: float) -> float:
    """Objective = max(structural level, entry + k x 1h ATR). Never a fixed percentage."""
    k = _OBJECTIVE_ATR_FLOOR.get(str(setup or ""), _DEFAULT_OBJECTIVE_ATR_FLOOR)
    floor = entry_price + k * max(0.0, atr_1h)
    return max(float(structural or 0.0), floor)


def move_potential(setup: str, ref_price: float, atr_1h: float, structural: float) -> float:
    """Expected move to the objective in 1h-ATR units. Ranking telemetry only, never a gate."""
    if ref_price <= 0 or atr_1h <= 0:
        return 0.0
    return (objective_level(setup, ref_price, atr_1h, structural) - ref_price) / atr_1h


def runner_stop(
    *,
    entry_price: float,
    highest_price: float,
    atr_1h: float,
    objective: float,
    estimated_roundtrip_cost: float,
) -> dict[str, Any]:
    """Current ratchet state. ``stop`` is 0.0 until the trade has proven itself."""
    hwm = max(float(highest_price or 0.0), float(entry_price or 0.0))
    activated = atr_1h > 0 and entry_price > 0 and (hwm - entry_price) >= RUNNER_ACTIVATION_ATR_1H * atr_1h
    objective_reached = objective > 0 and hwm >= objective
    if not activated:
        return {"activated": False, "objective_reached": objective_reached, "stop": 0.0, "trail_atr_1h": 0.0}
    trail_mult = RUNNER_TIGHT_TRAIL_ATR_1H if objective_reached else RUNNER_TRAIL_ATR_1H
    break_even = entry_price * (1.0 + max(0.0, estimated_roundtrip_cost))
    stop = max(break_even, hwm - trail_mult * atr_1h)
    return {"activated": True, "objective_reached": objective_reached, "stop": stop, "trail_atr_1h": trail_mult}
