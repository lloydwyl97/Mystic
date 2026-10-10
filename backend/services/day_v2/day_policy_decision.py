"""One DAY continuation decision. Live exits and exact-policy episodes both call this.

The function has no order, cash, or slot side effects. The live engine may sell
after it returns. A refused-candidate episode may only record the same result.
"""

from __future__ import annotations

import time
from typing import Any

from backend.services.continuation_surface import state_features
from backend.services.day_v2.live_exit_evaluator import evaluate_day_v2_exit


def decide_day_continuation(
    db_path: str,
    *,
    symbol: str,
    setup: str,
    exit_setup: str = "",
    regime: str,
    entry_price: float,
    mark: float,
    low: float,
    high: float,
    prev_net: float | None,
    age_sec: float,
    atr_at_entry: float,
    structural_anchor: float,
    target_price: float,
    entry_time: float,
    roundtrip_cost: float,
    atr_1h_at_entry: float = 0.0,
    objective_structural: float = 0.0,
    objective_atr_mult: float = 1.0,
    structural_emphasis: float = 1.0,
    runner_activation_mult: float = 1.0,
    runner_trail_mult: float = 1.0,
    runner_tighten_mult: float = 1.0,
    now: float | None = None,
) -> dict[str, Any]:
    """HOLD or EXIT for one DAY position state. ``mark`` is the executable bid."""
    moment = float(now if now is not None else time.time())
    entry = float(entry_price)
    price = float(mark)
    net = (price - entry) / entry - float(roundtrip_cost) if entry > 0 else 0.0
    features = state_features(
        entry=entry,
        mark=price,
        net=net,
        mfe=max(0.0, (float(high) - entry) / entry) if entry else 0.0,
        mae=max(0.0, (entry - float(low)) / entry) if entry else 0.0,
        high_water=max(float(high), price, entry),
        prev_net=prev_net,
        age_sec=max(0.0, float(age_sec)),
    )
    from backend.services.adaptive_learning import continuation_terminal
    from backend.services.continuation_surface import ADVANTAGE_VERSION, advantage_authority, installed_aggregator, surface_trace

    terminal = continuation_terminal(db_path, "DAY_V2", symbol, setup, regime, net, features=features, now=moment)
    trace = None
    if advantage_authority(db_path, "DAY_V2"):
        trace = surface_trace(
            db_path,
            "DAY_V2",
            symbol,
            setup,
            regime,
            features,
            moment,
            how=installed_aggregator(db_path, "DAY_V2"),
        )
    decision = evaluate_day_v2_exit(
        engine_id="DAY_V2",
        entry_price=entry,
        current_price=price,
        bar_low=float(low) if float(low) > 0 else price,
        highest_price=float(high) if float(high) > 0 else entry,
        atr_at_entry=float(atr_at_entry or 0.0),
        structural_anchor=float(structural_anchor or 0.0),
        target_price=float(target_price or 0.0),
        entry_time=float(entry_time or 0.0),
        estimated_roundtrip_cost=float(roundtrip_cost),
        setup=str(exit_setup or setup or ""),
        atr_1h_at_entry=float(atr_1h_at_entry or 0.0),
        objective_structural=float(objective_structural or 0.0),
        objective_atr_mult=float(objective_atr_mult or 1.0),
        structural_emphasis=float(structural_emphasis or 1.0),
        runner_activation_mult=float(runner_activation_mult or 1.0),
        runner_trail_mult=float(runner_trail_mult or 1.0),
        runner_tighten_mult=float(runner_tighten_mult or 1.0),
        expected_terminal_net=terminal,
        now=moment,
    )
    advantage = None if terminal is None else float(terminal) - float(net)
    action = "exit" if decision and str(decision.get("action") or "") == "sell" else "hold"
    reason = str((decision or {}).get("reason") or "")
    return {
        "features": features,
        "net": float(net),
        "terminal": None if terminal is None else float(terminal),
        "advantage": advantage,
        "decision": decision,
        "action": action,
        "reason": reason,
        "at": moment,
        "trace": trace,
        "continuation_version": ADVANTAGE_VERSION,
    }
