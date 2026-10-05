"""DAY V2 same-bar candidate ranking.

Orders the candidates that fired on one closed 15m bar so the strongest is
funded first when the DAY sleeve cannot fund all of them. Ranking decides
order only; it never removes a candidate.

Order, highest first:
  1. expected net edge after costs (``day_decision``): the candidate's
     hierarchically pooled current-version evidence, realized closes and
     lifecycle replays of unfilled qualified candidates. Cold, it is 0;
  2. executable net edge to the setup objective: (objective - ask) / ask minus
     the estimated round-trip cost, where objective is the same level the
     structure runner manages to (max(structural level, ask + k x 1h ATR));
  3. move potential in 1h-ATR units (the same distance, volatility-normalised);
  4. DAY universe order (deterministic tie-break).

The objective distance is a target, not evidence of edge, so it only orders
candidates whose expected net edge is equal (e.g. all cold).
"""

from __future__ import annotations

from typing import Any

from backend.services.day_v2.winner_contract import move_potential, objective_level


def executable_objective_edge(
    signal: Any,
    ask_price: float,
    roundtrip_cost: float,
    *,
    atr_mult: float = 1.0,
    structural_emphasis: float = 1.0,
) -> float:
    ask = float(ask_price or 0.0)
    if ask <= 0:
        return 0.0
    atr_1h = float(getattr(signal, "atr_1h", 0.0) or 0.0)
    structural = float(getattr(signal, "objective_structural", 0.0) or 0.0)
    objective = objective_level(
        str(signal.setup),
        ask,
        atr_1h,
        structural,
        atr_mult=atr_mult,
        structural_emphasis=structural_emphasis,
    )
    return (objective - ask) / ask - max(0.0, float(roundtrip_cost or 0.0))


def rank_components(
    signal: Any,
    ask_price: float,
    roundtrip_cost: float,
    *,
    atr_mult: float = 1.0,
    structural_emphasis: float = 1.0,
) -> dict[str, float]:
    return {
        "executable_objective_edge": round(
            executable_objective_edge(
                signal,
                ask_price,
                roundtrip_cost,
                atr_mult=atr_mult,
                structural_emphasis=structural_emphasis,
            ),
            8,
        ),
        "move_potential_atr_1h": round(
            move_potential(
                str(signal.setup),
                float(ask_price or 0.0),
                float(getattr(signal, "atr_1h", 0.0) or 0.0),
                float(getattr(signal, "objective_structural", 0.0) or 0.0),
                atr_mult=atr_mult,
                structural_emphasis=structural_emphasis,
            ),
            6,
        ),
        "objective_atr_mult": float(atr_mult),
        "structural_emphasis": float(structural_emphasis),
    }


def rank_day_candidates(candidates: list[dict[str, Any]], universe: list[str] | tuple[str, ...], roundtrip_cost: float) -> list[dict[str, Any]]:
    """Return every candidate, highest rank first. Adaptive state reorders; it never removes.

    A candidate may carry ``adaptive`` from ``day_decision``; its expected net
    edge is the score. Without it every score is 0 and the structural edge
    orders the bar.
    """
    order = {str(sym).upper(): i for i, sym in enumerate(universe)}
    for cand in candidates:
        adaptive = cand.get("adaptive") or {}
        atr_mult = float(adaptive.get("objective_atr_mult") or 1.0)
        emphasis = float(adaptive.get("structural_emphasis") or 1.0)
        cand["rank"] = rank_components(
            cand["signal"],
            cand["ask_price"],
            roundtrip_cost,
            atr_mult=atr_mult,
            structural_emphasis=emphasis,
        )
        cand["rank"]["expected_move"] = float(adaptive.get("expected_move") or 0.0)
        cand["rank"]["expected_net"] = float(adaptive.get("expected_net") or 0.0)
        cand["rank"]["uncertainty"] = float(adaptive.get("uncertainty") or 0.0)
        cand["rank"]["confidence"] = float(adaptive.get("confidence") or 0.0)
        cand["rank"]["size_mult"] = float(adaptive.get("size_mult") or 1.0)
        cand["rank"]["score"] = cand["rank"]["expected_net"]
    ranked = sorted(
        candidates,
        key=lambda c: (
            -c["rank"]["score"],
            -c["rank"]["executable_objective_edge"],
            -c["rank"]["move_potential_atr_1h"],
            order.get(str(c["symbol"]).upper(), len(order)),
        ),
    )
    for position, cand in enumerate(ranked, start=1):
        cand["rank"]["position"] = position
        cand["rank"]["of"] = len(ranked)
    return ranked


def clamp_to_sleeve(quantity: float, ask_price: float, remaining_budget: float) -> float:
    """Largest quantity <= ``quantity`` whose notional fits the remaining DAY sleeve."""
    qty = max(0.0, float(quantity or 0.0))
    ask = float(ask_price or 0.0)
    if ask <= 0:
        return 0.0
    # Stay a hair under the budget so float rounding never trips ENGINE_BUDGET_EXCEEDED.
    cap = max(0.0, float(remaining_budget or 0.0)) / ask * (1.0 - 1e-6)
    return min(qty, cap)


__all__ = ["clamp_to_sleeve", "executable_objective_edge", "rank_components", "rank_day_candidates"]
