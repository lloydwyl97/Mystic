"""Offline DAY entry-quality scorecard. Never imported by ranking or exits.

Computes grouped-decision analytics after outcomes exist. No threshold from
this module may block or authorize a live trade.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any

COINS = ("BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT")
HOLD = "HOLD"


def _num(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def scorecard_from_labeled_groups(groups: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate opportunity / selection / regret metrics. Analysis only."""
    n = len(groups)
    if n == 0:
        return {"n_groups": 0, "live_feed": False}
    selected_positive = 0
    hold_correct = 0
    negative_entry = 0
    best_coin_selected = 0
    opportunity = 0
    regret_best = 0.0
    regret_hold = 0.0
    pred_vs_real = []
    by_symbol: dict[str, list[float]] = defaultdict(list)
    cost_cover_by_score: dict[str, list[int]] = defaultdict(list)
    for g in groups:
        labels = dict(g.get("labels") or {})
        selected = str(g.get("selected_symbol") or HOLD)
        hold_net = _num((labels.get(HOLD) or {}).get("net_bps")) or 0.0
        coin_nets = {sym: _num((labels.get(sym) or {}).get("net_bps")) for sym in COINS}
        eligible = [sym for sym in COINS if coin_nets.get(sym) is not None]
        best = None
        best_net = hold_net
        for sym in eligible:
            val = coin_nets[sym]
            if val is not None and val > best_net:
                best = sym
                best_net = val
        if best is not None and best_net > 0:
            opportunity += 1
        sel_net = hold_net if selected == HOLD else (coin_nets.get(selected) or 0.0)
        if selected == HOLD:
            if hold_net >= best_net:
                hold_correct += 1
        else:
            if sel_net > 0:
                selected_positive += 1
            else:
                negative_entry += 1
            if selected == best:
                best_coin_selected += 1
            by_symbol[selected].append(sel_net)
        regret_best += best_net - sel_net
        regret_hold += hold_net - sel_net
        pred = _num(g.get("predicted_net_bps"))
        if pred is not None:
            pred_vs_real.append(pred - sel_net)
        score = _num(g.get("selected_final_score"))
        covered = 1 if (labels.get(selected) or {}).get("cost_cover") else 0
        if score is not None:
            bucket = f"{int(score * 10000) // 5 * 5}bps"
            cost_cover_by_score[bucket].append(covered)
    return {
        "n_groups": n,
        "live_feed": False,
        "opportunity_rate": opportunity / n,
        "best_coin_selection_rate": best_coin_selected / max(1, n - hold_correct),
        "selected_positive_rate": selected_positive / max(1, n - sum(1 for g in groups if str(g.get("selected_symbol") or HOLD) == HOLD)),
        "HOLD_accuracy": hold_correct / n,
        "negative_entry_rate": negative_entry / n,
        "regret_vs_best_coin": regret_best / n,
        "regret_vs_HOLD": regret_hold / n,
        "predicted_net_vs_realized_net": (sum(pred_vs_real) / len(pred_vs_real)) if pred_vs_real else None,
        "profit_by_symbol": {k: (sum(v) / len(v) if v else None) for k, v in by_symbol.items()},
        "cost_cover_rate_by_score": {k: (sum(v) / len(v) if v else None) for k, v in cost_cover_by_score.items()},
    }
