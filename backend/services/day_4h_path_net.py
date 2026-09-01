"""Path-aware DAY ranking value. Not an entry gate.

expected_path_net =
    P(favorable_path) * E[favorable_net_bps]
    - P(4h_break_first) * E[break_loss_bps]

Does not install a minimum distance, regime, VWAP, EMA, or wait blocker.
Does not change production ranking unless a caller explicitly uses the score.
"""

from __future__ import annotations

from typing import Any


def expected_path_net_bps(
    *,
    probability_favorable: float,
    expected_favorable_net_bps: float,
    probability_4h_break_first: float,
    expected_break_loss_bps: float,
) -> float:
    p_fav = max(0.0, min(1.0, float(probability_favorable)))
    p_break = max(0.0, min(1.0, float(probability_4h_break_first)))
    return (p_fav * float(expected_favorable_net_bps)) - (p_break * float(expected_break_loss_bps))


def path_net_from_features(features: dict[str, Any] | None) -> dict[str, float | str | None]:
    """Compute a ranking score from existing approved fields. No hard reject."""
    src = dict(features or {})
    p_buy = float(src.get("p_buy") or src.get("prob_buy") or 0.0)
    ml = float(src.get("ml_score") or 0.0)
    pred_ev = float(src.get("predicted_net_ev_bps") or src.get("selected_net_expected_value_bps") or 0.0)
    dist = float(src.get("distance_to_4h_break_bps") or 0.0)
    vel = float(src.get("velocity_toward_4h_break_bps") or 0.0)
    age = float(src.get("forming_4h_bar_age_min") or 0.0)
    ema = float(src.get("ema_alignment") or 0.0)
    ofi = float(src.get("ofi") or src.get("imbalance") or 0.0)

    # Distance/velocity only scale break-first probability — never a min-distance gate.
    closeness = max(0.0, min(1.0, (12.0 - dist) / 12.0)) if dist >= 0 else 1.0
    toward = max(0.0, min(1.0, vel / 8.0)) if vel > 0 else 0.0
    late_bar = max(0.0, min(1.0, age / 240.0))
    p_break = max(0.05, min(0.95, 0.20 + 0.45 * closeness + 0.25 * toward + 0.10 * late_bar - 0.08 * ema - 0.05 * ofi))
    p_fav = max(0.0, min(1.0, (1.0 - p_break) * max(0.05, p_buy)))
    fav = pred_ev if pred_ev != 0.0 else max(4.0, 12.0 + 20.0 * ml)
    break_loss = max(8.0, 18.0 + max(0.0, 10.0 - dist) * 0.8)
    score = expected_path_net_bps(
        probability_favorable=p_fav,
        expected_favorable_net_bps=fav,
        probability_4h_break_first=p_break,
        expected_break_loss_bps=break_loss,
    )
    return {
        "expected_path_net_bps": round(score, 4),
        "probability_favorable": round(p_fav, 4),
        "expected_favorable_net_bps": round(fav, 4),
        "probability_4h_break_first": round(p_break, 4),
        "expected_break_loss_bps": round(break_loss, 4),
        "ranking_only": 1.0,
    }
