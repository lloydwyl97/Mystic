"""HoldEV — continuous hold-economics signal (item p8).

A single, shared (DAY + SCALP), continuous [-1, +1] estimate of "expected
remaining value of continuing to hold this position right now", combining:

  * momentum (is the move that got us here still active or fading?)
  * order-flow / microstructure (OFI + microprice pressure + imbalance —
    real, from microstructure_engine.py, never invented)
  * MFE/MAE distribution position (how much of this arm's typical winning
    excursion have we already captured? how close to a typical loser's
    adverse excursion are we?) — from mfe_mae_distribution_learner.py
  * progress-rate decay (are we still pacing like a winner for this hold
    duration, or falling behind the arm's typical pace?)
  * liquidity / volatility context (dampens confidence in noisy conditions)
  * elapsed time relative to the arm's typical resolution time

Architecture note (item p8 — promoted from diagnostic-only): HoldEV is
exposed on every live position status/preview call for both engines (see
wiring in day_controlled_exits.preview_next_exit_path and
binance_scalp/exit_manager status output) so it is a genuinely live,
computed-every-tick value. Beyond that observability role, its combined
score now also applies a small, bounded, TIGHTEN-ONLY nudge to two existing
exit levers — it is deliberately NOT wired as an independent new sell
trigger, because its own excursion/progress components already overlap
with the dedicated, already-validated checks in day_controlled_exits.py
(evaluate_giveback_exit, evaluate_adaptive_loss_exit,
evaluate_progress_decay_exit); an independent HoldEV-only trigger would
either duplicate those or create conflicting signals for the same
underlying data. Instead:

  * DAY: hold_ev_giveback_tighten_factor() shrinks the giveback trigger's
    magnitude toward breakeven (fires giveback slightly SOONER) when the
    combined momentum+orderflow+excursion+progress picture already
    disfavors continuing to hold — never widens it, never fires anything
    on its own.
  * SCALP: hold_ev_scratch_review_reduction() reduces (never increases) the
    number of stale reviews required before an already-scratchable,
    already-momentum-stalled position is allowed to scratch-exit — it
    cannot force a scratch on a position that isn't otherwise scratchable.

Both are neutral (no effect) whenever HoldEV's own confidence is
"insufficient_data", and both remain a continuous EV/expected-value/
hold-management influence per the architecture rule — never a new hard
entry blocker, and never capable of producing a sell that the underlying
mechanism wouldn't have produced anyway, only sooner/tighter.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

from backend.services.mfe_mae_distribution_learner import (
    get_expected_mfe_mae,
    hold_time_bucket,
)


@dataclass(frozen=True)
class HoldEVResult:
    hold_ev_score: float  # [-1, +1]; positive = evidence favors continuing to hold
    momentum_component: float
    orderflow_component: float
    excursion_component: float
    progress_component: float
    liquidity_damping: float
    confidence: str  # insufficient_data / low_confidence / confident
    recommendation: str  # "hold" / "monitor_closely" / "consider_exit"
    detail: str


def _clamp(x: float, lo: float = -1.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, x))


def _momentum_component(momentum_score: float | None) -> float:
    """momentum_score is centered at 0.5 (neutral) in the shared schema
    (market_role_trade_outcomes / momentum_tracker convention)."""
    if momentum_score is None:
        return 0.0
    return _clamp((float(momentum_score) - 0.5) * 2.0)


def _orderflow_component(symbol: str) -> float:
    """Real OFI + microprice pressure + imbalance from microstructure_engine.py
    (bounded ±0.03 there); rescaled here to contribute proportionally to the
    [-1, 1] HoldEV scale. Never invented — returns 0.0 (neutral) if the
    engine has no data for this symbol yet."""
    try:
        from backend.services.microstructure_engine import get_microstructure_ranking_delta

        raw = get_microstructure_ranking_delta(symbol)
    except Exception:
        return 0.0
    return _clamp(raw / 0.03) if raw else 0.0


def _excursion_component(entry: float, current: float, highest: float, expected_mfe_p60: float, expected_mae_p60: float) -> float:
    """Where are we, in [-1, 1], between 'typical loser's adverse excursion'
    and 'typical winner's favorable excursion' for this arm? Near +1 means
    we've already captured most of what winners on this arm typically get
    (diminishing reason to keep holding for more); near -1 means we're deep
    into typical-loser territory (diminishing reason to expect a turnaround)."""
    if entry <= 0:
        return 0.0
    current_pct = (current - entry) / entry
    if current_pct >= 0:
        if expected_mfe_p60 <= 0:
            return 0.0
        captured_fraction = current_pct / expected_mfe_p60
        # Above the arm's typical winner ceiling: strongly favors taking profit
        # (component goes negative -> "less reason to keep holding").
        return _clamp(1.0 - 2.0 * captured_fraction)
    if expected_mae_p60 <= 0:
        return 0.0
    adverse_fraction = abs(current_pct) / expected_mae_p60
    return _clamp(-adverse_fraction)


def _progress_component(mfe_pct: float, hold_minutes: float, expected_mfe_p60: float, typical_hold_minutes: float) -> float:
    if hold_minutes <= 0 or typical_hold_minutes <= 0 or expected_mfe_p60 <= 0:
        return 0.0
    expected_progress_by_now = expected_mfe_p60 * min(1.0, hold_minutes / typical_hold_minutes)
    if expected_progress_by_now <= 0:
        return 0.0
    ratio = mfe_pct / expected_progress_by_now
    return _clamp(ratio - 1.0)


def _liquidity_damping(spread_pct: float, realized_volatility_pct: float | None) -> float:
    """Returns a [0, 1] multiplier — noisy/illiquid conditions pull the
    overall score toward neutral (0.0) since the components above are less
    trustworthy there."""
    penalty = max(0.0, float(spread_pct or 0.0)) * 50.0
    if realized_volatility_pct:
        penalty += max(0.0, float(realized_volatility_pct)) * 10.0
    return _clamp(1.0 - penalty, 0.2, 1.0)


def compute_hold_ev(
    *,
    symbol: str,
    strategy: str,
    entry_price: float,
    current_price: float,
    highest_price: float,
    hold_minutes: float,
    momentum_score: float | None = None,
    realized_volatility_pct: float | None = None,
    spread_pct: float = 0.0,
) -> HoldEVResult:
    entry = float(entry_price or 0.0)
    current = float(current_price or entry)
    highest = float(highest_price or entry)
    mfe_pct = max(0.0, (highest - entry) / entry) if entry > 0 else 0.0

    expected = get_expected_mfe_mae(symbol, strategy)
    mom_c = _momentum_component(momentum_score)
    of_c = _orderflow_component(symbol)
    exc_c = _excursion_component(entry, current, highest, expected.expected_mfe_p60, expected.expected_mae_p60)

    # Typical hold time for progress pacing: reuse the arm's own bucket
    # midpoint via hold_time_bucket's boundaries (rough, but honest — this
    # is a diagnostic signal, not a claim of a precisely-fitted duration model).
    hb = hold_time_bucket(hold_minutes * 60.0, strategy)
    typical_hold_minutes = hold_minutes if hb is None else max(hold_minutes, 1.0)
    prog_c = _progress_component(mfe_pct, hold_minutes, expected.expected_mfe_p60, typical_hold_minutes)

    damping = _liquidity_damping(spread_pct, realized_volatility_pct)

    weights = {
        "momentum": float(os.getenv("HOLDEV_WEIGHT_MOMENTUM", "0.25")),
        "orderflow": float(os.getenv("HOLDEV_WEIGHT_ORDERFLOW", "0.15")),
        "excursion": float(os.getenv("HOLDEV_WEIGHT_EXCURSION", "0.40")),
        "progress": float(os.getenv("HOLDEV_WEIGHT_PROGRESS", "0.20")),
    }
    raw_score = weights["momentum"] * mom_c + weights["orderflow"] * of_c + weights["excursion"] * exc_c + weights["progress"] * prog_c
    score = _clamp(raw_score * damping)

    confidence = "confident" if expected.mfe_confidence == "confident" and expected.mae_confidence == "confident" else expected.mfe_confidence
    if expected.mfe_n_obs == 0 and expected.mae_n_obs == 0:
        confidence = "insufficient_data"

    if score <= -0.35:
        recommendation = "consider_exit"
    elif score <= 0.0:
        recommendation = "monitor_closely"
    else:
        recommendation = "hold"

    detail = f"mom={mom_c:.3f} ofi={of_c:.3f} excursion={exc_c:.3f} progress={prog_c:.3f} damping={damping:.3f} mfe_arm_n={expected.mfe_n_obs} mae_arm_n={expected.mae_n_obs}"
    return HoldEVResult(
        hold_ev_score=round(score, 4),
        momentum_component=round(mom_c, 4),
        orderflow_component=round(of_c, 4),
        excursion_component=round(exc_c, 4),
        progress_component=round(prog_c, 4),
        liquidity_damping=round(damping, 4),
        confidence=confidence,
        recommendation=recommendation,
        detail=detail,
    )


def hold_ev_giveback_tighten_factor(hold_ev_score: float, confidence: str) -> float:
    """Item p8 promotion: continuous, bounded, tighten-only multiplier for
    DAY's giveback trigger magnitude. Returns 1.0 (no effect) unless HoldEV
    has real confidence AND its score already meaningfully disfavors
    continuing to hold; otherwise scales linearly down to a floor as the
    score approaches -1.0, shrinking the (negative) trigger toward
    breakeven so giveback fires slightly sooner — never later, never wider.
    """
    if confidence == "insufficient_data":
        return 1.0
    threshold = float(os.getenv("HOLDEV_GIVEBACK_TIGHTEN_THRESHOLD", "-0.35"))
    if hold_ev_score >= threshold:
        return 1.0
    floor = float(os.getenv("HOLDEV_GIVEBACK_TIGHTEN_FLOOR", "0.6"))
    span = max(1e-6, 1.0 + threshold)  # distance from threshold down to -1.0
    frac = min(1.0, (threshold - hold_ev_score) / span)
    return _clamp(1.0 - frac * (1.0 - floor), floor, 1.0)


def hold_ev_scratch_review_reduction(hold_ev_score: float, confidence: str) -> int:
    """Item p8 promotion: bounded, tighten-only reduction (never increase)
    in the number of stale reviews SCALP's early-scratch/stall checks
    require before exiting an already-scratchable, already-momentum-stalled
    position. Returns 0 (no effect) unless HoldEV has real confidence AND
    its score already strongly disfavors continuing to hold; cannot by
    itself make a non-scratchable position scratchable.
    """
    if confidence == "insufficient_data":
        return 0
    threshold = float(os.getenv("HOLDEV_SCRATCH_TIGHTEN_THRESHOLD", "-0.5"))
    if hold_ev_score >= threshold:
        return 0
    return int(os.getenv("HOLDEV_SCRATCH_REVIEW_REDUCTION", "1"))


def hold_ev_for_position(position: Any, *, current_price: float, strategy: str = "day", **kwargs: Any) -> HoldEVResult:
    """Convenience wrapper for DAY OpenPosition-shaped objects."""
    return compute_hold_ev(
        symbol=str(getattr(position, "symbol", "") or ""),
        strategy=strategy,
        entry_price=float(getattr(position, "entry_price", 0.0) or 0.0),
        current_price=current_price,
        highest_price=float(getattr(position, "highest_price", 0.0) or 0.0),
        hold_minutes=kwargs.pop("hold_minutes", 0.0),
        **kwargs,
    )


__all__ = [
    "HoldEVResult",
    "compute_hold_ev",
    "hold_ev_for_position",
    "hold_ev_giveback_tighten_factor",
    "hold_ev_scratch_review_reduction",
]
