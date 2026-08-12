"""EV-scaled SCALP position sizing — replaces the flat notional-cap-for-every-trade behavior.

Architecture v2 (2026-08-11): Mystic is a ranking/trading engine, not a
permission bot. Once a candidate clears every *mechanical* safety hard_block
(stale data, bad spread/impact, no net edge, duplicate position, exposure
cap) in ``scalp_candidate_ranking.py``, it is executable. Opinion-style
evidence — a strategy's own ``sig.passed``, MTF trend disagreement, regime
mismatch, symbol-level historical negative-EV, or a weak arm's evidence — no
longer blocks entry. It must instead show up as a SMALLER position, never a
refusal.

``compute_scalp_position_size`` returns a notional between a practical floor
and the existing exposure ceiling (``config.notional_cap_for_symbol`` /
``max_notional_paper`` / free cash — all unchanged, still mechanical caps),
scaled down by every piece of negative evidence collected during ranking:

    notional = base_cap
               * confidence_factor      (genuine pass vs soft-rank vs regime/MTF conflict)
               * arm_penalty_mult       (per (symbol, setup, regime) historical EV, from arm_blocker)
               * mtf_penalty_mult       (multi-timeframe trend disagreement)
               * symbol_stall_penalty   (symbol-level historical negative-EV, e.g. ETHUSDT/XRPUSDT)
               * volatility_adjustment  (inverse realized-vol — choppier symbol trades smaller)
               * liquidity_adjustment   (wider spread/impact at entry trades smaller)

This module never raises above the mechanical ceiling and never returns
below the caller-supplied minimum viable notional — it only decides *where
in that range* a given candidate's evidence puts it. It cannot cause a trade
to be skipped; ``INSUFFICIENT_CASH``/``MAX_OPEN_POSITIONS`` checks in
``paper_engine.py`` remain the only capital-availability gates.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass


@dataclass(frozen=True)
class SizingResult:
    notional: float
    confidence_factor: float
    volatility_adjustment: float
    liquidity_adjustment: float
    combined_multiplier: float
    reasoning: str


def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def _min_sizing_multiplier() -> float:
    # Floor on the combined multiplier so a weak-but-executable candidate
    # still gets a real (if small) position rather than a token amount that
    # can't clear exchange lot/notional minimums.
    return float(os.getenv("SCALP_DYNAMIC_SIZE_MIN_MULT", "0.15"))


def _confidence_factor(
    *,
    strategy_passed: bool,
    regime_mismatch: bool,
    symbol_stall_risk: bool,
) -> float:
    """Confidence multiplier from the categorical evidence gathered while ranking.

    A genuine strategy pass with no conflicting evidence sizes at 1.0. Every
    additional piece of opinion-evidence against the candidate compounds a
    discount — this is where "ranking not gating" actually protects capital.
    """
    base = 1.0 if strategy_passed else float(os.getenv("SCALP_SOFT_RANK_SIZE_MULT", "0.35"))
    if regime_mismatch:
        base *= float(os.getenv("SCALP_REGIME_MISMATCH_SIZE_MULT", "0.70"))
    if symbol_stall_risk:
        base *= float(os.getenv("SCALP_SYMBOL_STALL_RISK_SIZE_MULT", "0.45"))
    return _clamp(base, 0.05, 1.0)


def _volatility_adjustment(realized_volatility_pct: float | None) -> float:
    """Inverse-volatility sizing: 1 / (1 + k * realized_vol).

    ``realized_volatility_pct`` comes from ``MomentumDiagnostics`` (already
    computed live per symbol every tick — no new data dependency). None or 0
    means insufficient history; treat as neutral (1.0) rather than
    penalizing on missing data.
    """
    if realized_volatility_pct is None or realized_volatility_pct <= 0:
        return 1.0
    k = float(os.getenv("SCALP_VOL_SIZE_SENSITIVITY", "40.0"))
    return _clamp(1.0 / (1.0 + k * float(realized_volatility_pct)), 0.25, 1.0)


def _liquidity_adjustment(spread_pct: float, impact_pct: float) -> float:
    """Execution-cost-derived liquidity multiplier from the same spread/impact
    figures already computed for the mechanical net-edge check — thinner
    top-of-book liquidity sizes smaller even when it wasn't thin enough to
    hard-block."""
    cost = max(0.0, float(spread_pct or 0.0)) + max(0.0, float(impact_pct or 0.0))
    scale = float(os.getenv("SCALP_LIQUIDITY_SIZE_SCALE", "0.004"))  # 0.4% combined cost -> ~0.5x
    return _clamp(math.exp(-cost / scale), 0.30, 1.0)


def compute_scalp_position_size(
    *,
    base_cap: float,
    free_cash: float,
    min_notional: float = 5.0,
    strategy_passed: bool,
    arm_penalty_mult: float = 1.0,
    mtf_penalty_mult: float = 1.0,
    regime_mismatch: bool = False,
    symbol_stall_risk: bool = False,
    spread_pct: float = 0.0,
    impact_pct: float = 0.0,
    realized_volatility_pct: float | None = None,
    calibration_mult: float = 1.0,
) -> SizingResult:
    """Compute the EV-scaled notional for one SCALP entry.

    ``base_cap`` is the existing mechanical ceiling (already
    ``min(config.notional_cap_for_symbol(sym), free_cash)`` from the caller)
    — this function only scales *down* from it, never up.

    ``calibration_mult`` (item p12) is an additional continuous dampener from
    ``ai_calibration_tracker.calibration_confidence_multiplier`` — < 1.0 only
    when this symbol's model confidence has been measured (Brier/ECE) to be
    poorly calibrated recently. Defaults to 1.0 (neutral) so callers that
    don't pass it are unaffected.
    """
    confidence_factor = _confidence_factor(
        strategy_passed=strategy_passed,
        regime_mismatch=regime_mismatch,
        symbol_stall_risk=symbol_stall_risk,
    )
    vol_adj = _volatility_adjustment(realized_volatility_pct)
    liq_adj = _liquidity_adjustment(spread_pct, impact_pct)
    cal_adj = _clamp(float(calibration_mult), 0.05, 1.0)

    combined = confidence_factor * _clamp(arm_penalty_mult, 0.05, 1.0) * _clamp(mtf_penalty_mult, 0.05, 1.0) * vol_adj * liq_adj * cal_adj
    combined = max(combined, _min_sizing_multiplier())

    raw_notional = float(base_cap) * combined
    notional = _clamp(raw_notional, min(min_notional, free_cash), min(float(base_cap), float(free_cash)))
    notional = max(0.0, round(notional, 2))

    reasoning = (
        f"conf={confidence_factor:.3f} arm={arm_penalty_mult:.3f} mtf={mtf_penalty_mult:.3f} "
        f"vol_adj={vol_adj:.3f} liq_adj={liq_adj:.3f} cal_adj={cal_adj:.3f} combined={combined:.3f} "
        f"base_cap={base_cap:.2f} -> notional={notional:.2f}"
    )
    return SizingResult(
        notional=notional,
        confidence_factor=confidence_factor,
        volatility_adjustment=vol_adj,
        liquidity_adjustment=liq_adj,
        combined_multiplier=combined,
        reasoning=reasoning,
    )
