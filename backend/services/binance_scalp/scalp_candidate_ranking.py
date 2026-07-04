"""Rank scalp setups: soft penalties for setup misses, hard blocks for safety only."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

from backend.services.binance_scalp.scalp_regime_classifier import STRATEGY_NATIVE_REGIMES
from backend.services.binance_scalp.strategies.base import ScalpSetupSignal, StrategyMarketContext
from backend.services.binance_scalp.strategies.common import (
    check_spread,
    depth_check,
    target_reachable,
)

# Hard safety — never trade through these.
HARD_REJECT_REASONS: frozenset[str] = frozenset(
    {
        "SPREAD_TOO_WIDE",
        "DEPTH_OR_IMPACT_FAIL",
        "INSUFFICIENT_BARS",
        "STALE_DATA",
        "MOMENTUM_DATA_INSUFFICIENT",
        "TARGET_NOT_REACHABLE",
        "NO_EXECUTABLE_NET_EDGE",
    }
)

# Soft setup misses → partial rank score (failed checks are not tradeable edge).
# Prior report: NO_REJECTION_WICK dominated scratches; NOT_NEAR_SUPPORT had more winners.
SOFT_REJECT_SCORE: dict[str, float] = {
    "NO_REJECTION_WICK": 0.92,
    "NOT_NEAR_SUPPORT": 1.28,
    "NO_PULLBACK_RECOVERY": 1.05,
    "NO_VWAP_EMA_RECLAIM": 0.98,
    "MOMENTUM_NOT_FLIPPED": 0.90,
    "MOMENTUM_NOT_SUSTAINED": 0.85,
    "NO_REVERSAL_CONFIRM": 0.82,
    "NO_FAILED_BREAKOUT": 0.82,
    "NO_BREAKOUT": 0.78,
    "RANGE_TOO_WIDE": 0.55,
    "REGIME_BLOCKED": 0.48,
}


def _min_tradeable_score() -> float:
    return float(os.getenv("SCALP_MIN_TRADEABLE_SCORE", "1.45"))


def _rank_tie_margin() -> float:
    return float(os.getenv("SCALP_RANK_TIE_MARGIN", "0.06"))


def _min_confident_rank() -> float:
    return float(os.getenv("SCALP_MIN_CONFIDENT_RANK", "1.55"))


def _regime_mismatch_mult(setup_name: str, regime: str) -> float:
    native = STRATEGY_NATIVE_REGIMES.get(setup_name, frozenset())
    if regime in native:
        return 1.0
    return float(os.getenv("SCALP_REGIME_MISMATCH_MULT", "0.82"))


def _soft_base_score(reject_reason: str | None) -> float:
    if not reject_reason:
        return 0.0
    key = reject_reason.split(":", 1)[0] if reject_reason.startswith("REGIME_BLOCKED") else reject_reason
    if reject_reason.startswith("REGIME_BLOCKED"):
        return SOFT_REJECT_SCORE.get("REGIME_BLOCKED", 0.48)
    return SOFT_REJECT_SCORE.get(key, 0.72)


def _soft_momentum_boost(ctx: StrategyMarketContext) -> float:
    """Cap soft-entry momentum lift — tiny 15s upticks must not inflate weak setups to tradable."""
    mom = ctx.mom
    raw = max(0.0, float(getattr(mom, "mid_change_15s", 0) or 0) * 120.0)
    if getattr(mom, "momentum_confirmed", False):
        return min(0.16, raw)
    return min(0.05, raw * 0.4)


def _reachability_soft_mult(
    econ: Any,
    *,
    spread_pct: float,
    impact_pct: float,
    expected_move_pct: float,
    soft_entry: bool,
) -> tuple[float, float]:
    """Penalize soft entries whose projected move barely clears break-even."""
    if expected_move_pct <= 0:
        return 1.0, 0.0
    reachable, req = target_reachable(
        econ,
        spread_pct=spread_pct,
        impact_pct=impact_pct,
        expected_move_pct=expected_move_pct,
    )
    surplus = expected_move_pct - req
    if not reachable:
        if soft_entry:
            return 0.45, surplus
        # Passed strategies already verified reachability at signal time.
        return 1.0, surplus
    if not soft_entry:
        return 1.0, surplus
    min_soft = max(float(econ.min_projected_surplus_pct) * 2.5, 0.0008)
    if surplus >= min_soft * 2.0:
        return 1.0, surplus
    if surplus <= 0:
        return 0.5, surplus
    return max(0.55, min(1.0, surplus / min_soft)), surplus


@dataclass(frozen=True)
class RankedCandidate:
    signal: ScalpSetupSignal
    rank_score: float
    entry_eligible: bool
    hard_block: str | None
    regime: str
    regime_native: bool
    soft_reason: str | None = None
    reachability_surplus: float = 0.0
    selection_confidence: str = "normal"


def rank_setup_signal(
    sig: ScalpSetupSignal,
    *,
    regime: str,
    ctx: StrategyMarketContext,
) -> RankedCandidate:
    """Score a strategy signal; soft failures rank, hard failures block."""
    spread_ok, spread_reason = check_spread(ctx.snap, ctx.econ, ctx.config)
    if not spread_ok:
        return RankedCandidate(
            signal=sig,
            rank_score=0.0,
            entry_eligible=False,
            hard_block=spread_reason or "SPREAD_TOO_WIDE",
            regime=regime,
            regime_native=regime in STRATEGY_NATIVE_REGIMES.get(sig.setup_name, frozenset()),
            soft_reason=sig.reject_reason,
            selection_confidence="blocked",
        )

    depth_ok, impact, fill = depth_check(ctx.snap, ctx.notional_usd, ctx.econ)
    if not depth_ok:
        return RankedCandidate(
            signal=sig,
            rank_score=0.0,
            entry_eligible=False,
            hard_block="DEPTH_OR_IMPACT_FAIL",
            regime=regime,
            regime_native=regime in STRATEGY_NATIVE_REGIMES.get(sig.setup_name, frozenset()),
            soft_reason=sig.reject_reason,
            selection_confidence="blocked",
        )

    regime_mult = _regime_mismatch_mult(sig.setup_name, regime)
    native = regime in STRATEGY_NATIVE_REGIMES.get(sig.setup_name, frozenset())
    confidence = "normal"

    if sig.passed:
        rank_score = float(sig.score) * regime_mult
        hard_block = None
    else:
        reason = sig.reject_reason or ""
        if reason in HARD_REJECT_REASONS or reason.startswith("STRATEGY_ERROR"):
            return RankedCandidate(
                signal=sig,
                rank_score=0.0,
                entry_eligible=False,
                hard_block=reason or "HARD_REJECT",
                regime=regime,
                regime_native=native,
                soft_reason=reason,
                selection_confidence="blocked",
            )
        if reason == "TARGET_NOT_REACHABLE":
            return RankedCandidate(
                signal=sig,
                rank_score=0.0,
                entry_eligible=False,
                hard_block="NO_EXECUTABLE_NET_EDGE",
                regime=regime,
                regime_native=native,
                soft_reason=reason,
                selection_confidence="blocked",
            )
        base = _soft_base_score(reason)
        mom_boost = _soft_momentum_boost(ctx)
        rank_score = (base + mom_boost) * regime_mult
        hard_block = None

    expected = float(sig.expected_move_pct or 0.0)
    reach_surplus = 0.0
    if expected > 0:
        reach_mult, reach_surplus = _reachability_soft_mult(
            ctx.econ,
            spread_pct=ctx.snap.spread_pct,
            impact_pct=impact,
            expected_move_pct=expected,
            soft_entry=not sig.passed,
        )
        rank_score *= reach_mult
        reachable = reach_mult > 0.5 and reach_surplus >= float(ctx.econ.min_projected_surplus_pct)
        if not reachable and not sig.passed:
            return RankedCandidate(
                signal=sig,
                rank_score=round(rank_score, 4),
                entry_eligible=False,
                hard_block="NO_EXECUTABLE_NET_EDGE",
                regime=regime,
                regime_native=native,
                soft_reason=sig.reject_reason or "TARGET_NOT_REACHABLE",
                reachability_surplus=reach_surplus,
                selection_confidence="low_reachability",
            )

    min_score = _min_tradeable_score()
    entry_eligible = rank_score >= min_score and hard_block is None
    if not entry_eligible and hard_block is None:
        confidence = "below_min"

    return RankedCandidate(
        signal=sig,
        rank_score=round(rank_score, 4),
        entry_eligible=entry_eligible,
        hard_block=hard_block,
        regime=regime,
        regime_native=native,
        soft_reason=None if sig.passed else sig.reject_reason,
        reachability_surplus=reach_surplus,
        selection_confidence=confidence,
    )


def prepare_entry_signal(
    ranked: RankedCandidate,
    ctx: StrategyMarketContext,
) -> ScalpSetupSignal:
    """Promote a soft-ranked candidate to an executable entry signal."""
    sig = ranked.signal
    if sig.passed:
        return sig

    _, impact, fill = depth_check(ctx.snap, ctx.notional_usd, ctx.econ)
    expected = float(sig.expected_move_pct or 0.0)
    if expected <= 0:
        expected = max(
            ctx.econ.net_profit_target_pct + ctx.econ.entry_edge_buffer_pct,
            ctx.snap.spread_pct + impact + ctx.econ.min_projected_surplus_pct,
        )

    ctx_extra = dict(sig.setup_context or {})
    ctx_extra["soft_rank_entry"] = True
    ctx_extra["soft_reason"] = ranked.soft_reason
    ctx_extra["rank_score"] = ranked.rank_score
    ctx_extra["reachability_surplus"] = ranked.reachability_surplus

    return ScalpSetupSignal(
        symbol=sig.symbol,
        side=sig.side,
        score=ranked.rank_score,
        setup_name=sig.setup_name,
        confidence=max(sig.confidence, min(0.55, ranked.rank_score / 4.0)),
        entry_reason=f"ranked_soft:{ranked.soft_reason or 'setup'} score={ranked.rank_score:.2f}",
        invalidation_reason=sig.invalidation_reason or "soft_rank_invalidation",
        required_target_pct=ctx.econ.net_profit_target_pct,
        expected_move_pct=expected,
        spread_pct=ctx.snap.spread_pct,
        impact_pct=impact,
        depth_sufficient=True,
        limit_buy_price=fill,
        passed=True,
        reject_reason=None,
        setup_context=ctx_extra,
    )


def pick_best_ranked(candidates: list[RankedCandidate]) -> RankedCandidate | None:
    if not candidates:
        return None
    eligible = [c for c in candidates if c.entry_eligible]
    pool = eligible if eligible else candidates
    return max(
        pool,
        key=lambda c: (
            c.rank_score,
            c.reachability_surplus,
            1 if c.regime_native else 0,
            _soft_base_score(c.soft_reason),
            c.signal.confidence,
        ),
    )


def _global_tie_key(row: dict[str, Any]) -> tuple:
    """Secondary sort when rank scores cluster — not spread/BTC order."""
    meta = row.get("rank_meta") or {}
    soft = str(meta.get("soft_reason") or row.get("soft_reason") or "")
    soft_tier = _soft_base_score(soft.split(":")[0] if soft else None)
    intel = row.get("intelligence") or {}
    mem_delta = float(intel.get("memory_rank_delta") or 0)
    win_rate = float(intel.get("recent_scalp_win_rate") or intel.get("same_scalp_setup_today_net_pnl") or 0)
    regime_native = 1 if meta.get("regime_native") else 0
    mom = row.get("mom")
    m15 = float(getattr(mom, "mid_change_15s", 0) or 0) if mom else 0.0
    m30 = float(getattr(mom, "mid_change_30s", 0) or 0) if mom else 0.0
    sig = row.get("signal")
    passed = 1 if getattr(sig, "passed", False) else 0
    reach = float(meta.get("reachability_surplus") or 0)
    sym = str(row.get("symbol") or "")
    # Deprioritize BTC on pure weak ties (tight spread was winning every coin flip).
    sym_penalty = -0.002 if sym == "BTCUSDT" else 0.0
    return (
        passed,
        regime_native,
        soft_tier,
        reach,
        mem_delta,
        win_rate,
        m30,
        m15,
        sym_penalty,
        float(row.get("rank_score") or 0),
    )


def pick_best_global_candidate(rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    """
    Pick one global entry among per-symbol rows.

    Skips weak all-symbol ties (least-bad cluster) unless a candidate clears
    SCALP_MIN_CONFIDENT_RANK with margin over second place.
    """
    eligible = [r for r in rows if r.get("entry_eligible")]
    if not eligible:
        return None

    eligible.sort(key=lambda r: (-float(r.get("rank_score") or 0), *_global_tie_key(r)[::-1]))
    top_score = float(eligible[0].get("rank_score") or 0)
    second_score = float(eligible[1].get("rank_score") or 0) if len(eligible) > 1 else 0.0
    margin = top_score - second_score
    tie_margin = _rank_tie_margin()
    min_confident = _min_confident_rank()

    if top_score < min_confident and margin < tie_margin:
        return None

    if margin < tie_margin and len(eligible) > 1:
        tied = [r for r in eligible if abs(float(r.get("rank_score") or 0) - top_score) <= tie_margin + 1e-9]
        return max(tied, key=_global_tie_key)

    return eligible[0]


__all__ = [
    "HARD_REJECT_REASONS",
    "RankedCandidate",
    "SOFT_REJECT_SCORE",
    "pick_best_global_candidate",
    "pick_best_ranked",
    "prepare_entry_signal",
    "rank_setup_signal",
]
