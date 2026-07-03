"""Rank scalp setups: soft penalties for setup misses, hard blocks for safety only."""

from __future__ import annotations

import os
from dataclasses import dataclass

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

# Soft setup misses → partial rank score (paper ranking engine).
SOFT_REJECT_SCORE: dict[str, float] = {
    "NO_REJECTION_WICK": 1.85,
    "NOT_NEAR_SUPPORT": 1.65,
    "NO_PULLBACK_RECOVERY": 1.35,
    "NO_VWAP_EMA_RECLAIM": 1.25,
    "MOMENTUM_NOT_FLIPPED": 1.15,
    "MOMENTUM_NOT_SUSTAINED": 1.0,
    "NO_REVERSAL_CONFIRM": 0.95,
    "NO_FAILED_BREAKOUT": 0.95,
    "NO_BREAKOUT": 0.9,
    "RANGE_TOO_WIDE": 0.55,
    "REGIME_BLOCKED": 0.5,
}


def _min_tradeable_score() -> float:
    return float(os.getenv("SCALP_MIN_TRADEABLE_SCORE", "1.1"))


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
        return SOFT_REJECT_SCORE.get("REGIME_BLOCKED", 0.5)
    return SOFT_REJECT_SCORE.get(key, 0.75)


@dataclass(frozen=True)
class RankedCandidate:
    signal: ScalpSetupSignal
    rank_score: float
    entry_eligible: bool
    hard_block: str | None
    regime: str
    regime_native: bool
    soft_reason: str | None = None


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
        )

    regime_mult = _regime_mismatch_mult(sig.setup_name, regime)
    native = regime in STRATEGY_NATIVE_REGIMES.get(sig.setup_name, frozenset())

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
            )
        base = _soft_base_score(reason)
        mom_boost = max(0.0, float(getattr(ctx.mom, "mid_change_15s", 0) or 0) * 200.0)
        rank_score = (base + mom_boost) * regime_mult
        hard_block = None

    expected = float(sig.expected_move_pct or 0.0)
    if expected > 0:
        reachable, _ = target_reachable(
            ctx.econ,
            spread_pct=ctx.snap.spread_pct,
            impact_pct=impact,
            expected_move_pct=expected,
        )
        if not reachable:
            return RankedCandidate(
                signal=sig,
                rank_score=rank_score * 0.5,
                entry_eligible=False,
                hard_block="NO_EXECUTABLE_NET_EDGE",
                regime=regime,
                regime_native=native,
                soft_reason=sig.reject_reason or "TARGET_NOT_REACHABLE",
            )

    min_score = _min_tradeable_score()
    entry_eligible = rank_score >= min_score and hard_block is None

    return RankedCandidate(
        signal=sig,
        rank_score=round(rank_score, 4),
        entry_eligible=entry_eligible,
        hard_block=hard_block,
        regime=regime,
        regime_native=native,
        soft_reason=None if sig.passed else sig.reject_reason,
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
    return max(pool, key=lambda c: (c.rank_score, c.signal.confidence, -c.signal.spread_pct))


__all__ = [
    "HARD_REJECT_REASONS",
    "RankedCandidate",
    "pick_best_ranked",
    "prepare_entry_signal",
    "rank_setup_signal",
]
