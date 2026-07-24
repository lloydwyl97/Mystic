"""Rank scalp setups: soft penalties for setup misses, hard blocks for safety only."""

from __future__ import annotations

import contextlib
import os
from dataclasses import dataclass, field
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
    # Evidence review (repair-all Phase 8): entry rank score at the time of
    # buy showed no meaningful separation between winners and losers in 184
    # matched SCALP paper trades (NET_PROFIT_TARGET avg entry score 1.48 vs
    # EARLY_SCRATCH_EXIT avg 1.47, MOMENTUM_FAILED_EXIT avg 1.43) — score does
    # not discriminate outcome in the observed population, and a 1.45->1.35
    # floor trim had never actually been deployed (no "after" data exists to
    # justify it). Reverted to the last measured-good floor rather than
    # keeping an untested change. Override via SCALP_MIN_TRADEABLE_SCORE.
    return float(os.getenv("SCALP_MIN_TRADEABLE_SCORE", "1.45"))


def _rank_tie_margin() -> float:
    return float(os.getenv("SCALP_RANK_TIE_MARGIN", "0.06"))


def _min_confident_rank() -> float:
    # See _min_tradeable_score: reverted 1.45->1.55 for the same reason (no
    # measured evidence the trimmed floor improves net expectancy). Override
    # via SCALP_MIN_CONFIDENT_RANK.
    return float(os.getenv("SCALP_MIN_CONFIDENT_RANK", "1.55"))


def _regime_mismatch_mult(setup_name: str, regime: str) -> float:
    native = STRATEGY_NATIVE_REGIMES.get(setup_name, frozenset())
    if regime in native:
        return 1.0
    return float(os.getenv("SCALP_REGIME_MISMATCH_MULT", "0.82"))


def _symbol_stall_risk_blocklist() -> frozenset[str]:
    """Evidence (139 genuine-pass trades across 2 independently-running hosts
    since the 07-12 entry_eligible fix): no entry-time feature (rank_score,
    confidence, spread_pct, impact_pct, wick_rejection_pct, expected_move_pct)
    discriminates winners from losers on this population (correlations all
    <0.15) — but symbol identity does, consistently on both hosts:
      ETHUSDT: 45 trades, ~13% win rate, -$5.95 combined (worst)
      XRPUSDT: 32 trades, ~22% win rate, -$2.55 combined
      SOLUSDT: 37 trades, ~32% win rate, -$1.68 combined
      BTCUSDT: 21 trades, ~38% win rate, +$0.16 combined (only net-positive)
    Override via SCALP_STALL_RISK_SYMBOL_BLOCKLIST (comma-separated) or
    disable via SCALP_STALL_RISK_SYMBOL_GATE_ENABLED=false.
    """
    raw = os.getenv("SCALP_STALL_RISK_SYMBOL_BLOCKLIST", "ETHUSDT,XRPUSDT")
    return frozenset(s.strip().upper() for s in raw.split(",") if s.strip())


def _symbol_stall_risk_gate_enabled() -> bool:
    return str(os.getenv("SCALP_STALL_RISK_SYMBOL_GATE_ENABLED", "true")).strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )


def _require_regime_native() -> bool:
    """When true (default), only regime-native strategies may become entry_eligible."""
    return str(os.getenv("SCALP_REQUIRE_REGIME_NATIVE", "true")).strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )


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
    rank_score: float                      # final score (base + context adjustments)
    entry_eligible: bool
    hard_block: str | None
    regime: str
    regime_native: bool
    soft_reason: str | None = None
    reachability_surplus: float = 0.0
    selection_confidence: str = "normal"
    # Diagnostics (computed during scoring; may be None for hard-blocked or passed cases)
    base_score: float | None = None
    momentum_boost: float | None = None
    reachability_multiplier: float | None = None
    target_gap_pct: float | None = None
    # Market-role context breakdown (live + learned, never a gate)
    raw_rank_score: float = 0.0            # score before context adjustments
    live_context_adjustment: float = 0.0  # bounded ±0.04 from live market-role data
    learned_adjustment: float = 0.0       # bounded ±0.02 from outcome history
    role_sample_count: int = 0
    role_confidence: str = "insufficient_data"


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
    base_score: float | None = None
    mom_boost: float | None = None
    reach_mult_val: float | None = None
    target_gap_val: float | None = None

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
        base_score = _soft_base_score(reason)
        mom_boost = _soft_momentum_boost(ctx)
        rank_score = (base_score + mom_boost) * regime_mult
        hard_block = None

    expected = float(sig.expected_move_pct or 0.0)
    reach_surplus = 0.0
    reach_mult_val = 1.0
    target_gap_val = 0.0
    if expected > 0:
        reach_mult_val, reach_surplus = _reachability_soft_mult(
            ctx.econ,
            spread_pct=ctx.snap.spread_pct,
            impact_pct=impact,
            expected_move_pct=expected,
            soft_entry=not sig.passed,
        )
        rank_score *= reach_mult_val
        target_gap_val = reach_surplus
        reachable = reach_mult_val > 0.5 and reach_surplus >= float(ctx.econ.min_projected_surplus_pct)
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
                base_score=base_score,
                momentum_boost=mom_boost,
                reachability_multiplier=reach_mult_val,
                target_gap_pct=target_gap_val,
            )

    min_score = _min_tradeable_score()
    # Lever-4 fix (evidence: 188/188 observed SCALP entries over 8 days were
    # soft-rank promotions of setups the strategy itself REJECTED — never a
    # genuine sig.passed setup — net -$4.35, 15.3% win rate. Prior review
    # already found rank_score does not discriminate winners from losers for
    # these soft entries. Soft-rejected setups may still rank/score for
    # status/diagnostics display, but must never be promoted to an executable
    # trade. Only a strategy's own confirmed pass_signal() may enter live.
    entry_eligible = sig.passed and rank_score >= min_score and hard_block is None
    soft_reason = None if sig.passed else sig.reject_reason
    if entry_eligible and not native and _require_regime_native():
        entry_eligible = False
        confidence = "regime_mismatch"
        soft_reason = f"REGIME_BLOCKED:{regime}"
    elif not entry_eligible and hard_block is None:
        confidence = "below_min"

    if entry_eligible and _symbol_stall_risk_gate_enabled() and sig.symbol.upper() in _symbol_stall_risk_blocklist():
        entry_eligible = False
        confidence = "symbol_stall_risk_blocked"
        soft_reason = f"SYMBOL_STALL_RISK_GATE:{sig.symbol}"

    # ------------------------------------------------------------------
    # Market-role context adjustment — direct bounded addition to score.
    # Never changes eligibility, never blocks, never adds a gate.
    # live_adj ±0.04  (scaled from ±0.06 DAY range; SCALP scores are higher)
    # learned_adj ±0.02 from outcome history
    # ------------------------------------------------------------------
    raw_rank_score = rank_score
    live_ctx_adj = 0.0
    learned_adj = 0.0
    role_samples = 0
    role_conf_status = "insufficient_data"
    with contextlib.suppress(Exception):
        from backend.services.market_role_intelligence import get_cached_role_context as _gcrc
        from backend.services.market_role_outcome_learner import get_learning_stats as _gls

        _rctx = _gcrc(sig.symbol)
        if _rctx is not None:
            # Scale live delta: SCALP base scores are 1.0–2.0+, so we use a
            # proportional fraction of the ±0.06 DAY delta (target ±0.04 here).
            _raw_delta = _rctx.live_ranking_delta()
            live_ctx_adj = round(max(-0.04, min(0.04, _raw_delta * (0.04 / 0.06))), 5)

        _db = os.getenv("TRADING_DB_PATH", "/home/mystic/mystic/mystic_trading.db")
        _stats = _gls(_db, sig.symbol, "scalp")
        role_samples = _stats.sample_count
        role_conf_status = _stats.confidence_status
        learned_adj = round(max(-0.02, min(0.02, _stats.learned_adjustment)), 5)

    rank_score = round(rank_score + live_ctx_adj + learned_adj, 4)

    return RankedCandidate(
        signal=sig,
        rank_score=rank_score,
        entry_eligible=entry_eligible,
        hard_block=hard_block,
        regime=regime,
        regime_native=native,
        soft_reason=soft_reason,
        reachability_surplus=reach_surplus,
        selection_confidence=confidence,
        base_score=base_score,
        momentum_boost=mom_boost,
        reachability_multiplier=reach_mult_val,
        target_gap_pct=target_gap_val if target_gap_val is not None else reach_surplus,
        raw_rank_score=round(raw_rank_score, 4),
        live_context_adjustment=live_ctx_adj,
        learned_adjustment=learned_adj,
        role_sample_count=role_samples,
        role_confidence=role_conf_status,
    )


def prepare_entry_signal(
    ranked: RankedCandidate,
    ctx: StrategyMarketContext,
) -> ScalpSetupSignal:
    """Return an executable entry signal only for genuine strategy passes.

    Soft-rank promotion (forcing ``passed=True`` on rejects) is permanently
    disabled — ``entry_eligible`` already requires ``sig.passed``, so this
    path must never resurrect rejected setups into paper/live trades.
    """
    del ctx  # reserved for future genuine-pass enrichment; unused by design
    sig = ranked.signal
    if not sig.passed:
        import logging

        logging.getLogger(__name__).warning(
            "SOFT_RANK_PROMOTION_BLOCKED setup=%s symbol=%s soft_reason=%s — refusing entry",
            sig.setup_name,
            sig.symbol,
            ranked.soft_reason,
        )
        return sig
    return sig


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


def _role_ranking_delta(row: dict[str, Any]) -> float:
    """
    Market-role intelligence delta for SCALP tie-breaking.
    Reads from the signal's ctx_role_ranking_delta field (published by ai_market_context).
    Range: ±0.06.  Never used as a gate.
    """
    sym = str(row.get("symbol") or "")
    # Try signal fields first
    sig = row.get("signal")
    sig_delta = None
    if sig is not None:
        with contextlib.suppress(Exception):
            from backend.services.market_role_intelligence import get_role_ranking_delta
            sig_delta = get_role_ranking_delta(sym)
    if sig_delta is not None:
        return max(-0.06, min(0.06, sig_delta))
    return 0.0


def _global_tie_key(row: dict[str, Any]) -> tuple:
    """Secondary sort when rank scores cluster — not spread/BTC order."""
    meta = row.get("rank_meta") or {}
    soft = str(meta.get("soft_reason") or row.get("soft_reason") or "")
    soft_tier = _soft_base_score(soft.split(":")[0] if soft else None)
    intel = row.get("intelligence") or {}
    mem_delta = float(intel.get("memory_rank_delta") or 0)
    # Do not treat win_rate=0.0 as missing (falsy) and fall through to dollar PnL.
    _wr = intel.get("recent_scalp_win_rate")
    if _wr is None:
        win_rate = float(intel.get("same_scalp_setup_today_net_pnl") or 0)
    else:
        try:
            win_rate = float(_wr)
        except (TypeError, ValueError):
            win_rate = 0.0
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
    # Market-role intelligence soft delta (affects tie-breaking only, not eligibility)
    role_delta = _role_ranking_delta(row)
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
        role_delta,
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
