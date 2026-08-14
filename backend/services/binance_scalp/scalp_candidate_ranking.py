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


def min_tradeable_score() -> float:
    """Public alias of the soft reference tradeability score.

    NOT a permission gate — see architecture note in rank_setup_signal.
    Exposed for scalp_dynamic_sizing.py so the EV-sizing formula uses the
    same reference threshold rather than duplicating the magic default.
    """
    return _min_tradeable_score()


def min_confident_rank() -> float:
    """Public alias — see min_tradeable_score(). Not a gate."""
    return _min_confident_rank()


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
    raw = os.getenv("SCALP_STALL_RISK_SYMBOL_BLOCKLIST", "")
    return frozenset(s.strip().upper() for s in raw.split(",") if s.strip())


def _symbol_stall_risk_gate_enabled() -> bool:
    return str(os.getenv("SCALP_STALL_RISK_SYMBOL_GATE_ENABLED", "false")).strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )


def _require_regime_native() -> bool:
    """When true (default), only regime-native strategies may become entry_eligible."""
    return str(os.getenv("SCALP_REQUIRE_REGIME_NATIVE", "false")).strip().lower() in (
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
    rank_score: float  # final score (base + context adjustments)
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
    raw_rank_score: float = 0.0  # score before context adjustments
    live_context_adjustment: float = 0.0  # bounded ±0.04 from live market-role data
    learned_adjustment: float = 0.0  # bounded ±0.02 from outcome history
    role_sample_count: int = 0
    role_confidence: str = "insufficient_data"
    microstructure_adjustment: float = 0.0  # bounded ±0.03 from real OFI/microprice/imbalance
    # EV inputs for scalp_dynamic_sizing.py (never used to gate — sizing only)
    arm_penalty_mult: float = 1.0
    arm_stats: Any = None
    regime_mismatch: bool = False
    symbol_stall_risk: bool = False


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

    # Adaptive per-arm EV signal: (symbol, setup) combos with a consistently
    # negative historical expectancy over the last 30d of
    # scalp_learning_outcomes are ranking/size inputs, not permission gates
    # (architecture rule: opinion/expectancy evidence must influence ranking
    # and sizing, never become a new hard entry blocker). A heavily negative
    # arm still competes for the global pick — it will simply almost never
    # win against a healthier arm, and scalp_dynamic_sizing.py sizes it near
    # the practical floor even when it does win. Never blocks on insufficient
    # sample count (handled by arm_blocked itself).
    arm_penalty_mult = 1.0
    arm_stats_for_sizing: Any = None
    with contextlib.suppress(Exception):
        from backend.services.binance_scalp.scalp_arm_blocker import arm_blocked

        _arm_blk, _arm_reason, _arm_stats = arm_blocked(ctx.snap.symbol, sig.setup_name)
        arm_stats_for_sizing = _arm_stats
        if _arm_blk:
            # Strong rank penalty (not exclusion) — evidence-based negative EV.
            arm_penalty_mult = float(os.getenv("SCALP_ARM_NEGATIVE_EV_RANK_MULT", "0.20"))

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
        rank_score = float(sig.score) * regime_mult * arm_penalty_mult
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
        rank_score = (base_score + mom_boost) * regime_mult * arm_penalty_mult
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
    # Architecture v2 (2026-08-11, "ranking not gating" rule): the prior
    # "lever-4" fix required sig.passed AND rank_score>=min_score AND
    # hard_block is None as a single boolean permission before ANY candidate
    # could execute. That was itself an opinion/confidence-threshold gate —
    # exactly the pattern the current architecture rule forbids for anything
    # that is not mechanical safety. The original evidence it was fixing
    # (188/188 soft-rank entries, -$4.35, 15.3% WR, rank_score not
    # discriminating outcome) is still real and is NOT ignored — it is now
    # addressed on the sizing side: scalp_dynamic_sizing.py sizes
    # soft-rejected / regime-mismatched / arm-negative-EV / symbol-stall-risk
    # candidates down toward the practical notional floor instead of a
    # full-size bet, so a weak opinion costs little instead of nothing.
    #
    # entry_eligible now means ONLY "no mechanical safety hard_block fired"
    # (spread / depth / stale-data / momentum-data-insufficient / net-edge —
    # all handled above, before this point). rank_score is still fully
    # computed and still drives which of the four symbols wins the global
    # pick (pick_best_global_candidate) — a rejected setup can still be
    # ranked and traded (small), it can never be forbidden by an opinion
    # signal.
    entry_eligible = hard_block is None
    soft_reason = None if sig.passed else sig.reject_reason
    confidence = "genuine_pass" if sig.passed else "soft_rank_ranked"
    if rank_score < min_score:
        confidence = f"{confidence}_below_min_score"  # observability only — does not block

    regime_mismatch = not native and _require_regime_native()
    if regime_mismatch:
        # Opinion signal only now — already reflected in regime_mult above.
        confidence = f"{confidence}_regime_mismatch"
        soft_reason = soft_reason or f"REGIME_MISMATCH_RANKED:{regime}"

    symbol_stall_risk = _symbol_stall_risk_gate_enabled() and sig.symbol.upper() in _symbol_stall_risk_blocklist()
    if symbol_stall_risk:
        # Evidence-based *symbol-level* negative-EV signal (ETHUSDT/XRPUSDT
        # historically worse win rate on this arm-population) — penalize
        # rank/size, never exclude. The four top-4 symbols must all remain
        # ranked and eligible per the architecture rule.
        confidence = f"{confidence}_symbol_stall_risk"
        soft_reason = soft_reason or f"SYMBOL_STALL_RISK_RANKED:{sig.symbol}"
        rank_score = round(rank_score * float(os.getenv("SCALP_SYMBOL_STALL_RISK_RANK_MULT", "0.35")), 4)

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
        from backend.services.market_role_intelligence import fetch_role_ranking_delta_from_redis as _frrd
        from backend.services.market_role_outcome_learner import get_learning_stats as _gls

        # Redis is cross-process (ai_market_context writes; scalp runner reads)
        _raw_delta = _frrd(sig.symbol)
        live_ctx_adj = round(max(-0.04, min(0.04, _raw_delta * (0.04 / 0.06))), 5)

        _db = os.getenv("TRADING_DB_PATH", "/home/mystic/mystic/mystic_trading.db")
        _stats = _gls(_db, sig.symbol, "scalp")
        role_samples = _stats.sample_count
        role_conf_status = _stats.confidence_status
        learned_adj = round(max(-0.02, min(0.02, _stats.learned_adjustment)), 5)
        with contextlib.suppress(Exception):
            from backend.services.trade_learning_writer import consume_setup_outcomes_for_ranking

            _learned = consume_setup_outcomes_for_ranking(
                _db,
                sig.setup_name,
                features={
                    "volatility": getattr(ctx.mom, "realized_volatility_pct", None),
                    "momentum": getattr(ctx.mom, "mid_change_60s", None),
                    "regime": regime,
                    "model_probability": sig.confidence,
                    "market_regime": regime,
                },
            )
            if _learned.get("consumed") and int(_learned.get("n") or 0) >= 8:
                learned_adj = round(
                    max(-0.04, min(0.04, learned_adj + float(_learned.get("rank_delta") or 0.0))),
                    5,
                )

    # Real microstructure engine (OFI + aggressor flow + microprice pressure,
    # short 250ms-30s windows) — feeds this SCALP entry's rank_score only.
    # Never eligibility, never a gate. See microstructure_engine.py.
    micro_adj = 0.0
    with contextlib.suppress(Exception):
        from backend.services.microstructure_engine import get_microstructure_ranking_delta as _gmrd

        micro_adj = round(_gmrd(sig.symbol), 5)

    feature_adj = 0.0
    with contextlib.suppress(Exception):
        feats = (sig.setup_context or {}).get("features") or {}
        if feats:
            from backend.services.binance_scalp.scalp_setup_measurements import evidence_rank_delta

            feature_adj = round(evidence_rank_delta({sig.setup_name: feats}), 5)
    rank_score = round(rank_score + live_ctx_adj + learned_adj + micro_adj + feature_adj, 4)

    # Measurement: counters only — never flips eligibility (scalp_strategy_owner_v2).
    # Outcome is "hard_blocked" ONLY for mechanical safety (hard_block set).
    # Everything opinion-derived (soft-rank, regime mismatch, symbol stall
    # risk, arm negative-EV, below min score) is recorded as "ranked" —
    # it was penalized, not rejected — to keep telemetry honest about what
    # actually happened.
    with contextlib.suppress(Exception):
        from backend.services.scalp_gate_telemetry import record_gate_event

        from backend.services.binance_scalp.config import get_scalp_config

        _db = get_scalp_config().database_path
        if hard_block:
            record_gate_event(
                _db,
                reason=str(hard_block),
                symbol=sig.symbol,
                outcome="hard_blocked",
                setup=sig.setup_name,
                regime=regime,
            )
        elif sig.passed:
            record_gate_event(
                _db,
                gate_id="STRATEGY_PASS",
                symbol=sig.symbol,
                outcome="ranked",
                setup=sig.setup_name,
                regime=regime,
                detail=str(confidence),
            )
        else:
            record_gate_event(
                _db,
                gate_id="SOFT_RANK_RANKED",
                reason=str(sig.reject_reason or soft_reason or ""),
                symbol=sig.symbol,
                outcome="ranked",
                setup=sig.setup_name,
                regime=regime,
                detail=str(confidence),
            )
        if regime_mismatch:
            record_gate_event(
                _db,
                gate_id="REGIME_MISMATCH",
                symbol=sig.symbol,
                outcome="ranked",
                setup=sig.setup_name,
                regime=regime,
                detail=str(soft_reason),
            )
        if symbol_stall_risk:
            record_gate_event(
                _db,
                gate_id="SYMBOL_STALL_RISK",
                symbol=sig.symbol,
                outcome="ranked",
                setup=sig.setup_name,
                regime=regime,
                detail=f"rank_penalty_mult={os.getenv('SCALP_SYMBOL_STALL_RISK_RANK_MULT', '0.35')}",
            )
        if arm_penalty_mult < 1.0:
            record_gate_event(
                _db,
                gate_id="ARM_NEGATIVE_EV_RANKED",
                symbol=sig.symbol,
                outcome="ranked",
                setup=sig.setup_name,
                regime=regime,
                detail=f"arm_penalty_mult={arm_penalty_mult}",
            )

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
        microstructure_adjustment=micro_adj,
        arm_penalty_mult=arm_penalty_mult,
        arm_stats=arm_stats_for_sizing,
        regime_mismatch=regime_mismatch,
        symbol_stall_risk=symbol_stall_risk,
    )


def prepare_entry_signal(
    ranked: RankedCandidate,
    ctx: StrategyMarketContext,
) -> ScalpSetupSignal:
    """Return an executable entry signal for any candidate that is
    ``entry_eligible`` (i.e. no mechanical safety hard_block fired).

    Architecture v2 (2026-08-11): a strategy's own ``sig.passed`` is a
    strong, but no longer mandatory, ranking/confidence input — see the
    architecture note in ``rank_setup_signal``. This function stamps HONEST
    provenance (never forges ``passed=True``) so downstream sizing
    (``scalp_dynamic_sizing.py``) and learning attribution can tell a
    genuine strategy pass from an opinion-ranked promotion. Position size,
    not eligibility, is what protects capital on a weak/soft-rank/negative-EV
    candidate.
    """
    from dataclasses import replace

    sig = ranked.signal
    if not ranked.entry_eligible:
        # Mechanical safety hard_block — this path should not be reached by
        # the caller for a non-eligible candidate, but stay defensive.
        return sig

    ctx_map = dict(sig.setup_context or {})
    ctx_map["entry_owner"] = "strategy" if sig.passed else "ranking_ev"
    ctx_map["ml_role"] = "rank_size"
    ctx_map["decision_policy_version"] = "scalp_ranking_not_gating_v2"
    ctx_map["soft_rank_entry"] = not sig.passed
    ctx_map["regime_mismatch"] = bool(ranked.regime_mismatch)
    ctx_map["symbol_stall_risk"] = bool(ranked.symbol_stall_risk)
    ctx_map["arm_penalty_mult"] = float(ranked.arm_penalty_mult)
    ctx_map["rank_score_at_entry"] = float(ranked.rank_score)
    ctx_map["selection_confidence"] = str(ranked.selection_confidence)
    ctx_map["bar_closed"] = True

    with contextlib.suppress(Exception):
        from backend.services.binance_scalp.config import get_scalp_config
        from backend.services.scalp_gate_telemetry import record_gate_event

        _db = get_scalp_config().database_path
        record_gate_event(
            _db,
            gate_id="STRATEGY_PASS" if sig.passed else "SOFT_RANK_PROMOTED",
            symbol=sig.symbol,
            outcome="entered",
            setup=sig.setup_name,
            detail=f"rank_score={ranked.rank_score} confidence={ranked.selection_confidence}",
        )

    del ctx  # reserved for future genuine-pass enrichment; unused by design
    return replace(sig, setup_context=ctx_map)


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
    Reads Redis ai_context (cross-process). Range ±0.06. Never a gate.
    """
    sym = str(row.get("symbol") or "")
    with contextlib.suppress(Exception):
        from backend.services.market_role_intelligence import fetch_role_ranking_delta_from_redis

        return max(-0.06, min(0.06, fetch_role_ranking_delta_from_redis(sym)))
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
    # Rank the opportunity, never the coin identity.
    sym_penalty = 0.0
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
    "min_confident_rank",
    "min_tradeable_score",
    "pick_best_global_candidate",
    "pick_best_ranked",
    "prepare_entry_signal",
    "rank_setup_signal",
]
