"""Rank scalp setups: soft penalties for setup misses, hard blocks for safety only."""

from __future__ import annotations

import contextlib
import os
import time
from dataclasses import dataclass, field
from typing import Any

_LEARN_CACHE: dict[str, tuple[float, Any]] = {}
_LEARN_CACHE_TTL_SEC = 30.0


def _learn_cache_get(key: str):
    hit = _LEARN_CACHE.get(key)
    if hit and hit[0] > time.time():
        return True, hit[1]
    return False, None


def _learn_cache_set(key: str, value: Any) -> Any:
    _LEARN_CACHE[key] = (time.time() + _LEARN_CACHE_TTL_SEC, value)
    return value


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
    """When true, non-native regime is a rank/size label only — not entry_eligible."""
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
    # Observability only — already computed during ranking; never a gate.
    micro_ev: dict[str, Any] = field(default_factory=dict)
    rank_components: dict[str, Any] = field(default_factory=dict)


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

    depth_ok, impact, _fill = depth_check(ctx.snap, ctx.notional_usd, ctx.econ)
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
    # Soft-rank (sig.passed=False) still RANKS and may EXECUTE if mechanical
    # safety is clear. Opinion misses change score / size / telemetry only.
    # hard_block owns BUY permission. Never forge passed=True.
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
        stats_key = f"gls:{sig.symbol}:scalp"
        hit, cached_stats = _learn_cache_get(stats_key)
        if hit:
            _stats = cached_stats
        else:
            _stats = _learn_cache_set(stats_key, _gls(_db, sig.symbol, "scalp"))
        role_samples = _stats.sample_count
        role_conf_status = _stats.confidence_status
        learned_adj = round(max(-0.02, min(0.02, _stats.learned_adjustment)), 5)
        with contextlib.suppress(Exception):
            from backend.services.trade_learning_writer import consume_setup_outcomes_for_ranking

            consume_key = f"consume:{sig.setup_name}"
            hit_c, cached_learned = _learn_cache_get(consume_key)
            if hit_c:
                _learned = cached_learned
            else:
                _learned = _learn_cache_set(
                    consume_key,
                    consume_setup_outcomes_for_ranking(
                        _db,
                        sig.setup_name,
                        features={
                            "volatility": getattr(ctx.mom, "realized_volatility_pct", None),
                            "momentum": getattr(ctx.mom, "mid_change_60s", None),
                            "regime": regime,
                            "model_probability": sig.confidence,
                            "market_regime": regime,
                        },
                    ),
                )
            if _learned.get("consumed") and int(_learned.get("n") or 0) >= 8:
                learned_adj = round(
                    max(-0.04, min(0.04, learned_adj + float(_learned.get("rank_delta") or 0.0))),
                    5,
                )

    # Real microstructure features — ranking only, never eligibility.
    # select_v2: EV_10s is the primary four-coin key (frozen validation).
    # DAY get_microstructure_ranking_delta is not used here.
    micro_adj = 0.0
    micro_feats: dict = {}
    with contextlib.suppress(Exception):
        from backend.services.microstructure_engine import compute_features as _cmf

        micro_feats = _cmf(sig.symbol) or {}

    micro_learn_adj = 0.0
    micro_learn: dict = {}
    with contextlib.suppress(Exception):
        from backend.services.binance_scalp.config import get_scalp_config
        from backend.services.binance_scalp.scalp_micro_learning import micro_learning_adjustments

        micro_learn = micro_learning_adjustments(
            get_scalp_config().database_path,
            symbol=sig.symbol,
            ofi_5s=float(micro_feats.get("ofi_5s") or 0.0),
            obi_l5=float(micro_feats.get("obi_l5") or 0.0),
            adverse_selection_score=float(micro_feats.get("adverse_selection_score") or 0.0),
        )
        micro_learn_adj = round(float(micro_learn.get("rank_delta") or 0.0), 5)

    micro_ev: dict = {}
    with contextlib.suppress(Exception):
        from backend.services.binance_scalp.scalp_micro_ev import multi_horizon_ev

        micro_ev = multi_horizon_ev(micro_feats)

    feature_adj = 0.0
    with contextlib.suppress(Exception):
        feats = (sig.setup_context or {}).get("features") or {}
        if feats:
            from backend.services.binance_scalp.scalp_setup_measurements import evidence_rank_delta

            feature_adj = round(evidence_rank_delta({sig.setup_name: feats}), 5)
    rank_components: dict = {}
    with contextlib.suppress(Exception):
        from backend.services.binance_scalp.scalp_micro_rank import apply_repaired_rank

        rank_score, micro_adj, rank_components = apply_repaired_rank(
            static_rank=float(rank_score),
            feats=micro_feats,
            live_ctx_adj=float(live_ctx_adj),
            learned_adj=float(learned_adj),
            micro_learn_adj=float(micro_learn_adj),
            feature_adj=float(feature_adj),
            micro_ev=micro_ev,
        )
    if not rank_components:
        rank_score = round(rank_score + live_ctx_adj + learned_adj + micro_adj + micro_learn_adj + feature_adj, 4)

    # Measurement: counters only — never flips eligibility (scalp_strategy_owner_v2).
    # Outcome is "hard_blocked" ONLY for mechanical safety (hard_block set).
    # Everything opinion-derived (soft-rank, regime mismatch, symbol stall
    # risk, arm negative-EV, below min score) is recorded as "ranked" —
    # it was penalized, not rejected — to keep telemetry honest about what
    # actually happened.
    with contextlib.suppress(Exception):
        from backend.services.binance_scalp.config import get_scalp_config
        from backend.services.scalp_gate_telemetry import record_gate_event

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
        microstructure_adjustment=round(micro_adj + micro_learn_adj, 5),
        arm_penalty_mult=arm_penalty_mult,
        arm_stats=arm_stats_for_sizing,
        regime_mismatch=regime_mismatch,
        symbol_stall_risk=symbol_stall_risk,
        micro_ev=dict(micro_ev or {}),
        rank_components=dict(rank_components or {}),
    )


def prepare_entry_signal(
    ranked: RankedCandidate,
    ctx: StrategyMarketContext,
) -> ScalpSetupSignal:
    """Stamp provenance on an ``entry_eligible`` candidate.

    ``entry_eligible`` is mechanical safety only (``hard_block is None``).
    Soft-rank stays labeled (``soft_rank_entry``, ``entry_owner=ranking_ev``).
    Never forges ``passed=True``.
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
        from backend.services.binance_scalp.scalp_micro_contract import version_stamps
        from backend.services.binance_scalp.scalp_micro_ev import multi_horizon_ev
        from backend.services.microstructure_engine import compute_features

        _mf = compute_features(sig.symbol) or {}
        _ev = multi_horizon_ev(_mf)
        ctx_map.update(version_stamps())
        ctx_map["final_micro_rank_delta"] = float(ranked.microstructure_adjustment)
        ctx_map["learned_adjustment"] = float(ranked.learned_adjustment)
        ctx_map["rank_components"] = dict(getattr(ranked, "rank_components", None) or {})
        ctx_map["microstructure_features"] = {
            k: _mf.get(k)
            for k in (
                "ofi_1s",
                "ofi_3s",
                "ofi_5s",
                "ofi_15s",
                "ofi_30s",
                "obi_l1",
                "obi_l5",
                "obi_l10",
                "obi_l20",
                "weighted_depth_imbalance",
                "microprice_pressure",
                "microprice_accel",
                "agg_flow_imbalance_1s",
                "agg_flow_imbalance_5s",
                "trade_count_1s",
                "trade_count_5s",
                "flow_acceleration",
                "signed_volume_5s",
                "bid_cancelled_5s",
                "ask_cancelled_5s",
                "bid_replenished_5s",
                "ask_replenished_5s",
                "bid_absorption_score",
                "ask_absorption_score",
                "depth_fragility",
                "adverse_selection_score",
                "spread_pct",
            )
            if k in _mf
        }
        ctx_map.update({k: _ev[k] for k in _ev if k.startswith("EV_") or k.startswith("p_")})
        ctx_map["selection_micro_score"] = _ev.get("selection_micro_score")
        ctx_map["model_version"] = _ev.get("model_version")
        ctx_map["calibration_status"] = _ev.get("calibration_status")

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
    soft_tier = _soft_base_score(soft.split(":", maxsplit=1)[0] if soft else None)
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
    str(row.get("symbol") or "")
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


HOLD_ACTION_EV = 0.0
HOLD_ACTION_NAME = "HOLD"
DECISION_POLICY_HOLD_AS_ACTION = "scalp_hold_as_action_v1"


def _num(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _signal_field(row: dict[str, Any], name: str, default: float = 0.0) -> float:
    sig = row.get("signal")
    if sig is None:
        return _num((row.get("rank_meta") or {}).get(name), default)
    if isinstance(sig, dict):
        return _num(sig.get(name), default)
    return _num(getattr(sig, name, None), default)


def candidate_roundtrip_cost_pct(row: dict[str, Any]) -> float:
    """Spread + fees + slippage + impact. Same economics as live SCALP fills."""
    snap = row.get("snap")
    spread = _num(getattr(snap, "spread_pct", None) if snap is not None else None)
    if spread <= 0:
        spread = _signal_field(row, "spread_pct")
    impact = _signal_field(row, "impact_pct")
    if impact <= 0:
        impact = _num((row.get("rank_meta") or {}).get("impact_pct"))
    intel = row.get("intelligence") or {}
    slip = _num(intel.get("slippage_estimate"))
    try:
        from backend.services.binance_scalp.economics import ScalpEconomics

        econ = ScalpEconomics.from_env()
        if slip <= 0:
            slip = float(econ.slippage_buffer_pct)
        return float(econ.roundtrip_fee_pct) + spread + impact + slip
    except Exception:
        return 0.0004 + spread + impact + (slip if slip > 0 else 0.0001)


def candidate_expected_gross_pct(row: dict[str, Any]) -> float:
    """Gross move the candidate itself claimed. Missing/zero stays zero."""
    expected = _signal_field(row, "expected_move_pct")
    if expected > 0:
        return expected
    meta = row.get("rank_meta") or {}
    expected = _num(meta.get("expected_move_pct"))
    if expected > 0:
        return expected
    return _num((row.get("intelligence") or {}).get("expected_move_pct") or (row.get("intelligence") or {}).get("projected_gross_pct"))


def candidate_expected_net_ev(row: dict[str, Any]) -> float:
    """BUY expected net after costs. HOLD is a separate action with EV=0.

    If an accepted forward-net artifact is loaded, use its already-net
    prediction. Otherwise fall back to claimed gross minus costs.
    This is not a threshold gate.
    """
    with contextlib.suppress(Exception):
        from backend.services.binance_scalp.forward_net_predictor import predict_row_expected_net

        predicted = predict_row_expected_net(row)
        if predicted is not None:
            return float(predicted)
    return candidate_expected_gross_pct(row) - candidate_roundtrip_cost_pct(row)


def candidate_positive_net_probability(row: dict[str, Any]) -> float:
    """Probability the BUY's expected net is positive. Not a permission gate."""
    ev = candidate_expected_net_ev(row)
    cost = candidate_roundtrip_cost_pct(row)
    if ev <= HOLD_ACTION_EV:
        return 0.0
    conf = _signal_field(row, "confidence")
    if 0.0 < conf <= 1.0:
        return round(min(0.99, max(0.51, conf)), 4)
    surplus = ev / cost if cost > 0 else ev
    return round(min(0.95, max(0.51, 0.50 + surplus * 8.0)), 4)


def attach_action_predictions(row: dict[str, Any]) -> dict[str, Any]:
    """Stamp absolute predicted outcomes on a ranking row. Rank alone is not enough."""
    mh = row.get("multi_horizon_ev") or {}
    horizons = mh.get("horizons") if isinstance(mh, dict) else None
    expected_mfe = None
    expected_mae = None
    expected_hold = None
    if isinstance(horizons, list) and horizons:
        mfe_vals = [_num(h.get("expected_mfe_pct")) for h in horizons if isinstance(h, dict)]
        mae_vals = [_num(h.get("expected_mae_pct")) for h in horizons if isinstance(h, dict)]
        if mfe_vals:
            expected_mfe = round(sum(mfe_vals) / len(mfe_vals), 6)
        if mae_vals:
            expected_mae = round(sum(mae_vals) / len(mae_vals), 6)
        mid = horizons[len(horizons) // 2]
        if isinstance(mid, dict):
            expected_hold = mid.get("bucket")
    gross = candidate_expected_gross_pct(row)
    cost = candidate_roundtrip_cost_pct(row)
    ev = candidate_expected_net_ev(row)
    row["expected_gross_move"] = round(gross, 8)
    row["roundtrip_cost_pct"] = round(cost, 8)
    row["expected_net_ev"] = round(ev, 8)
    row["predicted_net_return"] = round(ev, 8)
    row["predicted_prob_positive_net"] = candidate_positive_net_probability(row)
    row["expected_mfe"] = expected_mfe
    row["expected_mae"] = expected_mae
    row["expected_hold"] = expected_hold
    row["hold_action_ev"] = HOLD_ACTION_EV
    row["action_name"] = f"BUY_{row.get('symbol')}"
    row["forward_net_model_version"] = ""
    with contextlib.suppress(Exception):
        from backend.services.binance_scalp.forward_net_predictor import load_accepted_artifact

        art = load_accepted_artifact()
        if art is not None:
            row["forward_net_model_version"] = art.version
            feats = {}
            meta = row.get("rank_meta") or {}
            meas = meta.get("setup_measurements") or row.get("setup_measurements") or {}
            if meas:
                from backend.services.binance_scalp.forward_net_predictor import (
                    flatten_measurements,
                    predict_artifact,
                )

                feats = flatten_measurements(meas, live_book=True)
                pred = predict_artifact(art, feats)
                row["predicted_prob_positive_net"] = round(float(pred["predicted_prob_positive_net"]), 4)
    return row


def rank_actions_with_hold(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """BUY actions plus HOLD(EV=0). Highest expected net wins."""
    actions: list[dict[str, Any]] = []
    for row in rows:
        stamped = attach_action_predictions(dict(row))
        actions.append(stamped)
    actions.append(
        {
            "action_name": HOLD_ACTION_NAME,
            "symbol": HOLD_ACTION_NAME,
            "entry_eligible": True,
            "expected_gross_move": 0.0,
            "roundtrip_cost_pct": 0.0,
            "expected_net_ev": HOLD_ACTION_EV,
            "predicted_net_return": HOLD_ACTION_EV,
            "predicted_prob_positive_net": 0.5,
            "expected_mfe": 0.0,
            "expected_mae": 0.0,
            "expected_hold": 0.0,
            "hold_action_ev": HOLD_ACTION_EV,
            "rank_score": 0.0,
        }
    )
    actions.sort(key=lambda r: (-_num(r.get("expected_net_ev")), -_num(r.get("rank_score"))))
    return actions


def pick_best_global_candidate(rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    """Select the highest select_v2 rank among hard-eligible candidates.

    Path-net / HOLD opinion does not skip a higher-ranked available coin.
    Negative EV_10s still trades if the candidate is entry_eligible.
    Already-open symbols are skipped. Hard safety stays on entry_eligible.
    """
    from backend.services.binance_scalp.scalp_micro_rank import EV_TIE_TOLERANCE, ev_scores_tied

    eligible = [r for r in rows if r.get("entry_eligible") and not r.get("already_open") and str(r.get("hard_block") or "") == ""]
    if not eligible:
        return None

    for row in eligible:
        attach_action_predictions(row)

    top_rank = max(_num(r.get("rank_score")) for r in eligible)
    clustered = [r for r in eligible if ev_scores_tied(_num(r.get("rank_score")), top_rank, tol=EV_TIE_TOLERANCE)]
    if len(clustered) > 1:
        return max(clustered, key=_global_tie_key)
    return clustered[0]


__all__ = [
    "DECISION_POLICY_HOLD_AS_ACTION",
    "HARD_REJECT_REASONS",
    "HOLD_ACTION_EV",
    "HOLD_ACTION_NAME",
    "SOFT_REJECT_SCORE",
    "RankedCandidate",
    "attach_action_predictions",
    "candidate_expected_net_ev",
    "min_confident_rank",
    "min_tradeable_score",
    "pick_best_global_candidate",
    "pick_best_ranked",
    "prepare_entry_signal",
    "rank_actions_with_hold",
    "rank_setup_signal",
]
