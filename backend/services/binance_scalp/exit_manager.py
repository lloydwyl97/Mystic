"""Stateful paper-only scalp exit manager — no blind stale-timeout sells."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

from backend.services.binance_scalp.config import ScalpConfig
from backend.services.binance_scalp.economics import ScalpEconomics
from backend.services.binance_scalp.market_reader import MarketSnapshot
from backend.services.binance_scalp.momentum_tracker import MomentumDiagnostics
from backend.services.binance_scalp.paper_spread_caps import uses_paper_spread_caps
from backend.services.binance_scalp.strategy_exit_rules import setup_invalidated

STATE_OPEN = "OPEN"
STATE_MAX_HOLD_REVIEW = "MAX_HOLD_REVIEW"
STATE_HEALTHY_HOLD = "HEALTHY_HOLD"
STATE_RECOVERY_HOLD = "RECOVERY_HOLD"

EXIT_NET_PROFIT_TARGET = "NET_PROFIT_TARGET"
EXIT_SETUP_INVALIDATED = "SETUP_INVALIDATED_EXIT"
EXIT_MOMENTUM_FAILED = "MOMENTUM_FAILED_EXIT"
EXIT_EARLY_SCRATCH = "EARLY_SCRATCH_EXIT"
EXIT_MAX_HOLD_HARD_LIMIT = "MAX_HOLD_HARD_LIMIT"
# Micro-TP: books a small green exit as soon as MFE has "armed" the exit and
# price gives back a configurable fraction. Prevents scalp from repeatedly
# reaching +0.15%+ then giving it all back and eventually hitting MAX_HOLD
# for a small loss — which is what the 14-sell history showed.
EXIT_MICRO_TP = "MICRO_TP_LOCK"
EXIT_PATH_EXECUTABLE_PROFIT = "PATH_EXECUTABLE_PROFIT"
# Bounded downside for the path-aware branch. That branch returns before the
# scratch/stall/momentum exits below, so without this a losing position has no
# exit at all until the 20-minute horizon. The first 53 paper closes split
# exactly two ways: 25 profit takes averaging +$0.0085 and 28 horizon timeouts
# averaging -$0.053, with no timeout ever closing green.
EXIT_PATH_MAX_ADVERSE_STOP = "PATH_MAX_ADVERSE_STOP"

DECISION_HOLD = "HOLD"
DECISION_SELL = "SELL"
PATH_AWARE_POLICY = "scalp_path_aware_v1"

RECOVERY_MIN_PCT = 0.00012
HIGHER_LOWS_MIN_REVIEWS = 2


def _path_aware_exit_enabled() -> bool:
    """Take profit at the first executable net that clears the floor, and cut
    losers at a bounded stop rather than at the horizon.

    This is an exit policy, not an entry gate. The scratch/stall opinion exits
    are still skipped so a later favorable print can be taken, but the downside
    is bounded so "skipped" cannot mean "unbounded".
    """
    return os.getenv("SCALP_PATH_AWARE_EXIT", "true").strip().lower() in {"1", "true", "yes", "on"}


def _path_min_executable_net_pct() -> float:
    """Minimum executable net before the path-aware branch books a winner.

    Deliberately low. Across the first 53 paper closes the best winner was
    +0.1015% net and the median was +0.0206%, so raising this floor to a
    conventional target books nothing at all — at 0.15% it would have taken
    zero of the 25 winners. The asymmetry is fixed on the loss side instead.
    """
    try:
        return float(os.getenv("SCALP_PATH_MIN_EXECUTABLE_NET_PCT", "0.0001"))
    except (TypeError, ValueError):
        return 0.0001


def _path_max_adverse_net_pct() -> float:
    """Executable net (as a positive magnitude) at which a loser is cut.

    Set beyond the model's own predicted MAE for every traded symbol (max
    observed 0.122%) so it fires on tail moves rather than on the adverse
    excursion a healthy position is expected to survive.
    """
    try:
        v = float(os.getenv("SCALP_PATH_MAX_ADVERSE_NET_PCT", "0.0015"))
    except (TypeError, ValueError):
        v = 0.0015
    return abs(v)


def _review_interval_sec() -> int:
    return int(os.getenv("SCALP_REVIEW_INTERVAL_SEC", "30"))


def _micro_tp_enabled() -> bool:
    return os.getenv("SCALP_MICRO_TP_ENABLED", "true").strip().lower() in ("1", "true", "yes", "on")


def _micro_tp_arm_pct() -> float:
    """MFE (as pct of entry) that arms the micro-TP lock. Once armed, we watch
    for a give-back to trigger the exit."""
    try:
        return float(os.getenv("SCALP_MICRO_TP_ARM_PCT", "0.0018"))
    except (TypeError, ValueError):
        return 0.0018


def _micro_tp_min_exec_net_pct() -> float:
    """Minimum executable net pct required at exit — do not fire micro-TP if
    the current fill would still be a loss after fees + spread."""
    try:
        return float(os.getenv("SCALP_MICRO_TP_MIN_NET_PCT", "0.0006"))
    except (TypeError, ValueError):
        return 0.0006


def _micro_tp_giveback_frac() -> float:
    """Fraction of armed MFE to give back before firing (0.35 = 35% giveback)."""
    try:
        v = float(os.getenv("SCALP_MICRO_TP_GIVEBACK_FRAC", "0.35"))
    except (TypeError, ValueError):
        v = 0.35
    # Clamp so we never accept absurd values
    return max(0.10, min(0.75, v))


def _micro_tp_exit(
    *,
    max_fav: float,
    fav: float,
    executable_net_pct: float,
    hold_sec: float,
) -> tuple[bool, str]:
    """Book any decent green if giveback exceeds threshold.

    Only fires when:
      * feature enabled (kill switch defaults on)
      * max_fav has reached the arm threshold (default 0.18%)
      * current executable net is still positive by min-exec floor
      * giveback from peak is >= giveback_frac * max_fav
    """
    if not _micro_tp_enabled():
        return False, ""
    if hold_sec < 30.0:
        # Skip the first 30s to avoid whipsaw on entry noise.
        return False, ""
    arm = _micro_tp_arm_pct()
    if max_fav < arm:
        return False, ""
    if executable_net_pct < _micro_tp_min_exec_net_pct():
        return False, ""
    giveback_frac = _micro_tp_giveback_frac()
    # Giveback = how much of the MFE we've lost from peak.
    # If max_fav = 0.20% and current fav = 0.10%, giveback_frac = (0.20-0.10)/0.20 = 0.50
    if max_fav <= 0:
        return False, ""
    giveback = max(0.0, (max_fav - fav) / max_fav)
    if giveback < giveback_frac:
        return False, ""
    return True, (f"micro_tp_lock mfe_pct={max_fav:.4f} fav_pct={fav:.4f} giveback={giveback:.2f} net_exec={executable_net_pct:.4f}")


def _review_trigger_sec(econ: ScalpEconomics) -> int:
    return int(os.getenv("SCALP_REVIEW_TRIGGER_SEC", str(econ.stale_scalp_timeout_sec)))


def _max_hold_hard_sec(econ: ScalpEconomics) -> int:
    raw = os.getenv("SCALP_MAX_HOLD_SEC", "")
    if raw and str(raw).strip():
        return int(raw)
    hold_min = os.getenv("SCALP_HOLD_MAX_MINUTES", "")
    if hold_min and str(hold_min).strip():
        return int(float(hold_min) * 60)
    return max(_review_trigger_sec(econ) * 4, 1200)


def _meaningful_recovery(recovery: float, max_fav: float, econ: ScalpEconomics) -> bool:
    """Tiny bid upticks off session low are not enough to defer scratch/exit."""
    target_progress = econ.net_profit_target_pct * _scratch_progress_frac()
    return recovery >= RECOVERY_MIN_PCT and max_fav >= target_progress * 0.4


def _is_genuine_pass_context(setup_context: dict | None) -> bool:
    ctx = setup_context or {}
    if ctx.get("soft_rank_entry"):
        return False
    if ctx.get("entry_owner") == "strategy":
        return True
    return bool(ctx.get("passed")) and not ctx.get("soft_rank_entry")


def _scratch_min_hold_sec(setup_name: str | None = None, setup_context: dict | None = None) -> int:
    base = int(os.getenv("SCALP_SCRATCH_MIN_HOLD_SEC", "180"))
    name = (setup_name or "").strip().lower()
    # Range bounce gets a slightly longer floor, but must still scratch before
    # MAX_HOLD (post soft-rank paper: delayed range scratch → max-hold bleed).
    if name == "range_bounce_scalp":
        base = int(os.getenv("SCALP_RANGE_SCRATCH_MIN_HOLD_SEC", str(max(base, 300))))
    # Replay of genuine-pass VWAP reclaim: 3/3 scratched at the short floor
    # finished red, while the same entries held to the 20m horizon were 66.7%
    # WR / +EV. Historical scratches that never had MFE stay on the short
    # floor (soft-rank). This is an exit-timing change, not an entry gate.
    if _is_genuine_pass_context(setup_context):
        return int(os.getenv("SCALP_GENUINE_SCRATCH_MIN_HOLD_SEC", str(max(base, 420))))
    return base


def _scratch_flat_upper_pct() -> float:
    """Executable net at or below this (flat / tiny green that missed target)."""
    return float(os.getenv("SCALP_SCRATCH_FLAT_UPPER_PCT", "0.0002"))


def _scratch_progress_frac() -> float:
    # Higher = require more MFE toward target before treating a hold as "progress".
    return float(os.getenv("SCALP_SCRATCH_PROGRESS_FRAC", "0.40"))


def _scratch_deep_loss_pct() -> float:
    return float(os.getenv("SCALP_SCRATCH_DEEP_LOSS_PCT", "-0.006"))


def _scratch_min_reviews(setup_name: str | None = None) -> int:
    base = int(os.getenv("SCALP_SCRATCH_MIN_REVIEWS", "1"))
    name = (setup_name or "").strip().lower()
    if name == "range_bounce_scalp":
        return int(os.getenv("SCALP_RANGE_SCRATCH_MIN_REVIEWS", str(max(base, 3))))
    return base


def _effective_scratch_min_reviews(setup_name: str | None = None, hold_ev_reduction: int = 0) -> int:
    """Item p8 promotion: HoldEV can reduce (never increase) the required
    review count, bounded at a floor of 1 — it can make an already-
    scratchable, already-momentum-stalled position scratch a review or two
    sooner, but can never remove the review-count floor entirely."""
    return max(1, _scratch_min_reviews(setup_name) - max(0, int(hold_ev_reduction or 0)))


def _stall_exit_hold_frac() -> float:
    # Default 50% of hard max-hold so stalled paper scalps exit ~10m, not 15–20m.
    return float(os.getenv("SCALP_STALL_EXIT_HOLD_FRAC", "0.50"))


def _stall_exit_min_sec(hard: int) -> int:
    raw = os.getenv("SCALP_STALL_EXIT_MIN_SEC", "")
    if raw and str(raw).strip():
        return int(raw)
    return int(hard * _stall_exit_hold_frac())


def _momentum_stalled(mom: MomentumDiagnostics) -> bool:
    """Flat/choppy or negative — not a live recovery push toward target."""
    if mom.flat_regime:
        return True
    if mom.bid_change_15s < 0 and mom.bid_change_30s < 0 and mom.mid_change_30s <= 0:
        return True
    return mom.bid_change_15s <= 0 and mom.bid_change_30s <= 0 and mom.mid_change_30s <= 0


def _scratchable_net(executable_net_pct: float, econ: ScalpEconomics) -> bool:
    """Flat, tiny green, or small loss — not a deep drawdown worth holding for recovery."""
    upper = max(_scratch_flat_upper_pct(), econ.net_profit_target_pct * 0.45)
    return _scratch_deep_loss_pct() < executable_net_pct <= upper


def _early_scratch_exit(
    *,
    hold_sec: float,
    hard: int,
    max_fav: float,
    executable_net_pct: float,
    mom: MomentumDiagnostics,
    recovery: float,
    econ: ScalpEconomics,
    stale_review_count: int,
    setup_name: str | None = None,
    setup_context: dict | None = None,
    hold_ev_reduction: int = 0,
) -> tuple[bool, str]:
    """Exit flat/slightly-negative scalps that stall before hard max-hold."""
    min_hold = _scratch_min_hold_sec(setup_name, setup_context)
    if hold_sec < min_hold or hold_sec >= hard:
        return False, ""
    if _meaningful_recovery(recovery, max_fav, econ):
        return False, ""
    target_progress = econ.net_profit_target_pct * _scratch_progress_frac()
    no_progress = max_fav < target_progress
    if not no_progress:
        return False, ""
    if not _momentum_stalled(mom):
        return False, ""
    if not _scratchable_net(executable_net_pct, econ):
        return False, ""
    min_reviews = _effective_scratch_min_reviews(setup_name, hold_ev_reduction)
    trigger = _review_trigger_sec(econ)
    if hold_sec >= trigger and stale_review_count >= min_reviews:
        return True, "stalled_no_progress_review_scratch"
    if hold_sec >= min_hold + 90 and stale_review_count >= min_reviews:
        return True, "stalled_no_progress_early_scratch"
    return False, ""


def _stall_before_max_hold(
    *,
    hold_sec: float,
    hard: int,
    max_fav: float,
    executable_net_pct: float,
    mom: MomentumDiagnostics,
    recovery: float,
    econ: ScalpEconomics,
    stale_review_count: int,
    setup_name: str | None = None,
    hold_ev_reduction: int = 0,
) -> tuple[bool, str]:
    """Cut prolonged no-progress holds before the hard ceiling."""
    if hold_sec >= hard:
        return False, ""
    stall_at = _stall_exit_min_sec(hard)
    if hold_sec < stall_at:
        return False, ""
    target_progress = econ.net_profit_target_pct * _scratch_progress_frac()
    if max_fav >= target_progress:
        return False, ""
    if _meaningful_recovery(recovery, max_fav, econ):
        return False, ""
    if not _scratchable_net(executable_net_pct, econ):
        return False, ""
    if not _momentum_stalled(mom):
        return False, ""
    if stale_review_count >= _effective_scratch_min_reviews(setup_name, hold_ev_reduction):
        return True, "stall_before_max_hold_no_progress"
    return False, ""


def _spread_cap(econ: ScalpEconomics, config: ScalpConfig, symbol: str) -> float:
    if uses_paper_spread_caps(
        scalp_live=config.scalp_live,
        calibration_mode=config.calibration_mode,
        scalp_paper_enabled=config.scalp_paper_enabled,
    ):
        return econ.spread_cap_for_symbol(symbol)
    return econ.spread_cap_pct


def _higher_lows(review_lows: list[float]) -> bool:
    if len(review_lows) < HIGHER_LOWS_MIN_REVIEWS:
        return False
    tail = review_lows[-HIGHER_LOWS_MIN_REVIEWS:]
    return all(tail[i] > tail[i - 1] for i in range(1, len(tail)))


@dataclass(frozen=True)
class PositionTrack:
    entry_price: float
    state: str
    max_favorable_pct: float
    max_adverse_pct: float
    session_low_bid: float
    stale_review_count: int
    review_lows: tuple[float, ...]
    setup_name: str
    setup_context: dict[str, Any]


@dataclass(frozen=True)
class ExitReviewResult:
    decision: str
    state: str
    reason: str
    exit_reason: str | None
    diagnostics: dict[str, Any]
    updated_track: PositionTrack


def evaluate_exit(
    *,
    track: PositionTrack,
    snap: MarketSnapshot,
    mom: MomentumDiagnostics,
    econ: ScalpEconomics,
    config: ScalpConfig,
    trade_id: str,
    hold_sec: float,
    executable_net_pct: float,
    profit_hit: bool,
    exit_spread_ok: bool,
    perform_review: bool,
) -> ExitReviewResult:
    entry = track.entry_price
    bid = snap.best_bid
    cap = _spread_cap(econ, config, snap.symbol)

    fav = (bid - entry) / entry if entry > 0 else 0.0
    adv = (entry - bid) / entry if entry > 0 else 0.0
    max_fav = max(track.max_favorable_pct, fav)
    max_adv = max(track.max_adverse_pct, adv)
    session_low = min(track.session_low_bid, bid)
    recovery = (bid - session_low) / entry if entry > 0 else 0.0
    recovering = recovery >= RECOVERY_MIN_PCT and bid > session_low
    review_lows = list(track.review_lows)
    stale_review_count = track.stale_review_count

    diag_base: dict[str, Any] = {
        "trade_id": trade_id,
        "symbol": snap.symbol,
        "setup_name": track.setup_name,
        "hold_seconds": round(hold_sec, 1),
        "current_bid": bid,
        "entry_price": entry,
        "executable_net_pct": executable_net_pct,
        "max_favorable_pct": max_fav,
        "max_adverse_pct": max_adv,
        "recovery_from_low_pct": recovery,
        "bid_change_15s": mom.bid_change_15s,
        "bid_change_30s": mom.bid_change_30s,
        "bid_change_60s": mom.bid_change_60s,
        "spread_pct": snap.spread_pct,
        "spread_cap_pct": cap,
        "spread_ok": snap.spread_pct <= cap,
        # Always present (not only on the review branch) so EARLY_SCRATCH_EXIT
        # closes can be attributed with the review count that triggered them.
        "stale_review_count": stale_review_count,
    }

    # HoldEV (item p8) — see hold_ev_engine.py's architecture note. Computed
    # on every review (hold and sell alike) so it is a live, observed signal
    # for SCALP, mirroring the DAY wiring in day_controlled_exits.py. Its
    # score also feeds hold_ev_scratch_review_reduction below — a bounded,
    # tighten-only reduction (never increase) in the stale-review count the
    # early-scratch/stall checks require; it never changes decision/state/
    # reason on its own and can never make an otherwise-non-scratchable
    # position scratchable.
    hold_ev_reduction = 0
    try:
        from backend.services.hold_ev_engine import compute_hold_ev, hold_ev_scratch_review_reduction

        _hev = compute_hold_ev(
            symbol=snap.symbol,
            strategy="scalp",
            entry_price=entry,
            current_price=bid,
            highest_price=entry * (1.0 + max_fav),
            hold_minutes=hold_sec / 60.0,
            realized_volatility_pct=getattr(mom, "realized_volatility_pct", None),
            spread_pct=snap.spread_pct,
        )
        diag_base["hold_ev_score"] = _hev.hold_ev_score
        diag_base["hold_ev_recommendation"] = _hev.recommendation
        diag_base["hold_ev_confidence"] = _hev.confidence
        hold_ev_reduction = hold_ev_scratch_review_reduction(_hev.hold_ev_score, _hev.confidence)
        diag_base["hold_ev_scratch_review_reduction"] = hold_ev_reduction
    except Exception:
        pass

    def _hold(state: str, reason: str) -> ExitReviewResult:
        d = {**diag_base, "decision": DECISION_HOLD, "state": state, "reason": reason, "higher_lows": _higher_lows(review_lows)}
        return ExitReviewResult(
            DECISION_HOLD,
            state,
            reason,
            None,
            d,
            PositionTrack(
                entry,
                state,
                max_fav,
                max_adv,
                session_low,
                stale_review_count,
                tuple(review_lows),
                track.setup_name,
                track.setup_context,
            ),
        )

    def _sell(state: str, reason: str, exit_reason: str) -> ExitReviewResult:
        d = {**diag_base, "decision": DECISION_SELL, "state": state, "reason": reason, "higher_lows": _higher_lows(review_lows)}
        return ExitReviewResult(
            DECISION_SELL,
            state,
            reason,
            exit_reason,
            d,
            PositionTrack(
                entry,
                state,
                max_fav,
                max_adv,
                session_low,
                stale_review_count,
                tuple(review_lows),
                track.setup_name,
                track.setup_context,
            ),
        )

    if profit_hit:
        return _sell(STATE_OPEN, "profit_target_met", EXIT_NET_PROFIT_TARGET)

    if _path_aware_exit_enabled():
        diag_base["exit_policy"] = PATH_AWARE_POLICY
        if executable_net_pct > _path_min_executable_net_pct():
            return _sell(STATE_OPEN, "path_first_executable_profit", EXIT_PATH_EXECUTABLE_PROFIT)
        mtp, mtp_reason = _micro_tp_exit(
            max_fav=max_fav,
            fav=fav,
            executable_net_pct=executable_net_pct,
            hold_sec=hold_sec,
        )
        if mtp:
            diag_base["micro_tp_trigger_detail"] = mtp_reason
            return _sell(STATE_OPEN, mtp_reason, EXIT_MICRO_TP)
        stop_pct = _path_max_adverse_net_pct()
        diag_base["path_max_adverse_net_pct"] = stop_pct
        diag_base["path_min_executable_net_pct"] = _path_min_executable_net_pct()
        if executable_net_pct <= -stop_pct:
            return _sell(
                STATE_MAX_HOLD_REVIEW,
                f"path_max_adverse_stop net={executable_net_pct:.5f} bound=-{stop_pct:.5f}",
                EXIT_PATH_MAX_ADVERSE_STOP,
            )
        hard = _max_hold_hard_sec(econ)
        if hold_sec >= hard:
            return _sell(STATE_MAX_HOLD_REVIEW, f"path_horizon_{hard}s", EXIT_MAX_HOLD_HARD_LIMIT)
        return _hold(STATE_OPEN, "path_awaiting_executable_profit")

    # Micro-TP: book any position that reached the arm threshold and gave back
    # a configurable share of MFE from peak. This is what turns "MFE=+0.20%
    # then times out at -0.15%" into "+0.13% booked". Fires BEFORE scratch/stall
    # so a position that's already had a good move doesn't get scratched for
    # small loss just because momentum stalled after the peak.
    mtp, mtp_reason = _micro_tp_exit(
        max_fav=max_fav,
        fav=fav,
        executable_net_pct=executable_net_pct,
        hold_sec=hold_sec,
    )
    if mtp:
        diag_base["micro_tp_arm_pct"] = _micro_tp_arm_pct()
        diag_base["micro_tp_giveback_frac"] = _micro_tp_giveback_frac()
        diag_base["micro_tp_trigger_detail"] = mtp_reason
        return _sell(STATE_OPEN, mtp_reason, EXIT_MICRO_TP)

    hard = _max_hold_hard_sec(econ)
    scratch, scratch_reason = _early_scratch_exit(
        hold_sec=hold_sec,
        hard=hard,
        max_fav=max_fav,
        executable_net_pct=executable_net_pct,
        mom=mom,
        recovery=recovery,
        econ=econ,
        stale_review_count=stale_review_count,
        setup_name=track.setup_name,
        setup_context=track.setup_context,
        hold_ev_reduction=hold_ev_reduction,
    )
    if scratch:
        target_progress = econ.net_profit_target_pct * _scratch_progress_frac()
        diag_base["scratch_trigger_detail"] = scratch_reason
        diag_base["scratch_target_progress_pct"] = target_progress
        diag_base["scratch_min_reviews"] = _effective_scratch_min_reviews(track.setup_name, hold_ev_reduction)
        diag_base["scratch_min_hold_sec"] = _scratch_min_hold_sec(track.setup_name, track.setup_context)
        diag_base["scratch_momentum_stalled"] = _momentum_stalled(mom)
        diag_base["scratch_flat_or_slight_neg"] = _scratchable_net(executable_net_pct, econ)
        return _sell(STATE_RECOVERY_HOLD, scratch_reason, EXIT_EARLY_SCRATCH)

    stall, stall_reason = _stall_before_max_hold(
        hold_sec=hold_sec,
        hard=hard,
        max_fav=max_fav,
        executable_net_pct=executable_net_pct,
        mom=mom,
        recovery=recovery,
        econ=econ,
        stale_review_count=stale_review_count,
        setup_name=track.setup_name,
        hold_ev_reduction=hold_ev_reduction,
    )
    if stall:
        diag_base["scratch_trigger_detail"] = stall_reason
        diag_base["stall_exit_min_sec"] = _stall_exit_min_sec(hard)
        return _sell(STATE_RECOVERY_HOLD, stall_reason, EXIT_EARLY_SCRATCH)

    if hold_sec >= hard:
        return _sell(STATE_MAX_HOLD_REVIEW, f"max_hold_hard_limit_{hard}s", EXIT_MAX_HOLD_HARD_LIMIT)

    trigger = _review_trigger_sec(econ)
    if hold_sec < trigger:
        return _hold(STATE_OPEN, "awaiting_profit_or_review")

    if not perform_review:
        return _hold(track.state or STATE_RECOVERY_HOLD, "between_review_interval")

    stale_review_count += 1
    review_lows.append(bid)
    if len(review_lows) > 8:
        review_lows = review_lows[-8:]
    hl = _higher_lows(review_lows)
    diag_base["higher_lows"] = hl
    diag_base["stale_review_count"] = stale_review_count

    invalidated, inv_reason = setup_invalidated(
        track.setup_name,
        track.setup_context,
        snap=snap,
        mom=mom,
        entry_price=entry,
        executable_net_pct=executable_net_pct,
    )

    spread_ok = snap.spread_pct <= cap and exit_spread_ok
    momentum_negative = mom.bid_change_15s < 0 and mom.bid_change_30s <= 0
    momentum_stalled = _momentum_stalled(mom)
    momentum_rising = mom.bid_change_15s > 0 or mom.mid_change_15s > 0
    below_entry = executable_net_pct < 0
    meaningful_rec = _meaningful_recovery(recovery, max_fav, econ)
    target_progress = econ.net_profit_target_pct * _scratch_progress_frac()

    scratch_ready = hold_sec >= _scratch_min_hold_sec(track.setup_name, track.setup_context)
    if (
        scratch_ready
        and stale_review_count >= _effective_scratch_min_reviews(track.setup_name, hold_ev_reduction)
        and max_fav < target_progress
        and _scratchable_net(executable_net_pct, econ)
        and momentum_stalled
        and not meaningful_rec
    ):
        diag_base["scratch_trigger_detail"] = "review_stall_no_progress"
        return _sell(STATE_RECOVERY_HOLD, "review_stall_no_progress", EXIT_EARLY_SCRATCH)

    if not spread_ok and not (invalidated and momentum_negative and below_entry):
        return _hold(STATE_RECOVERY_HOLD, "spread_too_wide_for_exit_wait")

    if momentum_rising and (fav >= 0 or meaningful_rec):
        return _hold(STATE_HEALTHY_HOLD, "momentum_rising_below_target")

    if below_entry and (meaningful_rec or hl):
        return _hold(STATE_RECOVERY_HOLD, "red_but_recovery_signs")

    if invalidated and below_entry and momentum_negative and not meaningful_rec:
        return _sell(EXIT_SETUP_INVALIDATED, inv_reason or "setup_invalidated", EXIT_SETUP_INVALIDATED)

    if scratch_ready and below_entry and momentum_stalled and not hl and not meaningful_rec and stale_review_count >= 1:
        return _sell(EXIT_MOMENTUM_FAILED, "no_recovery_negative_momentum", EXIT_MOMENTUM_FAILED)

    if not below_entry:
        return _hold(STATE_HEALTHY_HOLD, "green_but_below_target")

    return _hold(STATE_RECOVERY_HOLD, "review_inconclusive_hold")


def track_from_row(row: Any, diag: dict | None = None) -> PositionTrack:
    review_lows: list[float] = []
    session_low = float(row["entry_price"])
    setup_name = ""
    setup_context: dict[str, Any] = {}
    if diag:
        review_lows = list(diag.get("review_lows") or [])
        session_low = float(diag.get("session_low_bid") or session_low)
        setup_name = str(diag.get("setup_name") or "")
        setup_context = dict(diag.get("setup_context") or {})
        for key in ("passed", "soft_rank_entry", "entry_owner"):
            if key in diag and key not in setup_context:
                setup_context[key] = diag.get(key)
    slb = _col(row, "session_low_bid")
    if slb is not None:
        session_low = float(slb)
    return PositionTrack(
        entry_price=float(row["entry_price"]),
        state=str(_col(row, "state") or STATE_OPEN),
        max_favorable_pct=float(_col(row, "max_favorable_pct") or 0),
        max_adverse_pct=float(_col(row, "max_adverse_pct") or 0),
        session_low_bid=session_low,
        stale_review_count=int(_col(row, "stale_review_count") or 0),
        review_lows=tuple(review_lows),
        setup_name=setup_name,
        setup_context=setup_context,
    )


def _col(row: Any, name: str) -> Any:
    try:
        keys = row.keys() if hasattr(row, "keys") else []
        if keys and name not in keys:
            return None
        return row[name]
    except (KeyError, IndexError, TypeError):
        return None
