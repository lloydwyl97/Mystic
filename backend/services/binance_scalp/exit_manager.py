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

DECISION_HOLD = "HOLD"
DECISION_SELL = "SELL"

RECOVERY_MIN_PCT = 0.00012
HIGHER_LOWS_MIN_REVIEWS = 2


def _review_interval_sec() -> int:
    return int(os.getenv("SCALP_REVIEW_INTERVAL_SEC", "30"))


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


def _scratch_min_hold_sec() -> int:
    return int(os.getenv("SCALP_SCRATCH_MIN_HOLD_SEC", "180"))


def _scratch_flat_upper_pct() -> float:
    """Executable net at or below this (flat / tiny green that missed target)."""
    return float(os.getenv("SCALP_SCRATCH_FLAT_UPPER_PCT", "0.0002"))


def _scratch_progress_frac() -> float:
    return float(os.getenv("SCALP_SCRATCH_PROGRESS_FRAC", "0.35"))


def _scratch_deep_loss_pct() -> float:
    return float(os.getenv("SCALP_SCRATCH_DEEP_LOSS_PCT", "-0.006"))


def _scratch_min_reviews() -> int:
    return int(os.getenv("SCALP_SCRATCH_MIN_REVIEWS", "1"))


def _stall_exit_hold_frac() -> float:
    return float(os.getenv("SCALP_STALL_EXIT_HOLD_FRAC", "0.75"))


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
) -> tuple[bool, str]:
    """Exit flat/slightly-negative scalps that stall before hard max-hold."""
    if hold_sec < _scratch_min_hold_sec() or hold_sec >= hard:
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
    min_reviews = _scratch_min_reviews()
    trigger = _review_trigger_sec(econ)
    if hold_sec >= trigger and stale_review_count >= min_reviews:
        return True, "stalled_no_progress_review_scratch"
    if hold_sec >= _scratch_min_hold_sec() + 90 and stale_review_count >= min_reviews:
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
    if stale_review_count >= _scratch_min_reviews():
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
    )
    if scratch:
        target_progress = econ.net_profit_target_pct * _scratch_progress_frac()
        diag_base["scratch_trigger_detail"] = scratch_reason
        diag_base["scratch_target_progress_pct"] = target_progress
        diag_base["scratch_min_reviews"] = _scratch_min_reviews()
        diag_base["scratch_min_hold_sec"] = _scratch_min_hold_sec()
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

    if (
        stale_review_count >= _scratch_min_reviews()
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

    if below_entry and momentum_stalled and not hl and not meaningful_rec and stale_review_count >= 1:
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
