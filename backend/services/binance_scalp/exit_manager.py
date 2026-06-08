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
    return max(_review_trigger_sec(econ) * 4, 1200)


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
                entry, state, max_fav, max_adv, session_low, stale_review_count,
                tuple(review_lows), track.setup_name, track.setup_context,
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
                entry, state, max_fav, max_adv, session_low, stale_review_count,
                tuple(review_lows), track.setup_name, track.setup_context,
            ),
        )

    if profit_hit:
        return _sell(STATE_OPEN, "profit_target_met", EXIT_NET_PROFIT_TARGET)

    hard = _max_hold_hard_sec(econ)
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
    momentum_rising = mom.bid_change_15s > 0 or mom.mid_change_15s > 0
    below_entry = executable_net_pct < 0
    recovering = recovery >= RECOVERY_MIN_PCT and bid > session_low

    if not spread_ok and not (invalidated and momentum_negative and below_entry):
        return _hold(STATE_RECOVERY_HOLD, "spread_too_wide_for_exit_wait")

    if momentum_rising and (fav >= 0 or recovering):
        return _hold(STATE_HEALTHY_HOLD, "momentum_rising_below_target")

    if below_entry and (recovering or hl or momentum_rising):
        return _hold(STATE_RECOVERY_HOLD, "red_but_recovery_signs")

    if invalidated and below_entry and momentum_negative and not recovering:
        return _sell(EXIT_SETUP_INVALIDATED, inv_reason or "setup_invalidated", EXIT_SETUP_INVALIDATED)

    if below_entry and momentum_negative and not hl and not recovering and stale_review_count >= 1:
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
