"""
Controlled-risk DAY bracket exits — shared by paper and live via portfolio_engine.

Engine-managed sells: profit target, stop loss, time stop, trailing protection,
strategy-specific invalidation, plus catastrophic EXTREME_PROTECTION.
Net-profit exit remains one path among several — never the only sell path.
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass
from typing import Any

from backend.config.trading_economics import ESTIMATED_ROUNDTRIP_COST, MIN_NET_PROFIT_TO_SELL, min_net_profit_for_symbol
from backend.services.ai_regime_validation import blend_by_scalar, get_regime_validated_scalar
from backend.services.day_trade_thesis import (
    DAY_4H_BUNDLE_MISSING,
    EXIT_DAY_4H_STRUCTURE_BREAK,
    EXIT_DAY_RISK_FLOOR,
    EXIT_EXTREME_PROTECTION,
    EXIT_NET_PROFIT,
    EXIT_STOP_LOSS,
    EXIT_THESIS_INVALIDATION,
    EXIT_THESIS_WARNING,
    EXIT_TRAILING_STOP,
    SETUP_VWAP_REVERSION,
    day_4h_structure_snapshot,
    evaluate_extreme_protection,
    evaluate_thesis_exit,
    resolve_day_risk_floor_price,
    thesis_invalidated_live,
)

EXIT_VOLATILITY_STOP = "VOLATILITY_STOP_EXIT"
EXIT_TIME_STOP = "TIME_STOP_EXIT"
EXIT_FAILED_RECLAIM = "FAILED_RECLAIM_EXIT"
EXIT_STALL = "STALL_EXIT"
EXIT_STALL_DEAD = "STALL_EXIT_DEAD_NO_MFE"
EXIT_GIVEBACK = "GIVEBACK_EXIT"
EXIT_PROGRESS_DECAY = "PROGRESS_DECAY_EXIT"
EXIT_ADAPTIVE_LOSS = "ADAPTIVE_LOSS_EXIT"
EXIT_PATH_EXECUTABLE_PROFIT = "PATH_EXECUTABLE_PROFIT"
DAY_PATH_AWARE_POLICY = "day_path_aware_v1"
HOLD_4H_RISE = "PATH_AWARE_HOLD_4H_RISE"
HOLD_4H_MISSING = "PATH_AWARE_HOLD_4H_MISSING"
HOLD_4H_UNDECIDED = "PATH_AWARE_HOLD_4H_UNDECIDED"

# Reasons that are allowed to full-flatten a DAY position. Anything else holds.
DAY_FULL_FLATTEN_REASONS = frozenset(
    {
        EXIT_DAY_4H_STRUCTURE_BREAK,
        EXIT_DAY_RISK_FLOOR,
        EXIT_EXTREME_PROTECTION,
        EXIT_TRAILING_STOP,
        EXIT_GIVEBACK,
        EXIT_STALL_DEAD,
    }
)

logger = logging.getLogger(__name__)
_exit_policy_logged = False


def _path_aware_exit_enabled() -> bool:
    """DAY exit policy: hold the 4H thesis; sell only on structure break or extreme.

    When enabled (production default), the leftover net-profit / TP1 clips
    stay unreachable. An intact 4H hold still sells on: extreme protection,
    risk floor, an activated trailing stop (price pulled back through the
    ratchet), giveback-to-red, stall, or a later 4H structure break.
    Disabling path-aware restores the full leftover ladder.
    """
    raw = os.getenv("DAY_PATH_AWARE_EXIT")
    enabled = (raw if raw is not None else "true").strip().lower() in {"1", "true", "yes", "on"}
    global _exit_policy_logged
    if not _exit_policy_logged:
        _exit_policy_logged = True
        logger.warning(
            "DAY_EXIT_POLICY_RESOLVED path_aware=%s sell_reasons=%s source=%s",
            enabled,
            sorted(DAY_FULL_FLATTEN_REASONS) if enabled else "legacy_ladder",
            "env" if raw is not None else "code_default",
        )
    return enabled


def _path_min_executable_net_pct() -> float:
    try:
        return float(os.getenv("DAY_PATH_MIN_EXECUTABLE_NET_PCT", "0.0001"))
    except (TypeError, ValueError):
        return 0.0001


def _evaluate_path_aware_exit(
    *,
    position: Any,
    current_price: float,
    net_pnl_pct: float,
    hold_minutes: float,
    coin_profile: dict[str, Any],
    bundle: dict[str, Any] | None,
    entry: float,
    atr_pct: float,
) -> dict[str, Any]:
    snap4 = day_4h_structure_snapshot(bundle)
    extreme = evaluate_extreme_protection(
        entry_price=entry,
        mark=current_price,
        net_pnl_pct=net_pnl_pct,
        atr_pct=atr_pct,
        bundle=bundle,
    )
    if str(extreme.get("action")) == "sell":
        return {
            "action": "sell",
            "reason": EXIT_EXTREME_PROTECTION,
            "net_pnl_pct": net_pnl_pct,
            "hold_minutes": hold_minutes,
            "detail": "extreme_protection",
            **snap4,
            "extreme_protection_fired": True,
        }
    risk_floor_price = resolve_day_risk_floor_price(
        entry_price=entry,
        thesis_invalid_level=float(getattr(position, "thesis_invalid_level", 0.0) or 0.0),
        prior_4h_low=float(snap4.get("prior_4h_low") or 0.0),
        atr_pct=atr_pct,
    )
    base = {
        "net_pnl_pct": net_pnl_pct,
        "hold_minutes": hold_minutes,
        **snap4,
        "risk_floor_price": risk_floor_price,
        "extreme_protection_fired": False,
    }

    # Bounded adverse excursion. Must be checked before the 4H hold below, or the
    # hold makes it unreachable and the position can bleed unbounded for up to a
    # full 4H bar waiting for a close that may never come at a tolerable price.
    if risk_floor_price > 0 and current_price <= risk_floor_price:
        return {
            "action": "sell",
            "reason": EXIT_DAY_RISK_FLOOR,
            "detail": f"mark_at_or_below_risk_floor={risk_floor_price:.8f}",
            **base,
        }

    giveback_on_hold = os.getenv("DAY_GIVEBACK_ON_4H_HOLD", "true").lower() in ("1", "true", "yes", "on")
    if giveback_on_hold and snap4["htf_4h_rise_intact"]:
        gb = evaluate_giveback_exit(
            entry_price=entry,
            highest_price=float(getattr(position, "highest_price", entry) or entry),
            net_pnl_pct=net_pnl_pct,
            hold_minutes=hold_minutes,
            position=position,
        )
        if gb is not None:
            return {**gb, **base, "reason": EXIT_GIVEBACK}

    stall_on_hold = os.getenv("DAY_STALL_ON_4H_HOLD", "true").lower() in ("1", "true", "yes", "on")
    if stall_on_hold and snap4["htf_4h_rise_intact"]:
        stall = evaluate_stall_exit(
            entry_price=entry,
            highest_price=float(getattr(position, "highest_price", entry) or entry),
            net_pnl_pct=net_pnl_pct,
            hold_minutes=hold_minutes,
            max_hold_min=effective_max_hold_min(position, coin_profile),
            current_price=current_price,
            lowest_price=float(getattr(position, "lowest_price", 0.0) or 0.0),
        )
        if stall is not None and str(stall.get("action") or "") == "sell":
            return {**stall, **base, "reason": EXIT_STALL_DEAD}

    # Existing trail: once the high-water ratchet is armed, a pullback through
    # it is deterioration — not a "green enough" clip.
    trail_pct = float(getattr(position, "trail_pct", 0) or coin_profile.get("trail") or 0.005)
    highest = float(getattr(position, "highest_price", entry) or entry)
    trail = float(getattr(position, "trailing_stop_price", 0) or 0)
    if trail > 0 and highest >= entry * (1.0 + trail_pct) - 1e-12 and current_price <= trail:
        return {
            "action": "sell",
            "reason": EXIT_TRAILING_STOP,
            "detail": f"trail={trail:.8f}",
            **base,
        }

    # 4H still rising: never TP1 / leftover net-profit / time-stop.
    if snap4["htf_4h_rise_intact"]:
        return {
            "action": "hold",
            "reason": HOLD_4H_RISE,
            "detail": "4h_breakout_intact",
            **base,
        }

    # The only structural DAY sell. Never a 0.4% / 0.01% scalp clip.
    if snap4["htf_4h_rise_broken"]:
        return {
            "action": "sell",
            "reason": EXIT_DAY_4H_STRUCTURE_BREAK,
            "detail": "4h_close_below_prior_4h_low",
            **base,
        }

    # No 4H evidence is not permission to scalp-clip. Extreme protection above
    # is the only exit left in this state.
    if snap4["4h_bundle_missing"]:
        return {
            "action": "hold",
            "reason": HOLD_4H_MISSING,
            "detail": "4h_bundle_missing_no_scalp_clip",
            "diagnostic": DAY_4H_BUNDLE_MISSING,
            **base,
        }
    return {
        "action": "hold",
        "reason": HOLD_4H_UNDECIDED,
        "detail": "4h_not_intact_not_broken_no_scalp_clip",
        **base,
    }


PROGRESS_DECAY_HOLD_TOO_YOUNG = "PROGRESS_DECAY_HOLD_TOO_YOUNG"
PROGRESS_DECAY_GREEN = "PROGRESS_DECAY_GREEN"
PROGRESS_DECAY_NO_ARM_HISTORY = "PROGRESS_DECAY_NO_ARM_HISTORY"
PROGRESS_DECAY_NORMAL_PACE = "PROGRESS_DECAY_NORMAL_PACE"

# Telemetry hold reasons (action=hold; never force-sell).
STALL_HOLD_TOO_YOUNG = "STALL_HOLD_TOO_YOUNG"
STALL_HOLD_NOT_RED = "STALL_HOLD_NOT_RED"
STALL_HOLD_MFE_TOO_HIGH = "STALL_HOLD_MFE_TOO_HIGH"
STALL_HOLD_RECOVERY_PRESENT = "STALL_HOLD_RECOVERY_PRESENT"
STALL_HOLD_FLAT_NOT_DEAD = "STALL_HOLD_FLAT_NOT_DEAD"
STALL_HOLD_NET_PROFIT_ELIGIBLE = "STALL_HOLD_NET_PROFIT_ELIGIBLE"


def _bull_hold_extension_min() -> int:
    """Extra minutes granted to bull-regime positions beyond the coin profile ceiling."""
    return int(os.getenv("DAY_BULL_HOLD_EXTENSION_MIN", "120"))


def _bull_trail_multiplier() -> float:
    """Trail-pct multiplier applied when a bull position is trending strongly."""
    return float(os.getenv("DAY_BULL_TRAIL_MULTIPLIER", "2.0"))


def _bull_trail_mfe_threshold() -> float:
    """Minimum MFE fraction before bull trail widening activates (default 1.5%)."""
    return float(os.getenv("DAY_BULL_TRAIL_MFE_THRESHOLD", "0.015"))


def effective_max_hold_min(position: Any, coin_profile: dict[str, Any] | None = None) -> int:
    """Session ceiling: prefer current profile when stamped hold is shorter (day-trade upgrade).

    Bull-regime positions receive an extra DAY_BULL_HOLD_EXTENSION_MIN (default +120 min)
    so a trending position is not timed out before its target can be reached.
    """
    profile_hold = int((coin_profile or {}).get("max_hold_min") or 0)
    stamped = int(getattr(position, "max_hold_min", 0) or 0)
    if profile_hold > 0 and stamped > 0:
        base = max(stamped, profile_hold)
    else:
        base = stamped or profile_hold or 300
    regime = str(getattr(position, "day_route_regime_at_entry", "") or "").lower()
    if regime == "bull":
        scalar, _ = get_regime_validated_scalar(regime)
        base += int(round(_bull_hold_extension_min() * scalar))
    return base


ALLOWED_DAY_EXIT_REASONS = frozenset(
    {
        EXIT_NET_PROFIT,
        EXIT_PATH_EXECUTABLE_PROFIT,
        EXIT_VOLATILITY_STOP,
        EXIT_TIME_STOP,
        EXIT_STALL,
        EXIT_STALL_DEAD,
        EXIT_GIVEBACK,
        EXIT_PROGRESS_DECAY,
        EXIT_FAILED_RECLAIM,
        EXIT_EXTREME_PROTECTION,
        EXIT_THESIS_INVALIDATION,
        EXIT_STOP_LOSS,
        EXIT_TRAILING_STOP,
        "MANUAL_EXIT",
        "LEGACY_CLEANUP_EXIT",
        "LEGACY_INVENTORY_CLEANUP_EXIT",
        "ADMIN_CLEAR",
        "ALLWEATHER_ATR_STOP_EXIT",
        "ALLWEATHER_ATR_TARGET_EXIT",
        "ALLWEATHER_TIME_STOP_EXIT",
        EXIT_DAY_4H_STRUCTURE_BREAK,
        EXIT_DAY_RISK_FLOOR,
        "EMERGENCY_FLATTEN",
        "RESTART_FLATTEN",
    }
)

ENGINE_RISK_EXIT_PREFIXES = (
    EXIT_STOP_LOSS,
    EXIT_VOLATILITY_STOP,
    EXIT_TIME_STOP,
    EXIT_STALL,
    EXIT_STALL_DEAD,
    EXIT_GIVEBACK,
    EXIT_PROGRESS_DECAY,
    EXIT_TRAILING_STOP,
    EXIT_THESIS_INVALIDATION,
    EXIT_FAILED_RECLAIM,
    EXIT_EXTREME_PROTECTION,
    "ALLWEATHER",
)


def _stall_exit_enabled() -> bool:
    return os.getenv("DAY_STALL_EXIT_ENABLED", "true").lower() in ("1", "true", "yes", "on")


def _stall_min_hold_min() -> float:
    # Day-trade style: do not cut "dead" holds until the idea has had hours.
    return float(os.getenv("DAY_STALL_MIN_HOLD_MIN", "120"))


def _stall_max_mfe_pct() -> float:
    """Max mark MFE (fraction) allowed for a stall cut. Default 0.50%.

    Raised from 0.20% — positions that reached 0.20-0.50% favorable and came back
    are not dead inventory; they may still resolve toward target on the next leg.
    Only truly flat/dead positions (< 0.50% MFE ever seen) should be stall-cut.
    """
    return float(os.getenv("DAY_STALL_MAX_MFE_PCT", "0.0050"))


def _stall_min_adverse_pct() -> float:
    """Min MAE (fraction below entry) required to confirm a dead/worsening stall cut.

    Flat red trades with tiny adverse are not force-sold; TIME_STOP still owns the ceiling.
    """
    return float(os.getenv("DAY_STALL_MIN_ADVERSE_PCT", "0.0025"))


def _stall_recovery_pct() -> float:
    """If mark has reclaimed within this fraction of entry (from below), treat as recovery."""
    return float(os.getenv("DAY_STALL_RECOVERY_PCT", "0.0010"))


def evaluate_stall_exit(
    *,
    entry_price: float,
    highest_price: float,
    net_pnl_pct: float,
    hold_minutes: float,
    max_hold_min: int,
    current_price: float = 0.0,
    lowest_price: float = 0.0,
) -> dict[str, Any] | None:
    """
    Cut only *dead/worsening* DAY holds before hard time-stop.

    Low-MFE alone after 120m is not enough — Ocean evidence showed that pattern
    was the default losing disposal path. Require red + low MFE + confirmed
    adverse deterioration, and hold when recovery/improvement is present.

    Exit-only — no entry/ranking changes. Returns None to continue the exit
    chain, or a sell/hold telemetry dict (hold reasons never force-sell).
    """
    if not _stall_exit_enabled():
        return None
    entry = float(entry_price or 0.0)
    if entry <= 0:
        return None
    stall_min = _stall_min_hold_min()
    mark = float(current_price or 0.0)
    highest = float(highest_price or entry)
    lowest = float(lowest_price or 0.0)
    # Unit tests / older call sites may omit mark; approximate from fee-aware net.
    if mark <= 0 and net_pnl_pct is not None:
        mark = entry * (1.0 + float(net_pnl_pct))
    if lowest <= 0:
        lowest = min(mark, entry) if mark > 0 else entry
    mfe_pct = max(0.0, (highest - entry) / entry)
    mae_pct = max(0.0, (entry - lowest) / entry) if entry > 0 else 0.0
    max_mfe = _stall_max_mfe_pct()
    min_adverse = _stall_min_adverse_pct()
    recovery_band = _stall_recovery_pct()

    def _hold(reason: str, **extra: Any) -> dict[str, Any]:
        payload = {
            "action": "hold",
            "reason": reason,
            "net_pnl_pct": net_pnl_pct,
            "hold_minutes": hold_minutes,
            "mfe_pct": round(mfe_pct, 6),
            "mae_pct": round(mae_pct, 6),
            "detail": (f"stall_min={stall_min:.0f}m mfe={mfe_pct:.6f} mae={mae_pct:.6f} max_mfe={max_mfe:.6f} min_adverse={min_adverse:.6f}"),
        }
        payload.update(extra)
        return payload

    if hold_minutes < stall_min:
        return _hold(STALL_HOLD_TOO_YOUNG)
    # Never replace the hard ceiling — time-stop still owns max_hold.
    if hold_minutes + 1e-9 >= float(max_hold_min):
        return None
    # Never scratch greens / net-profit-eligible paths — profit exits own those.
    if net_pnl_pct + 1e-12 >= float(MIN_NET_PROFIT_TO_SELL):
        return _hold(STALL_HOLD_NET_PROFIT_ELIGIBLE)
    if net_pnl_pct >= 0.0:
        return _hold(STALL_HOLD_NOT_RED)
    if mfe_pct >= max_mfe:
        return _hold(STALL_HOLD_MFE_TOO_HIGH)

    # Never deteriorated enough — flat/chop inventory, not a dead disposal.
    if mae_pct < min_adverse:
        return _hold(STALL_HOLD_FLAT_NOT_DEAD)

    # Recovery / improvement after a real adverse print: reclaim toward entry
    # or lift off the low. Requires prior MAE so near-entry flats stay FLAT_NOT_DEAD.
    if mark > 0:
        near_entry_reclaim = mark >= entry * (1.0 - recovery_band)
        lifted_from_low = lowest > 0 and mark >= lowest * (1.0 + recovery_band)
        improving_vs_mid = lowest > 0 and highest > lowest and mark >= (lowest + entry) / 2.0
        if near_entry_reclaim or (lifted_from_low and improving_vs_mid):
            return _hold(
                STALL_HOLD_RECOVERY_PRESENT,
                near_entry_reclaim=near_entry_reclaim,
                lifted_from_low=lifted_from_low,
            )

    # Dead/worsening confirmation: adverse excursion still in force at the mark.
    still_adverse = mark > 0 and mark <= entry * (1.0 - min_adverse * 0.5)
    if not still_adverse:
        return _hold(STALL_HOLD_FLAT_NOT_DEAD)

    return {
        "action": "sell",
        "reason": EXIT_STALL_DEAD,
        "net_pnl_pct": net_pnl_pct,
        "hold_minutes": hold_minutes,
        "mfe_pct": round(mfe_pct, 6),
        "mae_pct": round(mae_pct, 6),
        "detail": (f"stall_min={stall_min:.0f}m mfe={mfe_pct:.6f} mae={mae_pct:.6f} max_mfe={max_mfe:.6f} min_adverse={min_adverse:.6f}"),
    }


def _giveback_exit_enabled() -> bool:
    return os.getenv("DAY_GIVEBACK_EXIT_ENABLED", "true").lower() in ("1", "true", "yes", "on")


def _giveback_min_hold_min() -> float:
    # Avoid 3-minute noise cuts; require a real day-trade development window.
    return float(os.getenv("DAY_GIVEBACK_MIN_HOLD_MIN", "20"))


def _giveback_min_mfe_pct() -> float:
    """Min favorable excursion (fraction) that must have been reached before a reversal counts as a giveback."""
    return float(os.getenv("DAY_GIVEBACK_MIN_MFE_PCT", "0.0025"))


def _giveback_trigger_pnl_pct() -> float:
    """Net pnl pct (negative fraction) that, once breached after MFE was reached, triggers an early cut."""
    return float(os.getenv("DAY_GIVEBACK_TRIGGER_PNL_PCT", "-0.0015"))


def evaluate_giveback_exit(
    *,
    entry_price: float,
    highest_price: float,
    net_pnl_pct: float,
    hold_minutes: float,
    position: Any = None,
) -> dict[str, Any] | None:
    """
    Cut DAY holds that reached meaningful favorable excursion and then reversed
    back to net-negative, instead of waiting for the stall floor to book a
    deeper loss.

    Defaults require ~20m development and a clearer MFE/giveback so 1–3m noise
    does not churn day trades. Exit-only — no entry/ranking changes.

    Adaptive MAE handling (item p7): when `position` is supplied and its
    (symbol, setup, regime) arm has enough losing-trade history, the arm's
    own historical MAE-among-losers percentile (day_adaptive_targets.py)
    replaces the fixed global -0.15% trigger — a "typical" reversal for a
    volatile arm may be much bigger than -0.15% (fires too eagerly on
    normal noise), while a calm arm's typical reversal may be much smaller
    (the fixed trigger waits too long). Falls back to the fixed constant
    whenever there isn't enough real history.

    HoldEV tightening (item p8 promotion): once the base trigger above is
    resolved, hold_ev_engine's combined momentum/orderflow/excursion/
    progress score can shrink (never widen) its magnitude toward breakeven
    when it already disfavors continuing to hold — see
    hold_ev_giveback_tighten_factor's docstring. Neutral (no effect) when
    HoldEV has insufficient data.
    """
    if not _giveback_exit_enabled():
        return None
    entry = float(entry_price or 0.0)
    if entry <= 0:
        return None
    if hold_minutes < _giveback_min_hold_min():
        return None
    highest = float(highest_price or entry)
    mfe_pct = max(0.0, (highest - entry) / entry)
    if mfe_pct < _giveback_min_mfe_pct():
        return None
    trigger = _giveback_trigger_pnl_pct()
    trigger_source = "fixed_default"
    if position is not None:
        try:
            from backend.services.day_adaptive_targets import adaptive_giveback_trigger_for_arm

            _adaptive = adaptive_giveback_trigger_for_arm(
                str(getattr(position, "symbol", "") or ""),
                str(getattr(position, "entry_thesis", "") or ""),
                str(getattr(position, "day_route_regime_at_entry", "") or ""),
            )
            if _adaptive.get("source") not in ("insufficient_data", "disabled"):
                trigger = float(_adaptive["trigger_pct"])
                trigger_source = str(_adaptive["source"])
        except Exception:
            pass
    hev_factor, hev_detail = _hold_ev_tighten(position, entry_price=entry, net_pnl_pct=net_pnl_pct, hold_minutes=hold_minutes)
    trigger *= hev_factor
    if net_pnl_pct + 1e-12 > trigger:
        return None
    return {
        "action": "sell",
        "reason": EXIT_GIVEBACK,
        "net_pnl_pct": net_pnl_pct,
        "hold_minutes": hold_minutes,
        "detail": f"mfe={mfe_pct:.6f} trigger={trigger:.6f} trigger_source={trigger_source} {hev_detail}",
    }


def _hold_ev_tighten(position: Any, *, entry_price: float, net_pnl_pct: float, hold_minutes: float) -> tuple[float, str]:
    """Shared helper: best-effort HoldEV giveback-tighten factor + detail
    string. Returns (1.0, "hev=unavailable") on any failure or missing
    position — always neutral, never blocks the caller."""
    if position is None or entry_price <= 0:
        return 1.0, "hev=unavailable"
    try:
        from backend.services.hold_ev_engine import hold_ev_for_position, hold_ev_giveback_tighten_factor

        approx_current = entry_price * (1.0 + float(net_pnl_pct or 0.0))
        _hev = hold_ev_for_position(position, current_price=approx_current, hold_minutes=hold_minutes)
        factor = hold_ev_giveback_tighten_factor(_hev.hold_ev_score, _hev.confidence)
        return factor, f"hev_score={_hev.hold_ev_score:.3f} hev_factor={factor:.3f}"
    except Exception:
        return 1.0, "hev=unavailable"


def _adaptive_loss_exit_enabled() -> bool:
    return os.getenv("DAY_ADAPTIVE_LOSS_EXIT_ENABLED", "true").strip().lower() in ("1", "true", "yes", "on")


def _adaptive_loss_min_hold_min() -> float:
    # Avoid 1-2 minute entry noise; require the position to have had a real
    # chance to move before treating adverse excursion as informative.
    return float(os.getenv("DAY_ADAPTIVE_LOSS_MIN_HOLD_MIN", "10"))


def evaluate_adaptive_loss_exit(
    *,
    entry_price: float,
    net_pnl_pct: float,
    hold_minutes: float,
    position: Any = None,
) -> dict[str, Any] | None:
    """Item p7 gap-closure: adaptive MAE-distribution-informed early exit for
    STRAIGHT losers — positions that go against entry WITHOUT ever building
    the meaningful favorable excursion that ``evaluate_giveback_exit``
    requires (its ``mfe_pct >= _giveback_min_mfe_pct()`` gate). Before this,
    a straight loser had NO adaptive MAE check at all between entry and the
    fixed per-coin ``stop_price``/``thesis_invalid_level`` (~1% typical) —
    the fixed stop was the only signal, contrary to item 7's intent that
    the distribution/EV-based normalcy check become PRIMARY for loss
    handling, with only the catastrophic hard stop remaining fixed.

    When the (symbol, setup, regime) arm has enough real LOSING-trade
    history, an adverse excursion beyond the arm's own losing-MAE
    percentile (default p75 — "worse than 75% of this arm's own historical
    losers already were") is treated as abnormal for this specific arm and
    exited early. This can only ever fire EARLIER/TIGHTER than the fixed
    stop_loss (never wider) — the fixed stop_loss and catastrophic extreme
    protection remain fully in place downstream as the mechanical backstop
    regardless of this check's outcome, and this check itself never
    overrides them, only pre-empts them when the arm's own real data says
    the current excursion is already abnormal.

    Honest fallback: with no position, insufficient arm history, or a
    cross-symbol-only pool, returns None — no early exit, identical to
    behavior before this item existed.
    """
    if not _adaptive_loss_exit_enabled() or position is None:
        return None
    entry = float(entry_price or 0.0)
    if entry <= 0 or hold_minutes < _adaptive_loss_min_hold_min():
        return None
    if net_pnl_pct >= 0.0:
        return None
    try:
        from backend.services.mfe_mae_distribution_learner import get_mae_distribution

        dist = get_mae_distribution(
            str(getattr(position, "symbol", "") or ""),
            "day",
            db_path=_db_path(),
        )
    except Exception:
        return None
    if dist.confidence_status == "insufficient_data" or dist.stratum_used == "strategy_cross_symbol":
        return None
    percentile_key = os.getenv("DAY_ADAPTIVE_LOSS_EXIT_PERCENTILE", "p75")
    abnormal_mae = float(dist.percentiles.get(percentile_key, 0.0))
    if abnormal_mae <= 0.0:
        return None
    current_mae = max(0.0, -net_pnl_pct)
    if current_mae + 1e-12 < abnormal_mae:
        return None
    return {
        "action": "sell",
        "reason": EXIT_ADAPTIVE_LOSS,
        "net_pnl_pct": net_pnl_pct,
        "hold_minutes": hold_minutes,
        "detail": f"mae={current_mae:.6f} abnormal_threshold={abnormal_mae:.6f} stratum={dist.stratum_used} n_obs={dist.n_obs}",
    }


def _progress_decay_enabled() -> bool:
    return os.getenv("DAY_PROGRESS_DECAY_EXIT_ENABLED", "true").strip().lower() in ("1", "true", "yes", "on")


def _progress_decay_min_hold_min() -> float:
    return float(os.getenv("DAY_PROGRESS_DECAY_MIN_HOLD_MIN", "30"))


def _progress_decay_min_ratio() -> float:
    """A position paced below this fraction of its arm's typical
    same-duration winner MFE pace is "decaying", not just quiet."""
    return float(os.getenv("DAY_PROGRESS_DECAY_MIN_RATIO", "0.35"))


def evaluate_progress_decay_exit(
    *,
    entry_price: float,
    highest_price: float,
    net_pnl_pct: float,
    hold_minutes: float,
    position: Any = None,
) -> dict[str, Any] | None:
    """Item p9: progress_rate = MFE / holding_time continuation decay.

    Replaces "wait for the fixed time-stop, then cut" with "if this red
    position's favorable-excursion pace is far below what winners on this
    exact arm typically show by this point in the hold, that's a genuine
    decay signal — independent of the catastrophic max-hold failsafe, which
    is untouched and still the hard ceiling."

    Never fires on a green position (profit exits own those), never fires
    before `DAY_PROGRESS_DECAY_MIN_HOLD_MIN`, and never fires without real
    per-arm history to compare against (no `position`, or the arm's MFE
    distribution for this hold-time bucket is insufficient_data) — there is
    no fixed-global fallback pace to guess at, unlike the trail/target/
    giveback adaptations above, because an invented "typical pace" would
    itself be exactly the kind of unvalidated opinion the architecture rule
    forbids injecting as a new blocker.
    """
    if not _progress_decay_enabled():
        return None
    entry = float(entry_price or 0.0)
    if entry <= 0 or hold_minutes <= 0:
        return None
    if net_pnl_pct is not None and net_pnl_pct >= 0.0:
        return None
    if hold_minutes < _progress_decay_min_hold_min():
        return None
    if position is None:
        return None
    highest = float(highest_price or entry)
    mfe_pct = max(0.0, (highest - entry) / entry)
    progress_rate = mfe_pct / hold_minutes

    try:
        from backend.services.mfe_mae_distribution_learner import get_mfe_distribution, hold_time_bucket

        symbol = str(getattr(position, "symbol", "") or "")
        hb = hold_time_bucket(hold_minutes * 60.0, "day")
        dist = get_mfe_distribution(symbol, "day", hold_bucket_filter=hb, db_path=_db_path())
    except Exception:
        return None
    if dist.confidence_status == "insufficient_data" or dist.n_obs <= 0:
        return None
    if dist.stratum_used == "strategy_cross_symbol":
        # Conservative: this is a novel signal, not (yet) validated the way
        # the trail/target/giveback adaptations above are — only trust the
        # symbol's own real history, never a cross-symbol pool, to decide to
        # cut a trade early.
        return None
    typical_mfe = float(dist.percentiles.get("p50", 0.0))
    if typical_mfe <= 0:
        return None
    typical_pace = typical_mfe / hold_minutes
    if typical_pace <= 0:
        return None
    ratio = progress_rate / typical_pace
    if ratio > _progress_decay_min_ratio():
        return None

    return {
        "action": "sell",
        "reason": EXIT_PROGRESS_DECAY,
        "net_pnl_pct": net_pnl_pct,
        "hold_minutes": hold_minutes,
        "detail": (f"progress_rate={progress_rate:.8f} typical_pace={typical_pace:.8f} ratio={ratio:.4f} arm_n_obs={dist.n_obs} arm_stratum={dist.stratum_used}"),
    }


def _db_path() -> str:
    from backend.database_schema import DATABASE_PATH

    return DATABASE_PATH


@dataclass(frozen=True)
class ControlledExitConfig:
    """Replay/live bracket parameters."""

    enabled: bool = False
    profit_floor_pct: float = 0.004
    atr_stop_mult: float = 1.0
    time_stop_hours: float = 48.0
    max_loss_pct: float = 0.015
    failed_reclaim_hours: float = 6.0
    failed_reclaim_buffer_pct: float = 0.0015
    use_fill_based_gate: bool = True


def effective_stop_price(entry_price: float, stop_price: float, thesis_invalid_level: float) -> float:
    """Long stop: highest valid sub-entry level (tightest protective stop)."""
    entry = float(entry_price or 0.0)
    if entry <= 0:
        return 0.0
    candidates = [float(v) for v in (stop_price, thesis_invalid_level) if v and float(v) > 0 and float(v) < entry]
    return max(candidates) if candidates else 0.0


def effective_target_price(entry_price: float, tp1: float, thesis_target: float, position: Any = None, bundle: dict[str, Any] | None = None) -> float:
    """Long target: nearest profit objective above entry.

    When `position` is supplied and its (symbol, setup, regime) arm has
    enough winning-trade history, the arm's actual MFE p60 (see
    day_adaptive_targets.py) joins the candidate pool. Because the final
    choice is still `min()` across all candidates, this can only ever pull
    the target CLOSER to entry (take profit at what winners on this arm
    actually reach) — never push it further away. Optional and additive;
    omitting `position` reproduces the exact prior behavior.

    When `bundle` (the DAY MTF OHLCV bundle) is also supplied, item p6's
    ATR-grid expectancy-selected target (day_adaptive_targets.atr_grid_target_candidate)
    joins the same candidate pool under the same min()-only-tightens rule.
    """
    entry = float(entry_price or 0.0)
    if entry <= 0:
        return 0.0
    candidates = [float(v) for v in (tp1, thesis_target) if v and float(v) > entry]
    if position is not None:
        try:
            from backend.services.day_adaptive_targets import adaptive_target_pct_for_arm

            _adaptive = adaptive_target_pct_for_arm(
                str(getattr(position, "symbol", "") or ""),
                str(getattr(position, "entry_thesis", "") or ""),
                str(getattr(position, "day_route_regime_at_entry", "") or ""),
            )
            _pct = float(_adaptive.get("target_pct") or 0.0)
            if _pct > 0:
                candidates.append(entry * (1.0 + _pct))
        except Exception:
            pass
        if bundle:
            try:
                from backend.services.day_adaptive_targets import atr_grid_target_candidate
                from backend.services.day_feature_stack_v2 import atr_pct_multi_period

                rows_1h = bundle.get("1h") if isinstance(bundle, dict) else None
                current_atr_pct = float(atr_pct_multi_period(rows_1h).get(14, 0.0)) if rows_1h else 0.0
                if current_atr_pct > 0:
                    _atr_grid = atr_grid_target_candidate(str(getattr(position, "symbol", "") or ""), current_atr_pct)
                    _atr_pct = float(_atr_grid.get("target_pct") or 0.0)
                    if _atr_pct > 0:
                        candidates.append(entry * (1.0 + _atr_pct))
            except Exception:
                pass
    return min(candidates) if candidates else 0.0


def stamp_open_position_exit_metadata(
    position: Any,
    *,
    fill_price: float,
    stop_price: float,
    tp1_price: float,
    tp2_price: float,
    coin_profile: dict[str, Any],
    thesis_invalid_level: float = 0.0,
    thesis_target_level: float = 0.0,
) -> None:
    """Stamp full exit metadata on a new OpenPosition at BUY time (paper/live shared)."""
    entry = float(fill_price or 0.0)
    if entry <= 0:
        return

    profile = coin_profile or {}
    sl_pct = float(profile.get("sl") or 0.010)
    trail_pct = float(profile.get("trail") or 0.005)
    max_hold = int(profile.get("max_hold_min") or 75)

    invalid = float(thesis_invalid_level or getattr(position, "thesis_invalid_level", 0.0) or 0.0)
    target = float(thesis_target_level or getattr(position, "thesis_target_level", 0.0) or 0.0)
    stop = float(stop_price or getattr(position, "stop_price", 0.0) or 0.0)

    if invalid > 0 and invalid < entry:
        position.thesis_invalid_level = invalid
    elif not invalid:
        position.thesis_invalid_level = entry * (1.0 - sl_pct)

    if stop <= 0 or stop >= entry:
        stop = float(position.thesis_invalid_level or entry * (1.0 - sl_pct))
    else:
        stop = max(stop, float(position.thesis_invalid_level or 0.0))
    position.stop_price = stop

    if target > entry:
        position.thesis_target_level = target
        position.take_profit_1_price = min(float(tp1_price or 0.0), target) if float(tp1_price or 0.0) > entry else target
    else:
        position.take_profit_1_price = float(tp1_price or entry * (1.0 + float(profile.get("tp") or 0.014)))

    position.take_profit_2_price = float(tp2_price or position.take_profit_1_price * 1.007)
    position.trailing_stop_price = entry * (1.0 - sl_pct)
    position.trail_pct = trail_pct
    position.max_hold_min = max_hold


def backfill_position_exit_metadata(position: Any, coin_profile: dict[str, Any]) -> list[str]:
    """Fill missing stop/target/trail/max-hold metadata from coin profile + thesis."""
    added: list[str] = []
    entry = float(getattr(position, "entry_price", 0.0) or 0.0)
    if entry <= 0:
        return added

    sl_pct = float(coin_profile.get("sl") or 0.010)
    tp_pct = float(coin_profile.get("tp") or 0.014)
    trail_pct = float(coin_profile.get("trail") or 0.005)
    max_hold_min = int(coin_profile.get("max_hold_min") or 75)

    stop = float(getattr(position, "stop_price", 0.0) or 0.0)
    if stop <= 0 or stop >= entry:
        position.stop_price = entry * (1.0 - sl_pct)
        added.append(f"stop_price={position.stop_price:.8f}")

    invalid = float(getattr(position, "thesis_invalid_level", 0.0) or 0.0)
    if invalid <= 0 or invalid >= entry:
        position.thesis_invalid_level = float(position.stop_price)
        added.append(f"thesis_invalid_level={position.thesis_invalid_level:.8f}")

    tp1 = float(getattr(position, "take_profit_1_price", 0.0) or 0.0)
    if tp1 <= 0 or tp1 <= entry:
        position.take_profit_1_price = entry * (1.0 + tp_pct)
        added.append(f"take_profit_1_price={position.take_profit_1_price:.8f}")

    tp2 = float(getattr(position, "take_profit_2_price", 0.0) or 0.0)
    if tp2 <= 0 or tp2 <= entry:
        position.take_profit_2_price = entry * (1.0 + tp_pct * 2.0)
        added.append(f"take_profit_2_price={position.take_profit_2_price:.8f}")

    trail = getattr(position, "trailing_stop_price", None)
    if trail is None or float(trail or 0.0) <= 0:
        position.trailing_stop_price = entry * (1.0 - sl_pct)
        added.append(f"trailing_stop_price={position.trailing_stop_price:.8f}")

    stamped_hold = int(getattr(position, "max_hold_min", 0) or 0)
    if stamped_hold <= 0 or stamped_hold < max_hold_min:
        position.max_hold_min = max_hold_min
        added.append(f"max_hold_min={max_hold_min}")

    if trail_pct and not getattr(position, "trail_pct", 0):
        position.trail_pct = trail_pct
        added.append(f"trail_pct={trail_pct}")

    return added


def _break_even_trigger_pct() -> float:
    """MFE fraction that must be reached before break-even ratchet activates.

    Default 0.30% is well above round-trip cost (~0.20%). Below trigger, no
    change. Above trigger, stop is lifted to entry_price + offset.
    """
    return float(os.getenv("DAY_BREAK_EVEN_TRIGGER_PCT", "0.0030"))


def _break_even_offset_pct() -> float:
    """After trigger, stop = entry * (1 + offset). Default +0.05% to cover exit slippage."""
    return float(os.getenv("DAY_BREAK_EVEN_OFFSET_PCT", "0.0005"))


def _mfe_trail_tier_1_pct() -> float:
    """MFE fraction at which the trail tightens to tier-1 (default 0.50%)."""
    return float(os.getenv("DAY_MFE_TRAIL_TIER1_MFE_PCT", "0.0050"))


def _mfe_trail_tier_1_trail_pct() -> float:
    """Trail distance used once tier-1 MFE is reached (default 0.30%)."""
    return float(os.getenv("DAY_MFE_TRAIL_TIER1_TRAIL_PCT", "0.0030"))


def _mfe_trail_tier_2_pct() -> float:
    """MFE fraction at which the trail tightens to tier-2 (default 1.00%)."""
    return float(os.getenv("DAY_MFE_TRAIL_TIER2_MFE_PCT", "0.0100"))


def _mfe_trail_tier_2_trail_pct() -> float:
    """Trail distance used once tier-2 MFE is reached (default 0.20%)."""
    return float(os.getenv("DAY_MFE_TRAIL_TIER2_TRAIL_PCT", "0.0020"))


def _break_even_enabled() -> bool:
    return os.getenv("DAY_BREAK_EVEN_TRAIL_ENABLED", "true").strip().lower() in ("1", "true", "yes", "on")


def apply_break_even_and_mfe_trail(position: Any, current_price: float) -> bool:
    """Ratchet stop above entry when MFE has cleared cost, then tighten trail
    by MFE tier. Never widens/lowers the stop; only ratchets upward.

    Order of protection once MFE clears:
    1. MFE ≥ trigger (default 0.30%): move stop to entry + 0.05%. Removes
       "won-then-gave-back-to-loss" losses on ~50% of stalled winners.
    2. MFE ≥ tier-1 (default 0.50%): trailing distance tightens to 0.30%.
    3. MFE ≥ tier-2 (default 1.00%): trailing distance tightens to 0.20%.

    Returns True if the position's stop or trailing_stop_price advanced.
    """
    if not _break_even_enabled():
        return False
    entry = float(getattr(position, "entry_price", 0.0) or 0.0)
    if entry <= 0 or current_price <= 0:
        return False
    highest = float(getattr(position, "highest_price", 0.0) or entry)
    mfe_pct = max(0.0, (highest - entry) / entry) if entry > 0 else 0.0
    if mfe_pct <= 0.0:
        return False

    changed = False

    trigger = _break_even_trigger_pct()
    if mfe_pct + 1e-12 >= trigger:
        be_stop = entry * (1.0 + _break_even_offset_pct())
        current_stop = float(getattr(position, "stop_price", 0.0) or 0.0)
        if be_stop > current_stop + 1e-12:
            position.stop_price = be_stop
            changed = True
        current_trail = float(getattr(position, "trailing_stop_price", 0.0) or 0.0)
        if be_stop > current_trail + 1e-12:
            position.trailing_stop_price = be_stop
            changed = True

    tier2 = _mfe_trail_tier_2_pct()
    tier1 = _mfe_trail_tier_1_pct()
    if mfe_pct + 1e-12 >= tier2:
        tightened = _mfe_trail_tier_2_trail_pct()
    elif mfe_pct + 1e-12 >= tier1:
        tightened = _mfe_trail_tier_1_trail_pct()
    else:
        tightened = None

    # Adaptive trail (day_adaptive_trail.py): once a (symbol, setup, regime)
    # arm has enough closed-winner history (default 4+ obs), its actual
    # MFE-giveback percentile is a better trail width than the fixed 0.20%/
    # 0.30% tier constants above. Only overrides when the arm has real
    # history (source == "arm_history") — insufficient-data and disabled
    # cases fall back to the fixed tiers computed above, unchanged.
    if tightened is not None and tightened > 0:
        try:
            from backend.services.day_adaptive_trail import adaptive_trail_pct_for_arm

            _adaptive = adaptive_trail_pct_for_arm(
                str(getattr(position, "symbol", "") or ""),
                str(getattr(position, "entry_thesis", "") or ""),
                str(getattr(position, "day_route_regime_at_entry", "") or ""),
            )
            if _adaptive.get("source") == "arm_history":
                tightened = float(_adaptive["trail_pct"])
        except Exception:
            pass

    if tightened is not None and tightened > 0:
        new_trail = highest * (1.0 - tightened)
        current_trail = float(getattr(position, "trailing_stop_price", 0.0) or 0.0)
        if new_trail > current_trail + 1e-12:
            position.trailing_stop_price = new_trail
            changed = True
        # Also ratchet the stop_price up so evaluate_engine_managed_exit's stop
        # gate uses the tightened level (not just the trailing gate).
        current_stop = float(getattr(position, "stop_price", 0.0) or 0.0)
        if new_trail > current_stop + 1e-12 and new_trail < entry * (1.0 + 0.02):
            # Safety cap: never lift stop above entry+2% (silly stop).
            position.stop_price = new_trail
            changed = True

    return changed


def refresh_trailing_stop(position: Any, current_price: float, coin_profile: dict[str, Any] | None = None) -> bool:
    """Ratchet trailing stop when price makes new highs; returns True if level changed.

    In bull regime, once price has moved DAY_BULL_TRAIL_MFE_THRESHOLD (default 1.5%)
    in our favour, the trail distance is widened by DAY_BULL_TRAIL_MULTIPLIER (default 2x)
    so normal intraday noise does not shake out a strongly trending position.

    After the base trailing update, apply_break_even_and_mfe_trail runs so
    once MFE clears round-trip cost the stop is lifted to break-even and the
    trail tightens by MFE tier. Both changes are ratchets — never widen.
    """
    entry = float(getattr(position, "entry_price", 0.0) or 0.0)
    if entry <= 0 or current_price <= 0:
        return False

    profile = coin_profile or {}
    trail_pct = float(getattr(position, "trail_pct", 0.0) or profile.get("trail") or 0.005)
    highest = float(getattr(position, "highest_price", 0.0) or entry)
    mfe_pct = max(0.0, (highest - entry) / entry) if entry > 0 else 0.0

    regime = str(getattr(position, "day_route_regime_at_entry", "") or "").lower()
    if regime == "bull" and mfe_pct >= _bull_trail_mfe_threshold():
        scalar, _ = get_regime_validated_scalar(regime)
        widened_trail_pct = min(trail_pct * _bull_trail_multiplier(), 0.025)
        trail_pct = blend_by_scalar(trail_pct, widened_trail_pct, scalar)

    activation = entry * (1.0 + trail_pct)
    base_changed = False
    if highest >= activation:
        new_trail = highest * (1.0 - trail_pct)
        current_trail = float(getattr(position, "trailing_stop_price", 0.0) or 0.0)
        floor = entry * (1.0 - float(profile.get("sl") or 0.010))
        new_trail = max(new_trail, floor, current_trail)
        if new_trail > current_trail + 1e-12:
            position.trailing_stop_price = new_trail
            base_changed = True

    # Break-even ratchet + tiered MFE-tightening apply regardless of the base
    # trail activation (they use their own MFE thresholds).
    tier_changed = apply_break_even_and_mfe_trail(position, current_price)
    return base_changed or tier_changed


def _trail_semantics(
    *,
    entry: float,
    current_price: float,
    position: Any,
    coin_profile: dict[str, Any],
    path_aware: bool,
    atr_pct: float,
    bundle: dict[str, Any] | None,
) -> dict[str, Any]:
    """Split the overloaded ``trailing_stop_price`` field into distinct concepts.

    Persisted ``trailing_stop_price`` is a high-water ratchet:
    ``highest * (1 - trail_distance)`` after ``highest >= entry * (1 + trail_distance)``.
    For a long that is an intended *stop below the high*, not a target above market.
    Path-aware sells when the mark pulls back through that ratchet.
    """
    highest = float(getattr(position, "highest_price", entry) or entry)
    trail_distance = float(getattr(position, "trail_pct", 0.0) or coin_profile.get("trail") or 0.005)
    trail_activation = entry * (1.0 + trail_distance) if entry > 0 else 0.0
    ratchet = float(getattr(position, "trailing_stop_price", 0.0) or 0.0)
    activated = bool(entry > 0 and highest >= trail_activation - 1e-12)
    executable_trail = ratchet if activated and ratchet > 0 else None
    snap4 = day_4h_structure_snapshot(bundle)
    hard_stop = resolve_day_risk_floor_price(
        entry_price=entry,
        thesis_invalid_level=float(getattr(position, "thesis_invalid_level", 0.0) or 0.0),
        prior_4h_low=float(snap4.get("prior_4h_low") or 0.0),
        atr_pct=atr_pct,
    )
    persisted_stop = float(getattr(position, "stop_price", 0.0) or 0.0)
    if hard_stop <= 0:
        hard_stop = persisted_stop if 0 < persisted_stop < entry else 0.0
    return {
        "high_water": highest,
        "trail_activation": trail_activation,
        "trail_distance": trail_distance,
        "ratchet_trail_price": ratchet,
        "executable_trailing_stop": executable_trail,
        "hard_stop": hard_stop,
        "persisted_stop_price": persisted_stop,
        "trailing_stop_in_exit_authority": bool(executable_trail is not None),
        "path_aware_exit": path_aware,
        "4h_bundle_present": bool(snap4.get("4h_bundle_present")),
        "prior_4h_low": snap4.get("prior_4h_low"),
        "htf_4h_rise_intact": snap4.get("htf_4h_rise_intact"),
        "htf_4h_rise_broken": snap4.get("htf_4h_rise_broken"),
    }


def preview_next_engine_exit(
    *,
    position: Any,
    current_price: float,
    net_pnl_pct: float,
    hold_minutes: float,
    coin_profile: dict[str, Any],
    bundle: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Read-only: next *executable* exit authority and trail-field split.

    Does not mutate exit behaviour. When path-aware is on, ``next_engine_exit``
    is the authority that can actually flatten the position: risk floor,
    activated trail (mark through the ratchet), giveback, stall, or 4H break.
    Leftover intact-floor NET_PROFIT is not an executable authority.
    """
    entry = float(getattr(position, "entry_price", 0.0) or 0.0)
    stop = effective_stop_price(entry, float(getattr(position, "stop_price", 0) or 0), float(getattr(position, "thesis_invalid_level", 0) or 0))
    target = effective_target_price(entry, float(getattr(position, "take_profit_1_price", 0) or 0), float(getattr(position, "thesis_target_level", 0) or 0), position, bundle)
    max_hold = effective_max_hold_min(position, coin_profile)
    if int(getattr(position, "max_hold_min", 0) or 0) < max_hold:
        position.max_hold_min = max_hold
    trail = float(getattr(position, "trailing_stop_price", 0) or 0)
    thesis = str(getattr(position, "entry_thesis", "") or "")

    highest = float(getattr(position, "highest_price", entry) or entry)
    mfe_pct = max(0.0, (highest - entry) / entry) if entry > 0 else 0.0
    invalid = float(getattr(position, "thesis_invalid_level", 0.0) or 0.0)
    atr_pct = 0.01
    if invalid > 0 and invalid < entry:
        atr_pct = max(0.008, (entry - invalid) / entry)
    path_aware = _path_aware_exit_enabled()
    trail_info = _trail_semantics(
        entry=entry,
        current_price=current_price,
        position=position,
        coin_profile=coin_profile,
        path_aware=path_aware,
        atr_pct=atr_pct,
        bundle=bundle,
    )
    _stall_preview = evaluate_stall_exit(
        entry_price=entry,
        highest_price=highest,
        net_pnl_pct=net_pnl_pct,
        hold_minutes=hold_minutes,
        max_hold_min=max_hold,
        current_price=float(current_price or 0.0),
        lowest_price=float(getattr(position, "lowest_price", 0.0) or 0.0),
    )
    stall_ready = bool(_stall_preview is not None and str(_stall_preview.get("action") or "") == "sell")
    giveback_ready = bool(_giveback_exit_enabled() and hold_minutes >= _giveback_min_hold_min() and mfe_pct >= _giveback_min_mfe_pct() and net_pnl_pct + 1e-12 <= _giveback_trigger_pnl_pct())
    checks = {
        "stop_loss": bool(stop > 0 and current_price <= stop),
        "trailing_stop": bool(trail > 0 and current_price <= trail and highest >= entry * (1 + float(coin_profile.get("trail") or 0.005))),
        "giveback_exit": giveback_ready,
        "stall_exit": stall_ready,
        "time_stop": bool(hold_minutes >= max_hold and net_pnl_pct + 1e-12 < float(MIN_NET_PROFIT_TO_SELL)),
        "profit_target": bool(target > 0 and current_price >= target and net_pnl_pct + 1e-12 >= float(MIN_NET_PROFIT_TO_SELL) * 0.45),
        "net_profit": bool(net_pnl_pct + 1e-12 >= float(MIN_NET_PROFIT_TO_SELL)),
        "risk_floor": bool(trail_info["hard_stop"] > 0 and current_price <= float(trail_info["hard_stop"])),
    }

    legacy_next = "none"
    priority = [
        ("stop_loss", EXIT_STOP_LOSS),
        ("trailing_stop", EXIT_TRAILING_STOP),
        ("giveback_exit", EXIT_GIVEBACK),
        ("stall_exit", EXIT_STALL_DEAD),
        ("time_stop", EXIT_TIME_STOP),
        ("profit_target", EXIT_NET_PROFIT),
        ("net_profit", EXIT_NET_PROFIT),
    ]
    for key, reason in priority:
        if checks[key]:
            legacy_next = reason
            break

    next_exit = legacy_next
    current_authority = legacy_next
    next_executable_condition = legacy_next
    if path_aware:
        managed = _evaluate_path_aware_exit(
            position=position,
            current_price=current_price,
            net_pnl_pct=net_pnl_pct,
            hold_minutes=hold_minutes,
            coin_profile=coin_profile,
            bundle=bundle,
            entry=entry,
            atr_pct=atr_pct,
        )
        current_authority = str(managed.get("reason") or HOLD_4H_MISSING)
        if str(managed.get("action") or "") == "sell":
            next_exit = current_authority
            next_executable_condition = current_authority
        else:
            if checks["trailing_stop"]:
                next_exit = EXIT_TRAILING_STOP
                next_executable_condition = EXIT_TRAILING_STOP
                current_authority = EXIT_TRAILING_STOP
            elif checks["risk_floor"]:
                next_exit = EXIT_DAY_RISK_FLOOR
                next_executable_condition = EXIT_DAY_RISK_FLOOR
            elif trail_info.get("executable_trailing_stop"):
                next_exit = current_authority
                next_executable_condition = EXIT_TRAILING_STOP
            elif trail_info["hard_stop"] > 0 and entry > 0:
                next_exit = current_authority
                next_executable_condition = f"{EXIT_DAY_RISK_FLOOR}_or_{EXIT_DAY_4H_STRUCTURE_BREAK}"
            else:
                next_exit = current_authority
                next_executable_condition = EXIT_DAY_4H_STRUCTURE_BREAK

    dist_stop_pct = ((current_price - stop) / entry) if stop > 0 and entry > 0 else None
    dist_hard_pct = (
        ((current_price - float(trail_info["hard_stop"])) / entry)
        if trail_info["hard_stop"] > 0 and entry > 0
        else None
    )
    dist_target_pct = ((target - current_price) / entry) if target > 0 and entry > 0 else None
    hold_remaining_min = max(0.0, max_hold - hold_minutes)

    can_stall = not any(
        [
            stop > 0,
            target > 0,
            max_hold > 0,
            trail > 0,
            bool(thesis),
            trail_info["hard_stop"] > 0,
        ]
    )

    hold_ev_payload: dict[str, Any] | None = None
    try:
        from backend.services.hold_ev_engine import hold_ev_for_position

        _hev = hold_ev_for_position(position, current_price=float(current_price or 0.0), hold_minutes=hold_minutes)
        hold_ev_payload = {
            "hold_ev_score": _hev.hold_ev_score,
            "recommendation": _hev.recommendation,
            "confidence": _hev.confidence,
            "detail": _hev.detail,
        }
    except Exception:
        hold_ev_payload = None

    return {
        "effective_stop": stop,
        "effective_target": target,
        "max_hold_min": max_hold,
        "hold_minutes": round(hold_minutes, 2),
        "hold_remaining_min": round(hold_remaining_min, 2),
        "trailing_stop_price": trail,
        "high_water": trail_info["high_water"],
        "trail_activation": trail_info["trail_activation"],
        "trail_distance": trail_info["trail_distance"],
        "executable_trailing_stop": trail_info["executable_trailing_stop"],
        "hard_stop": trail_info["hard_stop"],
        "current_exit_authority": current_authority,
        "next_executable_exit_condition": next_executable_condition,
        "trailing_stop_in_exit_authority": trail_info["trailing_stop_in_exit_authority"],
        "path_aware_exit": path_aware,
        "legacy_ladder_next_exit": legacy_next,
        "exit_checks": checks,
        "next_engine_exit": next_exit,
        "distance_to_stop_pct": round(dist_stop_pct, 6) if dist_stop_pct is not None else None,
        "distance_to_hard_stop_pct": round(dist_hard_pct, 6) if dist_hard_pct is not None else None,
        "distance_to_target_pct": round(dist_target_pct, 6) if dist_target_pct is not None else None,
        "can_be_stuck_indefinitely": can_stall,
        "hold_ev": hold_ev_payload,
    }


def evaluate_engine_managed_exit(
    *,
    position: Any,
    current_price: float,
    net_pnl_pct: float,
    hold_minutes: float,
    coin_profile: dict[str, Any],
    bundle: dict[str, Any] | None = None,
    bar_low: float | None = None,
) -> dict[str, Any]:
    """
    Shared paper/live exit manager. Risk exits bypass net-profit-only gate.
    Priority: stop -> trailing -> thesis invalidation -> failed reclaim -> time -> profit.
    """
    entry = float(getattr(position, "entry_price", 0.0) or 0.0)
    if entry <= 0 or current_price <= 0:
        return {"action": "hold", "reason": "missing_price"}

    setup = str(getattr(position, "entry_thesis", "") or "")
    entry_vwap = float(getattr(position, "entry_vwap", 0.0) or 0.0)
    invalid_level = float(getattr(position, "thesis_invalid_level", 0.0) or 0.0)
    target_level = float(getattr(position, "thesis_target_level", 0.0) or 0.0)
    atr_pct = 0.01
    if invalid_level > 0 and invalid_level < entry:
        atr_pct = max(0.008, (entry - invalid_level) / entry)

    if _path_aware_exit_enabled():
        return _evaluate_path_aware_exit(
            position=position,
            current_price=current_price,
            net_pnl_pct=net_pnl_pct,
            hold_minutes=hold_minutes,
            coin_profile=coin_profile,
            bundle=bundle,
            entry=entry,
            atr_pct=atr_pct,
        )

    extreme = evaluate_extreme_protection(
        entry_price=entry,
        mark=current_price,
        net_pnl_pct=net_pnl_pct,
        atr_pct=atr_pct,
        bundle=bundle,
    )
    if str(extreme.get("action")) == "sell":
        return {
            "action": "sell",
            "reason": EXIT_EXTREME_PROTECTION,
            "net_pnl_pct": net_pnl_pct,
            "hold_minutes": hold_minutes,
        }

    # Item p7 gap-closure: adaptive MAE-distribution check for STRAIGHT losers
    # (never built the favorable excursion evaluate_giveback_exit requires),
    # evaluated BEFORE the fixed stop-loss so the arm's own real losing-trade
    # history — not a fixed per-coin percentage — is the PRIMARY signal for
    # loss handling. Can only fire tighter/earlier than the fixed stop below,
    # never wider: the fixed stop_loss and catastrophic extreme protection
    # above remain the unconditional mechanical backstop regardless.
    adaptive_loss = evaluate_adaptive_loss_exit(
        entry_price=entry,
        net_pnl_pct=net_pnl_pct,
        hold_minutes=hold_minutes,
        position=position,
    )
    if adaptive_loss is not None:
        return adaptive_loss

    stop = effective_stop_price(entry, float(getattr(position, "stop_price", 0) or 0), invalid_level)
    low = float(bar_low if bar_low is not None else current_price)
    if stop > 0 and (current_price <= stop or low <= stop):
        return {
            "action": "sell",
            "reason": EXIT_STOP_LOSS,
            "net_pnl_pct": net_pnl_pct,
            "hold_minutes": hold_minutes,
            "detail": f"stop={stop:.8f}",
        }

    trail = float(getattr(position, "trailing_stop_price", 0) or 0)
    trail_pct = float(getattr(position, "trail_pct", 0) or coin_profile.get("trail") or 0.005)
    highest = float(getattr(position, "highest_price", entry) or entry)
    if trail > 0 and highest >= entry * (1.0 + trail_pct) and current_price <= trail:
        return {
            "action": "sell",
            "reason": EXIT_TRAILING_STOP,
            "net_pnl_pct": net_pnl_pct,
            "hold_minutes": hold_minutes,
            "detail": f"trail={trail:.8f}",
        }

    if setup and thesis_invalidated_live(
        setup,
        mark=current_price,
        invalid_level=invalid_level,
        bundle=bundle,
        entry_vwap=entry_vwap,
        entry_price=entry,
        atr_pct=atr_pct,
    ):
        return {
            "action": "sell",
            "reason": EXIT_THESIS_INVALIDATION,
            "net_pnl_pct": net_pnl_pct,
            "hold_minutes": hold_minutes,
            "detail": setup,
        }

    failed_reclaim_h = float(coin_profile.get("failed_reclaim_hours") or 6.0)
    hold_h = hold_minutes / 60.0
    if setup == SETUP_VWAP_REVERSION and hold_h >= failed_reclaim_h and net_pnl_pct + 1e-12 < float(MIN_NET_PROFIT_TO_SELL) * 0.5:
        ref = entry_vwap if entry_vwap > 0 else entry
        buf = float(coin_profile.get("failed_reclaim_buffer_pct") or 0.0015)
        if current_price < ref * (1.0 - buf):
            return {
                "action": "sell",
                "reason": EXIT_FAILED_RECLAIM,
                "net_pnl_pct": net_pnl_pct,
                "hold_minutes": hold_minutes,
            }

    _pos_regime = str(getattr(position, "day_route_regime_at_entry", "") or "").lower()
    if _pos_regime == "bull":
        # In bull regime require a deeper reversal before treating a pullback as a giveback —
        # normal bull noise can easily exceed the default -0.15% trigger on the way to target.
        # Leniency is scaled by validated edge: if "bull" hasn't shown a real forward-return
        # edge yet, blend back toward the standard (tighter) giveback thresholds.
        #
        # Item p7 gap-closure: the "tighter" side of that blend used to always be the
        # fixed global -0.15% constant, silently bypassing this arm's own adaptive
        # MAE-distribution trigger (day_adaptive_targets.adaptive_giveback_trigger_for_arm)
        # even when real per-arm history existed — the bull path is now grounded in the
        # SAME adaptive-or-fixed trigger the non-bull path uses, before applying the
        # bull-specific leniency blend on top of it.
        _highest = float(getattr(position, "highest_price", entry) or entry)
        _mfe = max(0.0, (_highest - entry) / entry) if entry > 0 else 0.0
        _bull_scalar, _ = get_regime_validated_scalar(_pos_regime)
        _base_trigger = _giveback_trigger_pnl_pct()
        _base_trigger_source = "fixed_default"
        try:
            from backend.services.day_adaptive_targets import adaptive_giveback_trigger_for_arm

            _adaptive_bull = adaptive_giveback_trigger_for_arm(
                str(getattr(position, "symbol", "") or ""),
                str(getattr(position, "entry_thesis", "") or ""),
                _pos_regime,
            )
            if _adaptive_bull.get("source") not in ("insufficient_data", "disabled"):
                _base_trigger = float(_adaptive_bull["trigger_pct"])
                _base_trigger_source = str(_adaptive_bull["source"])
        except Exception:
            pass
        _bull_mfe_thresh = blend_by_scalar(_giveback_min_mfe_pct(), float(os.getenv("DAY_BULL_GIVEBACK_MIN_MFE", "0.005")), _bull_scalar)
        _bull_trigger = blend_by_scalar(_base_trigger, float(os.getenv("DAY_BULL_GIVEBACK_TRIGGER", "-0.003")), _bull_scalar)
        # Item p8 promotion: same HoldEV tighten-only nudge as the non-bull
        # giveback path, applied on top of the bull-leniency blend above.
        _hev_factor, _hev_detail = _hold_ev_tighten(position, entry_price=entry, net_pnl_pct=net_pnl_pct, hold_minutes=hold_minutes)
        _bull_trigger *= _hev_factor
        if _mfe >= _bull_mfe_thresh and net_pnl_pct + 1e-12 <= _bull_trigger:
            giveback = {
                "action": "sell",
                "reason": EXIT_GIVEBACK,
                "net_pnl_pct": net_pnl_pct,
                "hold_minutes": hold_minutes,
                "detail": f"bull_giveback mfe={_mfe:.6f} trigger={_bull_trigger} base_trigger_source={_base_trigger_source} {_hev_detail}",
            }
        else:
            giveback = None
    else:
        giveback = evaluate_giveback_exit(
            entry_price=entry,
            highest_price=float(getattr(position, "highest_price", entry) or entry),
            net_pnl_pct=net_pnl_pct,
            hold_minutes=hold_minutes,
            position=position,
        )
    if giveback is not None:
        return giveback

    max_hold = effective_max_hold_min(position, coin_profile)
    if int(getattr(position, "max_hold_min", 0) or 0) < max_hold:
        position.max_hold_min = max_hold

    progress_decay = evaluate_progress_decay_exit(
        entry_price=entry,
        highest_price=float(getattr(position, "highest_price", entry) or entry),
        net_pnl_pct=net_pnl_pct,
        hold_minutes=hold_minutes,
        position=position,
    )
    if progress_decay is not None:
        return progress_decay

    _stall_regime = str(getattr(position, "day_route_regime_at_entry", "") or "").lower()
    # Full stall suppression is the strongest bull-regime bonus, so it requires the
    # strongest evidence bar: only skip the stall check once the label has shown a
    # real validated forward-return edge (scalar >= 0.7), not merely "not enough
    # data yet" (scalar defaults to 1.0 pre-data — see AI_REGIME_VALIDATION_MIN_SAMPLES).
    _stall_suppressed = False
    if _stall_regime == "bull":
        _stall_scalar, _ = get_regime_validated_scalar(_stall_regime)
        _stall_suppressed = _stall_scalar >= 0.7
    if not _stall_suppressed:
        stall = evaluate_stall_exit(
            entry_price=entry,
            highest_price=float(getattr(position, "highest_price", entry) or entry),
            net_pnl_pct=net_pnl_pct,
            hold_minutes=hold_minutes,
            max_hold_min=max_hold,
            current_price=float(current_price or 0.0),
            lowest_price=float(getattr(position, "lowest_price", 0.0) or 0.0),
        )
        if stall is not None and str(stall.get("action") or "") == "sell":
            return stall
        # Hold telemetry reasons (STALL_HOLD_*) continue the exit chain — do not force-sell.
    # Bull regime with validated edge: stall is suppressed — price consolidating
    # before next leg is normal. Without validated edge, stall check still applies.

    if hold_minutes >= max_hold and net_pnl_pct + 1e-12 < float(MIN_NET_PROFIT_TO_SELL):
        if net_pnl_pct >= 0.0:
            # Position is net-positive but below profit floor — let stop-loss / trailing stop /
            # target exit it cleanly rather than clocking out a winning trade. Applies in all
            # regimes: a profitable position should never be forcibly exited by a timer.
            pass
        else:
            return {
                "action": "sell",
                "reason": EXIT_TIME_STOP,
                "net_pnl_pct": net_pnl_pct,
                "hold_minutes": hold_minutes,
                "detail": f"max_hold_min={max_hold}",
            }

    # Resolve per-coin profit floor from the position's symbol.
    from backend.config.trading_economics import min_net_profit_for_symbol as _mnp

    _sym = str(getattr(position, "symbol", "") or "")
    _min_net = float(_mnp(_sym)) if _sym else float(MIN_NET_PROFIT_TO_SELL)

    target = effective_target_price(entry, float(getattr(position, "take_profit_1_price", 0) or 0), target_level, position, bundle)
    if target > 0 and current_price >= target and net_pnl_pct + 1e-12 >= _min_net * 0.45:
        return {
            "action": "sell",
            "reason": EXIT_NET_PROFIT,
            "net_pnl_pct": net_pnl_pct,
            "hold_minutes": hold_minutes,
            "detail": "target_hit",
        }

    if net_pnl_pct + 1e-12 >= _min_net:
        te = evaluate_thesis_exit(
            entry_thesis=setup,
            thesis_score=float(getattr(position, "thesis_score", 0.0) or 0.0),
            thesis_invalid_level=invalid_level,
            thesis_target_level=target_level,
            entry_vwap=entry_vwap,
            entry_price=entry,
            mark=current_price,
            bundle=bundle,
            symbol=_sym or None,
        )
        if str(te.get("action")) == "sell":
            return {
                "action": "sell",
                "reason": EXIT_NET_PROFIT,
                "net_pnl_pct": net_pnl_pct,
                "hold_minutes": hold_minutes,
                "detail": te.get("detail") or "profit_floor",
            }
        if str(te.get("action")) == "warn":
            return {"action": "hold", "reason": EXIT_THESIS_WARNING, "net_pnl_pct": net_pnl_pct}

    return {"action": "hold", "reason": "bracket_hold", "net_pnl_pct": net_pnl_pct, "hold_minutes": hold_minutes}


def evaluate_controlled_bracket_exit(
    *,
    entry_price: float,
    mark: float,
    bar_low: float,
    entry_ts: int,
    bar_ts: int,
    setup: str,
    invalid_level: float,
    atr_pct: float,
    net_pct_fill: float,
    net_pct_mid: float,
    bundle: dict[str, Any] | None,
    cfg: ControlledExitConfig,
    entry_vwap: float = 0.0,
) -> dict[str, Any]:
    """
    Bracket exit: profit target + volatility stop + time stop + failed reclaim.
    Replay helper — live engine uses evaluate_engine_managed_exit.
    """
    if entry_price <= 0 or mark <= 0:
        return {"action": "hold", "reason": "missing_price"}

    atr = max(0.003, float(atr_pct or 0.01))
    hold_h = max(0.0, (bar_ts - entry_ts) / 3600.0)
    gate = net_pct_fill if cfg.use_fill_based_gate else net_pct_mid

    extreme = evaluate_extreme_protection(
        entry_price=entry_price,
        mark=mark,
        net_pnl_pct=net_pct_mid,
        atr_pct=atr,
        bundle=bundle,
    )
    if str(extreme.get("action")) == "sell":
        return {
            "action": "sell",
            "reason": EXIT_EXTREME_PROTECTION,
            "net_pnl_pct": gate,
            "hold_hours": hold_h,
        }

    if not cfg.enabled:
        te = evaluate_thesis_exit(
            entry_thesis=setup,
            thesis_score=0.5,
            thesis_invalid_level=invalid_level,
            thesis_target_level=0.0,
            entry_vwap=entry_vwap,
            entry_price=entry_price,
            mark=mark,
            bundle=bundle,
        )
        if str(te.get("action")) == "warn":
            return {"action": "hold", "reason": EXIT_THESIS_WARNING, "net_pnl_pct": gate}
        if gate >= cfg.profit_floor_pct:
            return {"action": "sell", "reason": EXIT_NET_PROFIT, "net_pnl_pct": gate, "hold_hours": hold_h}
        return {"action": "hold", "reason": "profit_only_hold", "net_pnl_pct": gate}

    stop_dist = cfg.atr_stop_mult * atr
    stop_px = entry_price * (1.0 - stop_dist)
    eff_loss_pct = min(stop_dist, cfg.max_loss_pct)
    if bar_low <= stop_px or gate <= -eff_loss_pct:
        return {
            "action": "sell",
            "reason": EXIT_VOLATILITY_STOP,
            "net_pnl_pct": gate,
            "hold_hours": hold_h,
            "stop_dist_pct": stop_dist,
        }

    if setup == SETUP_VWAP_REVERSION and hold_h >= cfg.failed_reclaim_hours and gate < cfg.profit_floor_pct * 0.5:
        buf = cfg.failed_reclaim_buffer_pct
        ref = entry_vwap if entry_vwap > 0 else entry_price
        if mark < ref * (1.0 - buf):
            return {
                "action": "sell",
                "reason": EXIT_FAILED_RECLAIM,
                "net_pnl_pct": gate,
                "hold_hours": hold_h,
            }

    if hold_h >= cfg.time_stop_hours and gate < cfg.profit_floor_pct:
        return {
            "action": "sell",
            "reason": EXIT_TIME_STOP,
            "net_pnl_pct": gate,
            "hold_hours": hold_h,
        }

    if gate + 1e-12 >= cfg.profit_floor_pct:
        return {
            "action": "sell",
            "reason": EXIT_NET_PROFIT,
            "net_pnl_pct": gate,
            "hold_hours": hold_h,
        }

    te = evaluate_thesis_exit(
        entry_thesis=setup,
        thesis_score=0.5,
        thesis_invalid_level=invalid_level,
        thesis_target_level=0.0,
        entry_vwap=entry_vwap,
        entry_price=entry_price,
        mark=mark,
        bundle=bundle,
    )
    if str(te.get("action")) == "warn":
        return {"action": "hold", "reason": EXIT_THESIS_WARNING, "net_pnl_pct": gate}

    return {"action": "hold", "reason": "bracket_hold", "net_pnl_pct": gate, "hold_hours": hold_h}


_SETUP_REGIME_MISMATCH_BLOCKS = frozenset(
    {
        "REGIME_ROUTE_BEAR_NO_TREND_PULLBACK",
        "REGIME_ROUTE_RANGE_MR_ONLY",
        "REGIME_ROUTE_NEUTRAL_MR_ONLY",
        "REGIME_ROUTE_BULL_SETUP_MISMATCH",
        "ALLWEATHER_ROUTE_BULL_SETUP_MISMATCH",
        "ALLWEATHER_ROUTE_NEUTRAL_RANGE_ONLY",
        "ALLWEATHER_ROUTE_BEAR_REVERSAL_ONLY",
    }
)


class _PreBuyPositionView:
    """Minimal position view for shared exit-manager checks at entry."""

    def __init__(
        self,
        *,
        entry_price: float,
        stop_price: float,
        setup: str,
        invalid_level: float,
        target_level: float,
        entry_vwap: float,
        entry_ts: float,
        trail_pct: float,
        max_hold_min: int,
    ) -> None:
        self.entry_price = entry_price
        self.stop_price = stop_price
        self.entry_thesis = setup
        self.thesis_invalid_level = invalid_level
        self.thesis_target_level = target_level
        self.entry_vwap = entry_vwap
        self.entry_time = entry_ts
        self.highest_price = entry_price
        self.trailing_stop_price = 0.0
        self.trail_pct = trail_pct
        self.max_hold_min = max_hold_min
        self.thesis_score = 0.0
        self.take_profit_1_price = target_level


def evaluate_pre_buy_exit_consistency(
    *,
    setup: str,
    entry_price: float,
    stop_price: float,
    thesis_invalid_level: float,
    thesis_target_level: float,
    entry_vwap: float,
    entry_ts: float,
    coin_profile: dict[str, Any],
    bundle: dict[str, Any] | None,
    spread_pct: float = 0.0,
    day_regime: str = "",
    decision_data: dict[str, Any] | None = None,
    context_payload: dict[str, Any] | None = None,
    thesis_score: float = 0.0,
    bar_ts: float | None = None,
) -> dict[str, Any]:
    """
    Run the same immediate-exit checks the engine uses post-entry.
    Blocks self-contradictory buys (thesis invalid at entry, stop already hit, etc.).
    """
    from backend.services.day_regime_router import evaluate_day_entry_route

    dd = dict(decision_data or {})
    setup_s = str(setup or dd.get("setup_type") or dd.get("entry_thesis") or "")
    regime = str(day_regime or dd.get("day_route_regime") or "").strip().lower()
    entry = float(entry_price or 0.0)
    result: dict[str, Any] = {
        "allowed": True,
        "block_reason": "",
        "immediate_exit_reason": "",
        "setup_regime_compatible": True,
        "invalidation_at_entry": False,
        "entry_exit_state_consistent": True,
        "checks": {},
    }
    if entry <= 0 or not setup_s:
        result.update({"allowed": False, "block_reason": "ENTRY_EXIT_MISSING_PRICE_OR_SETUP", "entry_exit_state_consistent": False})
        return result

    route = evaluate_day_entry_route(
        setup_type=setup_s,
        day_regime=regime or "neutral",
        decision_data=dd,
        context_payload=context_payload,
        current_price=entry,
        thesis_score=float(thesis_score or dd.get("thesis_score") or 0.0),
        strategy_family=str(dd.get("strategy_family") or ""),
    )
    result["checks"]["route_allowed"] = bool(route.get("allowed"))
    result["checks"]["route_block_reason"] = str(route.get("block_reason") or "")
    if not route.get("allowed"):
        route_reason = str(route.get("block_reason") or "REGIME_ROUTE")
        # NON-BLOCKING BY DESIGN (continuation repair): every router outcome,
        # including "*_MFE_TOO_LOW", is an *expectation* about the trade's own
        # thesis target, not a measured execution fact. _vwap_expected_mfe_after_fees_ok
        # compares a PROJECTED move to the strategy's own thesis_target_level
        # against a constant ESTIMATED_ROUNDTRIP_COST — it is "expected favorable
        # excursion is insufficient", i.e. a trade-opinion/expected-value gate,
        # not a real-time spread/impact/liquidity safety fact (contrast with
        # SCALP's NET_EDGE_BELOW_MIN, which uses live order-book data and stays
        # a hard block there). Mystic is a ranking-and-trading engine: every
        # router outcome is surfaced as advisory penalty info only
        # (route_rank_delta/route_size_factor remain available for sizing/scoring).
        result["checks"]["route_regime_mismatch_advisory"] = route_reason
        result["checks"]["route_rank_delta"] = route.get("route_rank_delta", 0.0)
        result["checks"]["route_size_factor"] = route.get("route_size_factor", 1.0)
        # Fall through: continue with the remaining (non-opinion) entry/exit
        # state checks below instead of rejecting on the router's opinion alone.

    invalid_level = float(thesis_invalid_level or 0.0)
    atr_pct = 0.01
    if invalid_level > 0 and invalid_level < entry:
        atr_pct = max(0.008, (entry - invalid_level) / entry)

    invalidated = thesis_invalidated_live(
        setup_s,
        mark=entry,
        invalid_level=invalid_level,
        bundle=bundle,
        entry_vwap=float(entry_vwap or 0.0),
        entry_price=entry,
        atr_pct=atr_pct,
        spread_pct=float(spread_pct or 0.0),
    )
    result["checks"]["thesis_invalidated_live"] = invalidated
    if invalidated:
        result.update(
            {
                "allowed": False,
                "block_reason": "ENTRY_EXIT_THESIS_INVALID_AT_ENTRY",
                "immediate_exit_reason": EXIT_THESIS_INVALIDATION,
                "invalidation_at_entry": True,
                "entry_exit_state_consistent": False,
            }
        )
        return result

    stop = effective_stop_price(entry, float(stop_price or 0.0), invalid_level)
    if stop > 0 and entry <= stop:
        result.update(
            {
                "allowed": False,
                "block_reason": "ENTRY_EXIT_STOP_LOSS_AT_ENTRY",
                "immediate_exit_reason": EXIT_STOP_LOSS,
                "entry_exit_state_consistent": False,
            }
        )
        return result

    now_ts = float(bar_ts if bar_ts is not None else time.time())
    entry_time = float(entry_ts or now_ts)
    if entry_ts and entry_ts > now_ts + 5.0:
        result.update(
            {
                "allowed": False,
                "block_reason": "ENTRY_EXIT_BAD_ENTRY_TIMESTAMP",
                "immediate_exit_reason": EXIT_TIME_STOP,
                "entry_exit_state_consistent": False,
            }
        )
        return result

    pos = _PreBuyPositionView(
        entry_price=entry,
        stop_price=float(stop_price or 0.0),
        setup=setup_s,
        invalid_level=invalid_level,
        target_level=float(thesis_target_level or 0.0),
        entry_vwap=float(entry_vwap or 0.0),
        entry_ts=entry_time,
        trail_pct=float(coin_profile.get("trail") or 0.005),
        max_hold_min=int(coin_profile.get("max_hold_min") or 75),
    )
    managed = evaluate_engine_managed_exit(
        position=pos,
        current_price=entry,
        net_pnl_pct=-float(ESTIMATED_ROUNDTRIP_COST),
        hold_minutes=0.0,
        coin_profile=coin_profile,
        bundle=bundle,
        bar_low=entry,
    )
    immediate = str(managed.get("reason") or "")
    result["checks"]["managed_exit"] = immediate
    if str(managed.get("action")) == "sell" and immediate in (
        EXIT_THESIS_INVALIDATION,
        EXIT_STOP_LOSS,
        EXIT_FAILED_RECLAIM,
        EXIT_TIME_STOP,
        EXIT_EXTREME_PROTECTION,
    ):
        result.update(
            {
                "allowed": False,
                "block_reason": f"ENTRY_EXIT_IMMEDIATE_{immediate}",
                "immediate_exit_reason": immediate,
                "invalidation_at_entry": immediate == EXIT_THESIS_INVALIDATION,
                "entry_exit_state_consistent": False,
            }
        )
    return result
