"""
Controlled-risk DAY bracket exits — shared by paper and live via portfolio_engine.

Engine-managed sells: profit target, stop loss, time stop, trailing protection,
strategy-specific invalidation, plus catastrophic EXTREME_PROTECTION.
Net-profit exit remains one path among several — never the only sell path.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Any

from backend.config.trading_economics import ESTIMATED_ROUNDTRIP_COST, MIN_NET_PROFIT_TO_SELL
from backend.services.ai_regime_validation import blend_by_scalar, get_regime_validated_scalar
from backend.services.day_trade_thesis import (
    EXIT_EXTREME_PROTECTION,
    EXIT_NET_PROFIT,
    EXIT_STOP_LOSS,
    EXIT_THESIS_INVALIDATION,
    EXIT_THESIS_WARNING,
    EXIT_TRAILING_STOP,
    SETUP_VWAP_REVERSION,
    evaluate_extreme_protection,
    evaluate_thesis_exit,
    thesis_invalidated_live,
)

EXIT_VOLATILITY_STOP = "VOLATILITY_STOP_EXIT"
EXIT_TIME_STOP = "TIME_STOP_EXIT"
EXIT_FAILED_RECLAIM = "FAILED_RECLAIM_EXIT"
EXIT_STALL = "STALL_EXIT"
EXIT_STALL_DEAD = "STALL_EXIT_DEAD_NO_MFE"
EXIT_GIVEBACK = "GIVEBACK_EXIT"

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
        EXIT_VOLATILITY_STOP,
        EXIT_TIME_STOP,
        EXIT_STALL,
        EXIT_STALL_DEAD,
        EXIT_GIVEBACK,
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
    }
)

ENGINE_RISK_EXIT_PREFIXES = (
    EXIT_STOP_LOSS,
    EXIT_VOLATILITY_STOP,
    EXIT_TIME_STOP,
    EXIT_STALL,
    EXIT_STALL_DEAD,
    EXIT_GIVEBACK,
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
            "detail": (
                f"stall_min={stall_min:.0f}m mfe={mfe_pct:.6f} mae={mae_pct:.6f} "
                f"max_mfe={max_mfe:.6f} min_adverse={min_adverse:.6f}"
            ),
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
        "detail": (
            f"stall_min={stall_min:.0f}m mfe={mfe_pct:.6f} mae={mae_pct:.6f} "
            f"max_mfe={max_mfe:.6f} min_adverse={min_adverse:.6f}"
        ),
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
) -> dict[str, Any] | None:
    """
    Cut DAY holds that reached meaningful favorable excursion and then reversed
    back to net-negative, instead of waiting for the stall floor to book a
    deeper loss.

    Defaults require ~20m development and a clearer MFE/giveback so 1–3m noise
    does not churn day trades. Exit-only — no entry/ranking changes.
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
    if net_pnl_pct + 1e-12 > trigger:
        return None
    return {
        "action": "sell",
        "reason": EXIT_GIVEBACK,
        "net_pnl_pct": net_pnl_pct,
        "hold_minutes": hold_minutes,
        "detail": f"mfe={mfe_pct:.6f} trigger={trigger:.6f}",
    }


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


def effective_target_price(entry_price: float, tp1: float, thesis_target: float) -> float:
    """Long target: nearest profit objective above entry."""
    entry = float(entry_price or 0.0)
    if entry <= 0:
        return 0.0
    candidates = [float(v) for v in (tp1, thesis_target) if v and float(v) > entry]
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


def preview_next_engine_exit(
    *,
    position: Any,
    current_price: float,
    net_pnl_pct: float,
    hold_minutes: float,
    coin_profile: dict[str, Any],
    bundle: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Read-only: next exit path and whether position can stall indefinitely."""
    entry = float(getattr(position, "entry_price", 0.0) or 0.0)
    stop = effective_stop_price(entry, float(getattr(position, "stop_price", 0) or 0), float(getattr(position, "thesis_invalid_level", 0) or 0))
    target = effective_target_price(entry, float(getattr(position, "take_profit_1_price", 0) or 0), float(getattr(position, "thesis_target_level", 0) or 0))
    max_hold = effective_max_hold_min(position, coin_profile)
    if int(getattr(position, "max_hold_min", 0) or 0) < max_hold:
        position.max_hold_min = max_hold
    trail = float(getattr(position, "trailing_stop_price", 0) or 0)
    thesis = str(getattr(position, "entry_thesis", "") or "")

    highest = float(getattr(position, "highest_price", entry) or entry)
    mfe_pct = max(0.0, (highest - entry) / entry) if entry > 0 else 0.0
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
    }

    next_exit = "none"
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
            next_exit = reason
            break

    dist_stop_pct = ((current_price - stop) / entry) if stop > 0 and entry > 0 else None
    dist_target_pct = ((target - current_price) / entry) if target > 0 and entry > 0 else None
    hold_remaining_min = max(0.0, max_hold - hold_minutes)

    can_stall = not any(
        [
            stop > 0,
            target > 0,
            max_hold > 0,
            trail > 0,
            bool(thesis),
        ]
    )

    return {
        "effective_stop": stop,
        "effective_target": target,
        "max_hold_min": max_hold,
        "hold_minutes": round(hold_minutes, 2),
        "hold_remaining_min": round(hold_remaining_min, 2),
        "trailing_stop_price": trail,
        "exit_checks": checks,
        "next_engine_exit": next_exit,
        "distance_to_stop_pct": round(dist_stop_pct, 6) if dist_stop_pct is not None else None,
        "distance_to_target_pct": round(dist_target_pct, 6) if dist_target_pct is not None else None,
        "can_be_stuck_indefinitely": can_stall,
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
        _highest = float(getattr(position, "highest_price", entry) or entry)
        _mfe = max(0.0, (_highest - entry) / entry) if entry > 0 else 0.0
        _bull_scalar, _ = get_regime_validated_scalar(_pos_regime)
        _bull_mfe_thresh = blend_by_scalar(
            _giveback_min_mfe_pct(), float(os.getenv("DAY_BULL_GIVEBACK_MIN_MFE", "0.005")), _bull_scalar
        )
        _bull_trigger = blend_by_scalar(
            _giveback_trigger_pnl_pct(), float(os.getenv("DAY_BULL_GIVEBACK_TRIGGER", "-0.003")), _bull_scalar
        )
        if _mfe >= _bull_mfe_thresh and net_pnl_pct + 1e-12 <= _bull_trigger:
            giveback = {
                "action": "sell",
                "reason": EXIT_GIVEBACK,
                "net_pnl_pct": net_pnl_pct,
                "hold_minutes": hold_minutes,
                "detail": f"bull_giveback mfe={_mfe:.6f} trigger={_bull_trigger}",
            }
        else:
            giveback = None
    else:
        giveback = evaluate_giveback_exit(
            entry_price=entry,
            highest_price=float(getattr(position, "highest_price", entry) or entry),
            net_pnl_pct=net_pnl_pct,
            hold_minutes=hold_minutes,
        )
    if giveback is not None:
        return giveback

    max_hold = effective_max_hold_min(position, coin_profile)
    if int(getattr(position, "max_hold_min", 0) or 0) < max_hold:
        position.max_hold_min = max_hold
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

    target = effective_target_price(entry, float(getattr(position, "take_profit_1_price", 0) or 0), target_level)
    if target > 0 and current_price >= target and net_pnl_pct + 1e-12 >= float(MIN_NET_PROFIT_TO_SELL) * 0.45:
        return {
            "action": "sell",
            "reason": EXIT_NET_PROFIT,
            "net_pnl_pct": net_pnl_pct,
            "hold_minutes": hold_minutes,
            "detail": "target_hit",
        }

    if net_pnl_pct + 1e-12 >= float(MIN_NET_PROFIT_TO_SELL):
        te = evaluate_thesis_exit(
            entry_thesis=setup,
            thesis_score=float(getattr(position, "thesis_score", 0.0) or 0.0),
            thesis_invalid_level=invalid_level,
            thesis_target_level=target_level,
            entry_vwap=entry_vwap,
            entry_price=entry,
            mark=current_price,
            bundle=bundle,
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
