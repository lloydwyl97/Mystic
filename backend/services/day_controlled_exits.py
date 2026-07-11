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

ALLOWED_DAY_EXIT_REASONS = frozenset(
    {
        EXIT_NET_PROFIT,
        EXIT_VOLATILITY_STOP,
        EXIT_TIME_STOP,
        EXIT_STALL,
        EXIT_FAILED_RECLAIM,
        EXIT_EXTREME_PROTECTION,
        EXIT_THESIS_INVALIDATION,
        EXIT_STOP_LOSS,
        EXIT_TRAILING_STOP,
        "MANUAL_EXIT",
        "LEGACY_CLEANUP_EXIT",
        "LEGACY_INVENTORY_CLEANUP_EXIT",
        "ADMIN_CLEAR",
    }
)

ENGINE_RISK_EXIT_PREFIXES = (
    EXIT_STOP_LOSS,
    EXIT_VOLATILITY_STOP,
    EXIT_TIME_STOP,
    EXIT_STALL,
    EXIT_TRAILING_STOP,
    EXIT_THESIS_INVALIDATION,
    EXIT_FAILED_RECLAIM,
    EXIT_EXTREME_PROTECTION,
    "ALLWEATHER",
)


def _stall_exit_enabled() -> bool:
    return os.getenv("DAY_STALL_EXIT_ENABLED", "true").lower() in ("1", "true", "yes", "on")


def _stall_min_hold_min() -> float:
    return float(os.getenv("DAY_STALL_MIN_HOLD_MIN", "30"))


def _stall_max_mfe_pct() -> float:
    """Max mark MFE (fraction) allowed for a stall cut. Default 0.15%."""
    return float(os.getenv("DAY_STALL_MAX_MFE_PCT", "0.0015"))


def evaluate_stall_exit(
    *,
    entry_price: float,
    highest_price: float,
    net_pnl_pct: float,
    hold_minutes: float,
    max_hold_min: int,
) -> dict[str, Any] | None:
    """
    Cut dead DAY holds that never make meaningful progress before hard time-stop.

    Evidence: MANUAL/time-stop losers typically show no TP1 progress by 15–30m
    and then bleed until 75–90m. This is exit-only — no entry/ranking changes.
    """
    if not _stall_exit_enabled():
        return None
    entry = float(entry_price or 0.0)
    if entry <= 0:
        return None
    stall_min = _stall_min_hold_min()
    if hold_minutes < stall_min:
        return None
    # Never replace the hard ceiling — time-stop still owns max_hold.
    if hold_minutes + 1e-9 >= float(max_hold_min):
        return None
    # Only cut flat/losing paths — never scratch small greens before TP.
    if net_pnl_pct + 1e-12 >= 0.0:
        return None
    # Only cut when still below the net-profit floor (same eligibility as time-stop).
    if net_pnl_pct + 1e-12 >= float(MIN_NET_PROFIT_TO_SELL):
        return None
    highest = float(highest_price or entry)
    mfe_pct = max(0.0, (highest - entry) / entry)
    if mfe_pct >= _stall_max_mfe_pct():
        return None
    return {
        "action": "sell",
        "reason": EXIT_STALL,
        "net_pnl_pct": net_pnl_pct,
        "hold_minutes": hold_minutes,
        "detail": f"stall_min={stall_min:.0f}m mfe={mfe_pct:.6f} max_mfe={_stall_max_mfe_pct():.6f}",
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

    if not getattr(position, "max_hold_min", 0):
        position.max_hold_min = max_hold_min
        added.append(f"max_hold_min={max_hold_min}")

    if trail_pct and not getattr(position, "trail_pct", 0):
        position.trail_pct = trail_pct
        added.append(f"trail_pct={trail_pct}")

    return added


def refresh_trailing_stop(position: Any, current_price: float, coin_profile: dict[str, Any] | None = None) -> bool:
    """Ratchet trailing stop when price makes new highs; returns True if level changed."""
    entry = float(getattr(position, "entry_price", 0.0) or 0.0)
    if entry <= 0 or current_price <= 0:
        return False

    profile = coin_profile or {}
    trail_pct = float(getattr(position, "trail_pct", 0.0) or profile.get("trail") or 0.005)
    highest = float(getattr(position, "highest_price", 0.0) or entry)
    activation = entry * (1.0 + trail_pct)
    if highest < activation:
        return False

    new_trail = highest * (1.0 - trail_pct)
    current_trail = float(getattr(position, "trailing_stop_price", 0.0) or 0.0)
    floor = entry * (1.0 - float(profile.get("sl") or 0.010))
    new_trail = max(new_trail, floor, current_trail)
    if new_trail > current_trail + 1e-12:
        position.trailing_stop_price = new_trail
        return True
    return False


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
    max_hold = int(getattr(position, "max_hold_min", 0) or coin_profile.get("max_hold_min") or 75)
    trail = float(getattr(position, "trailing_stop_price", 0) or 0)
    thesis = str(getattr(position, "entry_thesis", "") or "")

    highest = float(getattr(position, "highest_price", entry) or entry)
    mfe_pct = max(0.0, (highest - entry) / entry) if entry > 0 else 0.0
    stall_ready = bool(
        _stall_exit_enabled()
        and hold_minutes >= _stall_min_hold_min()
        and hold_minutes + 1e-9 < float(max_hold)
        and net_pnl_pct + 1e-12 < 0.0
        and mfe_pct < _stall_max_mfe_pct()
    )
    checks = {
        "stop_loss": bool(stop > 0 and current_price <= stop),
        "trailing_stop": bool(trail > 0 and current_price <= trail and highest >= entry * (1 + float(coin_profile.get("trail") or 0.005))),
        "stall_exit": stall_ready,
        "time_stop": bool(hold_minutes >= max_hold and net_pnl_pct + 1e-12 < float(MIN_NET_PROFIT_TO_SELL)),
        "profit_target": bool(target > 0 and current_price >= target and net_pnl_pct + 1e-12 >= float(MIN_NET_PROFIT_TO_SELL) * 0.45),
        "net_profit": bool(net_pnl_pct + 1e-12 >= float(MIN_NET_PROFIT_TO_SELL)),
    }

    next_exit = "none"
    priority = [
        ("stop_loss", EXIT_STOP_LOSS),
        ("trailing_stop", EXIT_TRAILING_STOP),
        ("stall_exit", EXIT_STALL),
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

    max_hold = int(getattr(position, "max_hold_min", 0) or coin_profile.get("max_hold_min") or 75)
    stall = evaluate_stall_exit(
        entry_price=entry,
        highest_price=float(getattr(position, "highest_price", entry) or entry),
        net_pnl_pct=net_pnl_pct,
        hold_minutes=hold_minutes,
        max_hold_min=max_hold,
    )
    if stall is not None:
        return stall

    if hold_minutes >= max_hold and net_pnl_pct + 1e-12 < float(MIN_NET_PROFIT_TO_SELL):
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
        if str(te.get("action")) == "sell" or gate >= cfg.profit_floor_pct:
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
