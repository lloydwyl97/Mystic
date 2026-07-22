"""
TREND_BREAKOUT_PULLBACK_SLEEVE (formerly labeled ALLWEATHER_BREAKOUT_PULLBACK) — trend-only sleeve adapter.

This sleeve ONLY trades BREAKOUT and TREND_PULLBACK setups in trend_up (and limited qualifying neutral).
It is NOT an all-weather engine by itself.

It stays flat in trend_down, range, chop, and most neutral structures.

See sleeve_characteristics() for regimes traded / flat, expected frequency and idle time.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from backend.services.allweather_signal_engine import (
    REG_NEUTRAL,
    REG_RANGE,
    REG_TREND_DOWN,
    REG_TREND_UP,
    SETUP_BREAKOUT,
    SETUP_FAILED_BREAKDOWN_REVERSAL,
    SETUP_RANGE_BOUNCE,
    SETUP_TREND_PULLBACK,
    compute_state,
    diagnose_entry_state,
    entry_levels,
    entry_signal,
    exit_decision,
    normalize_bars,
)
from backend.services.day_trade_thesis import SETUP_BREAKOUT_CONTINUATION, SETUP_HTF_TREND_PULLBACK

# Honest sleeve identity (not "all weather")
SLEEVE_NAME = "TREND_BREAKOUT_PULLBACK_SLEEVE"
STRATEGY_FAMILY = "ALLWEATHER_BREAKOUT_PULLBACK"  # legacy internal tag kept for continuity in logs/DB
CANDIDATE_ID = "allweather_breakout_pullback_lab_1_5x"

EXIT_ATR_TARGET = "ALLWEATHER_ATR_TARGET_EXIT"
EXIT_ATR_STOP = "ALLWEATHER_ATR_STOP_EXIT"
EXIT_TIME_STOP = "ALLWEATHER_TIME_STOP_EXIT"

# Legacy aliases (still recognized on read paths)
_LEGACY_EXIT_MAP = {
    "ALLWEATHER_TARGET": EXIT_ATR_TARGET,
    "ALLWEATHER_STOP": EXIT_ATR_STOP,
    "ALLWEATHER_TIME_STOP": EXIT_TIME_STOP,
}

SHADOW_SNAPSHOT_PATH = Path(__file__).resolve().parents[2] / "scripts" / "replay_baselines" / "allweather_breakout_pullback_shadow_latest.json"

TOP_FOUR = frozenset({"BTC/USDT", "ETH/USDT", "SOL/USDT", "XRP/USDT"})

_telemetry: dict[str, Any] = {
    "evaluated_cycles": 0,
    "symbols_evaluated": 0,
    "would_buy_count": 0,
    "no_signal_count": 0,
    "eval_error_count": 0,
    "latest_no_signal_reasons": {},
    "last_bar_cycle_at": None,
}

# Idle alarm state (diagnostic only — not a trade blocker)
_last_executable_signal_ts: float | None = None  # unix ts of last time we had a would-buy / executable signal
_IDLE_ALARM_THRESHOLDS_H = (24.0, 48.0, 72.0)


def sleeve_characteristics() -> dict[str, Any]:
    """Honest reporting of what this sleeve does and does not trade."""
    return {
        "sleeve_name": SLEEVE_NAME,
        "internal_family": STRATEGY_FAMILY,
        "trades_regimes": [
            "trend_up (breakout/pullback)",
            "range (RANGE_BOUNCE)",
            "trend_down / bear (FAILED_BREAKDOWN_REVERSAL)",
            "neutral (limited)",
        ],
        "flat_regimes": ["pure chop with no reversal structure"],
        "expected_trade_frequency": {
            "trend_up": "low-moderate",
            "range / down": "moderate when reversal structures appear (to generate learnable outcomes)",
        },
        "notes": [
            "Expanded to produce trades and outcomes in current market regimes so the system learns and can improve.",
            "Paper and future live use identical rules. Conservative ATR brackets and exits preserved.",
            "Sitting idle for days is avoided by design now.",
        ],
    }


@dataclass
class AllweatherBpEvalOutcome:
    ok: bool = False
    eval_error: bool = False
    error_meta: dict[str, Any] | None = None
    no_signal_diag: dict[str, Any] | None = None
    signal: dict[str, Any] | None = None
    stop: float = 0.0
    target: float = 0.0
    current_price: float = 0.0
    atr: float = 0.0


def _env_bool(name: str, default: str) -> bool:
    return os.getenv(name, default).strip().lower() in ("1", "true", "yes", "on")


def execution_enabled() -> bool:
    """Real entries/exits — default OFF."""
    return _env_bool("ALLWEATHER_BREAKOUT_PULLBACK_ENABLED", "false")


def shadow_enabled() -> bool:
    """Evaluate + log hypothetical decisions — default ON for integration review."""
    return _env_bool("ALLWEATHER_BREAKOUT_PULLBACK_SHADOW", "true")


def adapter_active() -> bool:
    return execution_enabled() or shadow_enabled()


def uses_atr_bracket_exits() -> bool:
    """This family never uses MIN_NET_PROFIT_TO_SELL as primary exit."""
    return True


def normalize_exit_reason(reason: str) -> str:
    r = str(reason or "").strip()
    return _LEGACY_EXIT_MAP.get(r, r)


def allweather_setup_to_production(setup: str) -> str:
    if setup == SETUP_BREAKOUT:
        return SETUP_BREAKOUT_CONTINUATION
    if setup == SETUP_TREND_PULLBACK:
        return SETUP_HTF_TREND_PULLBACK
    if setup in (SETUP_FAILED_BREAKDOWN_REVERSAL, SETUP_RANGE_BOUNCE):
        # Map reversal/bounce to production thesis types that the router recognizes for non-bull
        return setup
    return str(setup or "")


def allweather_regime_to_day_regime(aw_regime: str) -> str:
    from backend.services.day_regime_router import (
        DAY_REGIME_BEAR,
        DAY_REGIME_BULL,
        DAY_REGIME_NEUTRAL,
        DAY_REGIME_RANGE,
    )

    if aw_regime == REG_TREND_UP:
        return DAY_REGIME_BULL
    if aw_regime == REG_TREND_DOWN:
        return DAY_REGIME_BEAR
    if aw_regime == REG_RANGE:
        return DAY_REGIME_RANGE
    return DAY_REGIME_NEUTRAL


def is_allweather_strategy_family(strategy_family: str | None) -> bool:
    return str(strategy_family or "").strip().upper() == STRATEGY_FAMILY


def is_allweather_position(position: Any) -> bool:
    sf = getattr(position, "strategy_family", None) or getattr(position, "entry_strategy_id", "")
    if is_allweather_strategy_family(str(sf)):
        return True
    thesis = str(getattr(position, "entry_thesis", "") or "")
    return thesis in (SETUP_BREAKOUT, SETUP_TREND_PULLBACK) and bool(getattr(position, "thesis_target_level", 0.0) and getattr(position, "thesis_invalid_level", 0.0))


def apply_signal_to_decision_data(
    dd: dict[str, Any],
    *,
    symbol: str,
    sig: dict[str, Any],
    current_price: float,
    atr: float,
) -> dict[str, Any]:
    """Stamp candidate fields for routing, ranking, and bracket exits."""
    target, stop = entry_levels(current_price, atr, float(sig["target_atr"]), float(sig["stop_atr"]))
    out = dict(dd or {})
    setup = str(sig["setup"])
    aw_regime = str(sig["regime"])
    out["sleeve_name"] = SLEEVE_NAME
    out["strategy_family"] = STRATEGY_FAMILY
    out["candidate_id"] = CANDIDATE_ID
    out["live_ai_strategy"] = STRATEGY_FAMILY
    out["setup_type"] = setup
    out["entry_thesis"] = setup
    out["allweather_setup"] = setup
    out["allweather_regime"] = aw_regime
    out["production_setup_type"] = allweather_setup_to_production(setup)
    out["day_route_regime"] = allweather_regime_to_day_regime(aw_regime)
    out["thesis_invalid_level"] = stop
    out["thesis_target_level"] = target
    out["thesis_score"] = max(float(out.get("thesis_score") or 0.0), 0.7)
    out["allweather_bracket_exit"] = True
    out["profit_floor_applies"] = False
    out["min_net_profit_floor_bypass"] = True
    return out


def evaluate_production_route(
    *,
    symbol: str,
    setup: str,
    aw_regime: str,
    decision_data: dict[str, Any],
    context_payload: dict[str, Any] | None = None,
    current_price: float = 0.0,
    thesis_score: float = 0.55,
) -> dict[str, Any]:
    """Family-aware production router — exempt from neutral-VWAP-only baseline kills."""
    from backend.services.day_regime_router import evaluate_day_entry_route

    prod_setup = allweather_setup_to_production(setup)
    day_regime = allweather_regime_to_day_regime(aw_regime)
    return evaluate_day_entry_route(
        setup_type=prod_setup,
        day_regime=day_regime,
        decision_data=decision_data,
        context_payload=context_payload,
        current_price=current_price,
        thesis_score=thesis_score,
        strategy_family=STRATEGY_FAMILY,
    )


def evaluate_production_bucket(
    *,
    symbol: str,
    setup: str,
    aw_regime: str,
    bucket_stats: dict | None = None,
) -> dict[str, Any]:
    from backend.services.day_bucket_quality import evaluate_bucket_entry

    prod_setup = allweather_setup_to_production(setup)
    day_regime = allweather_regime_to_day_regime(aw_regime)
    return evaluate_bucket_entry(
        symbol=symbol,
        regime=day_regime,
        setup=prod_setup,
        bucket_stats=bucket_stats,
        strategy_family=STRATEGY_FAMILY,
    )


def bracket_exit_decision(
    *,
    current_price: float,
    bar_low: float,
    bar_high: float,
    target_level: float,
    stop_level: float,
    hold_hours: float,
) -> dict[str, str] | None:
    dec = exit_decision(
        current_price=current_price,
        bar_low=bar_low,
        bar_high=bar_high,
        target_level=target_level,
        stop_level=stop_level,
        hold_hours=hold_hours,
    )
    if not dec:
        return None
    reason = normalize_exit_reason(str(dec.get("reason") or ""))
    return {"action": "sell", "reason": reason}


def write_shadow_snapshot(entry: dict[str, Any]) -> None:
    """Append hypothetical decision to rolling shadow artifact (no ledger impact)."""
    try:
        SHADOW_SNAPSHOT_PATH.parent.mkdir(parents=True, exist_ok=True)
        payload: dict[str, Any]
        if SHADOW_SNAPSHOT_PATH.exists():
            payload = json.loads(SHADOW_SNAPSHOT_PATH.read_text())
        else:
            payload = {
                "sleeve_name": SLEEVE_NAME,
                "strategy_family": STRATEGY_FAMILY,
                "candidate_id": CANDIDATE_ID,
                "shadow_enabled": True,
                "execution_enabled": False,
                "entries": [],
            }
        payload["updated_at"] = datetime.now(timezone.utc).isoformat()
        payload["shadow_enabled"] = shadow_enabled()
        payload["execution_enabled"] = execution_enabled()
        entries = list(payload.get("entries") or [])
        entry = {**entry, "ts": datetime.now(timezone.utc).isoformat()}
        entries.append(entry)
        payload["entries"] = entries[-500:]
        payload["entry_count"] = len(payload["entries"])
        SHADOW_SNAPSHOT_PATH.write_text(json.dumps(payload, indent=2))
    except Exception:
        pass


def get_telemetry_snapshot() -> dict[str, Any]:
    snap = dict(_telemetry)
    snap["sleeve_name"] = SLEEVE_NAME
    snap["sleeve_characteristics"] = sleeve_characteristics()
    return snap


def record_eval_outcome(outcome: AllweatherBpEvalOutcome, *, symbol: str) -> None:
    _telemetry["symbols_evaluated"] = int(_telemetry.get("symbols_evaluated") or 0) + 1
    if outcome.eval_error:
        _telemetry["eval_error_count"] = int(_telemetry.get("eval_error_count") or 0) + 1
    elif outcome.ok:
        _telemetry["would_buy_count"] = int(_telemetry.get("would_buy_count") or 0) + 1
        record_executable_signal()
    else:
        _telemetry["no_signal_count"] = int(_telemetry.get("no_signal_count") or 0) + 1
        if outcome.no_signal_diag:
            reasons = dict(_telemetry.get("latest_no_signal_reasons") or {})
            reasons[symbol] = outcome.no_signal_diag
            _telemetry["latest_no_signal_reasons"] = reasons


def record_executable_signal() -> None:
    """Call when we produce a would-buy / executable signal (not just no-signal)."""
    global _last_executable_signal_ts
    import time as _time

    _last_executable_signal_ts = _time.time()
    _telemetry["last_executable_signal_ts"] = _last_executable_signal_ts


def check_paper_idle_alarm() -> dict[str, Any]:
    """Diagnostic idle alarm. Returns current status; logs warnings at thresholds."""
    import time as _time
    import logging

    logger = logging.getLogger(__name__)
    now = _time.time()
    last = _last_executable_signal_ts or _telemetry.get("last_executable_signal_ts")
    if last is None:
        idle_h = 999.0
    else:
        idle_h = (now - float(last)) / 3600.0

    alarms = {}
    for thresh in _IDLE_ALARM_THRESHOLDS_H:
        key = f"idle_{int(thresh)}h"
        alarms[key] = idle_h >= thresh
        if alarms[key]:
            logger.warning(
                "ALLWEATHER_PAPER_IDLE_ALARM_%dh sleeve=%s idle_hours=%.1f — regime/fallback review recommended (diagnostic only)",
                int(thresh),
                SLEEVE_NAME,
                idle_h,
            )

    return {
        "sleeve": SLEEVE_NAME,
        "idle_hours": round(idle_h, 1) if idle_h < 999 else None,
        "last_executable_ts": last,
        "alarms": alarms,
        "thresholds_h": list(_IDLE_ALARM_THRESHOLDS_H),
        "is_diagnostic_only": True,
    }


def begin_bar_cycle_telemetry() -> None:
    _telemetry["evaluated_cycles"] = int(_telemetry.get("evaluated_cycles") or 0) + 1
    _telemetry["last_bar_cycle_at"] = datetime.now(timezone.utc).isoformat()


def write_shadow_heartbeat(
    *,
    open_positions: int = 0,
    real_orders_permitted: bool = False,
    kline_fetch_stats: dict[str, Any] | None = None,
) -> None:
    """Persist heartbeat artifact even when would_buy_count is zero."""
    if not shadow_enabled() and not execution_enabled():
        return
    try:
        SHADOW_SNAPSHOT_PATH.parent.mkdir(parents=True, exist_ok=True)
        payload: dict[str, Any] = {
            "strategy_family": STRATEGY_FAMILY,
            "candidate_id": CANDIDATE_ID,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "shadow_enabled": shadow_enabled(),
            "execution_enabled": execution_enabled(),
            "evaluated_cycles": int(_telemetry.get("evaluated_cycles") or 0),
            "symbols_evaluated": int(_telemetry.get("symbols_evaluated") or 0),
            "would_buy_count": int(_telemetry.get("would_buy_count") or 0),
            "no_signal_count": int(_telemetry.get("no_signal_count") or 0),
            "eval_error_count": int(_telemetry.get("eval_error_count") or 0),
            "latest_no_signal_reasons": dict(_telemetry.get("latest_no_signal_reasons") or {}),
            "open_positions": open_positions,
            "real_orders_permitted": bool(real_orders_permitted),
            "entries": [],
            "entry_count": 0,
            "heartbeat": True,
        }
        if SHADOW_SNAPSHOT_PATH.exists():
            try:
                prior = json.loads(SHADOW_SNAPSHOT_PATH.read_text())
                if isinstance(prior.get("entries"), list):
                    payload["entries"] = prior["entries"][-500:]
                    payload["entry_count"] = len(payload["entries"])
            except (json.JSONDecodeError, OSError):
                pass
        if kline_fetch_stats:
            payload["kline_fetch_stats"] = kline_fetch_stats
        # Idle alarm (diagnostic)
        try:
            payload["idle_alarm"] = check_paper_idle_alarm()
            payload["sleeve_name"] = SLEEVE_NAME
        except Exception:
            pass
        SHADOW_SNAPSHOT_PATH.write_text(json.dumps(payload, indent=2))
    except Exception:
        pass


async def evaluate_breakout_pullback_candidate(
    *,
    symbol: str,
    current_price: float,
    to_api_symbol_fn: Any,
) -> AllweatherBpEvalOutcome:
    """Evaluate AW signal with kline meta + structured no-signal diagnostics."""
    if not adapter_active():
        return AllweatherBpEvalOutcome(
            no_signal_diag=diagnose_entry_state(None, symbol=symbol, kline_fetch_failed=False),
        )
    from backend.services.live_market_data import live_market_data_service

    api_symbol = to_api_symbol_fn(symbol)
    fetch_meta = await live_market_data_service.get_ohlcv_with_meta(api_symbol, "1h", limit=260)
    if fetch_meta.get("kline_fetch_failed"):
        err = AllweatherBpEvalOutcome(
            eval_error=True,
            error_meta={
                "error_type": fetch_meta.get("error_type"),
                "endpoint": fetch_meta.get("endpoint"),
                "symbol": api_symbol,
                "timeframe": "1h",
                "retry_count": int(fetch_meta.get("retry_count") or 0),
                "used_cache": bool(fetch_meta.get("used_cache")),
                "recovered": bool(fetch_meta.get("recovered")),
            },
        )
        record_eval_outcome(err, symbol=symbol)
        return err

    raw = fetch_meta.get("rows")
    bars = normalize_bars(raw)
    state = compute_state(bars)
    ts = datetime.now(timezone.utc).isoformat()
    if state is None:
        diag = diagnose_entry_state(None, bar_count=len(bars), symbol=symbol, timestamp=ts)
        out = AllweatherBpEvalOutcome(no_signal_diag=diag)
        record_eval_outcome(out, symbol=symbol)
        return out

    sig = entry_signal(state)
    if fetch_meta.get("used_cache"):
        diag = diagnose_entry_state(state, bar_count=len(bars), symbol=symbol, timestamp=ts)
        diag["used_stale_kline_cache"] = True
        if not sig:
            out = AllweatherBpEvalOutcome(no_signal_diag=diag)
            record_eval_outcome(out, symbol=symbol)
            return out
        else:
            # Signal found but klines are stale — shadow-only, must not execute
            logger.debug(f"[AW_BP] {symbol} stale kline cache — signal present but not executable")
            diag["shadow_signal"] = sig
            out = AllweatherBpEvalOutcome(no_signal_diag=diag)
            record_eval_outcome(out, symbol=symbol)
            return out

    if not sig:
        diag = diagnose_entry_state(state, bar_count=len(bars), symbol=symbol, timestamp=ts)
        out = AllweatherBpEvalOutcome(no_signal_diag=diag)
        record_eval_outcome(out, symbol=symbol)
        return out

    cur = float(current_price or state.close or 0.0)
    target, stop = entry_levels(cur, state.atr, float(sig["target_atr"]), float(sig["stop_atr"]))
    if target <= 0 or stop <= 0:
        diag = diagnose_entry_state(state, bar_count=len(bars), symbol=symbol, timestamp=ts)
        diag["invalid_bracket_levels"] = True
        out = AllweatherBpEvalOutcome(no_signal_diag=diag)
        record_eval_outcome(out, symbol=symbol)
        return out

    out = AllweatherBpEvalOutcome(ok=True, signal=sig, stop=stop, target=target, current_price=cur, atr=state.atr)
    record_eval_outcome(out, symbol=symbol)
    return out


def log_shadow_entry(
    *,
    symbol: str,
    action: str,
    setup: str,
    aw_regime: str,
    price: float,
    stop: float,
    target: float,
    extra: dict[str, Any] | None = None,
) -> None:
    if not shadow_enabled():
        return
    write_shadow_snapshot(
        {
            "action": action,
            "symbol": symbol,
            "setup": setup,
            "aw_regime": aw_regime,
            "price": round(price, 8),
            "stop": round(stop, 8),
            "target": round(target, 8),
            "would_execute": execution_enabled(),
            **(extra or {}),
        }
    )


__all__ = [
    "CANDIDATE_ID",
    "EXIT_ATR_STOP",
    "EXIT_ATR_TARGET",
    "EXIT_TIME_STOP",
    "STRATEGY_FAMILY",
    "TOP_FOUR",
    "AllweatherBpEvalOutcome",
    "adapter_active",
    "allweather_regime_to_day_regime",
    "allweather_setup_to_production",
    "apply_signal_to_decision_data",
    "begin_bar_cycle_telemetry",
    "bracket_exit_decision",
    "compute_state",
    "entry_signal",
    "evaluate_breakout_pullback_candidate",
    "evaluate_production_bucket",
    "evaluate_production_route",
    "execution_enabled",
    "get_telemetry_snapshot",
    "is_allweather_position",
    "is_allweather_strategy_family",
    "log_shadow_entry",
    "normalize_bars",
    "normalize_exit_reason",
    "record_eval_outcome",
    "shadow_enabled",
    "uses_atr_bracket_exits",
    "write_shadow_heartbeat",
    "write_shadow_snapshot",
]
