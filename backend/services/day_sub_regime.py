"""Fast sub-regime detection layered on top of ai_context regime.

The main regime detector (`ai_market_context._market_regime_label`) uses
higher-TF EMA alignment + ADX which is a slow-moving classifier. It stays
`trending_up` even when the last few bars have clearly rejected a top and
started distribution. This module produces a fast sub-regime label from
recent-window OHLCV pattern that either agrees with the main regime
(`normal_up` / `normal_down`) or flags a divergence (`topping`,
`bottoming`, `climax_up`, `climax_down`, `distribution`, `accumulation`).

Read by:
* `day_candle_quality_gate` — extra size demotion when sub_regime disagrees
  with a BUY signal (e.g. main=trending_up but sub_regime=topping).
* `ai_signal_generator` — stamps to Redis for observability + learning.
* Learning ingestion — sub_regime becomes a labeled feature for future
  bandit routing (per (symbol, setup, sub_regime) instead of per (symbol,
  setup, regime)).

Kill switch: DAY_SUB_REGIME_ENABLED (default true).
"""

from __future__ import annotations

import os
from typing import Any


def sub_regime_enabled() -> bool:
    return os.getenv("DAY_SUB_REGIME_ENABLED", "true").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )


def _pct(a: float, b: float) -> float:
    """Percent change from a to b."""
    if a == 0:
        return 0.0
    return (b - a) / a


def compute_sub_regime(
    shape_bars: list,
    *,
    main_regime: str | None = None,
    recent_last_bar_vol_ratio: float = 1.0,
    recent_3bar_reversal_flag: int = 0,
    upper_wick_pct: float = 0.0,
    lower_wick_pct: float = 0.0,
    cs_bearish_engulfing: int = 0,
    cs_bullish_engulfing: int = 0,
    cs_shooting_star: int = 0,
    cs_hammer: int = 0,
) -> dict[str, Any]:
    """Return sub_regime label + fields.

    shape_bars: last ~25 bars, oldest-first, format [ts, o, h, l, c, v].
    main_regime: current ai_context regime label (trending_up / trending_down /
                 range / chop / sideways / bull / bear).

    Returns a dict with:
        sub_regime: one of {normal_up, normal_down, normal_range,
                            topping, bottoming, climax_up, climax_down,
                            distribution, accumulation, unknown}
        sub_regime_confidence: [0, 1]
        sub_regime_agrees_with_main: 0 | 1
        sub_regime_reason: short label
    """
    default = {
        "sub_regime": "unknown",
        "sub_regime_confidence": 0.5,
        "sub_regime_agrees_with_main": 1,
        "sub_regime_reason": "insufficient_data",
    }
    if not sub_regime_enabled():
        default["sub_regime"] = "disabled"
        return default
    if not shape_bars or len(shape_bars) < 6:
        return default

    try:
        closes = [float(b[4]) for b in shape_bars[-20:]]
        highs = [float(b[2]) for b in shape_bars[-20:]]
        lows = [float(b[3]) for b in shape_bars[-20:]]
        vols = [float(b[5]) for b in shape_bars[-20:]]
    except Exception:
        return default

    n = len(closes)
    # 5-bar momentum + 10-bar drift
    mom_5 = _pct(closes[-6], closes[-1]) if n >= 6 else 0.0
    drift_10 = _pct(closes[-11], closes[-1]) if n >= 11 else 0.0
    # Recent range vs prior range (expansion detector)
    rng_5 = max(highs[-5:]) - min(lows[-5:]) if n >= 5 else 0.0
    rng_prev5 = (
        max(highs[-10:-5]) - min(lows[-10:-5]) if n >= 10 else max(1e-12, rng_5)
    )
    rng_expansion = (rng_5 / rng_prev5) if rng_prev5 > 0 else 1.0
    # Volume expansion
    v_last5 = sum(vols[-5:]) / 5.0
    v_prev10 = sum(vols[-15:-5]) / 10.0 if n >= 15 else v_last5
    v_ratio_5v10 = (v_last5 / v_prev10) if v_prev10 > 0 else 1.0

    main = str(main_regime or "").strip().lower()
    main_up = main in ("trending_up", "bull", "uptrend")
    main_dn = main in ("trending_down", "bear", "downtrend")

    # Climax up: strong drift up + massive last-bar vol + rejection wick
    if (
        drift_10 > 0.008
        and recent_last_bar_vol_ratio >= 2.5
        and upper_wick_pct >= 0.45
    ):
        return {
            "sub_regime": "climax_up",
            "sub_regime_confidence": 0.85,
            "sub_regime_agrees_with_main": 0 if main_up else 1,
            "sub_regime_reason": f"drift10={drift_10:.4f}_vol_spike_upper_wick",
        }

    # Climax down: strong drift down + massive vol + hammer/lower_wick
    if (
        drift_10 < -0.008
        and recent_last_bar_vol_ratio >= 2.5
        and lower_wick_pct >= 0.45
    ):
        return {
            "sub_regime": "climax_down",
            "sub_regime_confidence": 0.85,
            "sub_regime_agrees_with_main": 0 if main_dn else 1,
            "sub_regime_reason": f"drift10={drift_10:.4f}_vol_spike_lower_wick",
        }

    # Topping: main regime says up but recent bars show reversal signals
    topping_signals = 0
    if recent_3bar_reversal_flag == 1:
        topping_signals += 1
    if cs_bearish_engulfing == 1 or cs_shooting_star == 1:
        topping_signals += 2
    if mom_5 < 0 and drift_10 > 0.003:  # recent stall after uptrend
        topping_signals += 1
    if upper_wick_pct >= 0.45:
        topping_signals += 1
    if main_up and topping_signals >= 2:
        conf = min(1.0, 0.55 + 0.15 * (topping_signals - 2))
        return {
            "sub_regime": "topping",
            "sub_regime_confidence": conf,
            "sub_regime_agrees_with_main": 0,
            "sub_regime_reason": f"topping_signals={topping_signals}",
        }

    # Bottoming: main regime says down but recent bars show reversal signals
    bottoming_signals = 0
    if cs_bullish_engulfing == 1 or cs_hammer == 1:
        bottoming_signals += 2
    if mom_5 > 0 and drift_10 < -0.003:  # recent bounce after downtrend
        bottoming_signals += 1
    if lower_wick_pct >= 0.45:
        bottoming_signals += 1
    if main_dn and bottoming_signals >= 2:
        conf = min(1.0, 0.55 + 0.15 * (bottoming_signals - 2))
        return {
            "sub_regime": "bottoming",
            "sub_regime_confidence": conf,
            "sub_regime_agrees_with_main": 0,
            "sub_regime_reason": f"bottoming_signals={bottoming_signals}",
        }

    # Distribution: volume up, price flat (accumulation-like but at highs)
    if rng_expansion < 0.9 and v_ratio_5v10 > 1.3 and mom_5 < 0.001 and drift_10 > 0.003:
        return {
            "sub_regime": "distribution",
            "sub_regime_confidence": 0.7,
            "sub_regime_agrees_with_main": 0 if main_up else 1,
            "sub_regime_reason": f"vol_5v10={v_ratio_5v10:.2f}_flat_price",
        }

    # Accumulation: volume up, price flat at recent lows
    if rng_expansion < 0.9 and v_ratio_5v10 > 1.3 and abs(mom_5) < 0.001 and drift_10 < -0.003:
        return {
            "sub_regime": "accumulation",
            "sub_regime_confidence": 0.7,
            "sub_regime_agrees_with_main": 0 if main_dn else 1,
            "sub_regime_reason": f"vol_5v10={v_ratio_5v10:.2f}_flat_at_lows",
        }

    # Normal: agrees with main regime
    if main_up:
        return {"sub_regime": "normal_up", "sub_regime_confidence": 0.7,
                "sub_regime_agrees_with_main": 1, "sub_regime_reason": "aligned_up"}
    if main_dn:
        return {"sub_regime": "normal_down", "sub_regime_confidence": 0.7,
                "sub_regime_agrees_with_main": 1, "sub_regime_reason": "aligned_down"}
    return {"sub_regime": "normal_range", "sub_regime_confidence": 0.6,
            "sub_regime_agrees_with_main": 1, "sub_regime_reason": "range_or_chop"}


__all__ = ["compute_sub_regime", "sub_regime_enabled"]
