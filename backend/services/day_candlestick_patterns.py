"""Candlestick pattern detection for DAY signal enrichment.

feature_builder.py already computes 27+ core technical indicators (RSI, MACD,
BBands, ADX, etc) but has zero classic candlestick pattern detectors. The
patterns a human reads on a chart — hammer, shooting star, doji, engulfing,
inside bar, outside bar — carry directional information that scalar
indicators do not encode.

These are computed off the same `_shape_bars` list ai_signal_generator
already has in memory (last ~25 bars of 1m OHLCV for DAY), stamped into the
Redis ai_signal payload as integer flags, propagated to decision_data by
portfolio_engine_integration, and read by day_candle_quality_gate for soft
demotion of BUY signals emitted on top-rejection patterns.

No ML retrain needed — these are separate soft signals, not features on
the 145-dim model vector. Kill switch: DAY_CANDLESTICK_PATTERNS_ENABLED.
"""

from __future__ import annotations

import os
from typing import Any


def candlestick_patterns_enabled() -> bool:
    return os.getenv("DAY_CANDLESTICK_PATTERNS_ENABLED", "true").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )


def _bar_geom(b: list | tuple) -> dict[str, float]:
    """Return {o,h,l,c,rng,body,upper_wick,lower_wick,body_pct,up_pct,low_pct,is_up,is_dn}."""
    o = float(b[1])
    h = float(b[2])
    l = float(b[3])
    c = float(b[4])
    rng = max(h - l, 1e-12)
    body_abs = abs(c - o)
    upper = h - max(c, o)
    lower = min(c, o) - l
    return {
        "o": o,
        "h": h,
        "l": l,
        "c": c,
        "rng": rng,
        "body": body_abs,
        "upper": upper,
        "lower": lower,
        "body_pct": body_abs / rng,
        "up_pct": upper / rng,
        "low_pct": lower / rng,
        "is_up": 1 if c > o else 0,
        "is_dn": 1 if c < o else 0,
    }


def _is_hammer(g: dict[str, float]) -> bool:
    """Long lower wick, small body, small/no upper wick — bullish rejection at lows."""
    return g["low_pct"] >= 0.55 and g["body_pct"] <= 0.35 and g["up_pct"] <= 0.20


def _is_shooting_star(g: dict[str, float]) -> bool:
    """Long upper wick, small body, small/no lower wick — bearish rejection at highs."""
    return g["up_pct"] >= 0.55 and g["body_pct"] <= 0.35 and g["low_pct"] <= 0.20


def _is_doji(g: dict[str, float]) -> bool:
    """Tiny body — indecision / potential reversal signal."""
    return g["body_pct"] <= 0.10 and g["rng"] > 0


def _is_bullish_engulfing(prev: dict[str, float], curr: dict[str, float]) -> bool:
    """Current bull candle body fully engulfs previous bear candle body."""
    return prev["is_dn"] == 1 and curr["is_up"] == 1 and curr["c"] >= prev["o"] and curr["o"] <= prev["c"] and curr["body"] > prev["body"] * 1.0


def _is_bearish_engulfing(prev: dict[str, float], curr: dict[str, float]) -> bool:
    """Current bear candle body fully engulfs previous bull candle body."""
    return prev["is_up"] == 1 and curr["is_dn"] == 1 and curr["o"] >= prev["c"] and curr["c"] <= prev["o"] and curr["body"] > prev["body"] * 1.0


def _is_inside_bar(prev: dict[str, float], curr: dict[str, float]) -> bool:
    """Current bar's range is contained within previous bar — consolidation."""
    return curr["h"] <= prev["h"] and curr["l"] >= prev["l"]


def _is_outside_bar(prev: dict[str, float], curr: dict[str, float]) -> bool:
    """Current bar's range fully contains previous — expansion / volatility."""
    return curr["h"] >= prev["h"] and curr["l"] <= prev["l"] and curr["rng"] > prev["rng"]


def _is_three_black_crows(bars: list[dict[str, float]]) -> bool:
    """Three consecutive bear bars, each closing lower than prior, body-dominant."""
    if len(bars) < 3:
        return False
    b1, b2, b3 = bars[-3], bars[-2], bars[-1]
    return b1["is_dn"] == 1 and b2["is_dn"] == 1 and b3["is_dn"] == 1 and b2["c"] < b1["c"] and b3["c"] < b2["c"] and b1["body_pct"] >= 0.5 and b2["body_pct"] >= 0.5 and b3["body_pct"] >= 0.5


def _is_three_white_soldiers(bars: list[dict[str, float]]) -> bool:
    """Three consecutive bull bars, each closing higher than prior, body-dominant."""
    if len(bars) < 3:
        return False
    b1, b2, b3 = bars[-3], bars[-2], bars[-1]
    return b1["is_up"] == 1 and b2["is_up"] == 1 and b3["is_up"] == 1 and b2["c"] > b1["c"] and b3["c"] > b2["c"] and b1["body_pct"] >= 0.5 and b2["body_pct"] >= 0.5 and b3["body_pct"] >= 0.5


def detect_patterns(shape_bars: list) -> dict[str, int]:
    """Detect candlestick patterns from the last N bars.

    Returns a dict of int flags (0/1). Missing bars → all zeros.
    Bearish patterns get suffix `_bear`, bullish `_bull`, neutral `_neutral`.

    Downstream consumers:
    * day_candlestick_patterns:candle_quality_gate → soft demote BUY when
      bearish patterns present (shooting_star / bearish_engulfing /
      three_black_crows).
    * ai_signal_generator → stamps as Redis fields
      cs_pat_{name} for observability + learning ingestion.
    """
    default = {
        "cs_pat_hammer_bull": 0,
        "cs_pat_shooting_star_bear": 0,
        "cs_pat_doji_neutral": 0,
        "cs_pat_bullish_engulfing_bull": 0,
        "cs_pat_bearish_engulfing_bear": 0,
        "cs_pat_inside_bar_neutral": 0,
        "cs_pat_outside_bar_neutral": 0,
        "cs_pat_three_white_soldiers_bull": 0,
        "cs_pat_three_black_crows_bear": 0,
    }
    if not candlestick_patterns_enabled():
        return default
    if not shape_bars or len(shape_bars) < 1:
        return default

    try:
        g_last = _bar_geom(shape_bars[-1])
    except Exception:
        return default

    # Single-bar patterns
    if _is_hammer(g_last):
        default["cs_pat_hammer_bull"] = 1
    if _is_shooting_star(g_last):
        default["cs_pat_shooting_star_bear"] = 1
    if _is_doji(g_last):
        default["cs_pat_doji_neutral"] = 1

    # 2-bar patterns
    if len(shape_bars) >= 2:
        try:
            g_prev = _bar_geom(shape_bars[-2])
            if _is_bullish_engulfing(g_prev, g_last):
                default["cs_pat_bullish_engulfing_bull"] = 1
            if _is_bearish_engulfing(g_prev, g_last):
                default["cs_pat_bearish_engulfing_bear"] = 1
            if _is_inside_bar(g_prev, g_last):
                default["cs_pat_inside_bar_neutral"] = 1
            if _is_outside_bar(g_prev, g_last):
                default["cs_pat_outside_bar_neutral"] = 1
        except Exception:
            pass

    # 3-bar patterns
    if len(shape_bars) >= 3:
        try:
            geoms3 = [_bar_geom(b) for b in shape_bars[-3:]]
            if _is_three_black_crows(geoms3):
                default["cs_pat_three_black_crows_bear"] = 1
            if _is_three_white_soldiers(geoms3):
                default["cs_pat_three_white_soldiers_bull"] = 1
        except Exception:
            pass

    return default


def net_bearish_pattern_score(flags: dict[str, Any]) -> float:
    """Aggregate a signed pattern-bias score in [-1, +1].

    Positive = bullish patterns dominate; negative = bearish patterns dominate.
    Used by day_candle_quality_gate to combine with volume/wick signals.
    """

    def _f(k: str) -> int:
        try:
            return int(float(flags.get(k, 0) or 0))
        except (TypeError, ValueError):
            return 0

    bull = _f("cs_pat_hammer_bull") * 0.6 + _f("cs_pat_bullish_engulfing_bull") * 0.8 + _f("cs_pat_three_white_soldiers_bull") * 1.0
    bear = _f("cs_pat_shooting_star_bear") * 0.6 + _f("cs_pat_bearish_engulfing_bear") * 0.8 + _f("cs_pat_three_black_crows_bear") * 1.0
    net = bull - bear
    return max(-1.0, min(1.0, net))


__all__ = [
    "candlestick_patterns_enabled",
    "detect_patterns",
    "net_bearish_pattern_score",
]
