"""
day_chart_pattern_detector — rule-based chart pattern recognition (not ML).

Detects classic technical structures from swing highs/lows on OHLCV closes:
double top / double bottom, higher-high+higher-low / lower-high+lower-low trend
structure, triangle/wedge convergence, and support/resistance breakout.

Deliberately NOT added to the 145-dim ML feature vector — that would force a
breaking dimension bump and a live-signal-generation gap while every model
retrains. Instead this feeds a small bounded ranking nudge (same pattern as
day_regime_transition.py's ±0.04 cap) and is captured into
ai_candidate_snapshots (chart_pattern_score/chart_pattern_label) so the forward
returns already computed for every snapshot let it be validated exactly like
regime labels — see ai_regime_validation.get_pattern_validated_scalar. Nudge
strength scales with proven edge, never blindly trusted.
"""

from __future__ import annotations

import logging
import math
from typing import Any

logger = logging.getLogger(__name__)

RANK_DELTA_CAP = 0.04
_DEFAULT_LOOKBACK = 3
_DEFAULT_TF = "15m"
_MIN_BARS = 2 * _DEFAULT_LOOKBACK + 5
_SWING_TOLERANCE_PCT = 0.006  # 0.6% — "equal" swing highs/lows for double top/bottom


def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        v = float(x)
        return v if math.isfinite(v) else default
    except (TypeError, ValueError):
        return default


def _extract_ohlc(rows: list[list]) -> tuple[list[float], list[float], list[float]]:
    closes: list[float] = []
    highs: list[float] = []
    lows: list[float] = []
    for r in rows or []:
        if not isinstance(r, (list, tuple)) or len(r) < 5:
            continue
        highs.append(_safe_float(r[2]))
        lows.append(_safe_float(r[3]))
        closes.append(_safe_float(r[4]))
    return closes, highs, lows


def find_swing_points(highs: list[float], lows: list[float], *, lookback: int = _DEFAULT_LOOKBACK) -> tuple[list[tuple[int, float]], list[tuple[int, float]]]:
    """Local-extrema swing points: bar i is a swing high if its high is the unique
    max within [i-lookback, i+lookback] (swing low: unique min of lows, symmetric).
    Returns (swing_highs, swing_lows) as (index, price) pairs, oldest-first."""
    n = len(highs)
    swing_highs: list[tuple[int, float]] = []
    swing_lows: list[tuple[int, float]] = []
    if n < (2 * lookback + 1):
        return swing_highs, swing_lows
    for i in range(lookback, n - lookback):
        window_h = highs[i - lookback : i + lookback + 1]
        if highs[i] == max(window_h) and window_h.count(highs[i]) == 1:
            swing_highs.append((i, highs[i]))
        window_l = lows[i - lookback : i + lookback + 1]
        if lows[i] == min(window_l) and window_l.count(lows[i]) == 1:
            swing_lows.append((i, lows[i]))
    return swing_highs, swing_lows


def _double_top_bottom(swing_highs: list[tuple[int, float]], swing_lows: list[tuple[int, float]], last_price: float) -> tuple[float, str | None]:
    if last_price <= 0:
        return 0.0, None
    if len(swing_highs) >= 2:
        (i1, h1), (i2, h2) = swing_highs[-2], swing_highs[-1]
        if i2 > i1 and abs(h1 - h2) / max(h1, h2, 1e-9) <= _SWING_TOLERANCE_PCT:
            trough = min((lo for idx, lo in swing_lows if i1 < idx < i2), default=None)
            if trough is not None and last_price < trough:
                return -0.80, "DOUBLE_TOP"
            return -0.35, "DOUBLE_TOP_FORMING"
    if len(swing_lows) >= 2:
        (i1, l1), (i2, l2) = swing_lows[-2], swing_lows[-1]
        if i2 > i1 and abs(l1 - l2) / max(l1, l2, 1e-9) <= _SWING_TOLERANCE_PCT:
            peak = max((hi for idx, hi in swing_highs if i1 < idx < i2), default=None)
            if peak is not None and last_price > peak:
                return 0.80, "DOUBLE_BOTTOM"
            return 0.35, "DOUBLE_BOTTOM_FORMING"
    return 0.0, None


def _trend_structure(swing_highs: list[tuple[int, float]], swing_lows: list[tuple[int, float]]) -> tuple[float, str | None]:
    if len(swing_highs) < 2 or len(swing_lows) < 2:
        return 0.0, None
    higher_high = swing_highs[-1][1] > swing_highs[-2][1]
    higher_low = swing_lows[-1][1] > swing_lows[-2][1]
    lower_high = swing_highs[-1][1] < swing_highs[-2][1]
    lower_low = swing_lows[-1][1] < swing_lows[-2][1]
    if higher_high and higher_low:
        return 0.50, "UPTREND_STRUCTURE"
    if lower_high and lower_low:
        return -0.50, "DOWNTREND_STRUCTURE"
    return 0.0, None


def _triangle_wedge(swing_highs: list[tuple[int, float]], swing_lows: list[tuple[int, float]]) -> tuple[float, str | None]:
    if len(swing_highs) < 2 or len(swing_lows) < 2:
        return 0.0, None
    high_slope = swing_highs[-1][1] - swing_highs[-2][1]
    low_slope = swing_lows[-1][1] - swing_lows[-2][1]
    if high_slope < 0 and low_slope > 0:
        return 0.20, "SYMMETRICAL_TRIANGLE"
    if high_slope < 0 and low_slope <= 0 and abs(low_slope) < abs(high_slope):
        return -0.25, "DESCENDING_TRIANGLE"
    if high_slope >= 0 and low_slope > 0 and abs(high_slope) < abs(low_slope):
        return 0.25, "ASCENDING_TRIANGLE"
    return 0.0, None


def _support_resistance_breakout(swing_highs: list[tuple[int, float]], swing_lows: list[tuple[int, float]], last_price: float) -> tuple[float, str | None]:
    if last_price <= 0:
        return 0.0, None
    if swing_highs and last_price > swing_highs[-1][1] * 1.001:
        return 0.60, "RESISTANCE_BREAKOUT"
    if swing_lows and last_price < swing_lows[-1][1] * 0.999:
        return -0.60, "SUPPORT_BREAKDOWN"
    return 0.0, None


def detect_chart_pattern(rows: list[list], *, lookback: int = _DEFAULT_LOOKBACK) -> dict[str, Any]:
    """
    rows: OHLCV rows [ts, open, high, low, close, volume], oldest-first, one timeframe.
    Returns a bounded score in [-1, 1], the winning label, and swing-point counts for
    explainability. Strongest-magnitude detected pattern wins when several fire at once
    (structural confirmations like breakout/double-top are inherently stronger signals
    than the softer trend/triangle reads, which this naturally respects via score magnitude).
    """
    closes, highs, lows = _extract_ohlc(rows)
    if len(closes) < _MIN_BARS:
        return {
            "chart_pattern_score": 0.0,
            "chart_pattern_label": "INSUFFICIENT_DATA",
            "chart_pattern_swing_highs": 0,
            "chart_pattern_swing_lows": 0,
        }

    swing_highs, swing_lows = find_swing_points(highs, lows, lookback=lookback)
    last_price = closes[-1]

    detections: list[tuple[float, str]] = []
    for score, label in (
        _double_top_bottom(swing_highs, swing_lows, last_price),
        _trend_structure(swing_highs, swing_lows),
        _triangle_wedge(swing_highs, swing_lows),
        _support_resistance_breakout(swing_highs, swing_lows, last_price),
    ):
        if label is not None:
            detections.append((score, label))

    if not detections:
        best_score, best_label = 0.0, "NO_PATTERN"
    else:
        best_score, best_label = max(detections, key=lambda d: abs(d[0]))

    return {
        "chart_pattern_score": round(max(-1.0, min(1.0, best_score)), 4),
        "chart_pattern_label": best_label,
        "chart_pattern_swing_highs": len(swing_highs),
        "chart_pattern_swing_lows": len(swing_lows),
    }


def _ccxt_symbol(symbol: str) -> str:
    s = (symbol or "").upper().replace("/", "")
    if s.endswith("USDT") and len(s) > 4:
        return f"{s[:-4]}/USDT"
    return f"{s}/USDT" if s else "BTC/USDT"


def get_chart_pattern_signal(symbol: str, *, timeframe: str = _DEFAULT_TF) -> dict[str, Any]:
    """Sync, cache-only signal lookup for the DAY candidate ranking path
    (portfolio_engine.py._score_candidate is sync). Returns a safe all-zero/neutral
    payload on any failure or missing cache — this is a bounded nudge, never a gate,
    so a miss must never block ranking."""
    neutral = {
        "chart_pattern_score": 0.0,
        "chart_pattern_label": "",
        "chart_pattern_swing_highs": 0,
        "chart_pattern_swing_lows": 0,
    }
    try:
        from backend.services.day_active_market_bundle import read_cached_day_active_bundle_sync

        bundle = read_cached_day_active_bundle_sync(_ccxt_symbol(symbol))
        if not bundle:
            return neutral
        rows = bundle.get(timeframe)
        if not isinstance(rows, list) or not rows:
            return neutral
        return detect_chart_pattern(rows)
    except Exception as exc:
        logger.debug("CHART_PATTERN_SIGNAL_FAILED %s: %s", symbol, exc)
        return neutral


def chart_pattern_rank_delta(pattern_info: dict[str, Any]) -> float:
    """Bounded ranking nudge (±RANK_DELTA_CAP) scaled by the pattern label's
    validated forward-return edge — defaults to full nudge until enough samples
    accumulate to say otherwise (same neutral-until-proven-otherwise contract as
    ai_regime_validation.get_regime_validated_scalar)."""
    score = _safe_float(pattern_info.get("chart_pattern_score"))
    label = str(pattern_info.get("chart_pattern_label") or "")
    if not label or label in ("NO_PATTERN", "INSUFFICIENT_DATA") or score == 0.0:
        return 0.0
    raw_delta = max(-RANK_DELTA_CAP, min(RANK_DELTA_CAP, score * RANK_DELTA_CAP))
    try:
        from backend.services.ai_regime_validation import blend_by_scalar, get_pattern_validated_scalar

        scalar, _detail = get_pattern_validated_scalar(label)
        return round(blend_by_scalar(0.0, raw_delta, scalar), 4)
    except Exception as exc:
        logger.debug("CHART_PATTERN_RANK_DELTA_VALIDATION_SKIPPED: %s", exc)
        return round(raw_delta, 4)


__all__ = [
    "RANK_DELTA_CAP",
    "chart_pattern_rank_delta",
    "detect_chart_pattern",
    "find_swing_points",
    "get_chart_pattern_signal",
]
