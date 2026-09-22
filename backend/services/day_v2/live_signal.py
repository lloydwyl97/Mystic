"""DAY V2 live signal evaluation.

Port of the deterministic signal detection logic from the qualifying
event-driven replay (scripts/research/day_v2_replay.py @ a88479a).

Only closed OHLCV bars are consumed — no look-ahead. Returns DayV2Signal
or None. The caller is responsible for ensuring the bar list is ordered
oldest-first and the last element is the most recently *closed* 15m bar.

No shadow checks here. This module is part of the live engine.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from typing import Any

# ---------------------------------------------------------------------------
# Setup family constants (must match replay and config)
# ---------------------------------------------------------------------------

SETUP_HTF_TREND_PULLBACK = "HTF_TREND_PULLBACK"
SETUP_BREAKOUT_CONTINUATION = "BREAKOUT_CONTINUATION"
SETUP_RANGE_BOUNCE = "RANGE_BOUNCE"
SETUP_VWAP_REVERSION = "VWAP_REVERSION"
SETUP_EXHAUSTION_MR = "EXHAUSTION_MR"

ENABLED_SETUPS: frozenset[str] = frozenset(
    {
        SETUP_HTF_TREND_PULLBACK,
        SETUP_BREAKOUT_CONTINUATION,
        SETUP_RANGE_BOUNCE,
        SETUP_VWAP_REVERSION,
        SETUP_EXHAUSTION_MR,
    }
)


@dataclass(frozen=True)
class DayV2Signal:
    """Entry signal produced by DAY V2 deterministic setup detection."""

    symbol: str  # normalized, e.g. "BTCUSDT"
    setup: str  # e.g. "HTF_TREND_PULLBACK"
    regime: str  # "bull" | "bear" | "neutral"
    structural_anchor: float  # price below which thesis is invalid
    target_price: float  # objective completion price
    atr: float  # 14-period ATR of the signal bar
    signal_bar_ts: int  # timestamp of the closed 15m bar that fired
    h1_bullish: bool
    opportunity_id: str  # deterministic 16-char hex ID for this opportunity


def _opportunity_id(symbol: str, setup: str, anchor: float) -> str:
    """Deterministic opportunity ID from symbol + setup + rounded anchor.

    Two repeated signals with the same setup family and structural anchor
    level (±0.001%) receive the same opportunity ID. This implements the
    continuity requirement: one continuous move cannot be repeatedly
    scalped under different IDs.
    """
    # Round anchor to 4 sig figs to tolerate minor float drift
    anchor_key = f"{anchor:.4g}"
    raw = f"DAY_V2:{symbol}:{setup}:{anchor_key}"
    return hashlib.sha1(raw.encode()).hexdigest()[:16].upper()


# ---------------------------------------------------------------------------
# Indicator helpers (identical to qualifying replay — no look-ahead)
# ---------------------------------------------------------------------------


def _sma(closes: list[float], n: int) -> float | None:
    if len(closes) < n:
        return None
    return sum(closes[-n:]) / n


def _rsi(closes: list[float], n: int = 14) -> float | None:
    if len(closes) < n + 1:
        return None
    diffs = [closes[i] - closes[i - 1] for i in range(len(closes) - n, len(closes))]
    gains = [max(0.0, d) for d in diffs]
    losses = [max(0.0, -d) for d in diffs]
    avg_gain = sum(gains) / n
    avg_loss = sum(losses) / n
    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return 100.0 - 100.0 / (1.0 + rs)


def _atr(highs: list[float], lows: list[float], closes: list[float], n: int = 14) -> float | None:
    if len(highs) < n + 1:
        return None
    trs = []
    for i in range(len(highs) - n, len(highs)):
        prev_c = closes[i - 1]
        tr = max(highs[i] - lows[i], abs(highs[i] - prev_c), abs(lows[i] - prev_c))
        trs.append(tr)
    return sum(trs) / n


def _bb_pct(closes: list[float], n: int = 20, k: float = 2.0) -> float | None:
    """Bollinger Band percent-B: 0 = lower band, 1 = upper band."""
    if len(closes) < n:
        return None
    window = closes[-n:]
    mean = sum(window) / n
    std = math.sqrt(sum((c - mean) ** 2 for c in window) / n)
    if std == 0:
        return 0.5
    lower = mean - k * std
    upper = mean + k * std
    return (closes[-1] - lower) / (upper - lower)


# ---------------------------------------------------------------------------
# Setup detection — no look-ahead, only bars[0..idx] visible
# ---------------------------------------------------------------------------


def _detect_setup(
    *,
    bars_15m: list[dict[str, Any]],
    bars_1h: list[dict[str, Any]],
    bars_4h: list[dict[str, Any]],
    idx: int,
) -> tuple[str, float, float, str, bool] | None:
    """Evaluate setup families at the closed bar at position idx.

    Returns (setup_name, structural_anchor, target_price, regime, h1_bullish)
    or None.

    Uses ONLY bars[0..idx]. Caller must not pass future bars.
    """
    if idx < 30:
        return None

    b0 = bars_15m[idx]  # the just-closed signal bar
    b1 = bars_15m[idx - 1]
    b2 = bars_15m[idx - 2]

    c0 = float(b0["close"])
    l0 = float(b0["low"])

    window_start = max(0, idx - 50)
    closes_15m = [float(b["close"]) for b in bars_15m[window_start : idx + 1]]
    highs_15m = [float(b["high"]) for b in bars_15m[window_start : idx + 1]]
    lows_15m = [float(b["low"]) for b in bars_15m[window_start : idx + 1]]

    rsi = _rsi(closes_15m, 14)
    atr = _atr(highs_15m, lows_15m, closes_15m, 14)
    bb = _bb_pct(closes_15m, 20, 2.0)
    sma20 = _sma(closes_15m, 20)
    high20 = max(highs_15m[-20:]) if len(highs_15m) >= 20 else None
    low20 = min(lows_15m[-20:]) if len(lows_15m) >= 20 else None

    if rsi is None or atr is None or bb is None or sma20 is None:
        return None

    # 4H regime: last 4H bar relative to 10-bar SMA
    regime = "neutral"
    ts0 = b0["ts"]
    if bars_4h:
        h4_past = [b for b in bars_4h if b["ts"] <= ts0]
        if len(h4_past) >= 10:
            h4_closes = [float(b["close"]) for b in h4_past[-11:]]
            h4_sma10 = _sma(h4_closes, 10)
            last_4h = h4_closes[-1]
            if h4_sma10:
                if last_4h > h4_sma10 * 1.005:
                    regime = "bull"
                elif last_4h < h4_sma10 * 0.995:
                    regime = "bear"

    # 1H context: last 1H close vs 5 bars ago
    h1_bullish = False
    if bars_1h:
        h1_past = [b for b in bars_1h if b["ts"] <= ts0]
        if len(h1_past) >= 5:
            h1_closes = [float(b["close"]) for b in h1_past[-6:]]
            h1_bullish = h1_closes[-1] > h1_closes[-5]

    b1c = float(b1["close"])
    b2c = float(b2["close"])

    # 1. HTF_TREND_PULLBACK
    # Bull 4H + 1H bullish + first green bar + near SMA20 + RSI recovering
    if regime == "bull" and h1_bullish and c0 > b1c and c0 < sma20 * 1.005 and 30 < rsi < 55 and SETUP_HTF_TREND_PULLBACK in ENABLED_SETUPS:
        anchor = c0 - 1.5 * atr
        target = c0 + 2.5 * atr
        return (SETUP_HTF_TREND_PULLBACK, anchor, target, regime, h1_bullish)

    # 2. RANGE_BOUNCE
    # Near-range-low + low BB + oversold RSI + first green bar
    if regime in ("neutral", "bear") and low20 is not None and c0 > b1c and c0 < low20 * 1.005 and bb < 0.30 and rsi < 45 and SETUP_RANGE_BOUNCE in ENABLED_SETUPS:
        anchor = low20 * 0.995
        target = c0 + 2.0 * atr
        return (SETUP_RANGE_BOUNCE, anchor, target, regime, h1_bullish)

    # 3. BREAKOUT_CONTINUATION
    # Decisive close above 20-bar high, two consecutive up bars
    if regime in ("bull", "neutral") and high20 is not None and c0 > high20 * 1.001 and rsi < 72 and c0 > b1c > b2c and SETUP_BREAKOUT_CONTINUATION in ENABLED_SETUPS:
        anchor = high20 * 0.995
        target = c0 + 2.0 * atr
        return (SETUP_BREAKOUT_CONTINUATION, anchor, target, regime, h1_bullish)

    # 4. VWAP_REVERSION
    # Oversold bounce after a decline, target SMA20
    if regime in ("bull", "neutral") and c0 > b1c and b1c < b2c and 20 < rsi < 40 and SETUP_VWAP_REVERSION in ENABLED_SETUPS:
        anchor = l0 * 0.998
        target = sma20
        if target > c0 * 1.003:
            return (SETUP_VWAP_REVERSION, anchor, target, regime, h1_bullish)

    # 5. EXHAUSTION_MR
    # Prior spike up (>2%), now retracing to SMA20
    closes_prior = closes_15m[:-1]
    rsi_prior = _rsi(closes_prior, 14) if len(closes_prior) >= 15 else None
    if rsi_prior is not None and rsi_prior < 45 and b1c > b2c * 1.02 and c0 < b1c and rsi < 50 and SETUP_EXHAUSTION_MR in ENABLED_SETUPS:
        anchor = l0 * 0.997
        target = sma20
        if target > c0 * 1.002:
            return (SETUP_EXHAUSTION_MR, anchor, target, regime, h1_bullish)

    return None


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def evaluate_entry_signal(
    symbol: str,
    bars_15m: list[dict[str, Any]],
    bars_1h: list[dict[str, Any]],
    bars_4h: list[dict[str, Any]],
) -> DayV2Signal | None:
    """Evaluate the most recent closed 15m bar for a DAY V2 entry signal.

    Args:
        symbol: Normalised symbol string, e.g. "BTCUSDT".
        bars_15m: Closed 15m bars, oldest-first. Last element = most recently closed.
        bars_1h:  Closed 1H bars, oldest-first.
        bars_4h:  Closed 4H bars, oldest-first.

    Returns:
        DayV2Signal if a setup fires on the last closed bar, else None.
    """
    if len(bars_15m) < 32:
        return None

    idx = len(bars_15m) - 1  # evaluate the most recently closed bar

    # Compute ATR for the signal bar (same window used by detect_setup)
    window_start = max(0, idx - 50)
    highs = [float(b["high"]) for b in bars_15m[window_start : idx + 1]]
    lows = [float(b["low"]) for b in bars_15m[window_start : idx + 1]]
    closes = [float(b["close"]) for b in bars_15m[window_start : idx + 1]]
    atr = _atr(highs, lows, closes, 14)
    if not atr or atr <= 0:
        return None

    result = _detect_setup(
        bars_15m=bars_15m,
        bars_1h=bars_1h,
        bars_4h=bars_4h,
        idx=idx,
    )
    if result is None:
        return None

    setup_name, anchor, target, regime, h1_bullish = result
    bar_ts = bars_15m[idx]["ts"]

    return DayV2Signal(
        symbol=symbol,
        setup=setup_name,
        regime=regime,
        structural_anchor=anchor,
        target_price=target,
        atr=atr,
        signal_bar_ts=int(bar_ts) if isinstance(bar_ts, (int, float)) else 0,
        h1_bullish=h1_bullish,
        opportunity_id=_opportunity_id(symbol, setup_name, anchor),
    )
