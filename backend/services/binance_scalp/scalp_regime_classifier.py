"""
Scalp market regime classifier — 1h bars, multi-label structure for regime-separated validation.

Does not blend regimes; each bar gets one primary scalp regime label.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

# Primary scalp regime labels (research validation)
REGIME_BULL_TREND = "bull_trend"
REGIME_BEAR_TREND = "bear_trend"
REGIME_RANGE = "range"
REGIME_CHOP = "chop"
REGIME_VOL_EXPANSION = "vol_expansion"
REGIME_VOL_CRUSH = "vol_crush"
REGIME_PUMP_CONTINUATION = "pump_continuation"
REGIME_DUMP_CONTINUATION = "dump_continuation"
REGIME_DUMP_REVERSAL = "dump_reversal"
REGIME_HIGH_VOL_BREAKOUT = "high_vol_breakout"
REGIME_LOW_VOL_DEAD = "low_vol_dead"

ALL_SCALP_REGIMES = (
    REGIME_BULL_TREND,
    REGIME_BEAR_TREND,
    REGIME_RANGE,
    REGIME_CHOP,
    REGIME_VOL_EXPANSION,
    REGIME_VOL_CRUSH,
    REGIME_PUMP_CONTINUATION,
    REGIME_DUMP_CONTINUATION,
    REGIME_DUMP_REVERSAL,
    REGIME_HIGH_VOL_BREAKOUT,
    REGIME_LOW_VOL_DEAD,
)

# Strategy → native regimes (router only enables in these)
STRATEGY_NATIVE_REGIMES: dict[str, frozenset[str]] = {
    "breakout_momentum": frozenset({REGIME_BULL_TREND, REGIME_VOL_EXPANSION, REGIME_HIGH_VOL_BREAKOUT, REGIME_PUMP_CONTINUATION}),
    "compression_breakout": frozenset({REGIME_VOL_CRUSH, REGIME_VOL_EXPANSION, REGIME_HIGH_VOL_BREAKOUT}),
    "volume_impulse_continuation": frozenset({REGIME_BULL_TREND, REGIME_PUMP_CONTINUATION, REGIME_HIGH_VOL_BREAKOUT}),
    "trend_pullback_micro": frozenset({REGIME_BULL_TREND, REGIME_PUMP_CONTINUATION}),
    "range_bounce_scalp": frozenset({REGIME_RANGE, REGIME_CHOP, REGIME_LOW_VOL_DEAD}),
    "vwap_ema_reclaim": frozenset({REGIME_RANGE, REGIME_CHOP, REGIME_DUMP_REVERSAL}),
    "failed_breakdown_reversal": frozenset({REGIME_DUMP_REVERSAL, REGIME_BEAR_TREND, REGIME_RANGE}),
    "failed_breakout_reversal": frozenset({REGIME_RANGE, REGIME_CHOP, REGIME_DUMP_CONTINUATION}),
    "orderbook_tape_scalp": frozenset({REGIME_VOL_EXPANSION, REGIME_HIGH_VOL_BREAKOUT, REGIME_PUMP_CONTINUATION}),
}


def _ema(values: list[float], period: int) -> float:
    if not values:
        return 0.0
    k = 2.0 / (period + 1)
    e = values[0]
    for v in values[1:]:
        e = v * k + e * (1 - k)
    return e


def _atr(bars: list[dict], period: int = 14) -> float:
    if len(bars) < period + 1:
        return 0.0
    trs = []
    for i in range(-period, 0):
        h, low, pc = bars[i]["high"], bars[i]["low"], bars[i - 1]["close"]
        trs.append(max(h - low, abs(h - pc), abs(low - pc)))
    return sum(trs) / len(trs)


def _adx(bars: list[dict], period: int = 14) -> float:
    if len(bars) < period + 2:
        return 0.0
    trs, pdm, mdm = [], [], []
    for i in range(-period - 1, 0):
        h, bar_low = bars[i]["high"], bars[i]["low"]
        ph, pl, pc = bars[i - 1]["high"], bars[i - 1]["low"], bars[i - 1]["close"]
        tr = max(h - bar_low, abs(h - pc), abs(bar_low - pc))
        up = h - ph
        dn = pl - bar_low
        trs.append(tr or 1e-9)
        pdm.append(up if (up > dn and up > 0) else 0.0)
        mdm.append(dn if (dn > up and dn > 0) else 0.0)
    atr = sum(trs)
    pdi = 100.0 * sum(pdm) / atr if atr else 0.0
    mdi = 100.0 * sum(mdm) / atr if atr else 0.0
    s = pdi + mdi
    return 100.0 * abs(pdi - mdi) / s if s else 0.0


@dataclass
class ScalpRegimeState:
    ts: int
    close: float
    regime: str
    adx: float
    atr_pct: float
    vol_ratio: float
    ema21: float
    ema55: float


def classify_scalp_regime(bars_1h: list[dict], idx: int) -> ScalpRegimeState | None:
    """Classify regime at bars_1h[idx] using lookback on same series."""
    if idx < 30 or idx >= len(bars_1h):
        return None
    window = bars_1h[max(0, idx - 200) : idx + 1]
    if len(window) < 30:
        return None
    closes = [b["close"] for b in window]
    c = closes[-1]
    e21 = _ema(closes, 21)
    e55 = _ema(closes, 55)
    e200 = _ema(closes, 200) if len(closes) >= 200 else e55
    adx = _adx(window)
    atr = _atr(window)
    atr_pct = atr / max(c, 1e-9)

    vol_recent = sum(b.get("volume", 0) for b in window[-6:])
    vol_prior = sum(b.get("volume", 0) for b in window[-12:-6]) or 1.0
    vol_ratio = vol_recent / vol_prior

    ret_6h = (c - closes[-7]) / max(closes[-7], 1e-9) if len(closes) >= 7 else 0.0
    ret_24h = (c - closes[-25]) / max(closes[-25], 1e-9) if len(closes) >= 25 else 0.0
    hi_20 = max(b["high"] for b in window[-20:])
    lo_20 = min(b["low"] for b in window[-20:])
    range_pct = (hi_20 - lo_20) / max(c, 1e-9)

    bar = bars_1h[idx]
    ts = int(bar.get("ts") or bar.get("epoch") or 0)

    # Low volume dead market
    if vol_ratio < 0.55 and atr_pct < 0.004 and adx < 16:
        regime = REGIME_LOW_VOL_DEAD
    # Vol crush (compression before expansion)
    elif atr_pct < 0.005 and range_pct < 0.012 and vol_ratio < 0.85:
        regime = REGIME_VOL_CRUSH
    # Pump / dump continuation
    elif ret_6h > 0.025 and vol_ratio > 1.3 and c > e21:
        regime = REGIME_PUMP_CONTINUATION
    elif ret_6h < -0.025 and vol_ratio > 1.2 and c < e21:
        regime = REGIME_DUMP_CONTINUATION
    # Dump reversal (capitulation bounce setup)
    elif ret_24h < -0.04 and ret_6h > 0.008 and c > lo_20 * 1.002:
        regime = REGIME_DUMP_REVERSAL
    # High vol breakout
    elif vol_ratio > 1.5 and atr_pct > 0.012 and c > hi_20 * 0.998:
        regime = REGIME_HIGH_VOL_BREAKOUT
    elif vol_ratio > 1.35 and atr_pct > 0.009:
        regime = REGIME_VOL_EXPANSION
    # Trend stacks
    elif e21 > e55 > e200 and adx >= 22:
        regime = REGIME_BULL_TREND
    elif e21 < e55 < e200 and adx >= 22:
        regime = REGIME_BEAR_TREND
    elif adx < 18 and range_pct < 0.025:
        regime = REGIME_RANGE
    elif adx < 22 and atr_pct > 0.01:
        regime = REGIME_CHOP
    else:
        regime = REGIME_RANGE

    return ScalpRegimeState(ts=ts, close=c, regime=regime, adx=adx, atr_pct=atr_pct, vol_ratio=vol_ratio, ema21=e21, ema55=e55)


def build_regime_index(bars_1h: list[dict]) -> dict[int, str]:
    """Map bar timestamp -> regime label."""
    out: dict[int, str] = {}
    for i in range(len(bars_1h)):
        st = classify_scalp_regime(bars_1h, i)
        if st and st.ts:
            out[st.ts] = st.regime
    return out


def regime_at_ts(regime_index: dict[int, str], epoch: int) -> str:
    """Nearest prior 1h regime for a given epoch (seconds)."""
    hour = epoch - (epoch % 3600)
    if hour in regime_index:
        return regime_index[hour]
    keys = [k for k in regime_index if k <= epoch]
    if not keys:
        return REGIME_RANGE
    return regime_index[max(keys)]


def summarize_regime_coverage(bars_1h: list[dict]) -> dict[str, Any]:
    """Hours/days per regime for one symbol series."""
    counts: dict[str, int] = {r: 0 for r in ALL_SCALP_REGIMES}
    for i in range(len(bars_1h)):
        st = classify_scalp_regime(bars_1h, i)
        if st:
            counts[st.regime] = counts.get(st.regime, 0) + 1
    total_h = sum(counts.values()) or 1
    return {
        "hours_by_regime": counts,
        "days_by_regime": {k: round(v / 24.0, 1) for k, v in counts.items()},
        "pct_by_regime": {k: round(100.0 * v / total_h, 2) for k, v in counts.items()},
        "total_hours": total_h,
        "total_days": round(total_h / 24.0, 1),
    }


def router_decision(
    *,
    current_regime: str,
    strategy: str,
    expectancy: float = 0.0,
    trades_per_month: float = 0.0,
    confidence: str = "low",
) -> dict[str, Any]:
    """Paper router view for one strategy at current regime."""
    native = STRATEGY_NATIVE_REGIMES.get(strategy, frozenset())
    allowed = current_regime in native and expectancy > 0
    if allowed:
        reason = f"native_regime_match expectancy={expectancy:.4f}"
    elif current_regime not in native:
        reason = f"blocked: regime={current_regime} not in native {sorted(native)}"
    else:
        reason = f"blocked: native regime {current_regime} but expectancy={expectancy:.4f}<=0"
    return {
        "current_regime": current_regime,
        "strategy": strategy,
        "allowed": allowed,
        "blocked": not allowed,
        "reason": reason,
        "expected_expectancy_usd": expectancy,
        "expected_trades_per_month": trades_per_month,
        "confidence": confidence,
    }
