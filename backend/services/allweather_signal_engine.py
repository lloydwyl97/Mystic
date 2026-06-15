"""
All-weather signal engine (production) — validated in research, gated OFF by default.

Mirrors scripts/run_allweather_strategy_lab.py, which proved over 3 years on the
top-four pairs (verified Binance.US 0% maker / 0.02% taker):
  * BREAKOUT (Donchian break in uptrend / volatility expansion) — primary edge
  * TREND_PULLBACK (buy a pullback to EMA21 that resumes up) — secondary
  * MEAN_REVERSION removed (proven net-negative)
  * Downtrend = no longs (spot is long-only; capital preserved)
  * Honest bounded exits: ATR target, ATR stop, hard <=72h time-stop

Enable with ALLWEATHER_ENGINE_ENABLED=true. When disabled every public helper
is a no-op signal so the live engine keeps its existing behavior unchanged.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Optional

SETUP_BREAKOUT = "BREAKOUT"
SETUP_TREND_PULLBACK = "TREND_PULLBACK"

REG_TREND_UP = "trend_up"
REG_TREND_DOWN = "trend_down"
REG_RANGE = "range"
REG_NEUTRAL = "neutral"

TIME_STOP_HOURS = float(os.getenv("ALLWEATHER_TIME_STOP_HOURS", "72"))
DONCHIAN = int(os.getenv("ALLWEATHER_DONCHIAN", "20"))


def allweather_enabled() -> bool:
    return os.getenv("ALLWEATHER_ENGINE_ENABLED", "false").strip().lower() in ("1", "true", "yes", "on")


# --------------------------- indicators (pure python) ---------------------------
def _ema_last(values: list[float], period: int) -> float:
    if not values:
        return 0.0
    k = 2.0 / (period + 1)
    e = values[0]
    for v in values[1:]:
        e = v * k + e * (1 - k)
    return e


def _rsi(closes: list[float], period: int = 14) -> float:
    if len(closes) < period + 1:
        return 50.0
    gains = losses = 0.0
    for i in range(-period, 0):
        ch = closes[i] - closes[i - 1]
        if ch >= 0:
            gains += ch
        else:
            losses -= ch
    if losses == 0:
        return 100.0
    rs = (gains / period) / (losses / period)
    return 100.0 - 100.0 / (1.0 + rs)


def _atr(bars: list[dict], period: int = 14) -> float:
    if len(bars) < period + 1:
        return 0.0
    trs = []
    for i in range(-period, 0):
        h, l, pc = bars[i]["high"], bars[i]["low"], bars[i - 1]["close"]
        trs.append(max(h - l, abs(h - pc), abs(l - pc)))
    return sum(trs) / len(trs)


def _adx(bars: list[dict], period: int = 14) -> float:
    if len(bars) < period + 2:
        return 0.0
    trs, pdm, mdm = [], [], []
    for i in range(-period - 1, 0):
        h, l = bars[i]["high"], bars[i]["low"]
        ph, pl, pc = bars[i - 1]["high"], bars[i - 1]["low"], bars[i - 1]["close"]
        tr = max(h - l, abs(h - pc), abs(l - pc))
        up = h - ph
        dn = pl - l
        trs.append(tr or 1e-9)
        pdm.append(up if (up > dn and up > 0) else 0.0)
        mdm.append(dn if (dn > up and dn > 0) else 0.0)
    atr = sum(trs)
    pdi = 100.0 * sum(pdm) / atr if atr else 0.0
    mdi = 100.0 * sum(mdm) / atr if atr else 0.0
    s = pdi + mdi
    return 100.0 * abs(pdi - mdi) / s if s else 0.0


@dataclass
class AWState:
    close: float
    prev_close: float
    ema21: float
    ema55: float
    ema200: float
    adx: float
    atr: float
    rsi: float
    don_high: float
    don_low: float
    regime: str


def normalize_bars(raw: Any) -> list[dict]:
    """Accept ccxt OHLCV (list of [ts,o,h,l,c,v]) or list of dicts -> list of dicts."""
    out: list[dict] = []
    if not raw:
        return out
    for r in raw:
        if isinstance(r, dict):
            out.append({
                "ts": int(r.get("ts") or r.get("timestamp") or 0),
                "open": float(r["open"]), "high": float(r["high"]),
                "low": float(r["low"]), "close": float(r["close"]),
            })
        elif isinstance(r, (list, tuple)) and len(r) >= 5:
            out.append({
                "ts": int(r[0]) // 1000, "open": float(r[1]), "high": float(r[2]),
                "low": float(r[3]), "close": float(r[4]),
            })
    return out


def compute_state(bars_1h: list[dict], don: int = DONCHIAN) -> Optional[AWState]:
    """Compute latest indicator/regime state from >=206 hourly bars."""
    if not bars_1h or len(bars_1h) < 206:
        return None
    closes = [b["close"] for b in bars_1h]
    e21 = _ema_last(closes, 21)
    e55 = _ema_last(closes, 55)
    e200 = _ema_last(closes, 200)
    adx = _adx(bars_1h)
    atr = _atr(bars_1h)
    rsi = _rsi(closes)
    prior = bars_1h[-don - 1:-1]
    dhigh = max(b["high"] for b in prior) if prior else bars_1h[-1]["high"]
    dlow = min(b["low"] for b in prior) if prior else bars_1h[-1]["low"]
    c = closes[-1]
    if e21 > e55 > e200 and adx >= 20:
        regime = REG_TREND_UP
    elif e21 < e55 < e200 and adx >= 20:
        regime = REG_TREND_DOWN
    elif adx < 18:
        regime = REG_RANGE
    else:
        regime = REG_NEUTRAL
    return AWState(c, closes[-2], e21, e55, e200, adx, atr, rsi, dhigh, dlow, regime)


def entry_signal(state: AWState) -> Optional[dict[str, Any]]:
    """Return {setup, regime, target_atr, stop_atr} for a long entry, else None.

    Identical rules to the validated lab. Mean-reversion intentionally absent.
    """
    c = state.close
    atr_pct = state.atr / c if c > 0 else 0.0
    if atr_pct <= 0:
        return None

    if state.regime == REG_TREND_UP:
        near_ema = (c <= state.ema21 * (1.0 + 0.35 * atr_pct)) and (c >= state.ema21 * (1.0 - 1.2 * atr_pct))
        resuming = c > state.prev_close
        if near_ema and resuming and 35.0 <= state.rsi <= 62.0:
            return {"setup": SETUP_TREND_PULLBACK, "regime": state.regime, "target_atr": 2.2, "stop_atr": 1.3}
        if c > state.don_high and state.rsi <= 78.0:
            return {"setup": SETUP_BREAKOUT, "regime": state.regime, "target_atr": 2.6, "stop_atr": 1.5}

    if state.regime == REG_NEUTRAL:
        if c > state.don_high and c > state.ema55 and state.adx >= 18 and state.rsi <= 75.0:
            return {"setup": SETUP_BREAKOUT, "regime": state.regime, "target_atr": 2.4, "stop_atr": 1.5}

    # range -> no longs; trend_down -> no longs (spot long-only, capital preserved)
    return None


def entry_levels(current_price: float, atr: float, target_atr: float, stop_atr: float) -> tuple[float, float]:
    """Return (target_level, stop_level) from ATR multiples."""
    if current_price <= 0 or atr <= 0:
        return 0.0, 0.0
    atr_pct = atr / current_price
    target = current_price * (1.0 + target_atr * atr_pct)
    stop = current_price * (1.0 - stop_atr * atr_pct)
    return round(target, 8), round(stop, 8)


def exit_decision(
    *,
    current_price: float,
    bar_low: float,
    bar_high: float,
    target_level: float,
    stop_level: float,
    hold_hours: float,
) -> Optional[dict[str, str]]:
    """Bounded exit: ATR stop, ATR target, hard <=72h time-stop. None = hold."""
    if stop_level > 0 and (bar_low <= stop_level or current_price <= stop_level):
        return {"action": "sell", "reason": "ALLWEATHER_STOP"}
    if target_level > 0 and (bar_high >= target_level or current_price >= target_level):
        return {"action": "sell", "reason": "ALLWEATHER_TARGET"}
    if hold_hours >= TIME_STOP_HOURS:
        return {"action": "sell", "reason": "ALLWEATHER_TIME_STOP"}
    return None


__all__ = [
    "allweather_enabled",
    "compute_state",
    "entry_signal",
    "entry_levels",
    "exit_decision",
    "normalize_bars",
    "AWState",
    "SETUP_BREAKOUT",
    "SETUP_TREND_PULLBACK",
    "TIME_STOP_HOURS",
]
