from __future__ import annotations

import contextlib
import logging
import math
from datetime import datetime, timezone
from typing import Any

import numpy as np

from backend.services.feature_mapping import FEATURE_MAPPING

logger = logging.getLogger(__name__)

try:
    import talib  # type: ignore[import-not-found]
except Exception as ex:
    logging.getLogger(__name__).debug("talib not available: %s", ex)
    talib = None

# Trust scores for feature health metadata (audit / learning guardrails).
FEATURE_TRUST_SCORES: dict[str, float] = {
    "LIVE": 1.0,
    "CALCULATED": 0.92,
    "CALCULATED_PROXY": 0.62,
    "WARMUP": 0.15,
    "FALLBACK": 0.10,
    "MISSING": 0.0,
    "STALE": 0.10,
    "UNSUPPORTED_FOR_SPOT": 0.0,
    "ZERO_DEFAULT": 0.0,
    "LOW_IMPORTANCE_TIME_FIELD_NORMAL": 0.35,
}

LEARNING_ALLOWED_STATUSES: frozenset[str] = frozenset({"LIVE", "CALCULATED"})


def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, float(v)))


def _volume_price_trend_rolling(cl: np.ndarray, vo: np.ndarray, *, window: int = 50) -> float:
    """Rolling VPT sum over last ``window`` bars, volume-normalized and clipped."""
    if cl.size < 3 or vo.size < 3:
        return 0.0
    n = min(int(window), int(cl.size) - 1)
    if n < 2:
        return 0.0
    start = cl.size - n
    acc = 0.0
    for i in range(start, cl.size):
        if i <= 0 or cl[i - 1] == 0:
            continue
        acc += float((cl[i] - cl[i - 1]) / cl[i - 1]) * float(vo[i])
    vol_scale = float(np.mean(vo[start:cl.size]))
    if vol_scale <= 1e-12:
        return 0.0
    return _clamp(acc / vol_scale, -5.0, 5.0)


def _orderbook_freshness_meta(
    orderbook: dict[str, Any] | None,
    orderbook_age_sec: float | None,
    *,
    threshold_sec: float = 45.0,
) -> dict[str, Any]:
    age = orderbook_age_sec
    updated_at = None
    if orderbook:
        updated_at = orderbook.get("updated_at") or orderbook.get("ts_utc")
        if age is None and updated_at is not None:
            import time

            with contextlib.suppress(Exception):
                age = max(0.0, time.time() - float(updated_at))
    stale = age is not None and age > threshold_sec
    return {
        "age_seconds": age,
        "orderbook_updated_at": updated_at,
        "freshness_status": "STALE" if stale else ("FRESH" if age is not None else "UNKNOWN"),
        "freshness_trust_modifier": 0.25 if stale else 1.0,
    }


def _record_feature_provenance(
    provenance: dict[str, dict[str, Any]] | None,
    name: str,
    status: str,
    source: str,
    *,
    age_seconds: float | None = None,
    trust_score: float | None = None,
    learning_allowed: bool | None = None,
    orderbook_updated_at: Any = None,
    freshness_status: str | None = None,
    freshness_trust_modifier: float | None = None,
) -> None:
    if provenance is None:
        return
    ts = float(trust_score if trust_score is not None else FEATURE_TRUST_SCORES.get(status, 0.5))
    la = bool(learning_allowed if learning_allowed is not None else status in LEARNING_ALLOWED_STATUSES)
    row: dict[str, Any] = {
        "status": status,
        "source": source,
        "age_seconds": age_seconds,
        "trust_score": round(ts, 4),
        "learning_allowed": la,
    }
    if orderbook_updated_at is not None:
        row["orderbook_updated_at"] = orderbook_updated_at
    if freshness_status is not None:
        row["freshness_status"] = freshness_status
    if freshness_trust_modifier is not None:
        row["freshness_trust_modifier"] = round(float(freshness_trust_modifier), 4)
    provenance[name] = row


def _swing_range_levels(hi: np.ndarray, lo: np.ndarray, lookback: int) -> tuple[float, float, float] | None:
    n = min(int(lookback), int(hi.size), int(lo.size))
    if n < 5:
        return None
    high_n = float(np.max(hi[-n:]))
    low_n = float(np.min(lo[-n:]))
    rng = high_n - low_n
    if rng <= 0:
        return None
    return low_n + rng * 0.236, low_n + rng * 0.382, low_n + rng * 0.618


def _ichimoku_from_ohlcv(hi: np.ndarray, lo: np.ndarray) -> tuple[float, float, float, float]:
    n = int(hi.size)
    w9, w26, w52 = min(9, n), min(26, n), min(52, n)
    tenkan = (float(np.max(hi[-w9:])) + float(np.min(lo[-w9:]))) / 2.0
    kijun = (float(np.max(hi[-w26:])) + float(np.min(lo[-w26:]))) / 2.0
    senkou_a = (tenkan + kijun) / 2.0
    senkou_b = (float(np.max(hi[-w52:])) + float(np.min(lo[-w52:]))) / 2.0
    return tenkan, kijun, senkou_a, senkou_b


def _safe(v: Any, default: float = 0.0) -> float:
    try:
        x = float(v)
        if not math.isfinite(x):
            return default
        return x
    except Exception as ex:
        logger.debug("_safe float failed for %r: %s", v, ex)
        return default


def _ohlcv_arrays(ohlcv: list[list]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    # ohlcv rows: [ts, open, high, low, close, volume]
    arr = np.asarray(ohlcv, dtype=float)
    ts = arr[:, 0]
    op = arr[:, 1]
    hi = arr[:, 2]
    lo = arr[:, 3]
    cl = arr[:, 4]
    vo = arr[:, 5]
    return ts, op, hi, lo, cl, vo


def _rolling_mean(x: np.ndarray, n: int) -> float:
    if x.size < n:
        return _safe(x[-1], 0.0) if x.size else 0.0
    return float(np.mean(x[-n:]))


def _rolling_std(x: np.ndarray, n: int) -> float:
    if x.size < n:
        return 0.0
    return float(np.std(x[-n:], ddof=0))


def _returns(cl: np.ndarray) -> np.ndarray:
    if cl.size < 2:
        return np.array([], dtype=float)
    r = np.diff(cl) / np.where(cl[:-1] == 0, 1.0, cl[:-1])
    return r


def _skewness(x: np.ndarray) -> float:
    if x.size < 3:
        return 0.0
    mu = float(np.mean(x))
    sigma = float(np.std(x, ddof=0))
    if sigma == 0.0:
        return 0.0
    m3 = float(np.mean((x - mu) ** 3))
    return m3 / (sigma**3)


def _nvi_pvi_from_ohlcv(cl: np.ndarray, vo: np.ndarray) -> tuple[float, float]:
    """NVI/PVI (classic volume-on-change rules). Returns scaled deltas (~small float range)."""
    if cl.size < 3 or vo.size < 3:
        return 0.0, 0.0
    nvi = 1000.0
    pvi = 1000.0
    for i in range(1, int(cl.size)):
        prev_c, c = float(cl[i - 1]), float(cl[i])
        ret = (c - prev_c) / prev_c if prev_c != 0 else 0.0
        pv, cv = float(vo[i - 1]), float(vo[i])
        if cv < pv:
            nvi *= 1.0 + ret
        elif cv > pv:
            pvi *= 1.0 + ret
    return float((nvi / 1000.0) - 1.0), float((pvi / 1000.0) - 1.0)


def _volume_profile_proxy_poc_vah_val(hi: np.ndarray, lo: np.ndarray, cl: np.ndarray, vo: np.ndarray, lookback: int = 50) -> tuple[float, float, float]:
    """OHLCV-only VP proxy when Redis volume_profile is absent (non-zero, varying)."""
    n = min(lookback, int(cl.size))
    if n < 5:
        return 0.0, 0.0, 0.0
    sl = slice(-n, None)
    h = hi[sl]
    lows = lo[sl]
    c = cl[sl]
    v = vo[sl]
    typ = (h + lows + c) / 3.0
    w = np.maximum(v, 1e-18)
    poc = float(np.average(typ, weights=w))
    vah = float(np.max(h))
    val = float(np.min(lows))
    return poc, vah, val


def _ema_arr(x: np.ndarray, n: int) -> np.ndarray:
    """Exponential moving average array."""
    a = 2.0 / (n + 1)
    out = np.empty_like(x, dtype=np.float64)
    out[0] = x[0]
    for i in range(1, x.size):
        out[i] = a * x[i] + (1 - a) * out[i - 1]
    return out


def _wilder_smooth_arr(x: np.ndarray, period: int) -> np.ndarray:
    """Wilder smoothing array (alpha = 1/period)."""
    out = np.empty(x.size, dtype=np.float64)
    out[0] = float(np.mean(x[:period])) if x.size >= period else float(x[0])
    for i in range(1, x.size):
        out[i] = (out[i - 1] * (period - 1) + float(x[i])) / period
    return out


def _true_range(hi: np.ndarray, lo: np.ndarray, cl: np.ndarray) -> np.ndarray:
    """True Range array (length n-1)."""
    return np.maximum(hi[1:] - lo[1:], np.maximum(np.abs(hi[1:] - cl[:-1]), np.abs(lo[1:] - cl[:-1])))


def build_feature_dict_from_ohlcv(
    *,
    symbol_ccxt: str,
    ohlcv: list[list],
    volume_profile: dict[str, Any] | None,
    orderbook: dict[str, Any] | None,
    sentiment: dict[str, Any] | None,
    ohlcv_1d: list[list] | None = None,
    provenance: dict[str, dict[str, Any]] | None = None,
    orderbook_age_sec: float | None = None,
) -> dict[str, float]:
    """
    Return a flat dict keyed by FEATURE_MAPPING names (124 keys).
    This is the ONLY canonical producer; training + inference must use it.

    ohlcv_1d: optional native daily candles (ascending) from the exchange. When provided with
    enough rows, change_24h / change_7d / change_30d are computed from **daily** closes so they
    match true calendar horizons. Without it, those fields use 1m bar counts (1440/10080/43200)
    and will be 0.0 when fewer than that many 1m bars exist — never implied from a partial 1m window.
    """
    out: dict[str, float] = dict.fromkeys(FEATURE_MAPPING.keys(), 0.0)

    if not ohlcv:
        for name in FEATURE_MAPPING:
            _record_feature_provenance(provenance, name, "MISSING", "no_1m_ohlcv", learning_allowed=False)
        return out

    ts, op, hi, lo, cl, vo = _ohlcv_arrays(ohlcv)
    n_bars = int(cl.size)
    ob_stale = orderbook_age_sec is not None and orderbook_age_sec > 45.0

    # -------------------------
    # Basic (1-10)
    # -------------------------
    out["price"] = _safe(cl[-1], 0.0)
    out["high"] = _safe(hi[-1], 0.0)
    out["low"] = _safe(lo[-1], 0.0)
    out["open"] = _safe(op[-1], 0.0)
    lv = _safe(vo[-1], 0.0)
    if lv < 1e-12 and vo.size >= 2:
        lv = float(np.mean(vo[-min(20, vo.size) :]))
    out["volume"] = lv

    # change_24h / 7d / 30d are defined in schema, but for 1m bars:
    # 24h=1440, 7d=10080, 30d=43200
    def _chg(bars: int) -> float:
        if cl.size < (bars + 1):
            return 0.0
        prev = cl[-(bars + 1)]
        if prev == 0:
            return 0.0
        return float((cl[-1] - prev) / prev)

    out["change_24h"] = _chg(1440)
    out["change_7d"] = _chg(10080)
    out["change_30d"] = _chg(43200)

    # True calendar horizons from native 1d series (overrides 1m-based zeros when 1m depth is < horizon).
    if ohlcv_1d and len(ohlcv_1d) >= 2:
        dcl = np.asarray([float(c[4]) for c in ohlcv_1d], dtype=float)
        if dcl.size >= 2 and float(dcl[-2]) != 0.0:
            out["change_24h"] = float((dcl[-1] - dcl[-2]) / dcl[-2])
        if dcl.size >= 8 and float(dcl[-8]) != 0.0:
            out["change_7d"] = float((dcl[-1] - dcl[-8]) / dcl[-8])
        if dcl.size >= 31 and float(dcl[-31]) != 0.0:
            out["change_30d"] = float((dcl[-1] - dcl[-31]) / dcl[-31])

    pr_n = min(14, int(cl.size))
    if pr_n >= 1:
        out["price_range"] = float(np.max(hi[-pr_n:]) - np.min(lo[-pr_n:]))
    else:
        out["price_range"] = float(out["high"] - out["low"])
    out["typical_price"] = float((out["high"] + out["low"] + out["price"]) / 3.0)

    # -------------------------
    # Technical (11-37)
    # Use talib when available; otherwise exact numpy equivalents where defined.
    # -------------------------
    out["ma_5"] = _rolling_mean(cl, 5)
    out["ma_10"] = _rolling_mean(cl, 10)
    out["ma_20"] = _rolling_mean(cl, 20)
    out["ma_50"] = _rolling_mean(cl, 50)
    out["ma_100"] = _rolling_mean(cl, 100)
    out["ma_200"] = _rolling_mean(cl, 200)

    if talib is not None and cl.size >= 50:
        out["ema_12"] = _safe(talib.EMA(cl, timeperiod=12)[-1], out["price"])
        out["ema_26"] = _safe(talib.EMA(cl, timeperiod=26)[-1], out["price"])
        out["ema_50"] = _safe(talib.EMA(cl, timeperiod=50)[-1], out["price"])

        out["rsi"] = _safe(talib.RSI(cl, timeperiod=14)[-1], 50.0)
        out["rsi_14"] = out["rsi"]

        # Stoch
        slowk, slowd = talib.STOCH(hi, lo, cl, fastk_period=14, slowk_period=3, slowk_matype=0, slowd_period=3, slowd_matype=0)
        out["stoch_k"] = _safe(slowk[-1], 50.0)
        out["stoch_d"] = _safe(slowd[-1], 50.0)

        out["williams_r"] = _safe(talib.WILLR(hi, lo, cl, timeperiod=14)[-1], -50.0)
        out["cci"] = _safe(talib.CCI(hi, lo, cl, timeperiod=20)[-1], 0.0)

        macd, macdsignal, macdhist = talib.MACD(cl, fastperiod=12, slowperiod=26, signalperiod=9)
        out["macd"] = _safe(macd[-1], 0.0)
        out["macd_signal"] = _safe(macdsignal[-1], 0.0)
        out["macd_histogram"] = _safe(macdhist[-1], 0.0)

        upper, middle, lower = talib.BBANDS(cl, timeperiod=20, nbdevup=2, nbdevdn=2, matype=0)
        out["bb_upper"] = _safe(upper[-1], out["price"])
        out["bb_middle"] = _safe(middle[-1], out["price"])
        out["bb_lower"] = _safe(lower[-1], out["price"])
        denom = out["bb_upper"] - out["bb_lower"]
        out["bb_position"] = 0.5 if denom == 0 else float((out["price"] - out["bb_lower"]) / denom)
        out["bb_width"] = 0.0 if out["bb_middle"] == 0 else float((out["bb_upper"] - out["bb_lower"]) / out["bb_middle"])

        out["obv"] = _safe(talib.OBV(cl, vo)[-1], 0.0)
        out["ad_line"] = _safe(talib.AD(hi, lo, cl, vo)[-1], 0.0)
        # CMF is not in talib; compute over 20 by definition:
        if cl.size >= 20:
            h20 = hi[-20:]
            l20 = lo[-20:]
            c20 = cl[-20:]
            v20 = vo[-20:]
            denom2 = np.where((h20 - l20) == 0, 1.0, (h20 - l20))
            mfm = ((c20 - l20) - (h20 - c20)) / denom2
            mfv = mfm * v20
            out["cmf"] = float(np.sum(mfv) / np.sum(v20)) if float(np.sum(v20)) != 0.0 else 0.0
        else:
            out["cmf"] = 0.0

        out["mfi"] = _safe(talib.MFI(hi, lo, cl, vo, timeperiod=14)[-1], 50.0)
    else:
        # Numpy fallbacks for when talib is unavailable
        out["ema_12"] = _safe(_ema_arr(cl, 12)[-1], out["price"]) if cl.size >= 12 else out["ma_20"]
        out["ema_26"] = _safe(_ema_arr(cl, 26)[-1], out["price"]) if cl.size >= 26 else out["ma_20"]
        out["ema_50"] = _safe(_ema_arr(cl, 50)[-1], out["price"]) if cl.size >= 50 else out["ma_50"]

        # RSI — Wilder smoothed
        if cl.size >= 15:
            d = np.diff(cl)
            g = np.maximum(d, 0.0)
            l_arr = np.maximum(-d, 0.0)
            ag = float(np.mean(g[:14]))
            al = float(np.mean(l_arr[:14]))
            for ix in range(14, len(g)):
                ag = (ag * 13 + float(g[ix])) / 14
                al = (al * 13 + float(l_arr[ix])) / 14
            rsi_v = (100.0 - 100.0 / (1.0 + ag / al)) if al > 1e-15 else (100.0 if ag > 1e-15 else 50.0)
            out["rsi"] = rsi_v
            out["rsi_14"] = rsi_v
        else:
            out["rsi"] = 50.0
            out["rsi_14"] = 50.0

        # Stochastic %K / %D
        if cl.size >= 16:
            fk_vals = []
            for off in range(min(5, cl.size - 14)):
                end = cl.size - off
                hh = float(np.max(hi[end - 14 : end]))
                ll = float(np.min(lo[end - 14 : end]))
                denom = hh - ll
                fk_vals.append(100.0 * (float(cl[end - 1]) - ll) / denom if denom > 0 else 50.0)
            fk_vals.reverse()
            out["stoch_k"] = float(np.mean(fk_vals[:3])) if len(fk_vals) >= 3 else fk_vals[-1]
            out["stoch_d"] = float(np.mean(fk_vals[:3])) if len(fk_vals) >= 3 else out["stoch_k"]
        else:
            out["stoch_k"] = 50.0
            out["stoch_d"] = 50.0

        # Williams %R
        if cl.size >= 14:
            hh14 = float(np.max(hi[-14:]))
            ll14 = float(np.min(lo[-14:]))
            denom = hh14 - ll14
            out["williams_r"] = -100.0 * (hh14 - float(cl[-1])) / denom if denom > 0 else -50.0
        else:
            out["williams_r"] = -50.0

        # CCI
        if cl.size >= 20:
            tp = (hi + lo + cl) / 3.0
            tp20 = tp[-20:]
            sma_tp = float(np.mean(tp20))
            mad = float(np.mean(np.abs(tp20 - sma_tp)))
            out["cci"] = (float(tp[-1]) - sma_tp) / (0.015 * mad) if mad > 1e-15 else 0.0
        else:
            out["cci"] = 0.0

        # MACD (12/26/9)
        if cl.size >= 26:
            ema12 = _ema_arr(cl, 12)
            ema26 = _ema_arr(cl, 26)
            macd_line = ema12 - ema26
            sig_line = _ema_arr(macd_line, 9) if macd_line.size >= 9 else macd_line
            out["macd"] = _safe(float(macd_line[-1]))
            out["macd_signal"] = _safe(float(sig_line[-1]))
            out["macd_histogram"] = _safe(float(macd_line[-1] - sig_line[-1]))
        else:
            out["macd"] = 0.0
            out["macd_signal"] = 0.0
            out["macd_histogram"] = 0.0

        # Bollinger Bands
        if cl.size >= 20:
            bb_mid = float(np.mean(cl[-20:]))
            bb_std = float(np.std(cl[-20:], ddof=0))
            bb_up = bb_mid + 2.0 * bb_std
            bb_lo = bb_mid - 2.0 * bb_std
            out["bb_upper"] = bb_up
            out["bb_middle"] = bb_mid
            out["bb_lower"] = bb_lo
            denom = bb_up - bb_lo
            out["bb_position"] = 0.5 if denom == 0 else float((out["price"] - bb_lo) / denom)
            out["bb_width"] = 0.0 if bb_mid == 0 else float(denom / bb_mid)
        else:
            out["bb_upper"] = out["price"]
            out["bb_middle"] = out["price"]
            out["bb_lower"] = out["price"]
            out["bb_position"] = 0.5
            out["bb_width"] = 0.0

        # OBV
        if cl.size >= 2:
            signs = np.sign(np.diff(cl))
            out["obv"] = float(np.sum(signs * vo[1:]))
        else:
            out["obv"] = 0.0

        # Accumulation/Distribution Line
        if cl.size >= 2:
            hl_diff = hi - lo
            with np.errstate(divide="ignore", invalid="ignore"):
                clv = np.where(hl_diff == 0, 0.0, ((cl - lo) - (hi - cl)) / hl_diff)
            out["ad_line"] = float(np.nansum(clv * vo))
        else:
            out["ad_line"] = 0.0

        # CMF
        if cl.size >= 20:
            h20, l20, c20, v20 = hi[-20:], lo[-20:], cl[-20:], vo[-20:]
            denom2 = np.where((h20 - l20) == 0, 1.0, (h20 - l20))
            mfm = ((c20 - l20) - (h20 - c20)) / denom2
            mfv = mfm * v20
            out["cmf"] = float(np.sum(mfv) / np.sum(v20)) if float(np.sum(v20)) != 0.0 else 0.0
        else:
            out["cmf"] = 0.0

        # MFI
        if cl.size >= 15:
            tp_arr = (hi + lo + cl) / 3.0
            tp_diff = np.diff(tp_arr)
            mf = tp_arr[1:] * vo[1:]
            pos_mf = np.where(tp_diff > 0, mf, 0.0)
            neg_mf = np.where(tp_diff < 0, mf, 0.0)
            pmf14 = float(np.sum(pos_mf[-14:]))
            nmf14 = float(np.sum(neg_mf[-14:]))
            out["mfi"] = 100.0 - 100.0 / (1.0 + pmf14 / nmf14) if nmf14 > 1e-15 else (100.0 if pmf14 > 1e-15 else 50.0)
        else:
            out["mfi"] = 50.0

    # -------------------------
    # Volatility (38-47)
    # -------------------------
    out["volatility"] = 0.0
    if cl.size >= 20:
        mu = _rolling_mean(cl, 20)
        out["volatility"] = 0.0 if mu == 0 else float(_rolling_std(cl, 20) / mu)

    if talib is not None and cl.size >= 20:
        out["atr"] = _safe(talib.ATR(hi, lo, cl, timeperiod=14)[-1], 0.0)
        out["natr"] = _safe(talib.NATR(hi, lo, cl, timeperiod=14)[-1], 0.0)
        ema20 = talib.EMA(cl, timeperiod=20)
        atr14 = talib.ATR(hi, lo, cl, timeperiod=14)
        out["keltner_upper"] = _safe(ema20[-1] + 2.0 * atr14[-1], out["price"])
        out["keltner_lower"] = _safe(ema20[-1] - 2.0 * atr14[-1], out["price"])
        out["donchian_upper"] = float(np.max(hi[-20:])) if hi.size >= 20 else out["high"]
        out["donchian_lower"] = float(np.min(lo[-20:])) if lo.size >= 20 else out["low"]
        out["parabolic_sar"] = _safe(talib.SAR(hi, lo, acceleration=0.02, maximum=0.2)[-1], out["price"])
    else:
        if cl.size >= 15:
            tr = _true_range(hi, lo, cl)
            atr_arr = _wilder_smooth_arr(tr, 14)
            atr_val = float(atr_arr[-1])
            out["atr"] = atr_val
            out["natr"] = (atr_val / float(cl[-1]) * 100.0) if float(cl[-1]) > 0 else 0.0
            ema20_val = float(_ema_arr(cl, 20)[-1]) if cl.size >= 20 else float(np.mean(cl[-20:]) if cl.size >= 20 else cl[-1])
            out["keltner_upper"] = ema20_val + 2.0 * atr_val
            out["keltner_lower"] = ema20_val - 2.0 * atr_val
        else:
            out["atr"] = 0.0
            out["natr"] = 0.0
            out["keltner_upper"] = out["price"]
            out["keltner_lower"] = out["price"]
        out["donchian_upper"] = float(np.max(hi[-20:])) if hi.size >= 20 else out["high"]
        out["donchian_lower"] = float(np.min(lo[-20:])) if lo.size >= 20 else out["low"]
        out["parabolic_sar"] = out["price"]

    # volatility_ratio = vol(5) / vol(20)
    v5 = _rolling_std(cl, 5)
    v20 = _rolling_std(cl, 20)
    out["volatility_ratio"] = 0.0 if v20 == 0.0 else float(v5 / v20)
    out["price_volatility"] = float(v20)

    # -------------------------
    # Momentum (48-62)
    # -------------------------
    # Momentum indicators — talib for ROC/MOM/PPO/TRIX/ULTOSC/BOP; rest is always numpy
    if talib is not None and cl.size >= 40:
        out["roc"] = _safe(talib.ROC(cl, timeperiod=10)[-1], 0.0)
        out["momentum"] = _safe(talib.MOM(cl, timeperiod=10)[-1], 0.0)
        out["ppo"] = _safe(talib.PPO(cl, fastperiod=12, slowperiod=26, matype=0)[-1], 0.0)
        out["trix"] = _safe(talib.TRIX(cl, timeperiod=15)[-1], 0.0)
        out["ultimate_oscillator"] = _safe(talib.ULTOSC(hi, lo, cl, timeperiod1=7, timeperiod2=14, timeperiod3=28)[-1], 50.0)
        bop_ser = talib.BOP(op, hi, lo, cl)
        out["balance_of_power"] = _clamp(float(np.nanmean(bop_ser[-min(5, len(bop_ser)) :])), -1.0, 1.0)
    else:
        # Numpy fallbacks for talib-specific momentum indicators
        if cl.size >= 11:
            out["roc"] = float((cl[-1] - cl[-11]) / cl[-11] * 100.0) if cl[-11] != 0 else 0.0
            out["momentum"] = float(cl[-1] - cl[-11])
        else:
            out["roc"] = 0.0
            out["momentum"] = 0.0
        if cl.size >= 26:
            e12 = _ema_arr(cl, 12)
            e26 = _ema_arr(cl, 26)
            out["ppo"] = float((e12[-1] - e26[-1]) / e26[-1] * 100.0) if e26[-1] != 0 else 0.0
        else:
            out["ppo"] = 0.0
        if cl.size >= 45:
            te1 = _ema_arr(cl, 15)
            te2 = _ema_arr(te1, 15)
            te3 = _ema_arr(te2, 15)
            out["trix"] = float((te3[-1] - te3[-2]) / te3[-2] * 100.0) if te3.size >= 2 and te3[-2] != 0 else 0.0
        else:
            out["trix"] = 0.0
        out["ultimate_oscillator"] = 50.0
        if cl.size >= 2:
            hl = hi[-1] - lo[-1]
            out["balance_of_power"] = _clamp(float((cl[-1] - op[-1]) / hl) if hl > 0 else 0.0, -1.0, 1.0)
        else:
            out["balance_of_power"] = 0.0

    _record_feature_provenance(
        provenance,
        "balance_of_power",
        "CALCULATED",
        "talib BOP or (close-open)/(high-low) on real OHLCV; clamped [-1,1]; near-zero valid when balanced",
        learning_allowed=True,
    )

    # Numpy-native momentum indicators (always computed)
    median = (hi + lo) / 2.0
    out["awesome_oscillator"] = float(_rolling_mean(median, 5) - _rolling_mean(median, 34))

    if cl.size >= 15:
        dm = ((hi[-14:] + lo[-14:]) / 2.0) - ((hi[-15:-1] + lo[-15:-1]) / 2.0)
        br = np.where((hi[-14:] - lo[-14:]) == 0, 1.0, (hi[-14:] - lo[-14:]))
        emv = dm / br * (vo[-14:] / 1e6)
        out["ease_of_movement"] = float(np.mean(emv))
    else:
        out["ease_of_movement"] = 0.0

    out["vortex_vi_plus"] = 1.0
    out["vortex_vi_minus"] = 1.0
    if cl.size >= 15:
        tr_v = np.maximum(hi[1:] - lo[1:], np.maximum(np.abs(hi[1:] - cl[:-1]), np.abs(lo[1:] - cl[:-1])))
        vm_plus = np.abs(hi[1:] - lo[:-1])
        vm_minus = np.abs(lo[1:] - hi[:-1])
        n_v = 14
        trn = float(np.sum(tr_v[-n_v:])) if tr_v.size >= n_v else float(np.sum(tr_v))
        vmp = float(np.sum(vm_plus[-n_v:])) if vm_plus.size >= n_v else float(np.sum(vm_plus))
        vmm = float(np.sum(vm_minus[-n_v:])) if vm_minus.size >= n_v else float(np.sum(vm_minus))
        out["vortex_vi_plus"] = 0.0 if trn == 0 else float(vmp / trn)
        out["vortex_vi_minus"] = 0.0 if trn == 0 else float(vmm / trn)

    out["kst"] = float(out["roc"])
    r = _returns(cl)
    out["tsi"] = float(np.mean(r[-25:]) * 100.0) if r.size >= 25 else 0.0

    if hi.size >= 25:
        period_a = 25
        hh_a = hi[-period_a:]
        ll_a = lo[-period_a:]
        days_since_hh = period_a - 1 - int(np.argmax(hh_a))
        days_since_ll = period_a - 1 - int(np.argmin(ll_a))
        out["aroon_up"] = float(100.0 * (period_a - 1 - days_since_hh) / (period_a - 1))
        out["aroon_down"] = float(100.0 * (period_a - 1 - days_since_ll) / (period_a - 1))
    else:
        out["aroon_up"] = 50.0
        out["aroon_down"] = 50.0

    if hi.size >= 35:
        rng = hi - lo
        ema1 = _ema_arr(rng, 9)
        ema2 = _ema_arr(ema1, 9)
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = np.where(ema2 == 0, 0.0, ema1 / ema2)
            ratio = np.nan_to_num(ratio, nan=0.0, posinf=0.0, neginf=0.0)
        out["mass_index"] = float(np.sum(ratio[-25:]))
    else:
        out["mass_index"] = 0.0

    # -------------------------
    # Trend (63-72)
    # -------------------------
    if talib is not None and cl.size >= 30:
        out["adx"] = _safe(talib.ADX(hi, lo, cl, timeperiod=14)[-1], 25.0)
        out["di_plus"] = _safe(talib.PLUS_DI(hi, lo, cl, timeperiod=14)[-1], 25.0)
        out["di_minus"] = _safe(talib.MINUS_DI(hi, lo, cl, timeperiod=14)[-1], 25.0)
        out["aroon_oscillator"] = _safe(talib.AROONOSC(hi, lo, timeperiod=14)[-1], 0.0)
    elif cl.size >= 29:
        # ADX / DI+/- numpy implementation (Wilder smoothed)
        up_move = np.diff(hi)
        dn_move = -np.diff(lo)
        plus_dm = np.where((up_move > dn_move) & (up_move > 0), up_move, 0.0)
        minus_dm = np.where((dn_move > up_move) & (dn_move > 0), dn_move, 0.0)
        tr_t = _true_range(hi, lo, cl)
        period_t = 14
        atr_s = float(np.mean(tr_t[:period_t]))
        plus_s = float(np.mean(plus_dm[:period_t]))
        minus_s = float(np.mean(minus_dm[:period_t]))
        dx_vals = []
        for ix in range(period_t, len(tr_t)):
            atr_s = (atr_s * (period_t - 1) + float(tr_t[ix])) / period_t
            plus_s = (plus_s * (period_t - 1) + float(plus_dm[ix])) / period_t
            minus_s = (minus_s * (period_t - 1) + float(minus_dm[ix])) / period_t
            if atr_s > 1e-15:
                pdi = 100.0 * plus_s / atr_s
                mdi = 100.0 * minus_s / atr_s
                di_sum = pdi + mdi
                dx_vals.append(100.0 * abs(pdi - mdi) / di_sum if di_sum > 1e-15 else 0.0)
        if len(dx_vals) >= period_t:
            adx_v = float(np.mean(dx_vals[:period_t]))
            for ix in range(period_t, len(dx_vals)):
                adx_v = (adx_v * (period_t - 1) + dx_vals[ix]) / period_t
            out["adx"] = float(np.clip(adx_v, 0.0, 100.0))
        else:
            out["adx"] = 25.0
        out["di_plus"] = float(np.clip(100.0 * plus_s / atr_s, 0, 100)) if atr_s > 1e-15 else 25.0
        out["di_minus"] = float(np.clip(100.0 * minus_s / atr_s, 0, 100)) if atr_s > 1e-15 else 25.0
        out["aroon_oscillator"] = out["aroon_up"] - out["aroon_down"]
    else:
        out["adx"] = 25.0
        out["di_plus"] = 25.0
        out["di_minus"] = 25.0
        out["aroon_oscillator"] = 0.0

    # Ichimoku — partial windows when fewer than 52 bars (never substitute price)
    tenkan, kijun, senkou_a, senkou_b = _ichimoku_from_ohlcv(hi, lo)
    out["ichimoku_tenkan"] = tenkan
    out["ichimoku_kijun"] = kijun
    out["ichimoku_senkou_a"] = senkou_a
    out["ichimoku_senkou_b"] = senkou_b
    ichi_status = "CALCULATED" if n_bars >= 52 else ("WARMUP" if n_bars >= 9 else "WARMUP")
    ichi_src = f"ichimoku native ohlcv bars={n_bars}"
    for iname in ("ichimoku_tenkan", "ichimoku_kijun", "ichimoku_senkou_a", "ichimoku_senkou_b"):
        _record_feature_provenance(provenance, iname, ichi_status, ichi_src, learning_allowed=ichi_status == "CALCULATED")

    out["psar"] = out["parabolic_sar"]
    out["trend_strength"] = float(out["adx"] / 100.0)  # deterministic scaling

    # -------------------------
    # Volume profile block (73-80)
    # -------------------------
    out["volume_ma_5"] = _rolling_mean(vo, 5)
    out["volume_ma_10"] = _rolling_mean(vo, 10)
    out["volume_ma_20"] = _rolling_mean(vo, 20)
    out["volume_ratio"] = 1.0 if vo.size < 2 or vo[-2] == 0 else float(vo[-1] / vo[-2])

    out["volume_price_trend"] = _volume_price_trend_rolling(cl, vo, window=50)
    _record_feature_provenance(
        provenance,
        "volume_price_trend",
        "CALCULATED",
        "rolling 50-bar VPT sum volume-normalized; not single-bar reset",
        learning_allowed=True,
    )

    nvi_v, pvi_v = _nvi_pvi_from_ohlcv(cl, vo)
    out["negative_volume_index"] = float(nvi_v)
    out["positive_volume_index"] = float(pvi_v)
    # volume_weighted_price (10-bar VWAP); WARMUP until enough history+volume (v6 may dedupe vs vwap slot)
    if cl.size >= 10 and float(np.sum(vo[-10:])) > 1e-12:
        out["volume_weighted_price"] = float(np.average(cl[-10:], weights=vo[-10:]))
        _record_feature_provenance(
            provenance,
            "volume_weighted_price",
            "CALCULATED",
            "10-bar volume-weighted price on 1m OHLCV",
            learning_allowed=True,
        )
    else:
        out["volume_weighted_price"] = out["price"]
        _record_feature_provenance(
            provenance,
            "volume_weighted_price",
            "WARMUP",
            "needs>=10 bars with volume; clears when history sufficient; v6 dedupe candidate vs vwap",
            learning_allowed=False,
        )

    # -------------------------
    # Sentiment (81-90) — use ``is not None`` so an empty dict still applies explicit zeros from merge
    # -------------------------
    if sentiment is not None:
        out["fear_greed_index"] = _safe(sentiment.get("fear_greed_index"), 0.0)
        out["social_sentiment"] = _safe(sentiment.get("social_sentiment"), 0.0)
        out["news_sentiment"] = _safe(sentiment.get("news_sentiment"), 0.0)
        # put_call_ratio handled after microstructure block (UNSUPPORTED unless options source)
        out["vix"] = _safe(sentiment.get("vix"), 0.0)
        out["market_cap"] = _safe(sentiment.get("market_cap"), 0.0)
        out["supply"] = _safe(sentiment.get("supply"), 0.0)
        out["circulating_supply"] = _safe(sentiment.get("circulating_supply"), 0.0)
        out["max_supply"] = _safe(sentiment.get("max_supply"), 0.0)
        out["market_dominance"] = _safe(sentiment.get("market_dominance"), 0.0)

    # -------------------------
    # Time (91-100) - derived from last candle timestamp
    # -------------------------
    last_ts = int(ts[-1])
    # timestamps may be ms; normalize to seconds
    dt = datetime.fromtimestamp(last_ts / 1000 if last_ts > 10_000_000_000 else last_ts, tz=timezone.utc)
    out["hour"] = float(dt.hour)
    out["day_of_week"] = float(dt.weekday())
    out["day_of_month"] = float(dt.day)
    out["month"] = float(dt.month)
    out["iso_weekday"] = float(dt.isoweekday())
    out["day_of_year"] = float(dt.timetuple().tm_yday)
    out["hour_12h"] = float(dt.hour % 12)
    out["minute"] = float(dt.minute)
    out["second"] = float(dt.second)
    out["seconds_since_midnight"] = float(dt.hour * 3600 + dt.minute * 60 + dt.second)
    _record_feature_provenance(
        provenance,
        "second",
        "LOW_IMPORTANCE_TIME_FIELD_NORMAL",
        "bar-close alignment; near-zero at minute boundaries; rank/learning neutral",
        trust_score=FEATURE_TRUST_SCORES["LOW_IMPORTANCE_TIME_FIELD_NORMAL"],
        learning_allowed=False,
    )

    # -------------------------
    # Advanced technical (101-108) — swing range / classic pivot (never price fallback)
    # -------------------------
    look = min(20, n_bars)
    fibs = _swing_range_levels(hi, lo, look)
    if fibs:
        out["fibonacci_retracement_23.6"], out["fibonacci_retracement_38.2"], out["fibonacci_retracement_61.8"] = fibs
        fib_status = "CALCULATED" if look >= 20 else "WARMUP"
        fib_src = f"swing_high_low lookback={look} 1m"
        for fname in ("fibonacci_retracement_23.6", "fibonacci_retracement_38.2", "fibonacci_retracement_61.8"):
            _record_feature_provenance(
                provenance,
                fname,
                fib_status,
                fib_src,
                learning_allowed=fib_status == "CALCULATED",
            )
    else:
        for fname in ("fibonacci_retracement_23.6", "fibonacci_retracement_38.2", "fibonacci_retracement_61.8"):
            out[fname] = 0.0
            _record_feature_provenance(provenance, fname, "WARMUP", "insufficient bars for swing range", learning_allowed=False)

    pivot = float((out["high"] + out["low"] + out["price"]) / 3.0)
    hl_rng = float(out["high"] - out["low"])
    out["pivot_point"] = pivot
    out["resistance_1"] = float(pivot + hl_rng * 0.382)
    out["resistance_2"] = float(pivot + hl_rng * 0.618)
    out["support_1"] = float(pivot - hl_rng * 0.382)
    out["support_2"] = float(pivot - hl_rng * 0.618)
    for fname in ("pivot_point", "resistance_1", "resistance_2", "support_1", "support_2"):
        _record_feature_provenance(provenance, fname, "CALCULATED", "classic pivot from session H/L/C 1m")

    # -------------------------
    # Advanced volume (109-116)
    # -------------------------
    if volume_profile:
        out["volume_profile_poc"] = _safe(volume_profile.get("poc"), 0.0)
        out["volume_profile_vah"] = _safe(volume_profile.get("vah"), 0.0)
        out["volume_profile_val"] = _safe(volume_profile.get("val"), 0.0)
        for fname in ("volume_profile_poc", "volume_profile_vah", "volume_profile_val"):
            _record_feature_provenance(
                provenance,
                fname,
                "CALCULATED_PROXY",
                "redis volume_profile hash proxy",
                trust_score=0.75,
                learning_allowed=False,
            )
    elif cl.size >= 20:
        poc_p, vah_p, val_p = _volume_profile_proxy_poc_vah_val(hi, lo, cl, vo, lookback=50)
        out["volume_profile_poc"] = poc_p
        out["volume_profile_vah"] = vah_p
        out["volume_profile_val"] = val_p
        for fname in ("volume_profile_poc", "volume_profile_vah", "volume_profile_val"):
            _record_feature_provenance(
                provenance,
                fname,
                "CALCULATED_PROXY",
                "ohlcv volume-by-price proxy; not live trade tape",
                trust_score=0.62,
                learning_allowed=False,
            )

    # vwap/twap over last 50 bars
    n = 50 if cl.size >= 50 else int(cl.size)
    if n > 0:
        voln = vo[-n:]
        if float(np.sum(voln)) != 0.0:
            out["vwap"] = float(np.average(cl[-n:], weights=voln))
        out["twap"] = float(np.mean(cl[-n:]))

        # volume delta / order_flow: volume * sign(close-open) — OHLCV proxy only
        sign = np.sign(cl[-n:] - op[-n:])
        delta = float(np.sum(voln * sign))
        out["volume_delta"] = delta
        out["volume_imbalance"] = 0.0 if float(np.sum(voln)) == 0.0 else float(delta / float(np.sum(voln)))
        out["order_flow"] = out["volume_imbalance"]
        for fname in ("volume_delta", "volume_imbalance", "order_flow"):
            _record_feature_provenance(
                provenance,
                fname,
                "CALCULATED_PROXY",
                f"ohlcv signed-volume proxy n={n}; not live trade tape",
                trust_score=0.62,
                learning_allowed=False,
            )

    # -------------------------
    # Microstructure (117-124)
    # -------------------------
    ob_fresh = _orderbook_freshness_meta(orderbook, orderbook_age_sec)
    if orderbook:
        for k in (
            "bid_ask_spread",
            "order_book_imbalance",
            "market_depth",
            "liquidity_score",
            "price_impact",
            "market_efficiency",
        ):
            if k in orderbook:
                out[k] = _safe(orderbook.get(k), out[k])
                ob_status = "STALE" if ob_stale else "LIVE"
                base_trust = FEATURE_TRUST_SCORES["STALE" if ob_stale else "LIVE"]
                mod = float(ob_fresh.get("freshness_trust_modifier") or 1.0)
                _record_feature_provenance(
                    provenance,
                    k,
                    ob_status,
                    "redis/live orderbook",
                    age_seconds=ob_fresh.get("age_seconds"),
                    trust_score=base_trust * mod,
                    learning_allowed=not ob_stale,
                    orderbook_updated_at=ob_fresh.get("orderbook_updated_at"),
                    freshness_status=str(ob_fresh.get("freshness_status") or "UNKNOWN"),
                    freshness_trust_modifier=mod,
                )

    # price_skewness from returns (real calculation)
    r = _returns(cl)
    if out["price_skewness"] == 0.0:
        out["price_skewness"] = float(_skewness(r))
    if r.size >= 3:
        _record_feature_provenance(provenance, "price_skewness", "CALCULATED", "log-return skewness 1m", learning_allowed=True)
    else:
        _record_feature_provenance(provenance, "price_skewness", "WARMUP", "insufficient returns", learning_allowed=False)

    # volatility_smile: NOT options surface — explicit proxy only
    if out["volatility_smile"] == 0.0 and r.size >= 20:
        v5p = _rolling_std(r, 5)
        v20p = _rolling_std(r, 20)
        v60p = _rolling_std(r, 60)
        out["volatility_smile"] = float(v5p - 2.0 * v20p + v60p)
    _record_feature_provenance(
        provenance,
        "volatility_smile",
        "UNSUPPORTED_FOR_SPOT",
        "vol term-structure proxy; not options smile — spot unsupported",
        trust_score=0.0,
        learning_allowed=False,
    )

    # put_call_ratio: no crypto spot options chain unless explicitly env-wired
    pcr_explicit = sentiment is not None and str(sentiment.get("_put_call_source") or "") == "options"
    if pcr_explicit and sentiment is not None:
        out["put_call_ratio"] = _safe(sentiment.get("put_call_ratio"), 0.0)
        _record_feature_provenance(provenance, "put_call_ratio", "LIVE", "explicit options source")
    else:
        out["put_call_ratio"] = 0.0
        _record_feature_provenance(
            provenance,
            "put_call_ratio",
            "UNSUPPORTED_FOR_SPOT",
            "no spot options chain wired",
            trust_score=0.0,
            learning_allowed=False,
        )

    _finalize_feature_provenance(
        out,
        provenance,
        n_bars=n_bars,
        sentiment=sentiment,
        volume_profile=volume_profile,
        orderbook=orderbook,
        orderbook_age_sec=orderbook_age_sec,
        ohlcv_1d=ohlcv_1d,
    )

    return out


def _finalize_feature_provenance(
    out: dict[str, float],
    provenance: dict[str, dict[str, Any]] | None,
    *,
    n_bars: int,
    sentiment: dict[str, Any] | None,
    volume_profile: dict[str, Any] | None,
    orderbook: dict[str, Any] | None,
    orderbook_age_sec: float | None,
    ohlcv_1d: list[list] | None,
) -> None:
    """Fill provenance for features not explicitly tagged during build."""
    if provenance is None:
        return
    price = float(out.get("price") or 0.0)
    ob_stale = orderbook_age_sec is not None and orderbook_age_sec > 45.0

    basic = ("price", "high", "low", "open", "volume", "change_24h", "change_7d", "change_30d", "price_range", "typical_price")
    for name in basic:
        if name in provenance:
            continue
        val = float(out.get(name) or 0.0)
        if name == "price" and val <= 0:
            _record_feature_provenance(provenance, name, "MISSING", "no close price", learning_allowed=False)
        elif name in ("change_24h", "change_7d", "change_30d") and val == 0.0 and not ohlcv_1d:
            _record_feature_provenance(provenance, name, "WARMUP", "needs 1d series or sufficient 1m depth", learning_allowed=False)
        elif val == 0.0 and name not in ("change_24h", "change_7d", "change_30d"):
            _record_feature_provenance(provenance, name, "ZERO_DEFAULT", "zero after build", learning_allowed=False)
        else:
            src = "1m ohlcv live" if name in ("price", "high", "low", "open", "volume", "price_range", "typical_price") else "1d/1m change calc"
            _record_feature_provenance(provenance, name, "LIVE" if "1m" in src else "CALCULATED", src)

    tech_names = [n for n, i in FEATURE_MAPPING.items() if 11 <= i <= 72]
    for name in tech_names:
        if name in provenance:
            continue
        val = float(out.get(name) or 0.0)
        min_bars = {
            "ma_200": 200, "ma_100": 100, "ma_50": 50, "ema_50": 50, "rsi": 15, "adx": 29,
            "aroon_up": 25, "mass_index": 35, "trix": 45, "ppo": 26,
        }.get(name, 20)
        if n_bars < min_bars:
            _record_feature_provenance(provenance, name, "WARMUP", f"needs>={min_bars} bars have {n_bars}", learning_allowed=False)
        else:
            engine = "talib" if talib is not None else "numpy"
            _record_feature_provenance(provenance, name, "CALCULATED", f"{engine} on 1m ohlcv (zero ok)")

    vol_names = [n for n, i in FEATURE_MAPPING.items() if 38 <= i <= 47]
    for name in vol_names:
        if name in provenance:
            continue
        if name == "parabolic_sar" and abs(float(out.get(name) or 0) - price) < 1e-9 and n_bars < 15:
            _record_feature_provenance(provenance, name, "WARMUP", "SAR needs more bars", learning_allowed=False)
        elif float(out.get(name) or 0) == 0.0:
            _record_feature_provenance(provenance, name, "WARMUP" if n_bars < 15 else "ZERO_DEFAULT", "atr/vol block", learning_allowed=n_bars >= 15)
        else:
            _record_feature_provenance(provenance, name, "CALCULATED", "volatility from 1m ohlcv")

    vol_prof = [n for n, i in FEATURE_MAPPING.items() if 70 <= i <= 80]
    for name in vol_prof:
        if name in provenance:
            continue
        if name in ("negative_volume_index", "positive_volume_index"):
            _record_feature_provenance(provenance, name, "CALCULATED", "NVI/PVI classic rules on 1m ohlcv")
        elif name == "volume_weighted_price" and name in provenance:
            pass
        elif name == "volume_weighted_price" and abs(float(out.get(name) or 0) - price) < 1e-9:
            _record_feature_provenance(provenance, name, "WARMUP", "vwap needs volume", learning_allowed=False)
        else:
            _record_feature_provenance(provenance, name, "CALCULATED", "volume block 1m")

    sent_names = [n for n, i in FEATURE_MAPPING.items() if 78 <= i <= 90]
    for name in sent_names:
        if name in provenance or name == "put_call_ratio":
            continue
        val = float(out.get(name) or 0.0)
        if sentiment is None:
            _record_feature_provenance(provenance, name, "MISSING", "no sentiment payload", learning_allowed=False)
        elif val == 0.0:
            if name in ("social_sentiment", "news_sentiment") and sentiment is not None:
                _record_feature_provenance(
                    provenance,
                    name,
                    "CALCULATED",
                    "sentiment neutral/zero from live pipeline",
                    trust_score=0.75,
                )
            else:
                _record_feature_provenance(provenance, name, "MISSING", "sentiment key unset", learning_allowed=False)
        else:
            _record_feature_provenance(provenance, name, "LIVE", "sentiment/redis/api")

    time_names = [n for n, i in FEATURE_MAPPING.items() if 88 <= i <= 100]
    for name in time_names:
        if name in provenance:
            continue
        _record_feature_provenance(provenance, name, "CALCULATED", "candle timestamp utc")

    adv_vol = [n for n, i in FEATURE_MAPPING.items() if 106 <= i <= 116]
    for name in adv_vol:
        if name in provenance:
            continue
        if name in ("volume_profile_poc", "volume_profile_vah", "volume_profile_val"):
            continue
        elif name in ("vwap", "twap"):
            _record_feature_provenance(provenance, name, "CALCULATED", "rolling vwap/twap 1m")
        else:
            _record_feature_provenance(provenance, name, "MISSING", "not computed", learning_allowed=False)

    micro = [n for n, i in FEATURE_MAPPING.items() if 114 <= i <= 124]
    for name in micro:
        if name in provenance:
            continue
        if not orderbook:
            _record_feature_provenance(provenance, name, "MISSING", "no orderbook", learning_allowed=False)
        elif ob_stale:
            _record_feature_provenance(provenance, name, "STALE", "orderbook age exceeded", age_seconds=orderbook_age_sec, learning_allowed=False)
        else:
            _record_feature_provenance(provenance, name, "MISSING", "orderbook key absent", learning_allowed=False)


def build_feature_vector_124(
    *,
    symbol_ccxt: str,
    ohlcv: list[list],
    volume_profile: dict[str, Any] | None,
    orderbook: dict[str, Any] | None,
    ohlcv_1d: list[list] | None = None,
    sentiment: dict[str, Any] | None = None,
    provenance: dict[str, dict[str, Any]] | None = None,
    orderbook_age_sec: float | None = None,
) -> list[float]:
    """
    Canonical 124-feature vector for training and live inference.

    Callers supply OHLCV plus optional overlays (volume profile, order book, daily candles).
    Training and inference must pass the same ``sentiment`` policy; use None so slots stay 0 until
    ``ai_feature_fundamentals.merge_canonical_sentiment_payload`` / Redis supply real values (v2 uses
    this 124-vector as the leading block of the 145-dim contract).
    """
    from backend.services.feature_mapping import dict_to_feature_vector

    feature_dict = build_feature_dict_from_ohlcv(
        symbol_ccxt=symbol_ccxt,
        ohlcv=ohlcv,
        volume_profile=volume_profile,
        orderbook=orderbook,
        sentiment=sentiment,
        ohlcv_1d=ohlcv_1d,
        provenance=provenance,
        orderbook_age_sec=orderbook_age_sec,
    )
    return dict_to_feature_vector(feature_dict)
