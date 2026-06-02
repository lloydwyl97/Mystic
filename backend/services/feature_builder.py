from __future__ import annotations

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
    l = lo[sl]
    c = cl[sl]
    v = vo[sl]
    typ = (h + l + c) / 3.0
    w = np.maximum(v, 1e-18)
    poc = float(np.average(typ, weights=w))
    vah = float(np.max(h))
    val = float(np.min(l))
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
        return out

    ts, op, hi, lo, cl, vo = _ohlcv_arrays(ohlcv)

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
            clv = np.where(hl_diff == 0, 0.0, ((cl - lo) - (hi - cl)) / hl_diff)
            out["ad_line"] = float(np.sum(clv * vo))
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
        out["balance_of_power"] = _safe(float(np.nanmean(bop_ser[-min(5, len(bop_ser)) :])), 0.0)
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
            out["balance_of_power"] = float((cl[-1] - op[-1]) / hl) if hl > 0 else 0.0
        else:
            out["balance_of_power"] = 0.0

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

    # Ichimoku
    if cl.size >= 52:
        tenkan = (float(np.max(hi[-9:])) + float(np.min(lo[-9:]))) / 2.0
        kijun = (float(np.max(hi[-26:])) + float(np.min(lo[-26:]))) / 2.0
        senkou_a = (tenkan + kijun) / 2.0
        senkou_b = (float(np.max(hi[-52:])) + float(np.min(lo[-52:]))) / 2.0
        out["ichimoku_tenkan"] = tenkan
        out["ichimoku_kijun"] = kijun
        out["ichimoku_senkou_a"] = senkou_a
        out["ichimoku_senkou_b"] = senkou_b
    else:
        out["ichimoku_tenkan"] = out["price"]
        out["ichimoku_kijun"] = out["price"]
        out["ichimoku_senkou_a"] = out["price"]
        out["ichimoku_senkou_b"] = out["price"]

    out["psar"] = out["parabolic_sar"]
    out["trend_strength"] = float(out["adx"] / 100.0)  # deterministic scaling

    # -------------------------
    # Volume profile block (73-80)
    # -------------------------
    out["volume_ma_5"] = _rolling_mean(vo, 5)
    out["volume_ma_10"] = _rolling_mean(vo, 10)
    out["volume_ma_20"] = _rolling_mean(vo, 20)
    out["volume_ratio"] = 1.0 if vo.size < 2 or vo[-2] == 0 else float(vo[-1] / vo[-2])

    # volume_price_trend: (Δprice/price)*volume
    if cl.size >= 2 and cl[-2] != 0:
        out["volume_price_trend"] = float(((cl[-1] - cl[-2]) / cl[-2]) * vo[-1])
    else:
        out["volume_price_trend"] = 0.0

    nvi_v, pvi_v = _nvi_pvi_from_ohlcv(cl, vo)
    out["negative_volume_index"] = float(nvi_v)
    out["positive_volume_index"] = float(pvi_v)
    # volume_weighted_price (VWAP over last 10)
    if cl.size >= 10 and float(np.sum(vo[-10:])) != 0.0:
        out["volume_weighted_price"] = float(np.average(cl[-10:], weights=vo[-10:]))
    else:
        out["volume_weighted_price"] = out["price"]

    # -------------------------
    # Sentiment (81-90) — use ``is not None`` so an empty dict still applies explicit zeros from merge
    # -------------------------
    if sentiment is not None:
        out["fear_greed_index"] = _safe(sentiment.get("fear_greed_index"), 0.0)
        out["social_sentiment"] = _safe(sentiment.get("social_sentiment"), 0.0)
        out["news_sentiment"] = _safe(sentiment.get("news_sentiment"), 0.0)
        out["put_call_ratio"] = _safe(sentiment.get("put_call_ratio"), 0.0)
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

    # -------------------------
    # Advanced technical (101-108)
    # -------------------------
    if cl.size >= 20:
        high_20 = float(np.max(hi[-20:]))
        low_20 = float(np.min(lo[-20:]))
        r20 = high_20 - low_20
        out["fibonacci_retracement_23.6"] = low_20 + r20 * 0.236
        out["fibonacci_retracement_38.2"] = low_20 + r20 * 0.382
        out["fibonacci_retracement_61.8"] = low_20 + r20 * 0.618
    else:
        out["fibonacci_retracement_23.6"] = out["price"]
        out["fibonacci_retracement_38.2"] = out["price"]
        out["fibonacci_retracement_61.8"] = out["price"]

    pivot = float((out["high"] + out["low"] + out["price"]) / 3.0)
    out["pivot_point"] = pivot
    out["resistance_1"] = float(pivot + (out["high"] - out["low"]) * 0.382)
    out["resistance_2"] = float(pivot + (out["high"] - out["low"]) * 0.618)
    out["support_1"] = float(pivot - (out["high"] - out["low"]) * 0.382)
    out["support_2"] = float(pivot - (out["high"] - out["low"]) * 0.618)

    # -------------------------
    # Advanced volume (109-116)
    # -------------------------
    if volume_profile:
        out["volume_profile_poc"] = _safe(volume_profile.get("poc"), 0.0)
        out["volume_profile_vah"] = _safe(volume_profile.get("vah"), 0.0)
        out["volume_profile_val"] = _safe(volume_profile.get("val"), 0.0)
    elif cl.size >= 20:
        poc_p, vah_p, val_p = _volume_profile_proxy_poc_vah_val(hi, lo, cl, vo, lookback=50)
        out["volume_profile_poc"] = poc_p
        out["volume_profile_vah"] = vah_p
        out["volume_profile_val"] = val_p

    # vwap/twap over last 50 bars
    n = 50 if cl.size >= 50 else int(cl.size)
    if n > 0:
        voln = vo[-n:]
        if float(np.sum(voln)) != 0.0:
            out["vwap"] = float(np.average(cl[-n:], weights=voln))
        out["twap"] = float(np.mean(cl[-n:]))

        # volume delta / order_flow: volume * sign(close-open)
        sign = np.sign(cl[-n:] - op[-n:])
        delta = float(np.sum(voln * sign))
        out["volume_delta"] = delta
        out["volume_imbalance"] = 0.0 if float(np.sum(voln)) == 0.0 else float(delta / float(np.sum(voln)))
        out["order_flow"] = out["volume_imbalance"]

    # -------------------------
    # Microstructure (117-124)
    # -------------------------
    if orderbook:
        for k in (
            "bid_ask_spread",
            "order_book_imbalance",
            "market_depth",
            "liquidity_score",
            "price_impact",
            "market_efficiency",
            "volatility_smile",
            "price_skewness",
        ):
            if k in orderbook:
                out[k] = _safe(orderbook.get(k), out[k])

    # Always compute volatility_smile and price_skewness proxies from returns if not already set
    r = _returns(cl)
    if out["price_skewness"] == 0.0:
        out["price_skewness"] = float(_skewness(r))
    if out["volatility_smile"] == 0.0:
        # volatility_smile proxy: vol(5) - 2*vol(20) + vol(60)
        v5p = _rolling_std(r, 5)
        v20p = _rolling_std(r, 20)
        v60p = _rolling_std(r, 60)
        out["volatility_smile"] = float(v5p - 2.0 * v20p + v60p)

    return out


def build_feature_vector_124(
    *,
    symbol_ccxt: str,
    ohlcv: list[list],
    volume_profile: dict[str, Any] | None,
    orderbook: dict[str, Any] | None,
    ohlcv_1d: list[list] | None = None,
    sentiment: dict[str, Any] | None = None,
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
    )
    return dict_to_feature_vector(feature_dict)
