"""
DAY feature-stack completion (items p15-p17, p19 of the institutional upgrade).

Pure, symbol-agnostic functions operating on OHLCV row lists already fetched
via ``day_active_market_bundle.async_fetch_day_active_ohlcv_bundle`` (rows
format: ``[ts_ms, open, high, low, close, volume]``). No network calls here —
callers pass in rows already cached by the existing DAY MTF bundle fetch, so
this module adds zero extra API load.

Covers:
  - p15: multi-horizon momentum vector (percent change over N bars per TF),
    distinct from the single 5-bar "price_momentum" scalar used by ranking.
  - p16: same-symbol historical RVOL (current bar volume vs this symbol's own
    rolling average), replacing the cross-symbol-median-only ``ctx_relative_volume``.
  - p17: ATR7/14/28 + realized volatility (stdev of log returns) at multiple
    bar windows + a same-symbol volatility percentile (where does the current
    ATR% rank against this symbol's own recent ATR% history).
  - p19: lagged cross-correlation of a symbol's returns against BTC's returns
    at short lags, to see whether the symbol tends to lead or lag BTC on the
    given TF (as *measured* evidence, not an assumed correlation-to-BTC snapshot).

Per the Mystic architecture rule: everything here is a continuous, additive
ranking/context input. Every function degrades to an explicit neutral value
on insufficient data (never raises, never returns a value that could look
like a pass/fail signal) so nothing here can accidentally become a hard gate.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

Row = list[float]
Rows = list[Row]

DEFAULT_MOMENTUM_HORIZONS: dict[str, int] = {
    "5m": 6,
    "15m": 8,
    "1h": 12,
    "4h": 12,
    "1d": 14,
}

DEFAULT_RVOL_TFS: tuple[str, ...] = ("1m", "5m", "15m", "30m", "1h")
RVOL_LOOKBACK_BARS = 20

DEFAULT_ATR_PERIODS: tuple[int, ...] = (7, 14, 28)
DEFAULT_REALIZED_VOL_WINDOWS: dict[str, int] = {
    "5m": 12,
    "15m": 16,
    "1h": 24,
}
VOL_PERCENTILE_LOOKBACK_BARS = 90


def _closes(rows: Rows | None) -> np.ndarray:
    if not rows:
        return np.array([], dtype=np.float64)
    try:
        return np.array([float(r[4]) for r in rows], dtype=np.float64)
    except (TypeError, IndexError, ValueError):
        return np.array([], dtype=np.float64)


def _volumes(rows: Rows | None) -> np.ndarray:
    if not rows:
        return np.array([], dtype=np.float64)
    try:
        return np.array([float(r[5]) for r in rows], dtype=np.float64)
    except (TypeError, IndexError, ValueError):
        return np.array([], dtype=np.float64)


def _atr_pct_for_period(rows: Rows | None, period: int) -> float:
    if not rows or len(rows) < period + 1:
        return 0.0
    highs = np.array([float(r[2]) for r in rows], dtype=np.float64)
    lows = np.array([float(r[3]) for r in rows], dtype=np.float64)
    closes = np.array([float(r[4]) for r in rows], dtype=np.float64)
    tr = np.maximum.reduce(
        [
            highs[1:] - lows[1:],
            np.abs(highs[1:] - closes[:-1]),
            np.abs(lows[1:] - closes[:-1]),
        ]
    )
    if len(tr) < period:
        return 0.0
    atr = float(np.mean(tr[-period:]))
    last_close = float(closes[-1]) or 1.0
    return atr / last_close


def momentum_pct(rows: Rows | None, lookback: int) -> float:
    """Percent change of close over `lookback` bars. 0.0 (neutral) if insufficient bars."""
    closes = _closes(rows)
    if len(closes) <= lookback:
        return 0.0
    base = float(closes[-lookback - 1])
    if base == 0.0:
        return 0.0
    return float((closes[-1] - base) / base)


def momentum_multi_horizon(bundle: dict[str, Rows], horizons: dict[str, int] | None = None) -> dict[str, float]:
    """DAY momentum vector across configured TFs. Each entry is a plain % change;
    missing/short TFs come back as 0.0 rather than being omitted, so downstream
    consumers always get a stable key set."""
    horizons = horizons or DEFAULT_MOMENTUM_HORIZONS
    out: dict[str, float] = {}
    for tf, lookback in horizons.items():
        rows = bundle.get(tf) if isinstance(bundle, dict) else None
        out[tf] = momentum_pct(rows, lookback)
    return out


def same_symbol_rvol(rows: Rows | None, lookback: int = RVOL_LOOKBACK_BARS) -> float:
    """Current bar volume / this symbol's own rolling mean volume over the
    preceding `lookback` bars on this TF. 1.0 (neutral — "typical volume") if
    insufficient history. This is the same-symbol counterpart to
    ai_market_context.py's cross-symbol-median ``ctx_relative_volume``."""
    vols = _volumes(rows)
    if len(vols) < lookback + 1:
        return 1.0
    baseline = float(np.mean(vols[-lookback - 1 : -1]))
    if baseline <= 0.0:
        return 1.0
    return float(vols[-1] / baseline)


def same_symbol_rvol_multi_horizon(bundle: dict[str, Rows], tfs: tuple[str, ...] = DEFAULT_RVOL_TFS) -> dict[str, float]:
    out: dict[str, float] = {}
    for tf in tfs:
        rows = bundle.get(tf) if isinstance(bundle, dict) else None
        out[tf] = same_symbol_rvol(rows)
    return out


def atr_pct_multi_period(rows: Rows | None, periods: tuple[int, ...] = DEFAULT_ATR_PERIODS) -> dict[int, float]:
    return {p: _atr_pct_for_period(rows, p) for p in periods}


def realized_vol_pct(rows: Rows | None, window: int) -> float:
    """Stdev of bar-over-bar log returns over `window` bars (raw, not annualized).
    0.0 if insufficient bars or a non-positive close is encountered."""
    closes = _closes(rows)
    if len(closes) < window + 1:
        return 0.0
    tail = closes[-(window + 1) :]
    if np.any(tail <= 0):
        return 0.0
    rets = np.diff(np.log(tail))
    if len(rets) < 2:
        return 0.0
    return float(np.std(rets))


def realized_vol_multi_horizon(bundle: dict[str, Rows], windows: dict[str, int] | None = None) -> dict[str, float]:
    windows = windows or DEFAULT_REALIZED_VOL_WINDOWS
    out: dict[str, float] = {}
    for tf, window in windows.items():
        rows = bundle.get(tf) if isinstance(bundle, dict) else None
        out[tf] = realized_vol_pct(rows, window)
    return out


def vol_percentile(
    rows: Rows | None,
    atr_period: int = 14,
    lookback: int = VOL_PERCENTILE_LOOKBACK_BARS,
) -> float:
    """Percentile rank (0..1) of the CURRENT ATR% within this symbol's OWN
    trailing `lookback`-bar ATR% history on this TF. 0.5 (neutral) if there
    isn't enough history to form a meaningful distribution. This is a
    same-symbol measure — never compares one symbol's vol against another's."""
    if not rows or len(rows) < atr_period + lookback + 1:
        return 0.5
    series = [_atr_pct_for_period(rows[: i + 1], atr_period) for i in range(len(rows) - lookback, len(rows))]
    if not series:
        return 0.5
    current = series[-1]
    rank = sum(1 for v in series if v <= current) / len(series)
    return float(rank)


@dataclass(frozen=True)
class LagCorrelationResult:
    tf: str
    best_lag_bars: int
    corr_at_best_lag: float
    same_bar_corr: float
    n_obs: int
    confidence: str  # insufficient_data | low_confidence | confident


def lagged_cross_correlation_vs_btc(
    symbol_rows: Rows | None,
    btc_rows: Rows | None,
    *,
    tf: str = "1m",
    max_lag_bars: int = 5,
    min_obs: int = 30,
) -> LagCorrelationResult:
    """Cross-correlation of a symbol's bar-over-bar returns against BTC's, at
    lags from -max_lag_bars..+max_lag_bars on the shared TF.

    Convention: lag > 0 means the SYMBOL's return at bar (t - lag) best
    matches BTC's return at bar t — i.e. the symbol LED BTC by `lag` bars.
    lag < 0 means the symbol LAGGED BTC by |lag| bars. lag == 0 means
    simultaneous (no measurable lead/lag).

    This is measured evidence for ranking, never an assumed same-bar
    correlation snapshot. Degrades to an explicit insufficient_data result
    (corr=0.0, lag=0) rather than raising or fabricating a confident value.
    """
    sym_closes = _closes(symbol_rows)
    btc_closes = _closes(btc_rows)
    n = min(len(sym_closes), len(btc_closes))
    if n < min_obs + max_lag_bars:
        return LagCorrelationResult(tf, 0, 0.0, 0.0, n, "insufficient_data")

    sym_closes = sym_closes[-n:]
    btc_closes = btc_closes[-n:]
    if np.any(sym_closes <= 0) or np.any(btc_closes <= 0):
        return LagCorrelationResult(tf, 0, 0.0, 0.0, n, "insufficient_data")

    sym_rets = np.diff(sym_closes) / sym_closes[:-1]
    btc_rets = np.diff(btc_closes) / btc_closes[:-1]
    m = len(sym_rets)

    def _corr_at_lag(lag: int) -> float:
        if lag > 0:
            a = sym_rets[: m - lag]
            b = btc_rets[lag:]
        elif lag < 0:
            a = sym_rets[-lag:]
            b = btc_rets[: m + lag]
        else:
            a = sym_rets
            b = btc_rets
        if len(a) < min_obs or np.std(a) < 1e-12 or np.std(b) < 1e-12:
            return 0.0
        c = float(np.corrcoef(a, b)[0, 1])
        return c if np.isfinite(c) else 0.0

    same_bar = _corr_at_lag(0)
    best_lag = 0
    best_corr = same_bar
    for lag in range(-max_lag_bars, max_lag_bars + 1):
        if lag == 0:
            continue
        c = _corr_at_lag(lag)
        if abs(c) > abs(best_corr):
            best_corr = c
            best_lag = lag

    if m - max_lag_bars < min_obs:
        confidence = "insufficient_data"
    elif m - max_lag_bars < min_obs * 2:
        confidence = "low_confidence"
    else:
        confidence = "confident"

    return LagCorrelationResult(tf, best_lag, float(best_corr), float(same_bar), m, confidence)


@dataclass(frozen=True)
class FeatureStackSnapshot:
    """Bundled output for a single symbol; every field additive/diagnostic —
    consumers append these as new ranking/context inputs, never as gates."""

    symbol: str
    momentum: dict[str, float] = field(default_factory=dict)
    rvol: dict[str, float] = field(default_factory=dict)
    atr_periods_1h: dict[int, float] = field(default_factory=dict)
    realized_vol: dict[str, float] = field(default_factory=dict)
    vol_percentile_1h: float = 0.5
    btc_lag: LagCorrelationResult | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "symbol": self.symbol,
            "momentum": dict(self.momentum),
            "rvol": dict(self.rvol),
            "atr_periods_1h": dict(self.atr_periods_1h),
            "realized_vol": dict(self.realized_vol),
            "vol_percentile_1h": self.vol_percentile_1h,
            "btc_lag": {
                "tf": self.btc_lag.tf,
                "best_lag_bars": self.btc_lag.best_lag_bars,
                "corr_at_best_lag": self.btc_lag.corr_at_best_lag,
                "same_bar_corr": self.btc_lag.same_bar_corr,
                "n_obs": self.btc_lag.n_obs,
                "confidence": self.btc_lag.confidence,
            }
            if self.btc_lag is not None
            else None,
        }


def compute_feature_stack_snapshot(
    symbol: str,
    bundle: dict[str, Rows],
    *,
    btc_bundle: dict[str, Rows] | None = None,
    lag_tf: str = "1m",
) -> FeatureStackSnapshot:
    """Single entry point combining p15/p16/p17/p19 for one symbol's already-fetched
    multi-TF bundle. Never raises — any component that fails silently falls back
    to its own neutral default (see individual function docstrings)."""
    momentum = momentum_multi_horizon(bundle)
    rvol = same_symbol_rvol_multi_horizon(bundle)
    rows_1h = bundle.get("1h") if isinstance(bundle, dict) else None
    atr_periods_1h = atr_pct_multi_period(rows_1h)
    realized_vol = realized_vol_multi_horizon(bundle)
    vp = vol_percentile(rows_1h)

    btc_lag: LagCorrelationResult | None = None
    if btc_bundle is not None and symbol.upper() != "BTCUSDT":
        sym_rows_lag = bundle.get(lag_tf) if isinstance(bundle, dict) else None
        btc_rows_lag = btc_bundle.get(lag_tf) if isinstance(btc_bundle, dict) else None
        btc_lag = lagged_cross_correlation_vs_btc(sym_rows_lag, btc_rows_lag, tf=lag_tf)

    return FeatureStackSnapshot(
        symbol=symbol,
        momentum=momentum,
        rvol=rvol,
        atr_periods_1h=atr_periods_1h,
        realized_vol=realized_vol,
        vol_percentile_1h=vp,
        btc_lag=btc_lag,
    )


_MOMENTUM_HORIZON_SCALES: dict[str, float] = {
    "5m": 0.01,
    "15m": 0.015,
    "1h": 0.03,
    "4h": 0.06,
    "1d": 0.10,
}


def momentum_rvol_confirmation_signal(momentum: dict[str, float], rvol: dict[str, float]) -> float:
    """Item p15/p16 ranking promotion: combine the multi-horizon momentum
    vector's average signed direction with same-symbol RVOL as a
    volume-confirmation multiplier (high relative volume amplifies the
    momentum reading toward ranking; low relative volume damps it toward
    neutral) — never a gate, bounded [-1, 1], neutral 0.0 with no data.
    """
    if not momentum:
        return 0.0
    signed_vals = [max(-1.0, min(1.0, pct / _MOMENTUM_HORIZON_SCALES.get(tf, 0.03))) for tf, pct in momentum.items()]
    if not signed_vals:
        return 0.0
    momentum_signal = sum(signed_vals) / len(signed_vals)

    if rvol:
        rvol_avg = sum(rvol.values()) / len(rvol)
        confirmation = max(0.6, min(1.4, 0.6 + 0.4 * rvol_avg))
    else:
        confirmation = 1.0

    return max(-1.0, min(1.0, momentum_signal * confirmation))


def btc_lag_predictive_signal(btc_lag: LagCorrelationResult | None, btc_recent_return: float) -> float:
    """Item p19 ranking promotion: if this symbol has a CONFIDENT *measured*
    tendency to lag BTC by N bars with correlation c, then BTC's own most
    recent N-bar return times c is a learned (not assumed) leading
    indicator for this symbol's near-term direction. Zero unless the
    measured relationship is confident and the symbol genuinely lags BTC
    (never fires from an assumed same-bar correlation, and never fires when
    the symbol leads/moves simultaneously with BTC, since BTC's past has no
    predictive content for that case)."""
    if btc_lag is None or btc_lag.confidence != "confident":
        return 0.0
    if btc_lag.best_lag_bars >= 0:
        return 0.0
    signal = btc_lag.corr_at_best_lag * max(-1.0, min(1.0, btc_recent_return / 0.01))
    return max(-1.0, min(1.0, signal))


__all__ = [
    "DEFAULT_ATR_PERIODS",
    "DEFAULT_MOMENTUM_HORIZONS",
    "DEFAULT_REALIZED_VOL_WINDOWS",
    "DEFAULT_RVOL_TFS",
    "FeatureStackSnapshot",
    "LagCorrelationResult",
    "atr_pct_multi_period",
    "btc_lag_predictive_signal",
    "compute_feature_stack_snapshot",
    "lagged_cross_correlation_vs_btc",
    "momentum_multi_horizon",
    "momentum_pct",
    "momentum_rvol_confirmation_signal",
    "realized_vol_multi_horizon",
    "realized_vol_pct",
    "same_symbol_rvol",
    "same_symbol_rvol_multi_horizon",
    "vol_percentile",
]
