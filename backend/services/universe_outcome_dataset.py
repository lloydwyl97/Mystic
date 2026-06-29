"""
Universe-expansion labeled dataset — research only.

Extends 145 base features with liquidity / cross-symbol context features.
"""

from __future__ import annotations

import bisect
import math
from typing import Any

import numpy as np

from backend.services.ai_outcome_dataset import (
    Economics,
    build_features_145,
    compute_forward_labels_at_idx,
)
from backend.services.ltf_pattern_miner import resample_bars

Bar = dict[str, Any]

UNIVERSE_EXTRA_NAMES = (
    "uni_liquidity_score",
    "uni_spread_pct",
    "uni_rel_volume_24h",
    "uni_volatility_rank",
    "uni_btc_beta",
    "uni_strength_rank_24h",
    "uni_market_risk_regime",
    "uni_momentum_rank",
    "uni_vwap_distance_pct",
    "uni_trend_1h",
    "uni_trend_4h",
    "uni_reclaim_5m",
    "uni_reclaim_15m",
    "uni_reclaim_30m",
)


def _trend_slope(bars: list[Bar], n: int = 24) -> float:
    if len(bars) < n:
        return 0.0
    closes = [b["close"] for b in bars[-n:]]
    if closes[0] <= 0:
        return 0.0
    return (closes[-1] - closes[0]) / closes[0]


def _reclaim_signal(bars: list[Bar], ts: int) -> float:
    """1.0 if close reclaimed above prior bar high after dip below prior low."""
    idx = bisect.bisect_right([int(b["ts"]) for b in bars], ts) - 1
    if idx < 2:
        return 0.0
    prev, cur = bars[idx - 1], bars[idx]
    if cur["low"] < prev["low"] and cur["close"] > prev["high"]:
        return 1.0
    if cur["close"] > prev["close"] and cur["low"] >= prev["low"]:
        return 0.5
    return 0.0


def _btc_beta(symbol_returns: np.ndarray, btc_returns: np.ndarray) -> float:
    n = min(len(symbol_returns), len(btc_returns))
    if n < 10:
        return 0.0
    s = symbol_returns[-n:]
    b = btc_returns[-n:]
    vb = float(np.var(b))
    if vb < 1e-12:
        return 0.0
    return float(np.cov(s, b)[0, 1] / vb)


def build_universe_extra_features(
    ts: int,
    bars_by_tf: dict[str, list[Bar]],
    *,
    meta: dict[str, Any],
    btc_1h: list[Bar],
    strength_ranks: dict[str, float],
    vol_ranks: dict[str, float],
    symbol: str,
) -> list[float]:
    i1h = bisect.bisect_right([int(b["ts"]) for b in bars_by_tf.get("1h", [])], ts) - 1
    h1 = bars_by_tf["1h"][: i1h + 1] if i1h >= 0 else []
    i4h = bisect.bisect_right([int(b["ts"]) for b in bars_by_tf.get("4h", [])], ts) - 1
    h4 = bars_by_tf.get("4h", [])[: i4h + 1] if i4h >= 0 else []

    vol_usd = float(meta.get("daily_volume_usd") or 0)
    liq_score = min(1.0, math.log10(max(vol_usd, 1)) / 7.0)
    spread = float(meta.get("half_spread_pct") or 0) * 2
    rel_vol = float(meta.get("relative_volume") or 1.0)

    sym_ret = np.array([(h1[i]["close"] / h1[i - 1]["close"] - 1) for i in range(1, len(h1))]) if len(h1) > 2 else np.array([])
    btc_i = bisect.bisect_right([int(b["ts"]) for b in btc_1h], ts) - 1
    btc_slice = btc_1h[: btc_i + 1] if btc_i >= 0 else []
    btc_ret = np.array([(btc_slice[i]["close"] / btc_slice[i - 1]["close"] - 1) for i in range(1, len(btc_slice))]) if len(btc_slice) > 2 else np.array([])

    vwap_dist = float(meta.get("vwap_distance_pct") or 0)
    risk_regime = 1.0 if meta.get("regime") in ("trending_down", "bear") else 0.0

    return [
        liq_score,
        spread,
        rel_vol,
        float(vol_ranks.get(symbol, 0.5)),
        _btc_beta(sym_ret, btc_ret),
        float(strength_ranks.get(symbol, 0.5)),
        risk_regime,
        float(strength_ranks.get(symbol, 0.5)),
        vwap_dist,
        _trend_slope(h1, 24),
        _trend_slope(h4, 20),
        _reclaim_signal(bars_by_tf.get("5m", []), ts),
        _reclaim_signal(bars_by_tf.get("15m", []), ts),
        _reclaim_signal(bars_by_tf.get("30m", []), ts),
    ]


def build_universe_dataset_rows(
    symbol: str,
    bars_by_tf: dict[str, list[Bar]],
    start_ts: int,
    end_ts: int,
    *,
    sample_sec: int,
    half_spread: float,
    universe_meta: dict[str, Any],
    btc_1h: list[Bar],
    strength_ranks: dict[str, float],
    vol_ranks: dict[str, float],
    scalp: bool = False,
) -> list[dict[str, Any]]:
    econ = Economics(half_spread=half_spread)
    bars_1m = bars_by_tf.get("1m", [])
    if len(bars_1m) < 300:
        return []

    ts_arr = np.array([int(b["ts"]) for b in bars_1m], dtype=np.int64)
    high_arr = np.array([b["high"] for b in bars_1m], dtype=np.float64)
    low_arr = np.array([b["low"] for b in bars_1m], dtype=np.float64)
    close_arr = np.array([b["close"] for b in bars_1m], dtype=np.float64)

    primary = bars_by_tf.get("1m" if scalp else "1h", [])
    rows: list[dict[str, Any]] = []
    last_ts = 0
    for bar in primary:
        ts = int(bar["ts"])
        if ts < start_ts or ts > end_ts:
            continue
        if ts - last_ts < sample_sec:
            continue
        last_ts = ts

        feat = build_features_145(symbol, ts, bars_by_tf, half_spread=half_spread)
        if feat is None:
            continue
        vec145, meta = feat
        meta.update(universe_meta)
        extra = build_universe_extra_features(
            ts,
            bars_by_tf,
            meta=meta,
            btc_1h=btc_1h,
            strength_ranks=strength_ranks,
            vol_ranks=vol_ranks,
            symbol=symbol,
        )
        full_vec = list(vec145) + extra

        idx_1m = bisect.bisect_right(ts_arr, ts) - 1
        labels = compute_forward_labels_at_idx(
            idx_1m,
            ts,
            bar["close"],
            ts_arr,
            high_arr,
            low_arr,
            close_arr,
            econ,
            scalp=scalp,
        )
        if not labels:
            continue

        rows.append(
            {
                "symbol": symbol,
                "timestamp": ts,
                "timeframe": "1m" if scalp else "1h",
                "features": full_vec,
                "feature_dim": len(full_vec),
                "meta": meta,
                "labels": labels,
                "entry_price": bar["close"],
                "half_spread": half_spread,
            }
        )
    return rows


def compute_symbol_ranks(symbols_meta: list[dict], bars_1h_by_sym: dict[str, list[Bar]]) -> tuple[dict[str, float], dict[str, float]]:
    strength: dict[str, float] = {}
    volat: dict[str, float] = {}
    for m in symbols_meta:
        sym = m["ccxt_symbol"]
        bars = bars_1h_by_sym.get(sym, [])
        if len(bars) < 30:
            strength[sym] = 0.0
            volat[sym] = 0.0
            continue
        closes = [b["close"] for b in bars[-48:]]
        rets = [(closes[i] / closes[i - 1] - 1) for i in range(1, len(closes))]
        strength[sym] = (closes[-1] / closes[0] - 1) if closes[0] > 0 else 0.0
        volat[sym] = float(np.std(rets)) if rets else 0.0

    def _rank(d: dict[str, float]) -> dict[str, float]:
        items = sorted(d.items(), key=lambda x: x[1])
        n = len(items)
        if n <= 1:
            return dict.fromkeys(d, 0.5)
        return {k: i / (n - 1) for i, (k, _) in enumerate(items)}

    return _rank(strength), _rank(volat)


def load_symbol_bars(
    api_sym: str,
    ccxt_sym: str,
    start_ms: int,
    end_ms: int,
    cache_dir: Any,
) -> dict[str, list[Bar]]:
    from backend.services.universe_eligibility import fetch_klines_cached

    sym_bars: dict[str, list] = {}
    for iv in ("1m", "5m", "15m", "1h"):
        sym_bars[iv] = fetch_klines_cached(api_sym, iv, start_ms, end_ms, cache_dir)
    sym_bars["30m"] = resample_bars(sym_bars["15m"], 30)
    sym_bars["4h"] = resample_bars(sym_bars["1h"], 240)
    sym_bars["1d"] = resample_bars(sym_bars["1h"], 1440)
    return sym_bars
