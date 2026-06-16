"""
AI outcome-driven labeled dataset — replay only, no live I/O.

Builds 145-dim feature rows + forward MFE/MAE / target-before-adverse labels from historical bars.
"""
from __future__ import annotations

import bisect
import math
from dataclasses import dataclass
from typing import Any

import numpy as np

from backend.config.day_active_timeframes import DAY_ACTIVE_TIMEFRAMES
from backend.config.trading_economics import ORDERBOOK_HALF_SPREAD_ESTIMATE, SLIPPAGE_BUFFER, TAKER_FEE
from backend.services.ai_decision_contract import AI_FEATURE_DIM_V2, CONTEXT_DIMS_DAY_FULL
from backend.services.ai_feature_v2 import context_vector_day_full_mtf
from backend.services.ai_market_context import _summarize_tf
from backend.services.feature_builder import build_feature_vector_124
from backend.services.feature_mapping import FEATURE_MAPPING
from backend.services.ltf_pattern_miner import bars_up_to, htf_context, resample_bars

Bar = dict[str, Any]

HORIZONS_SEC = {
    "1h": 3600,
    "3h": 10800,
    "6h": 21600,
    "12h": 43200,
    "24h": 86400,
    "48h": 172800,
    "72h": 259200,
}
SCALP_HORIZONS_SEC = {"5m": 300, "10m": 600, "15m": 900, "30m": 1800}

DAY_SAMPLE_TFS = ("1h",)
FEATURE_NAMES_145 = tuple(FEATURE_MAPPING.keys()) + tuple(CONTEXT_DIMS_DAY_FULL)


@dataclass
class Economics:
    taker_fee: float = TAKER_FEE
    half_spread: float = ORDERBOOK_HALF_SPREAD_ESTIMATE
    slippage: float = SLIPPAGE_BUFFER

    def roundtrip_pct(self) -> float:
        return 2 * self.taker_fee + 2 * self.half_spread + 2 * self.slippage

    def net_move_pct(self, gross_pct: float) -> float:
        return gross_pct - self.roundtrip_pct()


def _bars_to_ohlcv(bars: list[Bar]) -> list[list[float]]:
    return [[float(b["ts"]) * 1000, b["open"], b["high"], b["low"], b["close"], b["volume"]] for b in bars]


def _bar_index_up_to(bars: list[Bar], ts: int) -> int:
    if not bars:
        return -1
    ts_list = [int(b["ts"]) for b in bars]
    return bisect.bisect_right(ts_list, ts) - 1


def _candle_structure(bar: Bar) -> dict[str, float]:
    o, h, l, c = bar["open"], bar["high"], bar["low"], bar["close"]
    rng = max(h - l, 1e-12)
    body = abs(c - o)
    return {
        "body_pct": body / max(c, 1e-12),
        "upper_wick_pct": (h - max(o, c)) / rng,
        "lower_wick_pct": (min(o, c) - l) / rng,
        "bullish": 1.0 if c > o else 0.0,
    }


def _month_vec_from_1d(bars_1d: list[Bar]) -> list[float]:
    if len(bars_1d) < 22:
        return [0.0, 0.0]
    closes = [b["close"] for b in bars_1d[-26:]]
    if closes[0] <= 0:
        return [0.0, 0.0]
    log_ret = math.log(closes[-1] / closes[0])
    rets = [(closes[i] / closes[i - 1] - 1) for i in range(1, len(closes))]
    vol = float(np.std(rets)) if rets else 0.0
    return [log_ret, vol]


def build_features_145(
    symbol: str,
    ts: int,
    bars_by_tf: dict[str, list[Bar]],
    *,
    half_spread: float,
    idx_1m: int | None = None,
) -> tuple[list[float], dict[str, Any]] | None:
    bars_1m = bars_by_tf.get("1m", [])
    if idx_1m is None:
        idx_1m = _bar_index_up_to(bars_1m, ts)
    if idx_1m < 249:
        return None

    window = bars_1m[idx_1m - 249 : idx_1m + 1]
    ohlcv = _bars_to_ohlcv(window)

    bars_1d = bars_by_tf.get("1d")
    if bars_1d:
        i1d = _bar_index_up_to(bars_1d, ts)
        bars_1d_slice = bars_1d[: i1d + 1] if i1d >= 0 else []
    else:
        i1h = _bar_index_up_to(bars_by_tf["1h"], ts)
        bars_1d_slice = resample_bars(bars_by_tf["1h"][: i1h + 1], 1440) if i1h >= 0 else []
    ohlcv_1d = _bars_to_ohlcv(bars_1d_slice) if len(bars_1d_slice) >= 22 else None

    i1h = _bar_index_up_to(bars_by_tf["1h"], ts)
    if i1h < 24:
        return None
    h1 = bars_by_tf["1h"][: i1h + 1]
    i4h = _bar_index_up_to(bars_by_tf.get("4h", h1), ts)
    h4 = (bars_by_tf.get("4h") or [])[: i4h + 1] if i4h >= 0 else h1

    ctx_raw = htf_context(h1, h4, ts)
    vol_avg = sum(b["volume"] for b in h1[-24:]) / max(len(h1[-24:]), 1)
    rel_vol = h1[-1]["volume"] / max(vol_avg, 1e-9)
    vwap = sum((b["high"] + b["low"] + b["close"]) / 3 * b["volume"] for b in h1[-24:]) / max(
        sum(b["volume"] for b in h1[-24:]), 1e-9
    )
    price = bars_1m[idx_1m]["close"]

    ai_context = {
        "ctx_market_regime": ctx_raw.get("regime", "neutral"),
        "ctx_relative_volume": rel_vol,
        "ctx_spread_pct": half_spread * 2,
        "ctx_depth_imbalance": 0.0,
        "ctx_change_24h_pct": (price - h1[-25]["close"]) / h1[-25]["close"] if len(h1) >= 25 and h1[-25]["close"] > 0 else 0.0,
        "ctx_volume_24h_usd": sum(b["volume"] * b["close"] for b in h1[-24:]),
        "ctx_rs_btc": 0.0,
        "ctx_rs_eth": 0.0,
        "ctx_btc_dominance_proxy": 0.5,
        "ctx_sentiment_fear_greed": 0.0,
    }

    mtf_snaps: dict[str, dict] = {}
    for tf in DAY_ACTIVE_TIMEFRAMES:
        if tf == "1m":
            rows = ohlcv
        elif tf in bars_by_tf:
            idx = _bar_index_up_to(bars_by_tf[tf], ts)
            rows = _bars_to_ohlcv(bars_by_tf[tf][: idx + 1]) if idx >= 0 else None
        else:
            rows = None
        mtf_snaps[tf] = dict(_summarize_tf(rows))

    month_vec = _month_vec_from_1d(bars_1d_slice)
    tech124 = build_feature_vector_124(
        symbol_ccxt=symbol,
        ohlcv=ohlcv,
        volume_profile=None,
        orderbook={"spread_pct": half_spread * 2, "imbalance": 0.0},
        ohlcv_1d=ohlcv_1d,
        sentiment=None,
    )
    ctx21 = context_vector_day_full_mtf(ai_context, mtf_snapshots=mtf_snaps, month_four=month_vec)
    vec = list(tech124) + list(ctx21)
    if len(vec) != AI_FEATURE_DIM_V2:
        return None

    meta = {
        "regime": ctx_raw.get("regime"),
        "relative_volume": rel_vol,
        "vwap_distance_pct": (price - vwap) / vwap if vwap > 0 else 0.0,
        "spread_pct": half_spread * 2,
        "adx_proxy": ctx_raw.get("adx_proxy"),
        "rsi": ctx_raw.get("rsi"),
        "thesis_regime": ctx_raw.get("regime"),
        "router_regime": ctx_raw.get("regime"),
        **_candle_structure(bars_1m[idx_1m]),
    }
    return vec, meta


def _hit_key(tag: str) -> str:
    return "hit_100bp_before_75bp" if tag == "100_75" else f"hit_{tag}bp_before_{tag}bp"


def compute_forward_labels_at_idx(
    entry_idx: int,
    entry_ts: int,
    entry_price: float,
    ts_arr: np.ndarray,
    high_arr: np.ndarray,
    low_arr: np.ndarray,
    close_arr: np.ndarray,
    econ: Economics,
    *,
    scalp: bool = False,
) -> dict[str, Any]:
    if entry_idx < 0 or entry_idx >= len(ts_arr):
        return {}

    entry_fill = entry_price * (1 + econ.half_spread + econ.slippage)
    labels: dict[str, Any] = {}
    horizons = SCALP_HORIZONS_SEC if scalp else HORIZONS_SEC

    for name, sec in horizons.items():
        end_ts = entry_ts + sec
        j_end = bisect.bisect_right(ts_arr, end_ts, lo=entry_idx + 1)
        if j_end <= entry_idx + 1:
            labels[f"mfe_{name}"] = 0.0
            labels[f"mae_{name}"] = 0.0
            continue
        seg_hi = high_arr[entry_idx + 1 : j_end]
        seg_lo = low_arr[entry_idx + 1 : j_end]
        labels[f"mfe_{name}"] = float(np.max((seg_hi - entry_fill) / entry_fill))
        labels[f"mae_{name}"] = float(np.min((seg_lo - entry_fill) / entry_fill))

    targets = (
        (0.004, 0.004, "40"),
        (0.006, 0.006, "60"),
        (0.010, 0.0075, "100_75"),
    )
    max_hold_violation = False
    scalp_max_sec = 1800 if scalp else 259200
    j72 = bisect.bisect_right(ts_arr, entry_ts + scalp_max_sec, lo=entry_idx + 1)
    if j72 > entry_idx + 1 and (ts_arr[j72 - 1] - entry_ts) > scalp_max_sec:
        max_hold_violation = True

    seg_hi = high_arr[entry_idx + 1 : j72]
    seg_lo = low_arr[entry_idx + 1 : j72]
    seg_ts = ts_arr[entry_idx + 1 : j72]
    if seg_hi.size == 0:
        for _, _, tag in targets:
            labels[_hit_key(tag)] = False
        labels["time_to_profit_sec"] = None
        labels["time_to_adverse_sec"] = None
        labels["max_hold_violation"] = max_hold_violation
        labels["best_net_pct_72h"] = -999.0
        labels["best_net_hold_sec"] = 0
        labels["expected_net_pnl_pct"] = -999.0
        return labels

    net_hi = econ.net_move_pct((seg_hi - entry_fill) / entry_fill)
    net_lo = econ.net_move_pct((seg_lo - entry_fill) / entry_fill)
    holds = seg_ts - entry_ts

    first_profit: dict[str, int | None] = {}
    first_adverse: dict[str, int | None] = {}
    for tgt, adv, tag in targets:
        pi = np.where(net_hi >= tgt)[0]
        ai = np.where(net_lo <= -adv)[0]
        first_profit[tag] = int(holds[pi[0]]) if pi.size else None
        first_adverse[tag] = int(holds[ai[0]]) if ai.size else None
        fp, fa = first_profit[tag], first_adverse[tag]
        labels[_hit_key(tag)] = fp is not None and (fa is None or fp < fa)

    labels["time_to_profit_sec"] = first_profit.get("40")
    labels["time_to_adverse_sec"] = first_adverse.get("40")
    labels["max_hold_violation"] = max_hold_violation

    net_close = econ.net_move_pct((close_arr[entry_idx + 1 : j72] - entry_fill) / entry_fill)
    best_i = int(np.argmax(net_close))
    labels["best_net_pct_72h"] = float(net_close[best_i])
    labels["best_net_hold_sec"] = int(holds[best_i])
    labels["expected_net_pnl_pct"] = float(net_close[best_i])
    if scalp:
        labels["scalp_max_hold_sec"] = scalp_max_sec
    return labels


def build_dataset_rows(
    symbol: str,
    bars_by_tf: dict[str, list[Bar]],
    start_ts: int,
    end_ts: int,
    *,
    sample_sec: int = 900,
    half_spread: float,
    scalp: bool = False,
    timeframes: tuple[str, ...] | None = None,
) -> list[dict[str, Any]]:
    econ = Economics(half_spread=half_spread)
    bars_1m = bars_by_tf.get("1m", [])
    if len(bars_1m) < 300:
        return []

    ts_arr = np.array([int(b["ts"]) for b in bars_1m], dtype=np.int64)
    high_arr = np.array([b["high"] for b in bars_1m], dtype=np.float64)
    low_arr = np.array([b["low"] for b in bars_1m], dtype=np.float64)
    close_arr = np.array([b["close"] for b in bars_1m], dtype=np.float64)

    if scalp:
        primary_tfs = ("1m",)
    else:
        primary_tfs = timeframes or DAY_SAMPLE_TFS

    rows: list[dict[str, Any]] = []
    seen_ts: set[int] = set()
    built = 0

    for tf in primary_tfs:
        primary = bars_by_tf.get(tf, [])
        if not primary:
            continue
        last_ts = 0
        for bar in primary:
            ts = int(bar["ts"])
            if ts < start_ts or ts > end_ts:
                continue
            if ts - last_ts < sample_sec:
                continue
            if ts in seen_ts:
                continue
            last_ts = ts
            seen_ts.add(ts)

            idx_1m = _bar_index_up_to(bars_1m, ts)
            feat = build_features_145(symbol, ts, bars_by_tf, half_spread=half_spread, idx_1m=idx_1m)
            if feat is None:
                continue
            vec, meta = feat
            entry_price = bar["close"]
            labels = compute_forward_labels_at_idx(
                idx_1m, ts, entry_price, ts_arr, high_arr, low_arr, close_arr, econ, scalp=scalp
            )
            if not labels:
                continue

            rows.append({
                "symbol": symbol,
                "timestamp": ts,
                "timeframe": tf if not scalp else "1m",
                "features_145": vec,
                "meta": meta,
                "labels": labels,
                "entry_price": entry_price,
            })
            built += 1
            if built % 200 == 0:
                print(f"      {symbol} {tf}: {built} rows", flush=True)
    return rows
