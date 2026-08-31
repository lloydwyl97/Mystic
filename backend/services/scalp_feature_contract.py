"""Canonical SCALP feature vector contract — separate from DAY 145-dim v5."""

from __future__ import annotations

import math
from typing import Any

SCALP_FEATURE_VERSION = 1
SCALP_FEATURE_DIM = 40

# Fixed-order scalp intelligence vector (not DAY 145).
SCALP_FEATURE_NAMES: tuple[str, ...] = (
    "spread_pct",
    "order_book_imbalance",
    "orderbook_age_sec",
    "redis_spread_pct",
    "best_bid",
    "best_ask",
    "mid_price",
    "mid_change_15s",
    "mid_change_30s",
    "mid_change_60s",
    "bid_change_15s",
    "bid_change_30s",
    "bid_change_60s",
    "momentum_confirmed",
    "flat_regime",
    "recent_range_pct",
    "realized_volatility_pct",
    "last_n_ticks_up_count",
    "momentum_sample_count",
    "kline_return_1m",
    "kline_return_3m",
    "kline_volume_ratio",
    "kline_rsi_proxy",
    "kline_vwap_distance",
    "kline_ema9_distance",
    "kline_atr_pct",
    "kline_range_position",
    "projected_gross_pct",
    "breakout_signal",
    "surplus_pct",
    "micro_regime_score",
    "adx_1h",
    "atr_1h_pct",
    "impact_pct",
    "depth_sufficient_flag",
    "expected_move_pct",
    "signal_score",
    "signal_confidence",
    "required_target_pct",
    "same_setup_today_count",
)

SCALP_FEATURE_BLOCKS: dict[str, tuple[int, int]] = {
    "microstructure": (1, 7),
    "momentum": (8, 19),
    "kline_1m": (20, 27),
    "gross_estimate": (28, 30),
    "micro_regime": (31, 33),
    "execution": (34, 36),
    "signal_meta": (37, 39),
    "memory": (40, 40),
}

SCALP_TRUST_SCORES: dict[str, float] = {
    "LIVE": 1.0,
    "CALCULATED": 0.92,
    "CALCULATED_PROXY": 0.62,
    "WARMUP": 0.35,
    "FALLBACK": 0.15,
    "MISSING": 0.0,
    "STALE": 0.15,
    "ZERO_DEFAULT": 0.0,
    "PLACEHOLDER": 0.0,
    "UNSUPPORTED_FOR_SPOT": 0.0,
}

LEARNING_ALLOWED_STATUSES: frozenset[str] = frozenset({"LIVE", "CALCULATED"})

# Strategy router name → canonical scalp setup type for learning buckets.
STRATEGY_TO_SCALP_SETUP: dict[str, str] = {
    "breakout_momentum": "MICRO_BREAKOUT",
    "compression_breakout": "MICRO_BREAKOUT",
    "volume_impulse_continuation": "MOMENTUM_BURST",
    "trend_pullback_micro": "MICRO_PULLBACK_CONTINUATION",
    "range_bounce_scalp": "RANGE_EDGE_SCALP",
    "vwap_ema_reclaim": "VWAP_RECLAIM",
    "failed_breakdown_reversal": "LIQUIDITY_SWEEP_RECLAIM",
    "failed_breakout_reversal": "FAILED_MICRO_BREAKDOWN",
    "orderbook_tape_scalp": "SPREAD_SAFE_CONTINUATION",
}

ALL_SCALP_SETUPS: tuple[str, ...] = (
    "MICRO_BREAKOUT",
    "MICRO_PULLBACK_CONTINUATION",
    "VWAP_RECLAIM",
    "LIQUIDITY_SWEEP_RECLAIM",
    "MOMENTUM_BURST",
    "RANGE_EDGE_SCALP",
    "FAILED_MICRO_BREAKDOWN",
    "SPREAD_SAFE_CONTINUATION",
)

# Micro regime bucket prefix for learning (maps from scalp_regime_classifier labels).
MICRO_REGIME_BUCKET: dict[str, str] = {
    "bull_trend": "micro_bull",
    "bear_trend": "micro_bear",
    "range": "micro_range",
    "chop": "micro_chop",
    "vol_expansion": "micro_vol_expansion",
    "vol_crush": "micro_vol_crush",
    "pump_continuation": "micro_bull",
    "dump_continuation": "micro_bear",
    "dump_reversal": "micro_range",
    "high_vol_breakout": "micro_vol_expansion",
    "low_vol_dead": "micro_chop",
}


def scalp_setup_bucket(micro_regime: str, setup: str) -> str:
    prefix = MICRO_REGIME_BUCKET.get(str(micro_regime or "").lower(), "micro_unknown")
    st = str(setup or "UNKNOWN").upper()
    return f"{prefix}::{st}"


def _block_for_index(idx0: int) -> str:
    one = idx0 + 1
    for block, (lo, hi) in SCALP_FEATURE_BLOCKS.items():
        if lo <= one <= hi:
            return block
    return "unknown"


def get_feature_name(one_based: int) -> str:
    if 1 <= one_based <= len(SCALP_FEATURE_NAMES):
        return SCALP_FEATURE_NAMES[one_based - 1]
    return f"unknown_{one_based}"


def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, float(v)))


def _safe(v: Any, default: float = 0.0) -> float:
    try:
        x = float(v)
        if math.isnan(x):
            return default
        return x
    except (TypeError, ValueError):
        return default


def _kline_features(bars: list[dict]) -> dict[str, float]:
    if not bars or len(bars) < 2:
        return {
            "kline_return_1m": 0.0,
            "kline_return_3m": 0.0,
            "kline_volume_ratio": 1.0,
            "kline_rsi_proxy": 50.0,
            "kline_vwap_distance": 0.0,
            "kline_ema9_distance": 0.0,
            "kline_atr_pct": 0.0,
            "kline_range_position": 0.5,
        }
    closes = [_safe(b.get("close")) for b in bars[-30:]]
    vols = [_safe(b.get("volume")) for b in bars[-30:]]
    hi = max(_safe(b.get("high")) for b in bars[-20:])
    lo = min(_safe(b.get("low")) for b in bars[-20:])
    c0, c1 = closes[-1], closes[-2]
    ret1 = (c0 - c1) / c1 if c1 else 0.0
    c3 = closes[-4] if len(closes) >= 4 else closes[0]
    ret3 = (c0 - c3) / c3 if c3 else 0.0
    vol_ratio = vols[-1] / (sum(vols[-10:-1]) / max(1, len(vols[-10:-1]))) if len(vols) > 2 else 1.0
    gains = [max(0.0, closes[i] - closes[i - 1]) for i in range(1, len(closes))]
    losses = [max(0.0, closes[i - 1] - closes[i]) for i in range(1, len(closes))]
    rs = (sum(gains[-14:]) / max(sum(losses[-14:]), 1e-12)) if len(gains) >= 14 else 1.0
    rsi = 100.0 - 100.0 / (1.0 + rs)
    vwap_num = sum(_safe(b.get("close")) * _safe(b.get("volume")) for b in bars[-10:])
    vwap_den = sum(_safe(b.get("volume")) for b in bars[-10:])
    vwap = vwap_num / vwap_den if vwap_den else c0
    vwap_dist = (c0 - vwap) / vwap if vwap else 0.0
    ema9 = sum(closes[-9:]) / min(9, len(closes))
    ema_dist = (c0 - ema9) / ema9 if ema9 else 0.0
    trs = []
    for i in range(-14, 0):
        if i - 1 >= -len(bars):
            h, low, pc = _safe(bars[i].get("high")), _safe(bars[i].get("low")), _safe(bars[i - 1].get("close"))
            trs.append(max(h - low, abs(h - pc), abs(low - pc)))
    atr = sum(trs) / len(trs) if trs else 0.0
    atr_pct = atr / c0 if c0 else 0.0
    rng = hi - lo
    range_pos = (c0 - lo) / rng if rng > 0 else 0.5
    return {
        "kline_return_1m": _clamp(ret1, -0.05, 0.05),
        "kline_return_3m": _clamp(ret3, -0.10, 0.10),
        "kline_volume_ratio": _clamp(vol_ratio, 0.0, 5.0),
        "kline_rsi_proxy": _clamp(rsi, 0.0, 100.0),
        "kline_vwap_distance": _clamp(vwap_dist, -0.05, 0.05),
        "kline_ema9_distance": _clamp(ema_dist, -0.05, 0.05),
        "kline_atr_pct": _clamp(atr_pct, 0.0, 0.10),
        "kline_range_position": _clamp(range_pos, 0.0, 1.0),
    }


def build_scalp_feature_vector(
    *,
    snap: Any,
    mom: Any,
    bars_1m: list[dict] | None = None,
    signal: Any | None = None,
    memory: dict[str, Any] | None = None,
    micro_regime: str = "",
    gross: dict[str, Any] | None = None,
) -> list[float]:
    mem = memory or {}
    gross = gross or {}
    mom_d = mom.as_dict() if hasattr(mom, "as_dict") else (mom if isinstance(mom, dict) else {})
    sig_d = signal.as_dict() if hasattr(signal, "as_dict") else (signal if isinstance(signal, dict) else {})
    kf = _kline_features(bars_1m or [])

    ob_age = _safe(getattr(snap, "orderbook_age_sec", 0.0))
    spread = _safe(getattr(snap, "spread_pct", 0.0))
    redis_sp = getattr(snap, "redis_spread_pct", None)
    imbalance = _safe(getattr(snap, "order_book_imbalance", 0.0))

    regime_score = 0.5
    if micro_regime in ("bull_trend", "pump_continuation", "high_vol_breakout"):
        regime_score = 0.75
    elif micro_regime in ("bear_trend", "dump_continuation"):
        regime_score = 0.25
    elif micro_regime in ("range", "chop", "low_vol_dead"):
        regime_score = 0.5

    values = {
        "spread_pct": spread,
        "order_book_imbalance": imbalance,
        "orderbook_age_sec": ob_age,
        "redis_spread_pct": _safe(redis_sp, spread) if redis_sp is not None else spread,
        "best_bid": _safe(getattr(snap, "best_bid", 0.0)),
        "best_ask": _safe(getattr(snap, "best_ask", 0.0)),
        "mid_price": _safe(getattr(snap, "mid", 0.0)),
        "mid_change_15s": _safe(mom_d.get("mid_change_15s")),
        "mid_change_30s": _safe(mom_d.get("mid_change_30s")),
        "mid_change_60s": _safe(mom_d.get("mid_change_60s")),
        "bid_change_15s": _safe(mom_d.get("bid_change_15s")),
        "bid_change_30s": _safe(mom_d.get("bid_change_30s")),
        "bid_change_60s": _safe(mom_d.get("bid_change_60s")),
        "momentum_confirmed": 1.0 if mom_d.get("momentum_confirmed") else 0.0,
        "flat_regime": 1.0 if mom_d.get("flat_regime") else 0.0,
        "recent_range_pct": _safe(mom_d.get("recent_range_pct")),
        "realized_volatility_pct": _safe(mom_d.get("realized_volatility_pct")),
        "last_n_ticks_up_count": _safe(mom_d.get("last_n_ticks_up_count")),
        "momentum_sample_count": _safe(mom_d.get("sample_count")),
        **kf,
        "projected_gross_pct": _safe(gross.get("projected_gross_pct")),
        "breakout_signal": 1.0 if gross.get("breakout_signal") else 0.0,
        "surplus_pct": _safe(gross.get("surplus_pct")),
        "micro_regime_score": regime_score,
        "adx_1h": _safe(gross.get("adx_1h"), 25.0),
        "atr_1h_pct": _safe(gross.get("atr_1h_pct")),
        "impact_pct": _safe(sig_d.get("impact_pct")),
        "depth_sufficient_flag": 1.0 if sig_d.get("depth_sufficient", True) else 0.0,
        "expected_move_pct": _safe(sig_d.get("expected_move_pct")),
        "signal_score": _safe(sig_d.get("score")),
        "signal_confidence": _safe(sig_d.get("confidence")),
        "required_target_pct": _safe(sig_d.get("required_target_pct")),
        "same_setup_today_count": _safe(mem.get("same_scalp_setup_today_count")),
    }
    return [float(values.get(n, 0.0)) for n in SCALP_FEATURE_NAMES]


__all__ = [
    "ALL_SCALP_SETUPS",
    "LEARNING_ALLOWED_STATUSES",
    "SCALP_FEATURE_BLOCKS",
    "SCALP_FEATURE_DIM",
    "SCALP_FEATURE_NAMES",
    "SCALP_FEATURE_VERSION",
    "SCALP_TRUST_SCORES",
    "STRATEGY_TO_SCALP_SETUP",
    "_block_for_index",
    "build_scalp_feature_vector",
    "get_feature_name",
    "scalp_setup_bucket",
]
