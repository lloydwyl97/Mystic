"""Setup-specific scores for DAY ranking (soft inputs only — no gates)."""

from __future__ import annotations

import math
from typing import Any

from backend.services.day_trade_thesis import (
    SETUP_BREAKOUT_CONTINUATION,
    SETUP_FAILED_BREAKDOWN_REVERSAL,
    SETUP_HTF_TREND_PULLBACK,
    SETUP_RANGE_BOUNCE,
)

# User-specified setup names (rank-only; not all have dedicated classifiers yet)
SETUP_EXHAUSTION_REVERSAL = "EXHAUSTION_REVERSAL"
SETUP_REVERSAL_BREAKOUT = "REVERSAL_BREAKOUT"

ALL_SCORED_SETUPS: tuple[str, ...] = (
    SETUP_HTF_TREND_PULLBACK,
    SETUP_BREAKOUT_CONTINUATION,
    SETUP_RANGE_BOUNCE,
    SETUP_FAILED_BREAKDOWN_REVERSAL,
    SETUP_EXHAUSTION_REVERSAL,
    SETUP_REVERSAL_BREAKOUT,
)


def _f(dd: dict[str, Any], key: str, default: float = 0.0) -> float:
    raw = dd.get(key)
    try:
        v = float(raw)
        return v if math.isfinite(v) else default
    except (TypeError, ValueError):
        return default


def _clamp01(v: float) -> float:
    return max(0.0, min(1.0, v))


def _score_htf_trend_pullback(dd: dict[str, Any], blocks: dict[str, float]) -> float:
    ema = _f(dd, "ema_alignment", 0.5)
    mom = _f(dd, "price_momentum", 0.0)
    rsi = _f(dd, "rsi", 50.0)
    adx = _f(dd, "adx", 25.0)
    rs = _f(dd, "ctx_rs_btc", 0.0) + _f(dd, "ctx_rs_eth", 0.0)
    rsi_reset = 1.0 - abs(rsi - 45.0) / 45.0
    trend = blocks.get("trend_block_score", 0.5)
    vol_ctrl = 1.0 - min(1.0, _f(dd, "atr", 0.0) / max(_f(dd, "current_price", 1.0) * 0.05, 1e-9))
    parts = [
        0.22 * _clamp01(ema),
        0.15 * _clamp01((mom + 3.0) / 6.0),
        0.18 * _clamp01(rsi_reset),
        0.12 * _clamp01(adx / 40.0),
        0.13 * _clamp01((rs + 4.0) / 8.0),
        0.10 * trend,
        0.10 * _clamp01(vol_ctrl),
    ]
    return round(_clamp01(sum(parts)), 4)


def _score_breakout_continuation(dd: dict[str, Any], blocks: dict[str, float]) -> float:
    rel_vol = _f(dd, "ctx_relative_volume", 1.0)
    mom = _f(dd, "price_momentum", 0.0)
    spread = _f(dd, "spread_pct", 0.0)
    depth = abs(_f(dd, "ctx_depth_imbalance", 0.0))
    vol_block = blocks.get("volume_block_score", 0.5)
    ob = blocks.get("orderbook_block_score", 0.5)
    spread_ok = 1.0 - min(1.0, spread / 0.003)
    parts = [
        0.25 * _clamp01(rel_vol / 2.5),
        0.20 * _clamp01((mom + 2.0) / 5.0),
        0.15 * vol_block,
        0.15 * ob,
        0.15 * _clamp01(spread_ok),
        0.10 * _clamp01(depth),
    ]
    return round(_clamp01(sum(parts)), 4)


def _score_range_bounce(dd: dict[str, Any], blocks: dict[str, float]) -> float:
    rsi = _f(dd, "rsi", 50.0)
    chop = _f(dd, "chop_score", 0.5)
    vwap_dist = abs(_f(dd, "vwap_distance", 0.0))
    sr = blocks.get("trend_block_score", 0.5)
    rsi_ext = max(_clamp01((30.0 - rsi) / 30.0), _clamp01((rsi - 70.0) / 30.0))
    parts = [
        0.25 * rsi_ext,
        0.20 * _clamp01(chop),
        0.20 * _clamp01(1.0 - min(1.0, vwap_dist / 2.0)),
        0.20 * sr,
        0.15 * blocks.get("volatility_block_score", 0.5),
    ]
    return round(_clamp01(sum(parts)), 4)


def _score_failed_breakdown_reversal(dd: dict[str, Any], blocks: dict[str, float]) -> float:
    rsi = _f(dd, "rsi", 50.0)
    mom = _f(dd, "price_momentum", 0.0)
    spread = _f(dd, "spread_pct", 0.0)
    depth = _f(dd, "ctx_depth_imbalance", 0.0)
    rsi_rec = _clamp01((rsi - 25.0) / 35.0)
    mom_rev = _clamp01((mom + 1.5) / 3.0)
    spread_ok = 1.0 - min(1.0, spread / 0.004)
    parts = [
        0.22 * rsi_rec,
        0.20 * mom_rev,
        0.18 * _clamp01(abs(depth)),
        0.15 * _clamp01(spread_ok),
        0.15 * blocks.get("orderbook_block_score", 0.5),
        0.10 * blocks.get("sentiment_block_score", 0.5),
    ]
    return round(_clamp01(sum(parts)), 4)


def _score_exhaustion_reversal(dd: dict[str, Any], blocks: dict[str, float]) -> float:
    rsi = _f(dd, "rsi", 50.0)
    vol = blocks.get("volatility_block_score", 0.5)
    sent = blocks.get("sentiment_block_score", 0.5)
    rsi_ext = max(_clamp01((rsi - 75.0) / 25.0), _clamp01((25.0 - rsi) / 25.0))
    return round(_clamp01(0.40 * rsi_ext + 0.35 * vol + 0.25 * sent), 4)


def _score_reversal_breakout(dd: dict[str, Any], blocks: dict[str, float]) -> float:
    mom = _f(dd, "price_momentum", 0.0)
    vol = blocks.get("volume_block_score", 0.5)
    trend = blocks.get("trend_block_score", 0.5)
    return round(_clamp01(0.35 * _clamp01((mom + 2.0) / 5.0) + 0.35 * vol + 0.30 * trend), 4)


_SETUP_FN = {
    SETUP_HTF_TREND_PULLBACK: _score_htf_trend_pullback,
    SETUP_BREAKOUT_CONTINUATION: _score_breakout_continuation,
    SETUP_RANGE_BOUNCE: _score_range_bounce,
    SETUP_FAILED_BREAKDOWN_REVERSAL: _score_failed_breakdown_reversal,
    SETUP_EXHAUSTION_REVERSAL: _score_exhaustion_reversal,
    SETUP_REVERSAL_BREAKOUT: _score_reversal_breakout,
}


def compute_setup_score(setup: str, decision_data: dict[str, Any], block_scores: dict[str, float]) -> float:
    fn = _SETUP_FN.get(str(setup or "").strip().upper())
    if not fn:
        return 0.5
    return fn(decision_data, block_scores)


def compute_all_setup_scores(decision_data: dict[str, Any], block_scores: dict[str, float]) -> dict[str, float]:
    return {s: compute_setup_score(s, decision_data, block_scores) for s in ALL_SCORED_SETUPS}


def setup_score_rank_delta(active_setup: str, setup_scores: dict[str, float]) -> float:
    score = float(setup_scores.get(str(active_setup or "").strip().upper(), 0.5))
    return round(max(-0.05, min(0.05, (score - 0.55) * 0.12)), 4)


__all__ = [
    "ALL_SCORED_SETUPS",
    "compute_all_setup_scores",
    "compute_setup_score",
    "setup_score_rank_delta",
]
