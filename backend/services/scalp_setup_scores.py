"""SCALP setup-specific scores (rank only — no gates)."""

from __future__ import annotations

import math
from typing import Any

from backend.services.scalp_feature_contract import ALL_SCALP_SETUPS

SETUP_MICRO_BREAKOUT = "MICRO_BREAKOUT"
SETUP_MICRO_PULLBACK = "MICRO_PULLBACK_CONTINUATION"
SETUP_VWAP_RECLAIM = "VWAP_RECLAIM"
SETUP_LIQUIDITY_SWEEP = "LIQUIDITY_SWEEP_RECLAIM"
SETUP_MOMENTUM_BURST = "MOMENTUM_BURST"
SETUP_RANGE_EDGE = "RANGE_EDGE_SCALP"
SETUP_FAILED_MICRO = "FAILED_MICRO_BREAKDOWN"
SETUP_SPREAD_SAFE = "SPREAD_SAFE_CONTINUATION"


def _f(dd: dict[str, Any], key: str, default: float = 0.0) -> float:
    try:
        v = float(dd.get(key))
        return v if math.isfinite(v) else default
    except (TypeError, ValueError):
        return default


def _c01(v: float) -> float:
    return max(0.0, min(1.0, v))


def _score_micro_breakout(dd: dict[str, Any], blocks: dict[str, float]) -> float:
    parts = [
        0.25 * _c01(_f(dd, "mid_change_30s") / 0.002 + 0.5),
        0.20 * _c01(_f(dd, "kline_volume_ratio") / 2.0),
        0.15 * blocks.get("scalp_microstructure_score", 0.5),
        0.15 * _c01(1.0 - _f(dd, "spread_pct") / 0.003),
        0.10 * _c01(abs(_f(dd, "order_book_imbalance"))),
        0.10 * _c01(1.0 - _f(dd, "impact_pct") / 0.004),
        0.05 * _c01(_f(dd, "realized_volatility_pct") / 0.008),
    ]
    return round(_c01(sum(parts)), 4)


def _score_micro_pullback(dd: dict[str, Any], blocks: dict[str, float]) -> float:
    rsi = _f(dd, "kline_rsi_proxy", 50.0)
    rsi_reset = 1.0 - abs(rsi - 45.0) / 45.0
    parts = [
        0.22 * _c01(_f(dd, "kline_vwap_distance") / 0.002 + 0.5),
        0.20 * _c01(_f(dd, "mid_change_60s") / 0.003 + 0.5),
        0.18 * _c01(rsi_reset),
        0.15 * _c01(1.0 - _f(dd, "spread_pct") / 0.003),
        0.15 * blocks.get("scalp_momentum_score", 0.5),
        0.10 * _c01(_f(dd, "kline_volume_ratio") / 1.5),
    ]
    return round(_c01(sum(parts)), 4)


def _score_vwap_reclaim(dd: dict[str, Any], blocks: dict[str, float]) -> float:
    parts = [
        0.30 * _c01(-_f(dd, "kline_vwap_distance") / 0.002),
        0.20 * _c01(_f(dd, "kline_volume_ratio") / 2.0),
        0.20 * _c01(abs(_f(dd, "order_book_imbalance"))),
        0.15 * _c01(1.0 - _f(dd, "spread_pct") / 0.003),
        0.15 * _c01(_f(dd, "mid_change_15s") / 0.001 + 0.5),
    ]
    return round(_c01(sum(parts)), 4)


def _score_liquidity_sweep(dd: dict[str, Any], blocks: dict[str, float]) -> float:
    parts = [
        0.25 * _c01(_f(dd, "kline_range_position")),
        0.20 * _c01(_f(dd, "kline_volume_ratio") / 2.5),
        0.20 * blocks.get("scalp_depth_quality_score", 0.5),
        0.15 * _c01(_f(dd, "mid_change_15s") / 0.002 + 0.5),
        0.20 * _c01(1.0 - _f(dd, "spread_pct") / 0.004),
    ]
    return round(_c01(sum(parts)), 4)


def _score_momentum_burst(dd: dict[str, Any], blocks: dict[str, float]) -> float:
    parts = [
        0.30 * _c01(_f(dd, "mid_change_30s") / 0.003),
        0.25 * _c01(_f(dd, "kline_volume_ratio") / 2.5),
        0.20 * blocks.get("scalp_momentum_score", 0.5),
        0.15 * _c01(1.0 if dd.get("momentum_confirmed") else 0.0),
        0.10 * _c01(1.0 - _f(dd, "spread_pct") / 0.003),
    ]
    return round(_c01(sum(parts)), 4)


def _score_range_edge(dd: dict[str, Any], blocks: dict[str, float]) -> float:
    edge = abs(_f(dd, "kline_range_position") - 0.5) * 2.0
    parts = [
        0.25 * _c01(edge),
        0.20 * _c01(-abs(_f(dd, "kline_vwap_distance")) / 0.003 + 1.0),
        0.20 * _c01(1.0 - _f(dd, "kline_atr_pct") / 0.008),
        0.20 * _c01(1.0 - _f(dd, "spread_pct") / 0.003),
        0.15 * blocks.get("scalp_volatility_score", 0.5),
    ]
    return round(_c01(sum(parts)), 4)


def _score_failed_micro(dd: dict[str, Any], blocks: dict[str, float]) -> float:
    parts = [
        0.25 * _c01(1.0 - _f(dd, "kline_range_position")),
        0.25 * _c01(_f(dd, "mid_change_15s") / 0.002 + 0.5),
        0.20 * _c01(_f(dd, "kline_volume_ratio") / 2.0),
        0.15 * blocks.get("scalp_depth_quality_score", 0.5),
        0.15 * _c01(1.0 - _f(dd, "spread_pct") / 0.004),
    ]
    return round(_c01(sum(parts)), 4)


def _score_spread_safe(dd: dict[str, Any], blocks: dict[str, float]) -> float:
    parts = [
        0.35 * _c01(1.0 - _f(dd, "spread_pct") / 0.0025),
        0.25 * blocks.get("scalp_execution_quality_score", 0.5),
        0.20 * _c01(abs(_f(dd, "order_book_imbalance"))),
        0.20 * _c01(_f(dd, "mid_change_30s") / 0.002 + 0.5),
    ]
    return round(_c01(sum(parts)), 4)


_SETUP_FN = {
    SETUP_MICRO_BREAKOUT: _score_micro_breakout,
    SETUP_MICRO_PULLBACK: _score_micro_pullback,
    SETUP_VWAP_RECLAIM: _score_vwap_reclaim,
    SETUP_LIQUIDITY_SWEEP: _score_liquidity_sweep,
    SETUP_MOMENTUM_BURST: _score_momentum_burst,
    SETUP_RANGE_EDGE: _score_range_edge,
    SETUP_FAILED_MICRO: _score_failed_micro,
    SETUP_SPREAD_SAFE: _score_spread_safe,
}


def compute_all_setup_scores(data: dict[str, Any], blocks: dict[str, float]) -> dict[str, float]:
    return {name: _SETUP_FN.get(name, lambda _d, _b: 0.5)(data, blocks) for name in ALL_SCALP_SETUPS}


def compute_setup_score(setup: str, data: dict[str, Any], blocks: dict[str, float]) -> float:
    fn = _SETUP_FN.get(str(setup or "").upper())
    if not fn:
        return 0.5
    return fn(data, blocks)


def setup_score_rank_delta(setup: str, setup_scores: dict[str, float]) -> float:
    sc = float(setup_scores.get(str(setup or "").upper(), 0.5))
    return round(max(-0.05, min(0.05, (sc - 0.55) * 0.15)), 4)


__all__ = [
    "compute_all_setup_scores",
    "compute_setup_score",
    "setup_score_rank_delta",
]
