"""
ai_feature_v2 — canonical AI feature vector v2 (145 dims).

v2 = v1 (124-dim technical block from ``build_feature_vector_124``; **v3 callers must pass primary-clock OHLCV**)
    + 21 canonical context dims appended in CONTEXT_DIMS_V2 order.

This module is intentionally pure (no I/O). It composes outputs from
``backend.services.feature_builder`` and ``backend.services.ai_decision_contract``.

Layout (v2):
    [0..123]   = build_feature_vector_124(...)
    [124..144] = CONTEXT_DIMS_V2 in order

Both training (`backend.ai_training_pipeline`) and live inference
(`backend.services.ai_signal_generator`) MUST go through this module so the
feature contract cannot diverge between the two paths.
"""

from __future__ import annotations

import math
from typing import Any

from backend.config.day_active_timeframes import DAY_ACTIVE_TIMEFRAMES
from backend.services.ai_decision_contract import (
    AI_FEATURE_DIM_V1,
    AI_FEATURE_DIM_V2,
    CONTEXT_DIMS_DAY_FULL,
    CONTEXT_DIMS_V2,
)
from backend.services.feature_builder import build_feature_vector_124

# ---------------------------------------------------------------------------
# Helpers — convert raw ai_context payload (from ai_market_context) into the
# 21-dim ordered context vector.  All values are clamped to safe ranges so an
# unscaled outlier cannot blow up a tree-based model's leaves.
# ---------------------------------------------------------------------------


def _safe_float(v: Any, default: float = 0.0) -> float:
    try:
        if v is None:
            return float(default)
        return float(v)
    except (TypeError, ValueError):
        return float(default)


def _clip(x: float, lo: float, hi: float) -> float:
    if x < lo:
        return lo
    if x > hi:
        return hi
    return x


def _regime_to_signed(label: str | None) -> float:
    if not label:
        return 0.0
    s = str(label).strip().lower()
    if s in ("bull", "trend_up", "trending_up", "uptrend", "risk_on"):
        return 1.0
    if s in ("bear", "trend_down", "trending_down", "downtrend", "risk_off"):
        return -1.0
    return 0.0


def _liquidity_tier_norm(tier: Any) -> float:
    try:
        t = int(tier)
    except (TypeError, ValueError):
        t = 0
    if t <= 0:
        return 0.0
    if t >= 3:
        return 1.0
    return t / 3.0


def _ema_align_to_unit(v: Any) -> float:
    """Map an EMA-alignment score (commonly in [-1, +1]) to [0, 1]."""
    x = _safe_float(v, 0.0)
    return _clip((x + 1.0) * 0.5, 0.0, 1.0)


def _slope_norm(v: Any) -> float:
    """Slope is an unscaled % change; clip to a sane range so trees don't blow up."""
    x = _safe_float(v, 0.0)
    return _clip(x, -0.20, 0.20)


def _tf_dict(mtf: dict[str, Any] | None, tf: str) -> dict[str, Any]:
    if not isinstance(mtf, dict):
        return {}
    raw = mtf.get(tf)
    return raw if isinstance(raw, dict) else {}


def context_vector_from_ai_context(
    ai_context: dict[str, Any] | None,
    *,
    mtf: dict[str, Any] | None = None,
) -> list[float]:
    """
    Build the canonical 21-dim context vector from an ai_context payload.

    ``ai_context`` is the ai_context:{symbol} Redis hash (already string-coerced).
    ``mtf`` is the parsed mtf_json dict (per-timeframe stats). When omitted, we
    try to parse it from ``ai_context['mtf_json']``.
    """
    ctx = ai_context or {}
    if mtf is None:
        raw_mtf = ctx.get("mtf_json")
        if isinstance(raw_mtf, str) and raw_mtf:
            import json

            try:
                parsed = json.loads(raw_mtf)
                mtf = parsed if isinstance(parsed, dict) else {}
            except (ValueError, TypeError):
                mtf = {}
        elif isinstance(raw_mtf, dict):
            mtf = raw_mtf
        else:
            mtf = {}

    # MTF EMA alignments (5)
    align_5m = _ema_align_to_unit(_tf_dict(mtf, "5m").get("ema_align"))
    align_15m = _ema_align_to_unit(_tf_dict(mtf, "15m").get("ema_align"))
    align_1h = _ema_align_to_unit(_tf_dict(mtf, "1h").get("ema_align"))
    align_4h = _ema_align_to_unit(_tf_dict(mtf, "4h").get("ema_align"))
    align_1d = _ema_align_to_unit(_tf_dict(mtf, "1d").get("ema_align"))

    # MTF slopes (5)
    slope_5m = _slope_norm(_tf_dict(mtf, "5m").get("slope"))
    slope_15m = _slope_norm(_tf_dict(mtf, "15m").get("slope"))
    slope_1h = _slope_norm(_tf_dict(mtf, "1h").get("slope"))
    slope_4h = _slope_norm(_tf_dict(mtf, "4h").get("slope"))
    slope_1d = _slope_norm(_tf_dict(mtf, "1d").get("slope"))

    # 24h context (4)
    chg_24 = _clip(_safe_float(ctx.get("ctx_change_24h_pct"), 0.0), -0.5, 0.5)
    vol_24_usd = max(0.0, _safe_float(ctx.get("ctx_volume_24h_usd"), 0.0))
    vol_24_log = math.log1p(vol_24_usd / 1_000_000.0)  # ≈ 0..15
    rel_vol = _clip(_safe_float(ctx.get("ctx_relative_volume"), 1.0), 0.0, 5.0)
    liq_norm = _liquidity_tier_norm(ctx.get("ctx_liquidity_tier"))

    # Microstructure (2)
    spread = _clip(_safe_float(ctx.get("ctx_spread_pct"), 0.0), 0.0, 0.05)
    depth = _clip(_safe_float(ctx.get("ctx_depth_imbalance"), 0.0), -1.0, 1.0)

    # Cross-asset RS (3)
    rs_btc = _clip(_safe_float(ctx.get("ctx_rs_btc"), 0.0), -0.5, 0.5)
    rs_eth = _clip(_safe_float(ctx.get("ctx_rs_eth"), 0.0), -0.5, 0.5)
    btc_dom = _clip(_safe_float(ctx.get("ctx_btc_dominance_proxy"), 0.5), 0.0, 1.0)

    # Regime + sentiment (2)
    regime_signed = _regime_to_signed(ctx.get("ctx_market_regime"))
    sentiment = _clip(_safe_float(ctx.get("ctx_sentiment_fear_greed"), 0.0), -1.0, 1.0)

    out = [
        align_5m,
        align_15m,
        align_1h,
        align_4h,
        align_1d,
        slope_5m,
        slope_15m,
        slope_1h,
        slope_4h,
        slope_1d,
        chg_24,
        vol_24_log,
        rel_vol,
        liq_norm,
        spread,
        depth,
        rs_btc,
        rs_eth,
        btc_dom,
        regime_signed,
        sentiment,
    ]
    assert len(out) == len(CONTEXT_DIMS_V2), f"context_vector_from_ai_context produced {len(out)} dims, contract requires {len(CONTEXT_DIMS_V2)}"
    return out


def context_vector_day_full_mtf(
    ai_context: dict[str, Any] | None,
    *,
    mtf_snapshots: dict[str, dict[str, Any]],
    month_four: list[float],
) -> list[float]:
    """
    v5 DAY 21-D context: one native slope read per DAY_ACTIVE_TF, two month-from-daily
    scalars, mean ema_align, then condensed macro/micro/rs tail from Redis ai_context fields.
    """
    ctx = ai_context or {}
    front: list[float] = []
    emas: list[float] = []
    for tf in DAY_ACTIVE_TIMEFRAMES:
        snap = mtf_snapshots.get(tf) if isinstance(mtf_snapshots.get(tf), dict) else {}
        front.append(_slope_norm(snap.get("slope", 0.0)))
        emas.append(float(snap.get("ema_align", 0.5) or 0.5))
        emas[-1] = max(0.0, min(1.0, emas[-1]))
    mean_ema = sum(emas) / float(len(emas))

    mf = list(month_four) if isinstance(month_four, list) else []
    mon0 = _clip(_safe_float(mf[0] if len(mf) > 0 else 0.0, 0.0), -6.0, 6.0)
    mon1 = _clip(_safe_float(mf[1] if len(mf) > 1 else 0.0, 0.0), -6.0, 6.0)

    chg_24 = _clip(_safe_float(ctx.get("ctx_change_24h_pct"), 0.0), -0.5, 0.5)
    vol_24_usd = max(0.0, _safe_float(ctx.get("ctx_volume_24h_usd"), 0.0))
    vol_24_log = math.log1p(vol_24_usd / 1_000_000.0)
    rel_vol = _clip(_safe_float(ctx.get("ctx_relative_volume"), 1.0), 0.0, 5.0)
    spread = _clip(_safe_float(ctx.get("ctx_spread_pct"), 0.0), 0.0, 0.05)
    depth = _clip(_safe_float(ctx.get("ctx_depth_imbalance"), 0.0), -1.0, 1.0)
    rs_btc = _clip(_safe_float(ctx.get("ctx_rs_btc"), 0.0), -0.5, 0.5)
    rs_eth = _clip(_safe_float(ctx.get("ctx_rs_eth"), 0.0), -0.5, 0.5)
    rs_mean = _clip((rs_btc + rs_eth) / 2.0, -0.5, 0.5)
    btc_dom = _clip(_safe_float(ctx.get("ctx_btc_dominance_proxy"), 0.5), 0.0, 1.0)
    regime_signed = _regime_to_signed(ctx.get("ctx_market_regime"))
    sentiment = _clip(_safe_float(ctx.get("ctx_sentiment_fear_greed"), 0.0), -1.0, 1.0)
    regime_sent_blend = _clip((regime_signed + sentiment) / 2.0, -1.0, 1.0)

    out = front + [mon0, mon1, _clip(mean_ema, 0.0, 1.0)] + [chg_24, vol_24_log, rel_vol, spread, depth, rs_mean, btc_dom, regime_sent_blend]
    assert len(out) == len(CONTEXT_DIMS_DAY_FULL), f"context_vector_day_full_mtf produced {len(out)} dims != {CONTEXT_DIMS_DAY_FULL}"
    return out


def build_feature_vector_v2(
    *,
    symbol_ccxt: str,
    ohlcv: list[list],
    volume_profile: dict[str, Any] | None,
    orderbook: dict[str, Any] | None,
    ohlcv_1d: list[list] | None = None,
    sentiment: dict[str, Any] | None = None,
    ai_context: dict[str, Any] | None = None,
    ai_context_mtf: dict[str, Any] | None = None,
) -> list[float]:
    """
    Canonical AI feature vector v2 (145 dims).

    The first 124 dims are the legacy v1 vector (so all the live feature_builder
    code keeps producing them); the trailing 21 dims are the canonical context.
    """
    base = build_feature_vector_124(
        symbol_ccxt=symbol_ccxt,
        ohlcv=ohlcv,
        volume_profile=volume_profile,
        orderbook=orderbook,
        ohlcv_1d=ohlcv_1d,
        sentiment=sentiment,
    )
    if len(base) != AI_FEATURE_DIM_V1:
        raise ValueError(f"build_feature_vector_124 returned {len(base)} dims, contract requires {AI_FEATURE_DIM_V1}")
    ctx_part = context_vector_from_ai_context(ai_context, mtf=ai_context_mtf)
    out = list(base) + list(ctx_part)
    assert len(out) == AI_FEATURE_DIM_V2, f"build_feature_vector_v2 produced {len(out)} dims, contract requires {AI_FEATURE_DIM_V2}"
    return out


__all__ = [
    "build_feature_vector_v2",
    "context_vector_day_full_mtf",
    "context_vector_from_ai_context",
]
