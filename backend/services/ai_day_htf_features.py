"""
DAY trading model input — **124 dims** from real **native 1m** OHLCV (named FEATURE_MAPPING
indicators) + **21 dims** CONTEXT_DIMS_DAY_FULL (slopes x every DAY_ACTIVE_TF, month-from-daily
scalars, macro tail from Redis ai_context).

This replaces the retired v4 “stack four 31-block HTFs into 124” layout.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from backend.config.day_active_timeframes import DAY_ACTIVE_TIMEFRAMES
from backend.services.ai_decision_contract import AI_FEATURE_DIM_V1, AI_FEATURE_DIM_V2
from backend.services.ai_feature_v2 import context_vector_day_full_mtf
from backend.services.ai_market_context import _summarize_tf
from backend.services.feature_builder import build_feature_vector_124


def _safe(x: float, default: float = 0.0) -> float:
    try:
        v = float(x)
        return v if math.isfinite(v) else default
    except (TypeError, ValueError):
        return default


def _ohlcv_to_arrays(ohlcv: list[list]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    a = np.asarray(ohlcv, dtype=np.float64)
    if a.ndim != 2 or a.shape[1] < 6 or a.shape[0] < 2:
        return (
            np.array([], dtype=np.float64),
            np.array([], dtype=np.float64),
            np.array([], dtype=np.float64),
            np.array([], dtype=np.float64),
            np.array([], dtype=np.float64),
        )
    op, hi, lo, cl, vo = a[:, 1], a[:, 2], a[:, 3], a[:, 4], a[:, 5]
    return op, hi, lo, cl, vo


DAY_HTF_BLOCK_DIM = 31  # compat for compact_htf_block_31 callers (spike heuristic, dashboards)


def compact_htf_block_31(ohlcv: list[list]) -> list[float]:
    """Fixed 31-dim OHLCV summary (used by spike-fade heuristic and diagnostics)."""
    out = [0.0] * DAY_HTF_BLOCK_DIM
    op, hi, lo, cl, vo = _ohlcv_to_arrays(ohlcv)
    n = int(cl.size)
    if n < 5:
        return out

    c = cl
    last = _safe(c[-1], 0.0)
    if last <= 0:
        return out

    w = min(120, n)
    out[0] = _safe(last / (float(np.mean(c[-w:])) + 1e-12) - 1.0, 0.0)
    out[1] = _safe(c[-1] / c[-2] - 1.0, 0.0) if n >= 2 else 0.0
    out[2] = _safe(c[-1] / c[-6] - 1.0, 0.0) if n >= 6 else out[1]
    out[3] = _safe(c[-1] / c[-21] - 1.0, 0.0) if n >= 21 else out[2]
    out[4] = _safe((float(hi[-1]) - float(lo[-1])) / last, 0.0)
    vw = min(30, n)
    vm = float(np.mean(vo[-vw:])) + 1e-9
    out[5] = _safe(float(vo[-1]) / vm - 1.0, 0.0)

    d = np.diff(c[-15:])
    if d.size >= 14:
        gains = np.maximum(d, 0.0)
        losses = np.maximum(-d, 0.0)
        ag = float(np.mean(gains[-14:]))
        al = float(np.mean(losses[-14:])) + 1e-15
        rs = ag / al
        out[6] = _safe(100.0 - (100.0 / (1.0 + rs)), 50.0) / 100.0 - 0.5
    else:
        out[6] = 0.0

    def _ema(arr: np.ndarray, span: int) -> float:
        if arr.size < span:
            return float(arr[-1])
        alpha = 2.0 / (span + 1)
        e = float(arr[-span])
        for px in arr[-span + 1 :]:
            e = alpha * float(px) + (1 - alpha) * e
        return e

    e12 = _ema(c, min(12, n))
    e26 = _ema(c, min(26, n))
    out[7] = _safe((e12 - e26) / last, 0.0)

    look = min(50, n)
    hh = float(np.max(hi[-look:]))
    ll = float(np.min(lo[-look:]))
    out[8] = _safe((last - ll) / (hh - ll + 1e-12), 0.5) - 0.5

    m10 = float(np.mean(c[-min(10, n) :]))
    m30 = float(np.mean(c[-min(30, n) :]))
    out[9] = _safe((m10 - m30) / last, 0.0)

    lr = np.diff(np.log(c[-min(21, n) :] + 1e-12))
    if lr.size >= 3:
        out[10] = _safe(float(np.std(lr)), 0.0)
        out[11] = _safe(float(np.mean(((lr - float(np.mean(lr))) / (float(np.std(lr)) + 1e-9)) ** 3)), 0.0)
    else:
        out[10] = out[11] = 0.0

    seg = c[-min(30, n) :]
    peak = np.maximum.accumulate(seg)
    dd = float(np.min((seg - peak) / (peak + 1e-12)))
    out[12] = _safe(float(dd), 0.0)

    rng = float(hi[-1] - lo[-1]) + 1e-12
    abs(float(c[-1] - op[-1]))
    out[13] = _safe((float(hi[-1]) - max(float(c[-1]), float(op[-1]))) / rng, 0.0)
    out[14] = _safe((min(float(c[-1]), float(op[-1])) - float(lo[-1])) / rng, 0.0)

    up = 0
    for j in range(1, min(11, n)):
        if c[-j] > c[-j - 1]:
            up += 1
    out[15] = _safe(up / 10.0 - 0.5, 0.0)

    if n >= 15:
        trs: list[float] = []
        for i in range(-14, 0):
            h_, l_, pc = float(hi[i]), float(lo[i]), float(c[i - 1])
            trs.append(max(h_ - l_, abs(h_ - pc), abs(l_ - pc)))
        out[16] = _safe(float(np.mean(trs)) / last, 0.0)
    else:
        out[16] = 0.0

    if n >= 21:
        m = float(np.mean(c[-20:]))
        sd = float(np.std(c[-20:])) + 1e-12
        out[17] = _safe(2.0 * sd / m, 0.0)
    else:
        out[17] = 0.0

    if n >= 26:
        out[18] = _safe((last - float(np.mean(c[-26:]))) / last, 0.0)
    else:
        out[18] = 0.0

    for k, lag in enumerate((2, 3, 4, 5, 7, 11, 13, 17, 19, 21, 23, 25)):
        idx = 19 + k
        if idx >= DAY_HTF_BLOCK_DIM:
            break
        if n > lag:
            out[idx] = _safe(c[-1] / c[-1 - lag] - 1.0, 0.0)

    for i in range(DAY_HTF_BLOCK_DIM):
        out[i] = float(np.clip(out[i], -6.0, 6.0))
    return out


def build_day_htf_feature_vector_145(
    *,
    symbol_ccxt: str,
    day_bundle: dict[str, Any],
    volume_profile: dict[str, Any] | None,
    orderbook: dict[str, Any] | None,
    sentiment: dict[str, Any] | None,
    ai_context: dict[str, Any] | None,
    tech_provenance: dict[str, dict[str, Any]] | None = None,
    orderbook_age_sec: float | None = None,
) -> list[float]:
    """
    145 dims — v5 day: FEATURE_MAPPING technicals on native **1m** + CONTEXT_DIMS_DAY_FULL (21).

    ``day_bundle`` must have passed ``validate_day_active_bundle`` including ``_month_vec``.
    """
    rows_1m = day_bundle.get("1m") or []
    rows_1d = day_bundle.get("1d")
    ohlcv_1d = rows_1d if isinstance(rows_1d, list) and len(rows_1d) >= 2 else None
    tech124 = build_feature_vector_124(
        symbol_ccxt=symbol_ccxt,
        ohlcv=rows_1m if isinstance(rows_1m, list) else [],
        volume_profile=volume_profile,
        orderbook=orderbook,
        ohlcv_1d=ohlcv_1d,
        sentiment=sentiment,
        provenance=tech_provenance,
        orderbook_age_sec=orderbook_age_sec,
    )
    if len(tech124) != AI_FEATURE_DIM_V1:
        raise ValueError(f"day tech block expected {AI_FEATURE_DIM_V1}, got {len(tech124)}")

    snaps: dict[str, dict[str, Any]] = {}
    for tf in DAY_ACTIVE_TIMEFRAMES:
        snaps[tf] = dict(_summarize_tf(day_bundle.get(tf) if isinstance(day_bundle.get(tf), list) else None))

    month_vec = day_bundle.get("_month_vec")
    if not isinstance(month_vec, list) or len(month_vec) < 2:
        raise ValueError("build_day_htf_feature_vector_145 requires validated day_bundle['_month_vec']")

    ctx21 = context_vector_day_full_mtf(
        ai_context,
        mtf_snapshots=snaps,
        month_four=[float(x) for x in month_vec],
    )

    out = list(tech124) + ctx21
    if len(out) != AI_FEATURE_DIM_V2:
        raise ValueError(f"day+v5 ctx expected {AI_FEATURE_DIM_V2}, got {len(out)}")
    for i, v in enumerate(out):
        fv = float(v)
        out[i] = 0.0 if not math.isfinite(fv) else fv
    return out


def day_htf_layout_legend() -> dict[str, tuple[int, int]]:
    """Index ranges for telemetry (v5)."""
    return {
        "technical_124_named_primary_1m": (0, 124),
        "context_day_full_mtf_v5": (124, AI_FEATURE_DIM_V2),
    }


DAY_HTF_CCXT_ORDER = DAY_ACTIVE_TIMEFRAMES


__all__ = [
    "DAY_HTF_BLOCK_DIM",
    "DAY_HTF_CCXT_ORDER",
    "build_day_htf_feature_vector_145",
    "compact_htf_block_31",
    "day_htf_layout_legend",
]
