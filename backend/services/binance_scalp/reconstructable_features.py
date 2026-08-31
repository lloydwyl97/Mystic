"""Honest bar-reconstructable features. No fabricated order book. No future path."""

from __future__ import annotations

import math
from datetime import datetime
from typing import Any

FEATURE_KEYS: tuple[str, ...] = (
    # existing-style
    "ret_1",
    "ret_3",
    "ret_5",
    "ret_10",
    "ret_20",
    "momentum_slope",
    "momentum_accel",
    "realized_vol_10",
    "vol_expansion",
    "dist_high_20",
    "dist_low_20",
    "range_position",
    "pullback_depth",
    "pullback_recovery",
    "trend_persist",
    "rel_volume",
    "volume_accel",
    "vwap_dist",
    "projected_move",
    "compression_score",
    "reclaim_strength",
    "btc_ret_5",
    "rel_vs_btc_5",
    "market_vol_5",
    "hour_sin",
    "hour_cos",
    "evt_mom_expansion",
    "evt_vol_accel",
    "evt_vwap_recovery",
    "evt_compression_release",
    "evt_failed_move",
    "evt_pullback_reclaim",
)

FEATURE_GROUPS: dict[str, tuple[str, ...]] = {
    "price_dynamics": (
        "ret_1",
        "ret_3",
        "ret_5",
        "ret_10",
        "ret_20",
        "momentum_slope",
        "momentum_accel",
        "realized_vol_10",
        "vol_expansion",
        "dist_high_20",
        "dist_low_20",
        "range_position",
        "pullback_depth",
        "pullback_recovery",
        "trend_persist",
    ),
    "volume": ("rel_volume", "volume_accel"),
    "setup_proxy": ("projected_move", "compression_score", "reclaim_strength", "vwap_dist"),
    "cross_asset": ("btc_ret_5", "rel_vs_btc_5", "market_vol_5"),
    "time_structure": ("hour_sin", "hour_cos"),
    "event_strength": (
        "evt_mom_expansion",
        "evt_vol_accel",
        "evt_vwap_recovery",
        "evt_compression_release",
        "evt_failed_move",
        "evt_pullback_reclaim",
    ),
}

LIVE_ONLY = (
    "orderbook_imbalance",
    "imbalance_change",
    "depth_asymmetry",
    "microprice_vs_mid",
    "book_velocity",
)


def _safe(n: float, d: float) -> float:
    return n / d if d else 0.0


def _ret(closes: list[float], n: int) -> float:
    if len(closes) <= n or closes[-1 - n] <= 0:
        return 0.0
    return (closes[-1] - closes[-1 - n]) / closes[-1 - n]


def reconstructable_features(
    bars: list[dict[str, Any]],
    *,
    btc_ret_5: float = 0.0,
    market_vol_5: float = 0.0,
    ts: datetime | None = None,
    projected_move: float = 0.0,
) -> dict[str, float]:
    """Features from bars known at the current close. No future bars."""
    out = dict.fromkeys(FEATURE_KEYS, 0.0)
    if len(bars) < 8:
        return out
    closes = [float(b["close"]) for b in bars]
    highs = [float(b["high"]) for b in bars]
    lows = [float(b["low"]) for b in bars]
    vols = [float(b.get("volume") or 0.0) for b in bars]
    cur = closes[-1]
    if cur <= 0:
        return out
    out["ret_1"] = _ret(closes, 1)
    out["ret_3"] = _ret(closes, 3)
    out["ret_5"] = _ret(closes, 5)
    out["ret_10"] = _ret(closes, 10)
    out["ret_20"] = _ret(closes, 20)
    out["momentum_slope"] = out["ret_5"] - out["ret_10"]
    out["momentum_accel"] = out["ret_1"] - out["ret_5"]
    # realized vol of last 10 1-bar returns
    last10 = []
    for i in range(1, min(11, len(closes))):
        if closes[-1 - i] > 0:
            last10.append((closes[-i] - closes[-1 - i]) / closes[-1 - i])
    if last10:
        mean = sum(last10) / len(last10)
        out["realized_vol_10"] = math.sqrt(sum((x - mean) ** 2 for x in last10) / len(last10))
    vol5 = out["realized_vol_10"]
    prior = []
    for i in range(11, min(21, len(closes))):
        if closes[-1 - i] > 0:
            prior.append((closes[-i] - closes[-1 - i]) / closes[-1 - i])
    prior_vol = 0.0
    if prior:
        m = sum(prior) / len(prior)
        prior_vol = math.sqrt(sum((x - m) ** 2 for x in prior) / len(prior))
    out["vol_expansion"] = vol5 - prior_vol
    hi20 = max(highs[-20:]) if len(highs) >= 20 else max(highs)
    lo20 = min(lows[-20:]) if len(lows) >= 20 else min(lows)
    out["dist_high_20"] = _safe(hi20 - cur, cur)
    out["dist_low_20"] = _safe(cur - lo20, cur)
    out["range_position"] = _safe(cur - lo20, hi20 - lo20)
    ema5 = sum(closes[-5:]) / 5
    out["pullback_depth"] = _safe(ema5 - cur, ema5)
    out["pullback_recovery"] = max(0.0, out["ret_3"]) - max(0.0, out["pullback_depth"])
    ups = sum(1 for i in range(1, min(8, len(closes))) if closes[-i] > closes[-i - 1])
    out["trend_persist"] = ups / 7.0
    avg_vol = sum(vols[-20:]) / max(1, min(20, len(vols)))
    out["rel_volume"] = _safe(vols[-1], avg_vol)
    out["volume_accel"] = _safe(sum(vols[-3:]), sum(vols[-8:-3]) or 1.0)
    tp_v = 0.0
    vsum = 0.0
    for b in bars[-15:]:
        tp = (float(b["high"]) + float(b["low"]) + float(b["close"])) / 3.0
        v = float(b.get("volume") or 0.0)
        tp_v += tp * v
        vsum += v
    vwap = tp_v / vsum if vsum else cur
    out["vwap_dist"] = _safe(cur - vwap, vwap)
    out["projected_move"] = float(projected_move or 0.0)
    rng = _safe(max(highs[-10:]) - min(lows[-10:]), cur)
    out["compression_score"] = max(0.0, 0.004 - rng) * 250.0
    out["reclaim_strength"] = out["vwap_dist"] * 1000.0
    out["btc_ret_5"] = float(btc_ret_5)
    out["rel_vs_btc_5"] = out["ret_5"] - float(btc_ret_5)
    out["market_vol_5"] = float(market_vol_5)
    if ts is not None:
        hour = ts.hour + ts.minute / 60.0
        out["hour_sin"] = math.sin(2 * math.pi * hour / 24.0)
        out["hour_cos"] = math.cos(2 * math.pi * hour / 24.0)
    # continuous event strengths — not gates
    out["evt_mom_expansion"] = max(0.0, out["ret_1"]) * max(0.0, out["vol_expansion"]) * 1000.0
    out["evt_vol_accel"] = max(0.0, out["volume_accel"] - 1.0)
    out["evt_vwap_recovery"] = max(0.0, out["vwap_dist"]) * max(0.0, out["ret_3"]) * 1000.0
    out["evt_compression_release"] = out["compression_score"] * max(0.0, out["ret_1"]) * 100.0
    sweep = min(lows[-8:-1]) if len(lows) >= 8 else lo20
    out["evt_failed_move"] = max(0.0, _safe(cur - sweep, sweep)) * max(0.0, out["ret_1"]) * 1000.0
    out["evt_pullback_reclaim"] = max(0.0, out["pullback_depth"]) * max(0.0, out["ret_3"]) * 1000.0
    for k, v in list(out.items()):
        if not math.isfinite(v):
            out[k] = 0.0
    return out
