"""Continuous setup measurements for all nine SCALP modules.

Opinion conditions become features. Mechanical/economic checks stay separate.
"""

from __future__ import annotations

from typing import Any

from backend.services.binance_scalp.strategies.base import StrategyMarketContext
from backend.services.binance_scalp.strategies.common import estimate_expected_move_pct


def _vwap(bars: list[dict]) -> float:
    num = den = 0.0
    for b in bars:
        tp = (float(b["high"]) + float(b["low"]) + float(b["close"])) / 3.0
        v = float(b.get("volume") or 0.0)
        num += tp * v
        den += v
    return num / den if den > 0 else float(bars[-1]["close"])


def _ema(values: list[float], period: int) -> float:
    if not values:
        return 0.0
    k = 2.0 / (period + 1)
    ema = values[0]
    for v in values[1:]:
        ema = v * k + ema * (1 - k)
    return ema


def _safe_div(n: float, d: float) -> float:
    return n / d if d else 0.0


def measure_all_setups(ctx: StrategyMarketContext) -> dict[str, dict[str, float]]:
    """Return per-strategy continuous features. Never raises. Never gates."""
    out: dict[str, dict[str, float]] = {}
    bars = list(ctx.bars_1m or [])
    snap = ctx.snap
    mom = ctx.mom
    cur = float(getattr(snap, "mid", 0.0) or 0.0)
    spread = float(getattr(snap, "spread_pct", 0.0) or 0.0)
    imb = float(getattr(snap, "order_book_imbalance", 0.0) or 0.0)
    mid15 = float(getattr(mom, "mid_change_15s", 0.0) or 0.0)
    mid30 = float(getattr(mom, "mid_change_30s", 0.0) or 0.0)
    mid60 = float(getattr(mom, "mid_change_60s", 0.0) or 0.0)
    bid15 = float(getattr(mom, "bid_change_15s", 0.0) or 0.0)
    bid30 = float(getattr(mom, "bid_change_30s", 0.0) or 0.0)
    bid60 = float(getattr(mom, "bid_change_60s", 0.0) or 0.0)
    vol_state = float(getattr(mom, "realized_volatility_pct", 0.0) or 0.0)
    common = {
        "mid": cur,
        "spread_pct": spread,
        "orderbook_imbalance": imb,
        "momentum_15s": mid15,
        "momentum_30s": mid30,
        "momentum_60s": mid60,
        "bid_change_15s": bid15,
        "bid_change_30s": bid30,
        "bid_change_60s": bid60,
        "volatility_state": vol_state,
        "momentum_acceleration": mid15 - mid60,
        "price_confirmation_strength": (1.0 if mid15 > 0 and bid15 > 0 else 0.0) + max(0.0, mid15) * 2000.0,
        "n_bars": float(len(bars)),
    }

    closes = [float(b["close"]) for b in bars] if bars else []
    highs = [float(b["high"]) for b in bars] if bars else []
    lows = [float(b["low"]) for b in bars] if bars else []
    vols = [float(b.get("volume") or 0.0) for b in bars] if bars else []

    # vwap_ema_reclaim
    if len(bars) >= 8 and cur > 0:
        vwap = _vwap(bars[-15:] if len(bars) >= 15 else bars)
        ema_fast = _ema(closes[-8:], 5)
        ema_slow = _ema(closes[-15:] if len(closes) >= 15 else closes, 13)
        prior_low = min(lows[-5:]) if lows else cur
        out["vwap_ema_reclaim"] = {
            **common,
            "vwap_distance": _safe_div(cur - vwap, vwap),
            "reclaim_strength": _safe_div(cur - vwap, vwap) * 1000.0,
            "ema_relationship": _safe_div(ema_fast - ema_slow, ema_slow),
            "pullback_depth": _safe_div(vwap - prior_low, cur),
            "pullback_recovery_strength": (1.0 if bars[-1]["low"] > prior_low else 0.0) + max(0.0, mid30) * 1500.0,
            "projected_move": estimate_expected_move_pct(bars, structural=max(_safe_div(vwap - prior_low, cur), 0.0012), atr_mult=0.70, cap_pct=0.006),
        }
    else:
        out["vwap_ema_reclaim"] = {**common}

    # range_bounce
    if len(bars) >= 10 and cur > 0:
        window = bars[-15:] if len(bars) >= 15 else bars
        support = min(float(b["low"]) for b in window)
        hi = max(float(b["high"]) for b in window)
        last = bars[-1]
        br = float(last["high"]) - float(last["low"])
        wick = (min(float(last.get("open", last["close"])), float(last["close"])) - float(last["low"])) / br if br > 0 else 0.0
        out["range_bounce_scalp"] = {
            **common,
            "support_distance": _safe_div(cur - support, cur),
            "range_width": _safe_div(hi - support, cur),
            "reversal_strength": wick,
            "momentum_flip_strength": max(0.0, bid15) + max(0.0, mid15) + max(0.0, mid30),
            "projected_move": estimate_expected_move_pct(bars, structural=max(_safe_div(hi - cur, cur), 0.0008), atr_mult=0.65, cap_pct=0.006),
        }
    else:
        out["range_bounce_scalp"] = {**common}

    # breakout_momentum
    if len(bars) >= 8 and cur > 0:
        recent = bars[-6:-1]
        brk = max(float(b["high"]) for b in recent)
        rng = _safe_div(max(float(b["high"]) for b in recent) - min(float(b["low"]) for b in recent), cur)
        vr = sum(vols[-3:]) / (sum(vols[-6:-3]) or 1.0)
        out["breakout_momentum"] = {
            **common,
            "breakout_distance": _safe_div(cur - brk, brk),
            "breakout_strength": max(0.0, _safe_div(cur - brk, brk)) * 1000.0 + vr,
            "expansion_score": vr,
            "projected_move": estimate_expected_move_pct(bars, structural=min(rng * 0.55, 0.005), atr_mult=0.65, cap_pct=0.006),
        }
    else:
        out["breakout_momentum"] = {**common}

    # orderbook_tape
    bids = list(getattr(snap, "bids", None) or [])
    asks = list(getattr(snap, "asks", None) or [])
    bid_qty = sum(float(q) for _, q in bids[:5])
    ask_qty = sum(float(q) for _, q in asks[:5])
    out["orderbook_tape_scalp"] = {
        **common,
        "orderbook_imbalance": imb,
        "price_confirmation_strength": common["price_confirmation_strength"],
        "ask_bid_qty_ratio": _safe_div(ask_qty, bid_qty or 1.0),
        "projected_move": min(spread * 6 + imb * 0.002, 0.004),
    }

    # failed_breakdown
    if len(bars) >= 12 and cur > 0:
        sweep = min(float(b["low"]) for b in bars[-8:-1])
        vr = sum(vols[-3:]) / (sum(vols[-6:-3]) or 1.0)
        out["failed_breakdown_reversal"] = {
            **common,
            "reclaim_strength": _safe_div(cur - sweep, sweep),
            "volume_impulse_strength": vr,
            "reversal_strength": (1.0 if cur > sweep * 1.0008 else 0.0) + max(0.0, mid15) * 2000.0,
            "projected_move": estimate_expected_move_pct(bars, structural=0.0028, atr_mult=0.65, cap_pct=0.006),
        }
    else:
        out["failed_breakdown_reversal"] = {**common}

    # compression
    if len(bars) >= 15 and cur > 0:
        recent = bars[-10:]
        rng = _safe_div(max(float(b["high"]) for b in recent) - min(float(b["low"]) for b in recent), cur)
        vr = sum(vols[-3:]) / (sum(vols[-8:-3]) or 1.0)
        out["compression_breakout"] = {
            **common,
            "compression_score": max(0.0, 0.004 - rng) * 250.0,
            "expansion_score": vr,
            "range_width": rng,
            "projected_move": estimate_expected_move_pct(bars, structural=0.0030, atr_mult=0.70, cap_pct=0.006),
        }
    else:
        out["compression_breakout"] = {**common}

    # volume impulse
    if len(bars) >= 10 and cur > 0:
        vr = sum(vols[-2:]) / (sum(vols[-6:-2]) or 1.0)
        out["volume_impulse_continuation"] = {
            **common,
            "volume_impulse_strength": vr,
            "trend_alignment_strength": 1.0 if closes and cur > closes[-2] else 0.0,
            "projected_move": estimate_expected_move_pct(bars, structural=0.0026, atr_mult=0.70, cap_pct=0.006),
        }
    else:
        out["volume_impulse_continuation"] = {**common}

    # trend pullback
    if len(bars) >= 12 and cur > 0:
        ema5 = sum(closes[-5:]) / 5 if len(closes) >= 5 else cur
        out["trend_pullback_micro"] = {
            **common,
            "pullback_depth": _safe_div(ema5 - cur, ema5),
            "ema_relationship": _safe_div(cur - ema5, ema5),
            "trend_alignment_strength": 1.0 if mid15 > -0.00005 else 0.0,
            "projected_move": estimate_expected_move_pct(bars, structural=0.0025, atr_mult=0.60, cap_pct=0.006),
        }
    else:
        out["trend_pullback_micro"] = {**common}

    # failed breakout
    if len(bars) >= 10 and cur > 0:
        high = max(float(b["high"]) for b in bars[-6:-1])
        probed = max(float(b["high"]) for b in bars[-4:]) >= high * 0.9998
        out["failed_breakout_reversal"] = {
            **common,
            "breakout_distance": _safe_div(high - cur, high),
            "reversal_strength": (1.0 if probed and cur < high * 0.9995 else 0.0) + max(0.0, mid15) * 2000.0,
            "reclaim_strength": max(0.0, mid15) + max(0.0, bid15),
            "projected_move": estimate_expected_move_pct(bars, structural=0.0022, atr_mult=0.60, cap_pct=0.006),
        }
    else:
        out["failed_breakout_reversal"] = {**common}

    return out


def evidence_rank_delta(measurements: dict[str, dict[str, float]]) -> float:
    """Reserved for a calibrated measurement model.

    Ocean 1m replay (n=67761): naive reclaim/flip strength vs +5m net
    correlation was 0.0018 and the high tercile did not outperform the low
    tercile. Do not apply an uncalibrated linear combo to rank_score.
    Measurements are still stored for learning. Outcome-consume remains the
    live rank/size update path.
    """
    _ = measurements
    return 0.0


__all__ = ["evidence_rank_delta", "measure_all_setups"]
