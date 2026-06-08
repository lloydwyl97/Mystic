"""Phase 3e — momentum-based gross move projection for scalp entry."""

from __future__ import annotations

import os
from dataclasses import dataclass

from backend.services.binance_scalp.economics import ScalpEconomics
from backend.services.binance_scalp.market_reader import MarketSnapshot
from backend.services.binance_scalp.momentum_tracker import MomentumDiagnostics


def _float_env(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None or not str(raw).strip():
        return default
    return float(raw)


@dataclass(frozen=True)
class MomentumGrossEstimate:
    momentum_gross_estimate_pct: float
    recent_range_pct: float
    realized_volatility_pct: float
    trend_slope_30s: float
    trend_slope_60s: float
    trend_slope_15s: float
    breakout_strength_pct: float
    projected_gross_move_pct: float
    imbalance_boost_pct: float
    imbalance_raw: float
    data_sufficient: bool
    breakout_confirmed: bool
    range_sufficient: bool

    def as_dict(self) -> dict:
        return {
            "momentum_gross_estimate_pct": self.momentum_gross_estimate_pct,
            "recent_range_pct": self.recent_range_pct,
            "realized_volatility_pct": self.realized_volatility_pct,
            "trend_slope_30s": self.trend_slope_30s,
            "trend_slope_60s": self.trend_slope_60s,
            "trend_slope_15s": self.trend_slope_15s,
            "breakout_strength_pct": self.breakout_strength_pct,
            "projected_gross_move_pct": self.projected_gross_move_pct,
            "imbalance_boost_pct": self.imbalance_boost_pct,
            "imbalance_raw": self.imbalance_raw,
            "data_sufficient": self.data_sufficient,
            "breakout_confirmed": self.breakout_confirmed,
            "range_sufficient": self.range_sufficient,
        }


def _positive(*values: float) -> float:
    return max(0.0, *[float(v) for v in values])


def compute_from_tick_changes(
    *,
    mid: float,
    bid_change_15s: float,
    bid_change_30s: float,
    bid_change_60s: float,
    mid_change_15s: float,
    mid_change_30s: float,
    mid_change_60s: float,
    recent_range_pct: float,
    realized_volatility_pct: float,
    up_tick_count: int,
    history_sec: float,
    sample_count: int,
    imbalance: float,
    spread_pct: float,
) -> MomentumGrossEstimate:
    min_history = _float_env("SCALP_MOMENTUM_MIN_HISTORY_SEC", 30.0)
    min_range = _float_env("SCALP_MIN_RECENT_RANGE_PCT", 0.00008)

    data_sufficient = history_sec >= min_history and sample_count >= 4 and mid > 0
    range_sufficient = recent_range_pct >= min_range

    trend_15 = _positive(bid_change_15s, mid_change_15s)
    trend_30 = _positive(bid_change_30s, mid_change_30s)
    trend_60 = _positive(bid_change_60s, mid_change_60s)

    breakout = _positive(recent_range_pct * 0.5, trend_30 * 1.2, realized_volatility_pct * 0.6)
    if trend_30 <= 0:
        breakout = 0.0

    min_up = int(os.getenv("SCALP_MOMENTUM_MIN_UP_TICKS", "2"))
    breakout_confirmed = (
        trend_30 > 0
        and trend_60 > 0
        and up_tick_count >= min_up
        and breakout >= trend_30 * 0.20
    )

    momentum_gross = (
        0.25 * trend_30
        + 0.20 * trend_60
        + 0.15 * trend_15
        + 0.15 * breakout
        + 0.10 * _positive(realized_volatility_pct)
        + 0.15 * _positive(recent_range_pct)
    )
    if trend_60 > trend_30 > 0:
        momentum_gross += 0.10 * (trend_60 - trend_30)

    imbalance_boost = 0.0
    imb = max(0.0, float(imbalance))
    if trend_30 > 0 and imb > 0:
        cap = _float_env("SCALP_IMBALANCE_BOOST_CAP_PCT", 0.0004)
        imbalance_boost = min(cap, imb * spread_pct * 2.0)

    projected = momentum_gross + imbalance_boost
    if trend_30 > 0 and trend_60 > 0:
        carry = recent_range_pct * 0.65 + trend_30 * 0.50 + trend_60 * 0.25
        projected = max(projected, carry)

    return MomentumGrossEstimate(
        momentum_gross_estimate_pct=momentum_gross,
        recent_range_pct=recent_range_pct,
        realized_volatility_pct=realized_volatility_pct,
        trend_slope_30s=trend_30,
        trend_slope_60s=trend_60,
        trend_slope_15s=trend_15,
        breakout_strength_pct=breakout,
        projected_gross_move_pct=projected,
        imbalance_boost_pct=imbalance_boost,
        imbalance_raw=imb,
        data_sufficient=data_sufficient,
        breakout_confirmed=breakout_confirmed,
        range_sufficient=range_sufficient,
    )


def _range_and_vol_from_tracker(
    momentum: MomentumDiagnostics | None,
) -> tuple[float, float, float, float]:
    if momentum is None:
        return 0.0, 0.0, 0.0, 0.0
    mid60 = getattr(momentum, "mid_change_60s", 0.0)
    bid60 = getattr(momentum, "bid_change_60s", 0.0)
    recent_range = getattr(momentum, "recent_range_pct", 0.0)
    realized_vol = getattr(momentum, "realized_volatility_pct", 0.0)
    return mid60, bid60, recent_range, realized_vol


def compute_momentum_gross_estimate(
    snap: MarketSnapshot,
    momentum: MomentumDiagnostics | None,
    econ: ScalpEconomics | None = None,
) -> MomentumGrossEstimate:
    """Build projected gross move from live tick momentum (+ optional imbalance boost)."""
    _ = econ
    imb = snap.order_book_imbalance if snap.order_book_imbalance is not None else 0.0
    spread = snap.spread_pct
    mid = snap.mid

    if momentum is None:
        return compute_from_tick_changes(
            mid=mid,
            bid_change_15s=0.0,
            bid_change_30s=0.0,
            bid_change_60s=0.0,
            mid_change_15s=0.0,
            mid_change_30s=0.0,
            mid_change_60s=0.0,
            recent_range_pct=0.0,
            realized_volatility_pct=0.0,
            up_tick_count=0,
            history_sec=0.0,
            sample_count=0,
            imbalance=imb,
            spread_pct=spread,
        )

    mid60, bid60, recent_range, realized_vol = _range_and_vol_from_tracker(momentum)
    return compute_from_tick_changes(
        mid=mid,
        bid_change_15s=momentum.bid_change_15s,
        bid_change_30s=momentum.bid_change_30s,
        bid_change_60s=bid60,
        mid_change_15s=momentum.mid_change_15s,
        mid_change_30s=momentum.mid_change_30s,
        mid_change_60s=mid60,
        recent_range_pct=recent_range,
        realized_volatility_pct=realized_vol,
        up_tick_count=momentum.last_n_ticks_up_count,
        history_sec=momentum.history_sec,
        sample_count=momentum.sample_count,
        imbalance=imb,
        spread_pct=spread,
    )


def compute_from_1m_bars(
    bars: list[dict],
    idx: int,
    *,
    imbalance: float = 0.0,
    spread_pct: float = 0.0004,
) -> MomentumGrossEstimate:
    """Audit helper — derive momentum projection from 1m OHLCV leading into bar idx."""
    if idx < 6 or idx >= len(bars):
        return compute_from_tick_changes(
            mid=0.0,
            bid_change_15s=0.0,
            bid_change_30s=0.0,
            bid_change_60s=0.0,
            mid_change_15s=0.0,
            mid_change_30s=0.0,
            mid_change_60s=0.0,
            recent_range_pct=0.0,
            realized_volatility_pct=0.0,
            up_tick_count=0,
            history_sec=0.0,
            sample_count=0,
            imbalance=imbalance,
            spread_pct=spread_pct,
        )

    def chg(i0: int, i1: int) -> float:
        a, b = float(bars[i0]["close"]), float(bars[i1]["close"])
        return (b - a) / a if a > 0 else 0.0

    b0 = bars[idx]
    mid = float(b0["close"])
    mid15 = chg(idx - 1, idx)
    mid30 = chg(idx - 2, idx) / 2.0
    mid60 = chg(idx - 4, idx) / 4.0
    window = bars[idx - 5 : idx + 1]
    highs = [float(x["high"]) for x in window]
    lows = [float(x["low"]) for x in window]
    recent_range = (max(highs) - min(lows)) / mid if mid > 0 else 0.0
    rets = []
    for j in range(idx - 5, idx):
        p0, p1 = float(bars[j]["close"]), float(bars[j + 1]["close"])
        if p0 > 0:
            rets.append((p1 - p0) / p0)
    realized_vol = (
        (sum(r * r for r in rets) / len(rets)) ** 0.5 if rets else 0.0
    )
    up = sum(
        1
        for j in range(idx - 5, idx)
        if float(bars[j + 1]["close"]) > float(bars[j]["close"])
    )
    return compute_from_tick_changes(
        mid=mid,
        bid_change_15s=mid15,
        bid_change_30s=mid30,
        bid_change_60s=mid60,
        mid_change_15s=mid15,
        mid_change_30s=mid30,
        mid_change_60s=mid60,
        recent_range_pct=recent_range,
        realized_volatility_pct=realized_vol,
        up_tick_count=up,
        history_sec=300.0,
        sample_count=6,
        imbalance=imbalance,
        spread_pct=spread_pct,
    )
