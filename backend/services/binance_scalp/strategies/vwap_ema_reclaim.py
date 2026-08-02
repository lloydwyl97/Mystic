"""VWAP/EMA reclaim scalp — pullback ends, price reclaims trend."""

from __future__ import annotations

from backend.services.binance_scalp.strategies.base import ScalpSetupSignal, StrategyMarketContext
from backend.services.binance_scalp.strategies.common import (
    check_spread,
    depth_check,
    estimate_expected_move_pct,
    pass_signal,
    reject_signal,
    target_reachable,
)


def _vwap(bars: list[dict]) -> float:
    num = den = 0.0
    for b in bars:
        tp = (b["high"] + b["low"] + b["close"]) / 3.0
        v = b.get("volume", 0.0)
        num += tp * v
        den += v
    return num / den if den > 0 else bars[-1]["close"]


def _ema(values: list[float], period: int) -> float:
    if not values:
        return 0.0
    k = 2.0 / (period + 1)
    ema = values[0]
    for v in values[1:]:
        ema = v * k + ema * (1 - k)
    return ema


class VwapEmaReclaimStrategy:
    name = "vwap_ema_reclaim"

    def evaluate(self, ctx: StrategyMarketContext) -> ScalpSetupSignal:
        ok, reason = check_spread(ctx.snap, ctx.econ, ctx.config)
        if not ok:
            return reject_signal(ctx, self.name, reason or "SPREAD_TOO_WIDE")

        depth_ok, impact, fill = depth_check(ctx.snap, ctx.notional_usd, ctx.econ)
        if not depth_ok:
            return reject_signal(ctx, self.name, "DEPTH_OR_IMPACT_FAIL", impact=impact)

        bars = ctx.bars_1m
        if len(bars) < 15:
            return reject_signal(ctx, self.name, "INSUFFICIENT_BARS")

        closes = [b["close"] for b in bars]
        vwap = _vwap(bars[-15:])
        ema_fast = _ema(closes[-8:], 5)
        ema_slow = _ema(closes[-15:], 13)
        cur = ctx.snap.mid
        prior_low = min(b["low"] for b in bars[-5:])
        higher_low = bars[-1]["low"] > prior_low

        # Slightly tighter reclaim bands — paper soft reclaim was producing
        # max-hold losers without a real reclaim impulse.
        reclaimed_vwap = cur >= vwap * 0.9999
        ema_reclaim = ema_fast >= ema_slow * 0.9997
        if not (reclaimed_vwap and ema_reclaim):
            return reject_signal(ctx, self.name, "NO_VWAP_EMA_RECLAIM")

        mom = ctx.mom
        # Require higher_low + 60s mid lift in paper too (was paper-loosened).
        mom_ok = (
            mom.bid_change_15s > 0
            and mom.mid_change_15s > 0
            and mom.mid_change_30s > 0
            and mom.mid_change_60s > 0
            and higher_low
        )
        if not ctx.config.scalp_paper_enabled:
            mom_ok = mom_ok and mom.momentum_confirmed
        if not mom_ok:
            return reject_signal(ctx, self.name, "NO_PULLBACK_RECOVERY")

        structural = (vwap - prior_low) / cur if cur > 0 else 0.001
        structural = max(structural, 0.0012)
        expected = estimate_expected_move_pct(bars, structural=structural, atr_mult=0.70, cap_pct=0.006)
        reachable, _ = target_reachable(ctx.econ, spread_pct=ctx.snap.spread_pct, impact_pct=impact, expected_move_pct=expected)
        if not reachable:
            return reject_signal(ctx, self.name, "TARGET_NOT_REACHABLE", expected_move=expected, impact=impact)

        score = 2.35 + (cur - vwap) / vwap * 450 + (ema_fast - ema_slow) / ema_slow * 280
        return pass_signal(
            ctx,
            self.name,
            score=score,
            confidence=0.62,
            entry_reason=f"vwap_reclaim vwap={vwap:.4f} ema_fast>{ema_slow:.4f}",
            invalidation_reason="lost_vwap_or_ema_with_no_recovery",
            expected_move_pct=expected,
            impact_pct=impact,
            limit_buy=fill,
            setup_context={"vwap": vwap, "ema_fast": ema_fast, "ema_slow": ema_slow, "prior_low": prior_low},
        )
