"""Trend pullback micro — 1h trend_up context + 1m pullback into EMA/VWAP then resume.

Native for bull_trend / pump_continuation.
"""

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


class TrendPullbackMicroStrategy:
    name = "trend_pullback_micro"

    def evaluate(self, ctx: StrategyMarketContext) -> ScalpSetupSignal:
        ok, reason = check_spread(ctx.snap, ctx.econ, ctx.config)
        if not ok:
            return reject_signal(ctx, self.name, reason or "SPREAD_TOO_WIDE")

        depth_ok, impact, fill = depth_check(ctx.snap, ctx.notional_usd, ctx.econ)
        if not depth_ok:
            return reject_signal(ctx, self.name, "DEPTH_OR_IMPACT_FAIL", impact=impact)

        bars = ctx.bars_1m or []
        if len(bars) < 12:
            return reject_signal(ctx, self.name, "INSUFFICIENT_BARS")

        # simple ema pullback proxy on recent 1m
        closes = [b["close"] for b in bars[-8:]]
        ema5 = sum(closes[-5:]) / 5 if len(closes) >= 5 else closes[-1]
        pullback = ctx.snap.mid < ema5 * 1.0005 and ctx.snap.mid > ema5 * 0.998

        mom = ctx.mom
        resume = mom.mid_change_15s > -0.00005  # not strongly down

        if not (pullback and resume):
            return reject_signal(ctx, self.name, "NO_MICRO_PULLBACK")

        expected = estimate_expected_move_pct(bars, structural=0.0025, atr_mult=0.60, cap_pct=0.006)
        reachable, _ = target_reachable(
            ctx.econ,
            spread_pct=ctx.snap.spread_pct,
            impact_pct=impact,
            expected_move_pct=expected,
        )
        if not reachable:
            return reject_signal(ctx, self.name, "TARGET_NOT_REACHABLE", expected_move=expected, impact=impact)

        return pass_signal(
            ctx,
            self.name,
            score=0.55,
            confidence=0.50,
            entry_reason=f"micro_pullback_ema5_{ema5:.6f}",
            invalidation_reason="pullback_fails_to_resume",
            expected_move_pct=expected,
            impact_pct=impact,
            limit_buy=fill,
            setup_context={"ema5": ema5},
        )
