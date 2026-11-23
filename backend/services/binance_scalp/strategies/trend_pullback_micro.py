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

        # Require genuine 12m micro-uptrend and block post-blowoff falling-knife rollovers
        c12 = [b["close"] for b in bars[-12:]]
        high12 = max(b["high"] for b in bars[-12:])
        sma12 = sum(c12) / len(c12)
        closes = [b["close"] for b in bars[-8:]]
        ema5 = sum(closes[-5:]) / 5 if len(closes) >= 5 else closes[-1]
        uptrend = ema5 >= sma12 and c12[-1] >= c12[0] * 0.9995
        post_blowoff_rollover = high12 > 0 and (high12 - ctx.snap.mid) / high12 > 0.0040 and ctx.snap.mid < ema5
        pullback = uptrend and not post_blowoff_rollover and ctx.snap.mid < ema5 * 1.0005 and ctx.snap.mid > ema5 * 0.998

        mom = ctx.mom
        resume = mom.mid_change_15s > -0.00005  # not strongly down

        if not (pullback and resume):
            return reject_signal(ctx, self.name, "NO_MICRO_PULLBACK")

        structural = 0.0025

        expected = estimate_expected_move_pct(bars, structural=structural, atr_mult=0.60, cap_pct=0.006)
        reachable, _ = target_reachable(
            ctx.econ,
            spread_pct=ctx.snap.spread_pct,
            impact_pct=impact,
            expected_move_pct=expected,
        )
        if not reachable:
            return reject_signal(ctx, self.name, "TARGET_NOT_REACHABLE", expected_move=expected, impact=impact, directional_move=structural)

        cur = ctx.snap.mid
        score = 2.25 + max(0.0, mom.mid_change_15s) * 4000.0 + abs(cur - ema5) / max(ema5, 1e-12) * 200.0
        return pass_signal(
            ctx,
            self.name,
            score=score,
            confidence=0.50,
            entry_reason=f"micro_pullback_ema5_{ema5:.6f}",
            invalidation_reason="pullback_fails_to_resume",
            expected_move_pct=expected,
            directional_move_pct=structural,
            impact_pct=impact,
            limit_buy=fill,
            setup_context={"ema5": ema5},
        )
