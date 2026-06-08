"""Range bounce scalp — support rejection with momentum flip."""

from __future__ import annotations

from backend.services.binance_scalp.strategies.base import ScalpSetupSignal, StrategyMarketContext
from backend.services.binance_scalp.strategies.common import (
    check_spread,
    depth_check,
    pass_signal,
    reject_signal,
    target_reachable,
)


class RangeBounceScalpStrategy:
    name = "range_bounce_scalp"

    def evaluate(self, ctx: StrategyMarketContext) -> ScalpSetupSignal:
        ok, reason = check_spread(ctx.snap, ctx.econ, ctx.config)
        if not ok:
            return reject_signal(ctx, self.name, reason or "SPREAD_TOO_WIDE")

        depth_ok, impact, fill = depth_check(ctx.snap, ctx.notional_usd, ctx.econ)
        if not depth_ok:
            return reject_signal(ctx, self.name, "DEPTH_OR_IMPACT_FAIL", impact=impact)

        bars = ctx.bars_1m
        if len(bars) < 10:
            return reject_signal(ctx, self.name, "INSUFFICIENT_BARS")

        support_window = bars[-15:] if len(bars) >= 15 else bars
        support = min(b["low"] for b in support_window)
        cur = ctx.snap.mid
        dist_to_support = (cur - support) / cur if cur > 0 else 1.0
        if dist_to_support > 0.0018:
            return reject_signal(ctx, self.name, "NOT_NEAR_SUPPORT")

        last = bars[-1]
        wick_rejection = (last["close"] - last["low"]) / last["close"] if last["close"] > 0 else 0
        if wick_rejection < 0.0002 and ctx.mom.bid_change_15s <= 0:
            return reject_signal(ctx, self.name, "NO_REJECTION_WICK")

        mom = ctx.mom
        if not (
            mom.bid_change_15s > 0
            and mom.mid_change_15s > 0
            and mom.mid_change_30s > 0
        ):
            return reject_signal(ctx, self.name, "MOMENTUM_NOT_FLIPPED")

        recovery = (cur - support) / cur if cur > 0 else 0
        expected = min(recovery + 0.001, 0.0035)
        reachable, _ = target_reachable(ctx.econ, spread_pct=ctx.snap.spread_pct, impact_pct=impact, expected_move_pct=expected)
        if not reachable:
            return reject_signal(ctx, self.name, "TARGET_NOT_REACHABLE", expected_move=expected, impact=impact)

        score = 2.2 + wick_rejection * 400 + recovery * 300
        return pass_signal(
            ctx,
            self.name,
            score=score,
            confidence=0.6,
            entry_reason=f"support_bounce_{support:.6f}_wick={wick_rejection:.4f}",
            invalidation_reason="support_break_no_recovery",
            expected_move_pct=expected,
            impact_pct=impact,
            limit_buy=fill,
            setup_context={"support_level": support, "wick_rejection_pct": wick_rejection},
        )
