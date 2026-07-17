"""Failed breakout reversal — push above high then immediate rejection.

Native for range / chop / dump_continuation.
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


class FailedBreakoutReversalStrategy:
    name = "failed_breakout_reversal"

    def evaluate(self, ctx: StrategyMarketContext) -> ScalpSetupSignal:
        ok, reason = check_spread(ctx.snap, ctx.econ, ctx.config)
        if not ok:
            return reject_signal(ctx, self.name, reason or "SPREAD_TOO_WIDE")

        depth_ok, impact, fill = depth_check(ctx.snap, ctx.notional_usd, ctx.econ)
        if not depth_ok:
            return reject_signal(ctx, self.name, "DEPTH_OR_IMPACT_FAIL", impact=impact)

        bars = ctx.bars_1m or []
        if len(bars) < 10:
            return reject_signal(ctx, self.name, "INSUFFICIENT_BARS")

        recent = bars[-6:-1]
        high = max(b["high"] for b in recent)
        cur = ctx.snap.mid
        failed = cur < high * 0.9995

        mom = ctx.mom
        down_mom = mom.mid_change_15s < 0 or mom.bid_change_15s < 0

        if not (failed and down_mom):
            return reject_signal(ctx, self.name, "NO_FAILED_BREAKOUT")

        expected = estimate_expected_move_pct(bars, structural=0.0022, atr_mult=0.60, cap_pct=0.006)
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
            score=0.52,
            confidence=0.48,
            entry_reason=f"failed_breakout_reject_{high:.6f}",
            invalidation_reason="rejection_fails_to_follow_down",
            expected_move_pct=expected,
            impact_pct=impact,
            limit_buy=fill,
            setup_context={"failed_high": high},
        )
