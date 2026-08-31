"""Failed breakout reversal — probe high, reject, then reclaim with up momentum.

Long-only. Native for range / chop / dump_continuation.
Bearish chase (down-momentum after failed high) is rejected — that is a short
setup and must not emit BUY in this book.
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
        probed = max(b["high"] for b in bars[-4:]) >= high * 0.9998
        rejected_below_high = cur < high * 0.9995

        mom = ctx.mom
        # Long entry requires absorption + reclaim (up momentum), not dump chase.
        up_mom = mom.mid_change_15s > 0 and mom.bid_change_15s > 0

        if not (probed and rejected_below_high and up_mom):
            return reject_signal(ctx, self.name, "NO_FAILED_BREAKOUT_RECLAIM")

        expected = estimate_expected_move_pct(bars, structural=0.0022, atr_mult=0.60, cap_pct=0.006)
        reachable, _ = target_reachable(
            ctx.econ,
            spread_pct=ctx.snap.spread_pct,
            impact_pct=impact,
            expected_move_pct=expected,
        )
        if not reachable:
            return reject_signal(ctx, self.name, "TARGET_NOT_REACHABLE", expected_move=expected, impact=impact)

        # Same score band as working strategies (~2.0-3.0+); floor is SCALP_MIN_TRADEABLE_SCORE=1.45.
        score = 2.35 + max(0.0, (high - cur) / high) * 280.0 + mom.mid_change_15s * 3500.0
        return pass_signal(
            ctx,
            self.name,
            score=score,
            confidence=0.55,
            entry_reason=f"failed_breakout_reclaim_{high:.6f}",
            invalidation_reason="reclaim_fails_or_momentum_fades",
            expected_move_pct=expected,
            impact_pct=impact,
            limit_buy=fill,
            setup_context={"failed_high": high},
        )
