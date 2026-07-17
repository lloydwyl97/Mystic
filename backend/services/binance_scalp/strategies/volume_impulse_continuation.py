"""Volume impulse continuation — volume spike + price follow through.

Native for bull_trend / pump_continuation / high_vol_breakout.
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


class VolumeImpulseContinuationStrategy:
    name = "volume_impulse_continuation"

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

        vol_recent = sum(b["volume"] for b in bars[-2:])
        vol_prior = sum(b["volume"] for b in bars[-6:-2]) or 1.0
        impulse = vol_recent / vol_prior > 1.8

        price_up = ctx.snap.mid > bars[-2]["close"] if len(bars) > 1 else True

        if not (impulse and price_up):
            return reject_signal(ctx, self.name, "NO_VOLUME_IMPULSE")

        expected = estimate_expected_move_pct(bars, structural=0.0026, atr_mult=0.70, cap_pct=0.006)
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
            score=0.58,
            confidence=0.52,
            entry_reason=f"volume_impulse_{vol_recent / vol_prior:.2f}x",
            invalidation_reason="impulse_fails_to_continue",
            expected_move_pct=expected,
            impact_pct=impact,
            limit_buy=fill,
            setup_context={"vol_ratio": vol_recent / vol_prior},
        )
