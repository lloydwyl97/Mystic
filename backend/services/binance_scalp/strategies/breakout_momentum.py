"""Breakout momentum scalp — continuation after range break with rising bid/mid."""

from __future__ import annotations

from backend.services.binance_scalp.strategies.base import ScalpSetupSignal, StrategyMarketContext
from backend.services.binance_scalp.strategies.common import (
    check_spread,
    depth_check,
    pass_signal,
    reject_signal,
    target_reachable,
)


class BreakoutMomentumStrategy:
    name = "breakout_momentum"

    def evaluate(self, ctx: StrategyMarketContext) -> ScalpSetupSignal:
        ok, reason = check_spread(ctx.snap, ctx.econ, ctx.config)
        if not ok:
            return reject_signal(ctx, self.name, reason or "SPREAD_TOO_WIDE")

        depth_ok, impact, fill = depth_check(ctx.snap, ctx.notional_usd, ctx.econ)
        if not depth_ok:
            return reject_signal(ctx, self.name, "DEPTH_OR_IMPACT_FAIL", impact=impact)

        bars = ctx.bars_1m
        if len(bars) < 8:
            return reject_signal(ctx, self.name, "INSUFFICIENT_BARS")

        recent = bars[-6:-1]
        breakout_level = max(b["high"] for b in recent)
        cur = ctx.snap.mid
        range_pct = (max(b["high"] for b in recent) - min(b["low"] for b in recent)) / cur if cur > 0 else 0
        vol_recent = sum(b["volume"] for b in bars[-3:])
        vol_prior = sum(b["volume"] for b in bars[-6:-3]) or 1.0
        vol_expansion = vol_recent / vol_prior

        broke = cur > breakout_level * 1.00005
        if not broke:
            return reject_signal(ctx, self.name, "NO_BREAKOUT")

        mom = ctx.mom
        if not (
            mom.bid_change_15s > 0
            and mom.mid_change_15s > 0
            and mom.mid_change_30s > 0
        ):
            return reject_signal(ctx, self.name, "MOMENTUM_NOT_CONFIRMED")

        move_from_low = (cur - min(b["low"] for b in recent)) / cur if cur > 0 else 0
        if move_from_low > 0.006:
            return reject_signal(ctx, self.name, "MOVE_EXHAUSTED_CHASE")

        expected = min(range_pct * 0.55, 0.005)
        reachable, _ = target_reachable(ctx.econ, spread_pct=ctx.snap.spread_pct, impact_pct=impact, expected_move_pct=expected)
        if not reachable:
            return reject_signal(ctx, self.name, "TARGET_NOT_REACHABLE", expected_move=expected, impact=impact)

        score = 3.0 + vol_expansion + move_from_low * 200 + mom.bid_change_15s * 5000
        return pass_signal(
            ctx,
            self.name,
            score=score,
            confidence=min(0.95, 0.5 + vol_expansion * 0.1),
            entry_reason=f"breakout_above_{breakout_level:.6f}_vol_x{vol_expansion:.2f}",
            invalidation_reason="price_below_breakout_level_with_negative_momentum",
            expected_move_pct=expected,
            impact_pct=impact,
            limit_buy=fill,
            setup_context={
                "breakout_level": breakout_level,
                "range_pct": range_pct,
                "vol_expansion": vol_expansion,
            },
        )
