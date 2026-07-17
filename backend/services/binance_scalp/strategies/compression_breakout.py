"""Compression breakout scalp — low vol contraction then expansion breakout.

Native for vol_crush / vol_expansion / high_vol_breakout.
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


class CompressionBreakoutStrategy:
    name = "compression_breakout"

    def evaluate(self, ctx: StrategyMarketContext) -> ScalpSetupSignal:
        ok, reason = check_spread(ctx.snap, ctx.econ, ctx.config)
        if not ok:
            return reject_signal(ctx, self.name, reason or "SPREAD_TOO_WIDE")

        depth_ok, impact, fill = depth_check(ctx.snap, ctx.notional_usd, ctx.econ)
        if not depth_ok:
            return reject_signal(ctx, self.name, "DEPTH_OR_IMPACT_FAIL", impact=impact)

        bars = ctx.bars_1m or []
        if len(bars) < 15:
            return reject_signal(ctx, self.name, "INSUFFICIENT_BARS")

        recent = bars[-10:]
        highs = [b["high"] for b in recent]
        lows = [b["low"] for b in recent]
        range_pct = (max(highs) - min(lows)) / max(ctx.snap.mid, 1e-9)
        vol_recent = sum(b["volume"] for b in bars[-3:])
        vol_prior = sum(b["volume"] for b in bars[-8:-3]) or 1.0
        expansion = vol_recent / vol_prior > 1.4 and range_pct < 0.004

        if not expansion:
            return reject_signal(ctx, self.name, "NO_COMPRESSION_BREAK")

        expected = estimate_expected_move_pct(bars, structural=0.0030, atr_mult=0.70, cap_pct=0.006)
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
            score=0.6,
            confidence=0.55,
            entry_reason=f"compression_break_range_{range_pct:.5f}",
            invalidation_reason="no_follow_through_or_vol_fades",
            expected_move_pct=expected,
            impact_pct=impact,
            limit_buy=fill,
            setup_context={"range_pct": range_pct},
        )
