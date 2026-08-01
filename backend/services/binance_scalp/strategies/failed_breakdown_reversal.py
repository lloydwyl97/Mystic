"""Failed breakdown reversal scalp — sweep low then reclaim with volume/momentum flip.

Paper only. Regime-native for trend_down / range.
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


class FailedBreakdownReversalStrategy:
    name = "failed_breakdown_reversal"

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

        recent = bars[-8:-1]
        lows = [b["low"] for b in recent]
        sweep_low = min(lows)
        cur = ctx.snap.mid
        reclaimed = cur > sweep_low * 1.0008

        vol_recent = sum(b["volume"] for b in bars[-3:])
        vol_prior = sum(b["volume"] for b in bars[-6:-3]) or 1.0
        vol_ok = vol_recent > vol_prior * 1.1

        mom = ctx.mom
        mom_ok = mom.mid_change_15s > 0 or mom.bid_change_15s > 0

        if not (reclaimed and vol_ok and mom_ok):
            return reject_signal(ctx, self.name, "NO_REVERSAL_CONFIRM")

        expected = estimate_expected_move_pct(bars, structural=0.0028, atr_mult=0.65, cap_pct=0.006)
        reachable, _ = target_reachable(
            ctx.econ,
            spread_pct=ctx.snap.spread_pct,
            impact_pct=impact,
            expected_move_pct=expected,
        )
        if not reachable:
            return reject_signal(ctx, self.name, "TARGET_NOT_REACHABLE", expected_move=expected, impact=impact)

        # Align pass score with working strategies (~2.0–3.0+ vs floor 1.45).
        score = 2.45 + max(0.0, (cur - sweep_low) / max(sweep_low, 1e-12)) * 350.0 + (vol_recent / vol_prior) * 0.12
        return pass_signal(
            ctx,
            self.name,
            score=score,
            confidence=0.60,
            entry_reason=f"failed_breakdown_reclaim_{sweep_low:.6f}",
            invalidation_reason="reclaim_fails_or_momentum_fades",
            expected_move_pct=expected,
            impact_pct=impact,
            limit_buy=fill,
            setup_context={"sweep_low": sweep_low, "vol_ratio": vol_recent / vol_prior},
        )
