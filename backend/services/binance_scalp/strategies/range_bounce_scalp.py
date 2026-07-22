"""Range bounce scalp — support rejection with momentum flip."""

from __future__ import annotations

import os

from backend.services.binance_scalp.strategies.base import ScalpSetupSignal, StrategyMarketContext
from backend.services.binance_scalp.strategies.common import (
    check_spread,
    depth_check,
    estimate_expected_move_pct,
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
        support_cap = 0.002 if ctx.config.scalp_paper_enabled else 0.0015
        if dist_to_support > support_cap:
            return reject_signal(ctx, self.name, "NOT_NEAR_SUPPORT")

        hi = max(b["high"] for b in support_window)
        range_pct = (hi - support) / cur if cur > 0 else 1.0
        if range_pct > 0.008:
            return reject_signal(ctx, self.name, "RANGE_TOO_WIDE")

        last = bars[-1]
        # True lower-wick fraction: (min(open, close) - low) / (high - low).
        # Replaces the old (close - low) / close measure, which ignored the high
        # and was not range-normalized. Still NOT a hard gate — continuous
        # score/rank contribution only. Zero/absent wick must never remove an
        # otherwise-executable candidate; bounce evidence comes from the
        # momentum-flip checks below (live 15s/30s/60s order-book momentum).
        _bar_open = float(last.get("open", last["close"]))
        _bar_close = float(last["close"])
        _bar_high = float(last["high"])
        _bar_low = float(last["low"])
        _bar_range = _bar_high - _bar_low
        wick_rejection = (min(_bar_open, _bar_close) - _bar_low) / _bar_range if _bar_range > 0 else 0.0
        # Legacy close-above-low / close (kept for audit joins against older trades).
        wick_close_above_low_pct = (_bar_close - _bar_low) / _bar_close if _bar_close > 0 else 0.0

        # Optional wick floor (default off). Set SCALP_RANGE_MIN_WICK_FRAC>0 in
        # paper profile to drop the weakest rejection candles; zero-wick must
        # remain admissible when unset (score/rank still use wick_rejection).
        min_wick = float(os.getenv("SCALP_RANGE_MIN_WICK_FRAC", "0") or "0")
        if min_wick > 0.0 and wick_rejection < min_wick:
            return reject_signal(ctx, self.name, "WEAK_REJECTION_WICK")

        mom = ctx.mom
        if not (mom.bid_change_15s > 0 and mom.mid_change_15s > 0 and mom.mid_change_30s > 0):
            return reject_signal(ctx, self.name, "MOMENTUM_NOT_FLIPPED")
        if mom.bid_change_60s < -0.0001:
            return reject_signal(ctx, self.name, "MOMENTUM_NOT_SUSTAINED")
        # Prefer sustained 60s mid lift so we don't enter pure 15s blips that
        # get early-scratched before the bounce develops.
        min_mid_60 = float(os.getenv("SCALP_RANGE_MIN_MID_CHANGE_60S", "0.0") or "0.0")
        if float(getattr(mom, "mid_change_60s", 0.0) or 0.0) < min_mid_60:
            return reject_signal(ctx, self.name, "MOMENTUM_NOT_SUSTAINED")

        recovery = (cur - support) / cur if cur > 0 else 0
        # Project toward range high (bounce target), not a hard 0.25% micro-cap
        # that sits at/below net_profit_target and always fails reachability.
        to_high = (hi - cur) / cur if cur > 0 else 0.0
        structural = max(to_high, recovery + 0.0008)
        expected = estimate_expected_move_pct(bars, structural=structural, atr_mult=0.55, cap_pct=0.006)
        reachable, _ = target_reachable(ctx.econ, spread_pct=ctx.snap.spread_pct, impact_pct=impact, expected_move_pct=expected)
        if not reachable:
            return reject_signal(ctx, self.name, "TARGET_NOT_REACHABLE", expected_move=expected, impact=impact)

        # Weights rescaled for [0, 1] wick fraction (old formula used ~0.001–0.005
        # close-above-low ratios with *500 / *200 multipliers). Target similar
        # contribution magnitude: ~0.5–1.5 score points for a typical bounce wick.
        score = 2.5 + wick_rejection * 1.5 + recovery * 400
        confidence = min(0.72, 0.55 + wick_rejection * 0.25)
        return pass_signal(
            ctx,
            self.name,
            score=score,
            confidence=confidence,
            entry_reason=f"support_bounce_{support:.6f}_wick={wick_rejection:.4f}",
            invalidation_reason="support_break_no_recovery",
            expected_move_pct=expected,
            impact_pct=impact,
            limit_buy=fill,
            setup_context={
                "support_level": support,
                "wick_rejection_pct": wick_rejection,
                "wick_rejection_range_pct": wick_rejection,
                "wick_close_above_low_pct": wick_close_above_low_pct,
            },
        )
