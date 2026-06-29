"""Order book tape scalp — bid pressure confirmed by price movement."""

from __future__ import annotations

from backend.services.binance_scalp.strategies.base import ScalpSetupSignal, StrategyMarketContext
from backend.services.binance_scalp.strategies.common import (
    check_spread,
    depth_check,
    pass_signal,
    reject_signal,
    target_reachable,
)


class OrderbookTapeScalpStrategy:
    name = "orderbook_tape_scalp"

    def evaluate(self, ctx: StrategyMarketContext) -> ScalpSetupSignal:
        ok, reason = check_spread(ctx.snap, ctx.econ, ctx.config)
        if not ok:
            return reject_signal(ctx, self.name, reason or "SPREAD_TOO_WIDE")

        depth_ok, impact, fill = depth_check(ctx.snap, ctx.notional_usd, ctx.econ)
        if not depth_ok:
            return reject_signal(ctx, self.name, "DEPTH_OR_IMPACT_FAIL", impact=impact)

        imb = ctx.snap.order_book_imbalance
        if imb is None or imb < 0.10:
            return reject_signal(ctx, self.name, "IMBALANCE_ALONE_INSUFFICIENT")

        bid_qty = sum(q for _, q in ctx.snap.bids[:5])
        ask_qty = sum(q for _, q in ctx.snap.asks[:5])
        if ask_qty > bid_qty * 2.5:
            return reject_signal(ctx, self.name, "ASK_LIQUIDITY_TOO_THICK")

        mom = ctx.mom
        if not (mom.bid_change_15s > 0 and mom.mid_change_15s > 0 and mom.mid_change_30s > 0):
            return reject_signal(ctx, self.name, "PRICE_NOT_CONFIRMING_IMBALANCE")

        expected = min(ctx.snap.spread_pct * 6 + imb * 0.002, 0.004)
        reachable, _ = target_reachable(ctx.econ, spread_pct=ctx.snap.spread_pct, impact_pct=impact, expected_move_pct=expected)
        if not reachable:
            return reject_signal(ctx, self.name, "TARGET_NOT_REACHABLE", expected_move=expected, impact=impact)

        score = 2.0 + imb * 10 + mom.bid_change_15s * 4000
        return pass_signal(
            ctx,
            self.name,
            score=score,
            confidence=min(0.9, 0.4 + imb),
            entry_reason=f"bid_imbalance={imb:.3f}_bid_stepping",
            invalidation_reason="bid_pressure_gone_price_rollover",
            expected_move_pct=expected,
            impact_pct=impact,
            limit_buy=fill,
            setup_context={"imbalance": imb, "bid_qty_top5": bid_qty, "ask_qty_top5": ask_qty},
        )
