"""Per-strategy setup invalidation for paper exit manager."""

from __future__ import annotations

from typing import Any

from backend.services.binance_scalp.market_reader import MarketSnapshot
from backend.services.binance_scalp.momentum_tracker import MomentumDiagnostics


def setup_invalidated(
    setup_name: str,
    setup_context: dict[str, Any],
    *,
    snap: MarketSnapshot,
    mom: MomentumDiagnostics,
    entry_price: float,
    executable_net_pct: float,
) -> tuple[bool, str]:
    """Return (invalidated, reason)."""
    bid = snap.best_bid
    mid = snap.mid
    name = (setup_name or "").strip().lower()

    if name == "breakout_momentum":
        level = float(setup_context.get("breakout_level") or entry_price)
        if bid < level and mom.bid_change_15s < 0 and mom.bid_change_30s <= 0:
            return True, "below_breakout_level_negative_momentum"
        return False, ""

    if name == "vwap_ema_reclaim":
        vwap = float(setup_context.get("vwap") or entry_price)
        if mid < vwap * 0.999 and mom.bid_change_30s < 0 and executable_net_pct < 0:
            return True, "lost_vwap_no_recovery"
        return False, ""

    if name == "orderbook_tape_scalp":
        imb = snap.order_book_imbalance or 0.0
        if imb < 0.02 and mom.bid_change_15s < 0 and mom.mid_change_15s < 0:
            return True, "bid_pressure_gone_rollover"
        return False, ""

    if name == "range_bounce_scalp":
        support = float(setup_context.get("support_level") or entry_price * 0.998)
        if bid < support * 0.9998 and mom.bid_change_15s <= 0:
            return True, "support_broken_no_bid_recovery"
        if bid < support * 0.9995 and mom.bid_change_30s < 0:
            return True, "support_broken"
        return False, ""

    if executable_net_pct < -0.002 and mom.bid_change_15s < 0 and mom.bid_change_30s < 0:
        return True, "generic_setup_failed"
    return False, ""
