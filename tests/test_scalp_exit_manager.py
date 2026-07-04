"""Unit tests for scalp exit manager review/scratch paths."""

from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from backend.services.binance_scalp.config import get_scalp_config
from backend.services.binance_scalp.economics import ScalpEconomics
from backend.services.binance_scalp.exit_manager import (
    EXIT_EARLY_SCRATCH,
    EXIT_MAX_HOLD_HARD_LIMIT,
    PositionTrack,
    STATE_OPEN,
    evaluate_exit,
)
from backend.services.binance_scalp.momentum_tracker import MomentumDiagnostics


class _Snap:
    def __init__(self, symbol: str, bid: float, spread_pct: float = 0.0002) -> None:
        self.symbol = symbol
        self.best_bid = bid
        self.best_ask = bid * 1.0002
        self.mid = bid
        self.spread_pct = spread_pct


def _flat_mom() -> MomentumDiagnostics:
    return MomentumDiagnostics(
        mid_change_15s=0.0,
        mid_change_30s=0.0,
        mid_change_60s=0.0,
        bid_change_15s=0.0,
        bid_change_30s=0.0,
        bid_change_60s=0.0,
        last_n_ticks_up_count=0,
        sample_count=8,
        history_sec=60.0,
        recent_range_pct=0.0001,
        realized_volatility_pct=0.0001,
        momentum_confirmed=False,
        flat_regime=True,
    )


def test_early_scratch_on_stalled_flat_position():
    econ = ScalpEconomics.from_env()
    config = get_scalp_config()
    entry = 100.0
    bid = 99.98
    track = PositionTrack(
        entry_price=entry,
        state=STATE_OPEN,
        max_favorable_pct=0.0001,
        max_adverse_pct=0.0002,
        session_low_bid=bid,
        stale_review_count=1,
        review_lows=(bid,),
        setup_name="range_bounce_scalp",
        setup_context={},
    )
    review = evaluate_exit(
        track=track,
        snap=_Snap("BTCUSDT", bid),
        mom=_flat_mom(),
        econ=econ,
        config=config,
        trade_id="t1",
        hold_sec=float(econ.stale_scalp_timeout_sec + 30),
        executable_net_pct=-0.0003,
        profit_hit=False,
        exit_spread_ok=True,
        perform_review=False,
    )
    assert review.decision == "SELL", review.reason
    assert review.exit_reason == EXIT_EARLY_SCRATCH


def test_stall_before_max_hold_not_at_hard_ceiling():
    econ = ScalpEconomics.from_env()
    config = get_scalp_config()
    entry = 100.0
    bid = 99.97
    track = PositionTrack(
        entry_price=entry,
        state=STATE_OPEN,
        max_favorable_pct=0.0002,
        max_adverse_pct=0.0003,
        session_low_bid=bid,
        stale_review_count=2,
        review_lows=(bid, bid),
        setup_name="range_bounce_scalp",
        setup_context={},
    )
    review = evaluate_exit(
        track=track,
        snap=_Snap("ETHUSDT", bid),
        mom=_flat_mom(),
        econ=econ,
        config=config,
        trade_id="t2",
        hold_sec=910.0,
        executable_net_pct=-0.0004,
        profit_hit=False,
        exit_spread_ok=True,
        perform_review=False,
    )
    assert review.decision == "SELL", review.reason
    assert review.exit_reason == EXIT_EARLY_SCRATCH
    assert review.exit_reason != EXIT_MAX_HOLD_HARD_LIMIT
