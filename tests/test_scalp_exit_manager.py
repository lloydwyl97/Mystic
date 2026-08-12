"""Unit tests for scalp exit manager review/scratch paths."""

from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from unittest import mock

from backend.services.binance_scalp.config import get_scalp_config
from backend.services.binance_scalp.economics import ScalpEconomics
from backend.services.binance_scalp.exit_manager import (
    EXIT_EARLY_SCRATCH,
    EXIT_MAX_HOLD_HARD_LIMIT,
    PositionTrack,
    STATE_OPEN,
    _effective_scratch_min_reviews,
    _early_scratch_exit,
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
        max_favorable_pct=0.00001,
        max_adverse_pct=0.0002,
        session_low_bid=bid,
        stale_review_count=2,
        review_lows=(bid, bid),
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
        hold_sec=max(float(econ.stale_scalp_timeout_sec + 30), 610.0),
        executable_net_pct=-0.0003,
        profit_hit=False,
        exit_spread_ok=True,
        perform_review=True,
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
        perform_review=True,
    )
    assert review.decision == "SELL", review.reason
    assert review.exit_reason == EXIT_EARLY_SCRATCH
    assert review.exit_reason != EXIT_MAX_HOLD_HARD_LIMIT


# --- Item p8 promotion: HoldEV scratch-review-count reduction wiring ---


def test_effective_scratch_min_reviews_reduces_but_floors_at_one():
    from backend.services.binance_scalp.exit_manager import _scratch_min_reviews

    base = _scratch_min_reviews("range_bounce_scalp")
    assert base == 3
    assert _effective_scratch_min_reviews("range_bounce_scalp", 0) == base
    assert _effective_scratch_min_reviews("range_bounce_scalp", 1) == base - 1
    assert _effective_scratch_min_reviews("range_bounce_scalp", 999) == 1  # never below the floor


def test_early_scratch_exit_fires_sooner_with_hold_ev_reduction():
    econ = ScalpEconomics.from_env()
    kwargs = dict(
        hold_sec=610.0,
        hard=100000,
        max_fav=0.00001,
        executable_net_pct=-0.0003,
        mom=_flat_mom(),
        recovery=0.0,
        econ=econ,
        stale_review_count=2,
        setup_name="range_bounce_scalp",
    )
    fired_without, _ = _early_scratch_exit(**kwargs, hold_ev_reduction=0)
    assert fired_without is False  # 2 stale reviews < the arm's normal min of 3

    fired_with, reason = _early_scratch_exit(**kwargs, hold_ev_reduction=1)
    assert fired_with is True  # HoldEV's tighten-only reduction lowers the bar to 2
    assert reason


def test_evaluate_exit_applies_hold_ev_scratch_reduction_end_to_end():
    """Full evaluate_exit wiring: a position that would NOT scratch on its
    own stale-review count does scratch once HoldEV strongly disfavors
    continuing to hold, and does not exceed what the underlying scratch
    logic would otherwise allow."""
    econ = ScalpEconomics.from_env()
    config = get_scalp_config()
    entry = 100.0
    bid = 99.98
    track = PositionTrack(
        entry_price=entry,
        state=STATE_OPEN,
        max_favorable_pct=0.00001,
        max_adverse_pct=0.0002,
        session_low_bid=bid,
        stale_review_count=2,
        review_lows=(bid, bid),
        setup_name="range_bounce_scalp",
        setup_context={},
    )
    common_kwargs = dict(
        track=track,
        snap=_Snap("BTCUSDT", bid),
        mom=_flat_mom(),
        econ=econ,
        config=config,
        trade_id="t3",
        hold_sec=610.0,
        executable_net_pct=-0.0003,
        profit_hit=False,
        exit_spread_ok=True,
        perform_review=False,  # avoid the stale_review_count += 1 path so only the pre-review checks apply
    )
    with mock.patch("backend.services.hold_ev_engine.compute_hold_ev") as mock_compute, mock.patch("backend.services.hold_ev_engine.hold_ev_scratch_review_reduction", return_value=0):
        neutral = evaluate_exit(**common_kwargs)
    assert neutral.decision == "HOLD"

    with mock.patch("backend.services.hold_ev_engine.compute_hold_ev") as mock_compute2, mock.patch("backend.services.hold_ev_engine.hold_ev_scratch_review_reduction", return_value=1):
        tightened = evaluate_exit(**common_kwargs)
    assert tightened.decision == "SELL"
    assert tightened.exit_reason == EXIT_EARLY_SCRATCH
    assert tightened.diagnostics.get("hold_ev_scratch_review_reduction") == 1
