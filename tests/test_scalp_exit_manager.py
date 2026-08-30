"""Unit tests for scalp exit manager review/scratch paths."""

from __future__ import annotations

import inspect
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
    EXIT_PATH_EXECUTABLE_PROFIT,
    EXIT_PATH_MAX_ADVERSE_STOP,
    PositionTrack,
    STATE_OPEN,
    _effective_scratch_min_reviews,
    _early_scratch_exit,
    _max_hold_hard_sec,
    _path_max_adverse_net_pct,
    _path_min_executable_net_pct,
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


def test_early_scratch_on_stalled_flat_position(monkeypatch):
    monkeypatch.setenv("SCALP_PATH_AWARE_EXIT", "false")
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


def test_stall_before_max_hold_not_at_hard_ceiling(monkeypatch):
    monkeypatch.setenv("SCALP_PATH_AWARE_EXIT", "false")
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


def test_evaluate_exit_applies_hold_ev_scratch_reduction_end_to_end(monkeypatch):
    monkeypatch.setenv("SCALP_PATH_AWARE_EXIT", "false")
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


def _path_track(entry: float = 100.0, bid: float = 99.98) -> PositionTrack:
    return PositionTrack(
        entry_price=entry,
        state=STATE_OPEN,
        max_favorable_pct=0.00001,
        max_adverse_pct=0.0002,
        session_low_bid=bid,
        stale_review_count=4,
        review_lows=(bid, bid, bid),
        setup_name="vwap_ema_reclaim",
        setup_context={"soft_rank_entry": True},
    )


def test_path_aware_takes_first_executable_profit(monkeypatch):
    monkeypatch.setenv("SCALP_PATH_AWARE_EXIT", "true")
    review = evaluate_exit(
        track=_path_track(100.0, 100.08),
        snap=_Snap("BTCUSDT", 100.08),
        mom=_flat_mom(),
        econ=ScalpEconomics.from_env(),
        config=get_scalp_config(),
        trade_id="path-tp",
        hold_sec=45.0,
        executable_net_pct=0.0007,
        profit_hit=False,
        exit_spread_ok=True,
        perform_review=True,
    )
    assert review.decision == "SELL"
    assert review.exit_reason == EXIT_PATH_EXECUTABLE_PROFIT


def test_path_aware_profit_floor_matches_roundtrip_cost(monkeypatch):
    """+1bp crumbs are below taker+slip (6bp). Floor is the cost hurdle."""
    monkeypatch.delenv("SCALP_PATH_MIN_EXECUTABLE_NET_PCT", raising=False)
    assert _path_min_executable_net_pct() == 0.0006


def test_path_aware_does_not_clip_one_bp_crumb(monkeypatch):
    monkeypatch.setenv("SCALP_PATH_AWARE_EXIT", "true")
    monkeypatch.delenv("SCALP_PATH_MIN_EXECUTABLE_NET_PCT", raising=False)
    review = evaluate_exit(
        track=_path_track(100.0, 100.02),
        snap=_Snap("BTCUSDT", 100.02),
        mom=_flat_mom(),
        econ=ScalpEconomics.from_env(),
        config=get_scalp_config(),
        trade_id="path-crumb",
        hold_sec=45.0,
        executable_net_pct=0.0001,
        profit_hit=False,
        exit_spread_ok=True,
        perform_review=True,
    )
    assert review.decision != "SELL" or review.exit_reason != EXIT_PATH_EXECUTABLE_PROFIT


def test_path_aware_cuts_a_loser_at_the_bounded_stop(monkeypatch):
    """Without this the path-aware branch has no loss exit before 20 minutes."""
    monkeypatch.setenv("SCALP_PATH_AWARE_EXIT", "true")
    review = evaluate_exit(
        track=_path_track(100.0, 99.80),
        snap=_Snap("ETHUSDT", 99.80),
        mom=_flat_mom(),
        econ=ScalpEconomics.from_env(),
        config=get_scalp_config(),
        trade_id="path-stop",
        hold_sec=120.0,
        executable_net_pct=-0.0020,
        profit_hit=False,
        exit_spread_ok=True,
        perform_review=True,
    )
    assert review.decision == "SELL"
    assert review.exit_reason == EXIT_PATH_MAX_ADVERSE_STOP


def test_path_aware_stop_fires_well_before_the_horizon(monkeypatch):
    """The stop must not depend on hold time; 28/28 horizon exits were losses."""
    monkeypatch.setenv("SCALP_PATH_AWARE_EXIT", "true")
    review = evaluate_exit(
        track=_path_track(100.0, 99.85),
        snap=_Snap("XRPUSDT", 99.85),
        mom=_flat_mom(),
        econ=ScalpEconomics.from_env(),
        config=get_scalp_config(),
        trade_id="path-stop-early",
        hold_sec=5.0,
        executable_net_pct=-0.0015,
        profit_hit=False,
        exit_spread_ok=True,
        perform_review=False,
    )
    assert review.decision == "SELL"
    assert review.exit_reason == EXIT_PATH_MAX_ADVERSE_STOP


def test_path_aware_stop_sits_beyond_predicted_mae(monkeypatch):
    """Largest predicted MAE across traded symbols is 0.122%; the stop must clear
    it so normal adverse excursion does not close a healthy position."""
    monkeypatch.delenv("SCALP_PATH_MAX_ADVERSE_NET_PCT", raising=False)
    assert _path_max_adverse_net_pct() > 0.00122


def test_path_aware_stop_is_configurable(monkeypatch):
    monkeypatch.setenv("SCALP_PATH_AWARE_EXIT", "true")
    monkeypatch.setenv("SCALP_PATH_MAX_ADVERSE_NET_PCT", "0.0050")
    review = evaluate_exit(
        track=_path_track(100.0, 99.80),
        snap=_Snap("ETHUSDT", 99.80),
        mom=_flat_mom(),
        econ=ScalpEconomics.from_env(),
        config=get_scalp_config(),
        trade_id="path-stop-wide",
        hold_sec=120.0,
        executable_net_pct=-0.0020,
        profit_hit=False,
        exit_spread_ok=True,
        perform_review=True,
    )
    assert review.decision == "HOLD"


def test_path_aware_does_not_scratch_flat_loser(monkeypatch):
    monkeypatch.setenv("SCALP_PATH_AWARE_EXIT", "true")
    review = evaluate_exit(
        track=_path_track(),
        snap=_Snap("ETHUSDT", 99.98),
        mom=_flat_mom(),
        econ=ScalpEconomics.from_env(),
        config=get_scalp_config(),
        trade_id="path-hold",
        hold_sec=610.0,
        executable_net_pct=-0.0003,
        profit_hit=False,
        exit_spread_ok=True,
        perform_review=True,
    )
    assert review.decision == "HOLD"
    assert review.reason == "path_awaiting_executable_profit"


def test_path_aware_holds_all_four_symbols(monkeypatch):
    monkeypatch.setenv("SCALP_PATH_AWARE_EXIT", "true")
    for sym in ("BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT"):
        review = evaluate_exit(
            track=_path_track(),
            snap=_Snap(sym, 99.98),
            mom=_flat_mom(),
            econ=ScalpEconomics.from_env(),
            config=get_scalp_config(),
            trade_id=f"path-{sym}",
            hold_sec=200.0,
            executable_net_pct=-0.0002,
            profit_hit=False,
            exit_spread_ok=True,
            perform_review=False,
        )
        assert review.decision == "HOLD", sym


def test_scalp_max_hold_not_suppressed_by_4h():
    import backend.services.binance_scalp.exit_manager as em
    import backend.services.binance_scalp.paper_engine as pe

    src = inspect.getsource(em.evaluate_exit) + inspect.getsource(pe.BinanceScalpPaperEngine._try_exit)
    assert "htf_4h_rise_intact" not in src
    assert "_scalp_4h_rise_intact" not in src
    assert "4h_breakout_intact" not in src


def test_scalp_profit_hit_still_clips_like_scalp(monkeypatch):
    monkeypatch.setenv("SCALP_PATH_AWARE_EXIT", "true")
    review = evaluate_exit(
        track=_path_track(100.0, 100.25),
        snap=_Snap("SOLUSDT", 100.25),
        mom=_flat_mom(),
        econ=ScalpEconomics.from_env(),
        config=get_scalp_config(),
        trade_id="scalp-clip",
        hold_sec=2000.0,
        executable_net_pct=0.008,
        profit_hit=True,
        exit_spread_ok=True,
        perform_review=True,
    )
    assert review.decision == "SELL"
    assert review.exit_reason == "NET_PROFIT_TARGET"


def test_scalp_live_remains_false_and_cb_config_present():
    cfg = get_scalp_config()
    assert cfg.scalp_live is False
    assert cfg.max_consecutive_losses >= 1
    assert cfg.scalp_paper_enabled is True


def test_scalp_stays_short_horizon():
    """Max hold stays in scalp territory, not DAY-length holds."""
    econ = ScalpEconomics.from_env()
    assert _max_hold_hard_sec(econ) <= 3600


def test_scalp_module_has_no_day_thesis_dependency():
    import backend.services.binance_scalp.exit_manager as em

    src = inspect.getsource(em)
    assert "htf_4h_rise_intact" not in src
    assert "htf_4h_rise_broken" not in src
    assert "day_trade_thesis" not in src
