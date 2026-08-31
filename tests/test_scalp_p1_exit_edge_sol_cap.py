"""P1 SCALP retune: earlier stall exit defaults, reachability padding, SOL notional cap."""

from __future__ import annotations

import os

import pytest

from backend.services.binance_scalp.config import ScalpConfig
from backend.services.binance_scalp.economics import ScalpEconomics
from backend.services.binance_scalp.exit_manager import (
    EXIT_EARLY_SCRATCH,
    STATE_OPEN,
    PositionTrack,
    _stall_exit_hold_frac,
    _stall_exit_min_sec,
    evaluate_exit,
)
from backend.services.binance_scalp.momentum_tracker import MomentumDiagnostics
from backend.services.binance_scalp.strategies.common import target_reachable


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


def test_stall_exit_defaults_cut_before_three_quarter_hold(monkeypatch):
    monkeypatch.delenv("SCALP_STALL_EXIT_HOLD_FRAC", raising=False)
    monkeypatch.delenv("SCALP_STALL_EXIT_MIN_SEC", raising=False)
    assert _stall_exit_hold_frac() == pytest.approx(0.50)
    assert _stall_exit_min_sec(1200) == 600


def test_stall_before_max_hold_fires_near_half_ceiling(monkeypatch):
    monkeypatch.setenv("SCALP_PATH_AWARE_EXIT", "false")
    monkeypatch.setenv("SCALP_HOLD_MAX_MINUTES", "20")
    monkeypatch.setenv("SCALP_STALL_EXIT_HOLD_FRAC", "0.50")
    monkeypatch.setenv("SCALP_STALL_EXIT_MIN_SEC", "600")
    monkeypatch.setenv("SCALP_SCRATCH_PROGRESS_FRAC", "0.40")
    monkeypatch.setenv("SCALP_SCRATCH_MIN_REVIEWS", "1")
    monkeypatch.setenv("SCALP_STALE_TIMEOUT_SEC", "120")
    monkeypatch.setenv("SCALP_REVIEW_TRIGGER_SEC", "120")
    econ = ScalpEconomics.from_env()
    from backend.services.binance_scalp.config import get_scalp_config

    config = get_scalp_config()
    entry = 100.0
    bid = 99.97
    track = PositionTrack(
        entry_price=entry,
        state=STATE_OPEN,
        max_favorable_pct=0.00015,
        max_adverse_pct=0.0004,
        session_low_bid=bid,
        stale_review_count=2,
        review_lows=(bid, bid),
        setup_name="vwap_ema_reclaim",
        setup_context={},
    )
    review = evaluate_exit(
        track=track,
        snap=_Snap("SOLUSDT", bid),
        mom=_flat_mom(),
        econ=econ,
        config=config,
        trade_id="t-stall",
        hold_sec=620.0,
        executable_net_pct=-0.0004,
        profit_hit=False,
        exit_spread_ok=True,
        perform_review=True,
    )
    assert review.decision == "SELL", review.reason
    assert review.exit_reason == EXIT_EARLY_SCRATCH


def test_edge_buffer_defaults_trim_reachability_wall(monkeypatch):
    monkeypatch.delenv("SCALP_ENTRY_EDGE_BUFFER_PCT", raising=False)
    monkeypatch.delenv("SCALP_MIN_PROJECTED_SURPLUS_PCT", raising=False)
    econ = ScalpEconomics.from_env()
    assert econ.entry_edge_buffer_pct == pytest.approx(0.00025)
    assert econ.min_projected_surplus_pct == pytest.approx(0.00015)
    # With trimmed padding, ~0.38% expected move clears costs+target+surplus.
    ok_new, req = target_reachable(econ, spread_pct=0.0003, impact_pct=0.0001, expected_move_pct=0.0039)
    assert ok_new is True, f"req={req} buffer={econ.entry_edge_buffer_pct}"
    # Same move fails if padding is restored to the old 5bp/3bp wall.
    monkeypatch.setenv("SCALP_ENTRY_EDGE_BUFFER_PCT", "0.0005")
    monkeypatch.setenv("SCALP_MIN_PROJECTED_SURPLUS_PCT", "0.0003")
    econ_old = ScalpEconomics.from_env()
    ok_old, _ = target_reachable(econ_old, spread_pct=0.0003, impact_pct=0.0001, expected_move_pct=0.0039)
    assert ok_old is False


def test_sol_symbol_notional_cap(monkeypatch):
    monkeypatch.setenv("SCALP_MAX_NOTIONAL_PAPER", "150")
    monkeypatch.setenv("SCALP_SYMBOL_NOTIONAL_CAPS_JSON", '{"SOLUSDT":50}')
    cfg = ScalpConfig.from_env()
    assert cfg.notional_cap_for_symbol("SOLUSDT") == pytest.approx(50.0)
    assert cfg.notional_cap_for_symbol("BTCUSDT") == pytest.approx(150.0)
    assert cfg.notional_cap_for_symbol("SOL/USDT") == pytest.approx(50.0)
