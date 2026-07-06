"""Heartbeat math and thesis-valid capture (data only, no strategy changes)."""

from backend.services.ai_learning_ingestion import (
    _compute_mark_sell_net_usd,
    _resolve_thesis_valid,
)
from backend.services.day_trade_thesis import SETUP_HTF_TREND_PULLBACK


def test_mark_sell_net_usd_losing_position():
    # entry 100, mark 99.5, qty 10, entry_fee 0.20
    net = _compute_mark_sell_net_usd(
        entry_price=100.0,
        mark=99.5,
        quantity=10.0,
        entry_fee=0.20,
    )
    # proceeds = 995 - sell_fee; entry_cost = 1000.20
    assert net < 0
    assert abs(net - (-5.399)) < 0.05


def test_mark_sell_net_usd_winning_position():
    net = _compute_mark_sell_net_usd(
        entry_price=100.0,
        mark=101.0,
        quantity=10.0,
        entry_fee=0.20,
    )
    assert net > 0
    assert abs(net - 9.598) < 0.05


def test_mark_sell_net_usd_not_fraction_of_notional_bug():
    """Old formula (notional * net_pct) mis-scaled; true PnL is (mark-entry)*qty - fees."""
    entry, mark, qty = 100.0, 100.25, 25.0
    net = _compute_mark_sell_net_usd(entry_price=entry, mark=mark, quantity=qty, entry_fee=0.5)
    gross = (mark - entry) * qty
    assert abs(net - gross) > 1.0  # fees pull it below gross
    assert net < gross


def test_resolve_thesis_valid_stop_not_breached():
    valid = _resolve_thesis_valid(
        thesis_valid=None,
        entry_thesis=SETUP_HTF_TREND_PULLBACK,
        thesis_score=0.7,
        thesis_invalid_level=95.0,
        thesis_target_level=105.0,
        entry_vwap=100.0,
        entry_price=100.0,
        mark=98.0,
    )
    assert valid is True


def test_resolve_thesis_valid_stop_breached():
    valid = _resolve_thesis_valid(
        thesis_valid=None,
        entry_thesis=SETUP_HTF_TREND_PULLBACK,
        thesis_score=0.7,
        thesis_invalid_level=95.0,
        thesis_target_level=105.0,
        entry_vwap=100.0,
        entry_price=100.0,
        mark=94.0,
    )
    assert valid is False


def test_resolve_thesis_valid_no_thesis():
    valid = _resolve_thesis_valid(
        thesis_valid=None,
        entry_thesis="",
        thesis_score=0.0,
        thesis_invalid_level=0.0,
        thesis_target_level=0.0,
        entry_vwap=0.0,
        entry_price=100.0,
        mark=100.0,
    )
    assert valid is None
