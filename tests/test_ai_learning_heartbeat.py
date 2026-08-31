"""Heartbeat math and thesis-valid capture (data only, no strategy changes)."""

import os
import sqlite3
import tempfile

from backend.config.trading_economics import ESTIMATED_ROUNDTRIP_COST, TAKER_FEE
from backend.services.ai_learning_ingestion import (
    HEARTBEAT_CALC_VERSION,
    _compute_fee_estimate,
    _compute_gross_unrealized_pct,
    _compute_gross_unrealized_usd,
    _compute_mark_sell_net_usd,
    _resolve_thesis_valid,
    record_position_heartbeat,
)
from backend.services.day_trade_thesis import SETUP_HTF_TREND_PULLBACK


def test_gross_unrealized_pct_losing_long():
    pct = _compute_gross_unrealized_pct(entry_price=100.0, mark=99.5)
    assert abs(pct - (-0.005)) < 1e-12  # -0.5% as decimal fraction


def test_gross_unrealized_pct_winning_long():
    pct = _compute_gross_unrealized_pct(entry_price=100.0, mark=100.25)
    assert abs(pct - 0.0025) < 1e-12  # +0.25% as decimal fraction


def test_gross_unrealized_usd_losing_long():
    pnl = _compute_gross_unrealized_usd(entry_price=100.0, mark=99.5, quantity=10.0)
    assert abs(pnl - (-5.0)) < 1e-9


def test_net_unrealized_pct_subtracts_roundtrip_cost():
    gross = _compute_gross_unrealized_pct(entry_price=100.0, mark=100.25)
    net = gross - float(ESTIMATED_ROUNDTRIP_COST)
    assert net < gross


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


def test_fee_estimate_includes_entry_and_sell_side():
    fee = _compute_fee_estimate(mark=100.0, quantity=10.0, entry_fee=0.20)
    assert abs(fee - (0.20 + 10.0 * 100.0 * TAKER_FEE)) < 1e-9


def test_record_position_heartbeat_persists_calc_v2_fields():
    with tempfile.TemporaryDirectory() as tmp:
        db_path = os.path.join(tmp, "hb_test.db")
        record_position_heartbeat(
            symbol="BTC/USDT",
            trade_id="test_trade_1",
            entry_price=100.0,
            mark=99.5,
            entry_time_epoch=1_700_000_000.0,
            quantity=10.0,
            entry_fee=0.20,
            db_path=db_path,
        )
        with sqlite3.connect(db_path) as conn:
            row = conn.execute(
                """
                SELECT entry_price, mark, unrealized_pct, net_unrealized_pct,
                       quantity, gross_unrealized_pnl, net_unrealized_pnl,
                       fee_estimate, would_sell_now_net_usd, heartbeat_calc_version
                FROM ai_position_heartbeats WHERE trade_id='test_trade_1'
                """
            ).fetchone()
        assert row is not None
        ep, mk, up, nup, qty, gross_usd, net_usd, _fee, ws, ver = row
        assert ep == 100.0 and mk == 99.5 and qty == 10.0
        assert abs(up - (-0.005)) < 1e-12
        assert abs(nup - (-0.005 - float(ESTIMATED_ROUNDTRIP_COST))) < 1e-12
        assert abs(gross_usd - (-5.0)) < 1e-9
        assert abs(net_usd - ws) < 1e-9
        assert net_usd < gross_usd
        assert ver == HEARTBEAT_CALC_VERSION == 2
