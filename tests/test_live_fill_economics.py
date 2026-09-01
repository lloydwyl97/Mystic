"""Live Binance commission accounting — quote, base, mixed; no double-count."""

import sqlite3
import tempfile

import pytest

from backend.services.live_fill_economics import (
    apply_live_buy_economics,
    apply_live_sell_economics,
    extract_live_commission,
    live_round_trip_net,
    mode_scoped_equity_views,
    sum_dust_adjustment_pnl,
    sum_realized_pnl_by_mode,
)


def test_quote_commission_usdt():
    order = {
        "info": {"fills": [{"commission": "0.01307088", "commissionAsset": "USDT"}]},
    }
    comm = extract_live_commission(order, symbol="BTCUSDT", fill_price=77802.87)
    assert comm.fee_from_exchange is True
    assert comm.base_qty_reduction == 0.0
    assert abs(comm.usd - 0.01307088) < 1e-9
    assert abs(comm.quote_commission_usd - 0.01307088) < 1e-9


def test_base_commission_btc():
    order = {
        "trades": [{"commission": "0.00000017", "commissionAsset": "BTC"}],
    }
    px = 77137.14
    comm = extract_live_commission(order, symbol="BTC/USDT", fill_price=px)
    assert comm.fee_from_exchange is True
    assert abs(comm.base_qty_reduction - 0.00000017) < 1e-12
    assert abs(comm.usd - 0.00000017 * px) < 1e-9
    assert comm.quote_commission_usd == 0.0


def test_mixed_commission_assets():
    order = {
        "info": {
            "fills": [
                {"commission": "0.00000017", "commissionAsset": "BTC"},
                {"commission": "0.01307088", "commissionAsset": "USDT"},
            ]
        }
    }
    px = 77137.14
    comm = extract_live_commission(order, symbol="BTCUSDT", fill_price=px)
    assert comm.fee_from_exchange is True
    assert abs(comm.base_qty_reduction - 0.00000017) < 1e-12
    assert abs(comm.usd - (0.00000017 * px + 0.01307088)) < 1e-8


def test_buy_base_fee_reduces_inventory_not_double_cash():
    px = 77137.14
    filled = 0.00084
    comm = extract_live_commission(
        {"fee": {"cost": 0.00000017, "currency": "BTC"}},
        symbol="BTCUSDT",
        fill_price=px,
    )
    modeled = filled * px * 0.001
    qty, fee, cash = apply_live_buy_economics(
        filled_qty=filled,
        fill_price=px,
        modeled_fee=modeled,
        commission=comm,
    )
    assert abs(qty - (filled - 0.00000017)) < 1e-12
    assert abs(fee - 0.00000017 * px) < 1e-9
    assert abs(cash - filled * px) < 1e-8
    assert fee < modeled


def test_gross_vs_net_no_double_count():
    qty = 0.00084
    entry = 77137.14
    exit_px = 77802.87
    buy_fee = 0.00000017 * entry
    sell_fee = 0.01307088
    out = live_round_trip_net(
        quantity=qty,
        entry_price=entry,
        exit_price=exit_px,
        entry_commission_usd=buy_fee,
        exit_commission_usd=sell_fee,
    )
    gross = qty * (exit_px - entry)
    assert abs(out["gross_price_pnl"] - gross) < 1e-8
    assert abs(out["net_realized_pnl"] - (gross - buy_fee - sell_fee)) < 1e-8
    # Fill prices already include bid/ask; do not subtract extra slippage.
    assert out["exchange_commission_usd"] == round(buy_fee + sell_fee, 8)


def test_sell_uses_exchange_fee_not_model():
    comm = extract_live_commission(
        {"fee": {"cost": 0.01307088, "currency": "USDT"}},
        symbol="BTCUSDT",
        fill_price=77802.87,
    )
    modeled = 0.00084 * 77802.87 * 0.001
    fee, proceeds = apply_live_sell_economics(
        quantity=0.00084,
        fill_price=77802.87,
        modeled_fee=modeled,
        commission=comm,
    )
    assert abs(fee - 0.01307088) < 1e-9
    assert abs(proceeds - (0.00084 * 77802.87 - 0.01307088)) < 1e-8


def test_paper_model_used_when_no_exchange_fee():
    comm = extract_live_commission({}, symbol="BTCUSDT", fill_price=100.0)
    assert comm.fee_from_exchange is False
    qty, fee, cash = apply_live_buy_economics(
        filled_qty=1.0,
        fill_price=100.0,
        modeled_fee=0.1,
        commission=comm,
    )
    assert qty == 1.0
    assert fee == 0.1
    assert cash == 100.1


def _make_paper_trades_db(rows: list[dict]) -> str:
    """Create a temp SQLite with paper_trades rows, return path."""
    tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
    tmp.close()
    conn = sqlite3.connect(tmp.name)
    conn.execute("""
        CREATE TABLE paper_trades (
            id INTEGER PRIMARY KEY,
            side TEXT, pnl REAL, mode TEXT,
            is_synthetic INTEGER DEFAULT 0,
            exit_type TEXT
        )
    """)
    for r in rows:
        conn.execute(
            "INSERT INTO paper_trades (side, pnl, mode, is_synthetic, exit_type) VALUES (?,?,?,?,?)",
            (r["side"], r["pnl"], r["mode"], r.get("is_synthetic", 0), r.get("exit_type")),
        )
    conn.commit()
    conn.close()
    return tmp.name


def test_dust_writeoff_excluded_from_live_pnl():
    """DUST_WRITEOFF rows must not count toward live realized P&L."""
    path = _make_paper_trades_db(
        [
            {"side": "SELL", "pnl": -1.30, "mode": "live", "exit_type": None},
            {"side": "SELL", "pnl": -7.33, "mode": "live", "exit_type": "DUST_WRITEOFF"},
            {"side": "SELL", "pnl": 973.98, "mode": "paper", "exit_type": None},
        ]
    )
    live = sum_realized_pnl_by_mode(path, mode="live")
    paper = sum_realized_pnl_by_mode(path, mode="paper")
    # Dust must not contaminate live P&L
    assert abs(live - (-1.30)) < 1e-6, f"live={live} (expected -1.30, dust must be excluded)"
    assert abs(paper - 973.98) < 1e-6, f"paper={paper}"


def test_paper_dust_also_excluded():
    """DUST_WRITEOFF rows for paper mode are also excluded."""
    path = _make_paper_trades_db(
        [
            {"side": "SELL", "pnl": 10.0, "mode": "paper", "exit_type": None},
            {"side": "SELL", "pnl": -0.50, "mode": "paper", "exit_type": "DUST_WRITEOFF"},
        ]
    )
    paper = sum_realized_pnl_by_mode(path, mode="paper")
    assert abs(paper - 10.0) < 1e-6, f"paper={paper} (dust excluded)"


def test_dust_adjustment_sum_is_separate():
    path = _make_paper_trades_db(
        [
            {"side": "SELL", "pnl": -1.30, "mode": "live", "exit_type": None},
            {"side": "SELL", "pnl": -7.33, "mode": "live", "exit_type": "DUST_WRITEOFF"},
        ]
    )
    assert abs(sum_dust_adjustment_pnl(path) - (-7.33)) < 1e-6
    assert abs(sum_realized_pnl_by_mode(path, mode="live") - (-1.30)) < 1e-6


def test_live_performance_equity_is_cash_plus_positions_not_paper():
    views = mode_scoped_equity_views(
        account_equity=236.15,
        principal=236.95,
        live_realized=-1.34,
        paper_realized=973.98,
        unrealized=0.0,
        dust_adjustment=-10.13,
        is_live=True,
    )
    assert views["performance_equity"] == pytest.approx(236.15)
    assert views["cash_plus_positions_equity"] == pytest.approx(236.15)
    assert views["paper_realized_pnl"] == pytest.approx(973.98)
    assert views["live_realized_pnl"] == pytest.approx(-1.34)
    assert views["performance_equity_uses_live_account"] is True
    # Paper profit must not appear as live equity
    assert views["performance_equity"] < 300.0


def test_paper_mode_performance_equity_uses_paper_realized():
    views = mode_scoped_equity_views(
        account_equity=10000.0,
        principal=10000.0,
        live_realized=0.0,
        paper_realized=50.0,
        unrealized=10.0,
        dust_adjustment=0.0,
        is_live=False,
    )
    assert views["performance_equity"] == pytest.approx(10060.0)
