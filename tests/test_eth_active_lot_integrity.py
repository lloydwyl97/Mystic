"""Captured ETH 1587754176 / live 1587893573 quantity and reservation integrity."""

from __future__ import annotations

import sqlite3
from decimal import Decimal

from backend.services.day_entry_reservations import (
    STATUS_CONSUMED,
    consume_reservation,
    correct_filled_reservation_to_consumed,
    create_reservation,
    ensure_reservation_schema,
    reservation_status,
)
from backend.services.live_account_basis import persist_operational_json
from backend.services.live_exchange_equity import load_protected_preexisting_dust
from backend.services.live_fill_economics import (
    ETH_LOT_CORRECTION_ID as ECON_ID,
)
from backend.services.live_fill_economics import (
    active_lot_keeps_booked_qty,
    apply_eth_lot_integrity_correction,
    apply_live_buy_economics,
    apply_live_sell_economics,
    eth_captured_buy_1587754176,
    eth_current_buy_1587893573,
    extract_live_commission,
    plan_sell_quantity,
)
from backend.services.live_order_identity import extract_identity


def test_gross_minus_base_fee_is_net_once():
    named = eth_captured_buy_1587754176()
    order = {
        "filled": 0.0227,
        "average": 2638.42,
        "fee": {"cost": 0.00000454, "currency": "ETH"},
        "trades": [{"commission": "0.00000454", "commissionAsset": "ETH"}],
        "info": {
            "commission": "0.00000454",
            "commissionAsset": "ETH",
            "fills": [{"commission": "0.00000454", "commissionAsset": "ETH"}],
        },
    }
    comm = extract_live_commission(order, symbol="ETH/USDT", fill_price=2638.42)
    assert comm.base_qty_reduction == 0.00000454
    qty, fee, cash = apply_live_buy_economics(
        filled_qty=0.0227,
        fill_price=2638.42,
        modeled_fee=0.0359352804,
        commission=comm,
    )
    assert abs(qty - 0.02269546) < 1e-12
    assert qty == float(Decimal(named["net_credited"]))
    assert fee != 0.0359352804
    assert abs(cash - 59.892134) < 1e-8


def test_filled_reservation_becomes_consumed(tmp_path):
    db = str(tmp_path / "res.db")
    ensure_reservation_schema(db)
    ok, _reason, rid = create_reservation(
        db,
        decision_id="day_ETHUSDT_1789760671103",
        symbol="ETH/USDT",
        notional_usd=59.95,
        risk_usd=0.0,
        sleeve="ACTIVE",
    )
    assert ok
    assert consume_reservation(db, reservation_id=rid) is True
    assert reservation_status(db, rid) == STATUS_CONSUMED
    assert consume_reservation(db, reservation_id=rid) is False


def test_released_filled_reservation_is_corrected_consumed(tmp_path):
    db = str(tmp_path / "res2.db")
    ensure_reservation_schema(db)
    conn = sqlite3.connect(db)
    conn.execute(
        "INSERT INTO day_entry_reservations(reservation_id, decision_id, symbol, notional_usd, risk_usd, sleeve, status, created_at, expires_at, updated_at) VALUES (?,?,?,?,?,?,?,?,?,?)",
        ("res_b1271f7a936e4a82", "day_ETHUSDT_1789760671103", "ETH/USDT", 59.95, 0.0, "ACTIVE", "RELEASED", 1, 2, 3),
    )
    conn.commit()
    conn.close()
    first = correct_filled_reservation_to_consumed(db, reservation_id="res_b1271f7a936e4a82")
    second = correct_filled_reservation_to_consumed(db, reservation_id="res_b1271f7a936e4a82")
    assert first["changed"] is True
    assert first["previous_status"] == "RELEASED"
    assert first["status"] == STATUS_CONSUMED
    assert second["changed"] is False
    assert reservation_status(db, "res_b1271f7a936e4a82") == STATUS_CONSUMED


def test_active_pre_repair_lot_reconciles_without_second_trade(tmp_path):
    db = str(tmp_path / "eth.db")
    persist_operational_json(db, "seed", {"ok": True})
    conn = sqlite3.connect(db)
    conn.execute(
        """
        CREATE TABLE paper_trades (
            trade_id TEXT, side TEXT, symbol TEXT, quantity REAL, price REAL,
            fees_paid REAL, pnl REAL, pnl_usd_net REAL, order_id TEXT, mode TEXT,
            is_synthetic INTEGER DEFAULT 0
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE portfolio_engine_positions (
            symbol TEXT, quantity REAL, status TEXT, trade_id TEXT, entry_price REAL,
            entry_order_id TEXT, stop_price REAL, take_profit_1_price REAL,
            trailing_stop_price REAL, highest_price REAL, thesis_json TEXT
        )
        """
    )
    conn.execute(
        "INSERT INTO paper_trades VALUES (?,?,?,?,?,?,?,?,?,?,?)",
        ("mystic_ETH/USDT_1789761329846", "BUY", "ETH/USDT", 0.02268638, 2638.42, 0.0359352804, None, None, "1587754176", "live", 0),
    )
    conn.execute(
        "INSERT INTO paper_trades VALUES (?,?,?,?,?,?,?,?,?,?,?)",
        ("mystic_sell_ETH/USDT_1789768531658", "SELL", "ETH/USDT", 0.0226, 2626.95, 0.01187381, -0.3068942642725645, -0.3068942642725645, "1587872124", "live", 0),
    )
    conn.execute(
        "INSERT INTO portfolio_engine_positions VALUES (?,?,?,?,?,?,?,?,?,?,?)",
        (
            "ETH/USDT",
            0.0167,
            "ACTIVE",
            "mystic_ETH/USDT_1789770820249",
            2629.51,
            "1587893573",
            2603.2149,
            2666.32314,
            2603.2149,
            2629.755,
            '{"max_hold_min":300,"trail_pct":0.0025}',
        ),
    )
    conn.commit()
    conn.close()
    first = apply_eth_lot_integrity_correction(db, exchange_eth="0.01678348")
    second = apply_eth_lot_integrity_correction(db, exchange_eth="0.01678348")
    assert first["applied"] is True
    assert second["applied"] is True
    with sqlite3.connect(db) as c:
        n = c.execute(
            "SELECT COUNT(*) FROM live_accounting_corrections WHERE correction_id=?",
            (ECON_ID,),
        ).fetchone()[0]
        pos = c.execute("SELECT quantity, entry_price, stop_price, trade_id FROM portfolio_engine_positions WHERE symbol='ETH/USDT'").fetchone()
        buy = c.execute("SELECT quantity, fees_paid FROM paper_trades WHERE order_id='1587754176'").fetchone()
        trades = c.execute("SELECT COUNT(*) FROM paper_trades").fetchone()[0]
    assert n == 1
    assert trades == 2
    assert abs(pos[0] - 0.01659668) < 1e-12
    assert pos[1] == 2629.51
    assert abs(pos[2] - 2603.2149) < 1e-8
    assert pos[3] == "mystic_ETH/USDT_1789770820249"
    assert abs(buy[0] - 0.02269546) < 1e-12
    assert load_protected_preexisting_dust(db, "ETH/USDT") == Decimal("0.00018680")


def test_preexisting_dust_stays_separate_and_sell_cannot_borrow():
    live = eth_current_buy_1587893573()
    net = Decimal(live["net_credited"])
    planned = plan_sell_quantity(
        net_active_qty=net,
        exchange_free_qty="0.01678348",
        protected_dust_qty="0.00018680",
        qty_step="0.0001",
    )
    assert planned.sellable == Decimal("0.0165")
    assert planned.borrowed_from_dust == Decimal("0")
    assert planned.protected_dust == Decimal("0.00018680")
    assert planned.residual == net - Decimal("0.0165")
    assert active_lot_keeps_booked_qty(booked_qty=net, exchange_qty="0.01678348") is True
    assert active_lot_keeps_booked_qty(booked_qty="0.0167", exchange_qty="0.0165") is False


def test_restart_does_not_duplicate_correction_or_dust(tmp_path):
    db = str(tmp_path / "rst.db")
    persist_operational_json(db, "seed", {"ok": True})
    with sqlite3.connect(db) as conn:
        conn.execute("CREATE TABLE paper_trades (trade_id TEXT, side TEXT, quantity REAL, price REAL, fees_paid REAL, pnl REAL, pnl_usd_net REAL, order_id TEXT)")
        conn.execute(
            """
            CREATE TABLE portfolio_engine_positions (
                symbol TEXT, quantity REAL, status TEXT, trade_id TEXT, entry_price REAL,
                entry_order_id TEXT, stop_price REAL, take_profit_1_price REAL,
                trailing_stop_price REAL, highest_price REAL, thesis_json TEXT
            )
            """
        )
        conn.execute(
            "INSERT INTO portfolio_engine_positions VALUES (?,?,?,?,?,?,?,?,?,?,?)",
            ("ETH/USDT", 0.0167, "ACTIVE", "mystic_ETH/USDT_1789770820249", 2629.51, "1587893573", 2603.21, 2666.32, 2603.21, 2629.75, "{}"),
        )
        conn.commit()
    apply_eth_lot_integrity_correction(db, exchange_eth="0.01678348")
    ensure_reservation_schema(db)
    with sqlite3.connect(db) as conn:
        conn.execute(
            "INSERT INTO day_entry_reservations(reservation_id, decision_id, symbol, notional_usd, risk_usd, sleeve, status, created_at, expires_at, updated_at) VALUES (?,?,?,?,?,?,?,?,?,?)",
            ("res_9e7ebc24ecf34ffe", "day_ETHUSDT_1789770008654", "ETH/USDT", 43.7, 0.0, "ACTIVE", "RELEASED", 1, 2, 3),
        )
        conn.commit()
    replay = apply_eth_lot_integrity_correction(db, exchange_eth="0.01678348")
    assert replay["live_reservation"]["previous_status"] == "RELEASED"
    assert replay["live_reservation"]["status"] == STATUS_CONSUMED
    with sqlite3.connect(db) as conn:
        assert conn.execute("SELECT COUNT(*) FROM live_accounting_corrections").fetchone()[0] == 1
        assert conn.execute("SELECT COUNT(*) FROM portfolio_engine_positions").fetchone()[0] == 1
        qty = conn.execute("SELECT quantity FROM portfolio_engine_positions").fetchone()[0]
        assert reservation_status(db, "res_9e7ebc24ecf34ffe") == STATUS_CONSUMED
    assert abs(qty - 0.01659668) < 1e-12
    assert load_protected_preexisting_dust(db, "ETH/USDT") == Decimal("0.00018680")


def test_eventual_sell_uses_actual_exchange_commission():
    comm = extract_live_commission(
        {"fee": {"cost": 0.01187381, "currency": "USDT"}},
        symbol="ETH/USDT",
        fill_price=2626.95,
    )
    fee, _proceeds = apply_live_sell_economics(
        quantity=0.0226,
        fill_price=2626.95,
        modeled_fee=0.0359,
        commission=comm,
    )
    assert abs(fee - 0.01187381) < 1e-9
    assert fee != 0.0359
    ident = extract_identity(
        {"id": "1587754176", "filled": 0.0227, "average": 2638.42, "timestamp": 1789761330207},
        symbol="ETH/USDT",
        side="BUY",
        fee_amount=0.0359352804,
        fee_items=[{"amount": 0.00000454, "asset": "ETH", "usd": 0.01197843}],
        fee_from_exchange=True,
    )
    assert ident.fee_amount == 0.00000454
    assert ident.fee_asset == "ETH"
