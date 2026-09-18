"""Decision → intent → reservation → order → venue fill → position identity."""

from __future__ import annotations

import json
import time

from backend.services.day_entry_reservations import (
    STATUS_CONSUMED,
    consume_reservation,
    create_reservation,
    expire_stale,
    release_reservation,
    reservation_status,
)
from backend.services.day_trailing_buy_store import create_intent, load_intent, mark_order_accepted, mark_terminal
from backend.services.live_order_identity import extract_identity, record_fill


def test_info_fills_capture_venue_trade_ids_when_trades_list_is_empty():
    raw = {
        "id": "1587098371",
        "clientOrderId": "mystic_day_eth_1",
        "filled": 0.0236,
        "average": 2396.57,
        "timestamp": 1788600000000,
        "info": {
            "fills": [
                {"price": "2396.57", "qty": "0.0236", "commission": "0.0000236", "commissionAsset": "ETH", "tradeId": 991122},
            ]
        },
    }
    ident = extract_identity(raw, symbol="ETH/USDT", side="BUY", mystic_trade_id="t1", intent_id="i1")
    assert ident.fill_ids == ["991122"]
    assert ident.exchange_order_id == "1587098371"


def test_fetched_empty_fills_do_not_erase_create_reply_ids():
    from backend.services.protected_limit_execution import _merge_info_preserving_fills

    original = {"status": "FILLED", "fills": [{"tradeId": 7, "qty": "1"}]}
    fetched = {"status": "FILLED", "executedQty": "1"}
    merged = _merge_info_preserving_fills(original, fetched)
    assert merged["fills"][0]["tradeId"] == 7


def test_mark_order_accepted_never_blanks_known_identifiers(tmp_path):
    db = str(tmp_path / "id.db")
    ok, _, row = create_intent(
        db,
        fields={
            "decision_id": "dec-1",
            "symbol": "BTC/USDT",
            "arm_ask": 95000.0,
            "arm_bid": 94990.0,
            "arm_midpoint": 94995.0,
            "round_trip_cost_bps": 22.0,
            "spread_bps": 1.0,
            "required_improvement_bps": 3.0,
            "rebound_bps": 2.0,
            "min_dip_bps": 8.0,
            "expires_at": time.time() + 600,
            "reservation_id": "res_1",
            "quantity": 0.001,
        },
    )
    assert ok
    mark_order_accepted(db, row["intent_id"], order_id="10", fill_id="55", trade_id="mystic_1")
    mark_order_accepted(db, row["intent_id"], order_id="10")
    loaded = load_intent(db, row["intent_id"])
    assert loaded["order_id"] == "10"
    assert loaded["fill_id"] == "55"
    assert loaded["trade_id"] == "mystic_1"


def test_filled_reservation_is_consumed_and_cannot_expire(tmp_path):
    db = str(tmp_path / "res.db")
    ok, reason, rid = create_reservation(db, decision_id="dec-eth", symbol="ETH/USDT", notional_usd=50.0, ttl_sec=0.01)
    assert ok, reason
    assert consume_reservation(db, reservation_id=rid) is True
    assert reservation_status(db, rid) == STATUS_CONSUMED
    assert consume_reservation(db, reservation_id=rid) is False
    time.sleep(0.02)
    expire_stale(db)
    assert reservation_status(db, rid) == STATUS_CONSUMED
    release_reservation(db, reservation_id=rid, reason="EXPIRED")
    assert reservation_status(db, rid) == STATUS_CONSUMED


def test_live_fill_row_is_not_a_paper_trade(tmp_path):
    db = str(tmp_path / "fills.db")
    ident = extract_identity(
        {
            "id": "99",
            "clientOrderId": "mystic_live",
            "filled": 0.01,
            "average": 100.0,
            "timestamp": 1788600000000,
            "trades": [{"id": "t99", "order": "99"}],
            "info": {"fills": [{"tradeId": "t99"}]},
        },
        symbol="SOL/USDT",
        side="BUY",
        mystic_trade_id="mystic_sol_1",
        intent_id="int_1",
        decision_id="dec_1",
    )
    assert record_fill(db, ident) is True
    import sqlite3

    conn = sqlite3.connect(db)
    tables = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    assert "live_exchange_fills" in tables
    assert "paper_trades" not in tables
    mode_like = conn.execute("SELECT mystic_trade_id FROM live_exchange_fills").fetchone()[0]
    assert mode_like.startswith("mystic_")
    conn.close()


def test_position_identity_fields_exist_on_open_position():
    from backend.services.portfolio_engine import OpenPosition

    pos = OpenPosition(
        symbol="XRP/USDT",
        quantity=10.0,
        entry_price=0.62,
        entry_time=1.0,
        trade_id="t1",
        stop_price=0.6138,
        take_profit_1_price=0.62868,
        take_profit_2_price=0.633,
    )
    assert pos.entry_decision_id == ""
    assert pos.entry_intent_id == ""
    assert pos.entry_reservation_id == ""
    assert pos.entry_order_id == ""
    assert pos.entry_client_order_id == ""
    assert json.loads(pos.entry_fill_ids_json) == []
