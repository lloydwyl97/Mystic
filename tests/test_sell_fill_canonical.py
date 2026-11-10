"""A protected SELL is booked from venue commission, never from a missing one."""

from __future__ import annotations

import sqlite3

import pytest

from backend.services.day_mandatory_exit_execution import _combine_orders, run_mandatory_exit_ioc_loop
from backend.services.live_fill_economics import (
    align_proven_sell_qty,
    apply_live_sell_economics,
    extract_live_commission,
    merge_venue_trades_into_order,
    proven_sell_base_qty,
    proven_sell_quote,
    reconcile_lot_decision,
    sell_attempt_disposition,
    sell_commission_known,
)
from backend.services.protected_limit_execution import _LIVE_ORDER_OUTCOME, LIVE_OUTCOME_UNCONFIRMED
from backend.services.strategy_order_intent import (
    _finish_venue_lifecycle,
    allocate_intent,
    close_sell_order_ids,
    has_unbooked_executed_sell,
    mark_exchange_order,
    record_sell_attempt_result,
    reserved_unbooked_sell_qty,
    sell_quantity_state,
    unbooked_protective_sells,
)

MODELED = 0.0  # protected-limit maker schedule


def _full_create() -> dict:
    return {
        "id": "9001",
        "clientOrderId": "abc-i",
        "status": "closed",
        "filled": 0.4,
        "amount": 1.0,
        "average": 100.5,
        "cost": 40.2,
        "info": {
            "orderId": "9001",
            "executedQty": "0.40000000",
            "cummulativeQuoteQty": "40.20000000",
            "status": "FILLED",
            "fills": [
                {"price": "100.00000000", "qty": "0.15000000", "commission": "0.01500000", "commissionAsset": "USDT", "tradeId": "1"},
                {"price": "101.00000000", "qty": "0.10000000", "commission": "0.01010000", "commissionAsset": "USDT", "tradeId": "2"},
                {"price": "100.66666667", "qty": "0.15000000", "commission": "0.01510000", "commissionAsset": "USDT", "tradeId": "3"},
            ],
        },
    }


def test_full_create_is_canonical_without_a_reread():
    order = _full_create()
    assert sell_commission_known(order) is True
    assert proven_sell_base_qty(order) == pytest.approx(0.4)
    assert proven_sell_quote(order) == pytest.approx(0.15 * 100 + 0.10 * 101 + 0.15 * 100.66666667)
    comm = extract_live_commission(order, symbol="SOL/USDT", fill_price=100.5)
    assert comm.fee_from_exchange is True
    assert comm.usd == pytest.approx(0.015 + 0.0101 + 0.0151)
    fee, proceeds = apply_live_sell_economics(quantity=0.4, fill_price=100.5, modeled_fee=MODELED, commission=comm)
    assert fee == pytest.approx(comm.usd)
    assert proceeds == pytest.approx(proven_sell_quote(order) - comm.usd)
    assert sell_attempt_disposition(order, "") == "filled"


def test_summary_without_commission_is_not_booked():
    order = {
        "id": "9002",
        "filled": 0.4,
        "average": 100.0,
        "cost": 40.0,
        "status": "expired",
        "info": {"executedQty": "0.40000000", "cummulativeQuoteQty": "40.00000000", "status": "EXPIRED"},
    }
    assert sell_commission_known(order) is False
    assert sell_attempt_disposition(order, "") == "fee_pending"
    assert extract_live_commission(order, symbol="SOL/USDT", fill_price=100.0).fee_from_exchange is False


def test_partial_uses_venue_executed_qty_not_the_larger_summary():
    order = {
        "id": "9003",
        "filled": 0.00059,
        "average": 75907.45,
        "cost": 22.77078,
        "status": "closed",
        "_mystic_partial_fill": True,
        "info": {"executedQty": "0.00030000", "cummulativeQuoteQty": "22.77078000"},
        "fee": {"cost": 0.00455416, "currency": "USDT"},
    }
    aligned = align_proven_sell_qty(order)
    assert aligned["filled"] == pytest.approx(0.0003)
    assert sell_commission_known(aligned) is True
    comm = extract_live_commission(aligned, symbol="BTC/USDT", fill_price=75907.45)
    assert comm.usd == pytest.approx(0.00455416)


def test_explicit_zero_commission_is_known_and_not_replaced():
    order = {
        "id": "9004",
        "filled": 1.0,
        "info": {
            "executedQty": "1.0",
            "fills": [{"qty": "1.0", "price": "10", "commission": "0", "commissionAsset": "USDT", "tradeId": "z"}],
        },
    }
    assert sell_commission_known(order) is True
    comm = extract_live_commission(order, symbol="SOL/USDT", fill_price=10.0)
    assert comm.fee_from_exchange is True
    assert comm.usd == pytest.approx(0.0)
    fee, _proceeds = apply_live_sell_economics(quantity=1.0, fill_price=10.0, modeled_fee=0.02, commission=comm)
    assert fee == pytest.approx(0.0)


def test_quote_base_and_other_fee_assets():
    def order(asset: str, commission: str) -> dict:
        return {
            "id": "1",
            "filled": 1.0,
            "info": {
                "executedQty": "1",
                "cummulativeQuoteQty": "100",
                "fills": [{"qty": "1", "price": "100", "commission": commission, "commissionAsset": asset, "tradeId": "t"}],
            },
        }

    quote = extract_live_commission(order("USDT", "0.02"), symbol="SOL/USDT", fill_price=100.0)
    assert quote.quote_commission_usd == pytest.approx(0.02)
    assert quote.base_qty_reduction == pytest.approx(0.0)
    fee, proceeds = apply_live_sell_economics(quantity=1.0, fill_price=100.0, modeled_fee=0.0, commission=quote)
    assert fee == pytest.approx(0.02)
    assert proceeds == pytest.approx(99.98)

    base = extract_live_commission(order("SOL", "0.001"), symbol="SOL/USDT", fill_price=100.0)
    assert base.base_qty_reduction == pytest.approx(0.001)
    assert base.usd == pytest.approx(0.1)

    other = extract_live_commission(order("BNB", "0.0001"), symbol="SOL/USDT", fill_price=100.0)
    assert other.fee_from_exchange is True
    assert other.unpriced_assets == ("BNB",)
    assert sell_commission_known(order("BNB", "0.0001")) is True


def test_mytrades_completes_a_summary_and_a_second_merge_is_a_no_op():
    summary = {
        "id": "77",
        "filled": 0.4,
        "info": {"executedQty": "0.4", "cummulativeQuoteQty": "40"},
    }
    trades = [
        {"trade_id": "11", "order_id": "77", "qty": "0.4", "price": "100", "commission": "0.04", "commission_asset": "USDT"},
    ]
    merged = merge_venue_trades_into_order(summary, trades)
    assert sell_commission_known(merged) is True
    assert extract_live_commission(merged, symbol="SOL/USDT", fill_price=100.0).usd == pytest.approx(0.04)
    again = merge_venue_trades_into_order(merged, [{"trade_id": "99", "qty": "0.4", "price": "1", "commission": "9", "commission_asset": "USDT"}])
    assert extract_live_commission(again, symbol="SOL/USDT", fill_price=100.0).usd == pytest.approx(0.04)


def test_chunk_without_a_fee_asset_does_not_invent_one():
    bare = {"id": "1", "filled": 1.0, "average": 10.0, "cost": 10.0, "status": "closed", "info": {"executedQty": "1", "orderId": "1"}}
    priced = {
        "id": "2",
        "filled": 1.0,
        "average": 10.0,
        "cost": 10.0,
        "status": "closed",
        "info": {"executedQty": "1", "orderId": "2", "fills": [{"qty": "1", "price": "10", "commission": "0.002", "commissionAsset": "USDT", "tradeId": "t"}]},
    }
    combined = _combine_orders([bare, priced], 2.0)
    assert all(not row.get("_mystic_synthetic") for row in combined["info"]["fills"])
    assert sell_commission_known(combined) is False
    assert proven_sell_base_qty(combined) == pytest.approx(2.0)


def test_zero_fill_releases_and_unconfirmed_keeps_the_id(tmp_path):
    db = str(tmp_path / "intents.db")
    cid = allocate_intent(db, engine_id="DAY_V2", trade_id="lot-1", symbol="SOL/USDT", side="SELL", quantity=1.0, limit_price=10.0)
    assert record_sell_attempt_result(db, client_order_id=f"{cid}-i", order={"id": "5", "filled": 0, "status": "expired", "info": {"executedQty": "0"}}, outcome="EXPIRED_UNFILLED") == "release"
    assert reserved_unbooked_sell_qty(db, "lot-1") == pytest.approx(0.0)

    cid2 = allocate_intent(db, engine_id="DAY_V2", trade_id="lot-2", symbol="SOL/USDT", side="SELL", quantity=0.6, limit_price=10.0)
    assert record_sell_attempt_result(db, client_order_id=cid2, order=None, outcome="ORDER_UNCONFIRMED") == "keep"
    assert reserved_unbooked_sell_qty(db, "lot-2") == pytest.approx(0.6)


def test_fee_pending_reserves_base_until_the_close_is_booked(tmp_path):
    db = str(tmp_path / "pending.db")
    cid = allocate_intent(db, engine_id="DAY_V2", trade_id="lot-3", symbol="SOL/USDT", side="SELL", quantity=1.0, limit_price=10.0)
    order = {
        "id": "55",
        "clientOrderId": f"{cid}-i",
        "filled": 0.4,
        "info": {"executedQty": "0.4", "cummulativeQuoteQty": "40"},
    }
    assert record_sell_attempt_result(db, client_order_id=cid, order=order, outcome="") == "fee_pending"
    assert reserved_unbooked_sell_qty(db, "lot-3") == pytest.approx(0.4)
    # A later attempt for the remainder is a new id. The pending fill stays one row.
    cid_next = allocate_intent(db, engine_id="DAY_V2", trade_id="lot-3", symbol="SOL/USDT", side="SELL", quantity=0.6, limit_price=10.0)
    assert cid_next != cid
    assert len(unbooked_protective_sells(db, "lot-3")) == 1
    close_sell_order_ids(db, "lot-3", ["55"])
    assert reserved_unbooked_sell_qty(db, "lot-3") == pytest.approx(0.6)
    with sqlite3.connect(db) as conn:
        status = conn.execute("SELECT status FROM strategy_order_intents WHERE exchange_order_id='55'").fetchone()[0]
    assert status == "CLOSED"


def test_same_client_id_is_not_booked_twice(tmp_path):
    db = str(tmp_path / "once.db")
    cid = allocate_intent(db, engine_id="SCALP_V2", trade_id="lot-4", symbol="BTC/USDT", side="SELL", quantity=0.01, limit_price=100.0)
    order = {"id": "88", "filled": 0.01, "info": {"executedQty": "0.01", "fills": [{"qty": "0.01", "price": "100", "commission": "0.001", "commissionAsset": "USDT"}]}}
    assert record_sell_attempt_result(db, client_order_id=cid, order=order, outcome="") == "filled"
    assert record_sell_attempt_result(db, client_order_id=cid, order=order, outcome="") == "filled"
    with sqlite3.connect(db) as conn:
        assert conn.execute("SELECT COUNT(*) FROM strategy_order_intents WHERE trade_id='lot-4'").fetchone()[0] == 1
        assert conn.execute("SELECT filled_qty FROM strategy_order_intents WHERE trade_id='lot-4'").fetchone()[0] == pytest.approx(0.01)


@pytest.mark.asyncio
async def test_unconfirmed_attempt_does_not_place_a_second_order():
    calls = {"n": 0}

    async def preflight(qty, impact):
        return type("P", (), {"passed": True, "executable_qty": qty, "quantity": qty, "protected_limit_price": 10.0, "reject_reason": ""})()

    async def place(qty, limit):
        calls["n"] += 1
        _LIVE_ORDER_OUTCOME.set(LIVE_OUTCOME_UNCONFIRMED)

    out = await run_mandatory_exit_ioc_loop(quantity=1.0, preflight=preflight, place_ioc=place, is_meaningful=lambda q: q > 0, max_attempts=6)
    assert calls["n"] == 1
    assert out.abandoned_reason == "venue_unconfirmed"
    assert out.filled_qty == 0.0


def test_reconcile_does_not_shrink_a_fee_pending_lot():
    assert reconcile_lot_decision(booked_qty=1.0, exchange_qty=0.6, reserved_qty=0.4) == "keep"
    assert reconcile_lot_decision(booked_qty=1.0, exchange_qty=0.6, reserved_qty=0.0) == "shrink"
    assert reconcile_lot_decision(booked_qty=1.0, exchange_qty=0.5, reserved_qty=0.4) == "shortfall_hold"
    assert reconcile_lot_decision(booked_qty=1.0, exchange_qty=1.0, reserved_qty=0.0) == "keep"


def test_two_fee_pending_fills_reserve_both_and_survive_a_booked_id(tmp_path):
    db = str(tmp_path / "two.db")
    first = allocate_intent(db, engine_id="DAY_V2", trade_id="lot", symbol="SOL/USDT", side="SELL", quantity=0.25, limit_price=10.0)
    record_sell_attempt_result(db, client_order_id=first, order={"id": "A", "filled": 0.25, "info": {"executedQty": "0.25", "cummulativeQuoteQty": "2.5"}}, outcome="")
    second = allocate_intent(db, engine_id="DAY_V2", trade_id="lot", symbol="SOL/USDT", side="SELL", quantity=0.30, limit_price=10.0)
    record_sell_attempt_result(db, client_order_id=second, order={"id": "B", "filled": 0.30, "info": {"executedQty": "0.30", "cummulativeQuoteQty": "3"}}, outcome="")
    state = sell_quantity_state(db, "lot", 1.0)
    assert state["executed_unbooked_qty"] == pytest.approx(0.55)
    assert state["available_to_sell_qty"] == pytest.approx(0.45)
    assert state["economic_open_qty"] == pytest.approx(0.45)
    assert state["ledger_qty"] == pytest.approx(1.0)
    with sqlite3.connect(db) as conn:
        conn.execute("CREATE TABLE paper_trades (order_id TEXT, side TEXT)")
        conn.execute("INSERT INTO paper_trades (order_id, side) VALUES ('A', 'SELL')")
        conn.commit()
    after = sell_quantity_state(db, "lot", 1.0)
    assert after["executed_unbooked_qty"] == pytest.approx(0.30)
    assert after["reserved_qty"] == pytest.approx(0.30)


def test_resting_partial_reserves_the_executed_and_the_open_remainder(tmp_path):
    db = str(tmp_path / "rest.db")
    cid = allocate_intent(db, engine_id="DAY_V2", trade_id="lot", symbol="SOL/USDT", side="SELL", quantity=0.6, limit_price=10.0)
    mark_exchange_order(db, cid, "R", status="PARTIAL")
    with sqlite3.connect(db) as conn:
        conn.execute("UPDATE strategy_order_intents SET filled_qty=0.2, filled_cost=2.0 WHERE client_order_id=?", (cid,))
        conn.commit()
    state = sell_quantity_state(db, "lot", 1.0)
    assert state["executed_unbooked_qty"] == pytest.approx(0.2)
    assert state["working_sell_qty"] == pytest.approx(0.4)
    assert state["available_to_sell_qty"] == pytest.approx(0.4)
    _finish_venue_lifecycle(db, {"id": "R", "clientOrderId": cid, "filled": 0.2, "status": "canceled", "info": {"executedQty": "0.2", "cummulativeQuoteQty": "2"}}, {"adopted": False})
    canceled = sell_quantity_state(db, "lot", 1.0)
    assert canceled["working_sell_qty"] == pytest.approx(0.0)
    assert canceled["executed_unbooked_qty"] == pytest.approx(0.2)
    assert canceled["available_to_sell_qty"] == pytest.approx(0.8)
    assert has_unbooked_executed_sell(db) is True


def test_partial_mytrades_does_not_become_a_close_and_empty_read_is_not_a_zero_fill():
    summary = {"id": "77", "filled": 0.4, "status": "expired", "info": {"executedQty": "0.4", "cummulativeQuoteQty": "40"}}
    partial = merge_venue_trades_into_order(summary, [{"trade_id": "1", "qty": "0.25", "price": "100", "commission": "0.025", "commission_asset": "USDT"}])
    assert proven_sell_base_qty(partial) == pytest.approx(0.4)
    assert sell_commission_known(partial) is False
    assert sell_attempt_disposition(partial, "") == "fee_pending"
    untouched = merge_venue_trades_into_order(summary, [])
    assert untouched["info"]["executedQty"] == "0.4"
    assert sell_attempt_disposition(untouched, "") == "fee_pending"
    complete = merge_venue_trades_into_order(
        summary,
        [
            {"trade_id": "1", "qty": "0.25", "price": "100", "commission": "0.025", "commission_asset": "USDT"},
            {"trade_id": "2", "qty": "0.15", "price": "100", "commission": "0.015", "commission_asset": "USDT"},
        ],
    )
    assert sell_commission_known(complete) is True
    assert extract_live_commission(complete, symbol="SOL/USDT", fill_price=100.0).usd == pytest.approx(0.04)
