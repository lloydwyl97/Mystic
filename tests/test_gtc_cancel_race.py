"""A fill that lands after the OPEN read must survive the cancel."""

from __future__ import annotations

import sqlite3
import time
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from backend.services.portfolio_engine import PortfolioEngine, make_position_key
from backend.services.strategy_order_intent import (
    allocate_intent,
    apply_venue_order,
    intent_by_client,
    sell_intent_blocks,
    sell_quantity_state,
)
from tests.test_two_engine_architecture import DAY, SCALP, _engine, _lot


def _intent(db: str, *, qty: float = 1.0, age: float = 30.0, status: str = "OPEN", oid: str = "gtc-1") -> str:
    cid = allocate_intent(db, engine_id="DAY_V2", trade_id="t-btc", symbol="BTC/USDT", side="SELL", quantity=qty, limit_price=90.0)
    conn = sqlite3.connect(db)
    try:
        conn.execute(
            "UPDATE strategy_order_intents SET status=?, created_at=?, exchange_order_id=? WHERE client_order_id=?",
            (status, time.time() - age, oid, cid),
        )
        conn.commit()
    finally:
        conn.close()
    return cid


def _order(status: str, cid: str, *, filled: float = 0.0, oid: str = "gtc-1", fee: bool = True) -> dict:
    order = {
        "id": oid,
        "status": status,
        "filled": filled,
        "amount": 1.0,
        "average": 90.0,
        "price": 90.0,
        "cost": filled * 90.0,
        "clientOrderId": cid,
        "info": {"executedQty": str(filled), "cummulativeQuoteQty": str(filled * 90.0), "status": status.upper()},
    }
    if fee and filled > 0:
        order["fee"] = {"cost": 0.01, "currency": "USDT"}
        order["info"]["commission"] = "0.01"
        order["info"]["commissionAsset"] = "USDT"
    return order


def _engine_for(db: str, fetch, trades=None) -> PortfolioEngine:
    engine = PortfolioEngine.__new__(PortfolioEngine)
    engine.db_path = db
    engine._exit_reeval_requested = False
    engine._execution_book_is_fresh = AsyncMock(return_value={"best_bid": 89.0, "best_ask": 89.1, "book_age_ms": 40.0})
    engine._live_service = SimpleNamespace(
        fetch_order=fetch,
        fetch_order_trades=trades or AsyncMock(return_value={"status": "error"}),
        cancel_order=AsyncMock(return_value={"status": "success", "message": "cancelled"}),
    )
    return engine


@pytest.mark.asyncio
async def test_full_fill_between_probe_and_cancel_is_adopted_once(tmp_path):
    db = str(tmp_path / "i.db")
    cid = _intent(db)
    fetch = AsyncMock(return_value={"status": "success", "order": _order("closed", cid, filled=1.0)})
    engine = _engine_for(db, fetch)
    await engine._progress_working_sell("BTC/USDT", allow_cancel=True, cancel_only=True)
    row = intent_by_client(db, cid)
    assert row["status"] == "FILLED"
    assert row["filled_qty"] == pytest.approx(1.0)
    assert sell_intent_blocks(db, "BTC/USDT") is False
    assert engine._exit_reeval_requested is True
    await engine._progress_working_sell("BTC/USDT", allow_cancel=True, cancel_only=True)
    assert intent_by_client(db, cid)["filled_qty"] == pytest.approx(1.0)
    assert engine._live_service.cancel_order.await_count == 1


@pytest.mark.asyncio
async def test_partial_fill_before_cancel_keeps_the_unsold_remainder_unblocked(tmp_path):
    db = str(tmp_path / "i.db")
    cid = _intent(db, qty=1.0)
    fetch = AsyncMock(return_value={"status": "success", "order": _order("canceled", cid, filled=0.4)})
    engine = _engine_for(db, fetch)
    await engine._progress_working_sell("BTC/USDT", allow_cancel=True, cancel_only=True)
    row = intent_by_client(db, cid)
    assert row["filled_qty"] == pytest.approx(0.4)
    assert sell_intent_blocks(db, "BTC/USDT") is False
    state = sell_quantity_state(db, "t-btc", 1.0)
    assert state["executed_unbooked_qty"] == pytest.approx(0.4)
    assert state["working_sell_qty"] == pytest.approx(0.0)
    assert state["available_to_sell_qty"] == pytest.approx(0.6)


@pytest.mark.asyncio
async def test_cumulative_fill_replaces_the_earlier_partial_snapshot(tmp_path):
    db = str(tmp_path / "i.db")
    cid = _intent(db, qty=1.0, status="PENDING", oid="")
    first = apply_venue_order(db, _order("open", cid, filled=0.2, oid="gtc-1"))
    assert first["adopted"] is True
    assert intent_by_client(db, cid)["filled_qty"] == pytest.approx(0.2)
    fetch = AsyncMock(return_value={"status": "success", "order": _order("canceled", cid, filled=0.5)})
    engine = _engine_for(db, fetch)
    await engine._progress_working_sell("BTC/USDT", allow_cancel=True, cancel_only=True)
    assert intent_by_client(db, cid)["filled_qty"] == pytest.approx(0.5)
    await engine._progress_working_sell("BTC/USDT", allow_cancel=True, cancel_only=True)
    assert intent_by_client(db, cid)["filled_qty"] == pytest.approx(0.5)


@pytest.mark.asyncio
async def test_cancel_without_a_fee_reserves_the_executed_quantity_via_mytrades(tmp_path):
    db = str(tmp_path / "i.db")
    cid = _intent(db)
    trades = AsyncMock(
        return_value={
            "status": "success",
            "trades": [{"id": "trade-9", "qty": 0.4, "price": 90.0, "fee": {"cost": 0.02, "currency": "USDT"}, "order": "gtc-1"}],
        }
    )
    fetch = AsyncMock(return_value={"status": "success", "order": _order("canceled", cid, filled=0.4, fee=False)})
    engine = _engine_for(db, fetch, trades)
    await engine._progress_working_sell("BTC/USDT", allow_cancel=True, cancel_only=True)
    row = intent_by_client(db, cid)
    assert row["filled_qty"] == pytest.approx(0.4)
    assert sell_intent_blocks(db, "BTC/USDT") is False
    assert "trade-9" in str(row["fill_ids_json"])
    trades.assert_awaited()
    await engine._apply_probed_order(_order("canceled", cid, filled=0.4, fee=False), "BTC/USDT", cid, wait=False)
    assert intent_by_client(db, cid)["filled_qty"] == pytest.approx(0.4)


@pytest.mark.asyncio
async def test_lost_cancel_ack_keeps_the_intent_blocking_until_a_later_fill(tmp_path):
    db = str(tmp_path / "i.db")
    cid = _intent(db)

    async def fetch(*_a, **_k):
        raise TimeoutError("confirm")

    engine = _engine_for(db, fetch)
    await engine._progress_working_sell("BTC/USDT", allow_cancel=True, cancel_only=True)
    assert intent_by_client(db, cid)["status"] == "OPEN"
    assert sell_intent_blocks(db, "BTC/USDT") is True
    assert engine._exit_reeval_requested is False
    engine._live_service.fetch_order = AsyncMock(return_value={"status": "success", "order": _order("closed", cid, filled=1.0)})
    await engine._progress_working_sell("BTC/USDT", allow_cancel=False)
    assert intent_by_client(db, cid)["status"] == "FILLED"
    assert intent_by_client(db, cid)["filled_qty"] == pytest.approx(1.0)
    assert engine._live_service.cancel_order.await_count == 1


@pytest.mark.asyncio
async def test_confirm_read_refusal_does_not_release_the_symbol(tmp_path):
    db = str(tmp_path / "i.db")
    cid = _intent(db)
    engine = _engine_for(db, AsyncMock(return_value={"status": "error", "rate_limited": True}))
    await engine._progress_working_sell("BTC/USDT", allow_cancel=True, cancel_only=True)
    assert sell_intent_blocks(db, "BTC/USDT") is True
    engine2 = _engine_for(db, AsyncMock(return_value={"status": "error", "code": "-2013"}))
    await engine2._progress_working_sell("BTC/USDT", allow_cancel=True, cancel_only=True)
    assert intent_by_client(db, cid)["status"] == "OPEN"


@pytest.mark.asyncio
async def test_zero_fill_cancel_leaves_the_position_and_releases_the_intent(tmp_path):
    db = str(tmp_path / "i.db")
    cid = _intent(db)
    engine = _engine_for(db, AsyncMock(return_value={"status": "success", "order": _order("canceled", cid, filled=0.0, fee=False)}))
    await engine._progress_working_sell("BTC/USDT", allow_cancel=True, cancel_only=True)
    row = intent_by_client(db, cid)
    assert row["status"] == "CANCELED"
    assert row["filled_qty"] == pytest.approx(0.0)
    assert sell_intent_blocks(db, "BTC/USDT") is False
    state = sell_quantity_state(db, "t-btc", 1.0)
    assert state["reserved_qty"] == pytest.approx(0.0)
    assert state["available_to_sell_qty"] == pytest.approx(1.0)


@pytest.mark.asyncio
async def test_reservation_shrinks_only_after_the_cancel_is_confirmed(tmp_path):
    db = str(tmp_path / "i.db")
    cid = _intent(db, qty=0.6, status="PARTIAL")
    conn = sqlite3.connect(db)
    try:
        conn.execute("UPDATE strategy_order_intents SET filled_qty=0.2, filled_cost=18 WHERE client_order_id=?", (cid,))
        conn.commit()
    finally:
        conn.close()
    before = sell_quantity_state(db, "t-btc", 1.0)
    assert before["reserved_qty"] == pytest.approx(0.6)
    assert before["available_to_sell_qty"] == pytest.approx(0.4)
    engine = _engine_for(db, AsyncMock(return_value={"status": "success", "order": _order("canceled", cid, filled=0.2)}))
    await engine._progress_working_sell("BTC/USDT", allow_cancel=True, cancel_only=True)
    after = sell_quantity_state(db, "t-btc", 1.0)
    assert after["executed_unbooked_qty"] == pytest.approx(0.2)
    assert after["working_sell_qty"] == pytest.approx(0.0)
    assert after["available_to_sell_qty"] == pytest.approx(0.8)
    conn = sqlite3.connect(db)
    try:
        conn.execute("CREATE TABLE paper_trades (order_id TEXT, side TEXT, quantity REAL)")
        conn.execute("INSERT INTO paper_trades (order_id, side, quantity) VALUES ('gtc-1', 'SELL', 0.2)")
        conn.commit()
    finally:
        conn.close()
    booked = sell_quantity_state(db, "t-btc", 0.8)
    assert booked["reserved_qty"] == pytest.approx(0.0)
    assert booked["available_to_sell_qty"] == pytest.approx(0.8)


@pytest.mark.asyncio
async def test_restart_after_an_unconfirmed_cancel_books_the_fill_once(tmp_path):
    db = str(tmp_path / "i.db")
    cid = _intent(db, status="OPEN")
    engine = _engine_for(db, AsyncMock(side_effect=TimeoutError("lost")))
    await engine._progress_working_sell("BTC/USDT", allow_cancel=True, cancel_only=True)
    assert sell_intent_blocks(db, "BTC/USDT") is True
    restarted = _engine_for(db, AsyncMock(return_value={"status": "success", "order": _order("canceled", cid, filled=0.4)}))
    await restarted._recover_strategy_order_intents()
    assert intent_by_client(db, cid)["filled_qty"] == pytest.approx(0.4)
    await restarted._recover_strategy_order_intents()
    assert intent_by_client(db, cid)["filled_qty"] == pytest.approx(0.4)


@pytest.mark.asyncio
async def test_day_cancel_race_wakes_the_same_symbol_scalp_lot(tmp_path, monkeypatch):
    import backend.services.day_high_water as hw

    monkeypatch.setattr(hw, "load_feature_1m_candles", lambda *_a, **_k: [])
    eng = _engine(tmp_path)
    eng.run_trading_circuit_breaker_check = AsyncMock()
    eng._build_exit_check_telemetry = lambda *_a, **_k: {}
    eng._log_exit_check_telemetry = lambda *_a, **_k: None
    eng._persist_position_to_sqlite = AsyncMock()
    eng._emit_day_health_telemetry = AsyncMock()
    eng._learning_heartbeat_last = {"BTC/USDT": time.time()}
    eng._execution_book_is_fresh = AsyncMock(return_value={"best_bid": 89.0, "best_ask": 89.1, "book_age_ms": 20.0})
    eng._resolve_exit_monitor_mark = AsyncMock(return_value={"mark_used": 100.0, "price_source_stale": False, "mark_timestamp": time.time()})
    day = _lot("BTC/USDT", DAY, trade_id="t-btc")
    scalp = _lot("BTC/USDT", SCALP, trade_id="t-scalp")
    eng.open_positions[make_position_key(DAY, "BTC/USDT")] = day
    eng.open_positions[make_position_key(SCALP, "BTC/USDT")] = scalp
    cid = _intent(eng.db_path)
    calls = {"n": 0}

    async def fetch(*_a, **_k):
        calls["n"] += 1
        status = "open" if calls["n"] == 1 else "canceled"
        filled = 0.0 if calls["n"] == 1 else 0.4
        return {"status": "success", "order": _order(status, cid, filled=filled)}

    eng._live_service = SimpleNamespace(
        fetch_order=fetch,
        fetch_order_trades=AsyncMock(return_value={"status": "error"}),
        cancel_order=AsyncMock(return_value={"status": "success"}),
    )
    checked: list[str] = []

    async def _check(position, *_a, **_k):
        checked.append(str(position.engine_id))

    eng._check_exit_conditions = _check
    eng._scalp_exit_quote = AsyncMock(return_value=({"mark_used": 100.0, "price_source_stale": False, "mark_timestamp": time.time()}, 99.9))
    await eng.monitor_all_positions({}, 0)
    assert intent_by_client(eng.db_path, cid)["filled_qty"] == pytest.approx(0.4)
    assert sell_intent_blocks(eng.db_path, "BTC/USDT") is False
    assert eng._exit_reeval_requested is True
    assert SCALP in checked
    assert eng._live_service.cancel_order.await_count == 1
