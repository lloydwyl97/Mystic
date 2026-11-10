"""A protective SELL that is done at the venue must release, and a live one must not be duplicated."""

from __future__ import annotations

import sqlite3
import time
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from backend.services.portfolio_engine import PortfolioEngine
from backend.services.strategy_order_intent import (
    allocate_intent,
    apply_venue_order,
    intent_by_client,
    recovered_sell_order,
    sell_intent_blocks,
)
from backend.services.trade_learning_writer import TradeLearningRecord, record_trade_outcome


def _intent(db: str, *, status: str = "PENDING", qty: float = 1.0, limit_price: float = 90.0, age: float = 30.0) -> str:
    cid = allocate_intent(
        db,
        engine_id="DAY_V2",
        trade_id="t-btc",
        symbol="BTC/USDT",
        side="SELL",
        quantity=qty,
        limit_price=limit_price,
    )
    conn = sqlite3.connect(db)
    try:
        conn.execute(
            "UPDATE strategy_order_intents SET status=?, created_at=? WHERE client_order_id=?",
            (status, time.time() - age, cid),
        )
        conn.commit()
    finally:
        conn.close()
    return cid


def _order(status: str, *, filled: float = 0.0, oid: str = "99", cid: str = "") -> dict:
    order = {
        "id": oid,
        "status": status,
        "filled": filled,
        "amount": 1.0,
        "average": 90.0,
        "price": 90.0,
        "clientOrderId": cid,
    }
    if filled > 0:
        order["fee"] = {"cost": 0.01, "currency": "USDT"}
    return order


def test_expired_ioc_releases_the_intent_and_the_next_id_is_new(tmp_path):
    db = str(tmp_path / "i.db")
    cid = _intent(db)
    result = apply_venue_order(db, _order("expired", cid=cid))
    assert result["released"] is True
    assert intent_by_client(db, cid)["status"] == "EXPIRED"
    assert sell_intent_blocks(db, "BTC/USDT") is False
    nxt = allocate_intent(db, engine_id="DAY_V2", trade_id="t-btc", symbol="BTC/USDT", side="SELL", quantity=1.0, limit_price=85.0)
    assert nxt != cid


def test_open_and_partial_sells_block_a_second_order(tmp_path):
    db = str(tmp_path / "i.db")
    cid = _intent(db)
    apply_venue_order(db, _order("open", cid=cid))
    assert intent_by_client(db, cid)["status"] == "OPEN"
    assert sell_intent_blocks(db, "BTC/USDT") is True
    assert allocate_intent(db, engine_id="DAY_V2", trade_id="t-btc", symbol="BTC/USDT", side="SELL", quantity=1.0) == cid

    db2 = str(tmp_path / "p.db")
    cid2 = _intent(db2)
    apply_venue_order(db2, _order("open", filled=0.4, cid=cid2))
    assert intent_by_client(db2, cid2)["status"] == "PARTIAL"
    assert intent_by_client(db2, cid2)["filled_qty"] == pytest.approx(0.4)
    assert sell_intent_blocks(db2, "BTC/USDT") is True


def test_partial_fill_is_booked_once_and_does_not_resize_the_next_request(tmp_path):
    db = str(tmp_path / "i.db")
    cid = _intent(db, qty=1.0)
    apply_venue_order(db, _order("expired", filled=0.4, oid="sell-1", cid=cid))
    recovered = recovered_sell_order(db, "t-btc", requested_qty=1.0)
    assert recovered is not None
    assert recovered["filled"] == pytest.approx(0.4)
    assert recovered["filled"] < 1.0
    conn = sqlite3.connect(db)
    try:
        conn.execute("CREATE TABLE paper_trades (order_id TEXT, side TEXT, quantity REAL)")
        conn.execute("INSERT INTO paper_trades (order_id, side, quantity) VALUES ('sell-1', 'SELL', 0.4)")
        conn.commit()
    finally:
        conn.close()
    assert recovered_sell_order(db, "t-btc", requested_qty=0.6) is None
    assert sell_intent_blocks(db, "BTC/USDT") is False


def test_same_close_is_learned_once(tmp_path):
    db = str(tmp_path / "i.db")
    record = TradeLearningRecord(
        symbol="BTC/USDT",
        close_reason="STOP_LOSS_EXIT",
        net_profit_usd=-1.0,
        net_profit_pct=-0.01,
        exit_price=90.0,
        entry_price=100.0,
        quantity=0.4,
        event_key="live-close:sell-1",
    )
    assert record_trade_outcome(record, db_path=db) is True
    assert record_trade_outcome(record, db_path=db) is True
    conn = sqlite3.connect(db)
    try:
        assert conn.execute("SELECT COUNT(*) FROM trade_learning_outcomes WHERE event_key='live-close:sell-1'").fetchone()[0] == 1
    finally:
        conn.close()


def _engine(db: str, fetch) -> PortfolioEngine:
    engine = PortfolioEngine.__new__(PortfolioEngine)
    engine.db_path = db
    engine._live_service = SimpleNamespace(
        fetch_order=fetch,
        fetch_order_trades=AsyncMock(return_value={"status": "error"}),
        cancel_order=AsyncMock(return_value={"status": "success"}),
    )
    return engine


@pytest.mark.asyncio
async def test_restart_open_order_stays_blocking_and_expired_order_releases(tmp_path):
    db = str(tmp_path / "i.db")
    open_id = _intent(db, status="ACKED")
    engine = _engine(db, AsyncMock(return_value={"status": "success", "order": _order("open", oid="11", cid=open_id)}))
    await engine._recover_strategy_order_intents()
    assert intent_by_client(db, open_id)["status"] == "OPEN"
    assert sell_intent_blocks(db, "BTC/USDT") is True

    db2 = str(tmp_path / "e.db")
    dead = _intent(db2, status="PENDING")
    engine2 = _engine(db2, AsyncMock(return_value={"status": "success", "order": _order("expired", oid="12", cid=dead)}))
    await engine2._recover_strategy_order_intents()
    assert intent_by_client(db2, dead)["status"] == "EXPIRED"
    assert sell_intent_blocks(db2, "BTC/USDT") is False


@pytest.mark.asyncio
async def test_explicit_absence_releases_and_a_timeout_does_not(tmp_path):
    db = str(tmp_path / "i.db")
    cid = _intent(db)

    async def absent(*_a, **_k):
        return {"status": "error", "code": "-2013"}

    engine = _engine(db, absent)
    await engine._progress_working_sell("BTC/USDT", allow_cancel=True)
    assert intent_by_client(db, cid)["status"] == "NO_VENUE_ORDER"
    assert sell_intent_blocks(db, "BTC/USDT") is False

    db2 = str(tmp_path / "t.db")
    cid2 = _intent(db2)

    async def timeout(*_a, **_k):
        raise TimeoutError("venue")

    engine2 = _engine(db2, timeout)
    await engine2._progress_working_sell("BTC/USDT", allow_cancel=True)
    assert intent_by_client(db2, cid2)["status"] == "PENDING"
    assert sell_intent_blocks(db2, "BTC/USDT") is True
    engine2._live_service.cancel_order.assert_not_awaited()


@pytest.mark.asyncio
async def test_stale_book_does_not_cancel_and_a_fresh_book_cancels_a_timed_out_order(tmp_path, monkeypatch):
    db = str(tmp_path / "i.db")
    cid = _intent(db, status="OPEN", age=30.0)
    calls = {"n": 0}

    async def fetch(*_a, **_k):
        calls["n"] += 1
        status = "open" if calls["n"] == 1 else "canceled"
        return {"status": "success", "order": _order(status, oid="15", cid=cid)}

    engine = _engine(db, fetch)
    engine._execution_book_is_fresh = AsyncMock(return_value=None)
    await engine._progress_working_sell("BTC/USDT", allow_cancel=False)
    engine._live_service.cancel_order.assert_not_awaited()
    assert sell_intent_blocks(db, "BTC/USDT") is True

    calls["n"] = 0
    engine._execution_book_is_fresh = AsyncMock(return_value={"best_bid": 85.0, "best_ask": 85.1, "book_age_ms": 100.0})
    await engine._progress_working_sell("BTC/USDT", allow_cancel=True)
    engine._live_service.cancel_order.assert_awaited()
    assert intent_by_client(db, cid)["status"] == "CANCELED"
    assert engine._last_sell_probe["drift_bps"] == pytest.approx((90.0 - 85.0) / 85.0 * 10000.0)
    assert sell_intent_blocks(db, "BTC/USDT") is False


@pytest.mark.asyncio
async def test_cancel_confirm_timeout_does_not_allow_a_replacement(tmp_path):
    db = str(tmp_path / "i.db")
    cid = _intent(db, status="OPEN", age=30.0)
    calls = {"n": 0}

    async def fetch(*_a, **_k):
        calls["n"] += 1
        if calls["n"] == 1:
            return {"status": "success", "order": _order("open", oid="16", cid=cid)}
        raise TimeoutError("confirm")

    engine = _engine(db, fetch)
    engine._execution_book_is_fresh = AsyncMock(return_value={"best_bid": 85.0, "best_ask": 85.1, "book_age_ms": 50.0})
    await engine._progress_working_sell("BTC/USDT", allow_cancel=True)
    assert intent_by_client(db, cid)["status"] == "OPEN"
    assert sell_intent_blocks(db, "BTC/USDT") is True


@pytest.mark.asyncio
async def test_fill_during_cancel_is_the_only_quantity_that_can_be_booked(tmp_path):
    db = str(tmp_path / "i.db")
    cid = _intent(db, status="OPEN", qty=0.6, age=30.0)
    calls = {"n": 0}

    async def fetch(*_a, **_k):
        calls["n"] += 1
        filled = 0.0 if calls["n"] == 1 else 0.3
        status = "open" if calls["n"] == 1 else "canceled"
        return {"status": "success", "order": _order(status, filled=filled, oid="17", cid=cid)}

    engine = _engine(db, fetch)
    engine._execution_book_is_fresh = AsyncMock(return_value={"best_bid": 85.0, "best_ask": 85.1, "book_age_ms": 40.0})
    await engine._progress_working_sell("BTC/USDT", allow_cancel=True)
    recovered = recovered_sell_order(db, "t-btc", requested_qty=0.6)
    assert recovered is not None
    assert recovered["filled"] == pytest.approx(0.3)
    assert recovered["filled"] <= 0.3


@pytest.mark.asyncio
async def test_unresolved_partial_does_not_reach_the_sell_body(tmp_path):
    db = str(tmp_path / "i.db")
    _intent(db, status="PARTIAL")
    conn = sqlite3.connect(db)
    try:
        conn.execute("UPDATE strategy_order_intents SET filled_qty=0.4")
        conn.commit()
    finally:
        conn.close()
    engine = PortfolioEngine.__new__(PortfolioEngine)
    engine._live_execution_enabled = True
    engine._ownership_state_known = True
    engine.db_path = db
    engine._execute_sell_fifo_locked = AsyncMock(return_value={"placed": True})
    result = await engine.execute_sell_fifo("BTC/USDT", 1.0, 90.0, "CATASTROPHIC", "DAY_V2_CATASTROPHIC")
    assert result is None
    engine._execute_sell_fifo_locked.assert_not_awaited()


@pytest.mark.asyncio
async def test_released_intent_reaches_the_sell_body(tmp_path):
    db = str(tmp_path / "i.db")
    _intent(db, status="EXPIRED")
    engine = PortfolioEngine.__new__(PortfolioEngine)
    engine._live_execution_enabled = True
    engine._ownership_state_known = True
    engine.db_path = db
    engine._sell_execution_locks = {}
    engine._execute_sell_fifo_locked = AsyncMock(return_value={"placed": True})
    result = await engine.execute_sell_fifo("BTC/USDT", 1.0, 85.0, "CATASTROPHIC", "DAY_V2_CATASTROPHIC")
    assert result == {"placed": True}
    engine._execute_sell_fifo_locked.assert_awaited()


@pytest.mark.asyncio
async def test_ioc_expire_does_not_place_gtc_and_a_missing_ioc_does(monkeypatch):
    from backend.services import protected_limit_execution as ple

    async def _no_sleep(_s):
        return None

    async def _enrich(_svc, order, _sym):
        return order

    monkeypatch.setattr(ple.asyncio, "sleep", _no_sleep)
    monkeypatch.setattr(ple, "_enrich_live_order_fills", _enrich)
    monkeypatch.setattr(ple, "PROTECTED_LIMIT_ORDER_TIMEOUT_SEC", 0)
    placed: list[str] = []

    async def place(**kwargs):
        placed.append(str(kwargs.get("time_in_force")))
        return {"status": "success", "order": {"id": "1", "filled": 0.0, "amount": 1.0, "status": "expired"}}

    async def fetch(_exchange, order_id, _symbol):
        if str(order_id).endswith("-i"):
            return {"status": "error", "code": "-2013"}
        return {"status": "success", "order": {"id": "2", "filled": 1.0, "amount": 1.0, "status": "closed", "average": 90.0, "clientOrderId": str(order_id)}}

    venue = SimpleNamespace(
        place_order=place,
        fetch_order=fetch,
        cancel_order=AsyncMock(),
        _ensure_initialized=AsyncMock(),
    )
    out = await ple.execute_protected_limit_live(venue, symbol="BTCUSDT", side="sell", quantity=1.0, limit_price=90.0, client_order_id="a" * 32)
    assert out is None
    assert placed == ["IOC"]

    placed.clear()

    async def place_missing(**kwargs):
        placed.append(str(kwargs.get("time_in_force")))
        if kwargs.get("time_in_force") == "IOC":
            return {"status": "error", "message": "unsupported"}
        return {"status": "success", "order": {"id": "2", "filled": 1.0, "amount": 1.0, "status": "closed", "average": 90.0}}

    venue.place_order = place_missing
    out = await ple.execute_protected_limit_live(venue, symbol="BTCUSDT", side="sell", quantity=1.0, limit_price=90.0, client_order_id="b" * 32)
    assert out is not None
    assert float(out["filled"]) == pytest.approx(1.0)
    assert placed == ["IOC", "GTC"]
