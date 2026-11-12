"""An existing SELL is classified before this lot's mark is resolved."""

from __future__ import annotations

import inspect
import sqlite3
import time
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from backend.services.portfolio_engine import PortfolioEngine, make_position_key
from backend.services.strategy_order_intent import allocate_intent, intent_by_client, sell_intent_blocks
from tests.test_two_engine_architecture import DAY, SCALP, _engine, _lot

MARK = {"mark_used": 100.0, "price_source_stale": False, "mark_timestamp": time.time()}


def _intent(db: str, trade_id: str, *, status: str = "PENDING", qty: float = 1.0, age: float = 30.0) -> str:
    cid = allocate_intent(db, engine_id=DAY, trade_id=trade_id, symbol="BTC/USDT", side="SELL", quantity=qty, limit_price=90.0)
    conn = sqlite3.connect(db)
    try:
        conn.execute(
            "UPDATE strategy_order_intents SET status=?, created_at=?, filled_qty=? WHERE client_order_id=?",
            (status, time.time() - age, 0.4 if status == "FEE_PENDING" else 0.0, cid),
        )
        if status == "FEE_PENDING":
            conn.execute(
                "UPDATE strategy_order_intents SET exchange_order_id=?, filled_cost=? WHERE client_order_id=?",
                ("fee-1", 36.0, cid),
            )
        conn.commit()
    finally:
        conn.close()
    return cid


def _order(status: str, cid: str, *, filled: float = 0.0, oid: str = "99") -> dict:
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


def _monitor(tmp_path, monkeypatch, *, mark):
    import backend.services.day_high_water as hw

    monkeypatch.setattr(hw, "load_feature_1m_candles", lambda *_a, **_k: [])
    eng = _engine(tmp_path)
    eng.run_trading_circuit_breaker_check = AsyncMock()
    eng._build_exit_check_telemetry = lambda *_a, **_k: {}
    eng._log_exit_check_telemetry = lambda *_a, **_k: None
    eng._persist_position_to_sqlite = AsyncMock()
    eng._emit_day_health_telemetry = AsyncMock()
    eng._learning_heartbeat_last = {"BTC/USDT": time.time()}
    eng._execution_book_is_fresh = AsyncMock(return_value=None)
    eng.execute_sell_fifo = AsyncMock(return_value=None)
    eng._check_exit_conditions = AsyncMock(return_value=None)
    if isinstance(mark, Exception):
        eng._resolve_exit_monitor_mark = AsyncMock(side_effect=mark)
    else:
        eng._resolve_exit_monitor_mark = AsyncMock(return_value=mark)
    lot = _lot("BTC/USDT", DAY)
    lot.highest_price = 101.0
    lot.lowest_price = 99.0
    eng.open_positions[make_position_key(DAY, "BTC/USDT")] = lot
    return eng, lot


def _live(eng, fetch):
    eng._live_service = SimpleNamespace(
        fetch_order=fetch,
        fetch_order_trades=AsyncMock(return_value={"status": "error"}),
        cancel_order=AsyncMock(return_value={"status": "success"}),
    )


def test_status_probe_is_ahead_of_mark_resolution_in_source():
    body = inspect.getsource(PortfolioEngine.monitor_all_positions)
    assert body.index("_progress_working_sell") < body.index("_resolve_exit_monitor_mark")
    assert body.index("_exit_mark_is_usable") < body.index("_check_exit_conditions")


@pytest.mark.asyncio
async def test_lost_ack_fill_is_adopted_when_the_mark_raises(tmp_path, monkeypatch):
    eng, lot = _monitor(tmp_path, monkeypatch, mark=RuntimeError("mark-down"))
    cid = _intent(eng.db_path, lot.trade_id)

    async def fetch(*_a, **_k):
        return {"status": "success", "order": _order("closed", cid, filled=1.0, oid="fill-1")}

    _live(eng, fetch)
    await eng.monitor_all_positions({}, 0)
    assert intent_by_client(eng.db_path, cid)["status"] == "FILLED"
    assert intent_by_client(eng.db_path, cid)["filled_qty"] == pytest.approx(1.0)
    assert sell_intent_blocks(eng.db_path, "BTC/USDT") is False
    assert eng._exit_reeval_requested is True
    assert eng._check_exit_conditions.await_count == 0
    assert eng._last_order_probe_result == "closed"
    assert eng._last_mark_failure == "RuntimeError"
    eng.execute_sell_fifo.assert_not_awaited()


@pytest.mark.asyncio
async def test_expired_order_releases_the_symbol_when_the_mark_raises(tmp_path, monkeypatch):
    eng, lot = _monitor(tmp_path, monkeypatch, mark=RuntimeError("mark-down"))
    cid = _intent(eng.db_path, lot.trade_id)
    _live(eng, AsyncMock(return_value={"status": "success", "order": _order("expired", cid, oid="ex-1")}))
    await eng.monitor_all_positions({}, 0)
    assert intent_by_client(eng.db_path, cid)["status"] == "EXPIRED"
    assert sell_intent_blocks(eng.db_path, "BTC/USDT") is False
    assert eng._exit_reeval_requested is True
    eng._live_service.cancel_order.assert_not_awaited()


@pytest.mark.asyncio
async def test_partial_fill_stays_reserved_when_the_mark_raises(tmp_path, monkeypatch):
    eng, lot = _monitor(tmp_path, monkeypatch, mark=RuntimeError("mark-down"))
    cid = _intent(eng.db_path, lot.trade_id, qty=1.0)
    _live(eng, AsyncMock(return_value={"status": "success", "order": _order("open", cid, filled=0.4, oid="part-1")}))
    await eng.monitor_all_positions({}, 0)
    row = intent_by_client(eng.db_path, cid)
    assert row["status"] == "PARTIAL"
    assert row["filled_qty"] == pytest.approx(0.4)
    assert sell_intent_blocks(eng.db_path, "BTC/USDT") is True
    await eng.monitor_all_positions({}, 0)
    assert intent_by_client(eng.db_path, cid)["filled_qty"] == pytest.approx(0.4)
    eng._live_service.cancel_order.assert_not_awaited()


@pytest.mark.asyncio
async def test_open_order_is_probed_and_still_blocks_when_the_mark_raises(tmp_path, monkeypatch):
    eng, lot = _monitor(tmp_path, monkeypatch, mark=RuntimeError("mark-down"))
    cid = _intent(eng.db_path, lot.trade_id)
    fetch = AsyncMock(return_value={"status": "success", "order": _order("open", cid, oid="open-1")})
    _live(eng, fetch)
    await eng.monitor_all_positions({}, 0)
    assert intent_by_client(eng.db_path, cid)["status"] == "OPEN"
    assert sell_intent_blocks(eng.db_path, "BTC/USDT") is True
    assert fetch.await_count >= 1
    eng._live_service.cancel_order.assert_not_awaited()
    assert eng._last_order_probe_result == "open"


@pytest.mark.asyncio
async def test_stale_mark_still_probes_and_does_not_sell(tmp_path, monkeypatch):
    eng, lot = _monitor(
        tmp_path,
        monkeypatch,
        mark={"mark_used": 89.0, "price_source_stale": True, "mark_timestamp": time.time(), "mark_source": "stale"},
    )
    cid = _intent(eng.db_path, lot.trade_id)
    _live(eng, AsyncMock(return_value={"status": "success", "order": _order("open", cid, oid="stale-1")}))
    await eng.monitor_all_positions({}, 0)
    assert eng._check_exit_conditions.await_count == 0
    eng.execute_sell_fifo.assert_not_awaited()
    assert intent_by_client(eng.db_path, cid)["status"] == "OPEN"
    assert eng._last_order_probe_result == "open"


@pytest.mark.asyncio
async def test_fresh_book_is_required_before_cancel_after_a_successful_probe(tmp_path, monkeypatch):
    eng, lot = _monitor(tmp_path, monkeypatch, mark=dict(MARK))
    cid = _intent(eng.db_path, lot.trade_id, status="OPEN", age=30.0)
    conn = sqlite3.connect(eng.db_path)
    try:
        conn.execute("UPDATE strategy_order_intents SET exchange_order_id=? WHERE client_order_id=?", ("gtc-1", cid))
        conn.commit()
    finally:
        conn.close()
    _live(eng, AsyncMock(return_value={"status": "success", "order": _order("open", cid, oid="gtc-1")}))
    await eng.monitor_all_positions({}, 0)
    eng._live_service.cancel_order.assert_not_awaited()
    assert sell_intent_blocks(eng.db_path, "BTC/USDT") is True
    assert eng._check_exit_conditions.await_count == 1


@pytest.mark.asyncio
async def test_fee_pending_is_enriched_when_the_mark_raises(tmp_path, monkeypatch):
    eng, lot = _monitor(tmp_path, monkeypatch, mark=RuntimeError("mark-down"))
    _intent(eng.db_path, lot.trade_id, status="FEE_PENDING")
    eng._canonical_pending_sell = AsyncMock(return_value={"id": "fee-1", "average": 90.0, "filled": 0.4, "status": "closed"})
    _live(eng, AsyncMock(return_value={"status": "error", "code": "-2013"}))
    await eng.monitor_all_positions({}, 0)
    eng._canonical_pending_sell.assert_awaited()
    eng.execute_sell_fifo.assert_awaited()
    assert eng.execute_sell_fifo.await_args.kwargs["force_sell"] is True
    assert eng.execute_sell_fifo.await_args.args[2] == pytest.approx(90.0)


@pytest.mark.asyncio
async def test_day_terminal_sell_wakes_the_scalp_lot_when_the_day_mark_raises(tmp_path, monkeypatch):
    eng, day = _monitor(tmp_path, monkeypatch, mark=RuntimeError("day-mark"))
    scalp = _lot("BTC/USDT", SCALP)
    eng.open_positions[make_position_key(SCALP, "BTC/USDT")] = scalp
    eng._learning_heartbeat_last["BTC/USDT"] = time.time()
    cid = _intent(eng.db_path, day.trade_id)
    _live(eng, AsyncMock(return_value={"status": "success", "order": _order("expired", cid, oid="cross-1")}))

    async def scalp_quote(_symbol, executable_quote_only=False):
        return dict(MARK), 100.0

    eng._scalp_exit_quote = scalp_quote
    checked: list[str] = []

    async def _check(position, *_a, **_k):
        checked.append(str(position.engine_id))

    eng._check_exit_conditions = _check
    await eng.monitor_all_positions({}, 0)
    assert sell_intent_blocks(eng.db_path, "BTC/USDT") is False
    assert eng._exit_reeval_requested is True
    assert SCALP in checked
