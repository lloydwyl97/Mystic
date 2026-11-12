"""The book that gates a protective GTC cancel is exit work, not background work."""

from __future__ import annotations

import inspect
import time
from unittest.mock import AsyncMock

import pytest

from backend.services.portfolio_engine import PortfolioEngine, make_position_key
from backend.services.protected_limit_execution import ExecutableBook, _fetch_order_book
from backend.services.strategy_order_intent import intent_by_client, sell_intent_blocks
from backend.utils.binance_weight_limiter import RateLimited
from tests.test_gtc_cancel_race import _engine_for, _intent, _order
from tests.test_two_engine_architecture import DAY, SCALP, _engine, _lot


def test_protective_pending_is_cleared_when_the_mandatory_call_returns():
    from backend.services.day_mandatory_exit_execution import run_mandatory_exit_ioc_loop
    from backend.services.live_trading_service import LiveTradingService

    loop = inspect.getsource(run_mandatory_exit_ioc_loop)
    place = inspect.getsource(LiveTradingService._place_order_unfenced)
    assert loop.index("mark_protective_pending(1)") < loop.index("finally")
    assert loop.index("finally") < loop.index("mark_protective_pending(-1)")
    assert place.index("mark_protective_pending(1)") < place.index("finally")
    assert place.index("finally") < place.index("mark_protective_pending(-1)")


def test_only_the_cancel_book_asks_for_exit_priority():
    fetch = inspect.getsource(_fetch_order_book)
    cancel = inspect.getsource(PortfolioEngine._execution_book_is_fresh)
    assert 'priority: str = "background"' in fetch
    assert 'priority="exit"' in cancel
    assert "surface_errors=True" in cancel


def _fresh(priority: str) -> ExecutableBook:
    now = time.time()
    return ExecutableBook(
        bids=[[89.0, 1.0]],
        asks=[[89.2, 1.0]],
        last_update_id=9_000_000_000_000 + int(now),
        source_ts=now,
        receive_ts=now,
        exchange_ts=now,
    )


@pytest.mark.asyncio
async def test_old_gtc_cancel_book_uses_exit_priority_and_then_cancels(tmp_path, monkeypatch):
    db = str(tmp_path / "i.db")
    cid = _intent(db)
    seen: list[str] = []

    async def fetch(symbol, *, priority="background", surface_errors=False):
        seen.append(priority)
        return _fresh(priority)

    monkeypatch.setattr("backend.services.protected_limit_execution._fetch_order_book", fetch)
    engine = _engine_for(db, AsyncMock(return_value={"status": "success", "order": _order("canceled", cid, filled=0.0, fee=False)}))
    del engine._execution_book_is_fresh
    await engine._progress_working_sell("BTC/USDT", allow_cancel=True, cancel_only=True)
    assert seen == ["exit"]
    assert intent_by_client(db, cid)["status"] == "CANCELED"
    engine._live_service.cancel_order.assert_awaited()
    assert engine._live_service.cancel_order.await_args.kwargs["priority"] == "exit"


@pytest.mark.asyncio
async def test_weight_refusal_does_not_cancel_while_pending_would_block_background(tmp_path, monkeypatch):
    db = str(tmp_path / "i.db")
    cid = _intent(db)

    async def fetch(symbol, *, priority="background", surface_errors=False):
        assert priority == "exit"
        raise RateLimited("protective reserve")

    monkeypatch.setattr("backend.services.protected_limit_execution._fetch_order_book", fetch)
    engine = _engine_for(db, AsyncMock(return_value={"status": "success", "order": _order("open", cid)}))
    del engine._execution_book_is_fresh
    await engine._progress_working_sell("BTC/USDT", allow_cancel=True, cancel_only=True)
    engine._live_service.cancel_order.assert_not_awaited()
    assert intent_by_client(db, cid)["status"] == "OPEN"
    assert sell_intent_blocks(db, "BTC/USDT") is True


@pytest.mark.asyncio
async def test_depth_failure_and_confirm_failure_leave_the_gtc_blocking(tmp_path, monkeypatch):
    db = str(tmp_path / "i.db")
    cid = _intent(db)

    async def empty(symbol, *, priority="background", surface_errors=False):
        return None

    monkeypatch.setattr("backend.services.protected_limit_execution._fetch_order_book", empty)
    engine = _engine_for(db, AsyncMock(return_value={"status": "success", "order": _order("canceled", cid, filled=0.0, fee=False)}))
    del engine._execution_book_is_fresh
    await engine._progress_working_sell("BTC/USDT", allow_cancel=True, cancel_only=True)
    engine._live_service.cancel_order.assert_not_awaited()
    assert sell_intent_blocks(db, "BTC/USDT") is True

    async def fresh(symbol, *, priority="background", surface_errors=False):
        return _fresh(priority)

    monkeypatch.setattr("backend.services.protected_limit_execution._fetch_order_book", fresh)
    engine._live_service.fetch_order = AsyncMock(return_value={"status": "error", "rate_limited": True})
    await engine._progress_working_sell("BTC/USDT", allow_cancel=True, cancel_only=True)
    engine._live_service.cancel_order.assert_awaited()
    assert intent_by_client(db, cid)["status"] == "OPEN"
    assert sell_intent_blocks(db, "BTC/USDT") is True


@pytest.mark.asyncio
async def test_day_gtc_book_refusal_keeps_scalp_blocked_and_a_granted_book_releases_it(tmp_path, monkeypatch):
    import backend.services.day_high_water as hw

    monkeypatch.setattr(hw, "load_feature_1m_candles", lambda *_a, **_k: [])
    eng = _engine(tmp_path)
    eng.run_trading_circuit_breaker_check = AsyncMock()
    eng._build_exit_check_telemetry = lambda *_a, **_k: {}
    eng._log_exit_check_telemetry = lambda *_a, **_k: None
    eng._persist_position_to_sqlite = AsyncMock()
    eng._emit_day_health_telemetry = AsyncMock()
    eng._learning_heartbeat_last = {"BTC/USDT": time.time()}
    eng._resolve_exit_monitor_mark = AsyncMock(return_value={"mark_used": 100.0, "price_source_stale": False, "mark_timestamp": time.time()})
    eng._scalp_exit_quote = AsyncMock(return_value=({"mark_used": 95.0, "price_source_stale": False, "mark_timestamp": time.time()}, 94.9))
    day = _lot("BTC/USDT", DAY, trade_id="t-btc")
    scalp = _lot("BTC/USDT", SCALP, trade_id="t-scalp")
    eng.open_positions[make_position_key(DAY, "BTC/USDT")] = day
    eng.open_positions[make_position_key(SCALP, "BTC/USDT")] = scalp
    cid = _intent(eng.db_path)
    checked: list[str] = []

    async def _check(position, *_a, **_k):
        checked.append(str(position.engine_id))

    eng._check_exit_conditions = _check

    async def refused(symbol, *, priority="background", surface_errors=False):
        raise RateLimited("313")

    monkeypatch.setattr("backend.services.protected_limit_execution._fetch_order_book", refused)
    eng._live_service = AsyncMock()
    eng._live_service.fetch_order = AsyncMock(return_value={"status": "success", "order": _order("open", cid)})
    eng._live_service.fetch_order_trades = AsyncMock(return_value={"status": "error"})
    eng._live_service.cancel_order = AsyncMock(return_value={"status": "success"})
    await eng.monitor_all_positions({}, 0)
    assert sell_intent_blocks(eng.db_path, "BTC/USDT") is True
    eng._live_service.cancel_order.assert_not_awaited()
    assert SCALP in checked
    assert getattr(eng, "_exit_reeval_requested", False) is False

    async def granted(symbol, *, priority="background", surface_errors=False):
        assert priority == "exit"
        return _fresh(priority)

    monkeypatch.setattr("backend.services.protected_limit_execution._fetch_order_book", granted)
    calls = {"n": 0}

    async def fetch(*_a, **_k):
        calls["n"] += 1
        status = "open" if calls["n"] == 1 else "canceled"
        return {"status": "success", "order": _order(status, cid, filled=0.0, fee=False)}

    eng._live_service.fetch_order = fetch
    checked.clear()
    await eng.monitor_all_positions({}, 0)
    assert intent_by_client(eng.db_path, cid)["status"] == "CANCELED"
    assert sell_intent_blocks(eng.db_path, "BTC/USDT") is False
    assert eng._exit_reeval_requested is True
    assert SCALP in checked
    assert eng._live_service.cancel_order.await_count == 1
