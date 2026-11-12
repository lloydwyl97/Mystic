"""One lot's exception must not skip the other lots in the same exit pass."""

from __future__ import annotations

import asyncio
import time
from unittest.mock import AsyncMock

import pytest

from backend.services.portfolio_engine import make_position_key
from tests.test_two_engine_architecture import DAY, SCALP, _engine, _lot

MARK = {"mark_used": 100.0, "price_source_stale": False, "mark_timestamp": time.time()}


def _monitor(tmp_path, monkeypatch, symbols: list[tuple[str, str]]):
    import backend.services.day_high_water as hw

    monkeypatch.setattr(hw, "load_feature_1m_candles", lambda *_a, **_k: [])
    eng = _engine(tmp_path)
    eng.run_trading_circuit_breaker_check = AsyncMock()
    eng._build_exit_check_telemetry = lambda *_a, **_k: {}
    eng._log_exit_check_telemetry = lambda *_a, **_k: None
    eng._persist_position_to_sqlite = AsyncMock()
    eng._emit_day_health_telemetry = AsyncMock()
    eng._learning_heartbeat_last = {symbol: time.time() for _engine_id, symbol in symbols}
    progressed: list[str] = []

    async def _progress(symbol, allow_cancel=True):
        progressed.append(symbol)

    eng._progress_working_sell = _progress
    checked: list[str] = []

    async def _check(position, *_a, **_k):
        checked.append(str(position.symbol))
        if getattr(position, "_boom", False):
            raise RuntimeError("position-local")

    eng._check_exit_conditions = _check

    async def _mark(symbol):
        if symbol == "BTC/USDT":
            raise RuntimeError("btc-mark")
        return dict(MARK)

    eng._resolve_exit_monitor_mark = _mark
    for engine_id, symbol in symbols:
        lot = _lot(symbol, engine_id)
        lot.highest_price = 101.0
        lot.lowest_price = 99.0
        eng.open_positions[make_position_key(engine_id, symbol)] = lot
    return eng, checked, progressed


@pytest.mark.asyncio
async def test_first_position_exception_does_not_skip_the_rest(tmp_path, monkeypatch):
    eng, checked, progressed = _monitor(
        tmp_path,
        monkeypatch,
        [(DAY, "BTC/USDT"), (DAY, "ETH/USDT"), (DAY, "SOL/USDT"), (DAY, "XRP/USDT")],
    )
    eng._resolve_exit_monitor_mark = AsyncMock(return_value=dict(MARK))

    async def _check(position, *_a, **_k):
        checked.append(str(position.symbol))
        if position.symbol == "BTC/USDT":
            raise RuntimeError("btc")

    eng._check_exit_conditions = _check
    await eng.monitor_all_positions({}, 0)
    assert checked == ["BTC/USDT", "ETH/USDT", "SOL/USDT", "XRP/USDT"]
    assert progressed == ["BTC/USDT", "ETH/USDT", "SOL/USDT", "XRP/USDT"]


@pytest.mark.asyncio
async def test_middle_and_last_exceptions_leave_the_others(tmp_path, monkeypatch):
    eng, checked, _progressed = _monitor(
        tmp_path,
        monkeypatch,
        [(DAY, "BTC/USDT"), (DAY, "ETH/USDT"), (DAY, "SOL/USDT"), (DAY, "XRP/USDT")],
    )
    eng._resolve_exit_monitor_mark = AsyncMock(return_value=dict(MARK))

    async def _check(position, *_a, **_k):
        checked.append(str(position.symbol))
        if position.symbol == "ETH/USDT":
            raise RuntimeError("eth")

    eng._check_exit_conditions = _check
    await eng.monitor_all_positions({}, 0)
    assert checked == ["BTC/USDT", "ETH/USDT", "SOL/USDT", "XRP/USDT"]

    checked.clear()

    async def _check_last(position, *_a, **_k):
        checked.append(str(position.symbol))
        if position.symbol == "XRP/USDT":
            raise RuntimeError("xrp")

    eng._check_exit_conditions = _check_last
    await eng.monitor_all_positions({}, 0)
    assert checked == ["BTC/USDT", "ETH/USDT", "SOL/USDT", "XRP/USDT"]


@pytest.mark.asyncio
async def test_repeated_btc_failure_still_evaluates_eth(tmp_path, monkeypatch):
    eng, checked, _progressed = _monitor(tmp_path, monkeypatch, [(DAY, "BTC/USDT"), (DAY, "ETH/USDT")])
    eng._resolve_exit_monitor_mark = AsyncMock(return_value=dict(MARK))

    async def _check(position, *_a, **_k):
        checked.append(str(position.symbol))
        if position.symbol == "BTC/USDT":
            raise RuntimeError("btc")
        return {"symbol": "ETH/USDT", "protected": True}

    eng._check_exit_conditions = _check
    for _ in range(5):
        result = await eng.monitor_all_positions({}, 0)
        assert result == [{"symbol": "ETH/USDT", "protected": True}]
    assert checked.count("ETH/USDT") == 5
    assert checked.count("BTC/USDT") == 5


@pytest.mark.asyncio
async def test_day_exception_does_not_skip_scalp_progress(tmp_path, monkeypatch):
    eng, checked, progressed = _monitor(
        tmp_path,
        monkeypatch,
        [(DAY, "BTC/USDT"), (SCALP, "SOL/USDT")],
    )

    async def _scalp_quote(symbol, executable_quote_only=False):
        return dict(MARK), 100.0

    eng._scalp_exit_quote = _scalp_quote
    eng._resolve_exit_monitor_mark = AsyncMock(side_effect=RuntimeError("day-mark"))
    await eng.monitor_all_positions({}, 0)
    assert "SOL/USDT" in checked
    assert "SOL/USDT" in progressed
    assert "BTC/USDT" not in checked


@pytest.mark.asyncio
async def test_mark_failure_on_the_first_lot_still_progresses_the_rest(tmp_path, monkeypatch):
    eng, checked, progressed = _monitor(
        tmp_path,
        monkeypatch,
        [(DAY, "BTC/USDT"), (DAY, "ETH/USDT"), (DAY, "SOL/USDT"), (DAY, "XRP/USDT")],
    )
    await eng.monitor_all_positions({}, 0)
    assert checked == ["ETH/USDT", "SOL/USDT", "XRP/USDT"]
    assert progressed == ["ETH/USDT", "SOL/USDT", "XRP/USDT"]
    assert eng._last_exit_pass_stats == {"evaluated": 3, "failed": 1}


@pytest.mark.asyncio
async def test_watermark_failure_does_not_skip_protection(tmp_path, monkeypatch):
    eng, checked, _progressed = _monitor(tmp_path, monkeypatch, [(DAY, "BTC/USDT"), (DAY, "ETH/USDT")])
    eng._resolve_exit_monitor_mark = AsyncMock(return_value=dict(MARK))
    for lot in eng.open_positions.values():
        lot.lowest_price = 0.0

    async def _persist(position):
        if position.symbol == "BTC/USDT":
            raise RuntimeError("watermark")

    eng._persist_position_to_sqlite = _persist
    await eng.monitor_all_positions({}, 0)
    assert checked == ["BTC/USDT", "ETH/USDT"]
    assert eng._last_exit_pass_stats == {"evaluated": 2, "failed": 0}


@pytest.mark.asyncio
async def test_residual_lot_still_runs_when_an_earlier_lot_throws(tmp_path, monkeypatch):
    eng, checked, progressed = _monitor(
        tmp_path,
        monkeypatch,
        [(DAY, "BTC/USDT"), (DAY, "SOL/USDT")],
    )
    eng.open_positions[make_position_key(DAY, "SOL/USDT")].status = "EXIT_RESIDUAL_PENDING"
    await eng.monitor_all_positions({}, 0)
    assert checked == ["SOL/USDT"]
    assert progressed == ["SOL/USDT"]


@pytest.mark.asyncio
async def test_one_sell_attempt_when_the_check_raises_and_the_next_lot_runs(tmp_path, monkeypatch):
    eng, checked, progressed = _monitor(tmp_path, monkeypatch, [(DAY, "BTC/USDT"), (DAY, "ETH/USDT")])
    eng._resolve_exit_monitor_mark = AsyncMock(return_value=dict(MARK))
    sells: list[str] = []

    async def _check(position, *_a, **_k):
        checked.append(str(position.symbol))
        if position.symbol == "BTC/USDT":
            sells.append(position.symbol)
            raise RuntimeError("after-submit")

    eng._check_exit_conditions = _check
    await eng.monitor_all_positions({}, 0)
    assert sells == ["BTC/USDT"]
    assert checked == ["BTC/USDT", "ETH/USDT"]
    assert progressed == ["BTC/USDT", "ETH/USDT"]


@pytest.mark.asyncio
async def test_cancellation_is_not_swallowed(tmp_path, monkeypatch):
    eng, checked, _progressed = _monitor(tmp_path, monkeypatch, [(DAY, "BTC/USDT"), (DAY, "ETH/USDT")])

    async def _mark(_symbol):
        raise asyncio.CancelledError

    eng._resolve_exit_monitor_mark = _mark
    with pytest.raises(asyncio.CancelledError):
        await eng.monitor_all_positions({}, 0)
    assert checked == []
