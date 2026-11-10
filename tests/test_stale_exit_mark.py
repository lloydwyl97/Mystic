"""A stale mark cannot create a new low or reach a live SELL."""

from __future__ import annotations

import inspect
import math
import time
from unittest.mock import AsyncMock

from backend.services.portfolio_engine import PortfolioEngine, _exit_mark_is_usable, make_position_key
from backend.services.protected_limit_execution import (
    ExecutableBook,
    assess_book_freshness,
    reset_book_identity,
)
from tests.test_two_engine_architecture import DAY, _engine, _lot


def test_exit_mark_rejects_stale_nonfinite_and_nonpositive():
    fresh = {"mark_used": 99.5, "price_source_stale": False}
    assert _exit_mark_is_usable(fresh) is True
    assert _exit_mark_is_usable({"mark_used": 89.0, "price_source_stale": True}) is False
    assert _exit_mark_is_usable({"mark_used": float("nan"), "price_source_stale": False}) is False
    assert _exit_mark_is_usable({"mark_used": 0.0, "price_source_stale": False}) is False
    assert _exit_mark_is_usable({"mark_used": -1.0, "price_source_stale": False}) is False
    assert _exit_mark_is_usable(None) is False
    assert _exit_mark_is_usable({"mark_used": "nope", "price_source_stale": False}) is False
    assert math.isfinite(float(fresh["mark_used"]))


def test_monitor_checks_the_mark_before_watermarks_or_the_exit():
    body = inspect.getsource(PortfolioEngine.monitor_all_positions)
    usable = body.index("_exit_mark_is_usable")
    low = body.index("lowest_price = min")
    high = body.index("fold_high_water")
    check = body.index("_check_exit_conditions")
    assert usable < low
    assert usable < high
    assert usable < check
    assert "EXIT_STALE_MARK_RISK_ONLY" not in body


def _day_monitor(tmp_path, monkeypatch, mark):
    import backend.services.day_high_water as hw

    monkeypatch.setattr(hw, "load_feature_1m_candles", lambda *_a, **_k: [])
    eng = _engine(tmp_path)
    eng.run_trading_circuit_breaker_check = AsyncMock()
    eng._resolve_exit_monitor_mark = AsyncMock(return_value=mark)
    eng._build_exit_check_telemetry = lambda *_a, **_k: {}
    eng._log_exit_check_telemetry = lambda *_a, **_k: None
    eng._persist_position_to_sqlite = AsyncMock()
    eng._emit_day_health_telemetry = AsyncMock()
    eng._learning_heartbeat_last = {"BTC/USDT": time.time()}
    eng.execute_sell_fifo = AsyncMock(return_value=None)
    check = AsyncMock(return_value=None)
    eng._check_exit_conditions = check
    lot = _lot("BTC/USDT", DAY, price=100.0)
    lot.highest_price = 101.0
    lot.lowest_price = 99.0
    lot._last_mark_ts = 1_000.0
    eng.open_positions[make_position_key(DAY, "BTC/USDT")] = lot
    return eng, check, lot


async def test_stale_mark_does_not_create_a_low_or_a_sell(tmp_path, monkeypatch):
    eng, check, lot = _day_monitor(
        tmp_path,
        monkeypatch,
        {"mark_used": 89.0, "price_source_stale": True, "mark_timestamp": 2_000.0, "kline_1m_high": 130.0},
    )
    await eng.monitor_all_positions({}, 0)
    assert check.await_count == 0
    eng.execute_sell_fifo.assert_not_awaited()
    assert lot.lowest_price == 99.0
    assert lot.highest_price == 101.0


async def test_older_observation_does_not_move_a_valid_low(tmp_path, monkeypatch):
    eng, check, lot = _day_monitor(
        tmp_path,
        monkeypatch,
        {"mark_used": 89.0, "price_source_stale": False, "mark_timestamp": 900.0},
    )
    await eng.monitor_all_positions({}, 0)
    assert check.await_count == 0
    assert lot.lowest_price == 99.0
    assert lot.highest_price == 101.0


async def test_fresh_mark_after_a_stale_pass_keeps_the_stored_low(tmp_path, monkeypatch):
    stale = {"mark_used": 89.0, "price_source_stale": True, "mark_timestamp": 2_000.0}
    eng, check, lot = _day_monitor(tmp_path, monkeypatch, stale)
    lot.lowest_price = 90.0
    await eng.monitor_all_positions({}, 0)
    assert lot.lowest_price == 90.0
    eng._resolve_exit_monitor_mark = AsyncMock(return_value={"mark_used": 97.0, "price_source_stale": False, "mark_timestamp": 3_000.0})
    await eng.monitor_all_positions({}, 0)
    assert check.await_count == 1
    assert lot.lowest_price == 90.0
    assert check.await_args.args[1] == 97.0


def test_sell_preflight_rejects_a_stale_book_and_accepts_a_fresh_one():
    reset_book_identity()
    now = 10_000.0
    stale = ExecutableBook(
        bids=[[99.0, 1.0]],
        asks=[[99.1, 1.0]],
        last_update_id=1,
        source_ts=now - 30.0,
        receive_ts=now,
        exchange_ts=now - 30.0,
    )
    assert assess_book_freshness("BTC/USDT", stale, now=now)["freshness_result"] == "STALE"
    reset_book_identity()
    fresh = ExecutableBook(
        bids=[[99.0, 1.0]],
        asks=[[99.1, 1.0]],
        last_update_id=2,
        source_ts=now,
        receive_ts=now,
        exchange_ts=now,
    )
    assert assess_book_freshness("BTC/USDT", fresh, now=now)["freshness_result"] == "FRESH"


def test_stale_profitability_mark_does_not_replace_the_caller_price():
    body = inspect.getsource(PortfolioEngine._execute_sell_fifo_locked)
    assert 'sell_eval.get("mark_fresh") is not False' in body
