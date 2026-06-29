"""Regression: exit monitor must use fresh MTM marks, not stale integration.current_prices."""

from __future__ import annotations

import time
from unittest.mock import AsyncMock, patch

import pytest

from backend.services.day_controlled_exits import EXIT_STOP_LOSS, evaluate_engine_managed_exit
from backend.services.portfolio_engine import OpenPosition, PortfolioEngine, PriceCache


def _xrp_position() -> OpenPosition:
    return OpenPosition(
        symbol="XRP/USDT",
        quantity=3316.8,
        entry_price=1.1137719579112397,
        entry_time=time.time() - 3600 * 5,
        trade_id="mystic_XRP/USDT_test",
        stop_price=1.1086013,
        take_profit_1_price=1.1293647653219971,
        take_profit_2_price=0.0,
        trailing_stop_price=1.124947,
        thesis_invalid_level=1.1086013,
        thesis_target_level=1.1293647653219971,
        highest_price=1.1306,
        lowest_price=1.1008,
        max_hold_min=75,
        trail_pct=0.005,
        entry_thesis="RANGE_BOUNCE",
    )


def _sol_position() -> OpenPosition:
    return OpenPosition(
        symbol="SOL/USDT",
        quantity=52.206,
        entry_price=71.08285829215032,
        entry_time=time.time() - 3600 * 7,
        trade_id="mystic_SOL/USDT_test",
        stop_price=70.3934,
        take_profit_1_price=72.14910116653257,
        take_profit_2_price=0.0,
        trailing_stop_price=71.434935,
        thesis_invalid_level=70.01333889,
        thesis_target_level=72.14910116653257,
        highest_price=71.83,
        lowest_price=68.91,
        max_hold_min=75,
        trail_pct=0.0055,
        entry_thesis="RANGE_BOUNCE",
    )


def test_stale_integration_cache_does_not_trigger_stop_loss():
    """Stale high integration cache must NOT produce STOP_LOSS when fresh mark is below stop."""
    pos = _xrp_position()
    stale_mark = 1.1306  # above stop — mimics stale integration.current_prices
    fresh_mark = 1.1008  # below stop 1.1086013

    stale_eval = evaluate_engine_managed_exit(
        position=pos,
        current_price=stale_mark,
        net_pnl_pct=(stale_mark - pos.entry_price) / pos.entry_price - 0.0012,
        hold_minutes=300.0,
        coin_profile={"trail": 0.005},
        bundle=None,
        bar_low=stale_mark,
    )
    assert stale_eval.get("action") != "sell" or stale_eval.get("reason") != EXIT_STOP_LOSS

    fresh_eval = evaluate_engine_managed_exit(
        position=pos,
        current_price=fresh_mark,
        net_pnl_pct=(fresh_mark - pos.entry_price) / pos.entry_price - 0.0012,
        hold_minutes=300.0,
        coin_profile={"trail": 0.005},
        bundle=None,
        bar_low=fresh_mark,
    )
    assert fresh_eval.get("action") == "sell"
    assert fresh_eval.get("reason") == EXIT_STOP_LOSS


def test_sol_stale_vs_fresh_stop_loss_regression():
    pos = _sol_position()
    stale_mark = 71.83
    fresh_mark = 68.91

    stale_eval = evaluate_engine_managed_exit(
        position=pos,
        current_price=stale_mark,
        net_pnl_pct=(stale_mark - pos.entry_price) / pos.entry_price - 0.001,
        hold_minutes=400.0,
        coin_profile={"trail": 0.0055},
        bundle=None,
        bar_low=stale_mark,
    )
    assert stale_eval.get("reason") != EXIT_STOP_LOSS

    fresh_eval = evaluate_engine_managed_exit(
        position=pos,
        current_price=fresh_mark,
        net_pnl_pct=(fresh_mark - pos.entry_price) / pos.entry_price - 0.001,
        hold_minutes=400.0,
        coin_profile={"trail": 0.0055},
        bundle=None,
        bar_low=fresh_mark,
    )
    assert fresh_eval.get("action") == "sell"
    assert fresh_eval.get("reason") == EXIT_STOP_LOSS


@pytest.mark.asyncio
async def test_resolve_exit_monitor_mark_fail_closed_without_fresh_ticker():
    engine = object.__new__(PortfolioEngine)
    engine._price_cache = PriceCache(ttl_seconds=5)
    engine._live_service = None
    engine._live_execution_enabled = False

    with patch("backend.services.canonical_mark_price.fetch_canonical_mark", new=AsyncMock(return_value=None)):
        mark_info = await engine._resolve_exit_monitor_mark("XRP/USDT")

    assert mark_info["price_source_stale"] is True
    assert mark_info["mark_source"] in ("missing", "price_cache_stale")


@pytest.mark.asyncio
async def test_monitor_all_positions_stop_loss_uses_fresh_mtm_not_stale_cache():
    """Exact XRP failure: stale integration cache high, fresh canonical mark below stop → STOP_LOSS_EXIT."""
    from backend.services.canonical_mark_price import CanonicalMark

    engine = object.__new__(PortfolioEngine)
    engine.open_positions = {"XRP/USDT": _xrp_position()}
    engine._price_cache = PriceCache(ttl_seconds=5)
    engine._live_service = None
    engine._live_execution_enabled = False
    engine._exit_mark_price_source_stale = False
    engine._learning_heartbeat_last = {}
    engine._emit_day_health_telemetry = AsyncMock()
    engine._persist_position_to_sqlite = AsyncMock()

    fresh = CanonicalMark(
        symbol="XRP/USDT",
        symbol_format="XRPUSDT",
        mark=1.1008,
        bid=1.1007,
        ask=1.1009,
        mid=1.1008,
        last=1.1008,
        source="binance_book_ticker_mid",
        timestamp=time.time(),
        age_seconds=0.0,
        fresh=True,
    )

    sell_calls: list[str] = []

    async def _mock_sell(symbol, quantity, price, exit_type, exit_trigger, **kwargs):
        sell_calls.append(str(exit_trigger))
        return {"symbol": symbol, "exit_trigger": exit_trigger}

    engine.execute_sell_fifo = _mock_sell

    stale_integration_prices = {"XRP/USDT": 1.1306}
    current_bar = int(time.time() / 60) * 60

    with patch("backend.services.canonical_mark_price.fetch_canonical_mark", new=AsyncMock(return_value=fresh)):
        with patch("backend.services.ai_learning_ingestion.record_position_heartbeat"):
            exits = await engine.monitor_all_positions(stale_integration_prices, current_bar)

    assert sell_calls == [EXIT_STOP_LOSS]
    assert exits
    assert engine._exit_mark_price_source_stale is False


@pytest.mark.asyncio
async def test_exit_check_telemetry_flags_should_stop_loss_on_fresh_mark():
    engine = object.__new__(PortfolioEngine)
    pos = _xrp_position()
    mark_info = {
        "mark_used": 1.1008,
        "mark_source": "binance_book_ticker_mid",
        "mark_timestamp": time.time(),
        "mark_age_seconds": 0.0,
        "price_source_stale": False,
        "stale_mark_used": False,
        "bid": 1.1007,
        "ask": 1.1009,
        "mid": 1.1008,
        "last": 1.1008,
        "canonical_source": "binance_book_ticker_mid",
        "symbol_format": "XRPUSDT",
    }
    telemetry = engine._build_exit_check_telemetry(pos, mark_info)
    assert telemetry["should_stop_loss_fire"] is True
    assert telemetry["exit_reason"] == EXIT_STOP_LOSS
    assert telemetry["stale_mark_used"] is False
