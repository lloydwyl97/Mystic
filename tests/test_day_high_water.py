"""DAY high-water must capture brief live highs between slower mid samples."""

from __future__ import annotations

import time
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from backend.services.canonical_mark_price import CanonicalMark
from backend.services.day_controlled_exits import apply_break_even_and_mfe_trail, refresh_trailing_stop
from backend.services.day_high_water import (
    first_full_minute_epoch,
    fold_high_water,
    max_post_entry_1m_high,
    usable_kline_high,
)
from backend.services.portfolio_engine import OpenPosition, PortfolioEngine


def test_fold_high_water_is_monotonic():
    assert fold_high_water(80428.975, 80399.41) == pytest.approx(80428.975)
    assert fold_high_water(80428.975, 80399.41, 80435.11) == pytest.approx(80435.11)
    assert fold_high_water(80435.11, 80200.0, None, 0.0) == pytest.approx(80435.11)


def test_first_full_minute_skips_entry_candle():
    entry = datetime_epoch("2026-08-27T21:00:12Z")
    assert first_full_minute_epoch(entry) == datetime_epoch("2026-08-27T21:01:00Z")


def datetime_epoch(iso: str) -> float:
    from datetime import datetime, timezone

    return datetime.fromisoformat(iso.replace("Z", "+00:00")).timestamp()


def test_max_post_entry_1m_high_ignores_pre_entry_and_keeps_wick():
    entry = datetime_epoch("2026-08-27T21:00:12Z")
    candles = [
        {"open_time": datetime_epoch("2026-08-27T21:00:00Z") * 1000, "high": 81000.0},
        {"open_time": datetime_epoch("2026-08-27T22:18:00Z") * 1000, "high": 80435.11},
        {"open_time": datetime_epoch("2026-08-27T22:20:00Z") * 1000, "high": 80432.3},
    ]
    assert max_post_entry_1m_high(entry, candles) == pytest.approx(80435.11)


def test_entry_minute_pre_entry_wick_not_captured():
    """Synthetic: entry=100, same-minute pre-entry high=105, post-entry high=101 → 101."""
    entry = datetime_epoch("2026-08-27T21:00:12Z")
    candles = [
        {"open_time": datetime_epoch("2026-08-27T21:00:00Z") * 1000, "high": 105.0},
        {"open_time": datetime_epoch("2026-08-27T21:01:00Z") * 1000, "high": 101.0},
    ]
    reconstructed = fold_high_water(100.0, max_post_entry_1m_high(entry, candles))
    assert reconstructed == pytest.approx(101.0)
    assert reconstructed != pytest.approx(105.0)


def test_entry_minute_live_kline_high_rejected_without_trade_stamp():
    entry = datetime_epoch("2026-08-27T21:00:12Z")
    entry_minute_open = datetime_epoch("2026-08-27T21:00:00Z")
    first_full = datetime_epoch("2026-08-27T21:01:00Z")
    assert usable_kline_high(entry, entry_minute_open, 105.0) is None
    assert usable_kline_high(entry, None, 105.0) is None
    assert usable_kline_high(entry, first_full, 101.0) == pytest.approx(101.0)
    stored = fold_high_water(100.0, 100.4, usable_kline_high(entry, entry_minute_open, 105.0))
    assert stored == pytest.approx(100.4)


def test_persist_only_entry_minute_flush_does_not_leak():
    """feature_ohlcv ts is persist-now; an entry-minute bar flushed at 21:01:05 must not leak."""
    entry = datetime_epoch("2026-08-27T21:00:12Z")
    candles = [
        {"ts": "2026-08-27T21:01:05+00:00", "high": 105.0},
        {"ts": "2026-08-27T21:02:10+00:00", "high": 101.0},
    ]
    assert max_post_entry_1m_high(entry, candles) == pytest.approx(101.0)


def test_trail_stays_unarmed_when_valid_high_below_activation():
    pos = SimpleNamespace(
        symbol="XRP/USDT",
        entry_price=1.4542,
        highest_price=1.4542,
        trailing_stop_price=1.439658,
        stop_price=1.3959,
        trail_pct=0.005,
        day_route_regime_at_entry="",
    )
    pos.highest_price = fold_high_water(pos.highest_price, 1.455, 1.4579)
    refresh_trailing_stop(pos, 1.455, {"trail": 0.005, "sl": 0.010})
    assert pos.highest_price == pytest.approx(1.4579)
    assert pos.highest_price < 1.4542 * 1.005
    assert pos.trailing_stop_price == pytest.approx(1.439658)
    assert pos.trail_pct == pytest.approx(0.005)


def test_btc_wick_between_mid_samples_captures_high_and_arms():
    """Mark mid stays below activation; 1m high crosses it. Trail arms. No sell."""
    entry = 80114.0
    pos = SimpleNamespace(
        symbol="BTC/USDT",
        entry_price=entry,
        highest_price=80428.975,
        trailing_stop_price=80154.057,
        stop_price=80154.057,
        trail_pct=0.004,
        day_route_regime_at_entry="bull",
    )
    mark = 80399.41
    kline_high = 80435.11
    pos.highest_price = fold_high_water(pos.highest_price, mark, kline_high)
    assert pos.highest_price == pytest.approx(80435.11)
    activation = entry * 1.004
    assert pos.highest_price >= activation
    profile = {"trail": 0.0040, "sl": 0.008, "max_hold_min": 360}
    refresh_trailing_stop(pos, mark, profile)
    raw = 80435.11 * 0.996
    be = entry * 1.0005
    assert raw < be
    assert pos.trailing_stop_price == pytest.approx(be)
    assert pos.trail_pct == pytest.approx(0.004)
    # mid is still above BE — activation does not sell
    assert mark > pos.trailing_stop_price


def test_constant_profiles_unchanged_after_high_fold():
    cases = [
        ("BTC/USDT", 0.0040, 80114.0, 80435.11),
        ("ETH/USDT", 0.0045, 2500.0, 2520.0),
        ("SOL/USDT", 0.0055, 100.0, 101.0),
        ("XRP/USDT", 0.0050, 1.4542, 1.4579),
    ]
    for symbol, trail, entry, high in cases:
        pos = SimpleNamespace(
            symbol=symbol,
            entry_price=entry,
            highest_price=entry,
            trailing_stop_price=0.0,
            stop_price=entry * 0.99,
            trail_pct=trail,
            day_route_regime_at_entry="",
        )
        pos.highest_price = fold_high_water(pos.highest_price, entry + 1.0, high)
        refresh_trailing_stop(pos, entry + 1.0, {"trail": trail, "sl": 0.01})
        apply_break_even_and_mfe_trail(pos, high)
        assert pos.trail_pct == pytest.approx(trail)
        if pos.highest_price >= entry * (1.0 + trail) and pos.trailing_stop_price:
            dist = (pos.highest_price - pos.trailing_stop_price) / pos.highest_price
            be = entry * 1.0005
            if pos.trailing_stop_price > be + 1e-9:
                assert dist == pytest.approx(trail, abs=1e-6)


@pytest.mark.asyncio
async def test_monitor_folds_kline_high_and_does_not_sell_on_activation(tmp_path, monkeypatch):
    monkeypatch.setenv("DAY_PATH_AWARE_EXIT", "true")
    engine = PortfolioEngine.__new__(PortfolioEngine)
    engine.db_path = str(tmp_path / "hw.db")
    engine.open_positions = {}
    engine._price_cache = type("C", (), {"set": lambda *_a, **_k: None, "get_with_age": lambda *_a, **_k: (None, None)})()
    engine._live_service = None
    engine._live_execution_enabled = False
    engine._exit_mark_price_source_stale = False
    engine._learning_heartbeat_last = {}
    engine._emit_day_health_telemetry = AsyncMock()
    engine._persist_position_to_sqlite = AsyncMock()
    engine.run_trading_circuit_breaker_check = AsyncMock()
    engine.execute_sell_fifo = AsyncMock(return_value=None)

    pos = OpenPosition(
        symbol="BTC/USDT",
        quantity=0.00081,
        entry_price=80114.0,
        entry_time=datetime_epoch("2026-08-27T21:00:12Z"),
        trade_id="mystic_BTC/USDT_hw_test",
        stop_price=80154.057,
        take_profit_1_price=81075.0,
        take_profit_2_price=0.0,
        trailing_stop_price=80154.057,
        highest_price=80428.975,
        lowest_price=79999.42,
        trail_pct=0.004,
        max_hold_min=426,
        entry_thesis="HTF_TREND_PULLBACK",
        day_route_regime_at_entry="bull",
    )
    engine.open_positions = {"BTC/USDT": pos}

    mark = CanonicalMark(
        symbol="BTC/USDT",
        symbol_format="BTCUSDT",
        mark=80399.41,
        bid=80399.40,
        ask=80399.42,
        mid=80399.41,
        last=80399.41,
        source="binance_book_ticker_mid",
        timestamp=time.time(),
        age_seconds=0.0,
        fresh=True,
        kline_1m_close=80403.83,
        kline_1m_high=80435.11,
        kline_1m_open_time=datetime_epoch("2026-08-27T22:18:00Z") * 1000,
    )

    with (
        patch("backend.services.canonical_mark_price.fetch_canonical_mark", new=AsyncMock(return_value=mark)),
        patch("backend.services.day_high_water.load_feature_1m_candles", return_value=[]),
        patch("backend.services.ai_learning_ingestion.record_position_heartbeat"),
    ):
        exits = await engine.monitor_all_positions({"BTC/USDT": 80399.41}, int(time.time()))

    assert pos.highest_price == pytest.approx(80435.11)
    assert pos.highest_price >= 80114.0 * 1.004
    assert pos.trailing_stop_price == pytest.approx(80114.0 * 1.0005)
    assert pos.trail_pct == pytest.approx(0.004)
    assert engine.execute_sell_fifo.await_count == 0
    assert exits == []


@pytest.mark.asyncio
async def test_monitor_rejects_entry_minute_kline_high(tmp_path, monkeypatch):
    monkeypatch.setenv("DAY_PATH_AWARE_EXIT", "true")
    engine = PortfolioEngine.__new__(PortfolioEngine)
    engine.db_path = str(tmp_path / "hw2.db")
    engine._price_cache = type("C", (), {"set": lambda *_a, **_k: None, "get_with_age": lambda *_a, **_k: (None, None)})()
    engine._live_service = None
    engine._live_execution_enabled = False
    engine._exit_mark_price_source_stale = False
    engine._learning_heartbeat_last = {}
    engine._emit_day_health_telemetry = AsyncMock()
    engine._persist_position_to_sqlite = AsyncMock()
    engine.run_trading_circuit_breaker_check = AsyncMock()
    engine.execute_sell_fifo = AsyncMock(return_value=None)

    pos = OpenPosition(
        symbol="BTC/USDT",
        quantity=0.00081,
        entry_price=100.0,
        entry_time=datetime_epoch("2026-08-27T21:00:12Z"),
        trade_id="mystic_BTC/USDT_entry_min",
        stop_price=99.0,
        take_profit_1_price=102.0,
        take_profit_2_price=0.0,
        trailing_stop_price=99.0,
        highest_price=100.0,
        lowest_price=99.8,
        trail_pct=0.004,
        max_hold_min=426,
        entry_thesis="HTF_TREND_PULLBACK",
        day_route_regime_at_entry="",
    )
    engine.open_positions = {"BTC/USDT": pos}
    mark = CanonicalMark(
        symbol="BTC/USDT",
        symbol_format="BTCUSDT",
        mark=100.4,
        bid=100.3,
        ask=100.5,
        mid=100.4,
        last=100.4,
        source="binance_book_ticker_mid",
        timestamp=time.time(),
        age_seconds=0.0,
        fresh=True,
        kline_1m_close=100.4,
        kline_1m_high=105.0,
        kline_1m_open_time=datetime_epoch("2026-08-27T21:00:00Z") * 1000,
    )
    with (
        patch("backend.services.canonical_mark_price.fetch_canonical_mark", new=AsyncMock(return_value=mark)),
        patch("backend.services.day_high_water.load_feature_1m_candles", return_value=[]),
        patch("backend.services.ai_learning_ingestion.record_position_heartbeat"),
    ):
        await engine.monitor_all_positions({"BTC/USDT": 100.4}, int(time.time()))
    assert pos.highest_price == pytest.approx(100.4)
    assert pos.highest_price < 105.0
    assert engine.execute_sell_fifo.await_count == 0


def test_restart_backfill_reconstructs_missed_wick():
    entry = datetime_epoch("2026-08-27T21:00:12Z")
    stored = 80428.975
    candles = [{"ts": "2026-08-27T22:18:00+00:00", "high": 80435.11}]
    recovered = fold_high_water(stored, max_post_entry_1m_high(entry, candles))
    assert recovered == pytest.approx(80435.11)
    assert recovered >= stored
