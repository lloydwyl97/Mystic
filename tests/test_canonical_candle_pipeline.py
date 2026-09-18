"""Deterministic tests for the canonical candle pipeline and 4H/slot contracts."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from backend.config.canonical_candle_intervals import (
    CANONICAL_CANDLE_INTERVALS,
    CANONICAL_SYMBOLS,
    TELEMETRY_ONLY_NO_TRADE_AUTHORITY,
    align_open_ms,
    interval_ms,
)
from backend.config.timeframe_consumer_matrix import TIMEFRAME_CONSUMER_MATRIX
from backend.services.canonical_candle_pipeline import WRITER_ROLE, CanonicalCandlePipeline
from backend.services.canonical_candle_store import (
    aggregate_exact_from_1m,
    candle_dict,
    expected_open_ms_range,
    refuse_research_table_read,
    row_from_binance_kline,
)


def _kline(open_ms: int, o=100.0, h=101.0, low=99.0, c=100.5, v=1.25) -> list:
    return [open_ms, str(o), str(h), str(low), str(c), str(v), 0, "0", 0, "0", "0", "0"]


@pytest.mark.parametrize("interval", CANONICAL_CANDLE_INTERVALS)
def test_utc_alignment_every_interval(interval: str):
    raw = 1_700_000_123_456
    aligned = align_open_ms(raw, interval)
    assert aligned % interval_ms(interval) == 0
    assert aligned <= raw


def test_zero_volume_legitimate_candle_is_kept():
    now = 1_700_000_120_000
    row = row_from_binance_kline(_kline(1_700_000_040_000, v=0.0), symbol="BTCUSDT", interval="1m", now_ms=now)
    assert row is not None
    assert row["volume"] == 0.0
    assert row["zero_volume_legitimate"] is True
    assert row["completed"] is True


def test_forming_vs_completed_separation():
    now = 1_700_000_080_000
    forming_open = align_open_ms(now, "1m")
    closed_open = forming_open - 60_000
    closed = row_from_binance_kline(_kline(closed_open), symbol="ETHUSDT", interval="1m", now_ms=now)
    forming = row_from_binance_kline(_kline(forming_open), symbol="ETHUSDT", interval="1m", now_ms=now)
    assert closed["completed"] is True
    assert forming["forming"] is True


def test_3m_aggregation_requires_exact_three_1m():
    start = align_open_ms(1_700_000_000_000, "3m")
    bars = [
        candle_dict(symbol="SOLUSDT", interval="1m", open_ms=start, open_=1, high=2, low=0.5, close=1.5, volume=1, completed=True),
        candle_dict(symbol="SOLUSDT", interval="1m", open_ms=start + 60_000, open_=1.5, high=3, low=1.4, close=2.0, volume=2, completed=True),
        candle_dict(symbol="SOLUSDT", interval="1m", open_ms=start + 120_000, open_=2.0, high=2.2, low=1.8, close=2.1, volume=3, completed=True),
    ]
    out = aggregate_exact_from_1m(bars, "3m")
    assert len(out) == 1
    assert out[0]["open"] == 1
    assert out[0]["high"] == 3
    assert out[0]["low"] == 0.5
    assert out[0]["close"] == 2.1
    assert out[0]["volume"] == 6
    missing = bars[:2]
    assert aggregate_exact_from_1m(missing, "3m") == []


def test_expected_range_has_no_lookahead():
    start = align_open_ms(1_700_000_000_000, "1m")
    end = start + 4 * 60_000
    got = expected_open_ms_range(start, end, "1m")
    assert got[0] == start
    assert got[-1] == end
    assert all(ts <= end for ts in got)


@pytest.mark.asyncio
async def test_ingest_dedup_and_out_of_order(monkeypatch):
    pipe = CanonicalCandlePipeline()
    stored: list[dict] = []

    def fake_upsert(symbol, interval, candles):
        by_ts = {c["open_ms"]: c for c in stored}
        for c in candles:
            by_ts[c["open_ms"]] = c
        stored.clear()
        stored.extend(by_ts[ts] for ts in sorted(by_ts))
        return {"inserted": 1, "updated": 0, "skipped_forming": 0}

    monkeypatch.setattr("backend.services.canonical_candle_pipeline.upsert_completed_candles", fake_upsert)
    pipe.publish_redis = AsyncMock()
    now = align_open_ms(1_700_000_180_000, "1m")
    first = align_open_ms(1_700_000_000_000, "1m")
    second = first + 60_000
    pipe._now_ms = lambda: now  # type: ignore[method-assign]
    await pipe.ingest_klines("BTCUSDT", "1m", [_kline(second, c=11), _kline(first, c=10)], persist=True)
    await pipe.ingest_klines("BTCUSDT", "1m", [_kline(first, c=10.5)], persist=True)
    assert [c["open_ms"] for c in stored] == [first, second]
    assert stored[0]["close"] == 10.5


@pytest.mark.asyncio
async def test_backfill_and_gap_repair(monkeypatch):
    pipe = CanonicalCandlePipeline()
    start = align_open_ms(1_700_000_000_000, "1m")
    pages = {
        start: [_kline(start), _kline(start + 120_000)],
        start + 60_000: [_kline(start + 60_000)],
    }

    async def fake_fetch(symbol, interval, start_ms=None, end_ms=None, limit=1000):
        if start_ms in pages:
            return pages[start_ms]
        return [_kline(start_ms)] if start_ms else []

    stored: dict[int, dict] = {}

    def fake_upsert(symbol, interval, candles):
        for c in candles:
            stored[c["open_ms"]] = c
        return {"inserted": len(candles), "updated": 0, "skipped_forming": 0}

    def fake_load(symbol, interval, start_ms=None, end_ms=None, limit=None):
        return [stored[ts] for ts in sorted(stored) if (start_ms is None or ts >= start_ms) and (end_ms is None or ts <= end_ms)]

    monkeypatch.setattr(pipe, "fetch_binance", fake_fetch)
    monkeypatch.setattr("backend.services.canonical_candle_pipeline.upsert_completed_candles", fake_upsert)
    monkeypatch.setattr("backend.services.canonical_candle_pipeline.load_aligned_candles", fake_load)
    monkeypatch.setattr("backend.services.canonical_candle_store.load_aligned_candles", fake_load)
    pipe.publish_redis = AsyncMock()
    pipe._now_ms = lambda: start + 300_000  # type: ignore[method-assign]
    await pipe.backfill_range("XRPUSDT", "1m", start, start + 120_000)
    report = await pipe.repair_gaps("XRPUSDT", "1m", start, start + 120_000)
    assert start + 60_000 in stored
    assert report["after"]["missing_count"] == 0


@pytest.mark.asyncio
async def test_redis_flush_recovery_and_single_writer(monkeypatch):
    pipe = CanonicalCandlePipeline()
    redis_store: dict[str, str] = {}

    class FakeRedis:
        async def get(self, key):
            return redis_store.get(key)

        def pipeline(self):
            return self

        def set(self, key, val, ex=None):
            redis_store[key] = val
            return self

        async def execute(self):
            return True

    pipe._redis = AsyncMock(return_value=FakeRedis())

    def _upsert(*_args, **_kwargs):
        return {"inserted": 1, "updated": 0, "skipped_forming": 0}

    monkeypatch.setattr("backend.services.canonical_candle_pipeline.upsert_completed_candles", _upsert)
    now = 1_700_000_180_000
    pipe._now_ms = lambda: now  # type: ignore[method-assign]
    await pipe.ingest_klines("BTCUSDT", "5m", [_kline(align_open_ms(now - 300_000, "5m"), v=4.0)], persist=True)
    redis_store.clear()
    await pipe.ingest_klines("BTCUSDT", "5m", [_kline(align_open_ms(now - 300_000, "5m"), v=4.0)], persist=True)
    assert any(k.startswith("klines:BTCUSDT:5m") for k in redis_store)
    assert WRITER_ROLE == "canonical_candle_pipeline"


def test_research_table_is_retired():
    payload = refuse_research_table_read()
    assert payload["retired"] is True
    assert "september_2" in payload["reason"]


def test_consumer_matrix_documents_15m_shape_and_book_path():
    assert TIMEFRAME_CONSUMER_MATRIX["candle_shape_body_wick"]["timeframe"] == "15m"
    assert "order book" in TIMEFRAME_CONSUMER_MATRIX["trailing_buy_executable_bid_ask"]["source"]
    assert TIMEFRAME_CONSUMER_MATRIX["telemetry_only_4h"]["notes"] == TELEMETRY_ONLY_NO_TRADE_AUTHORITY


def test_all_four_coins_and_intervals_listed():
    assert CANONICAL_SYMBOLS == ("BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT")
    for tf in ("1m", "3m", "5m", "15m", "30m", "1h", "4h"):
        assert tf in CANONICAL_CANDLE_INTERVALS


def test_no_4h_buy_block():
    from backend.services.day_controlled_exits import evaluate_completed_4h_buy_hard_safety
    from backend.services.portfolio_engine import PortfolioEngine

    engine = PortfolioEngine.__new__(PortfolioEngine)
    for sym in ("BTC/USDT", "ETH/USDT", "SOL/USDT", "XRP/USDT"):
        blocked, reason = engine._intact_4h_slot_block(sym)
        assert blocked is False
        assert reason == ""
    hard = evaluate_completed_4h_buy_hard_safety(mark=100.0, bundle={"4h": [[1, 1, 1, 1, 0.5, 1], [2, 1, 1, 1, 0.4, 1]]})
    assert hard["allowed"] is True
    assert hard["authority"] == TELEMETRY_ONLY_NO_TRADE_AUTHORITY


def test_no_4h_sell_or_path_aware_restore(monkeypatch):
    from backend.services.day_controlled_exits import DAY_FULL_FLATTEN_REASONS, _evaluate_path_aware_exit, evaluate_engine_managed_exit

    monkeypatch.setenv("DAY_PATH_AWARE_EXIT", "true")
    pos = MagicMock()
    pos.entry_price = 100.0
    pos.highest_price = 100.2
    pos.trailing_stop_price = 0.0
    pos.trail_pct = 0.005
    pos.thesis_invalid_level = 90.0
    pos.lowest_price = 99.0
    pos.stop_price = 0.0
    pos.take_profit_1_price = 0.0
    pos.entry_thesis = ""
    bundle = {"4h": [[1, 100, 105, 95, 102, 1], [2, 101, 103, 90, 89, 1]]}
    path = _evaluate_path_aware_exit(
        position=pos,
        current_price=89.0,
        net_pnl_pct=-0.01,
        hold_minutes=10,
        coin_profile={"trail": 0.0025, "max_hold_min": 300},
        bundle=bundle,
        entry=100.0,
        atr_pct=0.01,
    )
    assert path["reason"] != "DAY_4H_STRUCTURE_BREAK_EXIT"
    assert path["action"] != "sell" or path["reason"] != "DAY_4H_STRUCTURE_BREAK_EXIT"
    managed = evaluate_engine_managed_exit(
        position=pos,
        current_price=89.0,
        net_pnl_pct=-0.01,
        hold_minutes=10,
        coin_profile={"trail": 0.0025, "tp": 0.014, "sl": 0.01, "max_hold_min": 300},
        bundle=bundle,
    )
    assert managed.get("reason") != "DAY_4H_STRUCTURE_BREAK_EXIT"
    assert "DAY_4H_STRUCTURE_BREAK_EXIT" not in DAY_FULL_FLATTEN_REASONS


def test_dust_does_not_occupy_slots():
    from backend.services.portfolio_engine import PortfolioEngine

    engine = PortfolioEngine.__new__(PortfolioEngine)
    dust = MagicMock()
    dust.status = "DUST_PENDING"
    dust.quantity = 0.001
    live = MagicMock()
    live.status = "ACTIVE"
    live.quantity = 0.02
    engine.open_positions = {"BTC/USDT": dust, "ETH/USDT": dust, "SOL/USDT": live, "XRP/USDT": dust}
    engine._pending_buy_symbols = lambda: []
    assert engine._count_live_slots() == 1


def test_fee_display_is_2_bps_not_20():
    from backend.config.trading_economics import _fee_fraction_to_bps

    assert _fee_fraction_to_bps(0.0002) == 2.0
    assert _fee_fraction_to_bps(0.002) == 2.0
    assert _fee_fraction_to_bps(0.02) == 2.0


def test_trailing_buy_remains_callable():
    import inspect

    from backend.services.day_trailing_buy import _pre_submit_safety

    assert callable(_pre_submit_safety)
    src = inspect.getsource(_pre_submit_safety)
    assert "COMPLETED_4H_ALREADY_INVALID" not in src


@pytest.mark.asyncio
async def test_refresh_live_skips_full_history_before_hydrate(monkeypatch):
    pipe = CanonicalCandlePipeline()
    pipe._cached_start_ms = 1_700_000_000_000
    pipe._hydrate_done = False
    pipe.fetch_binance = AsyncMock(return_value=[_kline(1_700_000_000_000)])
    pipe.ingest_klines = AsyncMock(return_value={"completed": 1, "forming": 0})
    pipe.repair_gaps = AsyncMock(return_value={})
    await pipe.refresh_live("BTCUSDT", "1m")
    pipe.repair_gaps.assert_not_awaited()
    pipe._hydrate_done = True
    pipe._last_backfill.clear()
    await pipe.refresh_live("BTCUSDT", "1m")
    assert pipe.repair_gaps.await_count == 1
    start_ms = pipe.repair_gaps.await_args.args[2]
    assert start_ms >= pipe._recent_window_start_ms("1m")


@pytest.mark.asyncio
async def test_api_contract_empty_is_error(monkeypatch):
    from backend.services.canonical_candle_pipeline import get_canonical_candles

    def _empty(*_args, **_kwargs):
        return []

    monkeypatch.setattr("backend.services.canonical_candle_pipeline.load_aligned_candles", _empty)
    pipe = CanonicalCandlePipeline()
    pipe._redis = AsyncMock(return_value=None)
    monkeypatch.setattr("backend.services.canonical_candle_pipeline.canonical_candle_pipeline", pipe)
    out = await get_canonical_candles("BTCUSDT", "3m")
    assert out["success"] is False
    assert out["error"] == "no_canonical_data"
