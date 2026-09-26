"""DAY V2 executable-price freshness and zero-size decision persistence."""

from __future__ import annotations

import json
import sqlite3
import time
from types import SimpleNamespace

import pytest

import backend.services.day_v2.config as day_v2_config
from backend.services import canonical_mark_price
from backend.services.portfolio_engine_integration import (
    _DAY_V2_EXECUTABLE_PRICE_STALE_SEC,
    PortfolioEngineIntegration,
)

SYMBOLS = ("BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT")
ENTRY_BAR = 1790421300 + 900 + 5  # a few seconds after the 11:30Z bar closes


class FakeRedis:
    def __init__(self) -> None:
        self.hashes: dict[str, dict[str, str]] = {}

    def set_price(self, norm: str, price: float, ts: float, *, field: str = "timestamp") -> None:
        self.hashes[f"price:{norm}"] = {"v": str(price), field: str(ts)}

    async def hget(self, key: str, field: str):
        return self.hashes.get(key, {}).get(field)


def _integration(tmp_path, redis_client=None, *, open_positions=None):
    integ = PortfolioEngineIntegration.__new__(PortfolioEngineIntegration)
    integ.current_prices = {}
    integ.redis_client = redis_client
    integ.sizing_calls = []

    def _size(**kwargs):
        integ.sizing_calls.append(kwargs)
        return integ.sizing_result

    async def _can_open(_symbol, _notional):
        return True, ""

    integ.sizing_result = (0.0, 0.0, 0.0)
    integ.engine = SimpleNamespace(
        db_path=str(tmp_path / "day.db"),
        open_positions=open_positions or {},
        calculate_position_size=_size,
        _can_open_position=_can_open,
        _total_equity=228.0,
        cash_balance=30.45,
    )
    return integ


@pytest.fixture
def no_canonical(monkeypatch):
    calls = []

    async def _none(symbol, *, use_cache=True):
        calls.append(symbol)

    monkeypatch.setattr(canonical_mark_price, "fetch_canonical_mark", _none)
    monkeypatch.setattr("backend.services.portfolio_engine_integration.asyncio.sleep", _no_sleep)
    return calls


async def _no_sleep(_s):
    return None


# ---------------------------------------------------------------- resolver


@pytest.mark.asyncio
@pytest.mark.parametrize("symbol", SYMBOLS)
async def test_fresh_redis_price_resolves_with_real_age(tmp_path, no_canonical, symbol):
    redis = FakeRedis()
    redis.set_price(symbol, 100.0, time.time() - 2.0)
    integ = _integration(tmp_path, redis)

    px, age = await integ._resolve_day_executable_price(symbol)

    assert px == 100.0
    assert 1.5 <= age <= 5.0
    assert no_canonical == []


@pytest.mark.asyncio
@pytest.mark.parametrize("symbol", SYMBOLS)
async def test_startup_cached_price_is_not_reused_after_market_moves(tmp_path, no_canonical, symbol):
    redis = FakeRedis()
    redis.set_price(symbol, 122.255, time.time())
    integ = _integration(tmp_path, redis)
    first, _ = await integ._resolve_day_executable_price(symbol)
    assert first == 122.255

    redis.set_price(symbol, 121.17, time.time())
    second, age = await integ._resolve_day_executable_price(symbol)

    assert second == 121.17
    assert age <= _DAY_V2_EXECUTABLE_PRICE_STALE_SEC


@pytest.mark.asyncio
@pytest.mark.parametrize("symbol", SYMBOLS)
async def test_untimestamped_cache_is_never_reported_fresh(tmp_path, no_canonical, symbol):
    ccxt = f"{symbol[:-4]}/USDT"
    integ = _integration(tmp_path, FakeRedis())
    integ.current_prices.update({symbol: 122.255, ccxt: 122.255})

    px, age = await integ._resolve_day_executable_price(symbol)

    assert (px, age) == (0.0, None)


@pytest.mark.asyncio
@pytest.mark.parametrize("symbol", SYMBOLS)
async def test_price_older_than_limit_is_rejected(tmp_path, no_canonical, symbol):
    redis = FakeRedis()
    redis.set_price(symbol, 122.255, time.time() - (_DAY_V2_EXECUTABLE_PRICE_STALE_SEC + 60))
    integ = _integration(tmp_path, redis)

    px, age = await integ._resolve_day_executable_price(symbol)

    assert (px, age) == (0.0, None)
    assert len(no_canonical) == 2


@pytest.mark.asyncio
async def test_reading_does_not_reset_age(tmp_path, no_canonical):
    redis = FakeRedis()
    source_ts = time.time() - 20.0
    redis.set_price("SOLUSDT", 121.17, source_ts)
    integ = _integration(tmp_path, redis)

    _, age1 = await integ._resolve_day_executable_price("SOLUSDT")
    _, age2 = await integ._resolve_day_executable_price("SOLUSDT")

    assert age1 >= 19.5
    assert age2 >= age1


@pytest.mark.asyncio
async def test_redis_ts_field_is_accepted_as_timestamp(tmp_path, no_canonical):
    redis = FakeRedis()
    redis.set_price("XRPUSDT", 1.541, time.time() - 3.0, field="ts")
    integ = _integration(tmp_path, redis)

    px, age = await integ._resolve_day_executable_price("XRPUSDT")

    assert px == 1.541
    assert 2.5 <= age <= 6.0


@pytest.mark.asyncio
async def test_stale_redis_falls_back_to_timestamped_canonical_mark(tmp_path, monkeypatch):
    redis = FakeRedis()
    redis.set_price("ETHUSDT", 2695.07, time.time() - 3600)
    stamped = time.time() - 1.0

    async def _mark(symbol, *, use_cache=True):
        return SimpleNamespace(ask=2690.99, mark=2690.96, timestamp=stamped, source="binance_book_ticker_mid")

    monkeypatch.setattr(canonical_mark_price, "fetch_canonical_mark", _mark)
    integ = _integration(tmp_path, redis)

    px, age = await integ._resolve_day_executable_price("ETHUSDT")

    assert px == 2690.99
    assert 0.5 <= age <= 5.0


@pytest.mark.asyncio
async def test_open_position_price_refresh_unchanged(tmp_path, monkeypatch):
    from backend.services import live_market_data

    async def _ticker(api_symbol):
        return {"price": 84100.0}

    monkeypatch.setattr(live_market_data.live_market_data_service, "get_ticker", _ticker)
    integ = _integration(tmp_path, FakeRedis(), open_positions={"BTC/USDT": SimpleNamespace()})

    await integ._refresh_prices()

    assert integ.current_prices == {"BTC/USDT": 84100.0}


# ---------------------------------------------------------------- DAY V2 cycle


def _signal(symbol: str) -> SimpleNamespace:
    return SimpleNamespace(
        symbol=symbol,
        setup="HTF_TREND_PULLBACK",
        regime="bull",
        structural_anchor=100.0,
        target_price=130.0,
        atr=0.3,
        signal_bar_ts=1790421300,
        h1_bullish=True,
        opportunity_id=f"opp_{symbol.lower()}",
    )


@pytest.fixture
def day_cycle(monkeypatch):
    state = {"signal": True, "intents": [], "claims": []}
    required_open = float(((ENTRY_BAR // 900) * 900) - 900)
    bars = [{"ts_epoch": required_open - 900 * i} for i in range(59, -1, -1)]

    monkeypatch.setattr(day_v2_config, "DAY_V2_ENABLED", True)
    monkeypatch.setattr(day_v2_config, "DAY_V2_UNIVERSE", SYMBOLS)
    monkeypatch.setattr("backend.services.candle_contract.load_closed_bars", lambda *_a, **_k: bars)
    monkeypatch.setattr(
        "backend.services.day_v2.live_signal.evaluate_entry_signal",
        lambda symbol, *_a: _signal(symbol) if state["signal"] else None,
    )
    monkeypatch.setattr(
        "backend.services.day_v2.live_signal.explain_no_signal",
        lambda *_a: {"closest": "RANGE_BOUNCE", "unmet": ["not_green"]},
    )
    monkeypatch.setattr("backend.services.day_v2.migrations.is_opportunity_consumed", lambda *_a: False)
    monkeypatch.setattr("backend.services.day_v2.frequency_guard.check_frequency_limit", lambda *_a: (True, ""))
    monkeypatch.setattr(
        "backend.services.day_v2.structural_entry.evaluate_structural_zone",
        lambda _s: SimpleNamespace(valid=True, reason="", reclaim_level=101.0, zone_low=99.0, zone_high=101.0),
    )

    def _claim(db_path, norm, engine, opp, notional, positions=None):
        state["claims"].append(norm)
        return True, "", f"res_{norm}"

    def _intent(db_path, signal, ask_price, qty, **kwargs):
        state["intents"].append({"symbol": signal.symbol, "ask": ask_price, "qty": qty})
        return {"intent_id": f"intent_{signal.symbol}"}

    monkeypatch.setattr("backend.services.two_engine_claim.claim_symbol", _claim)
    monkeypatch.setattr("backend.services.two_engine_claim.release_claim", lambda *_a, **_k: None)
    monkeypatch.setattr("backend.services.day_v2.live_entry.create_day_v2_intent", _intent)

    async def _no_mark(symbol, *, use_cache=True):
        return None

    monkeypatch.setattr(canonical_mark_price, "fetch_canonical_mark", _no_mark)
    monkeypatch.setattr("backend.services.portfolio_engine_integration.asyncio.sleep", _no_sleep)
    return state


def _fresh_redis(prices: dict[str, float]) -> FakeRedis:
    redis = FakeRedis()
    for sym, px in prices.items():
        redis.set_price(sym, px, time.time())
    return redis


def _decisions(db_path: str) -> list[tuple]:
    with sqlite3.connect(db_path) as conn:
        return conn.execute("SELECT symbol, cycle_ts, result, closest, unmet_json FROM day_v2_decisions ORDER BY id").fetchall()


LIVE = {"BTCUSDT": 84103.9, "ETHUSDT": 2690.99, "SOLUSDT": 121.17, "XRPUSDT": 1.5418}
STARTUP = {"BTCUSDT": 84120.27, "ETHUSDT": 2695.07, "SOLUSDT": 122.255, "XRPUSDT": 1.5741}


@pytest.mark.asyncio
async def test_zero_size_candidate_records_decision_without_order(tmp_path, day_cycle):
    integ = _integration(tmp_path, _fresh_redis(LIVE))

    await integ._process_day_v2_signals(ENTRY_BAR)

    rows = _decisions(integ.engine.db_path)
    assert [r[0] for r in rows] == list(SYMBOLS)
    for symbol, cycle_ts, result, closest, unmet_json in rows:
        unmet = json.loads(unmet_json)
        assert cycle_ts == float(ENTRY_BAR)
        assert result == "REJECTED:INSUFFICIENT_EXECUTABLE_CASH"
        assert closest == "HTF_TREND_PULLBACK"
        assert unmet[0] == "ZERO_SIZE"
        assert f"ask={LIVE[symbol]:.8f}" in unmet
        assert f"opportunity_id=opp_{symbol.lower()}" in unmet
    assert day_cycle["intents"] == []
    assert day_cycle["claims"] == []


@pytest.mark.asyncio
async def test_zero_size_decision_is_not_duplicated_for_same_bar(tmp_path, day_cycle):
    integ = _integration(tmp_path, _fresh_redis(LIVE))

    await integ._process_day_v2_signals(ENTRY_BAR)
    await integ._process_day_v2_signals(ENTRY_BAR)

    assert len(_decisions(integ.engine.db_path)) == len(SYMBOLS)


@pytest.mark.asyncio
async def test_no_signal_row_unchanged(tmp_path, day_cycle):
    day_cycle["signal"] = False
    integ = _integration(tmp_path, _fresh_redis(LIVE))

    await integ._process_day_v2_signals(ENTRY_BAR)

    rows = _decisions(integ.engine.db_path)
    assert [(r[2], r[3], json.loads(r[4])) for r in rows] == [("NO_SIGNAL", "RANGE_BOUNCE", ["not_green"])] * len(SYMBOLS)
    assert integ.sizing_calls == []


@pytest.mark.asyncio
async def test_nonzero_size_arms_intent_with_current_price(tmp_path, day_cycle):
    integ = _integration(tmp_path, _fresh_redis(LIVE))
    integ.current_prices.update(STARTUP)
    integ.sizing_result = (0.5, 0.0, 1.0)

    await integ._process_day_v2_signals(ENTRY_BAR)

    assert [r[2] for r in _decisions(integ.engine.db_path)] == ["ARMED"] * len(SYMBOLS)
    assert [c["current_price"] for c in integ.sizing_calls] == [LIVE[s] for s in SYMBOLS]
    assert day_cycle["intents"] == [{"symbol": s, "ask": LIVE[s], "qty": 0.5} for s in SYMBOLS]


@pytest.mark.asyncio
async def test_stale_price_rejects_before_sizing(tmp_path, day_cycle):
    redis = FakeRedis()
    for sym, px in STARTUP.items():
        redis.set_price(sym, px, time.time() - 3600)
    integ = _integration(tmp_path, redis)
    integ.current_prices.update(STARTUP)

    await integ._process_day_v2_signals(ENTRY_BAR)

    assert [r[2] for r in _decisions(integ.engine.db_path)] == ["MISSING_EXECUTABLE_PRICE"] * len(SYMBOLS)
    assert integ.sizing_calls == []
    assert day_cycle["intents"] == []
