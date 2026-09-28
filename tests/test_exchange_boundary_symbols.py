"""Exchange-boundary symbol contract for engine-scoped position keys.

Internal identity is '<engine_id>::<symbol>' (SCALP_V2::BTC/USDT, DAY_V2::BTC/USDT).
Every Binance.US request (klines, ticker/24hr, depth, orders, fetch_order,
myTrades, cancel) must carry only the market symbol; the engine prefix must
never reach the venue. Internal keys stay composite and distinct.
"""

from __future__ import annotations

import time
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

import backend.services.live_market_data as lmd
from backend.services import binance_rest_client as brc
from backend.services import execution_adapter as ea
from backend.services.live_trading_service import LiveTradingService, _to_binance_pair
from backend.services.portfolio_engine import (
    OpenPosition,
    PortfolioEngine,
    _to_api_symbol,
    make_position_key,
    split_position_key,
)
from backend.utils.canonical_symbol_formatter import CanonicalSymbolFormatter
from backend.utils.position_keys import split_engine_key, venue_symbol
from backend.utils.symbols import to_exchange_symbol

COINS = [("BTC", "BTC/USDT", "BTCUSDT"), ("ETH", "ETH/USDT", "ETHUSDT"), ("SOL", "SOL/USDT", "SOLUSDT"), ("XRP", "XRP/USDT", "XRPUSDT")]
ENGINES = ("SCALP_V2", "DAY_V2")


def _inputs(ccxt: str, api: str) -> list[str]:
    return [ccxt, api, *(f"{e}::{ccxt}" for e in ENGINES)]


def _assert_clean(value: object) -> None:
    text = str(value)
    for bad in ("SCALP_V2", "DAY_V2", "SCALPV2", "DAYV2", "::"):
        assert bad not in text, f"engine identity leaked: {text!r}"


# ---------------------------------------------------------------------------
# A-E. Formatter contract (bare and composite, all top-4)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(("base", "ccxt", "api"), COINS)
def test_formatters_strip_engine_prefix_for_every_coin(base, ccxt, api):
    for raw in _inputs(ccxt, api):
        assert CanonicalSymbolFormatter.to_exchange(raw) == api
        assert CanonicalSymbolFormatter.to_ccxt(raw) == ccxt
        assert CanonicalSymbolFormatter.to_base(raw) == base
        assert to_exchange_symbol(raw) == api
        assert lmd._to_ccxt_symbol(raw) == ccxt
        assert lmd._to_binance_pair(raw) == api
        assert _to_binance_pair(raw) == api
        assert ea._to_api_symbol(raw) == api
        assert ea._to_ccxt_symbol(raw) == ccxt
        assert _to_api_symbol(raw) == api
        assert brc._normalize_symbol(raw) == api


def test_venue_symbol_parser_matches_position_key_parser():
    assert split_engine_key("SCALP_V2::BTC/USDT") == ("SCALP_V2", "BTC/USDT")
    assert split_engine_key("BTC/USDT") == ("", "BTC/USDT")
    assert venue_symbol("DAY_V2::XRP/USDT") == "XRP/USDT"
    assert venue_symbol("XRPUSDT") == "XRPUSDT"
    for engine in ENGINES:
        key = make_position_key(engine, "BTCUSDT")
        assert split_position_key(key) == (engine, "BTC/USDT")
        assert split_engine_key(key) == split_position_key(key)


def test_day_bundle_cache_key_is_market_scoped():
    from backend.services.day_active_market_bundle import _bundle_cache_key, _normalize_ccxt_symbol

    for engine in ENGINES:
        assert _normalize_ccxt_symbol(f"{engine}::BTC/USDT") == "BTC/USDT"
        assert _bundle_cache_key(f"{engine}::BTC/USDT") == _bundle_cache_key("BTCUSDT")


# ---------------------------------------------------------------------------
# F-H. Market-data requests never carry the prefix
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize("timeframe", ["1m", "3m", "5m", "15m", "30m"])
async def test_klines_request_uses_bare_symbol(monkeypatch, timeframe):
    import backend.services.canonical_candle_store as store

    monkeypatch.setattr(store, "load_aligned_candles", lambda *_a, **_k: None)
    sent: list[dict] = []

    async def _get_json(url, params=None):
        sent.append({"url": url, **(params or {})})
        return [[1, "1", "1", "1", "1", "1"]]

    monkeypatch.setattr(lmd, "canonical_http_client", SimpleNamespace(get_json=_get_json))
    svc = lmd.live_market_data_service
    monkeypatch.setattr(svc, "_get_limiter", AsyncMock(return_value=SimpleNamespace(consume=AsyncMock())))
    for engine in ENGINES:
        meta = await svc.get_ohlcv_with_meta(f"{engine}::BTC/USDT", timeframe, limit=3)
        assert meta["symbol"] == "BTCUSDT"
    assert [p["symbol"] for p in sent] == ["BTCUSDT", "BTCUSDT"]
    for p in sent:
        _assert_clean(p)


@pytest.mark.asyncio
async def test_ticker_24hr_request_uses_bare_symbol(monkeypatch):
    sent: list[dict] = []

    class _Resp:
        def raise_for_status(self):
            return None

        def json(self):
            return {"lastPrice": "84000", "bidPrice": "83999", "askPrice": "84001", "volume": "1", "priceChangePercent": "0", "highPrice": "1", "lowPrice": "1"}

    class _Client:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc):
            return False

        async def get(self, url, params=None, timeout=None):
            sent.append({"url": url, **(params or {})})
            return _Resp()

    monkeypatch.setattr(lmd.httpx, "AsyncClient", _Client)
    svc = lmd.live_market_data_service
    monkeypatch.setattr(svc, "_get_limiter", AsyncMock(return_value=SimpleNamespace(consume=AsyncMock())))
    for engine in ENGINES:
        await svc.get_ticker(f"{engine}::BTC/USDT", force_refresh=True)
    assert [p["symbol"] for p in sent] == ["BTCUSDT", "BTCUSDT"]
    for p in sent:
        _assert_clean(p)


@pytest.mark.asyncio
async def test_order_book_request_uses_bare_symbol(monkeypatch):
    sent: list[str] = []

    class _Public:
        def __init__(self, *_a, **_k):
            pass

        def fetch_order_book(self, symbol, limit):
            sent.append(symbol)
            return {"bids": [], "asks": []}

    monkeypatch.setattr(lmd.ccxt, "binanceus", _Public)
    for engine in ENGINES:
        ob = await lmd.live_market_data_service.get_order_book(f"{engine}::ETH/USDT", limit=5)
        assert ob["fetch_failed"] is False
    assert sent == ["ETH/USDT", "ETH/USDT"]


@pytest.mark.asyncio
async def test_binance_rest_client_endpoints_use_bare_symbol():
    fake = SimpleNamespace(_request=AsyncMock(return_value={}))
    for engine in ENGINES:
        key = f"{engine}::SOL/USDT"
        await brc.BinanceREST.ticker_24h(fake, key)
        await brc.BinanceREST.klines(fake, key, "5m", 10)
        await brc.BinanceREST.depth(fake, key, 10)
        await brc.BinanceREST.order_market(fake, key, "BUY", quantity=1)
    symbols = [c.kwargs["params"]["symbol"] for c in fake._request.await_args_list]
    assert symbols == ["SOLUSDT"] * 8


# ---------------------------------------------------------------------------
# I-K. Order execution paths (BUY, SELL, fetch_order, myTrades, cancel)
# ---------------------------------------------------------------------------


class _FakeCcxt:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str]] = []

    def create_order(self, symbol, type, side, amount, price=None, params=None):
        self.calls.append((f"create_{side}", symbol))
        return {"id": "1", "status": "closed", "filled": amount, "amount": amount, "average": price or 1.0, "symbol": symbol}

    def fetch_order(self, id=None, symbol=None, params=None):
        self.calls.append(("fetch_order", symbol))
        return {"id": id or "1", "status": "closed", "symbol": symbol}

    def fetch_my_trades(self, symbol=None, params=None):
        self.calls.append(("fetch_my_trades", symbol))
        return []

    def cancel_order(self, id=None, symbol=None):
        self.calls.append(("cancel_order", symbol))
        return {}


@pytest.mark.asyncio
@pytest.mark.parametrize(("_base", "ccxt", "api"), COINS)
async def test_order_paths_never_send_engine_prefix(_base, ccxt, api):
    fake = _FakeCcxt()
    svc = SimpleNamespace(binance=fake, _ensure_initialized=AsyncMock())
    for engine in ENGINES:
        key = f"{engine}::{ccxt}"
        await LiveTradingService.place_order(svc, "binanceus", key, "limit", "buy", 1.0, price=1.0)
        await LiveTradingService.place_order(svc, "binanceus", key, "market", "sell", 1.0)
        await LiveTradingService.fetch_order(svc, "binanceus", "123", key)
        await LiveTradingService.fetch_order(svc, "binanceus", "cid-abc", key)
        await LiveTradingService.fetch_order_trades(svc, "binanceus", key, "123")
        await LiveTradingService.cancel_order(svc, "binanceus", "123", key)
    assert {name for name, _ in fake.calls} == {"create_buy", "create_sell", "fetch_order", "fetch_my_trades", "cancel_order"}
    assert all(sym == api for _, sym in fake.calls), fake.calls


# ---------------------------------------------------------------------------
# L-N. Internal keys stay composite; reconcile sums both engine lots
# ---------------------------------------------------------------------------


def _lot(symbol: str, engine_id: str, qty: float, price: float) -> OpenPosition:
    return OpenPosition(
        symbol=symbol,
        quantity=qty,
        entry_price=price,
        entry_time=time.time(),
        trade_id=f"t-{engine_id}",
        stop_price=price * 0.99,
        take_profit_1_price=price * 1.01,
        take_profit_2_price=price * 1.02,
        status="ACTIVE",
        engine_id=engine_id,
        original_position_cost=qty * price,
    )


def test_internal_keys_remain_composite_and_distinct():
    scalp = make_position_key("SCALP_V2", "BTC/USDT")
    day = make_position_key("DAY_V2", "BTC/USDT")
    assert scalp != day
    assert scalp == "SCALP_V2::BTC/USDT"
    assert day == "DAY_V2::BTC/USDT"
    assert to_exchange_symbol(scalp) == to_exchange_symbol(day) == "BTCUSDT"


@pytest.mark.asyncio
async def test_dual_engine_reconcile_uses_bare_balance_and_keeps_both_lots(tmp_path):
    eng = PortfolioEngine(db_path=str(tmp_path / "boundary.db"), principal=1000.0, test_mode=True)
    eng._ensure_db_schema()
    scalp = _lot("BTC/USDT", "SCALP_V2", 0.00045991, 84640.47)
    day = _lot("BTC/USDT", "DAY_V2", 0.00030000, 84000.00)
    scalp.highest_price = 85500.0
    eng.open_positions[make_position_key("SCALP_V2", "BTC/USDT")] = scalp
    eng.open_positions[make_position_key("DAY_V2", "BTC/USDT")] = day

    api = _to_api_symbol(next(iter(eng.open_positions)))
    base = api[:-4]
    balances = {"BTC": 0.00075991}
    await eng._reconcile_dual_engine_lots(
        symbol="BTC/USDT",
        lots=eng._symbol_lots("BTC/USDT"),
        exchange_qty=balances[base],
        qty_step=0.00001,
        source="test",
    )

    assert base == "BTC"
    assert set(eng.open_positions) == {"SCALP_V2::BTC/USDT", "DAY_V2::BTC/USDT"}
    assert eng.open_positions["SCALP_V2::BTC/USDT"].quantity == pytest.approx(0.00045991)
    assert eng.open_positions["DAY_V2::BTC/USDT"].quantity == pytest.approx(0.00030000)
    assert eng.open_positions["SCALP_V2::BTC/USDT"].highest_price == 85500.0
