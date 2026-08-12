"""Item p25: exchange-call fail-safe behavior — no ambiguous false-neutral returns."""

from __future__ import annotations

import sys
from pathlib import Path
from unittest import mock

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from backend.services import live_market_data as lmd


@pytest.fixture
def service(monkeypatch):
    monkeypatch.setattr(
        "backend.utils.binance_credentials.get_binance_us_credentials",
        lambda: ("key", "secret"),
    )
    return lmd.LiveMarketDataService()


@pytest.mark.asyncio
async def test_get_order_book_marks_fetch_failed_on_exception(service, monkeypatch):
    class _FakePublicClient:
        def fetch_order_book(self, *_a, **_k):
            raise RuntimeError("boom")

    monkeypatch.setattr(lmd.ccxt, "binanceus", lambda *_a, **_k: _FakePublicClient())
    result = await service.get_order_book("BTC/USDT", limit=10)
    assert result["fetch_failed"] is True
    assert result["bids"] == []
    assert result["asks"] == []
    assert "error" in result


@pytest.mark.asyncio
async def test_get_order_book_success_marks_fetch_failed_false(service, monkeypatch):
    class _FakePublicClient:
        def fetch_order_book(self, *_a, **_k):
            return {"bids": [[100.0, 1.0]], "asks": [[101.0, 1.0]]}

    monkeypatch.setattr(lmd.ccxt, "binanceus", lambda *_a, **_k: _FakePublicClient())
    result = await service.get_order_book("BTC/USDT", limit=10)
    assert result["fetch_failed"] is False
    assert result["bids"] == [[100.0, 1.0]]


@pytest.mark.asyncio
async def test_get_ticker_rejects_zero_price_response_with_no_cache(service, monkeypatch):
    class _FakeLimiter:
        async def consume(self, *_a, **_k):
            return None

    monkeypatch.setattr(service, "_get_limiter", mock.AsyncMock(return_value=_FakeLimiter()))

    class _FakeResponse:
        def raise_for_status(self):
            return None

        def json(self):
            return {"lastPrice": "0", "bidPrice": "0", "askPrice": "0"}

    class _FakeAsyncClient:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc):
            return False

        async def get(self, *_a, **_k):
            return _FakeResponse()

    monkeypatch.setattr(lmd.httpx, "AsyncClient", lambda *_a, **_k: _FakeAsyncClient())
    result = await service.get_ticker("BTC/USDT")
    assert result is None


@pytest.mark.asyncio
async def test_get_ticker_accepts_valid_nonzero_price(service, monkeypatch):
    class _FakeLimiter:
        async def consume(self, *_a, **_k):
            return None

    monkeypatch.setattr(service, "_get_limiter", mock.AsyncMock(return_value=_FakeLimiter()))

    class _FakeResponse:
        def raise_for_status(self):
            return None

        def json(self):
            return {
                "lastPrice": "50000.0",
                "bidPrice": "49999.0",
                "askPrice": "50001.0",
                "highPrice": "51000.0",
                "lowPrice": "49000.0",
                "volume": "10.0",
                "quoteVolume": "500000.0",
                "priceChange": "100.0",
                "priceChangePercent": "0.2",
                "closeTime": 1700000000000,
            }

    class _FakeAsyncClient:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc):
            return False

        async def get(self, *_a, **_k):
            return _FakeResponse()

    monkeypatch.setattr(lmd.httpx, "AsyncClient", lambda *_a, **_k: _FakeAsyncClient())
    result = await service.get_ticker("BTC/USDT")
    assert result is not None
    assert result["price"] == 50000.0
