"""Shared Binance.US request-weight accounting.

Measured costs are fixed here so a later edit cannot silently charge the
40-weight order-counter read as 1, or treat one process's local total as the
IP total.
"""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from backend.services.venue_order_budget import VenueOrderBudget
from backend.utils.binance_weight_limiter import (
    ACCOUNT_WEIGHT,
    ENDPOINT_WEIGHTS,
    EXCHANGE_INFO_WEIGHT,
    OPEN_ORDERS_ALL_SYMBOLS_WEIGHT,
    ORDER_STATUS_WEIGHT,
    ORDER_SUBMIT_WEIGHT,
    ORDER_USAGE_WEIGHT,
    RateLimited,
    depth_request_weight,
    header_used_weight,
    kline_request_weight,
    remaining_after_venue,
)


def test_measured_weights_match_binanceus_headers() -> None:
    assert ENDPOINT_WEIGHTS["/api/v3/rateLimit/order"] == 40
    assert ORDER_USAGE_WEIGHT == 40
    assert ENDPOINT_WEIGHTS["/api/v3/account"] == ACCOUNT_WEIGHT == 20
    assert ENDPOINT_WEIGHTS["/api/v3/exchangeInfo"] == EXCHANGE_INFO_WEIGHT == 20
    assert ENDPOINT_WEIGHTS["/api/v3/ticker/24hr"] == 2
    assert ENDPOINT_WEIGHTS["/api/v3/ticker/bookTicker"] == 2
    assert ENDPOINT_WEIGHTS["/api/v3/klines"] == 2
    assert ENDPOINT_WEIGHTS["/api/v3/depth"] == 5
    assert ENDPOINT_WEIGHTS["/api/v3/order"] == ORDER_STATUS_WEIGHT == 4
    assert ENDPOINT_WEIGHTS["/api/v3/myTrades"] == 20
    assert ENDPOINT_WEIGHTS["/api/v3/openOrders"] == 8
    assert OPEN_ORDERS_ALL_SYMBOLS_WEIGHT == 85
    assert ORDER_SUBMIT_WEIGHT == 1
    assert kline_request_weight(1) == 2
    assert kline_request_weight(300) == 2
    assert kline_request_weight(500) == 2
    assert kline_request_weight(1000) == 10
    assert depth_request_weight(5) == 5
    assert depth_request_weight(100) == 5


def test_venue_header_raises_usage_and_never_lowers_it() -> None:
    assert remaining_after_venue(1000, 1200, 1100) == 0
    assert remaining_after_venue(800, 400, 1100) == 700
    assert remaining_after_venue(100, 50, 1100) == 100
    assert remaining_after_venue(0, 10, 1100) == 0


def test_weight_header_parser() -> None:
    assert header_used_weight({"x-mbx-used-weight-1m": "64"}) == 64
    assert header_used_weight({"X-MBX-USED-WEIGHT-1M": "8"}) == 8
    assert header_used_weight({}) is None
    assert header_used_weight(None) is None


@pytest.mark.asyncio
async def test_order_counter_refresh_is_charged_at_40_before_the_request(monkeypatch) -> None:
    seen: list[tuple[str, int, bool]] = []

    class _Limiter:
        async def consume(self, path: str, weight: int | None = None, wait: bool = True, timeout: float = 5.0) -> None:
            seen.append((path, int(weight or 0), wait))

    monkeypatch.setattr(
        "backend.utils.binance_weight_limiter.BinanceWeightLimiter.create",
        AsyncMock(return_value=_Limiter()),
    )
    called = {"n": 0}

    def _fetch() -> list[dict[str, object]]:
        called["n"] += 1
        return []

    monkeypatch.setattr("backend.services.venue_order_budget._fetch_order_usage", _fetch)
    budget = VenueOrderBudget()
    assert await budget.refresh_order_usage() is False
    assert seen == [("/api/v3/rateLimit/order", 40, False)]
    assert called["n"] == 1


@pytest.mark.asyncio
async def test_order_counter_refresh_does_not_call_venue_without_weight(monkeypatch) -> None:
    class _Limiter:
        async def consume(self, path: str, weight: int | None = None, wait: bool = True, timeout: float = 5.0) -> None:
            raise RateLimited("empty")

    monkeypatch.setattr(
        "backend.utils.binance_weight_limiter.BinanceWeightLimiter.create",
        AsyncMock(return_value=_Limiter()),
    )

    def _fetch() -> list[dict[str, object]]:
        raise AssertionError("venue read must not run when the weight charge is refused")

    monkeypatch.setattr("backend.services.venue_order_budget._fetch_order_usage", _fetch)
    budget = VenueOrderBudget()
    assert await budget.refresh_order_usage() is False


@pytest.mark.asyncio
async def test_order_book_is_not_fetched_when_weight_is_refused(monkeypatch) -> None:
    from backend.services import live_market_data as lmd

    called: list[str] = []

    class _Client:
        def fetch_order_book(self, *_a, **_k):
            called.append("fetched")
            return {"bids": [[1.0, 1.0]], "asks": [[2.0, 1.0]]}

    monkeypatch.setattr(lmd.ccxt, "binanceus", lambda *_a, **_k: _Client())
    monkeypatch.setattr(
        "backend.utils.binance_credentials.get_binance_us_credentials",
        lambda: ("key", "secret"),
    )
    monkeypatch.setattr("backend.utils.binance_weight_limiter.try_consume_sync", lambda *_a, **_k: False)
    service = lmd.LiveMarketDataService()
    result = await service.get_order_book("BTC/USDT", limit=10)
    assert result["fetch_failed"] is True
    assert called == []
