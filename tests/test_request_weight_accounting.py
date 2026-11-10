"""Shared Binance.US request-weight accounting.

Measured costs are fixed here so a later edit cannot silently charge the
40-weight order-counter read as 1, or treat one process's local total as the
IP total.
"""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from backend.config.protected_execution import MANDATORY_EXIT_SAME_CALL_ATTEMPTS
from backend.services.portfolio_engine import COMBINED_ENGINE_MAX_POSITIONS
from backend.services.venue_order_budget import VenueOrderBudget
from backend.utils.binance_weight_limiter import (
    ACCOUNT_WEIGHT,
    DEPTH_WEIGHT,
    ENDPOINT_WEIGHTS,
    EXCHANGE_INFO_WEIGHT,
    MY_TRADES_WEIGHT,
    OPEN_ORDERS_ALL_SYMBOLS_WEIGHT,
    ORDER_CANCEL_WEIGHT,
    ORDER_STATUS_WEIGHT,
    ORDER_SUBMIT_WEIGHT,
    ORDER_USAGE_WEIGHT,
    PROTECTIVE_WEIGHT_RESERVE,
    RateLimited,
    alignment_state,
    apply_venue_header,
    carried_remaining,
    depth_request_weight,
    header_used_weight,
    kline_request_weight,
    remaining_after_venue,
    weight_grant_allowed,
    window_id,
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


def test_local_roll_keeps_a_fresh_venue_sample() -> None:
    now = 1_794_337_020.0
    assert window_id(now) == 1_794_337_020 // 60
    assert window_id(now - 0.1) == window_id(now) - 1
    # Five seconds and thirty seconds before the venue minute: still the same fresh sample.
    assert carried_remaining(1100, 363, now - 5, now) == 737
    assert carried_remaining(1100, 1050, now - 30, now) == 50
    assert carried_remaining(1100, 1050, now - 60, now) == 1100
    assert carried_remaining(1100, None, None, now) == 1100


def test_stale_lower_header_does_not_reopen_capacity() -> None:
    remaining, high_water, _when, hw_window, decision = apply_venue_header(
        remaining=50,
        budget=1100,
        high_water=1050,
        high_water_at=100.0,
        high_water_window=10,
        used=10,
        request_started_at=90.0,
        now_window=11,
    )
    assert decision == "stale_ignored"
    assert remaining == 50
    assert high_water == 1050
    assert hw_window == 10
    missing, *_rest = apply_venue_header(
        remaining=50,
        budget=1100,
        high_water=1050,
        high_water_at=100.0,
        high_water_window=10,
        used=10,
        request_started_at=None,
        now_window=11,
    )
    assert missing == 50


def test_proven_minute_reset_reopens_only_the_new_minute() -> None:
    remaining, high_water, _when, hw_window, decision = apply_venue_header(
        remaining=30,
        budget=1100,
        high_water=1050,
        high_water_at=100.0,
        high_water_window=10,
        used=10,
        request_started_at=110.0,
        now_window=11,
    )
    assert decision == "reset_confirmed"
    assert high_water == 10
    assert hw_window == 11
    assert remaining == 1070
    assert remaining <= 1100 - 10


def test_higher_header_never_increases_remaining() -> None:
    remaining, high_water, _when, hw_window, decision = apply_venue_header(
        remaining=800,
        budget=1100,
        high_water=100,
        high_water_at=1.0,
        high_water_window=4,
        used=400,
        request_started_at=2.0,
        now_window=4,
    )
    assert decision == "raised"
    assert remaining == 700
    assert high_water == 400
    assert hw_window == 4


def test_callers_share_one_carried_budget() -> None:
    remaining = carried_remaining(1100, 1050, 1.0, 2.0)
    for _ in range(4):
        assert remaining >= 5
        remaining -= 5
    assert remaining == 30
    assert alignment_state(11, 10, 1050) == "VENUE_AHEAD"
    assert alignment_state(11, 11, 10) == "ALIGNED"
    assert alignment_state(11, None, None) == "UNKNOWN"
    assert alignment_state(11, 12, 10) == "LOCAL_AHEAD"


def test_protective_reserve_covers_one_full_cycle_plus_one_recovery() -> None:
    clean = DEPTH_WEIGHT + ORDER_SUBMIT_WEIGHT
    recovery = ORDER_STATUS_WEIGHT + MY_TRADES_WEIGHT + ORDER_CANCEL_WEIGHT
    assert COMBINED_ENGINE_MAX_POSITIONS * MANDATORY_EXIT_SAME_CALL_ATTEMPTS * clean + recovery == PROTECTIVE_WEIGHT_RESERVE
    assert PROTECTIVE_WEIGHT_RESERVE < 1100


def test_background_cannot_take_the_exit_reserve() -> None:
    reserve = PROTECTIVE_WEIGHT_RESERVE
    for used in (1070, 1080, 1090, 1095, 1099):
        remaining = 1100 - used
        assert weight_grant_allowed(remaining, 5, priority="exit", reserve=reserve, pending=False) is (remaining >= 5)
        assert weight_grant_allowed(remaining, 2, priority="background", reserve=reserve, pending=False) is False
        assert weight_grant_allowed(remaining, 1, priority="entry", reserve=reserve, pending=False) is False
        assert weight_grant_allowed(remaining, 10, priority="background", reserve=reserve, pending=False) is False
        assert weight_grant_allowed(remaining, 20, priority="background", reserve=reserve, pending=False) is False
        assert weight_grant_allowed(remaining, 40, priority="background", reserve=reserve, pending=False) is False


def test_pending_sell_refuses_entry_and_background() -> None:
    assert weight_grant_allowed(500, 1, priority="entry", reserve=PROTECTIVE_WEIGHT_RESERVE, pending=True) is False
    assert weight_grant_allowed(500, 2, priority="background", reserve=PROTECTIVE_WEIGHT_RESERVE, pending=True) is False
    assert weight_grant_allowed(500, 5, priority="exit", reserve=PROTECTIVE_WEIGHT_RESERVE, pending=True) is True
    assert weight_grant_allowed(4, 5, priority="exit", reserve=PROTECTIVE_WEIGHT_RESERVE, pending=True) is False


def test_eight_clean_attempts_fit_in_the_reserve() -> None:
    remaining = PROTECTIVE_WEIGHT_RESERVE
    clean = DEPTH_WEIGHT + ORDER_SUBMIT_WEIGHT
    for _position in range(COMBINED_ENGINE_MAX_POSITIONS):
        for _attempt in range(MANDATORY_EXIT_SAME_CALL_ATTEMPTS):
            assert weight_grant_allowed(remaining, DEPTH_WEIGHT, priority="exit", reserve=PROTECTIVE_WEIGHT_RESERVE, pending=True)
            remaining -= DEPTH_WEIGHT
            assert weight_grant_allowed(remaining, ORDER_SUBMIT_WEIGHT, priority="exit", reserve=PROTECTIVE_WEIGHT_RESERVE, pending=True)
            remaining -= ORDER_SUBMIT_WEIGHT
            assert clean == DEPTH_WEIGHT + ORDER_SUBMIT_WEIGHT
    assert remaining == ORDER_STATUS_WEIGHT + MY_TRADES_WEIGHT + ORDER_CANCEL_WEIGHT


def test_exit_decision_mark_can_use_the_reserve_and_a_feature_read_cannot() -> None:
    remaining = 1100 - 800
    assert weight_grant_allowed(remaining, 2, priority="exit_decision", reserve=PROTECTIVE_WEIGHT_RESERVE, pending=False) is True
    assert weight_grant_allowed(remaining, 2, priority="background", reserve=PROTECTIVE_WEIGHT_RESERVE, pending=False) is False
    assert weight_grant_allowed(remaining, 2, priority="exit_decision", reserve=PROTECTIVE_WEIGHT_RESERVE, pending=True) is True
    assert weight_grant_allowed(remaining, 5, priority="exit", reserve=PROTECTIVE_WEIGHT_RESERVE, pending=False) is True
    assert weight_grant_allowed(4, 5, priority="exit_decision", reserve=PROTECTIVE_WEIGHT_RESERVE, pending=False) is False


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
