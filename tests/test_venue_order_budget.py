"""New-order count is a separate venue limit from request weight."""

from __future__ import annotations

import pytest

from backend.services.day_mandatory_exit_execution import impact_for_attempt, is_mandatory_day_flatten, run_mandatory_exit_ioc_loop
from backend.services.protected_limit_execution import _LIVE_ORDER_OUTCOME, LIVE_OUTCOME_RATE_LIMITED
from backend.services.venue_order_budget import VenueOrderBudget, is_venue_rate_limit, parse_order_limits

VENUE = [
    {"rateLimitType": "REQUEST_WEIGHT", "interval": "MINUTE", "intervalNum": 1, "limit": 6000},
    {"rateLimitType": "ORDERS", "interval": "SECOND", "intervalNum": 10, "limit": 100},
    {"rateLimitType": "ORDERS", "interval": "DAY", "intervalNum": 1, "limit": 200000},
]


def test_parser_keeps_order_windows_and_drops_weight():
    assert parse_order_limits(VENUE) == [(10.0, 100), (86400.0, 200000)]


def test_entries_leave_the_exit_reserve_and_exits_can_use_it():
    clock = {"t": 0.0}
    budget = VenueOrderBudget(clock=lambda: clock["t"])
    budget.set_limits(VENUE)
    reserve = 8
    for _ in range(100 - reserve):
        assert budget.try_acquire("entry") is True
    assert budget.try_acquire("entry") is False
    assert budget.try_acquire("exit") is True
    for _ in range(reserve - 1):
        assert budget.try_acquire("exit") is True
    assert budget.try_acquire("exit") is False
    clock["t"] = 10.0
    assert budget.try_acquire("exit") is True
    assert budget.try_acquire("entry") is True


def test_rate_limit_text_is_recognized():
    assert is_venue_rate_limit("HTTP 429 Too many new orders -1015")
    assert is_venue_rate_limit("418 banned")
    assert is_venue_rate_limit("Way too much request weight -1003")
    assert is_venue_rate_limit("Filter failure LOT_SIZE") is False


def test_learned_zero_fill_is_not_a_sticky_flatten_and_catastrophic_is():
    assert (
        is_mandatory_day_flatten(
            "DAY_V2_LEARNED_CONTINUATION",
            force_sell=True,
            exit_type_name="STRATEGY",
        )
        is False
    )
    assert (
        is_mandatory_day_flatten(
            "SCALP_V2_LEARNED_CONTINUATION",
            force_sell=True,
            exit_type_name="STRATEGY",
        )
        is False
    )
    assert (
        is_mandatory_day_flatten(
            "DAY_V2_CATASTROPHIC_PROTECTION",
            force_sell=True,
            exit_type_name="STRATEGY",
        )
        is True
    )
    assert (
        is_mandatory_day_flatten(
            "SCALP_V2_CATASTROPHIC_STOP",
            force_sell=True,
            exit_type_name="STRATEGY",
        )
        is True
    )


@pytest.mark.asyncio
async def test_rate_limit_stops_the_same_call_retry_loop():
    calls = {"n": 0}

    async def preflight(_qty, _impact):
        return type("Pf", (), {"passed": True, "executable_qty": 1.0, "quantity": 1.0, "protected_limit_price": 90.0, "reject_reason": ""})()

    async def place(_qty, _limit):
        calls["n"] += 1
        _LIVE_ORDER_OUTCOME.set(LIVE_OUTCOME_RATE_LIMITED)

    result = await run_mandatory_exit_ioc_loop(
        quantity=1.0,
        preflight=preflight,
        place_ioc=place,
        is_meaningful=lambda qty: qty > 0,
        max_attempts=6,
    )
    assert calls["n"] == 1
    assert result.abandoned_reason == "venue_rate_limit"


@pytest.mark.asyncio
async def test_zero_fill_retries_use_a_fresh_preflight_each_attempt():
    books = []

    async def preflight(qty, impact):
        books.append((qty, impact))
        return type(
            "Pf",
            (),
            {
                "passed": True,
                "executable_qty": qty,
                "quantity": qty,
                "protected_limit_price": 90.0 - len(books),
                "reject_reason": "",
            },
        )()

    async def place(_qty, limit):
        _LIVE_ORDER_OUTCOME.set("")
        return {"filled": 0.0, "price": limit}

    result = await run_mandatory_exit_ioc_loop(
        quantity=1.0,
        preflight=preflight,
        place_ioc=place,
        is_meaningful=lambda qty: qty > 0,
        max_attempts=3,
    )
    assert len(books) == 3
    assert [book[1] for book in books] == [impact_for_attempt(0), impact_for_attempt(1), impact_for_attempt(2)]
    assert result.abandoned_reason == "residual_after_same_call"
    assert result.combined_order is None
