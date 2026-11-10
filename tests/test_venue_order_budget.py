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


def _short_day_limits(per_10s: int) -> list[dict]:
    return [
        {"rateLimitType": "ORDERS", "interval": "SECOND", "intervalNum": 10, "limit": per_10s},
        {"rateLimitType": "ORDERS", "interval": "DAY", "intervalNum": 1, "limit": 200000},
    ]


def test_restart_does_not_report_an_empty_10_second_window():
    clock = {"t": 5.0}
    budget = VenueOrderBudget(clock=lambda: clock["t"])
    budget.set_limits(VENUE)
    budget.arm_restart_blackout()
    view = budget.snapshot()
    assert view["orders_10s_authoritative"] is False
    assert view["orders_used_10s"] is None
    assert budget.try_acquire("exit") is False
    assert budget.try_acquire("entry") is False
    clock["t"] = 14.999
    assert budget.try_acquire("exit") is False
    clock["t"] = 15.0
    assert budget.try_acquire("exit") is True
    assert budget.snapshot()["orders_10s_authoritative"] is False
    assert budget.snapshot()["orders_used_10s_local"] == 1
    assert budget.snapshot()["account_exclusive_assumption"] is False


def test_10_second_boundary_uses_the_monotonic_clock():
    clock = {"t": 0.0}
    budget = VenueOrderBudget(clock=lambda: clock["t"])
    budget.set_limits(_short_day_limits(1))
    assert budget.try_acquire("exit") is True
    clock["t"] = 9.999
    assert budget.try_acquire("exit") is False
    clock["t"] = 10.0
    assert budget.try_acquire("exit") is True
    clock["t"] = 10.001
    assert budget.try_acquire("exit") is False


def test_daily_count_survives_restart_and_resets_on_the_utc_day(tmp_path):
    path = tmp_path / "budget.json"
    day = {"value": "2026-11-10"}
    clock = {"t": 0.0}
    first = VenueOrderBudget(clock=lambda: clock["t"], state_path=str(path), day_fn=lambda: day["value"])
    first.set_limits(VENUE)
    assert first.try_acquire("exit") is True
    assert first.try_acquire("exit") is True
    restarted = VenueOrderBudget(clock=lambda: clock["t"], state_path=str(path), day_fn=lambda: day["value"])
    restarted.set_limits(VENUE)
    view = restarted.snapshot()
    assert view["orders_used_day_local"] == 2
    assert view["orders_day_authoritative"] is False
    assert view["orders_limit_day"] == 200000
    day["value"] = "2026-11-11"
    assert restarted.try_acquire("exit") is True
    assert restarted.snapshot()["orders_used_day"] == 1


def test_each_ioc_attempt_consumes_its_own_slot():
    budget = VenueOrderBudget(clock=lambda: 0.0)
    budget.set_limits(_short_day_limits(3))
    granted = [budget.try_acquire("exit") for _ in range(6)]
    assert granted == [True, True, True, False, False, False]


def test_lost_ack_does_not_refund_the_slot():
    budget = VenueOrderBudget(clock=lambda: 0.0)
    budget.set_limits(_short_day_limits(2))
    assert budget.try_acquire("exit") is True
    # The venue may have counted the POST even though the response never returned.
    assert budget.snapshot()["orders_used_10s"] == 1
    assert budget.try_acquire("exit") is True
    assert budget.snapshot()["orders_used_10s"] == 2
    assert budget.try_acquire("exit") is False


def test_rate_limit_response_fills_the_named_window():
    clock = {"t": 0.0}
    budget = VenueOrderBudget(clock=lambda: clock["t"])
    budget.set_limits(VENUE)
    budget.note_rate_limit('{"code":-1015,"msg":"Too many new orders; current limit is 100 orders per 10 SECOND."}')
    assert budget.try_acquire("exit") is False
    assert budget.snapshot()["last_rate_limit_error"].startswith('{"code":-1015')
    clock["t"] = 10.0
    assert budget.try_acquire("exit") is True


def test_concurrent_reservations_stay_inside_the_limit():
    import threading

    budget = VenueOrderBudget(clock=lambda: 0.0)
    budget.set_limits(_short_day_limits(10))
    priorities = ["exit"] * 8 + ["entry"] * 8
    results: list[tuple[str, bool]] = []
    barrier = threading.Barrier(16)

    def go(priority: str) -> None:
        barrier.wait()
        results.append((priority, budget.try_acquire(priority)))

    threads = [threading.Thread(target=go, args=(priority,)) for priority in priorities]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert sum(1 for _priority, ok in results if ok) <= 10
    assert sum(1 for priority, ok in results if priority == "entry" and ok) <= 2


def test_venue_usage_raises_local_count_and_never_lowers_it():
    budget = VenueOrderBudget(clock=lambda: 0.0)
    budget.set_limits(VENUE)
    for _ in range(40):
        assert budget.try_acquire("exit") is True
    assert budget.apply_venue_usage(
        [
            {"rateLimitType": "ORDERS", "interval": "SECOND", "intervalNum": 10, "limit": 100, "count": 63},
            {"rateLimitType": "ORDERS", "interval": "DAY", "intervalNum": 1, "limit": 200000, "count": 70},
        ]
    )
    view = budget.snapshot()
    assert view["orders_used_10s_effective"] == 63
    assert view["orders_used_day_effective"] == 70
    assert view["orders_10s_authoritative"] is False
    assert view["orders_day_authoritative"] is False
    assert view["venue_count_known"] is True
    assert view["external_order_activity_detected"] is True
    assert budget.apply_venue_usage(
        [
            {"rateLimitType": "ORDERS", "interval": "SECOND", "intervalNum": 10, "limit": 100, "count": 10},
            {"rateLimitType": "ORDERS", "interval": "DAY", "intervalNum": 1, "limit": 200000, "count": 5},
        ]
    )
    later = budget.snapshot()
    assert later["orders_used_10s_local"] == 63
    assert later["orders_used_day_local"] == 70


def test_external_burst_after_a_snapshot_merges_to_the_venue_total():
    clock = {"t": 0.0}
    budget = VenueOrderBudget(clock=lambda: clock["t"])
    budget.set_limits(VENUE)
    budget.apply_venue_usage(
        [
            {"rateLimitType": "ORDERS", "interval": "SECOND", "intervalNum": 10, "limit": 100, "count": 10},
            {"rateLimitType": "ORDERS", "interval": "DAY", "intervalNum": 1, "limit": 200000, "count": 10},
        ]
    )
    for _ in range(5):
        assert budget.try_acquire("exit") is True
    budget.apply_venue_usage(
        [
            {"rateLimitType": "ORDERS", "interval": "SECOND", "intervalNum": 10, "limit": 100, "count": 35},
            {"rateLimitType": "ORDERS", "interval": "DAY", "intervalNum": 1, "limit": 200000, "count": 35},
        ]
    )
    view = budget.snapshot()
    assert view["orders_used_10s_effective"] == 35
    assert view["orders_used_day_effective"] == 35
    assert view["local_orders_since_venue_snapshot"] == 0
    clock["t"] = 10.0
    aged = budget.snapshot()
    assert aged["orders_used_10s_effective"] == 0
    assert aged["venue_count_age_ms"] == 10000.0
    assert aged["orders_10s_authoritative"] is False


def test_second_portfolio_writer_fails_closed(tmp_path):
    from backend.services.venue_order_budget import hold_writer_lock

    path = tmp_path / "writer.lock"
    assert hold_writer_lock(str(path)) is not None
    assert hold_writer_lock(str(path)) is None
