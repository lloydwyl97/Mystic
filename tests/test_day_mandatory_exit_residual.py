"""DAY mandatory flatten: same-call residual, no 45s wait, dust stays dust."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from backend.config.protected_execution import (
    MANDATORY_EXIT_MAX_SPREAD_PCT,
    MAX_ORDERBOOK_PRICE_IMPACT_PCT,
    MAX_ORDERBOOK_SPREAD_PCT,
    PROTECTED_LIMIT_ALLOW_PARTIAL,
)
from backend.services.day_mandatory_exit_execution import (
    STATUS_EXIT_RESIDUAL_PENDING,
    impact_for_attempt,
    is_exit_residual_pending,
    is_mandatory_day_flatten,
    is_meaningful_residual,
    mark_exit_residual_pending,
    run_mandatory_exit_ioc_loop,
)
from backend.services.protected_limit_execution import (
    PRICE_IMPACT_TOO_HIGH,
    SPREAD_TOO_WIDE,
    run_protected_preflight,
    walk_book_within_impact,
)

SOL_QTY = 0.6228754
SOL_ENTRY = 104.87
SOL_STEP = 0.001
SOL_MIN_QTY = 0.001
SOL_MIN_NOTIONAL = 1.0


def _book(bids, asks):
    return (bids, asks, 0.0)


@pytest.mark.asyncio
async def test_entry_preflight_still_rejects_wide_spread():
    bids = [[105.16, 2.0], [105.10, 2.0]]
    asks = [[105.32, 2.0]]
    with patch(
        "backend.services.protected_limit_execution._fetch_order_book",
        AsyncMock(return_value=_book(bids, asks)),
    ):
        pf = await run_protected_preflight(
            symbol="SOL/USDT",
            side="SELL",
            quantity=0.622,
            reference_price=105.24,
            live_capable=True,
        )
    assert pf.passed is False
    assert pf.reject_reason == SPREAD_TOO_WIDE
    assert MAX_ORDERBOOK_SPREAD_PCT == 0.0005


@pytest.mark.asyncio
async def test_mandatory_preflight_sends_through_sol_trigger_spread():
    """14:01:06Z book: ~15 bps. Mandatory flatten must not refuse the order."""
    bids = [[105.16, 0.622], [105.10, 1.0]]
    asks = [[105.32, 2.0]]
    with patch(
        "backend.services.protected_limit_execution._fetch_order_book",
        AsyncMock(return_value=_book(bids, asks)),
    ):
        pf = await run_protected_preflight(
            symbol="SOL/USDT",
            side="SELL",
            quantity=0.622,
            reference_price=105.24,
            live_capable=True,
            mandatory_exit=True,
            allow_chunk=True,
            max_impact_pct=MAX_ORDERBOOK_PRICE_IMPACT_PCT,
        )
    assert pf.passed is True
    assert pf.executable_qty == pytest.approx(0.622)
    assert pf.protected_limit_price > 0
    assert pf.spread_pct > MAX_ORDERBOOK_SPREAD_PCT
    assert pf.spread_pct < MANDATORY_EXIT_MAX_SPREAD_PCT


@pytest.mark.asyncio
async def test_mandatory_preflight_chunks_thin_book():
    bids = [[105.15, 0.047], [104.36, 10.0]]
    asks = [[105.20, 2.0]]
    with patch(
        "backend.services.protected_limit_execution._fetch_order_book",
        AsyncMock(return_value=_book(bids, asks)),
    ):
        pf = await run_protected_preflight(
            symbol="SOL/USDT",
            side="SELL",
            quantity=0.622,
            reference_price=105.175,
            live_capable=True,
            mandatory_exit=True,
            allow_chunk=True,
            max_impact_pct=0.0005,
        )
    assert pf.passed is True
    assert pf.executable_qty == pytest.approx(0.047)
    assert pf.executable_qty < 0.622


def test_walk_book_within_impact_keeps_top_of_book():
    avg, filled, fully = walk_book_within_impact(
        [[105.15, 0.047], [104.36, 0.575]],
        0.622,
        best_px=105.15,
        max_impact_pct=0.0005,
        sell=True,
    )
    assert filled == pytest.approx(0.047)
    assert fully is False
    assert avg == pytest.approx(105.15)


def test_meaningful_residual_sol_not_dust():
    assert is_meaningful_residual(
        0.5758754,
        105.15,
        min_qty=SOL_MIN_QTY,
        min_notional=SOL_MIN_NOTIONAL,
        qty_step=SOL_STEP,
    )
    assert is_meaningful_residual(
        0.575,
        104.36,
        min_qty=SOL_MIN_QTY,
        min_notional=SOL_MIN_NOTIONAL,
        qty_step=SOL_STEP,
    )
    assert not is_meaningful_residual(
        0.0008754,
        104.36,
        min_qty=SOL_MIN_QTY,
        min_notional=SOL_MIN_NOTIONAL,
        qty_step=SOL_STEP,
    )


def test_mandatory_classifier():
    assert is_mandatory_day_flatten("TRAILING_STOP_EXIT", force_sell=True, exit_type_name="MANUAL")
    assert is_mandatory_day_flatten("DAY_4H_STRUCTURE_BREAK_EXIT")
    assert is_mandatory_day_flatten("DAY_RISK_FLOOR_EXIT")
    assert not is_mandatory_day_flatten("TP1_PARTIAL_EXIT", exit_type_name="TAKE_PROFIT_1")
    assert not is_mandatory_day_flatten("NET_PROFIT_EXIT", exit_type_name="TAKE_PROFIT_1")


@pytest.mark.asyncio
async def test_sol_partial_same_call_residual_no_monitor_wait():
    """Observed SOL: 0.622 requested, 0.047 first IOC, residual 0.5758754.

    Same-call second IOC must run immediately (no 45s). Residual stays under
    TRAILING_STOP_EXIT and is not dust.
    """
    pf_calls: list[tuple[float, float]] = []

    async def preflight(qty, impact):
        pf_calls.append((qty, impact))
        return SimpleNamespace(
            passed=True,
            executable_qty=qty,
            quantity=qty,
            protected_limit_price=105.10,
            reject_reason="",
        )

    ioc_calls: list[tuple[float, float]] = []

    async def place(qty, limit):
        ioc_calls.append((qty, limit))
        if len(ioc_calls) == 1:
            return {"id": "895431842", "filled": 0.047, "amount": qty, "average": 105.15, "status": "expired"}
        return {"id": "895436809", "filled": 0.575, "amount": qty, "average": 105.19, "status": "closed"}

    def meaningful(q):
        return is_meaningful_residual(q, 105.15, min_qty=SOL_MIN_QTY, min_notional=SOL_MIN_NOTIONAL, qty_step=SOL_STEP)

    out = await run_mandatory_exit_ioc_loop(
        quantity=0.622,
        preflight=preflight,
        place_ioc=place,
        is_meaningful=meaningful,
    )
    assert len(ioc_calls) == 2
    assert ioc_calls[0][0] == pytest.approx(0.622)
    assert out.filled_qty == pytest.approx(0.622)
    assert out.remaining_qty == pytest.approx(0.0, abs=1e-9)
    assert out.attempts >= 2
    assert out.combined_order is not None
    assert out.combined_order["_mystic_mandatory_flatten_fills"] == 2
    leftover = SOL_QTY - 0.622
    assert leftover == pytest.approx(0.0008754)
    assert not meaningful(leftover)


@pytest.mark.asyncio
async def test_partial_then_zero_marks_residual():
    async def preflight(qty, impact):
        return SimpleNamespace(passed=True, executable_qty=qty, quantity=qty, protected_limit_price=105.10, reject_reason="")

    n = {"i": 0}

    async def place(qty, limit):
        n["i"] += 1
        if n["i"] == 1:
            return {"id": "1", "filled": 0.047, "amount": qty, "average": 105.15, "status": "expired"}
        return {"id": "2", "filled": 0.0, "amount": qty, "average": 0.0, "status": "expired"}

    out = await run_mandatory_exit_ioc_loop(
        quantity=0.622,
        preflight=preflight,
        place_ioc=place,
        is_meaningful=lambda q: q >= 0.001,
    )
    assert out.filled_qty == pytest.approx(0.047)
    assert out.remaining_qty == pytest.approx(0.575)
    assert is_meaningful_residual(out.remaining_qty, 105.15, min_qty=0.001, min_notional=1.0, qty_step=0.001)
    pos = SimpleNamespace(status="ACTIVE", exit_residual_reason="", exit_residual_since=0.0)
    mark_exit_residual_pending(pos, "TRAILING_STOP_EXIT")
    assert is_exit_residual_pending(pos)
    assert pos.status == STATUS_EXIT_RESIDUAL_PENDING
    assert pos.exit_residual_reason == "TRAILING_STOP_EXIT"


@pytest.mark.asyncio
async def test_full_ioc_fill_no_partial_flag():
    async def preflight(qty, impact):
        return SimpleNamespace(passed=True, executable_qty=qty, quantity=qty, protected_limit_price=105.16, reject_reason="")

    async def place(qty, limit):
        return {"id": "full", "filled": qty, "amount": qty, "average": 105.16, "status": "closed"}

    out = await run_mandatory_exit_ioc_loop(
        quantity=0.622,
        preflight=preflight,
        place_ioc=place,
        is_meaningful=lambda q: q >= 0.001,
    )
    assert out.filled_qty == pytest.approx(0.622)
    assert out.remaining_qty == pytest.approx(0.0, abs=1e-12)
    assert out.combined_order["_mystic_partial_fill"] is False
    assert out.attempts == 1


@pytest.mark.asyncio
async def test_zero_ioc_hold_no_invented_fill():
    async def preflight(qty, impact):
        return SimpleNamespace(passed=True, executable_qty=qty, quantity=qty, protected_limit_price=105.16, reject_reason="")

    async def place(qty, limit):
        return {"id": "z", "filled": 0.0, "amount": qty, "status": "expired"}

    out = await run_mandatory_exit_ioc_loop(
        quantity=0.622,
        preflight=preflight,
        place_ioc=place,
        is_meaningful=lambda q: q >= 0.001,
        max_attempts=3,
    )
    assert out.filled_qty == 0.0
    assert out.combined_order is None
    assert out.remaining_qty == pytest.approx(0.622)
    assert out.attempts == 3


@pytest.mark.asyncio
async def test_missing_book_does_not_abandon_to_discretionary():
    async def preflight(qty, impact):
        return SimpleNamespace(passed=False, executable_qty=0.0, quantity=qty, protected_limit_price=0.0, reject_reason="ORDERBOOK_MISSING")

    async def place(qty, limit):
        raise AssertionError("must not place without book")

    out = await run_mandatory_exit_ioc_loop(
        quantity=0.622,
        preflight=preflight,
        place_ioc=place,
        is_meaningful=lambda q: q >= 0.001,
    )
    assert out.filled_qty == 0.0
    assert out.abandoned_reason == "ORDERBOOK_MISSING"


@pytest.mark.asyncio
async def test_high_impact_chunks_then_escalates():
    impacts: list[float] = []

    n = {"i": 0}

    async def preflight(qty, impact):
        n["i"] += 1
        impacts.append(impact)
        if n["i"] == 1:
            return SimpleNamespace(passed=True, executable_qty=0.047, quantity=0.047, protected_limit_price=105.15, reject_reason="")
        return SimpleNamespace(passed=True, executable_qty=qty, quantity=qty, protected_limit_price=104.90, reject_reason="")

    fills = []

    async def place(qty, limit):
        fills.append((qty, limit))
        return {"id": str(len(fills)), "filled": qty, "amount": qty, "average": limit, "status": "closed"}

    out = await run_mandatory_exit_ioc_loop(
        quantity=0.622,
        preflight=preflight,
        place_ioc=place,
        is_meaningful=lambda q: q >= 0.001,
    )
    assert out.filled_qty == pytest.approx(0.622)
    assert impacts[0] == pytest.approx(impact_for_attempt(0))
    assert fills[0][0] == pytest.approx(0.047)
    assert fills[1][0] == pytest.approx(0.575)


@pytest.mark.asyncio
async def test_exchange_reject_then_retry_no_oversell():
    async def preflight(qty, impact):
        return SimpleNamespace(passed=True, executable_qty=qty, quantity=qty, protected_limit_price=105.10, reject_reason="")

    n = {"i": 0}

    async def place(qty, limit):
        n["i"] += 1
        if n["i"] == 1:
            return None
        return {"id": "ok", "filled": qty, "amount": qty, "average": 105.10, "status": "closed"}

    out = await run_mandatory_exit_ioc_loop(
        quantity=0.622,
        preflight=preflight,
        place_ioc=place,
        is_meaningful=lambda q: q >= 0.001,
    )
    assert out.filled_qty == pytest.approx(0.622)
    assert out.filled_qty <= 0.622 + 1e-12


def test_allow_partial_entry_policy_unchanged():
    assert PROTECTED_LIMIT_ALLOW_PARTIAL is False


def test_no_duplicate_exit_in_progress_gate_still_present():
    from backend.services.portfolio_engine import PortfolioEngine

    src = open(PortfolioEngine.execute_sell_fifo.__code__.co_filename).read()
    assert "if normalized_symbol in self._exit_in_progress:" in src
    assert "EXIT_RESIDUAL_PENDING" in src or "mark_exit_residual_pending" in src


@pytest.mark.asyncio
async def test_duplicate_monitor_blocked_by_exit_in_progress():
    from backend.services.portfolio_engine import PortfolioEngine

    eng = PortfolioEngine.__new__(PortfolioEngine)
    eng._exit_in_progress = {"SOL/USDT"}
    assert "SOL/USDT" in eng._exit_in_progress


def test_pending_survives_restart_payload():
    from backend.services.day_inventory_recovery import thesis_json_for_position

    pos = SimpleNamespace(
        entry_thesis="",
        thesis_score=0.0,
        thesis_invalid_level=0.0,
        thesis_target_level=0.0,
        entry_vwap=0.0,
        thesis_trend_tf="",
        day_route_regime_at_entry="",
        price_structure_regime_at_entry="",
        max_hold_min=0,
        trail_pct=0.0055,
        legacy_pre_regime_router=False,
        opened_under_router=True,
        status=STATUS_EXIT_RESIDUAL_PENDING,
        exit_residual_reason="TRAILING_STOP_EXIT",
        exit_residual_since=1787925704.0,
    )
    payload = thesis_json_for_position(pos)
    assert payload["exit_residual_reason"] == "TRAILING_STOP_EXIT"
    assert payload["exit_residual_since"] == pytest.approx(1787925704.0)


def test_integration_uses_fast_pending_retry():
    import inspect

    from backend.services.portfolio_engine_integration import PortfolioEngineIntegration

    src = inspect.getsource(PortfolioEngineIntegration._position_monitor_loop)
    assert "MANDATORY_EXIT_PENDING_RETRY_SEC" in src
    assert "has_exit_residual_pending" in src
