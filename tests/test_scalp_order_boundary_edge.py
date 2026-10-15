"""SCALP reprices its edge at the order boundary and labels venue outcomes truthfully."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from backend.services import protected_limit_execution as ple
from backend.services.scalp_v2.executable_edge import edge_at_order_boundary

# SOL 2026-10-15 14:48Z: ranked +4.36 bps at ask 104.36; a bid traded at 104.44 at 14:48:16,
# so the ask at submission was at least 104.45.
SOL_EDGE = {"final_executable_edge_pct": 0.00043584, "live_cost_pct": 0.00075768}


def test_sol_incident_edge_is_gone_at_the_moved_ask():
    out = edge_at_order_boundary(SOL_EDGE, arm_price=104.36, best_ask=104.45, spread_pct=0.0001, impact_pct=0.0)
    assert out is not None
    assert out["ask_drift"] == pytest.approx(0.09 / 104.36)
    assert out["edge_now"] < 0
    assert out["eligible"] is False


def test_unchanged_ask_keeps_the_ranked_edge_with_the_current_cost():
    out = edge_at_order_boundary(SOL_EDGE, arm_price=104.36, best_ask=104.36, spread_pct=0.0001, impact_pct=0.0)
    assert out["ask_drift"] == 0.0
    assert out["edge_now"] == pytest.approx(0.00043584 + 0.00075768 - out["cost_now"])
    assert out["eligible"] is True


def test_wider_spread_and_impact_at_submission_are_charged():
    tight = edge_at_order_boundary(SOL_EDGE, arm_price=104.36, best_ask=104.36, spread_pct=0.0001, impact_pct=0.0)
    wide = edge_at_order_boundary(SOL_EDGE, arm_price=104.36, best_ask=104.36, spread_pct=0.0009, impact_pct=0.0004)
    assert wide["edge_now"] < tight["edge_now"]


def test_a_lower_ask_is_never_credited():
    same = edge_at_order_boundary(SOL_EDGE, arm_price=104.36, best_ask=104.36, spread_pct=0.0001, impact_pct=0.0)
    lower = edge_at_order_boundary(SOL_EDGE, arm_price=104.36, best_ask=104.00, spread_pct=0.0001, impact_pct=0.0)
    assert lower["edge_now"] == pytest.approx(same["edge_now"])


@pytest.mark.parametrize(
    ("edge", "arm", "ask"),
    [(None, 104.36, 104.36), ({}, 104.36, 104.36), ({"final_executable_edge_pct": 0.001}, 104.36, 104.36), (SOL_EDGE, 0.0, 104.36), (SOL_EDGE, 104.36, 0.0)],
)
def test_unusable_inputs_prove_nothing(edge, arm, ask):
    assert edge_at_order_boundary(edge, arm_price=arm, best_ask=ask, spread_pct=0.0001, impact_pct=0.0) is None


def _engine(tmp_path, monkeypatch, *, best_ask, adapter):
    from backend.services.portfolio_engine import PortfolioEngine

    engine = PortfolioEngine.__new__(PortfolioEngine)
    engine.open_positions = {}
    engine._available_balance = 500.0
    engine._total_equity = 500.0
    engine._global_cash_lock = asyncio.Lock()
    engine.db_path = str(tmp_path / "live.db")
    engine.last_buy_reject_reason = "STALE_FROM_EARLIER_ATTEMPT"
    engine._check_kill_switch_buy = lambda: (True, "")
    engine._normalize_order_amount = lambda **_k: (1.309, "ok", 136.6)
    engine._live_execution_enabled = True
    engine._live_service = object()
    engine._now_ms = lambda: 1
    preflight = MagicMock(passed=True, protected_limit_price=best_ask, reject_reason="", best_ask=best_ask, spread_pct=0.0001, price_impact_pct=0.0, book_age_sec=0.2)
    monkeypatch.setattr("backend.config.live_test_mode.can_place_live_orders_sync", lambda: (True, ""))
    monkeypatch.setattr("backend.services.protected_limit_execution.run_protected_preflight", AsyncMock(return_value=preflight))
    monkeypatch.setattr("backend.services.protected_limit_execution.execute_protected_limit_live", adapter)
    return engine


async def _buy(engine, adaptive_decision):
    return await engine.execute_scalp_v2_buy_live(
        "SOL/USDT",
        1.309,
        104.36,
        opportunity_id="opp-sol",
        entry_authority="SCALP_V2_CONFIRMED",
        adaptive_decision=adaptive_decision,
    )


@pytest.mark.asyncio
async def test_decayed_edge_sends_no_order(tmp_path, monkeypatch):
    adapter = AsyncMock(return_value=None)
    engine = _engine(tmp_path, monkeypatch, best_ask=104.45, adapter=adapter)

    assert await _buy(engine, {"executable_edge": SOL_EDGE}) is None
    adapter.assert_not_called()
    assert engine.last_buy_reject_reason == "EDGE_DECAYED_AT_ORDER"


@pytest.mark.asyncio
async def test_missing_ranked_edge_sends_no_order(tmp_path, monkeypatch):
    adapter = AsyncMock(return_value=None)
    engine = _engine(tmp_path, monkeypatch, best_ask=104.36, adapter=adapter)

    assert await _buy(engine, None) is None
    adapter.assert_not_called()
    assert engine.last_buy_reject_reason == "EDGE_UNPROVEN_AT_ORDER"


@pytest.mark.asyncio
async def test_kill_switch_block_is_labelled_not_stale(tmp_path, monkeypatch):
    adapter = AsyncMock(return_value=None)
    engine = _engine(tmp_path, monkeypatch, best_ask=104.36, adapter=adapter)
    engine._check_kill_switch_buy = lambda: (False, "KILL_SWITCH_PAUSE_BUYS: operator")

    assert await _buy(engine, {"executable_edge": SOL_EDGE}) is None
    adapter.assert_not_called()
    assert engine.last_buy_reject_reason == "KILL_OR_PAUSE"


class _Venue:
    def __init__(self, order):
        self._order = order
        self.fetch_order = AsyncMock(return_value={"status": "error"})
        self.cancel_order = AsyncMock(return_value={"status": "success"})

    async def _ensure_initialized(self):
        return None

    async def place_order(self, **_kwargs):
        return {"status": "success", "order": dict(self._order)}


@pytest.fixture
def _no_enrich(monkeypatch):
    async def _enrich(_svc, order, _sym):
        return order

    async def _no_sleep(_s):
        return None

    monkeypatch.setattr(ple, "_enrich_live_order_fills", _enrich)
    monkeypatch.setattr(ple.asyncio, "sleep", _no_sleep)


@pytest.mark.asyncio
@pytest.mark.usefixtures("_no_enrich")
async def test_unfilled_ioc_is_expired_unfilled_not_rejected():
    venue = _Venue({"id": "936042114", "filled": 0.0, "amount": 1.309, "status": "expired"})
    out = await ple.execute_protected_limit_live(venue, symbol="SOLUSDT", side="buy", quantity=1.309, limit_price=104.36)
    assert out is None
    assert ple.last_live_order_outcome() == ple.LIVE_OUTCOME_EXPIRED_UNFILLED


@pytest.mark.asyncio
@pytest.mark.usefixtures("_no_enrich")
async def test_venue_rejected_ioc_stays_exchange_rejected():
    venue = _Venue({"id": "1", "filled": 0.0, "amount": 1.309, "status": "rejected"})
    assert await ple.execute_protected_limit_live(venue, symbol="SOLUSDT", side="buy", quantity=1.309, limit_price=104.36) is None
    assert ple.last_live_order_outcome() == ple.LIVE_OUTCOME_EXCHANGE_REJECTED


@pytest.mark.asyncio
@pytest.mark.usefixtures("_no_enrich")
async def test_engine_records_the_venue_outcome(tmp_path, monkeypatch):
    venue = _Venue({"id": "936042114", "filled": 0.0, "amount": 1.309, "status": "expired"})
    real = ple.execute_protected_limit_live

    async def _adapter(_svc, **kwargs):
        return await real(venue, **kwargs)

    engine = _engine(tmp_path, monkeypatch, best_ask=104.36, adapter=_adapter)
    assert await _buy(engine, {"executable_edge": SOL_EDGE}) is None
    assert engine.last_buy_reject_reason == "EXPIRED_UNFILLED"
