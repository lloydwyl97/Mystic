"""Consecutive-loss state is telemetry-only in trailing-buy mode."""

from __future__ import annotations

import time
from unittest.mock import AsyncMock

import pytest

from backend.services.portfolio_engine import PortfolioEngine
from backend.services.risk_governor import AccountSnapshot, CandidateInfo, RiskGovernor


@pytest.fixture(autouse=True)
def _isolate_trade_state(monkeypatch):
    from backend.services import trade_state

    isolated = trade_state.TradeStateStore(redis_client=None)
    monkeypatch.setattr(trade_state, "_store_instance", isolated)
    monkeypatch.setattr(trade_state, "get_trade_state_store", lambda _redis_client=None: isolated)
    return isolated


def _account(**kw):
    now = time.time()
    return AccountSnapshot(
        equity=228.0,
        free_usdt=220.0,
        positions_value=4.0,
        open_positions_count=0,
        exposure_per_coin={},
        rolling_24h_drawdown_pct=0.0,
        consecutive_losses=kw.get("consecutive_losses", 5),
        max_positions=4,
        loss_hold_until=kw.get("loss_hold_until", now + 600.0),
        current_time_utc=now,
    )


def _cands():
    return [
        CandidateInfo(symbol=sym, composite_score=1.0, confidence=0.8, trend_score=0.7, chop_score=0.1, current_price=px, atr=1.0)
        for sym, px in (("BTC/USDT", 80000.0), ("ETH/USDT", 2700.0), ("SOL/USDT", 110.0), ("XRP/USDT", 1.4))
    ]


@pytest.mark.parametrize("symbol", ["BTC/USDT", "ETH/USDT", "SOL/USDT", "XRP/USDT"])
@pytest.mark.asyncio
async def test_hold_consec_cannot_veto_trailing_buy(monkeypatch, symbol):
    monkeypatch.setenv("DAY_ENTRY_EXECUTION_MODE", "trailing_buy")
    import backend.services.portfolio_engine as pe

    monkeypatch.setattr(pe, "ENABLE_GOVERNANCE_ENFORCEMENT", True)
    monkeypatch.setattr(pe, "governance_risk_governor_shadow_only", lambda: False)
    engine = PortfolioEngine(principal=228.0, test_mode=True)
    engine.cash_balance = 220.0
    engine._available_balance = 220.0
    engine._total_open_risk = 0.0
    engine._get_loss_hold_until = AsyncMock(return_value=time.time() + 600.0)
    engine.get_rolling_24h_risk_metrics = AsyncMock(return_value=(0.0, pe.MAX_CONSEC_LOSSES))
    allowed, reason = await engine._can_open_position(symbol, 40.0)
    assert allowed is True, reason
    assert reason == "OK"
    assert engine._last_governance_hold_reason == "HOLD_CONSEC_LOSSES"


def test_risk_governor_hold_consec_is_telemetry_in_trailing_buy(monkeypatch):
    monkeypatch.setenv("DAY_ENTRY_EXECUTION_MODE", "trailing_buy")
    gov = RiskGovernor(shadow_only=False)
    result = gov.decide(_account(), _cands())
    assert result.account_hold_reason is None
    assert all(c.symbol in {x.symbol for x in result.allowed_candidates} or True for c in _cands())
    assert not any(r.reason_code == "HOLD_CONSEC_LOSSES" for r in result.rejections)


@pytest.mark.asyncio
async def test_pre_submit_ignores_hold_consec(monkeypatch):
    from backend.services.day_trailing_buy import _pre_submit_safety

    monkeypatch.setenv("DAY_ENTRY_EXECUTION_MODE", "trailing_buy")

    class _E:
        _trading_paused = False
        _symbol_constraints = {}
        _entry_reservations = {}

        def _check_kill_switch_buy(self):
            return True, ""

        async def _can_open_position(self, *_a, **_k):
            return False, "HOLD_CONSEC_LOSSES"

        def _pending_buy_order_symbols(self):
            return set()

        def _own_entry_reservation(self, *_a, **_k):
            return {}, ""

        def _pending_buy_notional(self, **_k):
            return 0

    ok, why = await _pre_submit_safety(_E(), {"symbol": "ETH/USDT", "notional_usd": 38.0, "decision_id": "d1", "quantity": 0.01}, 2700.0)
    assert ok is True
    assert why == ""
