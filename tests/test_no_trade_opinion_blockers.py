"""
Regression: Mystic is a ranking/trading engine, not a permission bot.

Loss-streak pauses, sell cooldowns (symbol + global, wall-clock or bar-based),
the persistent close-ledger cooldown, and bear-regime "max one position" must
never block _can_open_position. Only genuine operational/safety conditions
(account status, trading pause, max positions, duplicate symbol, insufficient
cash, risk cap) are allowed to return False.
"""

from __future__ import annotations

import time
from unittest.mock import AsyncMock, MagicMock

import pytest

from backend.services.portfolio_engine import DAY_TRADE_SYMBOLS, PortfolioEngine


def _base_engine() -> PortfolioEngine:
    engine = PortfolioEngine(principal=25_000.0, test_mode=True)
    engine.cash_balance = 25_000.0
    engine._available_balance = 25_000.0
    engine._total_open_risk = 0.0
    engine._last_bar_timestamp = int(time.time())
    engine._bar_interval_seconds = 60
    engine._lookup_position_close_cooldown = MagicMock(return_value=0.0)
    engine._get_loss_hold_until = AsyncMock(return_value=None)
    engine._ensure_symbol_constraints = AsyncMock()
    return engine


@pytest.mark.asyncio
async def test_loss_streak_pause_does_not_block_buy():
    engine = _base_engine()
    engine._quality_filter_state.loss_streak_pause_until = time.time() + 7_200.0  # active "pause"
    allowed, reason = await engine._can_open_position("BTC/USDT", 100.0)
    assert allowed is True, f"loss-streak pause must not block buys, got reason={reason}"


@pytest.mark.asyncio
async def test_global_sell_cooldown_does_not_block_buy():
    engine = _base_engine()
    engine._quality_filter_state.global_cooldown_wall = time.time() + 600.0
    engine._quality_filter_state.global_cooldown_until = engine._last_bar_timestamp + 6000
    allowed, reason = await engine._can_open_position("ETH/USDT", 100.0)
    assert allowed is True, f"global sell cooldown must not block buys, got reason={reason}"


@pytest.mark.asyncio
async def test_symbol_sell_cooldown_does_not_block_buy():
    engine = _base_engine()
    engine._quality_filter_state.symbol_cooldown_wall["SOL/USDT"] = time.time() + 2_400.0
    engine._quality_filter_state.symbol_cooldowns["SOL/USDT"] = engine._last_bar_timestamp + 2400
    allowed, reason = await engine._can_open_position("SOL/USDT", 100.0)
    assert allowed is True, f"symbol sell cooldown must not block buys, got reason={reason}"


@pytest.mark.asyncio
async def test_persistent_close_ledger_cooldown_does_not_block_buy():
    engine = _base_engine()
    engine._lookup_position_close_cooldown = MagicMock(return_value=time.time() + 2_400.0)
    allowed, reason = await engine._can_open_position("XRP/USDT", 100.0)
    assert allowed is True, f"persistent close-ledger cooldown must not block buys, got reason={reason}"


@pytest.mark.asyncio
async def test_bear_regime_does_not_cap_to_one_day_position():
    engine = _base_engine()
    engine._is_bear_day_regime = MagicMock(return_value=True)
    engine._count_open_day_top4_positions = MagicMock(return_value=1)
    symbol = next(iter(DAY_TRADE_SYMBOLS))
    allowed, reason = await engine._can_open_position(symbol.replace("USDT", "/USDT") if "/" not in symbol else symbol, 100.0)
    assert allowed is True, f"bear regime must not hard-cap DAY positions to one, got reason={reason}"


@pytest.mark.asyncio
async def test_genuine_safety_blocks_still_enforced():
    """Sanity check: real operational conditions must still block (not everything was removed)."""
    from backend.services.portfolio_engine import AccountStatus

    engine = _base_engine()
    engine._account_status = AccountStatus.DELEVERAGING
    allowed, reason = await engine._can_open_position("BTC/USDT", 100.0)
    assert allowed is False
    assert reason == "DELEVERAGING_IN_PROGRESS"

    engine2 = _base_engine()
    engine2._available_balance = 10.0
    engine2.cash_balance = 10.0
    allowed2, reason2 = await engine2._can_open_position("BTC/USDT", 5_000.0)
    assert allowed2 is False
    assert "CASH" in reason2 or "INSUFFICIENT" in reason2


@pytest.mark.asyncio
async def test_portfolio_trade_state_exception_fails_open(monkeypatch):
    import backend.services.portfolio_engine as pe
    import backend.services.trade_state as trade_state

    class _BrokenStore:
        async def allow_new_entry_async(self, *_args, **_kwargs):
            raise RuntimeError("state backend unavailable")

    monkeypatch.setattr(pe, "ENABLE_TRADE_STATE_ENTRY_BLOCKING", True)
    monkeypatch.setattr(pe, "ENABLE_GOVERNANCE_ENFORCEMENT", False)
    monkeypatch.setattr(trade_state, "get_trade_state_store", lambda: _BrokenStore())

    allowed, reason = await _base_engine()._can_open_position("BTC/USDT", 100.0)

    assert allowed is True, reason


@pytest.mark.asyncio
async def test_trade_state_store_read_exception_fails_open():
    from backend.services.trade_state import TradeStateStore

    class _BrokenRedis:
        def hgetall(self, _key):
            raise RuntimeError("redis unavailable")

    allowed, reason = await TradeStateStore(_BrokenRedis()).allow_new_entry_async("ETH/USDT", "buy", time.time())

    assert allowed is True
    assert reason.startswith("GATE_ERROR_FAIL_OPEN:")


@pytest.mark.asyncio
async def test_loss_hold_does_not_preempt_operational_cash_check(monkeypatch):
    import backend.services.portfolio_engine as pe

    monkeypatch.setattr(pe, "ENABLE_TRADE_STATE_ENTRY_BLOCKING", False)
    monkeypatch.setattr(pe, "ENABLE_GOVERNANCE_ENFORCEMENT", True)
    monkeypatch.setattr(pe, "governance_risk_governor_shadow_only", lambda: False)
    engine = _base_engine()
    engine._available_balance = 0.0
    engine.cash_balance = 0.0
    engine._get_loss_hold_until = AsyncMock(return_value=time.time() + 600.0)
    engine.get_rolling_24h_risk_metrics = AsyncMock(return_value=(0.0, pe.MAX_CONSEC_LOSSES))

    allowed, reason = await engine._can_open_position("XRP/USDT", 100.0)

    assert allowed is False
    assert reason == "INSUFFICIENT_CASH"
    engine.get_rolling_24h_risk_metrics.assert_not_awaited()
