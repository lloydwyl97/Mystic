"""Live DAY and SCALP buys are not rejected by legacy thesis or exit-cooldown rules."""

from __future__ import annotations

import time
from pathlib import Path

import pytest

from backend.config.day_entry_execution import learned_exit_contract_buy
from backend.services.portfolio_engine import PortfolioEngine
from backend.services.trade_state import TradeStateStore


def test_live_buy_path_treats_learned_contract_as_telemetry():
    src = Path("backend/services/portfolio_engine.py").read_text()
    assert "learned_exit_contract_buy(entry_authority, fill_engine_id)" in src
    assert "learned exit contract — not enforced" in src


def test_learned_exit_contract_covers_confirmed_authorities_only():
    assert learned_exit_contract_buy("DAY_V2_CONFIRMED", "") is True
    assert learned_exit_contract_buy("SCALP_V2_CONFIRMED", "") is True
    assert learned_exit_contract_buy("", "DAY_V2") is True
    assert learned_exit_contract_buy("", "SCALP_V2") is True
    assert learned_exit_contract_buy("DAY_TRAILING_BUY_CONFIRMED", "") is False
    assert learned_exit_contract_buy("", "") is False


@pytest.mark.asyncio
async def test_learned_engine_cooldown_does_not_block_and_in_trade_still_does():
    store = TradeStateStore(redis_client=None)
    now = time.time()
    store.on_exit("BTC/USDT", 100.0, "LEARNED_CONTINUATION_EXIT", engine_id="DAY_V2")
    allowed, reason = await store.allow_new_entry_async("BTC/USDT", "buy", now, engine_id="DAY_V2")
    assert allowed is True
    assert reason == "ENTRY_ALLOWED"

    store.on_exit("ETH/USDT", 100.0, "LEARNED_CONTINUATION_EXIT", engine_id="SCALP_V2")
    scalp_ok, scalp_reason = await store.allow_new_entry_async("ETH/USDT", "buy", now, engine_id="SCALP_V2")
    assert scalp_ok is True, scalp_reason

    store.on_entry_fill("BTC/USDT", 100.0, engine_id="DAY_V2")
    blocked, blocked_reason = await store.allow_new_entry_async("BTC/USDT", "buy", now, engine_id="DAY_V2")
    assert blocked is False
    assert blocked_reason == "ALREADY_IN_TRADE"

    store.on_exit("XRP/USDT", 1.0, "NET_PROFIT_EXIT", engine_id="")
    legacy_ok, legacy_reason = await store.allow_new_entry_async("XRP/USDT", "buy", now, engine_id="")
    assert legacy_ok is False
    assert legacy_reason.startswith("COOLDOWN_ACTIVE_UNTIL_")


@pytest.mark.asyncio
async def test_portfolio_day_v2_cooldown_does_not_block(monkeypatch):
    import backend.services.portfolio_engine as pe
    from backend.services import trade_state

    isolated = trade_state.TradeStateStore(redis_client=None)
    monkeypatch.setattr(trade_state, "_store_instance", isolated)
    monkeypatch.setattr(trade_state, "get_trade_state_store", lambda _redis_client=None: isolated)
    monkeypatch.setattr(pe, "ENABLE_TRADE_STATE_ENTRY_BLOCKING", True)

    isolated.on_exit("SOL/USDT", 120.0, "LEARNED_CONTINUATION_EXIT", engine_id="DAY_V2")
    engine = PortfolioEngine(principal=25_000.0, test_mode=True)
    engine.cash_balance = 25_000.0
    engine._available_balance = 25_000.0
    engine._total_open_risk = 0.0
    engine._last_bar_timestamp = int(time.time())
    engine._bar_interval_seconds = 60
    allowed, reason = await engine._can_open_position("SOL/USDT", 100.0, engine_id="DAY_V2")
    assert allowed is True, reason
