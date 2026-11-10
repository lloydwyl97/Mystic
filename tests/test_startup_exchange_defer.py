"""Exit evaluation must not wait on a slow Binance account call."""

from __future__ import annotations

import asyncio
import inspect
import time

import pytest

from backend.services.portfolio_engine import PortfolioEngine
from backend.services.portfolio_engine_integration import PortfolioEngineIntegration


def test_account_reconcile_is_not_on_the_blocking_startup_path():
    init_src = inspect.getsource(PortfolioEngine.initialize_from_canonical_sources)
    assert "await self._live_bootstrap_reconcile()" not in init_src
    assert "await self._recover_strategy_order_intents()" not in init_src
    recovery = inspect.getsource(PortfolioEngineIntegration._run_startup_exchange_recovery)
    assert "_recover_strategy_order_intents" in recovery
    assert "_live_bootstrap_reconcile" in recovery
    start = inspect.getsource(PortfolioEngineIntegration.start)
    assert start.index("_position_monitor_loop") < start.index("_run_startup_exchange_recovery")
    loop = inspect.getsource(PortfolioEngineIntegration._position_monitor_loop)
    assert loop.index("_monitor_positions_once") < loop.index("_purge_stale_bd_exit_hard_pause_state")


@pytest.mark.asyncio
async def test_sell_is_held_until_venue_quantity_is_known():
    engine = PortfolioEngine.__new__(PortfolioEngine)
    engine._live_execution_enabled = True
    engine._ownership_state_known = False
    result = await engine.execute_sell_fifo("BTC/USDT", 0.01, 100.0, "CATASTROPHIC", "startup")
    assert result is None


@pytest.mark.asyncio
async def test_failed_account_call_does_not_mark_ownership_known():
    engine = PortfolioEngine.__new__(PortfolioEngine)
    engine._ownership_state_known = False
    engine._account_reconciliation_status = "pending"
    written: list[str] = []

    class _Slow:
        async def get_balance(self, *_args, **_kwargs):
            await asyncio.sleep(0.25)
            return {"status": "error", "balance": {}}

    engine._live_service = _Slow()
    engine._write_reconcile_state_file = lambda *_a, **_k: written.append("unavailable")
    started = time.perf_counter()
    task = asyncio.create_task(engine._live_bootstrap_reconcile())
    await asyncio.sleep(0)
    assert time.perf_counter() - started < 0.1
    await task
    assert engine._ownership_state_known is False
    assert engine._account_reconciliation_status == "unavailable"
    assert written == ["unavailable"]


@pytest.mark.asyncio
async def test_account_timeout_keeps_ownership_unknown():
    engine = PortfolioEngine.__new__(PortfolioEngine)
    engine._ownership_state_known = False

    class _Down:
        async def get_balance(self, *_args, **_kwargs):
            raise TimeoutError("binance down")

    engine._live_service = _Down()
    engine._write_reconcile_state_file = lambda *_a, **_k: None
    await engine._live_bootstrap_reconcile()
    assert engine._ownership_state_known is False
    assert engine._account_reconciliation_status == "unavailable"
