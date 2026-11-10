"""Exit evaluation must not wait on a slow Binance account call.

A held sell is not a stored order. The next pass reads the position again.
Catastrophic protection stays true because the low since entry is kept.
A learned or price-target exit is computed again and can disappear.
"""

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


def test_catastrophic_exit_stays_true_after_the_price_recovers():
    from backend.services.day_v2.live_exit_evaluator import evaluate_day_v2_exit
    from backend.services.scalp_v2.exit_evaluator import evaluate_scalp_v2_exit

    day = evaluate_day_v2_exit(
        engine_id="DAY_V2",
        entry_price=100.0,
        current_price=100.0,
        bar_low=90.0,
        highest_price=101.0,
        atr_at_entry=1.0,
        structural_anchor=0.0,
        target_price=110.0,
        entry_time=time.time() - 120.0,
        estimated_roundtrip_cost=0.00066,
    )
    assert day is not None and day["reason"] == "DAY_V2_CATASTROPHIC_PROTECTION"
    recovered = evaluate_day_v2_exit(
        engine_id="DAY_V2",
        entry_price=100.0,
        current_price=99.0,
        bar_low=99.5,
        highest_price=101.0,
        atr_at_entry=1.0,
        structural_anchor=0.0,
        target_price=110.0,
        entry_time=time.time() - 120.0,
        estimated_roundtrip_cost=0.00066,
        expected_terminal_net=0.01,
    )
    assert recovered is None

    class _Pos:
        engine_id = "SCALP_V2"
        entry_price = 100.0
        highest_price = 100.0
        lowest_price = 90.0
        symbol = "BTC/USDT"
        adaptive_decision = {}

    scalp = evaluate_scalp_v2_exit(position=_Pos(), current_price=100.0, net_pnl_pct=0.0, hold_minutes=1.0, bar_low=90.0, symbol="BTC/USDT")
    assert scalp["action"] == "sell" and scalp["reason"] == "SCALP_V2_CATASTROPHIC_STOP"
    _Pos.lowest_price = 99.5
    released = evaluate_scalp_v2_exit(
        position=_Pos(),
        current_price=100.0,
        net_pnl_pct=-0.0001,
        hold_minutes=1.0,
        bar_low=99.5,
        symbol="BTC/USDT",
        expected_terminal_net=0.01,
    )
    assert released.get("action") == "hold"


def test_day_hold_time_is_not_a_sell():
    from backend.services.day_v2.live_exit_evaluator import DAY_V2_MAX_HOLD_MINUTES, evaluate_day_v2_exit

    decision = evaluate_day_v2_exit(
        engine_id="DAY_V2",
        entry_price=100.0,
        current_price=100.0,
        bar_low=99.9,
        highest_price=100.2,
        atr_at_entry=1.0,
        structural_anchor=0.0,
        target_price=110.0,
        entry_time=time.time() - (DAY_V2_MAX_HOLD_MINUTES + 30) * 60.0,
        estimated_roundtrip_cost=0.00066,
        expected_terminal_net=0.01,
    )
    assert decision is None


@pytest.mark.asyncio
async def test_ownership_recovery_wakes_the_exit_loop_before_the_cadence():
    integ = object.__new__(PortfolioEngineIntegration)
    integ.is_running = True
    integ._exit_cycle_wake = asyncio.Event()
    started = time.perf_counter()

    async def _wait() -> None:
        await integ._scalp_fast_exit_wait(45)

    task = asyncio.create_task(_wait())
    await asyncio.sleep(0.05)
    integ._wake_exit_monitor()
    await asyncio.wait_for(task, timeout=1.0)
    assert time.perf_counter() - started < 1.0


@pytest.mark.asyncio
async def test_unresolved_sell_intent_blocks_a_second_sell(tmp_path):
    from backend.services.strategy_order_intent import _connect

    db = tmp_path / "intents.db"
    conn = _connect(str(db))
    now = time.time()
    conn.execute(
        """
        INSERT INTO strategy_order_intents (
            client_order_id, engine_id, trade_id, symbol, side, quantity, status, filled_qty, created_at, updated_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        ("abc", "DAY_V2", "t1", "BTC/USDT", "SELL", 1.0, "PENDING", 0.0, now, now),
    )
    conn.commit()
    conn.close()
    engine = PortfolioEngine.__new__(PortfolioEngine)
    engine._live_execution_enabled = True
    engine._ownership_state_known = True
    engine.db_path = str(db)
    result = await engine.execute_sell_fifo("BTC/USDT", 0.6, 100.0, "CATASTROPHIC", "startup")
    assert result is None
