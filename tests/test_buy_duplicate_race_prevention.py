"""
Regression: two independent BUY attempts for the same symbol must never both
execute. Root cause of a real incident: two BUYs for ETH/USDT executed ~60s
apart across consecutive bar cycles; the second silently overwrote the first
position's SQLite row (INSERT OR REPLACE) losing its cost basis while its
cash had already been debited — a real capital leak.

Two independent layers now guard against this:
  1. execute_buy_fifo acquires a per-symbol asyncio.Lock (previously declared
     in __init__ as `_buy_execution_locks` but never acquired anywhere) before
     delegating to the real implementation, serializing concurrent attempts.
  2. The "already open" duplicate check additionally does a fresh SQLite read
     (not just the in-memory `self.open_positions` dict), so even a stale/lost
     in-memory entry cannot let a duplicate buy through.
"""

from __future__ import annotations

import asyncio
import sqlite3
import tempfile
import time
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from backend.services.portfolio_engine import PortfolioEngine


@pytest.mark.asyncio
async def test_concurrent_buys_for_same_symbol_are_serialized_by_the_lock():
    engine = PortfolioEngine(principal=25_000.0, test_mode=True)
    call_order: list[str] = []

    async def _fake_impl(symbol, *args, **kwargs):
        call_order.append(f"start:{symbol}")
        await asyncio.sleep(0.02)  # simulate work inside the critical section
        call_order.append(f"end:{symbol}")
        return {"symbol": symbol}

    engine._execute_buy_fifo_locked = _fake_impl

    await asyncio.gather(
        engine.execute_buy_fifo("BTC/USDT", 1.0, 100.0, 95.0, 1.0, 0.7, 0, None, decision_id="d1"),
        engine.execute_buy_fifo("BTC/USDT", 1.0, 100.0, 95.0, 1.0, 0.7, 0, None, decision_id="d2"),
    )

    # Serialized: the second call's start must not appear before the first's end.
    assert call_order == ["start:BTC/USDT", "end:BTC/USDT", "start:BTC/USDT", "end:BTC/USDT"]


@pytest.mark.asyncio
async def test_different_symbols_are_not_serialized_against_each_other():
    engine = PortfolioEngine(principal=25_000.0, test_mode=True)
    started = []
    release = asyncio.Event()

    async def _fake_impl(symbol, *args, **kwargs):
        started.append(symbol)
        if len(started) < 2:
            await release.wait()
        else:
            release.set()
        return {"symbol": symbol}

    engine._execute_buy_fifo_locked = _fake_impl

    await asyncio.wait_for(
        asyncio.gather(
            engine.execute_buy_fifo("BTC/USDT", 1.0, 100.0, 95.0, 1.0, 0.7, 0, None),
            engine.execute_buy_fifo("ETH/USDT", 1.0, 100.0, 95.0, 1.0, 0.7, 0, None),
        ),
        timeout=2.0,
    )
    assert set(started) == {"BTC/USDT", "ETH/USDT"}


@pytest.mark.asyncio
async def test_lock_is_reused_across_calls_for_the_same_symbol():
    engine = PortfolioEngine(principal=25_000.0, test_mode=True)
    engine._execute_buy_fifo_locked = AsyncMock(return_value=None)

    await engine.execute_buy_fifo("XRP/USDT", 1.0, 1.0, 0.9, 0.01, 0.6, 0, None)
    lock1 = engine._buy_execution_locks.get("XRP/USDT")
    await engine.execute_buy_fifo("XRP/USDT", 1.0, 1.0, 0.9, 0.01, 0.6, 0, None)
    lock2 = engine._buy_execution_locks.get("XRP/USDT")

    assert lock1 is not None
    assert lock1 is lock2, "the same per-symbol lock object must be reused, not recreated each call"


@pytest.mark.asyncio
async def test_fresh_sqlite_read_catches_position_missing_from_stale_memory():
    """Directly exercises the fresh-read duplicate guard: a position exists in
    SQLite but is (deliberately, simulating staleness) absent from
    self.open_positions — the authoritative check must still find it."""
    with tempfile.TemporaryDirectory() as tmp:
        db_path = Path(tmp) / "dup.db"
        engine = PortfolioEngine(db_path=str(db_path), principal=25_000.0, test_mode=True)
        engine._ensure_db_schema()
        with sqlite3.connect(str(db_path)) as conn:
            conn.execute(
                """
                INSERT INTO portfolio_engine_positions (
                    symbol, quantity, entry_price, entry_time, trade_id,
                    stop_price, take_profit_1_price, take_profit_2_price,
                    highest_price, atr_at_entry, entry_bar_timestamp,
                    confidence_at_entry, last_updated, sleeve
                ) VALUES ('ETH/USDT', 2.06, 1818.0, ?, 'race-test-buy-1',
                          1787.0, 1841.0, 1853.0, 1818.0, 21.0, 0, 0.83, datetime('now'), 'ACTIVE')
                """,
                (time.time(),),
            )
            conn.commit()

        assert "ETH/USDT" not in engine.open_positions  # simulated staleness
        row = await asyncio.to_thread(engine._fetch_authoritative_open_position_sync, "ETH/USDT")
        assert row is not None, "fresh SQLite read must find the position even though memory does not have it"
        qty, trade_id, entry_price = row
        assert float(qty) == pytest.approx(2.06)
        assert trade_id == "race-test-buy-1"
