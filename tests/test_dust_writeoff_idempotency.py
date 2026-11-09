"""Unchanged exchange dust is one close, not a writeoff on every reconcile."""

from __future__ import annotations

import os
import sqlite3
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from backend.services.portfolio_engine import OpenPosition, PortfolioEngine, make_position_key
from backend.services.protected_external_inventory import consumes_strategy_slot
from backend.services.trade_learning_writer import _ensure_table

LEGACY = "LEGACY_DAY_LIVE"
ETH_QTY = 1.8e-07
SOL_QTY = 6.046e-05
ETH_PX = 3200.0
SOL_PX = 93.0


def _lot(symbol: str, qty: float, price: float, *, trade_id: str, entry_time: float) -> OpenPosition:
    return OpenPosition(
        symbol=symbol,
        quantity=qty,
        entry_price=price,
        entry_time=entry_time,
        trade_id=trade_id,
        stop_price=0.0,
        take_profit_1_price=0.0,
        take_profit_2_price=0.0,
        highest_price=price,
        lowest_price=price,
        status="DUST_PENDING",
        engine_id=LEGACY,
        dust_qty_canonical=qty,
        dust_detected_at=entry_time,
        original_position_cost=qty * price,
    )


def _engine(tmp_path: Path) -> PortfolioEngine:
    os.environ["MYSTIC_RECONCILE_STATE_FILE"] = str(tmp_path / "reconcile.json")
    eng = PortfolioEngine(db_path=str(tmp_path / "dust.db"), principal=1000.0, test_mode=True)
    eng._ensure_db_schema()
    eng._live_execution_enabled = True
    eng.cash_balance = 992.62
    eng._available_balance = 992.62
    eng._live_service = SimpleNamespace(
        get_balance=AsyncMock(return_value={"status": "success", "balance": {"total": {}, "free": {}}, "fetched_at": time.time()}),
        get_market_price=AsyncMock(return_value={"price": ETH_PX}),
        order_fence_active=lambda *_a, **_k: False,
    )
    eng._ensure_symbol_constraints = AsyncMock()
    eng._persist_position_to_sqlite = AsyncMock()
    eng._delete_position_from_sqlite = AsyncMock()
    original = eng._recompute_positions_values

    async def _mark(prices=None, allow_network_mtm=True):
        return await original({"ETH/USDT": ETH_PX, "SOL/USDT": SOL_PX, "XRP/USDT": 0.55}, allow_network_mtm=False)

    eng._recompute_positions_values = _mark
    return eng


def _counts(db: str) -> tuple[int, int]:
    with sqlite3.connect(db) as conn:
        ledger = conn.execute("SELECT COUNT(*) FROM position_close_ledger WHERE close_reason='DUST_WRITEOFF'").fetchone()[0]
        learning = 0
        if conn.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='trade_learning_outcomes'").fetchone():
            learning = conn.execute("SELECT COUNT(*) FROM trade_learning_outcomes WHERE close_reason='DUST_WRITEOFF'").fetchone()[0]
    return int(ledger), int(learning)


def _null_pnl(db: str) -> None:
    with sqlite3.connect(db) as conn:
        bad_ledger = conn.execute("SELECT COUNT(*) FROM position_close_ledger WHERE close_reason='DUST_WRITEOFF' AND realized_profit IS NOT NULL").fetchone()[0]
        bad_learning = conn.execute("SELECT COUNT(*) FROM trade_learning_outcomes WHERE close_reason='DUST_WRITEOFF' AND net_profit_usd IS NOT NULL").fetchone()[0]
    assert bad_ledger == 0
    assert bad_learning == 0


async def _passes(eng: PortfolioEngine, n: int) -> None:
    balances = {"ETH": 0.0, "SOL": 0.0, "XRP": 0.0, "USDT": 992.62}
    for _ in range(n):
        await eng.run_live_reconcile(balances, free_balances=dict(balances))


async def test_unchanged_dust_is_written_off_once_and_kept(tmp_path: Path):
    eng = _engine(tmp_path)
    eth = _lot("ETH/USDT", ETH_QTY, ETH_PX, trade_id="eth-dust", entry_time=1_793_957_646.809)
    sol = _lot("SOL/USDT", SOL_QTY, SOL_PX, trade_id="sol-dust", entry_time=1_793_965_059.818)
    eng.open_positions[make_position_key(LEGACY, "ETH/USDT")] = eth
    eng.open_positions[make_position_key(LEGACY, "SOL/USDT")] = sol

    await _passes(eng, 1)
    assert _counts(eng.db_path) == (2, 2)
    _null_pnl(eng.db_path)

    await _passes(eng, 4)
    assert _counts(eng.db_path) == (2, 2)
    assert eth.status == "DUST_PENDING"
    assert sol.status == "DUST_PENDING"
    assert eth.quantity == pytest.approx(ETH_QTY)
    assert sol.quantity == pytest.approx(SOL_QTY)
    assert consumes_strategy_slot(eth) is False
    assert consumes_strategy_slot(sol) is False
    assert eng._engine_held_count("DAY_V2") == 0
    assert eng._positions_value == pytest.approx(ETH_QTY * ETH_PX + SOL_QTY * SOL_PX)
    assert eng._total_equity == pytest.approx(eng.cash_balance + eng._positions_value)


async def test_quantity_change_is_a_new_event_and_then_stable(tmp_path: Path):
    eng = _engine(tmp_path)
    sol = _lot("SOL/USDT", SOL_QTY, SOL_PX, trade_id="sol-dust", entry_time=1_793_965_059.818)
    eng.open_positions[make_position_key(LEGACY, "SOL/USDT")] = sol
    await _passes(eng, 3)
    assert _counts(eng.db_path) == (1, 1)

    sol.quantity = SOL_QTY / 2
    sol.dust_qty_canonical = sol.quantity
    await _passes(eng, 1)
    assert _counts(eng.db_path) == (2, 2)
    await _passes(eng, 3)
    assert _counts(eng.db_path) == (2, 2)
    _null_pnl(eng.db_path)
    assert sol.quantity == pytest.approx(SOL_QTY / 2)
    assert consumes_strategy_slot(sol) is False


async def test_other_symbol_and_later_trade_are_not_suppressed(tmp_path: Path):
    eng = _engine(tmp_path)
    eth = _lot("ETH/USDT", ETH_QTY, ETH_PX, trade_id="eth-dust", entry_time=1_793_957_646.809)
    xrp = _lot("XRP/USDT", 0.0004, 0.55, trade_id="xrp-dust", entry_time=1_794_100_000.0)
    eng.open_positions[make_position_key(LEGACY, "ETH/USDT")] = eth
    eng.open_positions[make_position_key(LEGACY, "XRP/USDT")] = xrp
    await _passes(eng, 2)
    assert _counts(eng.db_path) == (2, 2)

    del eng.open_positions[make_position_key(LEGACY, "ETH/USDT")]
    nxt = _lot("ETH/USDT", 2.5e-07, 3100.0, trade_id="eth-dust-next", entry_time=1_794_200_000.0)
    eng.open_positions[make_position_key(LEGACY, "ETH/USDT")] = nxt
    await _passes(eng, 1)
    assert _counts(eng.db_path) == (3, 3)
    await _passes(eng, 3)
    assert _counts(eng.db_path) == (3, 3)
    assert nxt.status == "DUST_PENDING"
    assert consumes_strategy_slot(nxt) is False


async def test_legacy_rows_without_event_key_are_not_duplicated(tmp_path: Path):
    eng = _engine(tmp_path)
    entry_time = 1_793_957_646.809
    _ensure_table(eng.db_path)
    with sqlite3.connect(eng.db_path) as conn:
        conn.execute(
            """
            INSERT INTO position_close_ledger (
                symbol, closed_at, closed_at_epoch, close_reason, manual_sell,
                realized_profit, realized_profit_unknown, cooldown_until, quantity, entry_price, detail
            ) VALUES ('ETH/USDT', '2026-11-06T16:06:55+00:00', ?, 'DUST_WRITEOFF', 0, NULL, 1, 0, ?, ?, 'source=periodic_reconcile')
            """,
            (entry_time, ETH_QTY, ETH_PX),
        )
        conn.execute(
            """
            INSERT INTO trade_learning_outcomes (
                written_at_utc, mode, symbol, entry_timestamp, close_reason,
                dust_remaining_qty, engine_id, manual_sell_flag, realized_profit_unknown, net_profit_usd
            ) VALUES ('2026-11-06 16:06:55', 'live', 'ETH/USDT', ?, 'DUST_WRITEOFF', ?, 'LEGACY_DAY_LIVE', 0, 1, NULL)
            """,
            (entry_time, ETH_QTY),
        )
        conn.commit()
    eth = _lot("ETH/USDT", ETH_QTY, ETH_PX, trade_id="eth-dust", entry_time=entry_time)
    eng.open_positions[make_position_key(LEGACY, "ETH/USDT")] = eth
    await _passes(eng, 4)
    assert _counts(eng.db_path) == (1, 1)
    assert eth.quantity == pytest.approx(ETH_QTY)
    assert consumes_strategy_slot(eth) is False


async def test_gone_residual_is_not_written_off_again(tmp_path: Path):
    eng = _engine(tmp_path)
    sol = _lot("SOL/USDT", SOL_QTY, SOL_PX, trade_id="sol-dust", entry_time=1_793_965_059.818)
    key = make_position_key(LEGACY, "SOL/USDT")
    eng.open_positions[key] = sol
    await _passes(eng, 2)
    assert _counts(eng.db_path) == (1, 1)

    sol.quantity = 0.0
    await eng._remove_dust_position_canonical_cleanup("SOL/USDT", sol)
    assert key not in eng.open_positions
    await eng._handle_vanished_exchange_position("SOL/USDT", sol, source="periodic_reconcile")
    await _passes(eng, 3)
    assert _counts(eng.db_path) == (1, 1)
    _null_pnl(eng.db_path)
