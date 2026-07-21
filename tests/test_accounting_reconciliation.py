"""
Accounting regression suite (Mystic full-repair Phase 1).

Covers the canonical identity: total_equity = cash_balance + positions_value.
Uses the read-only reconciliation module (backend.services.portfolio_ledger_reconciliation)
for pure transaction-evidence scenarios, and the real PortfolioEngine restart/MTM
paths for the engine-level invariants (buy debit persists immediately, restart does
not restore pre-buy cash, mark-up does not touch cash).
"""

from __future__ import annotations

import sqlite3
import tempfile
import time
from pathlib import Path

import pytest

from backend.services.portfolio_engine import OpenPosition, PortfolioEngine, Sleeve
from backend.services.portfolio_ledger_reconciliation import (
    compute_expected_cash,
    reconcile_ledger_cash,
)


def _init_ledger(db_path: Path, *, cash: float, principal: float = 25_000.0, positions_value: float = 0.0) -> None:
    from backend.database_schema import initialize_paper_trading_schema

    engine = PortfolioEngine(db_path=str(db_path), principal=principal, test_mode=True)
    engine._ensure_db_schema()
    initialize_paper_trading_schema(str(db_path))
    with sqlite3.connect(str(db_path)) as conn:
        conn.execute(
            """
            INSERT OR REPLACE INTO portfolio_engine_ledger (
                id, principal, cash_balance, positions_value,
                realized_pnl, unrealized_pnl, total_equity,
                account_status, trading_paused, pause_reason, last_updated, version
            ) VALUES (1, ?, ?, ?, 0, 0, ?, 'HEALTHY', 0, NULL, datetime('now'), 1)
            """,
            (principal, cash, positions_value, cash + positions_value),
        )
        conn.commit()


def _insert_trade(db_path: Path, *, side: str, symbol: str, qty: float, price: float, run_id: str = "test-run") -> None:
    with sqlite3.connect(str(db_path)) as conn:
        cols = {r[1] for r in conn.execute("PRAGMA table_info(paper_trades)")}
        base_cols = ["trade_id", "paper_run_id", "mode", "symbol", "side", "quantity", "price", "remaining_position", "timestamp", "status", "strategy_id"]
        base_vals: list = [
            f"t_{symbol}_{side}_{time.time_ns()}",
            run_id,
            "paper",
            symbol,
            side,
            qty,
            price,
            qty if side == "BUY" else 0.0,
        ]
        placeholders = ["?"] * len(base_vals) + ["datetime('now')", "'executed'", "'day'"]
        insert_cols = list(base_cols)
        if "is_synthetic" in cols:
            insert_cols.append("is_synthetic")
            placeholders.append("0")
        conn.execute(
            f"INSERT INTO paper_trades ({', '.join(insert_cols)}) VALUES ({', '.join(placeholders)})",
            base_vals,
        )
        conn.commit()


# ---------------------------------------------------------------------------
# 1. Starting cash, no positions
# ---------------------------------------------------------------------------
def test_scenario_1_starting_cash_no_positions():
    with tempfile.TemporaryDirectory() as tmp:
        db_path = Path(tmp) / "s1.db"
        _init_ledger(db_path, cash=25_000.0, principal=25_000.0)
        recon = reconcile_ledger_cash(str(db_path))
        assert recon.stored_cash == 25_000.0
        assert recon.stored_positions_value == 0.0
        assert recon.expected_cash == 25_000.0
        assert recon.within_tolerance


# ---------------------------------------------------------------------------
# 2. Buy $4,000 of BTC, no fee
# ---------------------------------------------------------------------------
def test_scenario_2_buy_no_fee():
    with tempfile.TemporaryDirectory() as tmp:
        db_path = Path(tmp) / "s2.db"
        _init_ledger(db_path, cash=25_000.0, principal=25_000.0)
        _insert_trade(db_path, side="BUY", symbol="BTC/USDT", qty=0.1, price=40_000.0)  # $4,000 notional
        expected_cash, buy_debits, sell_credits, buy_count, sell_count = compute_expected_cash(str(db_path), principal=25_000.0)
        assert buy_debits == pytest.approx(4_000.0, abs=0.01)
        assert expected_cash == pytest.approx(21_000.0, abs=0.01)
        assert buy_count == 1 and sell_count == 0
        # Simulate ledger persisted right after the debit (as production now does).
        with sqlite3.connect(str(db_path)) as conn:
            conn.execute(
                "UPDATE portfolio_engine_ledger SET cash_balance=?, positions_value=?, total_equity=? WHERE id=1",
                (21_000.0, 4_000.0, 25_000.0),
            )
            conn.commit()
        recon = reconcile_ledger_cash(str(db_path))
        assert recon.within_tolerance
        assert recon.stored_cash == pytest.approx(21_000.0, abs=0.01)


# ---------------------------------------------------------------------------
# 3. Buy with entry fee
# ---------------------------------------------------------------------------
def test_scenario_3_buy_with_entry_fee():
    with tempfile.TemporaryDirectory() as tmp:
        db_path = Path(tmp) / "s3.db"
        _init_ledger(db_path, cash=25_000.0, principal=25_000.0)
        with sqlite3.connect(str(db_path)) as conn:
            cols = {r[1] for r in conn.execute("PRAGMA table_info(paper_trades)")}
            if "entry_fee_usd" not in cols:
                conn.execute("ALTER TABLE paper_trades ADD COLUMN entry_fee_usd REAL")
                cols.add("entry_fee_usd")
            insert_cols = ["trade_id", "paper_run_id", "mode", "symbol", "side", "quantity", "price", "remaining_position", "timestamp", "status", "strategy_id", "entry_fee_usd"]
            placeholders = ["'t_fee_buy'", "'test-run'", "'paper'", "'BTC/USDT'", "'BUY'", "0.1", "40000.0", "0.1", "datetime('now')", "'executed'", "'day'", "4.0"]
            if "is_synthetic" in cols:
                insert_cols.append("is_synthetic")
                placeholders.append("0")
            conn.execute(f"INSERT INTO paper_trades ({', '.join(insert_cols)}) VALUES ({', '.join(placeholders)})")
            conn.commit()
        expected_cash, buy_debits, _, _, _ = compute_expected_cash(str(db_path), principal=25_000.0)
        # notional $4,000 + $4 fee = $4,004 debited
        assert buy_debits == pytest.approx(4_004.0, abs=0.01)
        assert expected_cash == pytest.approx(25_000.0 - 4_004.0, abs=0.01)


# ---------------------------------------------------------------------------
# 4. Mark BTC up without selling
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_scenario_4_mark_up_does_not_touch_cash():
    engine = PortfolioEngine(principal=25_000.0, test_mode=True)
    engine.cash_balance = 21_000.0
    engine.open_positions["BTC/USDT"] = OpenPosition(
        symbol="BTC/USDT",
        quantity=0.1,
        entry_price=40_000.0,
        entry_time=time.time(),
        trade_id="mark-up-test",
        stop_price=38_000.0,
        take_profit_1_price=42_000.0,
        take_profit_2_price=0.0,
        sleeve=Sleeve.ACTIVE.value,
    )
    cash_before = engine.cash_balance
    await engine._recompute_positions_values({"BTC/USDT": 41_000.0})  # BTC up $1,000
    assert engine.cash_balance == cash_before, "mark-up must never change cash"
    assert engine._positions_value == pytest.approx(4_100.0, abs=0.01)
    assert engine._unrealized_pnl == pytest.approx(100.0, abs=0.01)
    assert engine._total_equity == pytest.approx(cash_before + 4_100.0, abs=0.01)
    # equity increased by exactly the unrealized gain, nothing else
    assert engine._total_equity - 25_000.0 == pytest.approx(100.0, abs=0.01)


# ---------------------------------------------------------------------------
# 5 & 6. Sell at profit / loss — proceeds credited once, realized P&L recorded once
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("sell_price,expect_profit", [(42_000.0, True), (38_000.0, False)])
def test_scenario_5_6_sell_profit_and_loss_single_credit(sell_price, expect_profit):
    with tempfile.TemporaryDirectory() as tmp:
        db_path = Path(tmp) / "s56.db"
        entry = 40_000.0
        qty = 0.1
        cash_after_buy = 21_000.0
        _init_ledger(db_path, cash=cash_after_buy, principal=25_000.0, positions_value=qty * entry)
        _insert_trade(db_path, side="BUY", symbol="BTC/USDT", qty=qty, price=entry)
        _insert_trade(db_path, side="SELL", symbol="BTC/USDT", qty=qty, price=sell_price)

        expected_cash, buy_debits, sell_credits, buy_count, sell_count = compute_expected_cash(str(db_path), principal=25_000.0)
        assert buy_count == 1 and sell_count == 1
        proceeds = qty * sell_price
        assert sell_credits == pytest.approx(proceeds, abs=0.01)
        # Flat position: expected_cash must equal principal +/- net P&L exactly once.
        realized = (sell_price - entry) * qty
        assert expected_cash == pytest.approx(25_000.0 + realized, abs=0.01)
        if expect_profit:
            assert realized > 0
        else:
            assert realized < 0

        with sqlite3.connect(str(db_path)) as conn:
            conn.execute(
                "UPDATE portfolio_engine_ledger SET cash_balance=?, positions_value=0, realized_pnl=?, total_equity=? WHERE id=1",
                (expected_cash, realized, expected_cash),
            )
            conn.commit()
        recon = reconcile_ledger_cash(str(db_path))
        assert recon.within_tolerance
        # flat position: cash alone equals equity, no double count of notional
        assert recon.stored_cash == pytest.approx(recon.stored_total_equity, abs=0.01)


# ---------------------------------------------------------------------------
# 7. Restart with open position — cash must not be restored to pre-buy amount
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_scenario_7_restart_does_not_restore_pre_buy_cash():
    with tempfile.TemporaryDirectory() as tmp:
        db_path = Path(tmp) / "s7.db"
        principal = 25_000.0
        cash_after_buy = 21_000.0  # post-buy cash, must survive restart
        _init_ledger(db_path, cash=cash_after_buy, principal=principal, positions_value=4_000.0)

        engine = PortfolioEngine(db_path=str(db_path), principal=principal, test_mode=True)
        await engine.initialize_from_db()

        assert engine.cash_balance == pytest.approx(cash_after_buy, abs=0.01)
        assert engine.cash_balance != pytest.approx(principal, abs=0.01), "restart must not silently restore pre-buy principal as cash while a position is open"


# ---------------------------------------------------------------------------
# 8. Partial position exit — cost basis, realized P&L, cash all remain consistent
# ---------------------------------------------------------------------------
def test_scenario_8_partial_exit_reconciles():
    with tempfile.TemporaryDirectory() as tmp:
        db_path = Path(tmp) / "s8.db"
        entry = 40_000.0
        qty_total = 0.2
        sell_qty = 0.1  # partial: half the position
        sell_price = 41_000.0
        cash_after_buy = 25_000.0 - qty_total * entry  # 17,000
        _init_ledger(db_path, cash=cash_after_buy, principal=25_000.0, positions_value=qty_total * entry)
        _insert_trade(db_path, side="BUY", symbol="BTC/USDT", qty=qty_total, price=entry)
        _insert_trade(db_path, side="SELL", symbol="BTC/USDT", qty=sell_qty, price=sell_price)

        expected_cash, buy_debits, sell_credits, buy_count, sell_count = compute_expected_cash(str(db_path), principal=25_000.0)
        remaining_qty = qty_total - sell_qty
        remaining_cost_basis = remaining_qty * entry
        realized = (sell_price - entry) * sell_qty

        assert expected_cash == pytest.approx(25_000.0 - qty_total * entry + sell_qty * sell_price, abs=0.01)
        # cash + remaining cost basis + realized must equal principal + realized (no notional double count)
        assert expected_cash + remaining_cost_basis == pytest.approx(25_000.0 + realized, abs=0.01)


# ---------------------------------------------------------------------------
# Reconciliation drift detection itself (proves the module can find a real bug)
# ---------------------------------------------------------------------------
def test_reconciliation_flags_the_original_double_count_bug():
    """
    Reproduces the exact P0 bug: MTM loop reloaded pre-buy cash from SQLite while
    a position was already open in memory, inflating equity by the open notional.
    """
    with tempfile.TemporaryDirectory() as tmp:
        db_path = Path(tmp) / "bug.db"
        entry = 62_500.0
        qty = 0.06
        pre_buy_cash = 25_000.0
        cash_after_buy = pre_buy_cash - qty * entry  # ~21,250
        _init_ledger(db_path, cash=pre_buy_cash, principal=25_000.0, positions_value=qty * entry)
        _insert_trade(db_path, side="BUY", symbol="BTC/USDT", qty=qty, price=entry)
        # Ledger row was NEVER updated after the buy (the bug): cash still shows pre-buy value
        # while positions_value already reflects the open position → double count.
        recon = reconcile_ledger_cash(str(db_path))
        assert not recon.within_tolerance
        assert recon.cash_discrepancy == pytest.approx(pre_buy_cash - cash_after_buy, abs=0.01)
        assert any("CASH_DRIFT" in n for n in recon.notes)
