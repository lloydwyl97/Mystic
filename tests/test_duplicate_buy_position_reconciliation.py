"""
Regression tests for the duplicate-buy transaction-level reconciliation added
during the final pre-push audit.

Background (real incident): a per-symbol asyncio.Lock in execute_buy_fifo was
declared but never acquired, so two independent BUY decisions for the same
symbol ~60s apart could both execute. Both debited cash correctly (each BUY
is a real, individually-correct transaction), but `portfolio_engine_positions`
only ever stores one row per symbol (INSERT OR REPLACE), so the earlier lot's
quantity/cost silently vanished from position tracking while its cash had
already been spent — an "orphaned" BUY. This pattern was found to have
recurred multiple times across ETH, SOL, and XRP over several days before the
lock-acquisition fix (see test_buy_duplicate_race_prevention.py) went in.

These tests cover the *replay-and-repair* mechanism that detects and fixes
that historical damage without ever touching cash (cash was already correct)
and without fabricating a SELL or rewriting any historical trade row.
"""

from __future__ import annotations

import sqlite3
import tempfile
import time
from pathlib import Path

import pytest

from backend.services.portfolio_engine import PortfolioEngine
from backend.services.portfolio_ledger_reconciliation import (
    reconcile_ledger_cash,
    repair_orphaned_buy_lot_via_position_merge,
    replay_symbol_position_from_trades,
)


def _init_db(db_path: Path, *, cash: float = 25_000.0, principal: float = 25_000.0) -> None:
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
            ) VALUES (1, ?, ?, 0, 0, 0, ?, 'HEALTHY', 0, NULL, datetime('now'), 1)
            """,
            (principal, cash, cash),
        )
        conn.commit()


def _insert_trade(db_path: Path, *, side: str, symbol: str, qty: float, price: float, ts: str, trade_id: str) -> None:
    with sqlite3.connect(str(db_path)) as conn:
        cols = {r[1] for r in conn.execute("PRAGMA table_info(paper_trades)")}
        insert_cols = ["trade_id", "paper_run_id", "mode", "symbol", "side", "quantity", "price", "remaining_position", "timestamp", "status", "strategy_id"]
        vals: list = [trade_id, "test-run", "paper", symbol, side, qty, price, qty if side == "BUY" else 0.0, ts, "executed", "day"]
        placeholders = ["?"] * len(vals)
        if "is_synthetic" in cols:
            insert_cols.append("is_synthetic")
            placeholders.append("0")
        conn.execute(
            f"INSERT INTO paper_trades ({', '.join(insert_cols)}) VALUES ({', '.join(placeholders)})",
            vals,
        )
        conn.commit()


def _insert_position(db_path: Path, *, symbol: str, qty: float, entry_price: float, trade_id: str) -> None:
    with sqlite3.connect(str(db_path)) as conn:
        conn.execute(
            """
            INSERT OR REPLACE INTO portfolio_engine_positions (
                symbol, quantity, entry_price, entry_time, trade_id,
                stop_price, take_profit_1_price, take_profit_2_price,
                trailing_stop_price, tp1_hit, highest_price, atr_at_entry,
                entry_bar_timestamp, confidence_at_entry, entry_fee, sleeve,
                entry_strategy_id, repair_add_count, last_repair_add_ts,
                repair_add_trade_ids, average_entry_after_repair,
                original_position_cost, thesis_json, last_updated
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 0, ?, ?, ?, ?, 0, 'ACTIVE', 'day', 0, 0, '[]', 0, ?, '{}', datetime('now'))
            """,
            (
                symbol, qty, entry_price, time.time(), trade_id,
                entry_price * 0.98, entry_price * 1.02, 0.0, 0.0,
                entry_price, entry_price * 0.01, int(time.time()), 0.5,
                qty * entry_price,
            ),
        )
        conn.commit()


def test_orphaned_buy_lot_detected_and_classified():
    """Two BUYs for the same symbol, only the second one tracked -> classification."""
    with tempfile.TemporaryDirectory() as tmp:
        db_path = Path(tmp) / "orphan.db"
        _init_db(db_path)
        _insert_trade(db_path, side="BUY", symbol="ETH/USDT", qty=2.063, price=1817.85, ts="2026-07-11T16:42:08", trade_id="buy1")
        _insert_trade(db_path, side="BUY", symbol="ETH/USDT", qty=2.0628, price=1818.18, ts="2026-07-11T16:43:09", trade_id="buy2")
        # Only the second BUY survives in the position table (simulating the
        # INSERT OR REPLACE overwrite race).
        _insert_position(db_path, symbol="ETH/USDT", qty=2.0628, entry_price=1818.18, trade_id="buy2")

        replay = replay_symbol_position_from_trades(str(db_path), "ETHUSDT")
        assert replay.classification == "ORPHANED_BUY_LOT_NOT_TRACKED"
        assert replay.replayed_open_qty == pytest.approx(2.063 + 2.0628, abs=1e-6)
        assert "buy1" in replay.orphaned_buy_trade_ids


def test_orphaned_buy_lot_merge_preserves_cash_and_combines_quantity():
    """Repair must merge quantity/cost, touch zero cash, and not fabricate a SELL."""
    with tempfile.TemporaryDirectory() as tmp:
        db_path = Path(tmp) / "merge.db"
        _init_db(db_path)
        _insert_trade(db_path, side="BUY", symbol="ETH/USDT", qty=2.063, price=1817.85, ts="2026-07-11T16:42:08", trade_id="buy1")
        _insert_trade(db_path, side="BUY", symbol="ETH/USDT", qty=2.0628, price=1818.18, ts="2026-07-11T16:43:09", trade_id="buy2")
        _insert_position(db_path, symbol="ETH/USDT", qty=2.0628, entry_price=1818.18, trade_id="buy2")

        with sqlite3.connect(str(db_path)) as conn:
            cash_before = conn.execute("SELECT cash_balance FROM portfolio_engine_ledger WHERE id=1").fetchone()[0]

        result = repair_orphaned_buy_lot_via_position_merge(str(db_path), "ETHUSDT", dry_run=False)
        assert result["applied"] is True
        assert result["cash_adjustment"] == 0.0
        assert result["merged_qty"] == pytest.approx(2.063 + 2.0628, abs=1e-6)

        with sqlite3.connect(str(db_path)) as conn:
            cash_after = conn.execute("SELECT cash_balance FROM portfolio_engine_ledger WHERE id=1").fetchone()[0]
            pos = conn.execute("SELECT quantity, repair_add_count FROM portfolio_engine_positions WHERE symbol='ETH/USDT'").fetchone()
            trades = conn.execute("SELECT COUNT(*) FROM paper_trades WHERE symbol='ETH/USDT'").fetchone()[0]
            sells = conn.execute("SELECT COUNT(*) FROM paper_trades WHERE symbol='ETH/USDT' AND side='SELL'").fetchone()[0]

        assert cash_after == cash_before, "cash must be untouched — both buys already correctly debited it"
        assert pos[0] == pytest.approx(2.063 + 2.0628, abs=1e-6)
        assert pos[1] == 1  # repair_add_count
        assert trades == 2, "no historical trade rows may be added or deleted"
        assert sells == 0, "must never fabricate a SELL to close out the orphaned lot"

        replay_after = replay_symbol_position_from_trades(str(db_path), "ETHUSDT")
        assert replay_after.classification == "CONSISTENT"


def test_untracked_position_entirely_missing_recreates_row_not_cash():
    """If no position row survives at all, recreate it from trade evidence — never touch cash."""
    with tempfile.TemporaryDirectory() as tmp:
        db_path = Path(tmp) / "missing.db"
        _init_db(db_path)
        _insert_trade(db_path, side="BUY", symbol="XRP/USDT", qty=3270.6, price=1.1102, ts="2026-07-10T13:47:07", trade_id="xrp_buy1")

        replay = replay_symbol_position_from_trades(str(db_path), "XRPUSDT")
        assert replay.classification == "UNTRACKED_POSITION_ENTIRELY_MISSING"

        with sqlite3.connect(str(db_path)) as conn:
            cash_before = conn.execute("SELECT cash_balance FROM portfolio_engine_ledger WHERE id=1").fetchone()[0]

        result = repair_orphaned_buy_lot_via_position_merge(str(db_path), "XRPUSDT", dry_run=False)
        assert result["applied"] is True
        assert result["cash_adjustment"] == 0.0

        with sqlite3.connect(str(db_path)) as conn:
            cash_after = conn.execute("SELECT cash_balance FROM portfolio_engine_ledger WHERE id=1").fetchone()[0]
            pos = conn.execute("SELECT quantity FROM portfolio_engine_positions WHERE symbol='XRP/USDT'").fetchone()

        assert cash_after == cash_before
        assert pos is not None
        assert pos[0] == pytest.approx(3270.6, abs=1e-6)


def test_pooled_average_replay_not_confused_by_clean_round_trips():
    """
    A strict per-lot FIFO replay mis-happened to misattribute later clean
    round trips as orphaned whenever an earlier duplicate-buy incident put
    two lots in the pool and a SELL exactly matched the *second* lot's
    quantity. The pooled weighted-average model must not repeat that bug:
    many subsequent perfectly-matched buy/sell round trips on top of one
    real orphaned lot must all classify as clean, with only the original
    orphan detected.
    """
    with tempfile.TemporaryDirectory() as tmp:
        db_path = Path(tmp) / "pooled.db"
        _init_db(db_path)
        # Duplicate-buy incident: two buys 60s apart.
        _insert_trade(db_path, side="BUY", symbol="ETH/USDT", qty=2.0946, price=1790.86, ts="2026-07-11T02:25:07", trade_id="orig_buy")
        _insert_trade(db_path, side="BUY", symbol="ETH/USDT", qty=2.0941, price=1791.07, ts="2026-07-11T02:26:07", trade_id="dup_buy")
        # SELL exactly matches the *second* lot's quantity (not the first's) —
        # the real-world observed pattern.
        _insert_trade(db_path, side="SELL", symbol="ETH/USDT", qty=2.0941, price=1795.13, ts="2026-07-11T03:26:44", trade_id="sell1")
        # Several subsequent clean round trips of a similar size.
        for i in range(5):
            _insert_trade(db_path, side="BUY", symbol="ETH/USDT", qty=2.09, price=1795.0 + i, ts=f"2026-07-11T0{4+i}:00:00", trade_id=f"rt_buy_{i}")
            _insert_trade(db_path, side="SELL", symbol="ETH/USDT", qty=2.09, price=1796.0 + i, ts=f"2026-07-11T0{4+i}:30:00", trade_id=f"rt_sell_{i}")
        # Only the most recent lot is tracked in the position table.
        _insert_position(db_path, symbol="ETH/USDT", qty=2.0946, entry_price=1790.86, trade_id="orig_buy")

        replay = replay_symbol_position_from_trades(str(db_path), "ETHUSDT")
        # The pool should net back down to exactly the original orphaned lot's
        # quantity (2.0946) after all the clean round trips cancel out —
        # meaning the stored position (which also shows 2.0946) is actually
        # CONSISTENT, not orphaned, despite the earlier duplicate-buy incident.
        assert replay.replayed_open_qty == pytest.approx(2.0946, abs=1e-4)
        assert replay.classification == "CONSISTENT"


def test_reconcile_ledger_cash_does_not_false_alarm_on_truncated_history():
    """
    When paper_trades history has been rotated/pruned (oldest surviving row
    postdates the ledger's true starting point), the principal-anchored cash
    formula cannot be evaluated correctly. The tool must detect that shape
    (clean equity_discrepancy + large cash_discrepancy) and report it as an
    explained, non-alarming truncated_history condition rather than a false
    CASH_DRIFT.
    """
    with tempfile.TemporaryDirectory() as tmp:
        db_path = Path(tmp) / "truncated.db"
        # Simulate a long-running, established ledger (many visible trades)
        # whose cash reflects far more history than the (rotated) paper_trades
        # table currently shows.
        _init_db(db_path, cash=21286.39, principal=25_000.0)
        with sqlite3.connect(str(db_path)) as conn:
            conn.execute(
                "UPDATE portfolio_engine_ledger SET positions_value=?, total_equity=? WHERE id=1",
                (14823.12, 21286.39 + 14823.12),
            )
            conn.commit()
        for i in range(30):
            _insert_trade(db_path, side="BUY", symbol="BTC/USDT", qty=0.01, price=64000.0, ts="2026-07-03T00:00:00", trade_id=f"recent_buy_{i}")
            _insert_trade(db_path, side="SELL", symbol="BTC/USDT", qty=0.01, price=64010.0, ts="2026-07-03T00:05:00", trade_id=f"recent_sell_{i}")

        recon = reconcile_ledger_cash(str(db_path))
        assert recon.truncated_history is True
        assert recon.equity_discrepancy == 0.0
        assert recon.within_tolerance is True
        assert any("CASH_FORMULA_NOT_APPLICABLE" in n for n in recon.notes)
