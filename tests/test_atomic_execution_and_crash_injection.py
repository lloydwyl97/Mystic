"""Atomic OPEN/CLOSE, orphan restore, SCALP money-DB split, crash injection."""

from __future__ import annotations

import asyncio
import sqlite3
import tempfile
import time
from pathlib import Path

import pytest

from backend.database_schema import initialize_paper_trading_schema
from backend.services.atomic_execution_book import (
    assert_cash_plus_marks_equals_equity,
    find_orphaned_day_buys,
    migrate_scalp_money_database,
    restore_orphaned_day_buys,
)
from backend.services.portfolio_engine import OpenPosition, PortfolioEngine, Sleeve
from backend.services.trade_learning_writer import (
    TradeLearningRecord,
    consume_setup_outcomes_for_ranking,
    record_trade_outcome,
)


def _init_engine(db_path: Path, *, cash: float = 10_000.0) -> PortfolioEngine:
    engine = PortfolioEngine(db_path=str(db_path), principal=cash, test_mode=True)
    engine._ensure_db_schema()
    initialize_paper_trading_schema(str(db_path))
    engine.cash_balance = cash
    engine._available_balance = cash
    engine._positions_value = 0.0
    engine._total_equity = cash
    engine._realized_pnl = 0.0
    engine._unrealized_pnl = 0.0
    asyncio.run(engine._persist_ledger_to_sqlite())
    return engine


def _position(*, symbol: str = "XRP/USDT", qty: float = 100.0, price: float = 1.0, trade_id: str = "t_open") -> OpenPosition:
    return OpenPosition(
        symbol=symbol,
        quantity=qty,
        entry_price=price,
        entry_time=time.time(),
        trade_id=trade_id,
        stop_price=price * 0.97,
        take_profit_1_price=price * 1.02,
        take_profit_2_price=price * 1.05,
        highest_price=price,
        lowest_price=price,
        atr_at_entry=0.01,
        entry_bar_timestamp=0,
        confidence_at_entry=0.5,
        entry_fee=0.1,
        sleeve=Sleeve.ACTIVE.value,
        original_position_cost=qty * price,
    )


def _trade_bind(trade_id: str, symbol: str, qty: float, price: float) -> tuple:
    ts = "2026-08-14T00:00:00+00:00"
    return (
        trade_id,
        "test-run",
        "paper",
        symbol,
        qty,
        price,
        qty,
        price * 0.97,
        price * 1.02,
        0.01,
        0,
        0.5,
        0.1,
        0.05,
        ts,
        "{}",
        "{}",
        Sleeve.ACTIVE.value,
        ts,
        None,
        "day",
        "{}",
    )


def _commit(engine: PortfolioEngine, pos: OpenPosition, *, cash: float, positions_value: float) -> None:
    engine._commit_atomic_day_open_sync(
        trade_bind=_trade_bind(pos.trade_id, pos.symbol, pos.quantity, pos.entry_price),
        position=pos,
        cash_balance=cash,
        positions_value=positions_value,
        realized_pnl=0.0,
        unrealized_pnl=0.0,
        total_equity=cash + positions_value,
        pre_ledger={"cash_balance": engine.cash_balance, "positions_value": 0.0, "total_equity": engine.cash_balance},
        fee=0.1,
        slippage_cost=0.05,
        quantity=pos.quantity,
        fill_price=pos.entry_price,
        symbol=pos.symbol,
        trade_id=pos.trade_id,
        entry_reason="test",
        sleeve=Sleeve.ACTIVE.value,
    )


def test_crash_before_buy_txn_leaves_no_orphan():
    with tempfile.TemporaryDirectory() as tmp:
        db = Path(tmp) / "day.db"
        engine = _init_engine(db)
        engine._inject_crash_before_buy_txn = True
        pos = _position(trade_id="crash_before")
        with pytest.raises(RuntimeError, match="INJECTED_CRASH_BEFORE_BUY_TXN"):
            _commit(engine, pos, cash=8500.0, positions_value=1500.0)
        assert find_orphaned_day_buys(str(db)) == []
        proof = assert_cash_plus_marks_equals_equity(str(db))
        assert proof["ok"] is True
        assert proof["cash"] == pytest.approx(10_000.0, abs=0.05)


def test_crash_during_buy_txn_rolls_back():
    with tempfile.TemporaryDirectory() as tmp:
        db = Path(tmp) / "day.db"
        engine = _init_engine(db)
        engine._inject_crash_during_buy_txn = True
        pos = _position(trade_id="crash_during")
        with pytest.raises(RuntimeError, match="INJECTED_CRASH_DURING_BUY_TXN"):
            _commit(engine, pos, cash=8500.0, positions_value=1500.0)
        assert find_orphaned_day_buys(str(db)) == []
        with sqlite3.connect(str(db)) as conn:
            n = conn.execute("SELECT COUNT(*) FROM paper_trades").fetchone()[0]
            p = conn.execute("SELECT COUNT(*) FROM portfolio_engine_positions").fetchone()[0]
        assert n == 0
        assert p == 0
        proof = assert_cash_plus_marks_equals_equity(str(db))
        assert proof["ok"] is True


def test_crash_after_commit_keeps_identity():
    with tempfile.TemporaryDirectory() as tmp:
        db = Path(tmp) / "day.db"
        engine = _init_engine(db)
        engine._inject_crash_after_buy_commit = True
        pos = _position(qty=1476.60871, price=1.0032, trade_id="crash_after")
        with pytest.raises(RuntimeError, match="INJECTED_CRASH_AFTER_BUY_COMMIT"):
            _commit(engine, pos, cash=8518.72, positions_value=1481.28)
        orphans = find_orphaned_day_buys(str(db))
        assert orphans == []
        proof = assert_cash_plus_marks_equals_equity(str(db))
        assert proof["ok"] is True
        assert proof["positions_value"] == pytest.approx(1481.28, abs=0.05)


def test_crash_during_mtm_after_commit_reload_restores_position():
    with tempfile.TemporaryDirectory() as tmp:
        db = Path(tmp) / "day.db"
        engine = _init_engine(db)
        pos = _position(symbol="XRP/USDT", qty=1476.60871, price=1.0032, trade_id="mtm_crash")
        _commit(engine, pos, cash=8518.72, positions_value=1481.28)
        engine._inject_crash_during_mtm = True
        with pytest.raises(RuntimeError, match="INJECTED_CRASH_DURING_MTM"):
            asyncio.run(engine._recompute_positions_values())
        engine2 = PortfolioEngine(db_path=str(db), principal=10_000.0, test_mode=True)
        engine2._ensure_db_schema()
        loaded = asyncio.run(engine2._load_positions_from_sqlite(allow_mutations=False))
        assert loaded >= 1
        assert any("XRP" in s for s in engine2.open_positions)
        proof = assert_cash_plus_marks_equals_equity(str(db))
        assert proof["ok"] is True


def test_orphan_restore_from_committed_buy_does_not_invent_cash():
    with tempfile.TemporaryDirectory() as tmp:
        db = Path(tmp) / "day.db"
        engine = _init_engine(db)
        with sqlite3.connect(str(db)) as conn:
            conn.execute(
                """
                INSERT INTO paper_trades (
                    trade_id, paper_run_id, mode, symbol, side, quantity, price,
                    remaining_position, timestamp, status
                ) VALUES ('orphan_980', 'test', 'paper', 'XRP/USDT', 'BUY', 1476.60871, 1.0032,
                          1476.60871, datetime('now'), 'executed')
                """
            )
            conn.execute(
                "UPDATE portfolio_engine_ledger SET cash_balance=8041.04, positions_value=0, total_equity=8041.04 WHERE id=1"
            )
            conn.commit()
        orphans = find_orphaned_day_buys(str(db))
        assert len(orphans) == 1
        restored = restore_orphaned_day_buys(str(db))
        assert len(restored) == 1
        proof = assert_cash_plus_marks_equals_equity(str(db))
        assert proof["ok"] is True
        assert proof["cash"] == pytest.approx(8041.04, abs=0.05)
        assert proof["positions_value"] > 1400
        assert find_orphaned_day_buys(str(db)) == []


def test_open_positions_swap_not_in_place_clear():
    engine = PortfolioEngine(principal=10_000.0, test_mode=True)
    engine.open_positions = {"ETH/USDT": _position(symbol="ETH/USDT", trade_id="a")}
    snapshot = list(engine.open_positions)
    rebuilt = {"XRP/USDT": _position(symbol="XRP/USDT", trade_id="b")}
    engine.open_positions = rebuilt
    assert snapshot == ["ETH/USDT"]
    assert list(engine.open_positions) == ["XRP/USDT"]


def test_status_exposes_failsafe_when_not_trading_paused():
    with tempfile.TemporaryDirectory() as tmp:
        db = Path(tmp) / "status.db"
        engine = _init_engine(db)
        engine._trading_paused = False
        from backend.services.portfolio_engine import KillSwitchMode

        engine._kill_switch_mode = KillSwitchMode.PAUSE_BUYS
        engine._kill_switch_reason = "CB:ACCOUNT_FAILSAFE equity=$8040"
        cap = engine.get_trading_capability_status()
        assert cap["failsafe_active"] is True
        assert cap["day_entry_enabled"] is False
        assert cap["no_trade_reason"]
        assert "kill_switch" in cap["no_trade_reason"]


def test_scalp_money_db_migrate_copies_and_isolates():
    with tempfile.TemporaryDirectory() as tmp:
        day_db = Path(tmp) / "mystic_trading.db"
        scalp_db = Path(tmp) / "mystic_scalp.db"
        from backend.services.binance_scalp.schema import init_scalp_schema

        init_scalp_schema(str(day_db), principal=1000.0)
        with sqlite3.connect(str(day_db)) as conn:
            conn.execute(
                """
                INSERT INTO scalp_paper_trades (trade_id, symbol, side, quantity, price, notional)
                VALUES ('s1', 'BTCUSDT', 'SELL', 0.001, 100000, 100)
                """
            )
            conn.commit()
        result = migrate_scalp_money_database(str(day_db), str(scalp_db))
        assert result["migrated"] is True
        with sqlite3.connect(str(scalp_db)) as conn:
            n = conn.execute("SELECT COUNT(*) FROM scalp_paper_trades").fetchone()[0]
        assert n == 1
        again = migrate_scalp_money_database(str(day_db), str(scalp_db))
        assert again["reason"] == "already_populated"


def test_learning_consume_affects_rank_delta():
    with tempfile.TemporaryDirectory() as tmp:
        db = Path(tmp) / "learn.db"
        from backend.config.trading_mode import TradingMode

        for i, pnl in enumerate([2.0, 1.5, 1.0, 0.8, 0.5, 0.4, 0.3, 0.2]):
            rec = TradeLearningRecord(
                symbol="BTCUSDT",
                entry_timestamp=1.0,
                exit_timestamp=2.0,
                entry_price=100.0,
                exit_price=101.0,
                quantity=1.0,
                fees_paid=0.1,
                slippage_cost=0.05,
                net_profit_usd=pnl,
                net_profit_pct=0.01,
                hold_seconds=60,
                close_reason="NET_PROFIT",
                extra={"setup_type_canonical": "RANGE_BOUNCE", "setup_type": "RANGE_BOUNCE"},
            )
            assert record_trade_outcome(rec, db_path=str(db), mode_override=TradingMode.PAPER) is True
        learned = consume_setup_outcomes_for_ranking(str(db), "RANGE_BOUNCE")
        assert learned["consumed"] is True
        assert learned["n"] >= 8
        assert learned["win_rate"] == 1.0
        assert learned["rank_delta"] > 0


def test_simultaneous_day_and_scalp_writes_do_not_lock():
    with tempfile.TemporaryDirectory() as tmp:
        day_db = Path(tmp) / "mystic_trading.db"
        scalp_db = Path(tmp) / "mystic_scalp.db"
        engine = _init_engine(day_db)
        from backend.services.binance_scalp.schema import init_scalp_schema

        init_scalp_schema(str(scalp_db), principal=1000.0)
        errors: list[str] = []

        def day_writer() -> None:
            try:
                pos = _position(trade_id=f"day_{time.time_ns()}")
                _commit(engine, pos, cash=9000.0, positions_value=1000.0)
            except Exception as exc:
                errors.append(f"day:{exc}")

        def scalp_writer() -> None:
            try:
                with sqlite3.connect(str(scalp_db), timeout=5) as conn:
                    conn.execute("BEGIN IMMEDIATE")
                    conn.execute(
                        """
                        INSERT INTO scalp_paper_trades (trade_id, symbol, side, quantity, price, notional)
                        VALUES (?, 'ETHUSDT', 'BUY', 0.01, 3000, 30)
                        """,
                        (f"scalp_{time.time_ns()}",),
                    )
                    conn.commit()
            except Exception as exc:
                errors.append(f"scalp:{exc}")

        import threading

        t1 = threading.Thread(target=day_writer)
        t2 = threading.Thread(target=scalp_writer)
        t1.start()
        t2.start()
        t1.join()
        t2.join()
        assert errors == []
        proof = assert_cash_plus_marks_equals_equity(str(day_db))
        assert proof["ok"] is True
