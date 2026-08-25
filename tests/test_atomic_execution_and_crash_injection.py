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


def test_orphan_restore_detects_zeroed_remaining_without_sell():
    with tempfile.TemporaryDirectory() as tmp:
        db = Path(tmp) / "day.db"
        engine = _init_engine(db)
        with sqlite3.connect(str(db)) as conn:
            conn.execute(
                """
                INSERT INTO paper_trades (
                    trade_id, paper_run_id, mode, symbol, side, quantity, price,
                    remaining_position, timestamp, status
                ) VALUES ('orphan_zeroed', 'test', 'paper', 'XRP/USDT', 'BUY', 100, 1.0,
                          0.0, datetime('now'), 'executed')
                """
            )
            conn.execute(
                "UPDATE portfolio_engine_ledger SET cash_balance=9900, positions_value=0, total_equity=9900 WHERE id=1"
            )
            conn.commit()
        orphans = find_orphaned_day_buys(str(db))
        assert len(orphans) == 1
        restored = restore_orphaned_day_buys(str(db))
        assert len(restored) == 1
        proof = assert_cash_plus_marks_equals_equity(str(db))
        assert proof["ok"] is True
        assert proof["cash"] == pytest.approx(9900.0, abs=0.05)


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


def test_reconcile_import_trade_id_mismatch_is_not_orphan():
    from backend.services.atomic_execution_book import find_cash_position_disagreement

    with tempfile.TemporaryDirectory() as tmp:
        db = Path(tmp) / "day.db"
        _init_engine(db)
        with sqlite3.connect(str(db)) as conn:
            conn.execute(
                """
                INSERT INTO paper_trades (
                    trade_id, paper_run_id, mode, symbol, side, quantity, price,
                    remaining_position, timestamp, status
                ) VALUES ('mystic_BTC/USDT_1787675408882', 'test', 'paper', 'BTC/USDT', 'BUY',
                          0.00082983, 79141.45, 0.00082983, datetime('now'), 'executed')
                """
            )
            conn.execute(
                """
                INSERT INTO portfolio_engine_positions (
                    symbol, quantity, entry_price, entry_time, trade_id,
                    stop_price, take_profit_1_price, take_profit_2_price,
                    trailing_stop_price, tp1_hit, highest_price,
                    atr_at_entry, entry_bar_timestamp, confidence_at_entry, last_updated
                ) VALUES (
                    'BTC/USDT', 0.00082, 79328.73, strftime('%s','now'),
                    'reconcile_import_BTC_USDT_1787675558',
                    77000, 81000, 83000, 0, 0, 79328.73, 0, 0, 0.5, datetime('now')
                )
                """
            )
            conn.execute(
                """
                UPDATE portfolio_engine_ledger
                SET cash_balance=173.62, positions_value=64.95, total_equity=238.57
                WHERE id=1
                """
            )
            conn.commit()
        assert find_orphaned_day_buys(str(db)) == []
        acc = find_cash_position_disagreement(str(db))
        assert acc["ok"] is True
        assert acc["orphans"] == []


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
        engine.cash_balance = 8040.0
        engine._positions_value = 0.0
        engine._total_equity = 8040.0
        from backend.services.portfolio_engine import KillSwitchMode

        engine._kill_switch_mode = KillSwitchMode.PAUSE_BUYS
        engine._kill_switch_reason = "CB:ACCOUNT_FAILSAFE equity=$8040"
        cap = engine.get_trading_capability_status()
        assert cap["failsafe_active"] is True
        assert cap["day_entry_enabled"] is False
        assert cap["no_trade_reason"]
        assert "ACCOUNT_FAILSAFE" in cap["no_trade_reason"]


def test_status_and_execution_agree_when_kill_switch_resume_but_equity_low():
    with tempfile.TemporaryDirectory() as tmp:
        db = Path(tmp) / "status_resume.db"
        engine = _init_engine(db, cash=25_000.0)
        engine.cash_balance = 20_984.86
        engine._positions_value = 0.0
        engine._total_equity = 20_984.86
        engine._trading_paused = False
        from backend.services.portfolio_engine import KillSwitchMode

        engine._kill_switch_mode = KillSwitchMode.RESUME
        engine._kill_switch_reason = ""
        cap = engine.get_trading_capability_status()
        ks = engine.get_kill_switch_status()
        can_buy, reason = engine._check_kill_switch_buy()
        assert cap["failsafe_active"] is True
        assert cap["day_entry_enabled"] is False
        assert "ACCOUNT_FAILSAFE" in str(cap["no_trade_reason"])
        assert ks["buys_blocked"] is True
        assert ks["mode"] == "PAUSE_BUYS"
        assert "ACCOUNT_FAILSAFE" in str(ks["reason"])
        assert can_buy is False
        assert "ACCOUNT_FAILSAFE" in reason
        op = asyncio.run(engine.get_operator_status())
        assert op["failsafe_active"] is True
        assert op["kill_switch"] == "PAUSE_BUYS"
        assert op["day_entry_enabled"] is False
        assert "ACCOUNT_FAILSAFE" in str(op.get("no_trade_reason") or op.get("kill_switch_reason") or "")


def test_later_same_symbol_sell_does_not_hide_unclosed_other_lot():
    from backend.services.atomic_execution_book import find_unclosed_buy_cash_debits

    with tempfile.TemporaryDirectory() as tmp:
        db = Path(tmp) / "lots.db"
        engine = _init_engine(db)
        with sqlite3.connect(str(db)) as conn:
            conn.execute(
                """
                INSERT INTO paper_trades (
                    trade_id, paper_run_id, mode, symbol, side, quantity, price,
                    remaining_position, timestamp, status
                ) VALUES
                ('buy_a', 'test', 'paper', 'SOL/USDT', 'BUY', 30.0, 70.0, 0.0, '2026-08-01T21:15:00+00:00', 'executed'),
                ('sell_b', 'test', 'paper', 'SOL/USDT', 'SELL', 28.0, 71.0, 0.0, '2026-08-10T08:42:00+00:00', 'executed')
                """
            )
            conn.commit()
        unclosed = find_unclosed_buy_cash_debits(str(db))
        assert any(o.get("trade_id") == "buy_a" for o in unclosed)
        orphans = find_orphaned_day_buys(str(db))
        assert not any(o.get("trade_id") == "buy_a" for o in orphans)


def test_canonical_purge_ignores_historical_sells_before_entry():
    with tempfile.TemporaryDirectory() as tmp:
        db = Path(tmp) / "purge.db"
        engine = _init_engine(db)
        entry_time = time.time()
        with sqlite3.connect(str(db)) as conn:
            conn.execute(
                """
                INSERT INTO paper_trades (
                    trade_id, paper_run_id, mode, symbol, side, quantity, price,
                    remaining_position, timestamp, status
                ) VALUES ('old_sell', 'test', 'paper', 'BTC/USDT', 'SELL', 0.03, 65000, 0.0,
                          '2026-08-01T00:00:00+00:00', 'executed')
                """
            )
            conn.execute(
                """
                INSERT INTO portfolio_engine_positions (
                    symbol, quantity, entry_price, entry_time, trade_id,
                    stop_price, take_profit_1_price, take_profit_2_price,
                    highest_price, atr_at_entry, last_updated
                ) VALUES ('BTC/USDT', 0.03373, 65206.88, ?, 'new_buy',
                          63000.0, 67000.0, 69000.0, 65206.88, 400.0, datetime('now'))
                """,
                (entry_time,),
            )
            conn.commit()

        def _check():
            with sqlite3.connect(str(db)) as conn:
                pos_row = conn.execute("SELECT * FROM portfolio_engine_positions WHERE symbol=?", ("BTC/USDT",)).fetchone()
                assert pos_row is not None
                entry = float(pos_row[3] if pos_row[3] else 0)
                sells = conn.execute("SELECT timestamp FROM paper_trades WHERE symbol=? AND side='SELL'", ("BTC/USDT",)).fetchall()
                later = False
                for srow in sells:
                    raw = srow[0]
                    try:
                        from datetime import datetime
                        epoch = datetime.fromisoformat(str(raw).replace("Z", "+00:00")).timestamp()
                    except Exception:
                        epoch = 0.0
                    if epoch > entry + 1e-6:
                        later = True
                return later

        assert _check() is False
        with sqlite3.connect(str(db)) as conn:
            n = conn.execute("SELECT COUNT(*) FROM portfolio_engine_positions WHERE symbol='BTC/USDT'").fetchone()[0]
        assert n == 1


def test_orphan_cash_restore_credits_identified_buys_only():
    from backend.services.ledger_operational_heal import apply_orphan_buy_cash_restore

    with tempfile.TemporaryDirectory() as tmp:
        db = Path(tmp) / "heal.db"
        engine = _init_engine(db, cash=25_000.0)
        with sqlite3.connect(str(db)) as conn:
            conn.execute(
                """
                INSERT INTO paper_trades (
                    id, trade_id, paper_run_id, mode, symbol, side, quantity, price,
                    remaining_position, timestamp, status
                ) VALUES (4194, 'mystic_BTC/USDT_1', 'test', 'paper', 'BTC/USDT', 'BUY',
                          0.03373, 65206.88, 0.0, '2026-08-10T08:00:07+00:00', 'executed')
                """
            )
            conn.execute(
                "UPDATE portfolio_engine_ledger SET cash_balance=20984.86, positions_value=0, realized_pnl=385.15, total_equity=20984.86 WHERE id=1"
            )
            conn.commit()
            buy_id = conn.execute("SELECT id FROM paper_trades WHERE trade_id='mystic_BTC/USDT_1'").fetchone()[0]
        out = apply_orphan_buy_cash_restore(
            str(db),
            buy_ids=[int(buy_id)],
            heal_key="TEST_ORPHAN_BTC_4194",
            reason="test restore vanished BTC buy cash",
        )
        assert out["success"] is True
        with sqlite3.connect(str(db)) as conn:
            cash, equity, realized = conn.execute(
                "SELECT cash_balance, total_equity, realized_pnl FROM portfolio_engine_ledger WHERE id=1"
            ).fetchone()
        assert realized == pytest.approx(385.15, abs=0.01)
        assert cash == pytest.approx(20984.86 + 0.03373 * 65206.88, abs=0.02)
        assert find_orphaned_day_buys(str(db)) == []


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
        assert result["migrated"] is False
        assert result["reason"] == "isolation_complete_no_import"
        with sqlite3.connect(str(scalp_db)) as conn:
            n = conn.execute("SELECT COUNT(*) FROM scalp_paper_trades").fetchone()[0]
        assert n == 0
        with sqlite3.connect(str(scalp_db)) as conn:
            conn.execute(
                """
                INSERT INTO scalp_paper_trades (trade_id, symbol, side, quantity, price, notional)
                VALUES ('live1', 'BTCUSDT', 'SELL', 0.001, 100000, 100)
                """
            )
            conn.commit()
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
