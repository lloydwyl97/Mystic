"""Writer-side SQLite lock storm regressions."""

from __future__ import annotations

import asyncio
import contextlib
import inspect
import sqlite3
import time
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from backend.services.circuit_breaker_service import TradingCircuitBreaker
from backend.services.portfolio_engine_integration import PortfolioEngineIntegration
from backend.utils import sqlite_runtime


def test_connect_rw_with_block_closes_underlying_connection(tmp_path: Path) -> None:
    """sqlite3.Connection.__exit__ only commits/rolls back — it never closes the fd.

    Every `with connect_rw(...) as conn:` call site (37 across the codebase) relies
    on the connection actually closing on exit. Before the fix, each such call leaked
    one open connection/fd; under sustained retry storms this compounded into
    cascading "database is locked" failures. connect_rw must return a wrapper whose
    `with` exit both preserves commit/rollback semantics AND closes the connection.
    """
    db = tmp_path / "autoclose.db"
    with sqlite3.connect(db) as c:
        c.execute("CREATE TABLE t(x INTEGER)")
        c.commit()

    with sqlite_runtime.connect_rw(db) as conn:
        conn.execute("INSERT INTO t (x) VALUES (1)")
        # sanity: proxy still behaves like a normal connection mid-block
        assert conn.execute("SELECT x FROM t").fetchone() == (1,)

    # After the `with` block exits, the underlying sqlite3 connection must be closed
    # (a closed connection raises ProgrammingError on any further use).
    with pytest.raises(sqlite3.ProgrammingError):
        conn.execute("SELECT 1")

    # Commit-on-success semantics must be preserved: the insert above should be durable.
    with sqlite3.connect(db) as c2:
        assert c2.execute("SELECT COUNT(*) FROM t").fetchone()[0] == 1


def test_connect_rw_with_block_closes_on_exception(tmp_path: Path) -> None:
    """Connection must close (and roll back) even when the `with` body raises."""
    db = tmp_path / "autoclose_err.db"
    with sqlite3.connect(db) as c:
        c.execute("CREATE TABLE t(x INTEGER)")
        c.commit()

    captured_conn = None
    with contextlib.suppress(RuntimeError), sqlite_runtime.connect_rw(db) as conn:
        captured_conn = conn
        conn.execute("INSERT INTO t (x) VALUES (1)")
        raise RuntimeError("boom")

    assert captured_conn is not None
    with pytest.raises(sqlite3.ProgrammingError):
        captured_conn.execute("SELECT 1")

    # Rolled back: the insert must not be durable since the block raised.
    with sqlite3.connect(db) as c2:
        assert c2.execute("SELECT COUNT(*) FROM t").fetchone()[0] == 0


def test_connect_ro_with_block_closes_underlying_connection(tmp_path: Path) -> None:
    """connect_ro hits every GET/status API request — leaks here compound fast."""
    db = tmp_path / "ro_autoclose.db"
    with sqlite3.connect(db) as c:
        c.execute("CREATE TABLE t(x INTEGER)")
        c.execute("INSERT INTO t (x) VALUES (1)")
        c.commit()

    with sqlite_runtime.connect_ro(db, timeout_sec=1.0) as conn:
        assert conn.execute("SELECT x FROM t").fetchone() == (1,)

    with pytest.raises(sqlite3.ProgrammingError):
        conn.execute("SELECT 1")


def test_connect_managed_closes_and_preserves_commit_semantics(tmp_path: Path) -> None:
    """connect_managed replaces bare `with sqlite3.connect(...) as conn:` call sites
    (21 in portfolio_engine.py) that bypassed connect_rw/connect_ro entirely and thus
    still leaked a connection/fd per call even after the connect_rw/connect_ro fix.
    """
    db = tmp_path / "managed.db"
    with sqlite3.connect(db) as c:
        c.execute("CREATE TABLE t(x INTEGER)")
        c.commit()

    with sqlite_runtime.connect_managed(db) as conn:
        conn.execute("INSERT INTO t (x) VALUES (1)")
        assert conn.execute("SELECT x FROM t").fetchone() == (1,)

    with pytest.raises(sqlite3.ProgrammingError):
        conn.execute("SELECT 1")

    with sqlite3.connect(db) as c2:
        assert c2.execute("SELECT COUNT(*) FROM t").fetchone()[0] == 1


def test_connect_managed_closes_and_rolls_back_on_exception(tmp_path: Path) -> None:
    db = tmp_path / "managed_err.db"
    with sqlite3.connect(db) as c:
        c.execute("CREATE TABLE t(x INTEGER)")
        c.commit()

    captured_conn = None
    with contextlib.suppress(RuntimeError), sqlite_runtime.connect_managed(db, timeout=5) as conn:
        captured_conn = conn
        conn.execute("INSERT INTO t (x) VALUES (1)")
        raise RuntimeError("boom")

    assert captured_conn is not None
    with pytest.raises(sqlite3.ProgrammingError):
        captured_conn.execute("SELECT 1")

    with sqlite3.connect(db) as c2:
        assert c2.execute("SELECT COUNT(*) FROM t").fetchone()[0] == 0


def test_writer_helper_sets_busy_timeout_and_retries(tmp_path: Path) -> None:
    db = tmp_path / "w.db"
    with sqlite3.connect(db) as c:
        c.execute("CREATE TABLE t(x INTEGER)")
        c.commit()
    conn = sqlite_runtime.connect_rw(db)
    try:
        row = conn.execute("PRAGMA busy_timeout").fetchone()
        assert row is not None
        assert int(row[0]) >= 1000
    finally:
        conn.close()
    assert sqlite_runtime.is_locked_error(sqlite3.OperationalError("database is locked"))
    calls = {"n": 0}

    def flaky():
        calls["n"] += 1
        if calls["n"] < 3:
            raise sqlite3.OperationalError("database is locked")
        return 42

    assert sqlite_runtime.run_locked_retry(flaky, max_attempts=5) == 42
    assert calls["n"] == 3


@pytest.mark.asyncio
async def test_circuit_breaker_skips_unchanged_persist(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(TradingCircuitBreaker, "_load_circuit_state", lambda _self: None)
    cb = TradingCircuitBreaker()
    cb.daily_loss_freeze_active = False
    cb.equity_circuit_breaker_active = False
    cb.account_failsafe_active = False
    cb.session_high_equity = 10000.0
    cb.last_daily_reset = "2026-08-03"
    writes: list[str] = []

    async def fake_set(key, _obj):
        writes.append(key)

    monkeypatch.setattr("backend.services.circuit_breaker_service.set_state", fake_set)
    monkeypatch.setenv("CIRCUIT_BREAKER_PERSIST_MIN_INTERVAL_SEC", "60")
    await cb.persist_circuit_state_async()
    first = len(writes)
    assert first >= 1
    await cb.persist_circuit_state_async()
    assert len(writes) == first  # unchanged + within throttle


@pytest.mark.asyncio
async def test_circuit_breaker_writes_on_state_change(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(TradingCircuitBreaker, "_load_circuit_state", lambda _self: None)
    cb = TradingCircuitBreaker()
    cb.daily_loss_freeze_active = False
    cb.equity_circuit_breaker_active = False
    cb.account_failsafe_active = False
    cb.session_high_equity = 10000.0
    cb.last_daily_reset = "2026-08-03"
    writes: list[str] = []

    async def fake_set(key, _obj):
        writes.append(key)

    monkeypatch.setattr("backend.services.circuit_breaker_service.set_state", fake_set)
    monkeypatch.setenv("CIRCUIT_BREAKER_PERSIST_MIN_INTERVAL_SEC", "60")
    await cb.persist_circuit_state_async()
    n0 = len(writes)
    cb.daily_loss_freeze_active = True
    await cb.persist_circuit_state_async()
    assert len(writes) > n0


@pytest.mark.asyncio
async def test_exit_failure_skips_cooldown_on_sqlite_lock() -> None:
    integ = PortfolioEngineIntegration.__new__(PortfolioEngineIntegration)
    integ.exit_failure_count = {}
    integ.exit_cooldown_until = {}
    integ.exit_hard_paused = set()
    integ.redis_client = None
    await integ._handle_exit_failure("BTC/USDT", "database is locked")
    assert integ.exit_failure_count.get("BTC/USDT", 0) == 0
    assert "BTC/USDT" not in integ.exit_cooldown_until


@pytest.mark.asyncio
async def test_exit_monitor_locked_error_no_cooldown(monkeypatch: pytest.MonkeyPatch) -> None:
    """EXIT_MONITORING_ERROR path with sqlite lock must not cooldown all symbols."""
    source = inspect.getsource(PortfolioEngineIntegration._monitor_positions_once)
    assert "EXIT_MONITORING_SQLITE_BUSY" in source or "is_locked_error" in source


def test_signal_redis_before_db_persist() -> None:
    from backend.services.ai_signal_generator import RealTimeAISignalGenerator

    source = inspect.getsource(RealTimeAISignalGenerator._generate_signal_for_symbol)
    redis_idx = source.index("pipe.hmset(key")
    db_idx = source.index("_save_signal_to_database")
    assert redis_idx < db_idx
    assert "redis signal preserved" in source


def test_retention_not_in_bar_or_exit_hot_path() -> None:
    from backend.services import portfolio_engine_integration as pei

    bar_src = inspect.getsource(pei.PortfolioEngineIntegration)
    # Retention lives in dedicated loop method, not inline in bar/exit once handlers.
    mon = inspect.getsource(pei.PortfolioEngineIntegration._monitor_positions_once)
    assert "LARGE_TABLE_RETENTION" not in mon
    assert "run_large_table_retention" not in mon
    assert "maybe_run_large_table_retention" not in mon
    assert "_large_table_retention_loop" in bar_src


def test_mtm_persist_fetches_marks_outside_lock() -> None:
    from backend.services.portfolio_engine_integration import PortfolioEngineIntegration

    source = inspect.getsource(PortfolioEngineIntegration._ledger_mtm_persist_loop)
    fetch_idx = source.index("_fetch_mtm_prices_for_open_positions")
    lock_idx = source.index("_sqlite_writer_lock")
    persist_idx = source.index("_persist_ledger_to_sqlite")
    assert fetch_idx < lock_idx < persist_idx


def test_scalp_sell_log_after_commit() -> None:
    from backend.services.binance_scalp.paper_engine import BinanceScalpPaperEngine

    exec_src = inspect.getsource(BinanceScalpPaperEngine._execute_sell)
    tick_src = inspect.getsource(BinanceScalpPaperEngine.tick)
    assert "_pending_sell_log" in exec_src
    assert "SCALP_PAPER_SELL" in tick_src
    assert tick_src.index("conn.commit()") < tick_src.index("SCALP_PAPER_SELL")
