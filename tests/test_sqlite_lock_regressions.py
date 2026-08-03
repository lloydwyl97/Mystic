"""Regressions: portfolio GET routes must not write/reconcile or hang on locks."""

from __future__ import annotations

import asyncio
import inspect
import sqlite3
import time
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import HTTPException

from backend.endpoints import portfolio_engine_endpoints
from backend.services.portfolio_engine import PortfolioEngine
from backend.utils.sqlite_runtime import connect_ro, is_locked_error


def test_scoreboard_today_route_source_has_no_update_scoreboard() -> None:
    source = inspect.getsource(portfolio_engine_endpoints.get_scoreboard_today)
    assert "await engine.update_scoreboard" not in source
    assert "await engine.get_scoreboard_today()" in source


def test_status_route_loads_without_mutations() -> None:
    source = inspect.getsource(portfolio_engine_endpoints.get_portfolio_status)
    assert "allow_mutations=False" in source
    assert "BEGIN IMMEDIATE" not in source


@pytest.mark.asyncio
async def test_get_scoreboard_does_not_call_update_or_reconcile(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    db_path = tmp_path / "score.db"
    engine = PortfolioEngine(db_path=str(db_path), principal=10_000.0, test_mode=True)
    engine._ensure_db_schema()

    update_spy = AsyncMock()
    fifo_spy = MagicMock()
    engine.update_scoreboard = update_spy
    engine._sync_paper_fifo_remaining_to_engine_positions_sync = fifo_spy
    engine.get_scoreboard_today = AsyncMock(
        return_value={"date": "2026-08-03", "status": "NO_DATA", "trades": 0}
    )

    monkeypatch.setattr(portfolio_engine_endpoints, "get_portfolio_engine", lambda: engine)

    result = await portfolio_engine_endpoints.get_scoreboard_today()
    assert result["success"] is True
    update_spy.assert_not_awaited()
    fifo_spy.assert_not_called()
    engine.get_scoreboard_today.assert_awaited_once()


@pytest.mark.asyncio
async def test_get_status_does_not_mutate_db(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    db_path = tmp_path / "status.db"
    engine = PortfolioEngine(db_path=str(db_path), principal=10_000.0, test_mode=True)
    engine._ensure_db_schema()
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            """
            UPDATE portfolio_engine_ledger
            SET cash_balance = 10000.0, positions_value = 0.0, realized_pnl = 0.0,
                unrealized_pnl = 0.0, total_equity = 10000.0, version = 1
            WHERE id = 1
            """
        )
        conn.commit()
        before = {
            "ledger": conn.execute("SELECT cash_balance, version FROM portfolio_engine_ledger WHERE id=1").fetchone(),
            "positions": conn.execute("SELECT COUNT(*) FROM portfolio_engine_positions").fetchone()[0],
            "paper": conn.execute("SELECT COUNT(*) FROM paper_trades").fetchone()[0],
            "scoreboard": conn.execute("SELECT COUNT(*) FROM portfolio_engine_scoreboard_daily").fetchone()[0],
            "audit": conn.execute("SELECT COUNT(*) FROM portfolio_engine_audit").fetchone()[0],
        }

    fifo_spy = MagicMock()
    engine._sync_paper_fifo_remaining_to_engine_positions_sync = fifo_spy
    monkeypatch.setattr(portfolio_engine_endpoints, "get_portfolio_engine", lambda: engine)
    monkeypatch.setattr(
        "backend.services.portfolio_engine.is_portfolio_engine_initialized",
        lambda: True,
    )
    monkeypatch.setenv("EXTERNAL_SUPERVISOR_MODE", "false")
    engine._live_execution_enabled = False
    engine._fetch_mtm_prices_for_open_positions = AsyncMock(return_value={})
    engine._recompute_positions_values = AsyncMock()
    monkeypatch.setattr(
        portfolio_engine_endpoints,
        "get_portfolio_integration",
        lambda: MagicMock(
            get_status=lambda: {
                "dust_pending_positions_current": 0,
                "dust_drift_events_total": 0,
                "dust_reconcile_runs_total": 0,
            }
        ),
    )
    monkeypatch.setattr(
        portfolio_engine_endpoints,
        "_sqlite_open_positions_count_sync",
        lambda: 0,
    )

    engine._fetch_mtm_prices_for_open_positions = AsyncMock(return_value={})
    result = await portfolio_engine_endpoints.get_portfolio_status()
    assert result["success"] is True
    engine._fetch_mtm_prices_for_open_positions.assert_not_awaited()

    with sqlite3.connect(db_path) as conn:
        after = {
            "ledger": conn.execute("SELECT cash_balance, version FROM portfolio_engine_ledger WHERE id=1").fetchone(),
            "positions": conn.execute("SELECT COUNT(*) FROM portfolio_engine_positions").fetchone()[0],
            "paper": conn.execute("SELECT COUNT(*) FROM paper_trades").fetchone()[0],
            "scoreboard": conn.execute("SELECT COUNT(*) FROM portfolio_engine_scoreboard_daily").fetchone()[0],
            "audit": conn.execute("SELECT COUNT(*) FROM portfolio_engine_audit").fetchone()[0],
        }
    assert before == after
    fifo_spy.assert_not_called()


@pytest.mark.asyncio
async def test_load_positions_allow_mutations_false_skips_fifo(tmp_path: Path) -> None:
    db_path = tmp_path / "fifo.db"
    engine = PortfolioEngine(db_path=str(db_path), principal=10_000.0, test_mode=True)
    engine._ensure_db_schema()
    fifo_spy = MagicMock()
    engine._sync_paper_fifo_remaining_to_engine_positions_sync = fifo_spy
    await engine._load_positions_from_sqlite(allow_mutations=False)
    fifo_spy.assert_not_called()

    await engine._load_positions_from_sqlite(allow_mutations=True)
    fifo_spy.assert_called()


@pytest.mark.asyncio
async def test_locked_db_scoreboard_fails_fast(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    db_path = tmp_path / "lock.db"
    engine = PortfolioEngine(db_path=str(db_path), principal=10_000.0, test_mode=True)
    engine._ensure_db_schema()

    def _locked(*_a, **_k):
        raise sqlite3.OperationalError("database is locked")

    monkeypatch.setattr("backend.services.portfolio_engine.connect_ro", _locked)

    started = time.monotonic()
    data = await asyncio.wait_for(engine.get_scoreboard_today(), timeout=2.0)
    elapsed = time.monotonic() - started
    assert elapsed < 1.5
    assert data.get("status") == "SQLITE_BUSY"
    assert data.get("degraded") is True


@pytest.mark.asyncio
async def test_locked_db_status_endpoint_bounded(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    db_path = tmp_path / "lock_status.db"
    engine = PortfolioEngine(db_path=str(db_path), principal=10_000.0, test_mode=True)
    engine._ensure_db_schema()
    monkeypatch.setattr(portfolio_engine_endpoints, "get_portfolio_engine", lambda: engine)
    monkeypatch.setattr(
        "backend.services.portfolio_engine.is_portfolio_engine_initialized",
        lambda: True,
    )
    monkeypatch.setenv("EXTERNAL_SUPERVISOR_MODE", "true")
    engine._live_execution_enabled = False

    async def _busy(*_a, **_k):
        raise sqlite3.OperationalError("database is locked")

    engine._load_ledger_from_sqlite = _busy
    engine._load_positions_from_sqlite = _busy
    monkeypatch.setattr(
        portfolio_engine_endpoints,
        "_sqlite_open_positions_count_sync",
        lambda: 0,
    )

    started = time.monotonic()
    with pytest.raises(HTTPException) as exc:
        await asyncio.wait_for(portfolio_engine_endpoints.get_portfolio_status(), timeout=2.0)
    elapsed = time.monotonic() - started
    assert elapsed < 1.5
    assert exc.value.status_code == 503
    assert "sqlite_busy" in str(exc.value.detail).lower() or "locked" in str(exc.value.detail).lower()


def test_connect_ro_sets_busy_timeout(tmp_path: Path) -> None:
    db = tmp_path / "ro.db"
    with sqlite3.connect(db) as c:
        c.execute("CREATE TABLE t(x INTEGER)")
        c.commit()
    conn = connect_ro(db, timeout_sec=1.5)
    try:
        row = conn.execute("PRAGMA busy_timeout").fetchone()
        assert row is not None
        assert int(row[0]) == 1500
        assert is_locked_error(sqlite3.OperationalError("database is locked"))
    finally:
        conn.close()


def test_get_scoreboard_route_not_writer_path() -> None:
    """FIFO reconcile / update_scoreboard must stay on writer or mutation paths."""
    score_src = inspect.getsource(portfolio_engine_endpoints.get_scoreboard_today)
    status_src = inspect.getsource(portfolio_engine_endpoints.get_portfolio_status)
    assert "await engine.update_scoreboard" not in score_src
    assert "_sync_paper_fifo_remaining_to_engine_positions_sync" not in status_src
    assert "allow_mutations=False" in status_src
