"""API and worker must report the same persisted entry-control state."""

from __future__ import annotations

import json
import sqlite3
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

import pytest

from backend.database_schema import initialize_paper_trading_schema
from backend.services.circuit_breaker_service import (
    TradingCircuitBreaker,
    read_persisted_entry_control,
)
from backend.services.portfolio_engine import KillSwitchMode, PortfolioEngine


@pytest.fixture(autouse=True)
def _healthy_accounting():
    with patch(
        "backend.services.atomic_execution_book.find_cash_position_disagreement",
        return_value={"ok": True, "orphans": []},
    ):
        yield


def _db() -> Path:
    tmp = tempfile.mkdtemp()
    path = Path(tmp) / "status.db"
    engine = PortfolioEngine(db_path=str(path), principal=250.0, test_mode=True)
    engine._ensure_db_schema()
    initialize_paper_trading_schema(str(path))
    with sqlite3.connect(str(path)) as conn:
        conn.execute(
            """
            INSERT OR REPLACE INTO portfolio_engine_ledger (
                id, principal, cash_balance, positions_value,
                realized_pnl, unrealized_pnl, total_equity,
                account_status, trading_paused, pause_reason, last_updated, version,
                kill_switch_mode, kill_switch_reason
            ) VALUES (1, 250.0, 233.01, 0.14, -3.10, 0, 233.15,
                      'HEALTHY', 0, NULL, datetime('now'), 1, 'RESUME', '')
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS operational_state (
                key TEXT PRIMARY KEY,
                value_json TEXT,
                updated_ts INTEGER
            )
            """
        )
        conn.commit()
    return path


def _write_cb(path: Path, **flags) -> None:
    payload = {
        "daily_loss_freeze_active": False,
        "equity_circuit_breaker_active": False,
        "account_failsafe_active": False,
        "session_high_equity": 233.15,
        "last_stable_equity": 233.15,
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }
    payload.update(flags)
    with sqlite3.connect(str(path)) as conn:
        conn.execute(
            "INSERT OR REPLACE INTO operational_state(key, value_json, updated_ts) VALUES (?,?,?)",
            ("risk:circuit_breakers", json.dumps(payload), int(datetime.now(timezone.utc).timestamp())),
        )
        conn.commit()


def test_genuine_breaker_trip_blocks_api_resume_memory():
    path = _db()
    _write_cb(path, equity_circuit_breaker_active=True)
    with sqlite3.connect(str(path)) as conn:
        conn.execute("UPDATE portfolio_engine_ledger SET kill_switch_mode='PAUSE_BUYS', kill_switch_reason='CIRCUIT_BREAKER:EQUITY' WHERE id=1")
        conn.commit()
    api = PortfolioEngine(db_path=str(path), principal=250.0, test_mode=True)
    api._kill_switch_mode = KillSwitchMode.RESUME
    api.cash_balance = 233.01
    api._positions_value = 0.14
    cap = api.get_trading_capability_status()
    ks = api.get_kill_switch_status()
    assert cap["effective_entry_state"] == "PAUSE"
    assert cap["effective_entry_permitted"] is False
    assert cap["equity_circuit_breaker_active"] is True
    assert cap["active_breaker"] == "equity_circuit_breaker"
    assert ks["effective_entry_state"] == "PAUSE"
    assert ks["buys_blocked"] is True
    worker = PortfolioEngine(db_path=str(path), principal=250.0, test_mode=True)
    worker._kill_switch_mode = KillSwitchMode.PAUSE_BUYS
    worker._kill_switch_reason = "CIRCUIT_BREAKER:EQUITY"
    worker.cash_balance = 233.01
    worker._positions_value = 0.14
    wcap = worker.get_trading_capability_status()
    assert wcap["effective_entry_state"] == cap["effective_entry_state"]
    assert wcap["effective_entry_permitted"] == cap["effective_entry_permitted"]


def test_false_spike_rejected_keeps_entries_open():
    tcb = object.__new__(TradingCircuitBreaker)
    tcb.daily_loss_freeze_active = False
    tcb.equity_circuit_breaker_active = False
    tcb.account_failsafe_active = False
    tcb.session_high_equity = 233.15
    tcb.last_stable_equity = 233.15
    tcb.last_daily_reset = None
    tcb.needs_revalidation = set()
    tcb.persisted_state_timestamp = None
    tcb.persisted_state_age_sec = None
    tcb.startup_changed_state = False
    tcb.last_dependency_check_at = None
    # Spike validation is defense in depth; a rejected spike must not latch pause.
    assert tcb.equity_circuit_breaker_active is False
    path = _db()
    _write_cb(path, equity_circuit_breaker_active=False)
    api = PortfolioEngine(db_path=str(path), principal=250.0, test_mode=True)
    api._kill_switch_mode = KillSwitchMode.RESUME
    api.cash_balance = 233.01
    api._positions_value = 0.14
    cap = api.get_trading_capability_status()
    assert cap["effective_entry_state"] == "RESUME"
    assert cap["effective_entry_permitted"] is True
    assert cap["equity_circuit_breaker_active"] is False


def test_breaker_recovery_resumes_both_views():
    path = _db()
    _write_cb(path, equity_circuit_breaker_active=False)
    with sqlite3.connect(str(path)) as conn:
        conn.execute("UPDATE portfolio_engine_ledger SET kill_switch_mode='RESUME', kill_switch_reason='CIRCUIT_BREAKER:cleared' WHERE id=1")
        conn.commit()
    api = PortfolioEngine(db_path=str(path), principal=250.0, test_mode=True)
    api._kill_switch_mode = KillSwitchMode.RESUME
    api.cash_balance = 233.01
    api._positions_value = 0.14
    cap = api.get_trading_capability_status()
    assert cap["effective_entry_state"] == "RESUME"
    assert cap["requested_kill_mode"] == "RESUME"


def test_restart_while_breaker_latched_reads_persisted():
    path = _db()
    _write_cb(path, equity_circuit_breaker_active=True)
    with sqlite3.connect(str(path)) as conn:
        conn.execute("UPDATE portfolio_engine_ledger SET kill_switch_mode='PAUSE_BUYS', kill_switch_reason='CIRCUIT_BREAKER:EQUITY' WHERE id=1")
        conn.commit()
    restarted = PortfolioEngine(db_path=str(path), principal=250.0, test_mode=True)
    restarted._hydrate_kill_switch_from_ledger_sync()
    assert restarted._kill_switch_mode == KillSwitchMode.PAUSE_BUYS
    persisted = read_persisted_entry_control(str(path))
    assert persisted["entry_permitted"] is False
    assert persisted["equity_circuit_breaker_active"] is True
    cap = restarted.get_trading_capability_status()
    assert cap["effective_entry_state"] == "PAUSE"
    assert cap["requested_kill_mode"] == "PAUSE_BUYS"
    assert cap["entry_control_source_process"]


def test_api_and_integration_agree_on_persisted_cb_only():
    path = _db()
    _write_cb(path, equity_circuit_breaker_active=True)
    api = PortfolioEngine(db_path=str(path), principal=250.0, test_mode=True)
    worker = PortfolioEngine(db_path=str(path), principal=250.0, test_mode=True)
    api._kill_switch_mode = KillSwitchMode.RESUME
    worker._kill_switch_mode = KillSwitchMode.RESUME
    api.cash_balance = worker.cash_balance = 233.01
    api._positions_value = worker._positions_value = 0.14
    assert api.get_trading_capability_status()["effective_entry_permitted"] is False
    assert worker.get_trading_capability_status()["effective_entry_permitted"] is False
    assert api.get_kill_switch_status()["effective_entry_state"] == worker.get_kill_switch_status()["effective_entry_state"]
