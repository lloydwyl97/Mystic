"""Routine ledger saves must not erase a persisted operator pause."""

from __future__ import annotations

import inspect
import re
import sqlite3

from backend.services import portfolio_engine

LEDGER_SCHEMA = """
CREATE TABLE portfolio_engine_ledger (
    id INTEGER PRIMARY KEY CHECK (id = 1),
    principal REAL NOT NULL,
    cash_balance REAL NOT NULL,
    positions_value REAL NOT NULL,
    realized_pnl REAL NOT NULL,
    unrealized_pnl REAL NOT NULL,
    total_equity REAL NOT NULL,
    account_status TEXT NOT NULL DEFAULT 'HEALTHY',
    trading_paused INTEGER NOT NULL DEFAULT 0,
    pause_reason TEXT,
    last_updated TEXT NOT NULL,
    version INTEGER NOT NULL DEFAULT 1,
    kill_switch_mode TEXT DEFAULT 'RESUME',
    kill_switch_reason TEXT,
    regime_state_json TEXT,
    startup_timestamp TEXT
)
"""


def _ledger_writes() -> list[str]:
    src = inspect.getsource(portfolio_engine)
    return re.findall(r'"""\s*(INSERT[^"]*?INTO portfolio_engine_ledger\b.*?)"""', src, flags=re.S)


def test_every_ledger_write_is_found():
    assert len(_ledger_writes()) >= 2


def test_ledger_saves_keep_the_operator_pause_and_other_columns(tmp_path):
    conn = sqlite3.connect(tmp_path / "ledger.db")
    conn.execute(LEDGER_SCHEMA)
    conn.execute("INSERT INTO portfolio_engine_ledger VALUES (1, 1000, 900, 100, 0, 0, 1000, 'HEALTHY', 0, NULL, 't0', 7, 'PAUSE_BUYS', 'operator', '{\"r\":1}', 's0')")
    for sql in _ledger_writes():
        conn.execute(sql, (1000.0, 905.1, 100.0, 1.0, 0.0, 1005.1, "HEALTHY", 0, None, "t1"))
    row = conn.execute("SELECT cash_balance, kill_switch_mode, kill_switch_reason, regime_state_json, startup_timestamp, version FROM portfolio_engine_ledger WHERE id=1").fetchone()
    assert row == (905.1, "PAUSE_BUYS", "operator", '{"r":1}', "s0", 7 + len(_ledger_writes()))


def test_first_ledger_save_still_creates_the_row(tmp_path):
    conn = sqlite3.connect(tmp_path / "ledger.db")
    conn.execute(LEDGER_SCHEMA)
    conn.execute(_ledger_writes()[0], (1000.0, 1000.0, 0.0, 0.0, 0.0, 1000.0, "HEALTHY", 0, None, "t1"))
    assert conn.execute("SELECT kill_switch_mode, version FROM portfolio_engine_ledger").fetchone() == ("RESUME", 1)
