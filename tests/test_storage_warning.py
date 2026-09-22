"""Disk headroom warnings and protected trading history."""

from __future__ import annotations

import pytest

from backend.services.sqlite_large_table_retention import (
    DISK_CRITICAL_FREE_GB,
    DISK_WARNING_FREE_GB,
    PROTECTED_TABLES,
    RETENTION_POLICIES,
    storage_report,
)


def test_trading_history_is_never_on_a_deletion_timer():
    policy = {p.table for p in RETENTION_POLICIES}
    for table in (
        "paper_trades",
        "live_exchange_fills",
        "feature_ohlcv",
        "portfolio_engine_ledger",
        "portfolio_engine_positions",
        "portfolio_engine_orders",
        "portfolio_engine_audit",
        "day_entry_reservations",
        "day_trailing_buy_intents",
        "scalp_paper_trades",
        "scalp_paper_ledger",
        "scalp_paper_positions",
    ):
        assert table not in policy
        assert table in PROTECTED_TABLES


def test_warning_bands_are_below_unsafe_utilization():
    assert DISK_WARNING_FREE_GB >= 5.0
    assert DISK_CRITICAL_FREE_GB >= 2.0
    assert DISK_CRITICAL_FREE_GB < DISK_WARNING_FREE_GB


def test_storage_report_warns_when_free_space_is_low(tmp_path, monkeypatch):
    db = tmp_path / "tiny.db"
    db.write_bytes(b"0")

    class _Usage:
        total = 10 * 1024**3
        used = 9 * 1024**3
        free = 1 * 1024**3

    monkeypatch.setattr("backend.services.sqlite_large_table_retention.shutil.disk_usage", lambda _p: _Usage())
    report = storage_report(db)
    assert report["severity"] == "CRITICAL"
    assert report["filesystem_free_gib"] == 1.0


def test_storage_report_accepts_mixed_epoch_and_iso_timestamps(tmp_path, monkeypatch):
    import sqlite3

    db = tmp_path / "mixed.db"
    conn = sqlite3.connect(db)
    conn.execute("CREATE TABLE paper_trades (timestamp REAL)")
    conn.execute("CREATE TABLE day_trailing_buy_intents (created_at TEXT)")
    conn.execute("INSERT INTO paper_trades VALUES (1789749000.0)")
    conn.execute("INSERT INTO day_trailing_buy_intents VALUES ('2026-09-18T16:30:00+00:00')")
    conn.commit()
    conn.close()

    class _Usage:
        total = 50 * 1024**3
        used = 20 * 1024**3
        free = 30 * 1024**3

    monkeypatch.setattr("backend.services.sqlite_large_table_retention.shutil.disk_usage", lambda _p: _Usage())
    report = storage_report(db)
    assert "error" not in report
    assert report["severity"] == "OK"
    assert report["learning_oldest"]
    assert report["learning_newest"]


def test_disk_pressure_reports_wal_and_blocks_only_at_critical(tmp_path, monkeypatch):
    from backend.services.sqlite_large_table_retention import disk_blocks_new_entries, disk_pressure, note_disk_pressure

    db = tmp_path / "live.db"
    db.write_bytes(b"db")
    wal = tmp_path / "live.db-wal"
    wal.write_bytes(b"wal-bytes")

    class _Critical:
        total = 48 * 1024**3
        used = 47 * 1024**3
        free = 1 * 1024**3

    class _Warning:
        total = 48 * 1024**3
        used = 44 * 1024**3
        free = 4 * 1024**3

    monkeypatch.setattr("backend.services.sqlite_large_table_retention.shutil.disk_usage", lambda _p: _Critical())
    report = disk_pressure(db)
    assert report["severity"] == "CRITICAL"
    assert report["wal_bytes"] == len(b"wal-bytes")
    assert report["db_bytes"] == 2
    assert report["blocks_new_entries"] is True
    assert report["exits_retained"] is True
    assert report["deletes_databases"] is False
    blocked, reason = disk_blocks_new_entries(db)
    assert blocked is True
    assert reason.startswith("DISK_CRITICAL")
    assert db.exists() and wal.exists()

    monkeypatch.setattr("backend.services.sqlite_large_table_retention.shutil.disk_usage", lambda _p: _Warning())
    warning = note_disk_pressure(db, min_interval_sec=0)
    assert warning["severity"] == "WARNING"
    assert warning["blocks_new_entries"] is False
    assert db.exists()


@pytest.mark.asyncio
async def test_critical_disk_blocks_new_entries_without_pausing_exits(monkeypatch):
    import backend.services.portfolio_engine as pe

    monkeypatch.setattr(
        "backend.services.sqlite_large_table_retention.disk_blocks_new_entries",
        lambda _path: (True, "DISK_CRITICAL free_gib=0.50 threshold_gib=2.0"),
    )
    engine = pe.PortfolioEngine(principal=228.0, test_mode=True)
    engine._trading_paused = False
    allowed, reason = await engine._can_open_position("BTC/USDT", 40.0)
    assert allowed is False
    assert reason.startswith("DISK_CRITICAL")
    assert engine._trading_paused is False


def test_storage_report_includes_wal_and_does_not_delete(tmp_path, monkeypatch):
    db = tmp_path / "ok.db"
    db.write_bytes(b"0")
    (tmp_path / "ok.db-wal").write_bytes(b"abcd")

    class _Usage:
        total = 50 * 1024**3
        used = 20 * 1024**3
        free = 30 * 1024**3

    monkeypatch.setattr("backend.services.sqlite_large_table_retention.shutil.disk_usage", lambda _p: _Usage())
    report = storage_report(db)
    assert report["wal_bytes"] == 4
    assert report["blocks_new_entries"] is False
    assert report["deletes_databases"] is False
    assert db.exists()


def test_storage_report_is_ok_with_headroom(tmp_path, monkeypatch):
    db = tmp_path / "ok.db"
    db.write_bytes(b"0")

    class _Usage:
        total = 50 * 1024**3
        used = 20 * 1024**3
        free = 30 * 1024**3

    monkeypatch.setattr("backend.services.sqlite_large_table_retention.shutil.disk_usage", lambda _p: _Usage())
    report = storage_report(db)
    assert report["severity"] == "OK"
