"""Disk headroom warnings and protected trading history."""

from __future__ import annotations

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
