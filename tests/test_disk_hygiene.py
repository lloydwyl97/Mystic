"""Local disk hygiene: logs, caches, and both SQLite files have a cleanup job."""

from __future__ import annotations

from pathlib import Path

from backend.services.sqlite_large_table_retention import (
    PROTECTED_TABLES,
    RETENTION_POLICIES,
    iter_retention_db_paths,
)

REPO = Path(__file__).resolve().parents[1]


def test_local_logrotate_covers_process_logs_and_cursor_debug():
    text = (REPO / "deploy/mystic-logrotate.conf").read_text()
    assert "/home/mystic/mystic/logs/*.log" in text
    assert "/home/mystic/.cursor/debug.log" in text
    assert "copytruncate" in text
    assert "size 20M" in text
    assert "rotate 5" in text
    assert "delaycompress" not in text


def test_disk_hygiene_script_cleans_caches_and_rotates_logs():
    text = (REPO / "scripts/disk_hygiene.sh").read_text()
    assert "logrotate" in text
    assert "mystic-logrotate.conf" in text
    assert "cache purge" in text
    assert ".cache/pip" in text
    assert ".cursor/worktrees" in text
    assert ".cursor-server/bin/linux-x64" in text
    assert ".cursor/debug.log" in text
    assert "VACUUM" not in text or "Does not VACUUM" in text


def test_offline_vacuum_covers_both_databases_and_holds_watchdog():
    text = (REPO / "scripts/offline_sqlite_vacuum_maintenance.sh").read_text()
    assert "mystic_trading.db" in text
    assert "mystic_scalp.db" in text
    assert "mystic_maintenance.lock" in text
    assert "binance_scalp.runner" in text
    assert "--auto-manage-services" in text


def test_scalp_money_tables_are_not_on_a_deletion_timer():
    by_table = {p.table: p for p in RETENTION_POLICIES}
    for table in ("scalp_paper_trades", "scalp_paper_ledger", "scalp_paper_positions"):
        assert table in PROTECTED_TABLES
        assert table not in by_table
    guard = (REPO / "backend/services/sqlite_large_table_retention.py").read_text()
    start = guard.find("def disk_pressure")
    end = guard.find("def storage_report")
    body = guard[start:end]
    assert "def disk_blocks_new_entries" in body
    assert "def note_disk_pressure" in body
    assert "unlink" not in body
    assert "os.remove" not in body
    assert "deletes_databases" in body


def test_iter_retention_db_paths_includes_scalp_when_present(tmp_path):
    day = tmp_path / "mystic_trading.db"
    scalp = tmp_path / "mystic_scalp.db"
    day.write_bytes(b"0")
    scalp.write_bytes(b"0")
    paths = iter_retention_db_paths(day, repo_root=tmp_path, env_get=lambda *_a, **_k: None)
    assert day in paths
    assert scalp in paths


def test_iter_retention_db_paths_skips_missing_scalp(tmp_path):
    day = tmp_path / "mystic_trading.db"
    day.write_bytes(b"0")
    paths = iter_retention_db_paths(day, repo_root=tmp_path, env_get=lambda *_a, **_k: None)
    assert paths == [day]
