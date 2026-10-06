"""Deploy-time DB snapshots outside the catalog are adopted, verified and retained by policy.

Ocean 2026-10-06: 15 GB of raw ``mystic_trading-pre-<sha>.db`` and bundle snapshots sat in
/home/mystic/backups outside retention while the disk reached 91%.
"""

from __future__ import annotations

import json
import os
import sqlite3
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from backend.services import mystic_maintenance as m

REPO = Path(__file__).resolve().parents[1]
NOW = datetime(2026, 10, 6, 15, 0, tzinfo=timezone.utc)


@pytest.fixture
def cfg(tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    home = tmp_path / "home"
    for d in (repo / "logs", home / "backups" / "db", tmp_path / "tmp"):
        d.mkdir(parents=True)
    monkeypatch.setenv("MAINT_LIVE_DB", str(repo / "mystic_trading.db"))
    monkeypatch.delenv("MAINT_BACKUP_DIR", raising=False)
    c = m.MaintConfig()
    c.repo, c.home, c.tmp_dir = repo, home, tmp_path / "tmp"
    c.lock_path = tmp_path / "run" / "maint.lock"
    c.deploy_locks = (tmp_path / "run" / "deploy.lock",)
    c.reboot_flag = tmp_path / "reboot-required"
    c.owner = "nobody-such-user"
    return c


def _db(path: Path, *, when: datetime, rows: int = 3) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(path) as conn:
        conn.execute("CREATE TABLE t (v TEXT)")
        conn.executemany("INSERT INTO t VALUES (?)", [(f"r{i}",) for i in range(rows)])
    os.utime(path, (when.timestamp(), when.timestamp()))
    return path


def _adopt_all(cfg, now=NOW):
    out = m.adopt_deploy_snapshots(cfg, dry_run=False, opened=set(), limit=10, now=now)
    assert out["status"] == "ok", out
    return out


def test_loose_and_bundle_snapshots_are_verified_and_cataloged(cfg):
    root = cfg.backups_root
    loose = _db(root / "mystic_trading-pre-ab2c39c.db", when=NOW - timedelta(hours=11))
    bundle = _db(root / "closure_b50c154_20261004T202830Z" / "mystic_trading.db", when=NOW - timedelta(days=2))
    (bundle.parent / "model.pkl").write_bytes(b"m")
    sha = {p.name: m.sha256_file(p) for p in (loose, bundle)}
    out = _adopt_all(cfg)
    assert out["found"] == 2 and len(out["adopted"]) == 2
    assert not loose.exists() and not bundle.exists()
    assert (bundle.parent / "model.pkl").exists()
    backups = {b.manifest["origin"]: b for b in m.list_backups(cfg.backup_dir)}
    for origin, info in backups.items():
        assert info.verified and info.manifest["reason"] == m.DEPLOY_SNAPSHOT_REASON
        assert info.manifest["sha256"] == sha[Path(origin).name] == m.sha256_file(info.path)
        assert int(info.ts.timestamp()) == int(datetime.strptime(info.manifest["created_utc"], "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc).timestamp())
    assert backups[str(loose)].ts == (NOW - timedelta(hours=11)).replace(microsecond=0)


def test_live_db_catalog_and_non_db_files_are_not_candidates(cfg):
    _db(cfg.live_db, when=NOW - timedelta(days=1))
    _db(cfg.backup_dir / m.backup_name(NOW - timedelta(days=1)), when=NOW - timedelta(days=1))
    (cfg.backups_root / "archive").mkdir()
    (cfg.backups_root / "archive" / "mystic_trading.db.pre_x.gz").write_bytes(b"gz")
    (cfg.backups_root / "adaptive-state-pre-x.db").write_bytes(b"x")
    assert m.deploy_snapshot_candidates(cfg) == []


def test_corrupt_snapshot_is_reported_and_left_in_place(cfg):
    bad = cfg.backups_root / "mystic_trading-pre-bad.db"
    bad.write_bytes(b"not a database" * 100)
    os.utime(bad, ((NOW - timedelta(hours=2)).timestamp(),) * 2)
    out = m.adopt_deploy_snapshots(cfg, dry_run=False, opened=set(), now=NOW)
    assert out["status"] == "error" and out["failed"]
    assert bad.exists() and m.list_backups(cfg.backup_dir) == []


def test_recent_open_wal_pending_and_dry_run_are_not_moved(cfg):
    recent = _db(cfg.backups_root / "mystic_trading-pre-new.db", when=NOW - timedelta(seconds=30))
    out = m.adopt_deploy_snapshots(cfg, dry_run=False, opened=set(), now=NOW)
    assert out["deferred"] == [str(recent)] and recent.exists()
    os.utime(recent, ((NOW - timedelta(hours=1)).timestamp(),) * 2)
    out = m.adopt_deploy_snapshots(cfg, dry_run=False, opened={str(recent)}, now=NOW)
    assert out["deferred"] == [str(recent)] and recent.exists()
    Path(f"{recent}-wal").write_bytes(b"w" * 32)
    out = m.adopt_deploy_snapshots(cfg, dry_run=False, opened=set(), now=NOW)
    assert out["skipped"] == [{"path": str(recent), "reason": "wal_not_empty"}] and recent.exists()
    Path(f"{recent}-wal").write_bytes(b"")
    out = m.adopt_deploy_snapshots(cfg, dry_run=True, opened=set(), now=NOW)
    assert out["adopted"][0]["dry_run"] and recent.exists() and m.list_backups(cfg.backup_dir) == []
    _adopt_all(cfg)
    assert not recent.exists() and not Path(f"{recent}-wal").exists()


def test_adopted_snapshots_are_not_immortal_but_newest_and_pins_survive(cfg):
    old = _db(cfg.backups_root / "mystic_trading-pre-old.db", when=NOW - timedelta(days=30))
    pinned = _db(cfg.backups_root / "mystic_trading-pre-keep.db", when=NOW - timedelta(days=29))
    recent = _db(cfg.backups_root / "mystic_trading-pre-recent.db", when=NOW - timedelta(hours=2))
    origin = {p.name for p in (old, pinned, recent)}
    _adopt_all(cfg)
    by_origin = {Path(b.manifest["origin"]).name: b for b in m.list_backups(cfg.backup_dir)}
    assert set(by_origin) == origin
    (cfg.backup_dir / f"{by_origin[pinned.name].path.name}.pin").write_text("")
    for d in range(1, 20):
        p = _db(cfg.backup_dir / m.backup_name(NOW - timedelta(days=d)), when=NOW - timedelta(days=d))
        m.manifest_path_for(p).write_text(json.dumps({"integrity": "ok", "sha256": m.sha256_file(p), "reason": "scheduled"}))
    res = m.apply_backup_retention(cfg, mode="normal", dry_run=True, opened=set(), now=NOW, compress=lambda _i: {"status": "ok"})
    deleted = {Path(a["path"]).name for a in res["actions"] if a["action"] == "delete_backup"}
    assert by_origin[old.name].path.name in deleted
    assert by_origin[pinned.name].path.name not in deleted
    assert res["newest_verified"] == by_origin[recent.name].path.name
    assert by_origin[recent.name].path.name not in deleted


def test_unmanaged_bundles_are_reported_not_hidden(cfg):
    (cfg.backups_root / "archive").mkdir()
    (cfg.backups_root / "archive" / "mystic_trading.db.pre_x.gz").write_bytes(b"g" * 500)
    (cfg.backups_root / "adaptive-state.db").write_bytes(b"a" * 50)
    res = m.apply_backup_retention(cfg, mode="normal", dry_run=True, opened=set(), now=NOW)
    assert [r["name"] for r in res["unmanaged"]] == ["archive", "adaptive-state.db"]
    assert res["unmanaged_bytes"] == 550


def test_run_adopts_after_verify_and_before_retention(cfg):
    calls: list[str] = []

    def owner(task, *args):
        calls.append(task)
        return {"status": "ok"}

    m.run_maintenance(cfg, dry_run=True, allow_reboot=False, owner_task=owner)
    assert calls.index("backup-verify") < calls.index("backup-adopt")


def test_failed_adoption_surfaces_in_run_errors(cfg):
    def owner(task, *args):
        return {"status": "error", "error": "deploy snapshot failed integrity_check: x"} if task == "backup-adopt" else {"status": "ok"}

    out = m.run_maintenance(cfg, dry_run=True, allow_reboot=False, owner_task=owner)
    assert "backup_adopt: deploy snapshot failed integrity_check: x" in out["errors"]


def test_cli_exposes_backup_adopt():
    src = (REPO / "scripts/mystic_maintenance.py").read_text()
    assert '"backup-adopt"' in src and "adopt_deploy_snapshots" in src


def test_adoption_preserves_bytes_exactly(cfg):
    snap = _db(cfg.backups_root / "mystic_trading-pre-big.db", when=NOW - timedelta(hours=3), rows=500)
    before = snap.read_bytes()
    _adopt_all(cfg)
    (info,) = m.list_backups(cfg.backup_dir)
    assert info.path.read_bytes() == before
    assert json.loads(info.manifest_path.read_text())["bytes"] == len(before)
    assert time.time() > 0
