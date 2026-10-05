"""Production maintenance: retention safety, temp patterns, disk thresholds, reboot gate."""

from __future__ import annotations

import json
import os
import sqlite3
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from backend.services import mystic_maintenance as m

NOW = datetime(2026, 10, 3, 12, 0, tzinfo=timezone.utc)


@pytest.fixture
def cfg(tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    home = tmp_path / "home"
    tmp = tmp_path / "tmp"
    for d in (repo / "logs", repo / "scripts", home / "backups" / "db", tmp):
        d.mkdir(parents=True)
    monkeypatch.setenv("MAINT_LIVE_DB", str(repo / "mystic_trading.db"))
    monkeypatch.delenv("MAINT_BACKUP_DIR", raising=False)
    monkeypatch.delenv("MAINT_STATUS_PATH", raising=False)
    c = m.MaintConfig()
    c.repo, c.home, c.tmp_dir = repo, home, tmp
    c.lock_path = tmp_path / "run" / "maint.lock"
    c.deploy_locks = (tmp_path / "run" / "deploy.lock", tmp / "mystic_maintenance.lock")
    c.reboot_flag = tmp_path / "reboot-required"
    c.owner = "nobody-such-user"
    return c


def _live_db(cfg, rows=50):
    conn = sqlite3.connect(cfg.live_db)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("CREATE TABLE paper_trades (id INTEGER PRIMARY KEY, v TEXT)")
    conn.executemany("INSERT INTO paper_trades (v) VALUES (?)", [(f"row{i}" * 20,) for i in range(rows)])
    conn.commit()
    conn.close()


def _fake_backup(cfg, ts: datetime, *, verified=True, compressed=False, payload=b"x" * 64) -> Path:
    name = m.backup_name(ts) + (".gz" if compressed else "")
    p = cfg.backup_dir / name
    p.write_bytes(payload)
    if verified:
        manifest = {"integrity": "ok", "sha256": "f" * 64, "verified_utc": m._iso(ts), "bytes": len(payload)}
        m.manifest_path_for(p).write_text(json.dumps(manifest))
    return p


def _old(path: Path, hours: float) -> None:
    t = time.time() - hours * 3600
    os.utime(path, (t, t))


# --- backup retention -------------------------------------------------------------


def test_retention_keeps_newest_and_generations(cfg):
    paths = [_fake_backup(cfg, NOW - timedelta(days=d), compressed=d > 0) for d in range(0, 30)]
    backups = m.list_backups(cfg.backup_dir)
    keep = m.select_retained(backups, now=NOW, keep_hourly=24, keep_daily=3, keep_weekly=2, pinned=set())
    assert paths[0] in keep
    assert {paths[0], paths[1], paths[2]} <= keep
    assert len(keep) <= 1 + 3 + 2
    assert paths[29] not in keep


def test_retention_never_deletes_newest_pinned_unverified_or_open(cfg):
    newest = _fake_backup(cfg, NOW)
    pinned = _fake_backup(cfg, NOW - timedelta(days=40), compressed=True)
    unverified = _fake_backup(cfg, NOW - timedelta(days=41), verified=False)
    opened = _fake_backup(cfg, NOW - timedelta(days=42), compressed=True)
    doomed = _fake_backup(cfg, NOW - timedelta(days=43), compressed=True)
    (cfg.backup_dir / m.PIN_FILE).write_text(f"# forensic\n{pinned.name}\n")
    out = m.apply_backup_retention(cfg, mode="aggressive", dry_run=False, opened={os.path.realpath(opened)}, now=NOW, compress=lambda _b: {"status": "ok"})
    for p in (newest, pinned, unverified, opened):
        assert p.exists(), p
    assert not doomed.exists()
    assert not m.manifest_path_for(doomed).exists()
    assert any(a["action"] == "skip_open_backup" for a in out["actions"])


def test_retention_ignores_files_it_did_not_create(cfg):
    _fake_backup(cfg, NOW)
    foreign = cfg.backup_dir / "mystic_trading.db.pre_audit_20260929T142427Z"
    foreign.write_bytes(b"keep me")
    partial = cfg.backup_dir / (m.backup_name(NOW - timedelta(days=50)) + ".partial")
    partial.write_bytes(b"in flight")
    m.apply_backup_retention(cfg, mode="critical", dry_run=False, opened=set(), now=NOW, compress=lambda _b: {"status": "ok"})
    assert foreign.exists() and partial.exists()


def test_backup_create_verify_compress_round_trip(cfg):
    _live_db(cfg)
    res = m.create_backup(cfg, dry_run=False)
    assert res["status"] == "ok", res
    [info] = m.list_backups(cfg.backup_dir)
    assert info.verified and not info.compressed
    assert oct(info.path.stat().st_mode & 0o777) == "0o600"
    assert sorted(p.name for p in cfg.backup_dir.iterdir()) == sorted([info.path.name, info.manifest_path.name])
    conn = sqlite3.connect(info.path)
    assert conn.execute("PRAGMA journal_mode").fetchone()[0] == "delete"
    assert conn.execute("SELECT COUNT(*) FROM paper_trades").fetchone()[0] == 50
    conn.close()
    out = m.compress_backup(info, owner=cfg.owner)
    assert out["status"] == "ok"
    [gz] = m.list_backups(cfg.backup_dir)
    assert gz.compressed and gz.manifest["compressed"] is True
    assert m.sha256_file(gz.path, opener=lambda p: __import__("gzip").open(p, "rb")) == info.manifest["sha256"]


def test_backup_skips_when_disk_cannot_hold_it(cfg, monkeypatch):
    _live_db(cfg)
    cfg.backup_free_reserve_bytes = 10**18
    res = m.create_backup(cfg, dry_run=False)
    assert res["status"] == "skipped"
    assert m.list_backups(cfg.backup_dir) == []


def test_backup_rejects_corrupt_copy(cfg, monkeypatch):
    _live_db(cfg)
    monkeypatch.setattr(m, "sqlite_integrity", lambda _p: "*** page 3: btree corrupt")
    res = m.create_backup(cfg, dry_run=False)
    assert res["status"] == "error"
    assert list(cfg.backup_dir.iterdir()) == []


def test_compress_aborts_if_source_changed(cfg):
    p = _fake_backup(cfg, NOW - timedelta(days=5))
    [info] = m.list_backups(cfg.backup_dir)
    out = m.compress_backup(info, owner=cfg.owner)
    assert out["status"] == "error"
    assert p.exists() and not p.with_name(p.name + ".gz").exists()


# --- temp + logs --------------------------------------------------------------------


def test_temp_cleanup_only_old_unopened_mystic_patterns(cfg):
    t = cfg.tmp_dir
    old_mystic = t / "replay_copy.db"
    young_mystic = t / "scalp_slim.db"
    foreign = t / "systemd-private-abc"
    generic = t / "st.json"
    legacy_lock = t / "mystic_maintenance.lock"
    pid = t / "mystic_ai_learning.pid"
    monitor = t / "mystic_monitor_run_until_epoch"
    open_one = t / "ocean_profit.json"
    keep = (young_mystic, foreign, generic, legacy_lock, pid, monitor, open_one)
    for p in (old_mystic, young_mystic, generic, legacy_lock, pid, monitor, open_one):
        p.write_text("x")
    foreign.mkdir()
    for p in (old_mystic, foreign, generic, legacy_lock, pid, monitor, open_one):
        _old(p, 72)
    out = m.clean_temp(cfg, mode="normal", dry_run=False, opened={os.path.realpath(open_one)})
    assert not old_mystic.exists()
    for p in keep:
        assert p.exists(), p
    assert out["bytes_reclaimed"] == 1


def test_aggressive_mode_shortens_temp_age(cfg):
    p = cfg.tmp_dir / "day_scratch.json"
    p.write_text("x")
    _old(p, 3)
    m.clean_temp(cfg, mode="normal", dry_run=False, opened=set())
    assert p.exists()
    m.clean_temp(cfg, mode="aggressive", dry_run=False, opened=set())
    assert not p.exists()


def test_log_snapshots_compressed_then_expired(cfg):
    logs = cfg.logs_dir
    snap = logs / "mystic_portfolio.log.pre_ca07c60"
    snap.write_text("line\n" * 1000)
    _old(snap, 48)
    active = logs / "mystic_portfolio.log"
    active.write_text("live")
    _old(active, 24 * 30)
    old_gz = logs / "mystic_portfolio.log.pre_old.gz"
    old_gz.write_bytes(b"gz")
    _old(old_gz, 24 * 20)
    m.apply_log_retention(cfg, mode="normal", dry_run=False, opened={os.path.realpath(active)})
    assert not snap.exists() and (logs / "mystic_portfolio.log.pre_ca07c60.gz").exists()
    assert active.exists()  # open by a process: never touched
    assert not old_gz.exists()


# --- disk guard ---------------------------------------------------------------------


@pytest.mark.parametrize(("pct", "mode"), [(10, "normal"), (74.9, "normal"), (75, "normal"), (85, "aggressive"), (91.9, "aggressive"), (92, "critical"), (99, "critical")])
def test_disk_thresholds(cfg, pct, mode):
    assert m.disk_mode(pct, cfg) == mode


# --- DB retention -------------------------------------------------------------------


def test_db_retention_uses_only_existing_policies_and_never_vacuums(tmp_path):
    from backend.services import sqlite_large_table_retention as r

    db = tmp_path / "t.db"
    conn = sqlite3.connect(db)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("CREATE TABLE ai_context_snapshots (id INTEGER PRIMARY KEY, ts_utc TEXT, blob TEXT)")
    conn.execute("CREATE TABLE paper_trades (id INTEGER PRIMARY KEY, created_at TEXT)")
    conn.execute("CREATE TABLE trade_learning_outcomes (id INTEGER PRIMARY KEY, created_at TEXT)")
    old = (datetime.now(timezone.utc) - timedelta(days=60)).isoformat()
    new = datetime.now(timezone.utc).isoformat()
    conn.executemany("INSERT INTO ai_context_snapshots (ts_utc, blob) VALUES (?, ?)", [(old, "x" * 500)] * 300 + [(new, "y" * 500)] * 10)
    conn.executemany("INSERT INTO paper_trades (created_at) VALUES (?)", [(old,)] * 20)
    conn.executemany("INSERT INTO trade_learning_outcomes (created_at) VALUES (?)", [(old,)] * 20)
    conn.commit()
    conn.close()
    assert "paper_trades" in r.PROTECTED_TABLES
    assert "trade_learning_outcomes" not in {p.table for p in r.RETENTION_POLICIES}
    out = m.db_table_retention(db, dry_run=False)
    t = out["tables"]["ai_context_snapshots"]
    assert (t["rows_before"], t["removed"], t["remaining"]) == (310, 300, 10)
    assert "paper_trades" not in out["tables"] and "trade_learning_outcomes" not in out["tables"]
    conn = sqlite3.connect(db)
    assert conn.execute("SELECT COUNT(*) FROM paper_trades").fetchone()[0] == 20
    assert conn.execute("SELECT COUNT(*) FROM trade_learning_outcomes").fetchone()[0] == 20
    conn.close()
    assert out["vacuum"] == "not_run"
    assert out["pages_after"] == out["pages_before"]  # space reused, file not rewritten


def test_db_report_reports_wal_and_checkpoints_only_when_large(cfg):
    _live_db(cfg)
    out = m.db_report(cfg.live_db, checkpoint_threshold_bytes=10**12, dry_run=False)
    assert out["journal_mode"] == "wal" and "checkpoint" not in out
    out = m.db_report(cfg.live_db, checkpoint_threshold_bytes=0, dry_run=False)
    assert out["checkpoint"]["mode"] == "PASSIVE"


# --- reboot gate --------------------------------------------------------------------


def _status(**over):
    data = {
        "positions_count": 0,
        "trailing_buy_intents": [],
        "exit_blocked_positions": [],
        "account_status": "HEALTHY",
        "accounting_healthy": True,
        "equity_invariant_ok": True,
        "dust_positions": [],
    }
    data.update(over)
    return {"success": True, "data": data}


def _gate(status=None, *, orders=0, code=200, lock=None, required=True):
    return m.evaluate_reboot_gate(reboot_required=required, deploy_lock=lock, status_code=code, status=status or _status(), open_orders=orders)


def test_gate_passes_when_flat_and_healthy():
    assert _gate() == []


def test_dust_does_not_block_reboot():
    assert _gate(_status(dust_positions=[{"symbol": "SOLUSDT", "value": 0.4}], dust_market_value=0.4)) == []


@pytest.mark.parametrize(
    ("kwargs", "reason"),
    [
        ({"status": _status(positions_count=1)}, "active_positions:1"),
        ({"orders": 2}, "open_exchange_orders:2"),
        ({"orders": None}, "open_orders_unknown"),
        ({"status": _status(account_status="DEGRADED")}, "account_status:DEGRADED"),
        ({"status": _status(accounting_healthy=False)}, "accounting_unhealthy"),
        ({"status": _status(trailing_buy_intents=[{"symbol": "BTCUSDT"}])}, "trailing_buy_intents:1"),
        ({"code": 503, "status": {}}, "status_unavailable:503"),
        ({"lock": Path("/run/mystic/deploy.lock")}, "deploy_lock_held:/run/mystic/deploy.lock"),
        ({"required": False}, "reboot_not_required"),
    ],
)
def test_gate_blocks(kwargs, reason):
    assert reason in _gate(**kwargs)


def _reboot_harness(cfg, *, statuses, orders=0):
    calls = {"reboot": 0, "backup": 0}
    seq = iter(statuses)

    def backup():
        calls["backup"] += 1
        _fake_backup(cfg, m.utc_now().replace(microsecond=0))
        return {"status": "ok"}

    out = m.maybe_reboot(
        cfg,
        dry_run=False,
        fetch_status=lambda: (200, next(seq)),
        open_orders=lambda: orders,
        make_backup=backup,
        reboot=lambda: calls.__setitem__("reboot", calls["reboot"] + 1),
        held_pending=lambda: [],
        upgrade_held=lambda _pkgs: {},
    )
    return out, calls


def test_reboot_takes_backup_then_rechecks_then_reboots(cfg):
    cfg.reboot_flag.write_text("*** System restart required ***")
    out, calls = _reboot_harness(cfg, statuses=[_status(), _status()])
    assert calls == {"reboot": 1, "backup": 1}
    assert out["rebooted"] is True
    assert json.loads(cfg.reboot_marker.read_text())["reason"] == "reboot-required"


def test_position_opened_during_backup_defers_reboot(cfg):
    cfg.reboot_flag.write_text("x")
    out, calls = _reboot_harness(cfg, statuses=[_status(), _status(positions_count=1)])
    assert calls["reboot"] == 0
    assert out["deferral_reason"] == "active_positions:1"


def test_open_order_defers_reboot_without_backup(cfg):
    cfg.reboot_flag.write_text("x")
    out, calls = _reboot_harness(cfg, statuses=[_status()], orders=1)
    assert calls == {"reboot": 0, "backup": 0}
    assert "open_exchange_orders:1" in out["deferral_reason"]


def test_no_reboot_when_not_required(cfg):
    out, calls = _reboot_harness(cfg, statuses=[])
    assert calls == {"reboot": 0, "backup": 0} and out["reboot_required"] is False


# --- orchestration ------------------------------------------------------------------


def test_maintenance_lock_is_exclusive(cfg):
    with m.maintenance_lock(cfg.lock_path), pytest.raises(m.MaintenanceLockBusyError), m.maintenance_lock(cfg.lock_path):
        pass
    with m.maintenance_lock(cfg.lock_path):
        pass  # released after the first holder exits


def test_concurrent_run_is_skipped(cfg):
    with m.maintenance_lock(cfg.lock_path):
        out = m.run_maintenance(cfg, dry_run=False, owner_task=lambda *_a: {}, reboot_fn=lambda: {})
    assert out["skipped"] == "maintenance_lock_busy"


def _populate(cfg):
    _live_db(cfg)
    tmp_old = cfg.tmp_dir / "rebuild_30a9a4f.log"
    tmp_old.write_text("x")
    _old(tmp_old, 72)
    snap = cfg.logs_dir / "mystic_portfolio.log.pre_x"
    snap.write_text("x")
    _old(snap, 72)
    base = m.utc_now().replace(microsecond=0) - timedelta(days=2)
    for d in range(0, 20):
        _fake_backup(cfg, base - timedelta(days=d), compressed=True)
    env = cfg.repo / ".env"
    env.write_text("SECRET=1")
    env.chmod(0o644)
    return [tmp_old, snap, env, *cfg.backup_dir.iterdir()]


def _owner_task_direct(cfg):
    def run(task, *args):
        dry = "--dry-run" in args
        if task == "backup-create":
            return m.create_backup(cfg, dry_run=dry)
        if task == "db-report":
            return m.db_report(cfg.live_db, checkpoint_threshold_bytes=cfg.wal_checkpoint_bytes, dry_run=dry)
        return m.db_table_retention(cfg.live_db, dry_run=dry)

    return run


def test_dry_run_deletes_nothing(cfg):
    files = _populate(cfg)
    before = {p: (p.stat().st_size, p.stat().st_mode) for p in files}
    out = m.run_maintenance(cfg, dry_run=True, owner_task=_owner_task_direct(cfg), reboot_fn=lambda: {"deferral_reason": "dry_run"})
    after = {p: (p.stat().st_size, p.stat().st_mode) for p in files}
    assert before == after
    assert not cfg.status_path.exists()
    assert out["temp"]["actions"] and out["backup_retention"]["actions"]
    assert all(a["dry_run"] for a in out["temp"]["actions"] + out["backup_retention"]["actions"] + out["permissions"])


def test_real_run_cleans_and_writes_status(cfg):
    files = _populate(cfg)
    out = m.run_maintenance(cfg, dry_run=False, owner_task=_owner_task_direct(cfg), reboot_fn=lambda: {"reboot_required": False, "deferral_reason": None})
    assert not files[0].exists()
    assert (cfg.logs_dir / "mystic_portfolio.log.pre_x.gz").exists()
    assert oct(files[2].stat().st_mode & 0o777) == "0o600"
    assert cfg.live_db.exists()
    assert out["backup_create"]["status"] == "ok"
    summary = m.maintenance_status_summary(cfg)
    assert summary["last_backup_verified_utc"] and summary["disk_used_pct"] is not None
    assert summary["reboot_required"] is False


def test_deploy_lock_makes_run_measure_only(cfg):
    files = _populate(cfg)
    cfg.deploy_locks[1].write_text("deploy")
    out = m.run_maintenance(cfg, dry_run=False, owner_task=_owner_task_direct(cfg), reboot_fn=lambda: {})
    assert out["skipped"].startswith("deploy_lock_held")
    assert all(p.exists() for p in files)


def test_deploy_lock_skip_keeps_the_verified_backup_stamp(cfg):
    """Ocean 2026-10-05 01:17: a deploy-lock skip replaced the status and cleared last_backup_verified_utc."""
    _live_db(cfg, rows=3)
    first = m.run_maintenance(cfg, dry_run=False, allow_reboot=False, owner_task=_owner_task_direct(cfg))
    stamped = first["backup_retention"]["newest_verified_utc"]
    assert stamped
    cfg.deploy_locks[1].write_text("deploy")
    skipped = m.run_maintenance(cfg, dry_run=False, allow_reboot=False, owner_task=_owner_task_direct(cfg))
    assert skipped["skipped"].startswith("deploy_lock_held")
    assert skipped["backup_retention"]["newest_verified_utc"] == stamped
    assert m.maintenance_status_summary(cfg)["last_backup_verified_utc"] == stamped


def test_status_reads_a_newer_catalog_stamp_than_the_stored_report(cfg):
    _write = {
        "finished_utc": "2026-10-05T01:17:01Z",
        "backup_retention": {"newest_verified_utc": "2026-10-04T20:29:27Z"},
        "disk_after": {"mode": "normal", "fs_used_pct": 73.0, "fs_free_gb": 12.0, "live_db_bytes": 0, "backup_bytes": 0},
    }
    cfg.status_path.write_text(json.dumps(_write))
    later = datetime(2026, 10, 5, 2, 40, tzinfo=timezone.utc)
    _fake_backup(cfg, later)
    summary = m.maintenance_status_summary(cfg)
    assert summary["last_backup_verified_utc"] == m._iso(later)


def test_open_orders_probe_loads_env_file_and_unknown_on_auth_failure(tmp_path, monkeypatch):
    import backend.services.live_readiness_service as lrs

    env = tmp_path / ".env"
    env.write_text("MAINT_TEST_PROBE_KEY=loaded\n")
    monkeypatch.delenv("MAINT_TEST_PROBE_KEY", raising=False)
    seen = {}

    async def ok():
        seen["key"] = os.environ.get("MAINT_TEST_PROBE_KEY")
        return {"binance_api_auth_status": "ok", "errors": [], "open_binance_orders_count": 2}

    monkeypatch.setattr(lrs, "_fetch_exchange_account_auth", ok)
    assert m.exchange_open_orders_count(env) == 2
    assert seen["key"] == "loaded"

    async def missing():
        return {"binance_api_auth_status": "missing_keys", "errors": ["missing"], "open_binance_orders_count": 0}

    monkeypatch.setattr(lrs, "_fetch_exchange_account_auth", missing)
    assert m.exchange_open_orders_count(env) is None
    monkeypatch.delenv("MAINT_TEST_PROBE_KEY", raising=False)
