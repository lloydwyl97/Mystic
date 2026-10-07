"""Production infrastructure maintenance: disk guard, backups, temp/log retention, WAL
hygiene and the safe-reboot gate.

Infrastructure only. Nothing here reads or changes strategy, entry/exit logic, sizing,
capital sleeves, ownership, accounting or breakers. The only writes to the live database
are the existing ``sqlite_large_table_retention`` policies (aggressive disk mode only) and a
PASSIVE WAL checkpoint. VACUUM is never run.

Safety invariants:
  * deletions are limited to explicit Mystic patterns and to backups this module created
    and verified; unknown files are reported, never removed
  * the live DB, the newest verified backup, pinned backups and any file open by a process
    are never deleted
  * a held deploy lock turns every run into measure-only
  * one run at a time (flock); ``dry_run`` performs no filesystem or database mutation
"""

from __future__ import annotations

import contextlib
import fcntl
import fnmatch
import gzip
import hashlib
import json
import logging
import os
import pwd
import re
import shutil
import sqlite3
import subprocess
import sys
import time
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger("mystic.maintenance")

GIB = 1024**3


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)) or default)
    except ValueError:
        return default


def _env_int(name: str, default: int) -> int:
    return int(_env_float(name, default))


@dataclass
class MaintConfig:
    repo: Path = Path(os.getenv("MYSTIC_REPO", "/home/mystic/mystic"))
    home: Path = Path(os.getenv("MYSTIC_HOME", "/home/mystic"))
    tmp_dir: Path = Path(os.getenv("MAINT_TMP_DIR", "/tmp"))
    lock_path: Path = Path(os.getenv("MAINT_LOCK_PATH", "/run/mystic/maintenance_job.lock"))
    deploy_locks: tuple[Path, ...] = (
        Path(os.getenv("MYSTIC_DEPLOY_LOCK", "/run/mystic/deploy.lock")),
        Path(os.getenv("MYSTIC_MAINTENANCE_LOCK", "/tmp/mystic_maintenance.lock")),
    )
    reboot_flag: Path = Path(os.getenv("MAINT_REBOOT_FLAG", "/var/run/reboot-required"))
    owner: str = os.getenv("MAINT_OWNER", "mystic")
    status_url: str = os.getenv("MAINT_STATUS_URL", "http://127.0.0.1:8000/api/portfolio-engine/status")
    task_health_url: str = os.getenv("MAINT_TASK_HEALTH_URL", "http://127.0.0.1:8000/api/system/task-health")
    # Backups: one per day is what a 4 GB DB on a 50 GB disk supports. The newest verified
    # backup stays uncompressed for fast recovery; older retained ones are gzipped.
    backup_interval_hours: float = _env_float("MAINT_BACKUP_INTERVAL_HOURS", 24)
    keep_hourly: int = _env_int("MAINT_BACKUP_KEEP_HOURLY", 24)
    keep_daily: int = _env_int("MAINT_BACKUP_KEEP_DAILY", 3)
    keep_weekly: int = _env_int("MAINT_BACKUP_KEEP_WEEKLY", 2)
    aggressive_keep_daily: int = _env_int("MAINT_BACKUP_AGGRESSIVE_KEEP_DAILY", 1)
    aggressive_keep_weekly: int = _env_int("MAINT_BACKUP_AGGRESSIVE_KEEP_WEEKLY", 0)
    backup_free_reserve_bytes: int = int(_env_float("MAINT_BACKUP_FREE_RESERVE_GB", 2.0) * GIB)
    reboot_backup_max_age_min: float = _env_float("MAINT_REBOOT_BACKUP_MAX_AGE_MIN", 60)
    temp_age_hours: float = _env_float("MAINT_TEMP_AGE_HOURS", 24)
    aggressive_temp_age_hours: float = _env_float("MAINT_AGGRESSIVE_TEMP_AGE_HOURS", 2)
    log_snapshot_compress_hours: float = _env_float("MAINT_LOG_SNAPSHOT_COMPRESS_HOURS", 24)
    log_snapshot_keep_days: float = _env_float("MAINT_LOG_SNAPSHOT_KEEP_DAYS", 14)
    aggressive_log_snapshot_keep_days: float = _env_float("MAINT_AGGRESSIVE_LOG_SNAPSHOT_KEEP_DAYS", 3)
    stale_log_days: float = _env_float("MAINT_STALE_LOG_DAYS", 14)
    normal_pct: float = _env_float("MAINT_DISK_NORMAL_PCT", 75)
    aggressive_pct: float = _env_float("MAINT_DISK_AGGRESSIVE_PCT", 85)
    critical_pct: float = _env_float("MAINT_DISK_CRITICAL_PCT", 92)
    wal_checkpoint_bytes: int = int(_env_float("MAINT_WAL_CHECKPOINT_MB", 256) * 1024**2)
    post_reboot_timeout_sec: float = _env_float("MAINT_POST_REBOOT_TIMEOUT_SEC", 900)
    core_process_patterns: tuple[str, ...] = (
        "venv/bin/python -m uvicorn backend.main:app",
        "start_live_market_data.py",
        "start_ai_signal_generator.py",
        "start_portfolio_engine_integration.py",
        "start_ai_market_context.py",
        "start_ai_learning.py",
    )

    @property
    def live_db(self) -> Path:
        return Path(os.getenv("MAINT_LIVE_DB", str(self.repo / "mystic_trading.db")))

    @property
    def backup_dir(self) -> Path:
        return Path(os.getenv("MAINT_BACKUP_DIR", str(self.home / "backups" / "db")))

    @property
    def backups_root(self) -> Path:
        return self.backup_dir.parent

    @property
    def logs_dir(self) -> Path:
        return self.repo / "logs"

    @property
    def status_path(self) -> Path:
        return Path(os.getenv("MAINT_STATUS_PATH", str(self.logs_dir / "maintenance_status.json")))

    @property
    def reboot_marker(self) -> Path:
        return self.logs_dir / "maintenance_reboot_marker.json"


# ---------------------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------------------


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _iso(dt: datetime | None) -> str | None:
    return dt.isoformat().replace("+00:00", "Z") if dt else None


def _action(actions: list[dict[str, Any]], kind: str, path: Path | str, *, dry_run: bool, **extra: Any) -> None:
    entry = {"action": kind, "path": str(path), "dry_run": dry_run, **extra}
    actions.append(entry)
    logger.info("MAINT %s %s", kind, json.dumps({k: v for k, v in entry.items() if k != "action"}, default=str))


def _size(path: Path) -> int:
    try:
        if path.is_symlink():
            return 0
        if path.is_file():
            return path.stat().st_size
        if path.is_dir():
            total = 0
            for root, _dirs, files in os.walk(path):
                for f in files:
                    with contextlib.suppress(OSError):
                        p = Path(root) / f
                        if not p.is_symlink():
                            total += p.stat().st_size
            return total
    except OSError:
        return 0
    return 0


def _newest_mtime(path: Path) -> float:
    try:
        newest = path.lstat().st_mtime
    except OSError:
        return time.time()
    if path.is_dir() and not path.is_symlink():
        for root, dirs, files in os.walk(path):
            for name in dirs + files:
                with contextlib.suppress(OSError):
                    newest = max(newest, (Path(root) / name).lstat().st_mtime)
    return newest


def open_paths(proc_root: Path = Path("/proc")) -> set[str]:
    """Real paths of every file currently open by any process we can inspect."""
    out: set[str] = set()
    try:
        pids = [p for p in proc_root.iterdir() if p.name.isdigit()]
    except OSError:
        return out
    for pid in pids:
        fd_dir = pid / "fd"
        try:
            fds = list(fd_dir.iterdir())
        except OSError:
            continue
        for fd in fds:
            with contextlib.suppress(OSError):
                target = str(fd.readlink())
                if target.startswith("/"):
                    out.add(target.removesuffix(" (deleted)"))
    return out


def _is_open(path: Path, opened: set[str]) -> bool:
    real = os.path.realpath(path)
    if real in opened:
        return True
    if path.is_dir():
        prefix = real.rstrip("/") + "/"
        return any(p.startswith(prefix) for p in opened)
    return False


def _remove(path: Path) -> None:
    if path.is_dir() and not path.is_symlink():
        shutil.rmtree(path)
    else:
        path.unlink()


def sha256_file(path: Path, *, opener: Callable[[Path], Any] | None = None) -> str:
    h = hashlib.sha256()
    with (opener or (lambda p: open(p, "rb")))(path) as fh:
        for chunk in iter(lambda: fh.read(4 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _write_json_atomic(path: Path, data: dict[str, Any], *, mode: int = 0o600) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with open(tmp, "w") as fh:
        json.dump(data, fh, indent=2, sort_keys=True, default=str)
        fh.flush()
        os.fsync(fh.fileno())
    tmp.chmod(mode)
    tmp.replace(path)


def _owner_ids(owner: str) -> tuple[int, int] | None:
    try:
        pw = pwd.getpwnam(owner)
    except KeyError:
        return None
    return pw.pw_uid, pw.pw_gid


def _chown(path: Path, owner: str) -> None:
    ids = _owner_ids(owner)
    if ids and os.geteuid() == 0:
        with contextlib.suppress(OSError):
            os.chown(path, *ids, follow_symlinks=False)


# ---------------------------------------------------------------------------------------
# Locks
# ---------------------------------------------------------------------------------------


class MaintenanceLockBusyError(RuntimeError):
    pass


@contextlib.contextmanager
def maintenance_lock(path: Path) -> Iterator[None]:
    """Exclusive non-blocking lock; raises ``MaintenanceLockBusyError`` if another run holds it."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fh = open(path, "a+")
    try:
        try:
            fcntl.flock(fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise MaintenanceLockBusyError(str(path)) from exc
        fh.seek(0)
        fh.truncate()
        fh.write(f"{os.getpid()} {_iso(utc_now())}\n")
        fh.flush()
        yield
    finally:
        with contextlib.suppress(OSError):
            fcntl.flock(fh.fileno(), fcntl.LOCK_UN)
        fh.close()


def deploy_lock_held(cfg: MaintConfig) -> Path | None:
    for p in cfg.deploy_locks:
        if p.exists() or p.is_symlink():
            return p
    return None


# ---------------------------------------------------------------------------------------
# Disk guard
# ---------------------------------------------------------------------------------------

LEGACY_BACKUP_GLOBS = ("mystic_trading.db.bak*", "mystic_trading.db.pre_*", "scalp_state_bak_*")


def legacy_backup_paths(cfg: MaintConfig) -> list[Path]:
    out: list[Path] = []
    for pattern in LEGACY_BACKUP_GLOBS:
        out.extend(sorted(cfg.home.glob(pattern)))
    return out


def disk_mode(used_pct: float, cfg: MaintConfig) -> str:
    if used_pct >= cfg.critical_pct:
        return "critical"
    if used_pct >= cfg.aggressive_pct:
        return "aggressive"
    return "normal"


def disk_snapshot(cfg: MaintConfig) -> dict[str, Any]:
    usage = shutil.disk_usage(cfg.live_db.parent if cfg.live_db.parent.exists() else "/")
    used_pct = 100.0 * usage.used / usage.total if usage.total else 0.0
    wal = Path(f"{cfg.live_db}-wal")
    backup_bytes = _size(cfg.backups_root) + sum(_size(p) for p in legacy_backup_paths(cfg))
    return {
        "measured_utc": _iso(utc_now()),
        "fs_total_bytes": usage.total,
        "fs_used_bytes": usage.used,
        "fs_free_bytes": usage.free,
        "fs_used_pct": round(used_pct, 2),
        "fs_free_gb": round(usage.free / GIB, 2),
        "live_db_bytes": _size(cfg.live_db),
        "live_wal_bytes": _size(wal),
        "backup_bytes": backup_bytes,
        "logs_bytes": _size(cfg.logs_dir),
        "temp_bytes": sum(_size(p) for p in iter_mystic_temp(cfg)),
        "mode": disk_mode(used_pct, cfg),
    }


# ---------------------------------------------------------------------------------------
# Permissions
# ---------------------------------------------------------------------------------------

CREDENTIAL_GLOBS_REPO = (".env", ".env.bak*", ".env.*.bak*", "core_only_local.env")
SENSITIVE_GLOBS_HOME = (
    "mystic_trading.db.*",
    "scalp_state_bak_*",
    "pre_audit_snapshot_*.json",
    "pre_deploy_*.json",
    "scalp_v2_label_backfill_pre_*.json",
    "*.env",
    ".env*",
)


def harden_permissions(cfg: MaintConfig, *, dry_run: bool) -> list[dict[str, Any]]:
    """Credential files and backups: owner-only (600 files / 700 dirs). Never loosens."""
    actions: list[dict[str, Any]] = []
    targets: list[tuple[Path, int]] = []
    for pattern in CREDENTIAL_GLOBS_REPO:
        targets += [(p, 0o600) for p in cfg.repo.glob(pattern) if p.is_file() and not p.is_symlink()]
    for pattern in SENSITIVE_GLOBS_HOME:
        targets += [(p, 0o600) for p in cfg.home.glob(pattern) if p.is_file() and not p.is_symlink()]
    if cfg.backups_root.is_dir():
        targets.append((cfg.backups_root, 0o700))
        for root, dirs, files in os.walk(cfg.backups_root):
            targets += [(Path(root) / d, 0o700) for d in dirs]
            targets += [(Path(root) / f, 0o600) for f in files]
    ids = _owner_ids(cfg.owner)
    for path, mode in targets:
        try:
            st = path.lstat()
        except OSError:
            continue
        if path.is_symlink():
            continue
        current = st.st_mode & 0o777
        if current & ~mode:
            _action(actions, "chmod", path, dry_run=dry_run, mode_from=oct(current), mode_to=oct(mode))
            if not dry_run:
                path.chmod(current & mode)
        if ids and os.geteuid() == 0 and (st.st_uid, st.st_gid) != ids:
            _action(actions, "chown", path, dry_run=dry_run, owner=cfg.owner)
            if not dry_run:
                _chown(path, cfg.owner)
    return actions


# ---------------------------------------------------------------------------------------
# Backups
# ---------------------------------------------------------------------------------------

BACKUP_RE = re.compile(r"^mystic_trading_(\d{8}T\d{6}Z)\.db(\.gz)?$")
PIN_FILE = "PINNED"


@dataclass
class BackupInfo:
    path: Path
    ts: datetime
    compressed: bool
    manifest: dict[str, Any] = field(default_factory=dict)

    @property
    def verified(self) -> bool:
        return self.manifest.get("integrity") == "ok" and bool(self.manifest.get("sha256"))

    @property
    def manifest_path(self) -> Path:
        return manifest_path_for(self.path)


def manifest_path_for(path: Path) -> Path:
    base = path.name.removesuffix(".gz")
    return path.with_name(f"{base}.json")


def backup_name(ts: datetime) -> str:
    return f"mystic_trading_{ts.strftime('%Y%m%dT%H%M%SZ')}.db"


def list_backups(backup_dir: Path) -> list[BackupInfo]:
    out: list[BackupInfo] = []
    if not backup_dir.is_dir():
        return out
    for p in backup_dir.iterdir():
        m = BACKUP_RE.match(p.name)
        if not m or not p.is_file():
            continue
        ts = datetime.strptime(m.group(1), "%Y%m%dT%H%M%SZ").replace(tzinfo=timezone.utc)
        manifest: dict[str, Any] = {}
        with contextlib.suppress(OSError, ValueError):
            manifest = json.loads(manifest_path_for(p).read_text())
        out.append(BackupInfo(p, ts, bool(m.group(2)), manifest))
    out.sort(key=lambda b: (b.ts, not b.compressed))
    return out


def pinned_names(backup_dir: Path) -> set[str]:
    names: set[str] = set()
    pin = backup_dir / PIN_FILE
    with contextlib.suppress(OSError):
        for raw in pin.read_text().splitlines():
            line = raw.split("#", 1)[0].strip()
            if line:
                names.add(line)
                names.add(line.removesuffix(".gz"))
                names.add(f"{line.removesuffix('.gz')}.gz")
    if backup_dir.is_dir():
        for p in backup_dir.glob("*.pin"):
            stem = p.name.removesuffix(".pin")
            names |= {stem, stem.removesuffix(".gz"), f"{stem.removesuffix('.gz')}.gz"}
    return names


def newest_verified(backups: Iterable[BackupInfo]) -> BackupInfo | None:
    verified = [b for b in backups if b.verified]
    return max(verified, key=lambda b: b.ts) if verified else None


def select_retained(
    backups: list[BackupInfo],
    *,
    now: datetime,
    keep_hourly: int,
    keep_daily: int,
    keep_weekly: int,
    pinned: set[str],
) -> set[Path]:
    """Generation policy. Returns the set of backup paths to keep.

    Always kept: newest verified, pinned, and anything unverified (never judged, never
    deleted). Then the newest per hour for ``keep_hourly`` hours, the newest per day for
    ``keep_daily`` distinct days and the newest per ISO week for ``keep_weekly`` weeks.
    """
    keep: set[Path] = set()
    newest = newest_verified(backups)
    if newest:
        keep.add(newest.path)
    for b in backups:
        if not b.verified or b.path.name in pinned:
            keep.add(b.path)
    verified = sorted((b for b in backups if b.verified), key=lambda b: b.ts, reverse=True)

    def _bucketed(key: Callable[[BackupInfo], Any], limit: int, candidates: list[BackupInfo]) -> None:
        seen: list[Any] = []
        for b in candidates:
            k = key(b)
            if k in seen:
                continue
            if len(seen) >= limit:
                break
            seen.append(k)
            keep.add(b.path)

    hourly = [b for b in verified if now - b.ts <= timedelta(hours=keep_hourly)]
    _bucketed(lambda b: b.ts.strftime("%Y%m%d%H"), keep_hourly, hourly)
    _bucketed(lambda b: b.ts.date(), keep_daily, verified)
    _bucketed(lambda b: tuple(b.ts.isocalendar())[:2], keep_weekly, verified)
    return keep


def sqlite_integrity(path: Path, *, immutable: bool = False) -> str:
    # immutable: never create -wal/-shm sidecars or take locks on a finished backup file.
    conn = sqlite3.connect(f"file:{path}?mode=ro{'&immutable=1' if immutable else ''}", uri=True)
    try:
        rows = conn.execute("PRAGMA integrity_check").fetchall()
    finally:
        conn.close()
    result = "; ".join(str(r[0]) for r in rows[:5])
    return "ok" if result == "ok" else result


def create_backup(cfg: MaintConfig, *, dry_run: bool, reason: str = "scheduled") -> dict[str, Any]:
    """Online SQLite backup of the live DB, integrity-checked, then atomically published."""
    db = cfg.live_db
    if not db.is_file():
        return {"status": "error", "error": f"live db missing: {db}"}
    db_bytes = db.stat().st_size + _size(Path(f"{db}-wal"))
    cfg.backup_dir.mkdir(parents=True, exist_ok=True)
    free = shutil.disk_usage(cfg.backup_dir).free
    need = int(db_bytes * 1.05) + cfg.backup_free_reserve_bytes
    if free < need:
        return {"status": "skipped", "reason": "insufficient_free_space", "free_bytes": free, "need_bytes": need}
    ts = utc_now().replace(microsecond=0)
    final = cfg.backup_dir / backup_name(ts)
    if dry_run:
        return {"status": "dry_run", "would_write": str(final), "estimated_bytes": db_bytes}
    partial = final.with_name(final.name + ".partial")
    started = time.monotonic()
    try:
        src = sqlite3.connect(str(db), timeout=60)
        dst = sqlite3.connect(str(partial))
        try:
            src.backup(dst)
            # A self-contained single file: no -wal/-shm sidecars appear when it is read
            # (the app re-enables WAL on its own connections after a restore).
            dst.execute("PRAGMA journal_mode=DELETE")
        finally:
            dst.close()
            src.close()
        partial.chmod(0o600)
        integrity = sqlite_integrity(partial)
        if integrity != "ok":
            partial.unlink(missing_ok=True)
            return {"status": "error", "error": f"integrity_check failed: {integrity[:200]}"}
        digest = sha256_file(partial)
        with open(partial, "rb") as fh:
            os.fsync(fh.fileno())
        manifest = {
            "name": final.name,
            "source": str(db),
            "created_utc": _iso(ts),
            "verified_utc": _iso(utc_now()),
            "integrity": "ok",
            "sha256": digest,
            "bytes": partial.stat().st_size,
            "compressed": False,
            "reason": reason,
        }
        _write_json_atomic(manifest_path_for(final), manifest)
        partial.replace(final)
        _chown(final, cfg.owner)
        _chown(manifest_path_for(final), cfg.owner)
    except Exception as exc:
        partial.unlink(missing_ok=True)
        return {"status": "error", "error": str(exc)[:300]}
    finally:
        for suffix in ("-wal", "-shm", "-journal"):
            Path(f"{partial}{suffix}").unlink(missing_ok=True)
    out = {"status": "ok", "path": str(final), "bytes": manifest["bytes"], "elapsed_sec": round(time.monotonic() - started, 1)}
    logger.info("MAINT backup_created %s", json.dumps(out))
    return out


ADOPT_MIN_AGE_SEC = 600
# Not created by this module. Once verified it follows the same keep/compress/delete
# policy as scheduled backups; a pin is the only way to keep one forever.
ADOPTED_REASON = "adopted_unverified"
DEPLOY_SNAPSHOT_REASON = "deploy_snapshot"
DEPLOY_SNAPSHOT_GLOB = "mystic_trading*.db"


def verify_unverified_backups(
    cfg: MaintConfig,
    *,
    dry_run: bool,
    opened: set[str] | None = None,
    limit: int = 1,
    now: datetime | None = None,
) -> dict[str, Any]:
    """Verify backups published without a manifest (e.g. a deploy-time ``.backup()``).

    A file that passes integrity_check gets a manifest and joins normal retention; a
    failing one is reported and left in place. Files still being written (recent mtime,
    open by a process) and compressed files are not touched. At most ``limit`` per run.
    """
    ref = (now or utc_now()).timestamp()
    pending = [b for b in list_backups(cfg.backup_dir) if not b.manifest and not b.compressed]
    out: dict[str, Any] = {"status": "ok", "unverified_found": len(pending), "verified": [], "failed": [], "deferred": []}
    for info in pending:
        if len(out["verified"]) + len(out["failed"]) >= limit:
            out["deferred"].append(info.path.name)
            continue
        if ref - info.path.stat().st_mtime < ADOPT_MIN_AGE_SEC or _is_open(info.path, opened if opened is not None else open_paths()):
            out["deferred"].append(info.path.name)
            continue
        if dry_run:
            out["verified"].append({"name": info.path.name, "dry_run": True})
            continue
        try:
            integrity = sqlite_integrity(info.path, immutable=True)
        except sqlite3.Error as exc:
            integrity = f"unreadable: {exc}"
        if integrity != "ok":
            out["failed"].append({"name": info.path.name, "integrity": integrity[:200]})
            out["status"] = "error"
            out["error"] = f"unverified backup failed integrity_check: {info.path.name}"
            continue
        manifest = {
            "name": info.path.name,
            "source": "external_backup_adopted",
            "created_utc": _iso(info.ts),
            "verified_utc": _iso(utc_now()),
            "integrity": "ok",
            "sha256": sha256_file(info.path),
            "bytes": info.path.stat().st_size,
            "compressed": False,
            "reason": ADOPTED_REASON,
        }
        _write_json_atomic(manifest_path_for(info.path), manifest)
        _chown(manifest_path_for(info.path), cfg.owner)
        out["verified"].append({"name": info.path.name, "sha256": manifest["sha256"]})
        logger.info("MAINT backup_verified %s", json.dumps({"name": info.path.name, "reason": ADOPTED_REASON}))
    return out


def deploy_snapshot_candidates(cfg: MaintConfig) -> list[Path]:
    """Raw DB snapshots left in the backups root (or one directory below) by deploys."""
    root = cfg.backups_root
    found: list[Path] = []
    if not root.is_dir():
        return found
    for entry in sorted(root.iterdir()):
        if entry.is_symlink() or entry == cfg.backup_dir:
            continue
        children = sorted(entry.iterdir()) if entry.is_dir() else [entry]
        found += [p for p in children if p.is_file() and not p.is_symlink() and fnmatch.fnmatch(p.name, DEPLOY_SNAPSHOT_GLOB)]
    live = cfg.live_db.resolve()
    return [p for p in found if p.resolve() != live]


def _checkpoint_snapshot(path: Path) -> None:
    """Fold a snapshot's WAL into the database so the file stands alone."""
    conn = sqlite3.connect(str(path), timeout=60)
    try:
        conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    finally:
        conn.close()


def _catalog_path(cfg: MaintConfig, ts: datetime) -> Path:
    while True:
        final = cfg.backup_dir / backup_name(ts)
        if not final.exists() and not Path(f"{final}.gz").exists() and not manifest_path_for(final).exists():
            return final
        ts += timedelta(seconds=1)


def adopt_deploy_snapshots(
    cfg: MaintConfig,
    *,
    dry_run: bool,
    opened: set[str] | None = None,
    limit: int = 1,
    now: datetime | None = None,
) -> dict[str, Any]:
    """Move verified deploy-time snapshots into the catalog so retention manages them.

    A non-empty WAL is checkpointed into the database, then the file is integrity
    checked, hashed, cataloged and compressed. The raw snapshot and its WAL are
    removed only after the gzip round trip matches. Recent, open or cross-filesystem
    files are skipped. Nothing is skipped only because a WAL exists.
    """
    ref = (now or utc_now()).timestamp()
    pending = deploy_snapshot_candidates(cfg)
    out: dict[str, Any] = {"status": "ok", "found": len(pending), "adopted": [], "failed": [], "deferred": [], "skipped": []}
    if pending and not dry_run:
        cfg.backup_dir.mkdir(parents=True, exist_ok=True)
    for path in pending:
        if len(out["adopted"]) + len(out["failed"]) >= limit:
            out["deferred"].append(str(path))
            continue
        st = path.stat()
        if ref - st.st_mtime < ADOPT_MIN_AGE_SEC or _is_open(path, opened if opened is not None else open_paths()):
            out["deferred"].append(str(path))
            continue
        wal = Path(f"{path}-wal")
        wal_pending = wal.exists() and wal.stat().st_size > 0
        if cfg.backup_dir.exists() and st.st_dev != cfg.backup_dir.stat().st_dev:
            out["skipped"].append({"path": str(path), "reason": "other_filesystem"})
            continue
        ts = datetime.fromtimestamp(int(st.st_mtime), tz=timezone.utc)
        final = _catalog_path(cfg, ts)
        if dry_run:
            out["adopted"].append({"origin": str(path), "name": final.name, "bytes": st.st_size, "wal": wal_pending, "dry_run": True})
            continue
        if wal_pending:
            try:
                _checkpoint_snapshot(path)
            except sqlite3.Error as exc:
                out["failed"].append({"path": str(path), "integrity": f"wal_checkpoint:{exc}"[:200]})
                out["status"] = "error"
                out["error"] = f"deploy snapshot wal checkpoint failed: {path}"
                continue
            if wal.exists() and wal.stat().st_size > 0:
                out["failed"].append({"path": str(path), "integrity": "wal_checkpoint_incomplete"})
                out["status"] = "error"
                out["error"] = f"deploy snapshot wal checkpoint incomplete: {path}"
                continue
        try:
            integrity = sqlite_integrity(path, immutable=True)
        except sqlite3.Error as exc:
            integrity = f"unreadable: {exc}"
        if integrity != "ok":
            out["failed"].append({"path": str(path), "integrity": integrity[:200]})
            out["status"] = "error"
            out["error"] = f"deploy snapshot failed integrity_check: {path}"
            continue
        digest = sha256_file(path)
        manifest = {
            "name": final.name,
            "source": DEPLOY_SNAPSHOT_REASON,
            "origin": str(path),
            "created_utc": _iso(ts),
            "verified_utc": _iso(utc_now()),
            "integrity": "ok",
            "sha256": digest,
            "bytes": path.stat().st_size,
            "compressed": False,
            "wal_checkpointed": wal_pending,
            "reason": DEPLOY_SNAPSHOT_REASON,
        }
        _write_json_atomic(manifest_path_for(final), manifest)
        path.rename(final)
        for suffix in ("-wal", "-shm"):
            Path(f"{path}{suffix}").unlink(missing_ok=True)
        _chown(final, cfg.owner)
        _chown(manifest_path_for(final), cfg.owner)
        if wal_pending:
            info = next((b for b in list_backups(cfg.backup_dir) if b.path == final), None)
            compressed = compress_backup(info, owner=cfg.owner) if info is not None else {"status": "error", "error": "catalog miss"}
            if compressed.get("status") != "ok":
                out["failed"].append({"path": str(final), "integrity": f"compress:{compressed.get('error')}"[:200]})
                out["status"] = "error"
                out["error"] = f"deploy snapshot compress failed: {final}"
                continue
        out["adopted"].append({"origin": str(path), "name": final.name, "sha256": digest, "bytes": manifest["bytes"], "compressed": wal_pending})
        logger.info("MAINT deploy_snapshot_adopted %s", json.dumps({"origin": str(path), "name": final.name, "compressed": wal_pending}))
    return out


def unmanaged_backups(cfg: MaintConfig) -> list[dict[str, Any]]:
    """Everything in the backups root outside the catalog, largest first, so nothing hides."""
    root = cfg.backups_root
    if not root.is_dir():
        return []
    now = time.time()
    rows = [{"name": p.name, "bytes": _size(p), "age_days": round((now - _newest_mtime(p)) / 86400.0, 2)} for p in root.iterdir() if p != cfg.backup_dir and not p.is_symlink()]
    return sorted(rows, key=lambda r: r["bytes"], reverse=True)


def compress_backup(info: BackupInfo, *, owner: str = "mystic") -> dict[str, Any]:
    """gzip a verified backup; the original is removed only after the round trip matches."""
    src = info.path
    gz = src.with_name(src.name + ".gz")
    partial = gz.with_name(gz.name + ".partial")
    expected = info.manifest.get("sha256")
    h = hashlib.sha256()
    try:
        with open(src, "rb") as fin, gzip.open(partial, "wb", compresslevel=6) as fout:
            for chunk in iter(lambda: fin.read(4 * 1024 * 1024), b""):
                h.update(chunk)
                fout.write(chunk)
        if h.hexdigest() != expected:
            partial.unlink(missing_ok=True)
            return {"status": "error", "error": "source changed since verification"}
        if sha256_file(partial, opener=lambda p: gzip.open(p, "rb")) != expected:
            partial.unlink(missing_ok=True)
            return {"status": "error", "error": "gzip round-trip mismatch"}
        partial.chmod(0o600)
        partial.replace(gz)
        _chown(gz, owner)
        manifest = dict(info.manifest)
        manifest.update({"compressed": True, "gz_bytes": gz.stat().st_size, "compressed_utc": _iso(utc_now())})
        _write_json_atomic(manifest_path_for(src), manifest)
        _chown(manifest_path_for(src), owner)
        src.unlink()
    except Exception as exc:
        partial.unlink(missing_ok=True)
        return {"status": "error", "error": str(exc)[:300]}
    return {"status": "ok", "path": str(gz), "bytes_before": info.manifest.get("bytes"), "bytes_after": gz.stat().st_size}


def apply_backup_retention(
    cfg: MaintConfig,
    *,
    mode: str,
    dry_run: bool,
    opened: set[str],
    now: datetime | None = None,
    compress: Callable[[BackupInfo], dict[str, Any]] | None = None,
) -> dict[str, Any]:
    backups = list_backups(cfg.backup_dir)
    aggressive = mode in ("aggressive", "critical")
    pinned = pinned_names(cfg.backup_dir)
    keep = select_retained(
        backups,
        now=now or utc_now(),
        keep_hourly=cfg.keep_hourly,
        keep_daily=cfg.aggressive_keep_daily if aggressive else cfg.keep_daily,
        keep_weekly=cfg.aggressive_keep_weekly if aggressive else cfg.keep_weekly,
        pinned=pinned,
    )
    newest = newest_verified(backups)
    actions: list[dict[str, Any]] = []
    reclaimed = 0
    errors: list[str] = []
    for b in backups:
        if b.path in keep or not b.verified or b.path.name in pinned or (newest and b.path == newest.path):
            continue
        if _is_open(b.path, opened):
            _action(actions, "skip_open_backup", b.path, dry_run=dry_run)
            continue
        size = _size(b.path)
        _action(actions, "delete_backup", b.path, dry_run=dry_run, bytes=size)
        if not dry_run:
            b.path.unlink(missing_ok=True)
            if not any(o.path != b.path and manifest_path_for(o.path) == b.manifest_path for o in backups):
                b.manifest_path.unlink(missing_ok=True)
        reclaimed += size
    for b in backups:
        if b.path not in keep or b.compressed or not b.verified or (newest and b.path == newest.path):
            continue
        if _is_open(b.path, opened):
            continue
        size = _size(b.path)
        _action(actions, "compress_backup", b.path, dry_run=dry_run, bytes=size)
        if dry_run:
            continue
        res = (compress or (lambda info: compress_backup(info, owner=cfg.owner)))(b)
        if res.get("status") == "ok":
            reclaimed += size - int(res.get("bytes_after") or 0)
        else:
            errors.append(f"compress {b.path.name}: {res.get('error')}")
    unmanaged = unmanaged_backups(cfg)
    return {
        "backups": len(backups),
        "unmanaged": unmanaged,
        "unmanaged_bytes": sum(r["bytes"] for r in unmanaged),
        "kept": sorted(p.name for p in keep),
        "pinned": sorted(n for n in pinned if BACKUP_RE.match(n)),
        "newest_verified": newest.path.name if newest else None,
        "newest_verified_utc": newest.manifest.get("verified_utc") if newest else None,
        "actions": actions,
        "bytes_reclaimed": reclaimed,
        "errors": errors,
    }


def backup_due(cfg: MaintConfig, *, now: datetime | None = None) -> bool:
    newest = newest_verified(list_backups(cfg.backup_dir))
    if newest is None:
        return True
    return (now or utc_now()) - newest.ts >= timedelta(hours=cfg.backup_interval_hours)


# ---------------------------------------------------------------------------------------
# Temp files
# ---------------------------------------------------------------------------------------

# Top-level names in /tmp that only Mystic tooling creates. Deliberately narrow: generic
# names (``st.json``, ``a_db.py``) are never matched automatically.
MYSTIC_TMP_PATTERNS: tuple[str, ...] = (
    "mystic_*",
    "mystic-*",
    "ocean_*",
    "scalp_*",
    "day_*",
    "replay_*",
    "rebuild_*",
    "pyc_check*",
    "micro_export*",
    "portfolio_pre_*",
)
# Never touched even when a pattern matches: lock files (the legacy deploy lock lives at
# /tmp/mystic_maintenance.lock), pid files, sockets and monitor control files.
MYSTIC_TMP_EXCLUDE: tuple[str, ...] = ("*.lock", "*.sock", "*.pid", "*_until_epoch")


def iter_mystic_temp(cfg: MaintConfig) -> Iterator[Path]:
    try:
        entries = list(cfg.tmp_dir.iterdir())
    except OSError:
        return
    excluded = {str(p) for p in cfg.deploy_locks}
    for p in entries:
        name = p.name
        if str(p) in excluded or any(fnmatch.fnmatch(name, x) for x in MYSTIC_TMP_EXCLUDE):
            continue
        if any(fnmatch.fnmatch(name, pat) for pat in MYSTIC_TMP_PATTERNS):
            yield p


def clean_temp(cfg: MaintConfig, *, mode: str, dry_run: bool, opened: set[str], now: float | None = None) -> dict[str, Any]:
    age_h = cfg.aggressive_temp_age_hours if mode in ("aggressive", "critical") else cfg.temp_age_hours
    cutoff = (now or time.time()) - age_h * 3600
    allowed_uids = {0}
    ids = _owner_ids(cfg.owner)
    if ids:
        allowed_uids.add(ids[0])
    allowed_uids.add(os.geteuid())
    actions: list[dict[str, Any]] = []
    reclaimed = 0
    for p in iter_mystic_temp(cfg):
        try:
            st = p.lstat()
        except OSError:
            continue
        if st.st_uid not in allowed_uids:
            continue
        if _newest_mtime(p) > cutoff:
            continue
        if _is_open(p, opened):
            _action(actions, "skip_open_temp", p, dry_run=dry_run)
            continue
        size = _size(p)
        _action(actions, "delete_temp", p, dry_run=dry_run, bytes=size)
        if not dry_run:
            with contextlib.suppress(OSError):
                _remove(p)
        reclaimed += size
    return {"age_hours": age_h, "actions": actions, "bytes_reclaimed": reclaimed}


# ---------------------------------------------------------------------------------------
# Logs (logrotate owns logs/*.log; this handles what it cannot see)
# ---------------------------------------------------------------------------------------

LOG_SNAPSHOT_GLOBS = ("*.pre_*", "*.prerestart*", "*.predeploy*")


def _gzip_file(src: Path, dst: Path) -> None:
    partial = dst.with_name(dst.name + ".partial")
    expected = sha256_file(src)
    with open(src, "rb") as fin, gzip.open(partial, "wb", compresslevel=6) as fout:
        shutil.copyfileobj(fin, fout, 4 * 1024 * 1024)
    if sha256_file(partial, opener=lambda p: gzip.open(p, "rb")) != expected:
        partial.unlink(missing_ok=True)
        raise OSError(f"gzip round-trip mismatch for {src}")
    shutil.copystat(src, partial)
    partial.replace(dst)
    src.unlink()


def apply_log_retention(cfg: MaintConfig, *, mode: str, dry_run: bool, opened: set[str], now: float | None = None) -> dict[str, Any]:
    now = now or time.time()
    keep_days = cfg.aggressive_log_snapshot_keep_days if mode in ("aggressive", "critical") else cfg.log_snapshot_keep_days
    actions: list[dict[str, Any]] = []
    reclaimed = 0
    errors: list[str] = []
    if not cfg.logs_dir.is_dir():
        return {"actions": actions, "bytes_reclaimed": 0, "errors": errors}
    candidates: set[Path] = set()
    for pattern in LOG_SNAPSHOT_GLOBS:
        candidates |= {p for p in cfg.logs_dir.glob(pattern) if p.is_file()}
    # Abandoned *.log files (process retired) that logrotate keeps forever because they
    # never grow past its size rule.
    for p in cfg.logs_dir.glob("*.log"):
        with contextlib.suppress(OSError):
            if p.is_file() and now - p.stat().st_mtime > cfg.stale_log_days * 86400 and not _is_open(p, opened):
                candidates.add(p)
    for p in sorted(candidates):
        try:
            mtime = p.stat().st_mtime
        except OSError:
            continue
        if _is_open(p, opened):
            continue
        size = _size(p)
        if p.name.endswith(".gz"):
            if now - mtime > keep_days * 86400:
                _action(actions, "delete_log_snapshot", p, dry_run=dry_run, bytes=size)
                if not dry_run:
                    p.unlink(missing_ok=True)
                reclaimed += size
            continue
        if now - mtime < cfg.log_snapshot_compress_hours * 3600:
            continue
        dst = p.with_name(p.name + ".gz") if not p.name.endswith(".log") else p.with_name(f"{p.name}.stale.gz")
        _action(actions, "compress_log", p, dry_run=dry_run, bytes=size)
        if dry_run:
            continue
        try:
            _gzip_file(p, dst)
            reclaimed += size - _size(dst)
        except Exception as exc:
            errors.append(f"{p.name}: {exc}")
    return {"keep_days": keep_days, "actions": actions, "bytes_reclaimed": reclaimed, "errors": errors}


# ---------------------------------------------------------------------------------------
# Database (WAL hygiene + existing table retention)
# ---------------------------------------------------------------------------------------


def db_report(db_path: Path, *, checkpoint_threshold_bytes: int, dry_run: bool) -> dict[str, Any]:
    out: dict[str, Any] = {"db": str(db_path)}
    wal = Path(f"{db_path}-wal")
    out["wal_bytes_before"] = _size(wal)
    conn = sqlite3.connect(str(db_path), timeout=30)
    try:
        conn.execute("PRAGMA busy_timeout=30000")
        for pragma in ("journal_mode", "wal_autocheckpoint", "page_size", "page_count", "freelist_count"):
            out[pragma] = conn.execute(f"PRAGMA {pragma}").fetchone()[0]
        if out["wal_bytes_before"] >= checkpoint_threshold_bytes:
            if dry_run:
                out["checkpoint"] = "would_run_passive"
            else:
                busy, log_frames, ckpt = conn.execute("PRAGMA wal_checkpoint(PASSIVE)").fetchone()
                out["checkpoint"] = {"mode": "PASSIVE", "busy": busy, "log_frames": log_frames, "checkpointed": ckpt}
    finally:
        conn.close()
    out["wal_bytes_after"] = _size(wal)
    return out


def db_table_retention(db_path: Path, *, dry_run: bool) -> dict[str, Any]:
    """Run the existing, documented retention policies once (no new tables, no new ages)."""
    from backend.services import sqlite_large_table_retention as r

    policy_tables = [p.table for p in r.RETENTION_POLICIES]

    def _counts() -> dict[str, int]:
        out: dict[str, int] = {}
        conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True, timeout=30)
        try:
            for t in policy_tables:
                if r._table_exists(conn, t):
                    out[t] = int(conn.execute(f'SELECT COUNT(*) FROM "{t}"').fetchone()[0])
            out["__page_count"] = int(conn.execute("PRAGMA page_count").fetchone()[0])
            out["__freelist_count"] = int(conn.execute("PRAGMA freelist_count").fetchone()[0])
        finally:
            conn.close()
        return out

    before = _counts()
    if dry_run:
        return {"dry_run": True, "rows_before": before, "would_run": "run_large_table_retention"}
    summary = r.run_large_table_retention(db_path)
    after = _counts()
    tables = {}
    for t in policy_tables:
        if t in before:
            removed = int((summary.get("tables", {}).get(t) or {}).get("deleted") or 0)
            tables[t] = {"rows_before": before[t], "removed": removed, "remaining": after.get(t)}
    return {
        "tables": tables,
        "pages_before": before.get("__page_count"),
        "pages_after": after.get("__page_count"),
        "freelist_after": after.get("__freelist_count"),
        "total_deleted": summary.get("total_deleted", 0),
        "vacuum": "not_run",
    }


# ---------------------------------------------------------------------------------------
# Safe reboot
# ---------------------------------------------------------------------------------------


def http_json(url: str, *, timeout: float = 15.0) -> tuple[int, dict[str, Any] | None]:
    import urllib.error
    import urllib.request

    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            return resp.status, json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        return exc.code, None
    except Exception:
        return 0, None


def exchange_open_orders_count(env_file: Path | None = None) -> int | None:
    """Open Binance.US orders via the existing readiness probe. ``None`` = unknown.

    Cron does not inherit the app environment, so credentials are loaded from the same
    ``.env`` start_mystic.sh sources (in-process only; never logged).
    """
    import asyncio

    try:
        if env_file is not None and env_file.is_file():
            from dotenv import load_dotenv

            load_dotenv(env_file, override=False)
        from backend.services.live_readiness_service import _fetch_exchange_account_auth

        res = asyncio.run(_fetch_exchange_account_auth())
    except Exception:
        return None
    if res.get("binance_api_auth_status") != "ok" or res.get("errors"):
        return None
    count = res.get("open_binance_orders_count")
    return int(count) if count is not None else None


def evaluate_reboot_gate(
    *,
    reboot_required: bool,
    deploy_lock: Path | None,
    status_code: int,
    status: dict[str, Any] | None,
    open_orders: int | None,
) -> list[str]:
    """Reasons a reboot must be deferred. Empty list = safe. Dust never counts."""
    reasons: list[str] = []
    if not reboot_required:
        reasons.append("reboot_not_required")
    if deploy_lock is not None:
        reasons.append(f"deploy_lock_held:{deploy_lock}")
    data = (status or {}).get("data") if isinstance(status, dict) else None
    if status_code != 200 or not isinstance(data, dict):
        reasons.append(f"status_unavailable:{status_code}")
        return reasons
    if int(data.get("positions_count") or 0) > 0:
        reasons.append(f"active_positions:{data.get('positions_count')}")
    if data.get("trailing_buy_intents"):
        reasons.append(f"trailing_buy_intents:{len(data.get('trailing_buy_intents') or [])}")
    if data.get("exit_blocked_positions"):
        reasons.append("exit_blocked_positions")
    if open_orders is None:
        reasons.append("open_orders_unknown")
    elif open_orders > 0:
        reasons.append(f"open_exchange_orders:{open_orders}")
    if str(data.get("account_status") or "").upper() != "HEALTHY":
        reasons.append(f"account_status:{data.get('account_status')}")
    if data.get("accounting_healthy") is not True:
        reasons.append("accounting_unhealthy")
    if data.get("equity_invariant_ok") is False:
        reasons.append("equity_invariant_failed")
    return reasons


def _systemctl_reboot() -> None:
    subprocess.run(["systemctl", "reboot"], check=False)


# Held back from unattended-upgrades (deploy/apt-52mystic-unattended-upgrades) because
# their maintainer scripts restart services Mystic depends on.
HELD_PACKAGES: tuple[str, ...] = ("redis-server", "redis-tools")


def held_upgrades_pending() -> list[str]:
    try:
        res = subprocess.run(
            ["apt-get", "-s", "install", "--only-upgrade", *HELD_PACKAGES],
            capture_output=True,
            text=True,
            timeout=120,
            check=False,
            env={**os.environ, "DEBIAN_FRONTEND": "noninteractive", "LC_ALL": "C"},
        )
    except Exception:
        return []
    return sorted({line.split()[1] for line in res.stdout.splitlines() if line.startswith("Inst ")})


def _upgrade_held_packages(pkgs: list[str]) -> dict[str, Any]:
    res = subprocess.run(
        ["apt-get", "install", "-y", "--only-upgrade", "-o", "Dpkg::Options::=--force-confold", *pkgs],
        capture_output=True,
        text=True,
        timeout=900,
        check=False,
        env={**os.environ, "DEBIAN_FRONTEND": "noninteractive", "NEEDRESTART_MODE": "l"},
    )
    return {"packages": pkgs, "rc": res.returncode, "tail": res.stdout.strip()[-300:]}


def maybe_reboot(
    cfg: MaintConfig,
    *,
    dry_run: bool,
    fetch_status: Callable[[], tuple[int, dict[str, Any] | None]] | None = None,
    open_orders: Callable[[], int | None] | None = None,
    make_backup: Callable[[], dict[str, Any]] | None = None,
    reboot: Callable[[], None] | None = None,
    held_pending: Callable[[], list[str]] | None = None,
    upgrade_held: Callable[[list[str]], dict[str, Any]] | None = None,
) -> dict[str, Any]:
    fetch_status = fetch_status or (lambda: http_json(cfg.status_url))
    open_orders = open_orders or (lambda: exchange_open_orders_count(cfg.repo / ".env"))
    held = (held_pending or held_upgrades_pending)()
    # Held packages are reported. They are not a reboot reason: their maintainer
    # scripts restart services the book depends on, and a pending upgrade of a
    # blacklisted package must not make the host reboot-eligible.
    required = cfg.reboot_flag.exists()
    out: dict[str, Any] = {"reboot_required": required, "reboot_flag": cfg.reboot_flag.exists(), "held_upgrades": held, "rebooted": False}
    if not required:
        out["deferral_reason"] = None
        return out

    def _gate() -> list[str]:
        code, status = fetch_status()
        return evaluate_reboot_gate(
            reboot_required=required,
            deploy_lock=deploy_lock_held(cfg),
            status_code=code,
            status=status,
            open_orders=open_orders(),
        )

    reasons = _gate()
    if reasons:
        out["deferral_reason"] = ",".join(reasons)
        logger.info("MAINT reboot_deferred %s", out["deferral_reason"])
        return out
    newest = newest_verified(list_backups(cfg.backup_dir))
    fresh = newest is not None and utc_now() - newest.ts <= timedelta(minutes=cfg.reboot_backup_max_age_min)
    if dry_run:
        out["deferral_reason"] = "dry_run"
        out["would_backup"] = not fresh
        return out
    if not fresh:
        res = (make_backup or (lambda: create_backup(cfg, dry_run=False, reason="pre_reboot")))()
        out["pre_reboot_backup"] = res
        if res.get("status") != "ok":
            out["deferral_reason"] = f"pre_reboot_backup_failed:{res.get('reason') or res.get('error')}"
            return out
    # Positions or orders could have appeared while the backup ran.
    reasons = _gate()
    if reasons:
        out["deferral_reason"] = ",".join(reasons)
        return out
    marker = {
        "requested_utc": _iso(utc_now()),
        "reason": "reboot-required",
        "pkgs": _read_text(Path(f"{cfg.reboot_flag}.pkgs")),
        "held_upgrades": held,
    }
    _write_json_atomic(cfg.reboot_marker, marker, mode=0o644)
    logger.warning("MAINT reboot_now %s", json.dumps(marker))
    out["rebooted"] = True
    out["deferral_reason"] = None
    (reboot or _systemctl_reboot)()
    return out


def _read_text(path: Path) -> str | None:
    try:
        return path.read_text().strip()[:2000]
    except OSError:
        return None


def _pgrep_count(pattern: str) -> int:
    try:
        res = subprocess.run(["pgrep", "-fc", pattern], capture_output=True, text=True, timeout=10, check=False)
        return int((res.stdout or "0").strip() or 0)
    except Exception:
        return 0


def _port_listening(port: int) -> bool:
    import socket

    with contextlib.suppress(OSError), socket.create_connection(("127.0.0.1", port), timeout=3):
        return True
    return False


def post_reboot_checks(cfg: MaintConfig) -> list[str]:
    """Failures (empty = healthy) for the post-reboot contract."""
    failures: list[str] = []
    for pattern in cfg.core_process_patterns:
        n = _pgrep_count(pattern)
        if n == 0:
            failures.append(f"process_missing:{pattern}")
        elif "uvicorn" in pattern and n != 1:
            failures.append(f"uvicorn_count:{n}")
    if not _port_listening(8000):
        failures.append("port_8000_not_listening")
    code, status = http_json(cfg.status_url)
    data = (status or {}).get("data") if status else None
    if code != 200 or not isinstance(data, dict):
        failures.append(f"status_http:{code}")
    else:
        live = data.get("always_live") or {}
        if str(live.get("execution_mode") or "").upper() != "LIVE":
            failures.append(f"execution_mode:{live.get('execution_mode')}")
        if str(data.get("kill_switch_mode") or "").upper() != "RESUME":
            failures.append(f"kill_switch:{data.get('kill_switch_mode')}")
        if str(data.get("account_status") or "").upper() != "HEALTHY":
            failures.append(f"account_status:{data.get('account_status')}")
        if data.get("accounting_healthy") is not True:
            failures.append("accounting_unhealthy")
    code, health = http_json(cfg.task_health_url)
    tasks = (health or {}).get("tasks") or {}
    for name in ("order_book_collector:ws_messages", "agg_trade_collector:ws_messages"):
        if (tasks.get(name) or {}).get("status") != "OK":
            failures.append(f"market_data_stale:{name}")
    if not _learning_running():
        failures.append("learning_not_running")
    return failures


def _learning_running() -> bool:
    try:
        import redis

        raw = redis.Redis(host="127.0.0.1", port=6379, socket_timeout=3).get("ai_learning_stats")
        stats = json.loads(raw) if raw else {}
    except Exception:
        return False
    return stats.get("status") == "active" and bool(stats.get("pipeline_running"))


def post_reboot_verify(cfg: MaintConfig, *, sleep: Callable[[float], None] = time.sleep) -> dict[str, Any]:
    marker = None
    with contextlib.suppress(OSError, ValueError):
        marker = json.loads(cfg.reboot_marker.read_text())
    deadline = time.monotonic() + cfg.post_reboot_timeout_sec
    failures: list[str] = ["not_checked"]
    while time.monotonic() < deadline:
        failures = post_reboot_checks(cfg)
        if not failures:
            break
        sleep(30)
    out = {"verified_utc": _iso(utc_now()), "maintenance_reboot": marker, "ok": not failures, "failures": failures}
    if failures:
        logger.critical("MAINT POST_REBOOT_VERIFY_FAILED %s", json.dumps(out))
    else:
        logger.info("MAINT post_reboot_verified %s", json.dumps(out))
        cfg.reboot_marker.unlink(missing_ok=True)
    return out


# ---------------------------------------------------------------------------------------
# Status
# ---------------------------------------------------------------------------------------


def read_status(cfg: MaintConfig | None = None) -> dict[str, Any]:
    cfg = cfg or MaintConfig()
    try:
        return json.loads(cfg.status_path.read_text())
    except (OSError, ValueError):
        return {}


def maintenance_status_summary(cfg: MaintConfig | None = None) -> dict[str, Any]:
    """Compact block for the portfolio status endpoint."""
    s = read_status(cfg)
    disk = s.get("disk_after") or s.get("disk_before") or {}
    backup = s.get("backup_retention") or {}
    reboot = s.get("reboot") or {}
    verified_utc = backup.get("newest_verified_utc")
    newest = newest_verified(list_backups(cfg.backup_dir))
    catalog_utc = newest.manifest.get("verified_utc") if newest else None
    if catalog_utc and (not verified_utc or str(catalog_utc) > str(verified_utc)):
        verified_utc = catalog_utc
    return {
        "last_maintenance_utc": s.get("finished_utc"),
        "dry_run": s.get("dry_run"),
        "disk_mode": disk.get("mode"),
        "disk_used_pct": disk.get("fs_used_pct"),
        "free_gb": disk.get("fs_free_gb"),
        "live_db_gb": round((disk.get("live_db_bytes") or 0) / GIB, 3) if disk else None,
        "backup_gb": round((disk.get("backup_bytes") or 0) / GIB, 3) if disk else None,
        "backup_unmanaged_gb": round((backup.get("unmanaged_bytes") or 0) / GIB, 3) if backup else None,
        "last_backup_verified_utc": verified_utc,
        "last_retention_run_utc": s.get("finished_utc") if backup else None,
        "bytes_reclaimed": s.get("bytes_reclaimed"),
        "reboot_required": reboot.get("reboot_required"),
        "reboot_deferral_reason": reboot.get("deferral_reason"),
        "last_post_reboot_verify": s.get("post_reboot_verify"),
        "errors": s.get("errors") or [],
    }


# ---------------------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------------------


def _run_owner_task(cfg: MaintConfig, task: str, *args: str) -> dict[str, Any]:
    """Run a DB task as the app owner so no root-owned -wal/-shm/backup files appear."""
    script = cfg.repo / "scripts" / "mystic_maintenance.py"
    cmd = [sys.executable, str(script), task, *args]
    if os.geteuid() == 0 and cfg.owner != "root":
        cmd = ["runuser", "-u", cfg.owner, "--", "nice", "-n", "19", "ionice", "-c3", *cmd]
    env = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1"}
    res = subprocess.run(cmd, capture_output=True, text=True, cwd=str(cfg.repo), env=env, check=False)
    try:
        return json.loads(res.stdout.strip().splitlines()[-1])
    except (IndexError, ValueError):
        return {"status": "error", "error": f"rc={res.returncode} {res.stderr.strip()[-300:]}"}


def run_maintenance(
    cfg: MaintConfig,
    *,
    dry_run: bool,
    allow_reboot: bool = True,
    owner_task: Callable[..., dict[str, Any]] | None = None,
    reboot_fn: Callable[..., dict[str, Any]] | None = None,
) -> dict[str, Any]:
    owner_task = owner_task or (lambda task, *a: _run_owner_task(cfg, task, *a))
    started = utc_now()
    report: dict[str, Any] = {"started_utc": _iso(started), "dry_run": dry_run, "errors": []}
    try:
        with maintenance_lock(cfg.lock_path):
            before = disk_snapshot(cfg)
            report["disk_before"] = before
            mode = before["mode"]
            if mode == "critical":
                logger.critical("MAINT DISK_CRITICAL used_pct=%s free_gb=%s", before["fs_used_pct"], before["fs_free_gb"])
            lock = deploy_lock_held(cfg)
            if lock is not None:
                report["skipped"] = f"deploy_lock_held:{lock}"
                report["disk_after"] = before
                return _finish(cfg, report, dry_run=dry_run)
            opened = open_paths()
            report["permissions"] = harden_permissions(cfg, dry_run=dry_run)
            report["temp"] = clean_temp(cfg, mode=mode, dry_run=dry_run, opened=opened)
            report["logs"] = apply_log_retention(cfg, mode=mode, dry_run=dry_run, opened=opened)
            report["errors"] += report["logs"].get("errors", [])
            if backup_due(cfg):
                report["backup_create"] = owner_task("backup-create", *(["--dry-run"] if dry_run else []))
                if report["backup_create"].get("status") == "error":
                    report["errors"].append(f"backup: {report['backup_create'].get('error')}")
                elif report["backup_create"].get("status") == "skipped":
                    report["errors"].append(f"backup_due_not_written: {report['backup_create'].get('reason')}")
            report["backup_verify"] = owner_task("backup-verify", *(["--dry-run"] if dry_run else []))
            if report["backup_verify"].get("status") == "error":
                report["errors"].append(f"backup_verify: {report['backup_verify'].get('error')}")
            report["backup_adopt"] = owner_task("backup-adopt", *(["--dry-run"] if dry_run else []))
            if report["backup_adopt"].get("status") == "error":
                report["errors"].append(f"backup_adopt: {report['backup_adopt'].get('error')}")
            # In-process (root) so files held open by any user's process are visible.
            report["backup_retention"] = apply_backup_retention(cfg, mode=mode, dry_run=dry_run, opened=opened)
            report["errors"] += report["backup_retention"].get("errors", [])
            report["db"] = owner_task("db-report", *(["--dry-run"] if dry_run else []))
            if mode in ("aggressive", "critical"):
                report["db_retention"] = owner_task("db-retention", *(["--dry-run"] if dry_run else []))
            after = disk_snapshot(cfg)
            report["disk_after"] = after
            report["bytes_reclaimed"] = max(0, after["fs_free_bytes"] - before["fs_free_bytes"]) if not dry_run else 0
            report["bytes_reclaimable_dry_run"] = sum(int((report.get(k) or {}).get("bytes_reclaimed") or 0) for k in ("temp", "logs", "backup_retention")) if dry_run else None
            if allow_reboot:
                report["reboot"] = (reboot_fn or (lambda: maybe_reboot(cfg, dry_run=dry_run, make_backup=lambda: owner_task("backup-create"))))()
            else:
                report["reboot"] = {"reboot_required": cfg.reboot_flag.exists(), "deferral_reason": "reboot_disabled_for_run"}
            return _finish(cfg, report, dry_run=dry_run)
    except MaintenanceLockBusyError:
        logger.info("MAINT skipped: another maintenance run holds %s", cfg.lock_path)
        return {"skipped": "maintenance_lock_busy", "dry_run": dry_run}


def _finish(cfg: MaintConfig, report: dict[str, Any], *, dry_run: bool) -> dict[str, Any]:
    report["finished_utc"] = _iso(utc_now())
    prev = read_status(cfg)
    if "post_reboot_verify" in prev and "post_reboot_verify" not in report:
        report["post_reboot_verify"] = prev["post_reboot_verify"]
    # A deploy-lock skip returns before retention runs. Replacing the status
    # file with that partial report cleared newest_verified_utc (Ocean 2026-10-05
    # 01:17) even though verified backups were still in the catalog.
    if "backup_retention" not in report and prev.get("backup_retention"):
        report["backup_retention"] = prev["backup_retention"]
    if not dry_run:
        _write_json_atomic(cfg.status_path, report, mode=0o644)
        _chown(cfg.status_path, cfg.owner)
    return report


def record_post_reboot(cfg: MaintConfig, result: dict[str, Any]) -> None:
    s = read_status(cfg)
    s["post_reboot_verify"] = result
    if not result.get("ok"):
        s.setdefault("errors", []).append("post_reboot_verify_failed")
    _write_json_atomic(cfg.status_path, s, mode=0o644)
    _chown(cfg.status_path, cfg.owner)
