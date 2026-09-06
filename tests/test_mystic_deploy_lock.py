"""Operations-only deploy lock and watchdog suppression.

These tests never start the real Mystic stack. The watchdog is invoked with
MYSTIC_WATCHDOG_START_CMD pointing at a marker-touching stub so a leak cannot
become a trading-process start.
"""

from __future__ import annotations

import os
import stat
import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
LOCK_SH = REPO / "scripts" / "mystic_deploy_lock.sh"
WATCHDOG = REPO / "watchdog_mystic.sh"


def _run(cmd: list[str], env: dict[str, str], cwd: Path | None = None) -> subprocess.CompletedProcess[str]:
    merged = os.environ.copy()
    merged.update(env)
    return subprocess.run(cmd, cwd=str(cwd or REPO), env=merged, text=True, capture_output=True, check=False)


def _lock_env(tmp_path: Path, **extra: str) -> dict[str, str]:
    env = {
        "MYSTIC_DEPLOY_LOCK": str(tmp_path / "deploy.lock"),
        "MYSTIC_MAINTENANCE_LOCK": str(tmp_path / "maintenance.lock"),
        "MYSTIC_WATCHDOG_LOG": str(tmp_path / "watchdog.log"),
        "MYSTIC_WATCHDOG_FLOCK": str(tmp_path / "flock"),
        "MYSTIC_WATCHDOG_REPO": str(tmp_path),
        "MYSTIC_WATCHDOG_START_CMD": f"touch '{tmp_path / 'started'}'",
    }
    env.update(extra)
    return env


def _log(tmp_path: Path) -> str:
    path = tmp_path / "watchdog.log"
    return path.read_text() if path.exists() else ""


def test_acquire_status_release_roundtrip(tmp_path: Path):
    env = _lock_env(tmp_path)
    acquired = _run([str(LOCK_SH), "acquire", "--sha", "abc123", "--reason", "test"], env)
    assert acquired.returncode == 0
    status = _run([str(LOCK_SH), "status"], env)
    assert status.returncode == 0
    assert "LOCKED" in status.stdout
    assert "abc123" in status.stdout
    assert "created_at" in status.stdout
    assert "operator" in status.stdout
    released = _run([str(LOCK_SH), "release"], env)
    assert released.returncode == 0
    assert _run([str(LOCK_SH), "status"], env).returncode == 1


def test_acquire_refuses_to_overwrite(tmp_path: Path):
    env = _lock_env(tmp_path)
    assert _run([str(LOCK_SH), "acquire", "--sha", "one"], env).returncode == 0
    second = _run([str(LOCK_SH), "acquire", "--sha", "two"], env)
    assert second.returncode == 1
    assert "abc123" not in (tmp_path / "deploy.lock").read_text()
    assert "one" in (tmp_path / "deploy.lock").read_text()
    assert "two" not in (tmp_path / "deploy.lock").read_text()


def test_status_is_read_only(tmp_path: Path):
    env = _lock_env(tmp_path)
    _run([str(LOCK_SH), "acquire", "--sha", "stay"], env)
    before = (tmp_path / "deploy.lock").read_text()
    _run([str(LOCK_SH), "status"], env)
    assert (tmp_path / "deploy.lock").read_text() == before


def test_no_auto_expiry_field(tmp_path: Path):
    env = _lock_env(tmp_path)
    _run([str(LOCK_SH), "acquire"], env)
    text = (tmp_path / "deploy.lock").read_text()
    assert "expires" not in text.lower()
    assert "ttl" not in text.lower()


def test_watchdog_no_lock_service_down_starts_stub(tmp_path: Path):
    env = _lock_env(tmp_path, MYSTIC_WATCHDOG_FORCE_MISSING="1")
    out = _run([str(WATCHDOG)], env)
    assert out.returncode == 0
    assert (tmp_path / "started").exists()
    assert "restarting core stack" in _log(tmp_path)
    assert "WATCHDOG_SUPPRESSED_DEPLOYMENT_LOCK" not in _log(tmp_path)


def test_watchdog_lock_plus_service_down_does_not_start(tmp_path: Path):
    env = _lock_env(tmp_path, MYSTIC_WATCHDOG_FORCE_MISSING="1")
    assert _run([str(LOCK_SH), "acquire", "--sha", "deploy-sha"], env).returncode == 0
    out = _run([str(WATCHDOG)], env)
    assert out.returncode == 0
    assert not (tmp_path / "started").exists()
    assert "WATCHDOG_SUPPRESSED_DEPLOYMENT_LOCK" in _log(tmp_path)


def test_watchdog_lock_plus_service_healthy_does_nothing(tmp_path: Path):
    env = _lock_env(tmp_path, MYSTIC_WATCHDOG_FORCE_MISSING="0")
    _run([str(LOCK_SH), "acquire"], env)
    _run([str(WATCHDOG)], env)
    assert not (tmp_path / "started").exists()
    assert "restarting core stack" not in _log(tmp_path)


def test_lock_removal_resumes_normal_watchdog(tmp_path: Path):
    env = _lock_env(tmp_path, MYSTIC_WATCHDOG_FORCE_MISSING="1")
    _run([str(LOCK_SH), "acquire"], env)
    _run([str(WATCHDOG)], env)
    assert not (tmp_path / "started").exists()
    _run([str(LOCK_SH), "release"], env)
    _run([str(WATCHDOG)], env)
    assert (tmp_path / "started").exists()


def test_legacy_maintenance_lock_also_suppresses(tmp_path: Path):
    env = _lock_env(tmp_path, MYSTIC_WATCHDOG_FORCE_MISSING="1")
    (tmp_path / "maintenance.lock").write_text("legacy\n")
    _run([str(WATCHDOG)], env)
    assert not (tmp_path / "started").exists()
    assert "WATCHDOG_SUPPRESSED_DEPLOYMENT_LOCK" in _log(tmp_path)


def test_malformed_directory_lock_fails_safe(tmp_path: Path):
    env = _lock_env(tmp_path, MYSTIC_WATCHDOG_FORCE_MISSING="1")
    (tmp_path / "deploy.lock").mkdir()
    _run([str(WATCHDOG)], env)
    assert not (tmp_path / "started").exists()
    assert "WATCHDOG_SUPPRESSED_MALFORMED_LOCK" in _log(tmp_path)


def test_malformed_unreadable_lock_fails_safe(tmp_path: Path):
    env = _lock_env(tmp_path, MYSTIC_WATCHDOG_FORCE_MISSING="1")
    lock = tmp_path / "deploy.lock"
    lock.write_text("{not-json")
    lock.chmod(stat.S_IWUSR)
    # Still a regular file; unreadable content is still a lock (fail safe).
    _run([str(WATCHDOG)], env)
    assert not (tmp_path / "started").exists()
    assert "WATCHDOG_SUPPRESSED_DEPLOYMENT_LOCK" in _log(tmp_path)


def test_lock_script_check_matches_watchdog(tmp_path: Path):
    env = _lock_env(tmp_path)
    assert _run([str(LOCK_SH), "check"], env).returncode == 1
    _run([str(LOCK_SH), "acquire"], env)
    check = _run([str(LOCK_SH), "check"], env)
    assert check.returncode == 0
    assert "WATCHDOG_SUPPRESSED_DEPLOYMENT_LOCK" in check.stdout


def test_watchdog_shared_flock_defaults_to_run_dir():
    text = WATCHDOG.read_text()
    assert "/run/mystic/watchdog.flock" in text
    assert "chmod 666" in text
    tmpfiles = REPO / "deploy" / "tmpfiles-mystic.conf"
    assert tmpfiles.is_file()
    assert "d /run/mystic 0775 mystic mystic" in tmpfiles.read_text()


def test_watchdog_is_not_a_trading_gate():
    text = WATCHDOG.read_text()
    assert "start_mystic.sh is intentionally NOT gated" in text
    assert "WATCHDOG_SUPPRESSED_DEPLOYMENT_LOCK" in text
    start = (REPO / "start_mystic.sh").read_text()
    assert "starting approved processes while watchdog is suppressed" in start
