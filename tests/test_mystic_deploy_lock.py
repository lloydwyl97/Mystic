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


def test_legacy_maintenance_lock_without_owner_does_not_suppress(tmp_path: Path):
    env = _lock_env(tmp_path, MYSTIC_WATCHDOG_FORCE_MISSING="1")
    (tmp_path / "maintenance.lock").write_text("legacy\n")
    _run([str(WATCHDOG)], env)
    assert (tmp_path / "started").exists()
    assert "DEPLOY_LOCK_OWNER_DEAD" in _log(tmp_path)
    assert not (tmp_path / "maintenance.lock").exists()


def test_malformed_directory_lock_does_not_suppress_forever(tmp_path: Path):
    env = _lock_env(tmp_path, MYSTIC_WATCHDOG_FORCE_MISSING="1")
    (tmp_path / "deploy.lock").mkdir()
    _run([str(WATCHDOG)], env)
    assert (tmp_path / "started").exists()
    assert "DEPLOY_LOCK_STALE" in _log(tmp_path)
    assert not (tmp_path / "deploy.lock").exists()


def test_malformed_unreadable_lock_does_not_suppress_forever(tmp_path: Path):
    env = _lock_env(tmp_path, MYSTIC_WATCHDOG_FORCE_MISSING="1")
    lock = tmp_path / "deploy.lock"
    lock.write_text("{not-json")
    lock.chmod(stat.S_IWUSR)
    _run([str(WATCHDOG)], env)
    assert (tmp_path / "started").exists()
    assert "DEPLOY_LOCK_STALE" in _log(tmp_path)


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
    assert "/tmp/mystic_watchdog.lock" not in text
    assert "WATCHDOG_CANONICAL_FLOCK_UNAVAILABLE" in text
    tmpfiles = REPO / "deploy" / "tmpfiles-mystic.conf"
    assert tmpfiles.is_file()
    assert "d /run/mystic 0775 mystic mystic" in tmpfiles.read_text()


def test_watchdog_same_canonical_flock_after_simulated_reboot(tmp_path):
    run_dir = tmp_path / "run" / "mystic"
    flock = run_dir / "watchdog.flock"
    assert not run_dir.exists()
    mystic_env = _lock_env(
        tmp_path,
        MYSTIC_WATCHDOG_FLOCK=str(flock),
        MYSTIC_WATCHDOG_FORCE_MISSING="0",
    )
    root_env = dict(mystic_env)
    mystic = _run([str(WATCHDOG)], mystic_env)
    root = _run([str(WATCHDOG)], root_env)
    assert mystic.returncode == 0
    assert root.returncode == 0
    assert flock.exists()
    assert "WATCHDOG_CANONICAL_FLOCK_UNAVAILABLE" not in _log(tmp_path)
    assert "/tmp/mystic_watchdog.lock" not in _log(tmp_path)


def test_watchdog_canonical_flock_unavailable_fails_closed(tmp_path):
    blocker = tmp_path / "not_a_dir"
    blocker.write_text("x")
    flock = blocker / "watchdog.flock"
    env = _lock_env(tmp_path, MYSTIC_WATCHDOG_FLOCK=str(flock), MYSTIC_WATCHDOG_FORCE_MISSING="1")
    out = _run([str(WATCHDOG)], env)
    assert out.returncode == 1
    assert "WATCHDOG_CANONICAL_FLOCK_UNAVAILABLE" in (out.stderr + _log(tmp_path))
    assert not (tmp_path / "started").exists()


def test_parallel_acquire_exactly_one_winner(tmp_path):
    env = _lock_env(tmp_path)
    first = subprocess.Popen(
        [str(LOCK_SH), "acquire", "--sha", "one"],
        env={**os.environ, **env},
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    second = subprocess.Popen(
        [str(LOCK_SH), "acquire", "--sha", "two"],
        env={**os.environ, **env},
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    codes = sorted([first.wait(), second.wait()])
    assert codes == [0, 1]
    text = (tmp_path / "deploy.lock").read_text()
    assert ("one" in text) ^ ("two" in text)
    assert _run([str(LOCK_SH), "status"], env).returncode == 0
    assert _run([str(LOCK_SH), "release"], env).returncode == 0
    assert _run([str(LOCK_SH), "status"], env).returncode == 1


def _live_owner() -> subprocess.Popen[str]:
    return subprocess.Popen(["sleep", "300"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def test_sigkill_owner_is_recovered_and_supervision_resumes(tmp_path: Path):
    owner = _live_owner()
    try:
        env = _lock_env(tmp_path, MYSTIC_DEPLOY_OWNER_PID=str(owner.pid), MYSTIC_WATCHDOG_FORCE_MISSING="1")
        acquired = _run([str(LOCK_SH), "acquire", "--sha", "deadbeef"], env)
        assert acquired.returncode == 0
        assert "DEPLOY_LOCK_ACQUIRED" in acquired.stdout
        text = (tmp_path / "deploy.lock").read_text()
        assert f'"pid": "{owner.pid}"' in text
        assert "starttime_ticks" in text
        assert "deadbeef" in text
        owner.kill()
        owner.wait(timeout=5)
        owner = None
        check = _run([str(LOCK_SH), "check"], env)
        assert check.returncode == 3
        assert "DEPLOY_LOCK_OWNER_DEAD" in check.stdout
        assert "DEPLOY_LOCK_RECOVERED" in check.stdout
        assert not (tmp_path / "deploy.lock").exists()
        watched = _run([str(WATCHDOG)], env)
        assert watched.returncode == 0
        assert (tmp_path / "started").exists()
    finally:
        if owner is not None and owner.poll() is None:
            owner.kill()


def test_shell_loss_leaves_the_lock_until_check(tmp_path: Path):
    env = _lock_env(tmp_path)
    script = f"'{LOCK_SH}' acquire --reason shell-loss; kill -9 $$"
    subprocess.run(["bash", "-c", script], env={**os.environ, **env}, check=False)
    assert (tmp_path / "deploy.lock").is_file()
    check = _run([str(LOCK_SH), "check"], env)
    assert check.returncode == 3
    assert not (tmp_path / "deploy.lock").exists()


def test_pid_reuse_with_a_different_start_time_is_not_the_owner(tmp_path: Path):
    owner = _live_owner()
    try:
        env = _lock_env(tmp_path, MYSTIC_DEPLOY_OWNER_PID=str(owner.pid))
        assert _run([str(LOCK_SH), "acquire"], env).returncode == 0
        text = (tmp_path / "deploy.lock").read_text()
        current = text.split('"starttime_ticks": "')[1].split('"')[0]
        (tmp_path / "deploy.lock").write_text(text.replace(f'"starttime_ticks": "{current}"', '"starttime_ticks": "1"'))
        held = _run([str(LOCK_SH), "check"], env)
        assert held.returncode == 3
        assert owner.poll() is None
        assert not (tmp_path / "deploy.lock").exists()
    finally:
        if owner.poll() is None:
            owner.kill()


def test_live_owner_blocks_the_watchdog_and_a_second_acquire(tmp_path: Path):
    owner = _live_owner()
    try:
        env = _lock_env(tmp_path, MYSTIC_DEPLOY_OWNER_PID=str(owner.pid), MYSTIC_WATCHDOG_FORCE_MISSING="1")
        assert _run([str(LOCK_SH), "acquire", "--sha", "aaa"], env).returncode == 0
        second = _run([str(LOCK_SH), "acquire", "--sha", "bbb"], env)
        assert second.returncode == 1
        assert "bbb" not in (tmp_path / "deploy.lock").read_text()
        assert _run([str(LOCK_SH), "check"], env).returncode == 0
        watched = _run([str(WATCHDOG)], env)
        assert watched.returncode == 0
        assert not (tmp_path / "started").exists()
        assert "WATCHDOG_SUPPRESSED_DEPLOYMENT_LOCK" in _log(tmp_path)
    finally:
        if owner.poll() is None:
            owner.kill()


def test_stale_lock_does_not_block_a_portfolio_restart_decision(tmp_path: Path):
    env = _lock_env(tmp_path)
    (tmp_path / "deploy.lock").write_text("{")
    check = _run([str(LOCK_SH), "check"], env)
    assert check.returncode == 3
    from scripts.core_supervisor import SupervisorState, choose_action

    action = choose_action(
        {"market_data", "api", "signal", "context", "learning"},
        now=10.0,
        state=SupervisorState(),
        redis_ok=True,
        db_ok=True,
        deploy_locked=False,
        lifecycle_busy=False,
    )
    assert action.mode == "portfolio"


def test_watchdog_is_not_a_trading_gate():
    text = WATCHDOG.read_text()
    assert "start_mystic.sh is intentionally NOT gated" in text
    assert "WATCHDOG_SUPPRESSED_DEPLOYMENT_LOCK" in text
    start = (REPO / "start_mystic.sh").read_text()
    assert "starting approved processes while watchdog is suppressed" in start
