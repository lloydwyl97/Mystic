#!/usr/bin/env python3
"""Fast supervision for the six Mystic core processes.

The five-minute watchdog remains a backstop. This loop notices a dead
portfolio within one poll and restarts only that process when Redis and
SQLite are up. A dead market-data process is restarted the same way.
Three or more workers missing, or a missing portfolio while Redis is down,
still uses ``start_mystic.sh core``.

The loop does not place orders. Portfolio startup still blocks buys until
ownership is known and the first exit pass has run.
"""

from __future__ import annotations

import argparse
import contextlib
import fcntl
import os
import subprocess
import sys
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from pathlib import Path

POLL_SEC = float(os.getenv("CORE_SUPERVISOR_POLL_SEC", "2"))
STORM_LIMIT = 3
STORM_WINDOW_SEC = 120.0
BACKOFF_SEC = 60.0
FULL_CORE_MISSING = 3
BUSY_STALE_SEC = 90.0

# Highest priority first. Portfolio owns exit monitoring.
SERVICES: tuple[tuple[str, str, str], ...] = (
    ("portfolio", "start_portfolio_engine_integration.py", "portfolio"),
    ("market_data", "start_live_market_data.py", "live_md"),
    ("api", "venv/bin/python -m uvicorn backend.main:app", "backend"),
    ("signal", "start_ai_signal_generator.py", "signal"),
    ("context", "start_ai_market_context.py", "ai_context"),
    ("learning", "start_ai_learning.py", "learning"),
)


@dataclass
class Action:
    kind: str
    service: str = ""
    mode: str = ""
    events: list[str] = field(default_factory=list)


@dataclass
class SupervisorState:
    restarts: dict[str, list[float]] = field(default_factory=dict)
    backoff_until: dict[str, float] = field(default_factory=dict)
    lost_at: dict[str, float] = field(default_factory=dict)
    recovery_started: dict[str, float] = field(default_factory=dict)
    log_offset: int = 0


def service_names() -> tuple[str, ...]:
    return tuple(name for name, _pattern, _mode in SERVICES)


def choose_action(
    alive: Iterable[str],
    *,
    now: float,
    state: SupervisorState,
    redis_ok: bool,
    db_ok: bool,
    deploy_locked: bool,
    lifecycle_busy: bool,
    portfolio_exit_ready: bool = False,
) -> Action:
    """One decision. Does not start a process."""
    present = set(alive)
    missing = [name for name in service_names() if name not in present]
    events: list[str] = []
    for name in service_names():
        if name in present:
            if name in state.lost_at and name in state.recovery_started:
                if name == "portfolio" and not portfolio_exit_ready:
                    continue
                elapsed = now - state.lost_at[name]
                if name == "portfolio":
                    events.append(f"PORTFOLIO_EXIT_PROTECTION_RESTORED elapsed_s={elapsed:.3f}")
                state.lost_at.pop(name, None)
                state.recovery_started.pop(name, None)
            else:
                state.lost_at.pop(name, None)
        elif name not in state.lost_at:
            state.lost_at[name] = now
            if name == "portfolio":
                events.append("PORTFOLIO_PROCESS_LOST elapsed_s=0.000")
    if deploy_locked:
        events.append("CORE_SUPERVISOR_SKIP reason=deploy_lock")
        return Action("skip", events=events)
    if lifecycle_busy:
        events.append("CORE_SUPERVISOR_SKIP reason=lifecycle_lock")
        return Action("skip", events=events)
    if not missing:
        return Action("none", events=events)
    if not db_ok:
        events.append("CORE_SUPERVISOR_RESTART_FAILED reason=sqlite_missing")
        return Action("skip", events=events)
    portfolio_down = "portfolio" in missing
    if (not redis_ok and portfolio_down) or len(missing) >= FULL_CORE_MISSING:
        if _backing_off(state, "core", now):
            events.append("CORE_SUPERVISOR_RESTART_FAILED reason=restart_storm service=core")
            return Action("skip", events=events)
        events.append("CORE_SUPERVISOR_RESTART mode=core reason=dependency")
        return Action("full", service="core", mode="core", events=events)
    target = missing[0]
    if _backing_off(state, target, now):
        events.append(f"CORE_SUPERVISOR_RESTART_FAILED reason=restart_storm service={target}")
        return Action("skip", events=events)
    mode = next(mode for name, _pattern, mode in SERVICES if name == target)
    elapsed = now - state.lost_at.get(target, now)
    if target == "portfolio":
        events.append(f"PORTFOLIO_RECOVERY_STARTED mode=portfolio elapsed_s={elapsed:.3f}")
    else:
        events.append(f"CORE_SUPERVISOR_RESTART service={target} mode={mode} elapsed_s={elapsed:.3f}")
    return Action("targeted", service=target, mode=mode, events=events)


def note_restart(state: SupervisorState, service: str, now: float) -> None:
    stamps = [stamp for stamp in state.restarts.get(service, []) if now - stamp < STORM_WINDOW_SEC]
    stamps.append(now)
    state.restarts[service] = stamps
    state.recovery_started[service] = now
    if len(stamps) >= STORM_LIMIT:
        state.backoff_until[service] = now + BACKOFF_SEC


def _backing_off(state: SupervisorState, service: str, now: float) -> bool:
    return now < state.backoff_until.get(service, 0.0)


def portfolio_exit_ready(log_path: Path, offset: int) -> bool:
    if not log_path.is_file():
        return False
    with log_path.open("rb") as handle:
        handle.seek(0, os.SEEK_END)
        if handle.tell() < offset:
            offset = 0
        handle.seek(max(0, offset))
        return b"EXIT_MONITOR_LOOP" in handle.read()


def log_offset(log_path: Path) -> int:
    try:
        return log_path.stat().st_size
    except OSError:
        return 0


class Supervisor:
    def __init__(
        self,
        *,
        repo: Path,
        alive: Callable[[], set[str]],
        redis_ok: Callable[[], bool],
        db_ok: Callable[[], bool],
        deploy_locked: Callable[[], bool],
        lifecycle_busy: Callable[[], bool],
        start: Callable[[str], int],
        log: Callable[[str], None],
        portfolio_log: Path,
        busy_path: Path,
        poll_sec: float = POLL_SEC,
        sleep: Callable[[float], None] = time.sleep,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.repo = repo
        self.alive = alive
        self.redis_ok = redis_ok
        self.db_ok = db_ok
        self.deploy_locked = deploy_locked
        self.lifecycle_busy = lifecycle_busy
        self.start = start
        self.log = log
        self.portfolio_log = portfolio_log
        self.busy_path = busy_path
        self.poll_sec = poll_sec
        self.sleep = sleep
        self.clock = clock
        self.state = SupervisorState()

    def tick(self) -> Action:
        now = self.clock()
        action = choose_action(
            self.alive(),
            now=now,
            state=self.state,
            redis_ok=self.redis_ok(),
            db_ok=self.db_ok(),
            deploy_locked=self.deploy_locked(),
            lifecycle_busy=self.lifecycle_busy(),
            portfolio_exit_ready=portfolio_exit_ready(self.portfolio_log, self.state.log_offset),
        )
        for event in action.events:
            self.log(event)
        if action.kind in {"targeted", "full"}:
            self._restart(action, now)
        return action

    def _restart(self, action: Action, now: float) -> None:
        service = "core" if action.kind == "full" else action.service
        if service == "portfolio":
            self.state.log_offset = log_offset(self.portfolio_log)
        self.busy_path.parent.mkdir(parents=True, exist_ok=True)
        self.busy_path.write_text(f"{service} {now}\n")
        try:
            code = self.start(action.mode)
        finally:
            with contextlib.suppress(OSError):
                self.busy_path.unlink()
        note_restart(self.state, service, now)
        if code != 0:
            self.log(f"CORE_SUPERVISOR_RESTART_FAILED service={service} rc={code}")

    def run(self, *, ticks: int | None = None) -> None:
        done = 0
        while ticks is None or done < ticks:
            self.tick()
            done += 1
            if ticks is None or done < ticks:
                self.sleep(self.poll_sec)


def _pgrep_alive() -> set[str]:
    found: set[str] = set()
    for name, pattern, _mode in SERVICES:
        result = subprocess.run(["pgrep", "-f", pattern], capture_output=True, check=False)
        if result.returncode == 0 and result.stdout.strip():
            found.add(name)
    return found


def _redis_ok() -> bool:
    result = subprocess.run(["redis-cli", "ping"], capture_output=True, text=True, check=False)
    return result.returncode == 0 and "PONG" in (result.stdout or "")


def _db_ok(repo: Path) -> bool:
    return (repo / "mystic_trading.db").is_file()


def _path_exists(path: Path) -> bool:
    return path.exists()


def _lifecycle_busy(path: Path) -> bool:
    if not path.exists():
        return False
    try:
        handle = open(path, "a+")
    except OSError:
        return False
    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        handle.close()
        return True
    fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
    handle.close()
    return False


def _start_mode(repo: Path, mode: str) -> int:
    result = subprocess.run(["./start_mystic.sh", mode], cwd=str(repo), check=False)
    return result.returncode


def _emit(log_path: Path, message: str) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    with log_path.open("a") as handle:
        handle.write(f"{stamp} {message}\n")


def _lock(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = open(path, "a+")
    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        handle.close()
        return None
    return handle


def build(repo: Path) -> Supervisor:
    log_path = repo / "logs" / "core_supervisor.log"
    return Supervisor(
        repo=repo,
        alive=_pgrep_alive,
        redis_ok=_redis_ok,
        db_ok=lambda: _db_ok(repo),
        deploy_locked=lambda: (
            _path_exists(Path(os.getenv("MYSTIC_DEPLOY_LOCK", "/run/mystic/deploy.lock"))) or _path_exists(Path(os.getenv("MYSTIC_MAINTENANCE_LOCK", "/tmp/mystic_maintenance.lock")))
        ),
        lifecycle_busy=lambda: _lifecycle_busy(repo / "logs" / "mystic_lifecycle.lock"),
        start=lambda mode: _start_mode(repo, mode),
        log=lambda message: _emit(log_path, message),
        portfolio_log=repo / "logs" / "mystic_portfolio.log",
        busy_path=Path(os.getenv("MYSTIC_SUPERVISOR_BUSY", "/run/mystic/core_supervisor.busy")),
    )


def ensure(repo: Path) -> int:
    """Start the loop if this host does not already have one. Cron calls this."""
    held = _lock(Path(os.getenv("MYSTIC_SUPERVISOR_LOCK", "/run/mystic/core_supervisor.lock")))
    if held is None:
        return 0
    held.close()
    subprocess.Popen(
        [sys.executable, str(Path(__file__).resolve()), "run"],
        cwd=str(repo),
        start_new_session=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["run", "ensure"])
    parser.add_argument("--ticks", type=int, default=0)
    args = parser.parse_args(argv)
    repo = Path(__file__).resolve().parents[1]
    if args.command == "ensure":
        return ensure(repo)
    held = _lock(Path(os.getenv("MYSTIC_SUPERVISOR_LOCK", "/run/mystic/core_supervisor.lock")))
    if held is None:
        return 0
    try:
        build(repo).run(ticks=args.ticks or None)
    finally:
        held.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
