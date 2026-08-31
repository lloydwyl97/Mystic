"""Process-level singleton via pidfile + exclusive flock.

Used by core workers so a second launch cannot become a second authority.
The flock is released automatically when the process exits.
"""

from __future__ import annotations

import atexit
import fcntl
import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)

_HELD_FDS: dict[str, int] = {}

SCALP_RUNNER_PIDFILE = "/tmp/mystic_scalp_runner.pid"
AI_MARKET_CONTEXT_PIDFILE = "/tmp/mystic_ai_market_context.pid"
AI_LEARNING_PIDFILE = "/tmp/mystic_ai_learning.pid"


class ProcessAlreadyRunningError(RuntimeError):
    """Another live process already holds the singleton lock."""


ProcessAlreadyRunning = ProcessAlreadyRunningError


def acquire_process_singleton(pid_file: str | Path, *, label: str) -> None:
    """Abort if another process already holds ``pid_file``.

    Keeps the file descriptor open so the exclusive flock survives for the
    life of this process. Stale pidfiles from crashed processes are reused.
    """
    path = Path(pid_file)
    path.parent.mkdir(parents=True, exist_ok=True)
    key = str(path.resolve()) if path.exists() else str(path)
    fd = os.open(str(path), os.O_CREAT | os.O_RDWR, 0o644)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        os.close(fd)
        holder = _read_pid(path)
        logger.error("ABORT: %s already running (pidfile=%s holder=%s)", label, path, holder)
        raise ProcessAlreadyRunningError(f"{label} already running (pidfile={path} holder={holder})") from exc
    os.ftruncate(fd, 0)
    os.write(fd, f"{os.getpid()}\n".encode())
    os.fsync(fd)
    _HELD_FDS[key] = fd

    def _release() -> None:
        held = _HELD_FDS.pop(key, None)
        if held is None:
            return
        try:
            fcntl.flock(held, fcntl.LOCK_UN)
        except OSError:
            pass
        os.close(held)
        try:
            if path.exists() and path.read_text().strip() == str(os.getpid()):
                path.unlink()
        except OSError:
            pass

    atexit.register(_release)


def _read_pid(path: Path) -> str:
    try:
        return path.read_text().strip() or "?"
    except OSError:
        return "?"
