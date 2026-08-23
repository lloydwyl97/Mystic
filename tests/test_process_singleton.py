"""Core-worker singleton: pidfile flock plus start_mystic.sh duplicate guards."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

REPO = Path("/home/mystic/mystic")


def test_second_process_cannot_acquire_same_pidfile(tmp_path: Path):
    from backend.utils.process_singleton import ProcessAlreadyRunning, acquire_process_singleton

    pidfile = tmp_path / "worker.pid"
    acquire_process_singleton(pidfile, label="test-worker")
    script = (
        "from backend.utils.process_singleton import ProcessAlreadyRunning, acquire_process_singleton\n"
        f"try:\n"
        f"    acquire_process_singleton({str(pidfile)!r}, label='child')\n"
        "except ProcessAlreadyRunning:\n"
        "    raise SystemExit(17)\n"
        "raise SystemExit(0)\n"
    )
    proc = subprocess.run([sys.executable, "-c", script], cwd=str(REPO), check=False)
    assert proc.returncode == 17


def test_start_mystic_has_lifecycle_lock_and_skip_guards():
    text = (REPO / "start_mystic.sh").read_text()
    assert "mystic_lifecycle.lock" in text
    assert "acquire_lifecycle_lock" in text
    assert "refuse_duplicate_or_collapse" in text
    assert text.count("9>&-") >= 6
    for pattern, label in (
        ("start_live_market_data.py", "Live Market Data"),
        ("start_ai_signal_generator.py", "AI Signal Generator"),
        ("start_portfolio_engine_integration.py", "Portfolio Engine Integration"),
        ("start_ai_market_context.py", "AI Market Context"),
        ("start_ai_learning.py", "AI Learning"),
        ("backend.services.binance_scalp.runner", "Scalp Paper Runner"),
    ):
        assert pattern in text
        assert f'refuse_duplicate_or_collapse "{pattern}"' in text
    stop = (REPO / "stop_mystic.sh").read_text()
    assert "mystic_lifecycle.lock" in stop
    assert "flock" in stop


def test_scalp_runner_and_context_and_learning_use_process_singleton():
    runner = (REPO / "backend/services/binance_scalp/runner.py").read_text()
    context = (REPO / "start_ai_market_context.py").read_text()
    learning = (REPO / "start_ai_learning.py").read_text()
    assert "acquire_process_singleton" in runner
    assert "SCALP_RUNNER_PIDFILE" in runner
    assert "acquire_process_singleton" in context
    assert "AI_MARKET_CONTEXT_PIDFILE" in context
    assert "acquire_process_singleton" in learning
    assert "AI_LEARNING_PIDFILE" in learning
