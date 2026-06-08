#!/usr/bin/env python3
"""
Short paper-only scalp calibration session (default 30 min max).

Sets SCALP_CALIBRATION_MODE + SCALP_PAPER_ENABLED for duration, then always tears down.
Never enables SCALP_LIVE. Writes only scalp tables.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import sqlite3
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
ENV_PATH = REPO / ".env"
sys.path.insert(0, str(REPO))

from backend.services.binance_scalp.session_cleanup import (  # noqa: E402
    clear_entry_armed,
    close_open_positions,
    read_env,
    restore_env_backup,
    restore_safe_env,
    set_env_keys,
    stop_scalp_runner,
    verify_safe_state,
)


def _scalp_summary(db_path: str) -> dict:
    if not Path(db_path).exists():
        return {"error": "no_db"}
    with sqlite3.connect(db_path) as conn:
        open_n = conn.execute(
            "SELECT COUNT(*) FROM scalp_paper_positions WHERE status='OPEN'"
        ).fetchone()[0]
        trades = conn.execute(
            "SELECT COUNT(*) FROM scalp_paper_trades WHERE created_at >= datetime('now','-2 hours')"
        ).fetchone()[0]
        sells = conn.execute(
            """
            SELECT COUNT(*), COALESCE(SUM(pnl_usd),0)
            FROM scalp_paper_trades
            WHERE side='SELL' AND created_at >= datetime('now','-2 hours')
              AND exit_reason NOT IN ('SCALP_TEST_ORPHAN_RESET','SCALP_TEST_SESSION_END_RESET')
            """
        ).fetchone()
        rejects = conn.execute(
            """
            SELECT reason, COUNT(*) FROM scalp_rejects
            WHERE created_at >= datetime('now','-2 hours')
            GROUP BY reason ORDER BY 2 DESC LIMIT 8
            """
        ).fetchall()
    return {
        "open_positions": open_n,
        "trades_2h": trades,
        "sells_2h": sells[0] if sells else 0,
        "net_pnl_2h": float(sells[1]) if sells else 0.0,
        "reject_reasons_2h": dict(rejects),
    }


def _teardown(
    db_path: str,
    *,
    backup: dict[str, str],
    proc: subprocess.Popen | None,
    grace_sec: int = 120,
) -> dict:
    """Stop runner, optional grace for strategy exit, then force safe state."""
    if proc is not None and proc.poll() is None:
        proc.send_signal(signal.SIGINT)
        try:
            proc.wait(timeout=15)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=5)

    open_before = _scalp_summary(db_path).get("open_positions", 0)
    if open_before > 0 and grace_sec > 0:
        venv_py = REPO / "venv" / "bin" / "python3"
        runner = REPO / "backend" / "services" / "binance_scalp" / "runner.py"
        grace_proc = subprocess.Popen(
            [str(venv_py), str(runner)],
            cwd=str(REPO),
            env={**os.environ, "PYTHONPATH": str(REPO)},
        )
        deadline = time.time() + grace_sec
        while time.time() < deadline:
            open_n = _scalp_summary(db_path).get("open_positions", 0)
            if open_n == 0:
                break
            time.sleep(5)
        if grace_proc.poll() is None:
            grace_proc.send_signal(signal.SIGINT)
            try:
                grace_proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                grace_proc.kill()

    stop_scalp_runner()
    restore_env_backup(backup)
    restore_safe_env()
    clear_entry_armed()
    closed = close_open_positions(db_path, reason="SCALP_TEST_SESSION_END_RESET")
    verify = verify_safe_state(db_path)
    return {
        "open_positions_before_teardown": open_before,
        "session_end_resets": closed,
        **verify,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--minutes", type=int, default=30)
    parser.add_argument(
        "--profile",
        choices=["strict", "moderate", "fast"],
        default="moderate",
    )
    parser.add_argument(
        "--replay-gate-pass",
        action="store_true",
        help="Required to start paper session when replay gate failed",
    )
    parser.add_argument("--grace-sec", type=int, default=120)
    args = parser.parse_args()

    replay_report = REPO / "scripts" / "scalp_strategy_replay_report.json"
    replay_gate_result = "unknown"
    if replay_report.exists():
        try:
            replay_gate_result = (
                "pass" if json.loads(replay_report.read_text()).get("replay_pass") else "fail"
            )
        except (json.JSONDecodeError, OSError):
            replay_gate_result = "unreadable"

    paper_session_started = False
    paper_session_reason = ""
    if replay_gate_result == "fail" and not args.replay_gate_pass:
        paper_session_reason = "skipped_replay_gate_failed"
        summary = {
            "started_utc": None,
            "ended_utc": datetime.now(timezone.utc).isoformat(),
            "paper_session_started": False,
            "paper_session_reason": paper_session_reason,
            "replay_gate_result": replay_gate_result,
            "runner_stopped": verify_safe_state(
                os.getenv("DATABASE_PATH", str(REPO / "mystic_trading.db"))
            ).get("runner_stopped"),
            "open_positions_end": verify_safe_state(
                os.getenv("DATABASE_PATH", str(REPO / "mystic_trading.db"))
            ).get("open_positions_end"),
            "safe_state_verified": verify_safe_state(
                os.getenv("DATABASE_PATH", str(REPO / "mystic_trading.db"))
            ).get("safe_state_verified"),
        }
        print(json.dumps(summary, indent=2, default=str))
        return 0

    duration_sec = max(60, min(args.minutes * 60, 1800))
    stale_sec = os.getenv("SCALP_STALE_TIMEOUT_SEC", "")
    backup = set_env_keys(
        {
            "SCALP_CALIBRATION_MODE": "true",
            "SCALP_CALIBRATION_PROFILE": args.profile,
            "SCALP_PAPER_ENABLED": "true",
            "SCALP_LIVE": "false",
            **({"SCALP_STALE_TIMEOUT_SEC": stale_sec} if stale_sec else {}),
        }
    )

    venv_py = REPO / "venv" / "bin" / "python3"
    runner = REPO / "backend" / "services" / "binance_scalp" / "runner.py"
    proc: subprocess.Popen | None = None
    started = datetime.now(timezone.utc).isoformat()
    paper_session_started = True
    paper_session_reason = "replay_gate_pass_or_override"
    db = os.getenv("DATABASE_PATH", str(REPO / "mystic_trading.db"))
    teardown: dict = {}

    try:
        proc = subprocess.Popen(
            [str(venv_py), str(runner)],
            cwd=str(REPO),
            env={**os.environ, "PYTHONPATH": str(REPO)},
        )
        time.sleep(duration_sec)
    except KeyboardInterrupt:
        paper_session_reason = "interrupted"
    finally:
        teardown = _teardown(db, backup=backup, proc=proc, grace_sec=args.grace_sec)

    summary = {
        "started_utc": started,
        "ended_utc": datetime.now(timezone.utc).isoformat(),
        "duration_sec": duration_sec,
        "profile": args.profile,
        "replay_gate_result": replay_gate_result,
        "paper_session_started": paper_session_started,
        "paper_session_reason": paper_session_reason,
        "runner_stopped": teardown.get("runner_stopped"),
        "open_positions_end": teardown.get("open_positions_end"),
        "safe_state_verified": teardown.get("safe_state_verified"),
        "teardown": teardown,
        "scalp_live": read_env().get("SCALP_LIVE", "false"),
        "scalp_paper_enabled": read_env().get("SCALP_PAPER_ENABLED", "false"),
        "calibration_mode": read_env().get("SCALP_CALIBRATION_MODE", "false"),
        "session": _scalp_summary(db),
    }
    print(json.dumps(summary, indent=2, default=str))
    return 0 if teardown.get("safe_state_verified") else 1


if __name__ == "__main__":
    raise SystemExit(main())
