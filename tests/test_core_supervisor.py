"""Fast core supervisor. These tests never start the production stack."""

from __future__ import annotations

import os
import sqlite3
import subprocess
import time
from pathlib import Path

import pytest

from scripts.core_supervisor import POLL_SEC, Supervisor, SupervisorState, choose_action, portfolio_exit_ready

REPO = Path(__file__).resolve().parents[1]
ALL = {"portfolio", "market_data", "api", "signal", "context", "learning"}


def _decide(alive, **kwargs):
    state = kwargs.pop("state", SupervisorState())
    return choose_action(
        alive,
        now=kwargs.pop("now", 100.0),
        state=state,
        redis_ok=kwargs.pop("redis_ok", True),
        db_ok=kwargs.pop("db_ok", True),
        deploy_locked=kwargs.pop("deploy_locked", False),
        lifecycle_busy=kwargs.pop("lifecycle_busy", False),
        portfolio_exit_ready=kwargs.pop("portfolio_exit_ready", False),
    )


def _book(path: Path) -> None:
    conn = sqlite3.connect(path)
    conn.execute("CREATE TABLE portfolio_engine_positions (symbol TEXT, engine_id TEXT, status TEXT, quantity REAL)")
    conn.execute("CREATE TABLE strategy_order_intents (side TEXT, status TEXT, symbol TEXT, filled_qty REAL)")
    conn.execute("INSERT INTO portfolio_engine_positions VALUES ('BTC/USDT', 'DAY_V2', 'ACTIVE', 0.01)")
    conn.commit()
    conn.close()


def test_portfolio_death_restarts_only_portfolio():
    action = _decide(ALL - {"portfolio"})
    assert action.kind == "targeted"
    assert action.mode == "portfolio"
    assert any(event.startswith("PORTFOLIO_PROCESS_LOST") for event in action.events)
    assert any(event.startswith("PORTFOLIO_RECOVERY_STARTED") for event in action.events)


def test_market_data_death_is_targeted():
    action = _decide(ALL - {"market_data"})
    assert action.mode == "live_md"


def test_learning_death_does_not_restart_portfolio():
    action = _decide(ALL - {"learning"})
    assert action.mode == "learning"
    assert action.kind == "targeted"


def test_redis_down_with_portfolio_down_uses_full_core():
    action = _decide(ALL - {"portfolio"}, redis_ok=False)
    assert action.kind == "full"
    assert action.mode == "core"


def test_three_missing_workers_use_full_core():
    action = _decide(ALL - {"portfolio", "market_data", "api"})
    assert action.kind == "full"


def test_missing_database_does_not_spawn():
    action = _decide(ALL - {"portfolio"}, db_ok=False)
    assert action.kind == "skip"
    assert any("sqlite_missing" in event for event in action.events)


def test_deploy_lock_and_lifecycle_lock_suppress_restart():
    assert _decide(ALL - {"portfolio"}, deploy_locked=True).kind == "skip"
    assert _decide(ALL - {"portfolio"}, lifecycle_busy=True).kind == "skip"


def test_restart_storm_backs_off():
    state = SupervisorState()
    state.restarts["portfolio"] = [90.0, 95.0, 99.0]
    state.backoff_until["portfolio"] = 160.0
    action = _decide(ALL - {"portfolio"}, state=state, now=120.0)
    assert action.kind == "skip"
    assert any("restart_storm" in event for event in action.events)


def test_exit_protection_event_waits_for_the_monitor_line(tmp_path: Path):
    log = tmp_path / "mystic_portfolio.log"
    log.write_text("old EXIT_MONITOR_LOOP\n")
    state = SupervisorState(lost_at={"portfolio": 10.0}, recovery_started={"portfolio": 12.0}, log_offset=log.stat().st_size)
    waiting = _decide(ALL, state=state, now=30.0, portfolio_exit_ready=False)
    assert not any("EXIT_PROTECTION_RESTORED" in event for event in waiting.events)
    assert "portfolio" in state.lost_at
    log.write_text("old EXIT_MONITOR_LOOP\nEXIT_MONITOR_LOOP interval_sec=45\n")
    ready = portfolio_exit_ready(log, state.log_offset)
    restored = _decide(ALL, state=state, now=31.0, portfolio_exit_ready=ready)
    assert any(event.startswith("PORTFOLIO_EXIT_PROTECTION_RESTORED") for event in restored.events)


def test_supervision_does_not_touch_open_lots(tmp_path: Path):
    db = tmp_path / "mystic_trading.db"
    _book(db)
    conn = sqlite3.connect(db)
    conn.execute("INSERT INTO portfolio_engine_positions VALUES ('BTC/USDT', 'SCALP_V2', 'ACTIVE', 0.02)")
    conn.execute("INSERT INTO strategy_order_intents VALUES ('SELL', 'WORKING', 'BTC/USDT', 0.01)")
    conn.execute("INSERT INTO strategy_order_intents VALUES ('BUY', 'ACKED', 'ETH/USDT', 0)")
    conn.execute("INSERT INTO strategy_order_intents VALUES ('BUY', 'PARTIAL', 'SOL/USDT', 0.5)")
    conn.execute("INSERT INTO strategy_order_intents VALUES ('SELL', 'FEE_PENDING', 'XRP/USDT', 10)")
    conn.commit()
    before = db.read_bytes()
    action = _decide(ALL - {"portfolio"})
    assert action.mode == "portfolio"
    assert db.read_bytes() == before
    rows = sqlite3.connect(db).execute("SELECT engine_id, status, COUNT(*) FROM portfolio_engine_positions GROUP BY 1, 2").fetchall()
    assert ("DAY_V2", "ACTIVE", 1) in rows
    assert ("SCALP_V2", "ACTIVE", 1) in rows


def test_measured_detection_is_within_one_poll(tmp_path: Path):
    """Wall-clock: a portfolio that dies between ticks is acted on at the next poll."""
    alive = set(ALL)
    started: list[tuple[float, str]] = []
    death = {"at": 0.0}

    def current() -> set[str]:
        if death["at"] and time.monotonic() >= death["at"]:
            return alive - {"portfolio"}
        return set(alive)

    supervisor = Supervisor(
        repo=tmp_path,
        alive=current,
        redis_ok=lambda: True,
        db_ok=lambda: True,
        deploy_locked=lambda: False,
        lifecycle_busy=lambda: False,
        start=lambda mode: started.append((time.monotonic(), mode)) or 0,
        log=lambda _message: None,
        portfolio_log=tmp_path / "mystic_portfolio.log",
        busy_path=tmp_path / "busy",
        poll_sec=0.05,
        clock=time.monotonic,
    )
    supervisor.tick()
    death["at"] = time.monotonic()
    deadline = time.monotonic() + 1.0
    while time.monotonic() < deadline and not started:
        supervisor.sleep(supervisor.poll_sec)
        supervisor.tick()
    assert started
    detected = started[0][0] - death["at"]
    assert started[0][1] == "portfolio"
    assert detected <= POLL_SEC + 0.2


def test_repeated_crashes_do_not_overlap(tmp_path: Path):
    calls = {"n": 0}

    def start(_mode: str) -> int:
        calls["n"] += 1
        if calls["n"] > 1:
            raise AssertionError("overlapping start")
        return 0

    supervisor = Supervisor(
        repo=tmp_path,
        alive=lambda: ALL - {"portfolio"},
        redis_ok=lambda: True,
        db_ok=lambda: True,
        deploy_locked=lambda: False,
        lifecycle_busy=lambda: False,
        start=start,
        log=lambda _message: None,
        portfolio_log=tmp_path / "log",
        busy_path=tmp_path / "busy",
        clock=lambda: 50.0 + calls["n"],
    )
    supervisor.tick()
    supervisor.state.backoff_until["portfolio"] = 10_000.0
    supervisor.tick()
    assert calls["n"] == 1
    assert not (tmp_path / "busy").exists()


def test_boot_cron_keeps_the_45_second_delay_and_starts_the_supervisor():
    text = (REPO / "deploy" / "cron.mystic-core-supervisor").read_text()
    assert "@reboot sleep 45 &&" in text
    assert "scripts/core_supervisor.py ensure" in text
    assert "sleep 45" in text


def test_watchdog_defers_while_the_fast_supervisor_is_restarting(tmp_path: Path):
    env = os.environ.copy()
    env.update(
        {
            "MYSTIC_DEPLOY_LOCK": str(tmp_path / "deploy.lock"),
            "MYSTIC_MAINTENANCE_LOCK": str(tmp_path / "maintenance.lock"),
            "MYSTIC_WATCHDOG_LOG": str(tmp_path / "watchdog.log"),
            "MYSTIC_WATCHDOG_FLOCK": str(tmp_path / "flock"),
            "MYSTIC_WATCHDOG_REPO": str(tmp_path),
            "MYSTIC_WATCHDOG_START_CMD": f"touch '{tmp_path / 'started'}'",
            "MYSTIC_WATCHDOG_FORCE_MISSING": "1",
            "MYSTIC_SUPERVISOR_BUSY": str(tmp_path / "busy"),
        }
    )
    (tmp_path / "busy").write_text("portfolio\n")
    out = subprocess.run(["bash", str(REPO / "watchdog_mystic.sh")], cwd=str(REPO), env=env, text=True, capture_output=True, check=False)
    assert out.returncode == 0
    assert not (tmp_path / "started").exists()
    assert "WATCHDOG_DEFER_FAST_SUPERVISOR" in (tmp_path / "watchdog.log").read_text()
