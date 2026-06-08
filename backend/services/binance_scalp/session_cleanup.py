"""Paper-only scalp session teardown — stop runner, restore env, close orphans."""

from __future__ import annotations

import json
import os
import re
import signal
import sqlite3
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
ENV_PATH = REPO_ROOT / ".env"
RUNNER_REL = "backend/services/binance_scalp/runner.py"
SERVICE_NAME = "mystic-scalp-paper.service"

CLEANUP_EXIT_REASONS = frozenset(
    {
        "SCALP_TEST_ORPHAN_RESET",
        "SCALP_TEST_SESSION_END_RESET",
    }
)


def _venv_python() -> Path:
    return REPO_ROOT / "venv" / "bin" / "python3"


def read_env() -> dict[str, str]:
    out: dict[str, str] = {}
    if not ENV_PATH.exists():
        return out
    for line in ENV_PATH.read_text().splitlines():
        if "=" in line and not line.strip().startswith("#"):
            k, v = line.split("=", 1)
            out[k.strip()] = v.strip()
    return out


def set_env_keys(updates: dict[str, str]) -> dict[str, str]:
    backup = read_env()
    lines = ENV_PATH.read_text().splitlines() if ENV_PATH.exists() else []
    seen: set[str] = set()
    new_lines: list[str] = []
    for line in lines:
        if "=" in line and not line.strip().startswith("#"):
            k = line.split("=", 1)[0].strip()
            if k in updates:
                new_lines.append(f"{k}={updates[k]}")
                seen.add(k)
                continue
        new_lines.append(line)
    for k, v in updates.items():
        if k not in seen:
            new_lines.append(f"{k}={v}")
    ENV_PATH.write_text("\n".join(new_lines) + "\n")
    return {k: backup.get(k, "") for k in updates}


def restore_env_backup(backup: dict[str, str]) -> None:
    safe = {
        "SCALP_PAPER_ENABLED": "false",
        "SCALP_LIVE": "false",
        "SCALP_CALIBRATION_MODE": "false",
    }
    merged = {**backup, **safe}
    set_env_keys(merged)


def restore_safe_env() -> None:
    set_env_keys(
        {
            "SCALP_PAPER_ENABLED": "false",
            "SCALP_LIVE": "false",
            "SCALP_CALIBRATION_MODE": "false",
        }
    )


def stop_scalp_runner(*, grace_sec: float = 15.0) -> dict[str, Any]:
    subprocess.run(
        ["systemctl", "--user", "stop", SERVICE_NAME],
        capture_output=True,
        text=True,
        check=False,
    )
    procs = _find_runner_pids()
    stopped: list[int] = []
    for pid in procs:
        try:
            os.kill(pid, signal.SIGTERM)
            stopped.append(pid)
        except ProcessLookupError:
            pass
    deadline = time.time() + grace_sec
    while time.time() < deadline:
        if not _find_runner_pids():
            break
        time.sleep(0.5)
    remaining = _find_runner_pids()
    for pid in remaining:
        try:
            os.kill(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        else:
            stopped.append(pid)
    time.sleep(0.5)
    return {
        "stopped_pids": stopped,
        "runner_remaining": _find_runner_pids(),
        "service_active": _service_active(),
    }


def _find_runner_pids() -> list[int]:
    proc = subprocess.run(
        ["pgrep", "-f", RUNNER_REL],
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0 or not proc.stdout.strip():
        return []
    return [int(x) for x in proc.stdout.split() if x.strip().isdigit()]


def _service_active() -> bool:
    proc = subprocess.run(
        ["systemctl", "--user", "is-active", SERVICE_NAME],
        capture_output=True,
        text=True,
        check=False,
    )
    return proc.stdout.strip() == "active"


def clear_entry_armed(redis_url: str | None = None, *, prefix: str | None = None) -> bool:
    try:
        import redis

        from backend.services.binance_scalp.config import get_scalp_config
        from backend.services.binance_scalp.scalp_control import clear_entry_armed as _clear

        cfg = get_scalp_config()
        url = redis_url or cfg.redis_url
        pref = prefix or cfg.redis_key_prefix
        client = redis.from_url(url, decode_responses=True)
        _clear(client, prefix=pref)
        return True
    except Exception:
        return False


def open_positions(db_path: str) -> list[dict[str, Any]]:
    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            "SELECT * FROM scalp_paper_positions WHERE status='OPEN'"
        ).fetchall()
    return [dict(r) for r in rows]


def close_open_positions(
    db_path: str,
    *,
    reason: str = "SCALP_TEST_ORPHAN_RESET",
) -> list[dict[str, Any]]:
    """Close paper orphans — excluded from scalp_scoreboard_daily."""
    if reason not in CLEANUP_EXIT_REASONS:
        raise ValueError(f"invalid cleanup reason: {reason}")
    closed: list[dict[str, Any]] = []
    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            "SELECT * FROM scalp_paper_positions WHERE status='OPEN'"
        ).fetchall()
        if not rows:
            return closed
        ledger = conn.execute(
            "SELECT cash_balance, positions_value, realized_pnl, total_equity "
            "FROM scalp_paper_ledger WHERE id=1"
        ).fetchone()
        pre = dict(ledger) if ledger else {}

        for row in rows:
            sym = str(row["symbol"])
            entry = float(row["entry_price"])
            qty = float(row["quantity"])
            trade_id = str(row["trade_id"])
            exit_price = entry
            notional = qty * exit_price
            fee = 0.0
            slip = 0.0
            net_usd = 0.0
            net_pct = 0.0
            sell_tid = f"{trade_id}_CLEANUP"
            diag = {
                "cleanup": True,
                "exclude_from_strategy_pnl": True,
                "historical_only": False,
                "exit_reason": reason,
            }
            conn.execute(
                """
                INSERT INTO scalp_paper_trades
                (trade_id, symbol, exchange, strategy_id, side, quantity, price,
                 notional, fee_usd, slippage_usd, pnl_usd, pnl_pct, entry_price,
                 exit_reason, diagnostics_json)
                VALUES (?, ?, ?, ?, 'SELL', ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    sell_tid,
                    sym,
                    row["exchange"],
                    row["strategy_id"],
                    qty,
                    exit_price,
                    notional,
                    fee,
                    slip,
                    net_usd,
                    net_pct,
                    entry,
                    reason,
                    json.dumps(diag),
                ),
            )
            conn.execute(
                "UPDATE scalp_paper_positions SET status='CLOSED', state=? WHERE id=?",
                (reason, row["id"]),
            )
            pos_cost = entry * qty
            cash = float(pre.get("cash_balance", 0)) + notional
            pos_val = max(0.0, float(pre.get("positions_value", 0)) - pos_cost)
            pre["cash_balance"] = cash
            pre["positions_value"] = pos_val
            conn.execute(
                """
                UPDATE scalp_paper_ledger SET
                  cash_balance = ?,
                  positions_value = ?,
                  total_equity = ? + ?,
                  updated_at = datetime('now')
                WHERE id = 1
                """,
                (cash, pos_val, cash, pos_val),
            )
            conn.execute(
                """
                INSERT INTO scalp_trade_audit
                (trade_id, action, symbol, qty, price, pre_ledger_json, post_ledger_json, reason)
                VALUES (?, 'SELL', ?, ?, ?, ?, ?, ?)
                """,
                (
                    sell_tid,
                    sym,
                    qty,
                    exit_price,
                    json.dumps(pre),
                    json.dumps({"cash_balance": cash, "positions_value": pos_val}),
                    reason,
                ),
            )
            closed.append({"symbol": sym, "trade_id": trade_id, "reason": reason})
        conn.commit()
    return closed


def verify_safe_state(db_path: str) -> dict[str, Any]:
    env = read_env()
    armed_proc = subprocess.run(
        [
            str(_venv_python()),
            "-c",
            (
                "import redis; "
                "r=redis.from_url('redis://127.0.0.1:6379/0', decode_responses=True); "
                "print(r.get('scalp:control:entry_armed') or '0')"
            ),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    armed = (armed_proc.stdout or "").strip()
    open_n = len(open_positions(db_path))
    runners = _find_runner_pids()
    ok = (
        env.get("SCALP_PAPER_ENABLED", "false").lower() == "false"
        and env.get("SCALP_LIVE", "false").lower() == "false"
        and env.get("SCALP_CALIBRATION_MODE", "false").lower() in {"false", "", "0"}
        and armed in {"0", "", "None"}
        and open_n == 0
        and not runners
        and not _service_active()
    )
    return {
        "safe_state_verified": ok,
        "scalp_paper_enabled": env.get("SCALP_PAPER_ENABLED", ""),
        "scalp_live": env.get("SCALP_LIVE", ""),
        "scalp_calibration_mode": env.get("SCALP_CALIBRATION_MODE", ""),
        "entry_armed": armed,
        "open_positions_end": open_n,
        "runner_pids": runners,
        "runner_stopped": not bool(runners),
        "service_active": _service_active(),
    }


def full_teardown(
    db_path: str,
    *,
    close_reason: str = "SCALP_TEST_ORPHAN_RESET",
) -> dict[str, Any]:
    runner = stop_scalp_runner()
    restore_safe_env()
    armed_ok = clear_entry_armed()
    closed = close_open_positions(db_path, reason=close_reason)
    verify = verify_safe_state(db_path)
    return {
        "runner": runner,
        "entry_armed_cleared": armed_ok,
        "closed_positions": closed,
        **verify,
    }
