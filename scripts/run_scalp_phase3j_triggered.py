#!/usr/bin/env python3
"""Phase 3j — warm-state triggered paper validation with entry_armed handoff."""

from __future__ import annotations

import json
import sqlite3
import subprocess
import sys
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import redis

from backend.services.binance_scalp.config import get_scalp_config
from backend.services.binance_scalp.economics import ScalpEconomics
from backend.services.binance_scalp.scalp_control import (
    control_key,
    is_entry_armed,
    set_entry_armed,
)
from scripts.watch_scalp_entry_opportunity import watch_loop

OUT_DIR = Path("/tmp/scalp_phase3j")
ENV_PATH = REPO / ".env"
DB = REPO / "mystic_trading.db"
WARM_SEC = 60.0
ARMED_RUN_SEC = 900.0
WATCH_MAX_SEC = 7200.0


def _set_paper_enabled(enabled: bool) -> None:
    text = ENV_PATH.read_text()
    flag = "true" if enabled else "false"
    lines = [f"SCALP_PAPER_ENABLED={flag}" if line.startswith("SCALP_PAPER_ENABLED=") else line for line in text.splitlines()]
    if not any(line.startswith("SCALP_PAPER_ENABLED=") for line in lines):
        lines.append(f"SCALP_PAPER_ENABLED={flag}")
    ENV_PATH.write_text("\n".join(lines) + ("\n" if text.endswith("\n") else ""))


def _day_snapshot() -> dict:
    with sqlite3.connect(DB) as conn:
        ledger = conn.execute("SELECT cash_balance, total_equity FROM portfolio_engine_ledger WHERE id=1").fetchone()
        xrp = conn.execute("SELECT symbol, quantity, entry_price FROM portfolio_engine_positions WHERE symbol LIKE '%XRP%'").fetchall()
        paper_n = conn.execute("SELECT COUNT(*) FROM paper_trades").fetchone()[0]
    ai = sorted(k for k in subprocess.check_output(["redis-cli", "KEYS", "ai_signal:day:*"], text=True).split() if k)
    scalp_keys = sorted(k for k in subprocess.check_output(["redis-cli", "KEYS", "scalp:*"], text=True).split() if k)
    try:
        with urllib.request.urlopen("http://127.0.0.1:8000/health", timeout=5) as r:
            health = r.status
    except Exception:
        health = 0
    return {
        "health": health,
        "ledger": {"cash_balance": ledger[0], "total_equity": ledger[1]} if ledger else None,
        "xrp": [{"symbol": r[0], "quantity": r[1], "entry_price": r[2]} for r in xrp],
        "paper_trades_count": paper_n,
        "ai_signal_day_keys": ai,
        "redis_scalp_keys": scalp_keys,
    }


def _start_paper() -> None:
    subprocess.run(["systemctl", "--user", "daemon-reload"], check=False)
    subprocess.run(["systemctl", "--user", "restart", "mystic-scalp-paper.service"], check=True)
    time.sleep(5)


def _stop_paper(cfg) -> None:
    subprocess.run(["systemctl", "--user", "stop", "mystic-scalp-paper.service"], check=False)
    time.sleep(2)
    client = redis.from_url(cfg.redis_url, decode_responses=True)
    set_entry_armed(client, prefix=cfg.redis_key_prefix, armed=False)
    _set_paper_enabled(False)


def _collect_trades(since: str) -> dict:
    with sqlite3.connect(DB) as conn:
        conn.row_factory = sqlite3.Row
        buys = conn.execute(
            "SELECT * FROM scalp_paper_trades WHERE side='BUY' AND created_at >= ?",
            (since,),
        ).fetchall()
        sells = conn.execute(
            "SELECT * FROM scalp_paper_trades WHERE side='SELL' AND created_at >= ?",
            (since,),
        ).fetchall()
        rejects = {
            r["reason"]: r["cnt"]
            for r in conn.execute(
                "SELECT reason, COUNT(*) cnt FROM scalp_rejects WHERE created_at >= ? GROUP BY reason",
                (since,),
            )
        }
        would_enter = conn.execute(
            "SELECT COUNT(*) FROM scalp_rejects WHERE reason='WOULD_ENTER_NOT_ARMED' AND created_at >= ?",
            (since,),
        ).fetchone()[0]
        open_pos = conn.execute("SELECT * FROM scalp_paper_positions WHERE status='OPEN'").fetchall()

    closes = []
    missed = []
    for sell in sells:
        diag = json.loads(sell["diagnostics_json"] or "{}")
        gate = diag.get("exit_gate") or {}
        closes.append(
            {
                "trade_id": str(sell["trade_id"]).replace("_SELL", ""),
                "symbol": sell["symbol"],
                "entry_price": sell["entry_price"],
                "exit_price": sell["price"],
                "exit_reason": sell["exit_reason"],
                "pnl_usd": sell["pnl_usd"],
                "exit_gate": gate,
            }
        )
        if sell["exit_reason"] == "STALE_SCALP_TIMEOUT":
            net_pct = gate.get("net_pct")
            target = gate.get("target_pct")
            if net_pct is not None and target is not None and float(net_pct) >= float(target):
                missed.append(str(sell["trade_id"]))

    return {
        "buys": len(buys),
        "sells": len(sells),
        "closes": closes,
        "missed_target_at_stale": missed,
        "rejects": rejects,
        "would_enter_not_armed": would_enter,
        "open_positions": [dict(r) for r in open_pos],
    }


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    cfg = get_scalp_config()
    if cfg.scalp_live:
        print("SCALP_LIVE must be false", file=sys.stderr)
        return 1

    client = redis.from_url(cfg.redis_url, decode_responses=True)
    prefix = cfg.redis_key_prefix
    test_start = datetime.now(timezone.utc)
    test_start_sql = test_start.strftime("%Y-%m-%d %H:%M:%S")
    day_before = _day_snapshot()
    (OUT_DIR / "day_before.json").write_text(json.dumps(day_before, indent=2))

    set_entry_armed(client, prefix=prefix, armed=False)
    _set_paper_enabled(True)
    _start_paper()
    warm_start = time.time()
    time.sleep(WARM_SEC)
    warm_duration = time.time() - warm_start

    armed_at: float | None = None
    trigger_rows: list[dict] | None = None

    def on_pass(rows: list[dict]) -> None:
        nonlocal armed_at, trigger_rows
        set_entry_armed(client, prefix=prefix, armed=True)
        armed_at = time.time()
        trigger_rows = rows
        (OUT_DIR / "trigger.json").write_text(json.dumps(rows, indent=2, default=str))

    watch_start = time.time()
    stats = watch_loop(
        interval_sec=5.0,
        max_sec=WATCH_MAX_SEC,
        log_path=OUT_DIR / "watch_events.jsonl",
        on_pass=on_pass,
    )
    watch_duration = time.time() - watch_start

    trade_result = {
        "buys": 0,
        "sells": 0,
        "closes": [],
        "missed_target_at_stale": [],
        "rejects": {},
        "would_enter_not_armed": 0,
        "open_positions": [],
    }
    armed_run_sec = 0.0

    if armed_at is not None:
        armed_run_sec = time.time() - armed_at
        remaining = max(0.0, ARMED_RUN_SEC - armed_run_sec)
        if remaining > 0:
            time.sleep(remaining)
        trade_result = _collect_trades(test_start_sql)
        if trade_result["open_positions"]:
            econ = ScalpEconomics.from_env()
            time.sleep(float(econ.stale_scalp_timeout_sec) + 45)
            trade_result = _collect_trades(test_start_sql)

    _stop_paper(cfg)
    day_after = _day_snapshot()
    mem = {}
    with open("/proc/meminfo") as f:
        for line in f:
            if line.startswith(("MemAvailable:", "SwapFree:")):
                k, v = line.split(":")
                mem[k.strip()] = int(v.split()[0])

    report = {
        "files_changed": [
            "backend/services/binance_scalp/scalp_control.py",
            "backend/services/binance_scalp/paper_engine.py",
            "backend/services/binance_scalp/runner.py",
            "scripts/scalp_arm_entries.py",
            "scripts/run_scalp_phase3j_triggered.py",
        ],
        "2_warm_state_design": "Redis scalp:control:entry_armed; paper runs warm with entries disarmed; watcher arms on OPPORTUNITY_PASS",
        "3_engine_warm_duration_sec": warm_duration,
        "4_watcher_pass_event": trigger_rows,
        "4_near_pass_count": stats.near_pass_count,
        "4_pass_count": stats.pass_count,
        "5_engine_entered_after_arm": trade_result["buys"] > 0,
        "6_trades_opened": trade_result["buys"],
        "6_trades_closed": trade_result["sells"],
        "7_exit_diagnostics": trade_result["closes"],
        "8_missed_target_check": trade_result["missed_target_at_stale"] or ["none"],
        "9_rejects_by_reason": trade_result["rejects"],
        "9_would_enter_not_armed_count": trade_result["would_enter_not_armed"],
        "10_redis_control_keys": {
            "entry_armed_key": control_key(prefix, "entry_armed"),
            "entry_armed_now": is_entry_armed(client, prefix=prefix),
            "redis_scalp_keys_after": day_after.get("redis_scalp_keys"),
        },
        "11_day_untouched": {
            "health": day_after["health"],
            "ledger_unchanged": day_before["ledger"] == day_after["ledger"],
            "xrp_unchanged": day_before["xrp"] == day_after["xrp"],
            "paper_trades_unchanged": day_before["paper_trades_count"] == day_after["paper_trades_count"],
            "ai_signal_unchanged": day_before["ai_signal_day_keys"] == day_after["ai_signal_day_keys"],
        },
        "12_memory_kb": mem,
        "13_safe_to_continue": day_after["health"] == 200 and not trade_result["open_positions"],
        "best_btc": stats.best_btc,
        "best_eth": stats.best_eth,
        "armed_run_sec": armed_run_sec,
        "watch_duration_sec": watch_duration,
        "no_pass": armed_at is None,
    }
    (OUT_DIR / "final_report.json").write_text(json.dumps(report, indent=2, default=str))
    print(json.dumps(report, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
