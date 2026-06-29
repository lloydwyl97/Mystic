#!/usr/bin/env python3
"""Phase 3i — triggered paper validation when entry gate fully passes."""

from __future__ import annotations

import json
import os
import sqlite3
import subprocess
import sys
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from backend.services.binance_scalp.config import get_scalp_config
from backend.services.binance_scalp.economics import ScalpEconomics
from backend.services.binance_scalp.market_reader import ScalpMarketReader
from backend.services.binance_scalp.momentum_tracker import MomentumTracker
from scripts.watch_scalp_entry_opportunity import watch_loop

OUT_DIR = Path("/tmp/scalp_phase3i")
ENV_PATH = REPO / ".env"
DB = REPO / "mystic_trading.db"
WATCH_MAX_SEC = 7200.0
PAPER_RUN_SEC = 900.0  # 15 min (within 10-20)


def _day_snapshot() -> dict:
    with sqlite3.connect(DB) as conn:
        ledger = conn.execute("SELECT cash_balance, total_equity FROM portfolio_engine_ledger WHERE id=1").fetchone()
        xrp = conn.execute("SELECT symbol, quantity, entry_price FROM portfolio_engine_positions WHERE symbol LIKE '%XRP%'").fetchall()
        paper_n = conn.execute("SELECT COUNT(*) FROM paper_trades").fetchone()[0]
    ai = sorted(k for k in subprocess.check_output(["redis-cli", "KEYS", "ai_signal:day:*"], text=True).split() if k)
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
    }


def _set_paper_enabled(enabled: bool) -> None:
    text = ENV_PATH.read_text()
    flag = "true" if enabled else "false"
    if "SCALP_PAPER_ENABLED=" in text:
        lines = []
        for line in text.splitlines():
            if line.startswith("SCALP_PAPER_ENABLED="):
                lines.append(f"SCALP_PAPER_ENABLED={flag}")
            else:
                lines.append(line)
        ENV_PATH.write_text("\n".join(lines) + ("\n" if text.endswith("\n") else ""))
    else:
        ENV_PATH.write_text(text + f"\nSCALP_PAPER_ENABLED={flag}\n")


def _start_paper() -> None:
    subprocess.run(["systemctl", "--user", "daemon-reload"], check=False)
    subprocess.run(["systemctl", "--user", "restart", "mystic-scalp-paper.service"], check=True)
    time.sleep(5)


def _stop_paper() -> None:
    subprocess.run(["systemctl", "--user", "stop", "mystic-scalp-paper.service"], check=False)
    time.sleep(2)
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
        "open_positions": [dict(r) for r in open_pos],
    }


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    watch_start = datetime.now(timezone.utc)
    watch_start_sql = watch_start.strftime("%Y-%m-%d %H:%M:%S")
    day_before = _day_snapshot()
    (OUT_DIR / "day_before.json").write_text(json.dumps(day_before, indent=2))

    paper_started = False
    paper_trigger_row: list[dict] | None = None
    paper_since: str | None = None

    def on_pass(rows: list[dict]) -> None:
        nonlocal paper_started, paper_trigger_row, paper_since
        paper_started = True
        paper_trigger_row = rows
        paper_since = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
        _set_paper_enabled(True)
        _start_paper()
        (OUT_DIR / "trigger.json").write_text(json.dumps(rows, indent=2, default=str))

    stats = watch_loop(
        interval_sec=5.0,
        max_sec=WATCH_MAX_SEC,
        log_path=OUT_DIR / "watch_events.jsonl",
        on_pass=on_pass,
    )

    watch_duration = time.time() - watch_start.timestamp()
    trade_result: dict = {"buys": 0, "sells": 0, "closes": [], "missed_target_at_stale": [], "rejects": {}, "open_positions": []}

    if paper_started:
        time.sleep(PAPER_RUN_SEC)
        _stop_paper()
        trade_result = _collect_trades(paper_since or watch_start_sql)
        if trade_result["open_positions"]:
            econ = ScalpEconomics.from_env()
            _set_paper_enabled(True)
            _start_paper()
            time.sleep(float(econ.stale_scalp_timeout_sec) + 45)
            _stop_paper()
            trade_result = _collect_trades(paper_since or watch_start_sql)
    else:
        _set_paper_enabled(False)
        subprocess.run(["systemctl", "--user", "stop", "mystic-scalp-paper.service"], check=False)

    day_after = _day_snapshot()
    mem = {}
    with open("/proc/meminfo") as f:
        for line in f:
            if line.startswith(("MemAvailable:", "SwapFree:")):
                k, v = line.split(":")
                mem[k.strip()] = int(v.split()[0])

    report = {
        "1_watch_duration_sec": watch_duration,
        "1_watch_duration_min": round(watch_duration / 60, 1),
        "2_near_pass_count": stats.near_pass_count,
        "2_pass_count": stats.pass_count,
        "3_best_btc_setup": stats.best_btc,
        "4_best_eth_setup": stats.best_eth,
        "5_paper_service_started": paper_started,
        "6_trades_opened": trade_result["buys"],
        "6_trades_closed": trade_result["sells"],
        "7_exit_diagnostics": trade_result["closes"],
        "8_missed_target_check": trade_result["missed_target_at_stale"] or ["none"],
        "9_rejects_during_paper": trade_result["rejects"],
        "10_day_untouched": {
            "health": day_after["health"],
            "ledger_unchanged": day_before["ledger"] == day_after["ledger"],
            "xrp_unchanged": day_before["xrp"] == day_after["xrp"],
            "paper_trades_unchanged": day_before["paper_trades_count"] == day_after["paper_trades_count"],
            "ai_signal_unchanged": day_before["ai_signal_day_keys"] == day_after["ai_signal_day_keys"],
        },
        "11_memory_kb": mem,
        "12_safe_to_continue": day_after["health"] == 200 and not trade_result["open_positions"],
        "no_valid_setup": not paper_started,
        "trigger_rows": paper_trigger_row,
    }
    (OUT_DIR / "final_report.json").write_text(json.dumps(report, indent=2, default=str))
    print(json.dumps(report, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
