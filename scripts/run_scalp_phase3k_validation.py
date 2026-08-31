#!/usr/bin/env python3
"""Phase 3k — short volatile-window warm/arm paper validation."""

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

OUT = Path("/tmp/scalp_phase3k")
ENV = REPO / ".env"
DB = REPO / "mystic_trading.db"
WARM_SEC = 60.0
ARMED_SEC = 900.0
WATCH_MAX_SEC = 5400.0  # 90 min volatile window


def _set_paper(on: bool) -> None:
    text = ENV.read_text()
    flag = "true" if on else "false"
    lines = [(f"SCALP_PAPER_ENABLED={flag}" if line.startswith("SCALP_PAPER_ENABLED=") else line) for line in text.splitlines()]
    if not any(line.startswith("SCALP_PAPER_ENABLED=") for line in lines):
        lines.append(f"SCALP_PAPER_ENABLED={flag}")
    ENV.write_text("\n".join(lines) + ("\n" if text.endswith("\n") else ""))


def _day() -> dict:
    with sqlite3.connect(DB) as c:
        led = c.execute("SELECT cash_balance, total_equity FROM portfolio_engine_ledger WHERE id=1").fetchone()
        xrp = c.execute("SELECT symbol, quantity, entry_price FROM portfolio_engine_positions WHERE symbol LIKE '%XRP%'").fetchall()
        pn = c.execute("SELECT COUNT(*) FROM paper_trades").fetchone()[0]
    ai = sorted(k for k in subprocess.check_output(["redis-cli", "KEYS", "ai_signal:day:*"], text=True).split() if k)
    try:
        h = urllib.request.urlopen("http://127.0.0.1:8000/health", timeout=5).status
    except Exception:
        h = 0
    return {
        "health": h,
        "ledger": {"cash_balance": led[0], "total_equity": led[1]} if led else None,
        "xrp": [{"symbol": r[0], "quantity": r[1], "entry_price": r[2]} for r in xrp],
        "paper_trades": pn,
        "ai_signal_day": ai,
    }


def _start() -> None:
    subprocess.run(["systemctl", "--user", "daemon-reload"], check=False)
    subprocess.run(["systemctl", "--user", "restart", "mystic-scalp-paper.service"], check=True)
    time.sleep(5)


def _stop(cfg) -> None:
    subprocess.run(["systemctl", "--user", "stop", "mystic-scalp-paper.service"], check=False)
    time.sleep(2)
    r = redis.from_url(cfg.redis_url, decode_responses=True)
    set_entry_armed(r, prefix=cfg.redis_key_prefix, armed=False)
    _set_paper(False)


def _arm() -> str:
    subprocess.run(
        [str(REPO / "venv/bin/python3"), str(REPO / "scripts/scalp_arm_entries.py"), "arm"],
        check=True,
    )
    return datetime.now(timezone.utc).isoformat()


def _trades(since: str) -> dict:
    with sqlite3.connect(DB) as c:
        c.row_factory = sqlite3.Row
        buys = c.execute("SELECT * FROM scalp_paper_trades WHERE side='BUY' AND created_at>=?", (since,)).fetchall()
        sells = c.execute("SELECT * FROM scalp_paper_trades WHERE side='SELL' AND created_at>=?", (since,)).fetchall()
        rejects = {
            r["reason"]: r["cnt"]
            for r in c.execute(
                "SELECT reason, COUNT(*) cnt FROM scalp_rejects WHERE created_at>=? GROUP BY reason",
                (since,),
            )
        }
        ledger = dict(c.execute("SELECT cash_balance, positions_value, realized_pnl, total_equity FROM scalp_paper_ledger WHERE id=1").fetchone() or {})
        sb = [dict(r) for r in c.execute("SELECT * FROM scalp_scoreboard_daily ORDER BY day DESC LIMIT 3").fetchall()]
        open_p = c.execute("SELECT * FROM scalp_paper_positions WHERE status='OPEN'").fetchall()

    closes = []
    missed = []
    for s in sells:
        d = json.loads(s["diagnostics_json"] or "{}")
        g = d.get("exit_gate") or {}
        pf = d.get("preflight") or {}
        closes.append(
            {
                "symbol": s["symbol"],
                "entry_price": s["entry_price"],
                "exit_price": s["price"],
                "exit_reason": s["exit_reason"],
                "pnl_usd": s["pnl_usd"],
                "best_bid": g.get("current_bid"),
                "expected_sell_fill": g.get("expected_sell_fill"),
                "executable_exit_net_pct": g.get("net_pct"),
                "target_pct": g.get("target_pct"),
                "sell_preflight_pass": g.get("sell_preflight_pass"),
                "sell_preflight": pf,
                "exit_gate": g,
            }
        )
        if s["exit_reason"] == "STALE_SCALP_TIMEOUT":
            np, tp = g.get("net_pct"), g.get("target_pct")
            if np is not None and tp is not None and float(np) >= float(tp):
                missed.append(s["trade_id"])

    buy_detail = None
    if buys:
        b = buys[-1]
        buy_detail = {
            "trade_id": b["trade_id"],
            "symbol": b["symbol"],
            "price": b["price"],
            "qty": b["quantity"],
            "created_at": b["created_at"],
        }

    return {
        "buys": len(buys),
        "sells": len(sells),
        "buy": buy_detail,
        "closes": closes,
        "missed": missed,
        "rejects": rejects,
        "ledger": ledger,
        "scoreboard": sb,
        "open": [dict(r) for r in open_p],
    }


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    cfg = get_scalp_config()
    r = redis.from_url(cfg.redis_url, decode_responses=True)
    t0 = datetime.now(timezone.utc)
    since = t0.strftime("%Y-%m-%d %H:%M:%S")
    before = _day()
    (OUT / "day_before.json").write_text(json.dumps(before, indent=2))

    set_entry_armed(r, prefix=cfg.redis_key_prefix, armed=False)
    _set_paper(True)
    _start()
    assert not is_entry_armed(r, prefix=cfg.redis_key_prefix)

    warm_t0 = time.time()
    time.sleep(WARM_SEC)
    warm_sec = time.time() - warm_t0

    trigger: list[dict] | None = None
    arm_iso: str | None = None

    def on_pass(rows: list[dict]) -> None:
        nonlocal trigger, arm_iso
        trigger = rows
        arm_iso = _arm()
        (OUT / "trigger.json").write_text(json.dumps(rows, indent=2, default=str))

    w0 = time.time()
    stats = watch_loop(
        interval_sec=5.0,
        max_sec=WATCH_MAX_SEC,
        log_path=OUT / "watch.jsonl",
        on_pass=on_pass,
    )
    watch_sec = time.time() - w0

    tr = {"buys": 0, "sells": 0, "closes": [], "missed": [], "rejects": {}, "ledger": {}, "scoreboard": [], "open": [], "buy": None}
    if arm_iso:
        time.sleep(ARMED_SEC)
        tr = _trades(since)
        if tr["open"]:
            econ = ScalpEconomics.from_env()
            time.sleep(float(econ.stale_scalp_timeout_sec) + 45)
            tr = _trades(since)

    _stop(cfg)
    after = _day()
    mem = {}
    with open("/proc/meminfo") as f:
        for line in f:
            if line.startswith(("MemAvailable:", "SwapFree:")):
                k, v = line.split(":")
                mem[k.strip()] = int(v.split()[0])

    pass_row = next((x for x in (trigger or []) if x.get("opportunity") == "OPPORTUNITY_PASS"), None)
    rep = {
        "1_warm_duration_sec": warm_sec,
        "2_watch_duration_sec": watch_sec,
        "3_opportunity_pass": pass_row,
        "3_near_pass_count": stats.near_pass_count,
        "4_arm_time_utc": arm_iso,
        "5_engine_entered": tr["buys"] > 0,
        "6_trade_opened": tr["buy"],
        "7_exit_reason": tr["closes"][0]["exit_reason"] if tr["closes"] else None,
        "8_exit_diagnostics": tr["closes"][0] if tr["closes"] else None,
        "9_missed_target": tr["missed"] or ["none"],
        "10_ledger_scoreboard": {"ledger": tr["ledger"], "scoreboard": tr["scoreboard"]},
        "11_rejects": tr["rejects"],
        "12_day_untouched": {
            "health": after["health"],
            "ledger_ok": before["ledger"] == after["ledger"],
            "xrp_ok": before["xrp"] == after["xrp"],
            "paper_trades_ok": before["paper_trades"] == after["paper_trades"],
            "ai_ok": before["ai_signal_day"] == after["ai_signal_day"],
        },
        "13_memory_kb": mem,
        "14_safe_to_continue": after["health"] == 200 and not tr["open"] and not is_entry_armed(r, prefix=cfg.redis_key_prefix),
        "entry_armed_cleared": not is_entry_armed(r, prefix=cfg.redis_key_prefix),
        "no_pass": arm_iso is None,
    }
    (OUT / "final_report.json").write_text(json.dumps(rep, indent=2, default=str))
    print(json.dumps(rep, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
