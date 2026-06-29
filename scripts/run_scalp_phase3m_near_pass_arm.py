#!/usr/bin/env python3
"""Phase 3m — pre-arm on HIGH_QUALITY_NEAR_PASS; engine full gate still required."""

from __future__ import annotations

import json
import sqlite3
import subprocess
import sys
import time
import urllib.request
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import redis
from backend.services.binance_scalp.config import get_scalp_config
from backend.services.binance_scalp.economics import ScalpEconomics
from backend.services.binance_scalp.scalp_control import is_entry_armed, set_entry_armed
from scripts.watch_scalp_entry_opportunity import watch_loop

OUT = Path("/tmp/scalp_phase3m")
ENV = REPO / ".env"
DB = REPO / "mystic_trading.db"
WARM_SEC = 60.0
WATCH_MAX_SEC = 1800.0
TOTAL_MAX_SEC = 1860.0  # warm + 30 min


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
        open_scalp = c.execute("SELECT COUNT(*) FROM scalp_paper_positions WHERE status='OPEN'").fetchone()[0]
    ai = sorted(k for k in subprocess.check_output(["redis-cli", "KEYS", "ai_signal:day:*"], text=True).split() if k)
    try:
        h = urllib.request.urlopen("http://127.0.0.1:8000/health", timeout=5).status
    except Exception:
        h = 0
    cfg = get_scalp_config()
    return {
        "health": h,
        "ledger": {"cash_balance": led[0], "total_equity": led[1]} if led else None,
        "xrp": [{"symbol": r[0], "quantity": r[1], "entry_price": r[2]} for r in xrp],
        "paper_trades": pn,
        "open_scalp": open_scalp,
        "ai_signal_day": ai,
        "SCALP_LIVE": cfg.scalp_live,
        "SCALP_PAPER_ENABLED": cfg.scalp_paper_enabled,
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


def _collect(since: str, arm_iso: str | None = None) -> dict:
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
        post_arm_rejects: Counter[str] = Counter()
        if arm_iso:
            arm_sql = arm_iso[:19].replace("T", " ")
            rows = c.execute(
                "SELECT reason, detail FROM scalp_rejects WHERE created_at>=?",
                (arm_sql,),
            ).fetchall()
            for r in rows:
                post_arm_rejects[r[0]] += 1
        led = dict(
            zip(
                ["cash", "pos", "realized", "equity"],
                c.execute("SELECT cash_balance, positions_value, realized_pnl, total_equity FROM scalp_paper_ledger WHERE id=1").fetchone() or (None, None, None, None),
                strict=False,
            )
        )
        sb = [dict(r) for r in c.execute("SELECT * FROM scalp_scoreboard_daily ORDER BY day DESC LIMIT 3").fetchall()]
        open_p = c.execute("SELECT * FROM scalp_paper_positions WHERE status='OPEN'").fetchall()

    close = None
    missed = []
    buy_row = None
    if buys:
        buy_row = dict(buys[-1])
    for s in sells:
        d = json.loads(s["diagnostics_json"] or "{}")
        g = d.get("exit_gate") or {}
        pf = d.get("preflight") or {}
        close = {
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
        if s["exit_reason"] == "STALE_SCALP_TIMEOUT":
            np, tp = g.get("net_pct"), g.get("target_pct")
            if np is not None and tp is not None and float(np) >= float(tp):
                missed.append(s["trade_id"])

    return {
        "buys": len(buys),
        "sells": len(sells),
        "buy": buy_row,
        "close": close,
        "missed": missed,
        "rejects": rejects,
        "post_arm_rejects": dict(post_arm_rejects),
        "ledger": led,
        "scoreboard": sb,
        "open": [dict(r) for r in open_p],
    }


def _wait_until_done(since: str, arm_iso: str | None, deadline: float) -> dict:
    econ = ScalpEconomics.from_env()
    stale_extra = float(econ.stale_scalp_timeout_sec) + 30.0
    entry_seen_at: float | None = None

    while time.time() < deadline:
        tr = _collect(since, arm_iso)
        if tr["sells"] > 0:
            return tr
        if tr["buys"] > 0 and tr["open"]:
            if entry_seen_at is None:
                entry_seen_at = time.time()
            if time.time() - entry_seen_at >= stale_extra:
                return _collect(since, arm_iso)
        time.sleep(5.0)
    return _collect(since, arm_iso)


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    cfg = get_scalp_config()
    if cfg.scalp_live:
        print("SCALP_LIVE must be false", file=sys.stderr)
        return 1

    r = redis.from_url(cfg.redis_url, decode_responses=True)
    test_start_epoch = time.time()
    deadline = test_start_epoch + TOTAL_MAX_SEC
    since = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
    before = _day()
    (OUT / "day_before.json").write_text(json.dumps(before, indent=2))

    set_entry_armed(r, prefix=cfg.redis_key_prefix, armed=False)
    _set_paper(True)
    _start()
    time.sleep(WARM_SEC)

    arm_event: dict | None = None
    arm_iso: str | None = None

    def on_arm(rows: list[dict], event: dict) -> None:
        nonlocal arm_event, arm_iso
        arm_event = event
        arm_iso = _arm()
        (OUT / "arm_event.json").write_text(json.dumps(event, indent=2, default=str))
        (OUT / "trigger_rows.json").write_text(json.dumps(rows, indent=2, default=str))

    watch_budget = min(WATCH_MAX_SEC, max(0.0, deadline - time.time()))
    w0 = time.time()
    stats = watch_loop(
        interval_sec=5.0,
        max_sec=watch_budget,
        log_path=OUT / "watch.jsonl",
        on_arm=on_arm,
        arm_on_high_quality_near_pass=True,
    )
    watch_sec = time.time() - w0

    tr = _wait_until_done(since, arm_iso, deadline)

    _stop(cfg)
    after = _day()
    mem = {}
    with open("/proc/meminfo") as f:
        for line in f:
            if line.startswith(("MemAvailable:", "SwapFree:", "MemTotal:", "SwapTotal:")):
                k, v = line.split(":")
                mem[k.strip()] = int(v.split()[0])

    gate_fail_reason = None
    if arm_iso and tr["buys"] == 0:
        post = tr.get("post_arm_rejects") or {}
        gate_fail_reason = post if post else "no_post_arm_rejects_logged"

    rep = {
        "1_watch_duration_sec": watch_sec,
        "2_near_pass_arm_events": stats.arm_events,
        "2_hq_near_pass_arm_count": stats.hq_near_pass_arm_count,
        "2_near_pass_count": stats.near_pass_count,
        "3_opportunity_pass_events": [e for e in stats.events if e.get("event") == "OPPORTUNITY_PASS"],
        "3_pass_count": stats.pass_count,
        "4_arm_event": arm_event,
        "4_arm_time_utc": arm_iso,
        "5_engine_entered": tr["buys"] > 0,
        "6_gate_fail_if_no_entry": gate_fail_reason,
        "7_trade": tr.get("buy"),
        "8_exit_reason": (tr.get("close") or {}).get("exit_reason"),
        "9_exit_diagnostics": tr.get("close"),
        "10_missed_target": tr["missed"] or ["none"],
        "11_rejects": tr["rejects"],
        "12_day_untouched": {
            "health": after["health"],
            "ledger_ok": before["ledger"] == after["ledger"],
            "xrp_ok": before["xrp"] == after["xrp"],
            "paper_trades_ok": before["paper_trades"] == after["paper_trades"],
            "ai_ok": before["ai_signal_day"] == after["ai_signal_day"],
            "scalp_live": after["SCALP_LIVE"],
            "scalp_paper_restored": not after["SCALP_PAPER_ENABLED"],
        },
        "13_memory_kb": mem,
        "14_safe_to_continue": (
            after["health"] == 200 and not tr["open"] and not is_entry_armed(r, prefix=cfg.redis_key_prefix) and not after["SCALP_PAPER_ENABLED"] and before["ledger"] == after["ledger"]
        ),
        "round_trip_complete": tr["buys"] > 0 and tr["sells"] > 0,
        "entry_armed_cleared": not is_entry_armed(r, prefix=cfg.redis_key_prefix),
        "no_arm": arm_iso is None,
    }
    (OUT / "final_report.json").write_text(json.dumps(rep, indent=2, default=str))
    print(json.dumps(rep, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
