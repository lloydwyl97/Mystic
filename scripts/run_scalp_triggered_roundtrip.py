#!/usr/bin/env python3
"""Short triggered paper round-trip validation — warm/disarm, arm on OPPORTUNITY_PASS."""

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

import redis  # noqa: E402

from backend.services.binance_scalp.config import get_scalp_config  # noqa: E402
from backend.services.binance_scalp.economics import ScalpEconomics  # noqa: E402
from backend.services.binance_scalp.scalp_control import is_entry_armed, set_entry_armed  # noqa: E402
from scripts.watch_scalp_entry_opportunity import watch_loop  # noqa: E402

OUT = Path("/tmp/scalp_triggered_roundtrip")
ENV = REPO / ".env"
DB = REPO / "mystic_trading.db"
WARM_SEC = 60.0
WATCH_MAX_SEC = 5400.0
POST_ARM_MAX_SEC = 1200.0  # 20 min after arm


def _set_paper(on: bool) -> None:
    text = ENV.read_text()
    flag = "true" if on else "false"
    lines = [
        (f"SCALP_PAPER_ENABLED={flag}" if l.startswith("SCALP_PAPER_ENABLED=") else l)
        for l in text.splitlines()
    ]
    if not any(l.startswith("SCALP_PAPER_ENABLED=") for l in lines):
        lines.append(f"SCALP_PAPER_ENABLED={flag}")
    ENV.write_text("\n".join(lines) + ("\n" if text.endswith("\n") else ""))


def _day() -> dict:
    with sqlite3.connect(DB) as c:
        led = c.execute(
            "SELECT cash_balance, total_equity FROM portfolio_engine_ledger WHERE id=1"
        ).fetchone()
        xrp = c.execute(
            "SELECT symbol, quantity, entry_price FROM portfolio_engine_positions WHERE symbol LIKE '%XRP%'"
        ).fetchall()
        pn = c.execute("SELECT COUNT(*) FROM paper_trades").fetchone()[0]
        open_scalp = c.execute(
            "SELECT COUNT(*) FROM scalp_paper_positions WHERE status='OPEN'"
        ).fetchone()[0]
    ai = sorted(
        k for k in subprocess.check_output(["redis-cli", "KEYS", "ai_signal:day:*"], text=True).split() if k
    )
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


def _collect(since: str) -> dict:
    with sqlite3.connect(DB) as c:
        c.row_factory = sqlite3.Row
        buys = c.execute(
            "SELECT * FROM scalp_paper_trades WHERE side='BUY' AND created_at>=?", (since,)
        ).fetchall()
        sells = c.execute(
            "SELECT * FROM scalp_paper_trades WHERE side='SELL' AND created_at>=?", (since,)
        ).fetchall()
        rejects = {
            r["reason"]: r["cnt"]
            for r in c.execute(
                "SELECT reason, COUNT(*) cnt FROM scalp_rejects WHERE created_at>=? GROUP BY reason",
                (since,),
            )
        }
        led = dict(
            zip(
                ["cash", "pos", "realized", "equity"],
                c.execute(
                    "SELECT cash_balance, positions_value, realized_pnl, total_equity FROM scalp_paper_ledger WHERE id=1"
                ).fetchone()
                or (None, None, None, None),
            )
        )
        sb = [dict(r) for r in c.execute(
            "SELECT * FROM scalp_scoreboard_daily ORDER BY day DESC LIMIT 3"
        ).fetchall()]
        open_p = c.execute(
            "SELECT * FROM scalp_paper_positions WHERE status='OPEN'"
        ).fetchall()

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
        "ledger": led,
        "scoreboard": sb,
        "open": [dict(r) for r in open_p],
    }


def _wait_post_arm(since: str, arm_epoch: float) -> tuple[float, dict]:
    """Poll until sell, open+stale window, or 20 min post-arm."""
    econ = ScalpEconomics.from_env()
    deadline = arm_epoch + POST_ARM_MAX_SEC
    stale_extra = float(econ.stale_scalp_timeout_sec) + 30.0
    entry_seen_at: float | None = None

    while time.time() < deadline:
        tr = _collect(since)
        if tr["sells"] > 0:
            return time.time() - arm_epoch, tr
        if tr["buys"] > 0 and tr["open"]:
            if entry_seen_at is None:
                entry_seen_at = time.time()
            if time.time() - entry_seen_at >= stale_extra:
                tr = _collect(since)
                return time.time() - arm_epoch, tr
        time.sleep(5.0)

    return time.time() - arm_epoch, _collect(since)


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    cfg = get_scalp_config()
    if cfg.scalp_live:
        print("SCALP_LIVE must be false", file=sys.stderr)
        return 1

    r = redis.from_url(cfg.redis_url, decode_responses=True)
    since = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
    before = _day()
    (OUT / "day_before.json").write_text(json.dumps(before, indent=2))

    set_entry_armed(r, prefix=cfg.redis_key_prefix, armed=False)
    _set_paper(True)
    _start()

    assert not is_entry_armed(r, prefix=cfg.redis_key_prefix)
    after_start = _day()
    (OUT / "pre_watch.json").write_text(json.dumps(after_start, indent=2))

    trigger: list[dict] | None = None
    arm_iso: str | None = None
    arm_epoch: float | None = None

    def on_pass(rows: list[dict]) -> None:
        nonlocal trigger, arm_iso, arm_epoch
        trigger = rows
        arm_iso = _arm()
        arm_epoch = time.time()
        (OUT / "trigger.json").write_text(json.dumps(rows, indent=2, default=str))

    w0 = time.time()
    stats = watch_loop(
        interval_sec=5.0,
        max_sec=WATCH_MAX_SEC,
        log_path=OUT / "watch.jsonl",
        on_pass=on_pass,
    )
    watch_sec = time.time() - w0

    post_arm_sec = 0.0
    tr = _collect(since)
    if arm_epoch is not None:
        post_arm_sec, tr = _wait_post_arm(since, arm_epoch)

    _stop(cfg)
    after = _day()
    mem = {}
    with open("/proc/meminfo") as f:
        for line in f:
            if line.startswith(("MemAvailable:", "SwapFree:", "MemTotal:", "SwapTotal:")):
                k, v = line.split(":")
                mem[k.strip()] = int(v.split()[0])

    pass_row = next((x for x in (trigger or []) if x.get("opportunity") == "OPPORTUNITY_PASS"), None)
    close = tr.get("close")
    rep = {
        "1_watch_duration_sec": watch_sec,
        "2_opportunity_pass": pass_row,
        "2_near_pass_count": stats.near_pass_count,
        "3_arm_time_utc": arm_iso,
        "4_trade_opened": tr["buys"] > 0,
        "5_symbol": (tr["buy"] or {}).get("symbol"),
        "6_entry_price": (tr["buy"] or {}).get("price"),
        "7_exit_price": (close or {}).get("exit_price"),
        "8_exit_reason": (close or {}).get("exit_reason"),
        "9_exit_diagnostics": close,
        "10_missed_target": tr["missed"] or ["none"],
        "11_ledger_scoreboard": {"ledger": tr["ledger"], "scoreboard": tr["scoreboard"]},
        "12_rejects": tr["rejects"],
        "13_day_untouched": {
            "health": after["health"],
            "ledger_ok": before["ledger"] == after["ledger"],
            "xrp_ok": before["xrp"] == after["xrp"],
            "paper_trades_ok": before["paper_trades"] == after["paper_trades"],
            "ai_ok": before["ai_signal_day"] == after["ai_signal_day"],
            "scalp_live": after["SCALP_LIVE"],
            "scalp_paper_restored": not after["SCALP_PAPER_ENABLED"],
        },
        "14_memory_kb": mem,
        "15_safe_to_continue": (
            after["health"] == 200
            and not tr["open"]
            and not is_entry_armed(r, prefix=cfg.redis_key_prefix)
            and not after["SCALP_PAPER_ENABLED"]
            and before["ledger"] == after["ledger"]
        ),
        "post_arm_sec": post_arm_sec,
        "entry_armed_cleared": not is_entry_armed(r, prefix=cfg.redis_key_prefix),
        "no_pass": arm_iso is None,
        "round_trip_complete": tr["buys"] > 0 and tr["sells"] > 0,
    }
    (OUT / "final_report.json").write_text(json.dumps(rep, indent=2, default=str))
    print(json.dumps(rep, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
