#!/usr/bin/env python3
"""Collect Binance scalp soak metrics — scalp_* tables only."""

from __future__ import annotations

import json
import sqlite3
import subprocess
import sys
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

DB = Path("/home/mystic/mystic/mystic_trading.db")
REJECT_REASONS = (
    "NET_EDGE_BELOW_MIN",
    "SPREAD_TOO_WIDE",
    "PRICE_IMPACT_TOO_HIGH",
    "DEPTH_INSUFFICIENT",
    "MAX_OPEN_SCALPS",
    "FEE_MODEL_UNVERIFIED",
)


def _mem_swap() -> dict:
    mem = {}
    with open("/proc/meminfo") as f:
        for line in f:
            if line.startswith(("MemTotal:", "MemAvailable:", "SwapTotal:", "SwapFree:")):
                k, v = line.split(":")
                mem[k.strip()] = int(v.strip().split()[0])  # kB
    return mem


def _redis_scalp_keys() -> list[str]:
    out = subprocess.check_output(["redis-cli", "KEYS", "scalp:*"], text=True)
    return sorted(k for k in out.strip().split("\n") if k)


def _day_health() -> dict:
    try:
        with urllib.request.urlopen("http://127.0.0.1:8000/health", timeout=5) as r:
            return {"http_code": r.status, "body": r.read().decode()[:200]}
    except Exception as exc:
        return {"http_code": 0, "error": str(exc)}


def _day_snapshot(conn: sqlite3.Connection) -> dict:
    ledger = conn.execute("SELECT cash_balance, total_equity FROM portfolio_engine_ledger WHERE id=1").fetchone()
    positions = [dict(r) for r in conn.execute("SELECT symbol, quantity, entry_price FROM portfolio_engine_positions")]
    paper_count = conn.execute("SELECT COUNT(*) FROM paper_trades").fetchone()[0]
    return {
        "ledger": {"cash_balance": ledger[0], "total_equity": ledger[1]} if ledger else None,
        "positions": positions,
        "paper_trades_count": paper_count,
    }


def collect(*, since_ts: str | None = None) -> dict:
    with sqlite3.connect(DB) as conn:
        conn.row_factory = sqlite3.Row
        ledger = dict(conn.execute("SELECT * FROM scalp_paper_ledger WHERE id=1").fetchone())
        scoreboard = [dict(r) for r in conn.execute("SELECT * FROM scalp_scoreboard_daily ORDER BY day")]
        positions = [dict(r) for r in conn.execute("SELECT * FROM scalp_paper_positions ORDER BY id")]
        trades = [dict(r) for r in conn.execute("SELECT * FROM scalp_paper_trades ORDER BY id")]
        open_pos = [dict(r) for r in conn.execute("SELECT * FROM scalp_paper_positions WHERE status='OPEN'")]

        reject_q = "SELECT reason, COUNT(*) as cnt FROM scalp_rejects"
        params: tuple = ()
        if since_ts:
            reject_q += " WHERE created_at >= ?"
            params = (since_ts,)
        reject_q += " GROUP BY reason ORDER BY cnt DESC"
        rejects = {r["reason"]: r["cnt"] for r in conn.execute(reject_q, params)}

        day = _day_snapshot(conn)

    ai_day = subprocess.check_output(["redis-cli", "KEYS", "ai_signal:day:*"], text=True)
    ai_ctx = subprocess.check_output(["redis-cli", "KEYS", "ai_context:*"], text=True)

    sells = [t for t in trades if t["side"] == "SELL"]
    buys = [t for t in trades if t["side"] == "BUY"]
    if since_ts:
        sells = [t for t in sells if (t.get("created_at") or "") >= since_ts[:10]]
        buys = [t for t in buys if (t.get("created_at") or "") >= since_ts[:10]]

    return {
        "collected_at": datetime.now(timezone.utc).isoformat(),
        "since_ts": since_ts,
        "scalp_ledger": {k: ledger[k] for k in ("cash_balance", "positions_value", "realized_pnl", "total_equity")},
        "scoreboard": scoreboard,
        "open_positions": open_pos,
        "all_positions": positions,
        "buys": len(buys),
        "sells": len(sells),
        "trades_detail": trades,
        "rejects": rejects,
        "tracked_rejects": {r: rejects.get(r, 0) for r in REJECT_REASONS},
        "redis_scalp_keys": _redis_scalp_keys(),
        "ai_signal_day_keys": sorted(k for k in ai_day.strip().split("\n") if k),
        "ai_context_keys": sorted(k for k in ai_ctx.strip().split("\n") if k),
        "memory_kb": _mem_swap(),
        "day_health": _day_health(),
        "day": day,
    }


def diff_report(before: dict, after: dict, *, duration_sec: float, scan_ticks: int) -> dict:
    b_trades = before["trades_detail"]
    a_trades = after["trades_detail"]
    new_trades = a_trades[len(b_trades) :]

    b_rejects_total = sum(before["rejects"].values())
    a_rejects_total = sum(after["rejects"].values())

    def reject_delta(reason: str) -> int:
        return after["rejects"].get(reason, 0) - before["rejects"].get(reason, 0)

    sells = [t for t in new_trades if t["side"] == "SELL"]
    buys = [t for t in new_trades if t["side"] == "BUY"]
    pnls = [float(t["pnl_usd"]) for t in sells if t.get("pnl_usd") is not None]
    wins = sum(1 for p in pnls if p > 0)
    losses = sum(1 for p in pnls if p <= 0)

    exit_reasons: dict[str, int] = {}
    hold_times: list[float] = []
    btc_buys = btc_sells = eth_buys = eth_sells = 0
    for t in new_trades:
        sym = t["symbol"]
        if t["side"] == "BUY":
            if sym == "BTCUSDT":
                btc_buys += 1
            elif sym == "ETHUSDT":
                eth_buys += 1
        else:
            if sym == "BTCUSDT":
                btc_sells += 1
            elif sym == "ETHUSDT":
                eth_sells += 1
            reason = t.get("exit_reason") or "UNKNOWN"
            exit_reasons[reason] = exit_reasons.get(reason, 0) + 1
            tid = t["trade_id"].replace("_SELL", "")
            buy = next((x for x in a_trades if x["trade_id"] == tid and x["side"] == "BUY"), None)
            if buy and buy.get("created_at") and t.get("created_at"):
                try:
                    fmt = "%Y-%m-%d %H:%M:%S"
                    t0 = datetime.strptime(buy["created_at"], fmt).replace(tzinfo=timezone.utc)
                    t1 = datetime.strptime(t["created_at"], fmt).replace(tzinfo=timezone.utc)
                    hold_times.append((t1 - t0).total_seconds())
                except ValueError:
                    pass

    return {
        "duration_sec": duration_sec,
        "scan_ticks_estimated": scan_ticks,
        "buys_new": len(buys),
        "sells_new": len(sells),
        "rejects_new_total": a_rejects_total - b_rejects_total,
        "rejects_by_reason_delta": {r: reject_delta(r) for r in REJECT_REASONS},
        "all_rejects_delta": {k: after["rejects"].get(k, 0) - before["rejects"].get(k, 0) for k in set(before["rejects"]) | set(after["rejects"])},
        "per_trade_pnl": pnls,
        "wins": wins,
        "losses": losses,
        "total_pnl": sum(pnls),
        "avg_hold_sec": sum(hold_times) / len(hold_times) if hold_times else None,
        "hold_times_sec": hold_times,
        "exit_reasons": exit_reasons,
        "btc": {"buys": btc_buys, "sells": btc_sells},
        "eth": {"buys": eth_buys, "sells": eth_sells},
        "ledger_before": before["scalp_ledger"],
        "ledger_after": after["scalp_ledger"],
        "scoreboard_before": before["scoreboard"],
        "scoreboard_after": after["scoreboard"],
        "open_at_end": after["open_positions"],
        "memory_before_kb": before["memory_kb"],
        "memory_after_kb": after["memory_kb"],
        "redis_scalp_before": before["redis_scalp_keys"],
        "redis_scalp_after": after["redis_scalp_keys"],
        "day_before": before["day"],
        "day_after": after["day"],
        "day_health_before": before["day_health"],
        "day_health_after": after["day_health"],
        "ai_signal_day_before": before["ai_signal_day_keys"],
        "ai_signal_day_after": after["ai_signal_day_keys"],
        "ai_context_before": before["ai_context_keys"],
        "ai_context_after": after["ai_context_keys"],
    }


def main() -> int:
    if len(sys.argv) < 2:
        print("usage: collect_scalp_soak_metrics.py baseline|after|report", file=sys.stderr)
        return 1
    cmd = sys.argv[1]
    if cmd == "baseline":
        print(json.dumps(collect(), indent=2))
        return 0
    if cmd == "after":
        since = sys.argv[2] if len(sys.argv) > 2 else None
        print(json.dumps(collect(since_ts=since), indent=2))
        return 0
    if cmd == "report":
        before = json.loads(Path(sys.argv[2]).read_text())
        after = json.loads(Path(sys.argv[3]).read_text())
        duration = float(sys.argv[4])
        ticks = int(sys.argv[5])
        journal_errors = Path(sys.argv[6]).read_text() if len(sys.argv) > 6 else ""
        rep = diff_report(before, after, duration_sec=duration, scan_ticks=ticks)
        rep["journal_errors"] = journal_errors.strip() or None
        print(json.dumps(rep, indent=2, default=str))
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
