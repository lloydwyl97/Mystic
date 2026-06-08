#!/usr/bin/env python3
"""Phase 3h — exit-fix validation soak report."""

from __future__ import annotations

import json
import sqlite3
import subprocess
import sys
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

DB = REPO / "mystic_trading.db"


def _mem() -> dict:
    out: dict = {}
    with open("/proc/meminfo") as f:
        for line in f:
            if line.startswith(("MemAvailable:", "SwapFree:", "SwapTotal:")):
                k, v = line.split(":")
                out[k.strip()] = int(v.split()[0])
    return out


def build_report(*, soak_start: str, duration_sec: float) -> dict:
    with sqlite3.connect(DB) as conn:
        conn.row_factory = sqlite3.Row
        buys = conn.execute(
            "SELECT * FROM scalp_paper_trades WHERE side='BUY' AND created_at >= ? ORDER BY id",
            (soak_start,),
        ).fetchall()
        sells = conn.execute(
            "SELECT * FROM scalp_paper_trades WHERE side='SELL' AND created_at >= ? ORDER BY id",
            (soak_start,),
        ).fetchall()
        open_pos = conn.execute(
            "SELECT * FROM scalp_paper_positions WHERE status='OPEN'"
        ).fetchall()
        rejects = {
            r["reason"]: r["cnt"]
            for r in conn.execute(
                "SELECT reason, COUNT(*) cnt FROM scalp_rejects WHERE created_at >= ? GROUP BY reason",
                (soak_start,),
            )
        }
        day_ledger = conn.execute(
            "SELECT cash_balance, total_equity FROM portfolio_engine_ledger WHERE id=1"
        ).fetchone()
        xrp = conn.execute(
            "SELECT symbol, quantity, entry_price FROM portfolio_engine_positions WHERE symbol LIKE '%XRP%'"
        ).fetchall()
        paper_n = conn.execute("SELECT COUNT(*) FROM paper_trades").fetchone()[0]

    baseline_path = Path("/tmp/scalp_phase3h/baseline.json")
    day_before = json.loads(baseline_path.read_text())["day"] if baseline_path.exists() else {}

    closes = []
    profit_exits = stale_exits = 0
    missed_target_notes = []
    for sell in sells:
        diag = json.loads(sell["diagnostics_json"] or "{}")
        exit_gate = diag.get("exit_gate") or {}
        pf = diag.get("preflight") or {}
        buy_tid = str(sell["trade_id"]).replace("_SELL", "")
        buy = next((b for b in buys if b["trade_id"] == buy_tid), None)
        entry_p = float(sell["entry_price"] or (buy["price"] if buy else 0))
        reason = sell["exit_reason"] or "UNKNOWN"
        if reason == "NET_PROFIT_TARGET":
            profit_exits += 1
        elif reason == "STALE_SCALP_TIMEOUT":
            stale_exits += 1
        net_pct = exit_gate.get("net_pct", pf.get("expected_net_edge_pct"))
        target = exit_gate.get("target_pct")
        if reason == "STALE_SCALP_TIMEOUT" and net_pct is not None and target is not None:
            if float(net_pct) >= float(target):
                missed_target_notes.append(
                    f"{sell['trade_id']}: stale but net_pct={net_pct} >= target at close"
                )
        closes.append(
            {
                "trade_id": buy_tid,
                "symbol": sell["symbol"],
                "entry_price": entry_p,
                "exit_price": float(sell["price"]),
                "best_bid": exit_gate.get("current_bid"),
                "expected_sell_fill": exit_gate.get("expected_sell_fill"),
                "executable_exit_net_pct": net_pct,
                "target_pct": target,
                "exit_reason": reason,
                "pnl_usd": float(sell["pnl_usd"] or 0),
                "exit_gate": exit_gate,
                "has_exit_gate": bool(exit_gate),
            }
        )

    pnls = [c["pnl_usd"] for c in closes]
    wins = sum(1 for p in pnls if p > 0)
    losses = sum(1 for p in pnls if p <= 0)
    btc = [c for c in closes if c["symbol"] == "BTCUSDT"]
    eth = [c for c in closes if c["symbol"] == "ETHUSDT"]

    try:
        with urllib.request.urlopen("http://127.0.0.1:8000/health", timeout=5) as r:
            health = {"http_code": r.status}
    except Exception as exc:
        health = {"http_code": 0, "error": str(exc)}

    ai = sorted(
        k
        for k in subprocess.check_output(["redis-cli", "KEYS", "ai_signal:day:*"], text=True).split()
        if k
    )
    ai_before = baseline_path.exists() and json.loads(baseline_path.read_text()).get(
        "ai_signal_day_keys", []
    )

    phase3f_profit_rate = 1 / 8 if 8 else 0
    phase3h_profit_rate = profit_exits / len(sells) if sells else None

    return {
        "1_runtime_duration_sec": duration_sec,
        "1_runtime_duration_min": round(duration_sec / 60, 1),
        "2_trades_opened": len(buys),
        "2_trades_closed": len(sells),
        "3_profit_target_exits": profit_exits,
        "3_stale_timeout_exits": stale_exits,
        "4_wins": wins,
        "4_losses": losses,
        "4_pnl_usd": sum(pnls),
        "5_exit_diagnostics_summary": closes,
        "5_all_have_exit_gate": all(c["has_exit_gate"] for c in closes) if closes else True,
        "6_missed_target_check": missed_target_notes or ["none at close tick"],
        "7_rejects_by_reason": rejects,
        "8_btc": {
            "closed": len(btc),
            "pnl": sum(c["pnl_usd"] for c in btc),
            "profit_exits": sum(1 for c in btc if c["exit_reason"] == "NET_PROFIT_TARGET"),
        },
        "8_eth": {
            "closed": len(eth),
            "pnl": sum(c["pnl_usd"] for c in eth),
            "profit_exits": sum(1 for c in eth if c["exit_reason"] == "NET_PROFIT_TARGET"),
        },
        "9_open_position_at_end": [dict(r) for r in open_pos],
        "10_day_untouched": {
            "health": health,
            "ledger_unchanged": day_before.get("ledger") == {
                "cash_balance": day_ledger[0],
                "total_equity": day_ledger[1],
            }
            if day_ledger and day_before.get("ledger")
            else None,
            "xrp": [
                {"symbol": r[0], "quantity": r[1], "entry_price": r[2]} for r in xrp
            ],
            "paper_trades_count": paper_n,
            "paper_trades_vs_baseline": paper_n == day_before.get("paper_trades_count"),
            "ai_signal_day_unchanged": ai == ai_before,
        },
        "11_memory_kb": _mem(),
        "12_safe_to_continue": health.get("http_code") == 200 and len(open_pos) == 0,
        "phase3f_comparison": {
            "phase3f_profit_exits": 1,
            "phase3f_total_closed": 8,
            "phase3f_profit_rate": phase3f_profit_rate,
            "phase3h_profit_rate": phase3h_profit_rate,
            "improved": (
                profit_exits > 1
                if sells
                else None
            ),
            "zero_trades": len(sells) == 0,
        },
        "soak_start": soak_start,
    }


def main() -> int:
    soak_start = Path("/tmp/scalp_phase3h/start.txt").read_text().splitlines()[0]
    start_epoch = int(Path("/tmp/scalp_phase3h/start.txt").read_text().splitlines()[1])
    duration = __import__("time").time() - start_epoch
    print(json.dumps(build_report(soak_start=soak_start, duration_sec=duration), indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
