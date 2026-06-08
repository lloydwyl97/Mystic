#!/usr/bin/env python3
"""Phase 3f soak final report — trades, diagnostics, DAY isolation checks."""

from __future__ import annotations

import json
import sqlite3
import statistics
import subprocess
import sys
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from backend.services.binance_scalp.economics import ScalpEconomics  # noqa: E402
from scripts.run_scalp_phase3b_audit import analyze_trade  # noqa: E402

DB = REPO / "mystic_trading.db"
REJECT_REASONS = (
    "NET_EDGE_BELOW_MIN",
    "SPREAD_TOO_WIDE",
    "PRICE_IMPACT_TOO_HIGH",
    "DEPTH_INSUFFICIENT",
    "MAX_OPEN_SCALPS",
    "FEE_MODEL_UNVERIFIED",
    "NET_PROFIT_TARGET_NOT_MET",
    "PROJECTED_EDGE_BELOW_REQUIRED",
    "PROJECTED_SURPLUS_BELOW_MIN",
    "INSUFFICIENT_CASH",
)


def _mem_swap() -> dict:
    mem: dict[str, int] = {}
    with open("/proc/meminfo") as f:
        for line in f:
            if line.startswith(("MemTotal:", "MemAvailable:", "SwapTotal:", "SwapFree:")):
                k, v = line.split(":")
                mem[k.strip()] = int(v.strip().split()[0])
    return mem


def _redis_mem() -> dict:
    out = subprocess.check_output(["redis-cli", "info", "memory"], text=True)
    d: dict[str, str] = {}
    for line in out.splitlines():
        if ":" in line and not line.startswith("#"):
            k, v = line.split(":", 1)
            if k in ("used_memory_human", "used_memory", "used_memory_rss"):
                d[k] = v.strip()
    return d


def _day_snapshot(conn: sqlite3.Connection) -> dict:
    ledger = conn.execute(
        "SELECT cash_balance, total_equity FROM portfolio_engine_ledger WHERE id=1"
    ).fetchone()
    positions = [
        dict(zip(["symbol", "quantity", "entry_price"], r))
        for r in conn.execute(
            "SELECT symbol, quantity, entry_price FROM portfolio_engine_positions"
        ).fetchall()
    ]
    paper_count = conn.execute("SELECT COUNT(*) FROM paper_trades").fetchone()[0]
    return {
        "ledger": {"cash_balance": ledger[0], "total_equity": ledger[1]} if ledger else None,
        "positions": positions,
        "paper_trades_count": paper_count,
    }


def _entry_diag_from_buy(conn: sqlite3.Connection, buy_tid: str) -> dict:
    buy = conn.execute(
        "SELECT diagnostics_json FROM scalp_paper_trades WHERE trade_id=? AND side='BUY'",
        (buy_tid,),
    ).fetchone()
    pos = conn.execute(
        "SELECT diagnostics_json FROM scalp_paper_positions WHERE trade_id=?",
        (buy_tid,),
    ).fetchone()
    raw = None
    if pos and pos[0]:
        raw = pos[0]
    elif buy and buy[0]:
        raw = buy[0]
    if not raw:
        return {}
    data = json.loads(raw)
    ed = data.get("entry_diagnostics") or {}
    pf = data.get("entry_preflight") or data.get("preflight") or {}
    reach = data.get("reachability") or pf.get("reachability") or {}
    return {
        "projected_gross": ed.get("projected_gross_move_pct") or reach.get("projected_gross_move_pct"),
        "required_gross": ed.get("required_gross_move_pct") or reach.get("required_gross_move_pct"),
        "projected_surplus": ed.get("projected_surplus_pct") or reach.get("projected_surplus_pct"),
        "momentum_gross_estimate_pct": ed.get("momentum_gross_estimate_pct") or reach.get("momentum_gross_estimate_pct"),
        "mid_change_15s": ed.get("mid_change_15s") or reach.get("mid_change_15s"),
        "mid_change_30s": ed.get("mid_change_30s") or reach.get("mid_change_30s"),
        "mid_change_60s": ed.get("mid_change_60s") or reach.get("mid_change_60s"),
        "bid_change_15s": ed.get("bid_change_15s") or reach.get("bid_change_15s"),
        "bid_change_30s": ed.get("bid_change_30s") or reach.get("bid_change_30s"),
        "bid_change_60s": ed.get("bid_change_60s") or reach.get("bid_change_60s"),
        "up_tick_count": ed.get("up_tick_count") or reach.get("up_tick_count"),
        "breakout_strength_pct": ed.get("breakout_strength_pct") or reach.get("breakout_strength_pct"),
        "spread": pf.get("spread_pct"),
        "buy_impact": pf.get("buy_impact_pct"),
        "sell_impact": pf.get("sell_impact_pct"),
    }


def build_report(
    *,
    soak_start: str,
    duration_sec: float,
    scan_ticks: int,
    baseline: dict,
    after: dict,
    hourly_memory: list[dict],
    journal_errors: str,
) -> dict:
    econ = ScalpEconomics.from_env()
    with sqlite3.connect(DB) as conn:
        conn.row_factory = sqlite3.Row
        new_buys = conn.execute(
            "SELECT * FROM scalp_paper_trades WHERE side='BUY' AND created_at >= ? ORDER BY id",
            (soak_start,),
        ).fetchall()
        new_sells = conn.execute(
            "SELECT * FROM scalp_paper_trades WHERE side='SELL' AND created_at >= ? ORDER BY id",
            (soak_start,),
        ).fetchall()
        open_pos = conn.execute(
            "SELECT * FROM scalp_paper_positions WHERE status='OPEN'"
        ).fetchall()
        reject_rows = conn.execute(
            """
            SELECT reason, COUNT(*) as cnt FROM scalp_rejects
            WHERE created_at >= ? GROUP BY reason ORDER BY cnt DESC
            """,
            (soak_start,),
        ).fetchall()
        rejects = {str(r["reason"]): int(r["cnt"]) for r in reject_rows}

    per_trade = []
    with sqlite3.connect(DB) as conn:
        conn.row_factory = sqlite3.Row
        for sell in new_sells:
            t = analyze_trade(conn, sell, econ)
            buy_tid = str(sell["trade_id"]).replace("_SELL", "")
            t["entry_gate"] = _entry_diag_from_buy(conn, buy_tid)
            per_trade.append(t)

    pnls = [float(s["pnl_usd"] or 0) for s in new_sells]
    wins = sum(1 for p in pnls if p > 0)
    losses = sum(1 for p in pnls if p <= 0)
    exit_reasons: dict[str, int] = {}
    for s in new_sells:
        r = str(s["exit_reason"] or "UNKNOWN")
        exit_reasons[r] = exit_reasons.get(r, 0) + 1

    btc_buys = sum(1 for b in new_buys if b["symbol"] == "BTCUSDT")
    eth_buys = sum(1 for b in new_buys if b["symbol"] == "ETHUSDT")
    btc_sells = [t for t in per_trade if t["symbol"] == "BTCUSDT"]
    eth_sells = [t for t in per_trade if t["symbol"] == "ETHUSDT"]

    def sym_stats(trades: list[dict]) -> dict:
        if not trades:
            return {"trades": 0, "wins": 0, "losses": 0, "pnl": 0.0, "target_reached_rate": None}
        pn = [t["net_pnl_usd"] for t in trades]
        return {
            "trades": len(trades),
            "wins": sum(1 for p in pn if p > 0),
            "losses": sum(1 for p in pn if p <= 0),
            "pnl": sum(pn),
            "avg_max_favorable_pct": statistics.mean([t["max_favorable_move_pct"] for t in trades]),
            "target_reached_rate": sum(1 for t in trades if t["price_reached_profit_target"]) / len(trades),
        }

    day_before = baseline.get("day", {})
    day_after = after.get("day", {})
    xrp_before = next(
        (p for p in day_before.get("positions", []) if "XRP" in str(p.get("symbol", "")).upper()),
        None,
    )
    xrp_after = next(
        (p for p in day_after.get("positions", []) if "XRP" in str(p.get("symbol", "")).upper()),
        None,
    )
    day_untouched = (
        day_before.get("paper_trades_count") == day_after.get("paper_trades_count")
        and day_before.get("ledger") == day_after.get("ledger")
        and xrp_before == xrp_after
        and baseline.get("ai_signal_day_keys") == after.get("ai_signal_day_keys")
    )

    mem_before = baseline.get("memory_kb", {})
    mem_after = after.get("memory_kb", {})
    mem_avail_before = mem_before.get("MemAvailable", 0)
    mem_avail_after = mem_after.get("MemAvailable", 0)
    mem_stable = mem_avail_after >= mem_avail_before * 0.85 if mem_avail_before else True

    safe = (
        day_untouched
        and after.get("day_health", {}).get("http_code") == 200
        and not journal_errors.strip()
        and baseline.get("scalp_ledger") != {}  # sanity
    )

    return {
        "1_runtime_duration_sec": duration_sec,
        "1_runtime_duration_h": round(duration_sec / 3600, 2),
        "2_trades_opened": len(new_buys),
        "2_trades_closed": len(new_sells),
        "3_profit_target_exits": exit_reasons.get("NET_PROFIT_TARGET", 0),
        "3_stale_timeout_exits": exit_reasons.get("STALE_SCALP_TIMEOUT", 0),
        "4_wins": wins,
        "4_losses": losses,
        "4_pnl_usd": sum(pnls),
        "5_rejects_by_reason": rejects,
        "5_tracked_rejects": {r: rejects.get(r, 0) for r in REJECT_REASONS},
        "6_per_trade_diagnostics": per_trade,
        "7_btc": sym_stats(btc_sells) | {"buys": btc_buys},
        "7_eth": sym_stats(eth_sells) | {"buys": eth_buys},
        "8_open_position_at_end": [dict(r) for r in open_pos],
        "9_memory_before_kb": mem_before,
        "9_memory_after_kb": mem_after,
        "9_memory_hourly": hourly_memory,
        "9_swap_before_kb": {
            "SwapTotal": mem_before.get("SwapTotal"),
            "SwapFree": mem_before.get("SwapFree"),
        },
        "9_swap_after_kb": {
            "SwapTotal": mem_after.get("SwapTotal"),
            "SwapFree": mem_after.get("SwapFree"),
        },
        "9_redis_memory": _redis_mem(),
        "10_day_health": after.get("day_health"),
        "10_day_untouched": day_untouched,
        "10_day_before": day_before,
        "10_day_after": day_after,
        "10_ai_signal_day_unchanged": baseline.get("ai_signal_day_keys") == after.get("ai_signal_day_keys"),
        "11_scan_count": scan_ticks,
        "11_journal_errors": journal_errors.strip() or None,
        "11_memory_stable": mem_stable,
        "12_safe_to_continue": safe and mem_stable,
        "meta": {
            "soak_start": soak_start,
            "collected_at": datetime.now(timezone.utc).isoformat(),
            "scalp_ledger_after": after.get("scalp_ledger"),
        },
    }


def main() -> int:
    if len(sys.argv) < 2:
        print("usage: run_scalp_phase3f_soak_report.py REPORT_INPUT.json", file=sys.stderr)
        return 1
    payload = json.loads(Path(sys.argv[1]).read_text())
    report = build_report(**payload)
    print(json.dumps(report, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
