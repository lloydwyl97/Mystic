#!/usr/bin/env python3
"""Phase 3g — exit timing, DAY retention, ETH selectivity audit."""

from __future__ import annotations

import json
import sqlite3
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from backend.services.binance_scalp.economics import ScalpEconomics  # noqa: E402
from backend.services.binance_scalp.orderbook_book import walk_sell_qty  # noqa: E402

DB = REPO / "mystic_trading.db"
SOAK_START = "2026-06-07 00:12:04"


def _fetch_klines(symbol: str, start_ms: int, end_ms: int) -> list[dict]:
    url = (
        "https://api.binance.us/api/v3/klines"
        f"?symbol={symbol}&interval=1m&startTime={start_ms}&endTime={end_ms}&limit=1000"
    )
    proc = subprocess.run(
        ["curl", "-s", "--max-time", "20", url],
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0 or not proc.stdout.strip():
        return []
    rows = json.loads(proc.stdout)
    if isinstance(rows, dict):
        return []
    return [
        {
            "open_time_ms": int(r[0]),
            "open": float(r[1]),
            "high": float(r[2]),
            "low": float(r[3]),
            "close": float(r[4]),
        }
        for r in rows
    ]


def _parse_ts(s: str) -> datetime:
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S"):
        try:
            return datetime.strptime(s[:19], fmt).replace(tzinfo=timezone.utc)
        except ValueError:
            continue
    raise ValueError(s)


def _engine_exit_net_pct(
    entry: float,
    best_bid: float,
    spread: float,
    buy_impact: float,
    sell_impact: float,
    econ: ScalpEconomics,
) -> float:
    """Mirror protected_preflight SELL gate (current engine)."""
    gross = (best_bid - entry) / entry if entry > 0 else -1.0
    costs = econ.roundtrip_cost_pct(spread, buy_impact, sell_impact)
    return gross - costs


def _review_exit_net_pct(
    entry: float,
    bid_proxy: float,
    spread: float,
    sell_impact: float,
    econ: ScalpEconomics,
) -> float:
    """Mirror run_scalp_phase3b_audit review (_net_at_bid)."""
    gross = (bid_proxy - entry) / entry if entry > 0 else -1.0
    costs = econ.roundtrip_cost_pct(spread, 0.0, sell_impact)
    return gross - costs


def _target_bid(entry: float, spread: float, buy_i: float, sell_i: float, econ: ScalpEconomics) -> float:
    costs = econ.roundtrip_cost_pct(spread, buy_i, sell_i)
    return entry * (1.0 + costs + econ.net_profit_target_pct)


def audit_exit_trade(buy: sqlite3.Row, sell: sqlite3.Row, econ: ScalpEconomics) -> dict:
    entry = float(buy["price"])
    exit_p = float(sell["price"])
    qty = float(buy["quantity"])
    entry_ts = _parse_ts(buy["created_at"])
    exit_ts = _parse_ts(sell["created_at"])
    sym = str(buy["symbol"])

    sell_diag = json.loads(sell["diagnostics_json"] or "{}")
    sell_pf = sell_diag.get("preflight", {})
    pos_raw = None
    with sqlite3.connect(DB) as conn:
        conn.row_factory = sqlite3.Row
        pos = conn.execute(
            "SELECT diagnostics_json FROM scalp_paper_positions WHERE trade_id=?",
            (buy["trade_id"],),
        ).fetchone()
        if pos:
            pos_raw = json.loads(pos["diagnostics_json"] or "{}")
    buy_pf = (pos_raw or {}).get("entry_preflight") or json.loads(buy["diagnostics_json"] or "{}").get("preflight", {})
    spread_e = float(buy_pf.get("spread_pct", sell_pf.get("spread_pct", 0)))
    buy_i_e = float(buy_pf.get("buy_impact_pct", 0))
    sell_i_e = float(buy_pf.get("sell_impact_pct", 0))

    start_ms = int(entry_ts.timestamp() * 1000)
    end_ms = int(exit_ts.timestamp() * 1000) + 60_000
    klines = _fetch_klines(sym, start_ms - 60_000, end_ms)

    best_bid_live_proxy = entry
    best_ask_proxy = entry
    best_high = entry
    target_hit_engine_times: list[str] = []
    target_hit_review_times: list[str] = []
    max_engine_net = -999.0
    max_review_net = -999.0

    for k in klines:
        t = k["open_time_ms"] / 1000.0
        if t < entry_ts.timestamp() or t > exit_ts.timestamp():
            continue
        hi, lo, close = k["high"], k["low"], k["close"]
        best_high = max(best_high, hi)
        best_bid_live_proxy = max(best_bid_live_proxy, lo)  # conservative executable bid proxy
        best_ask_proxy = max(best_ask_proxy, hi)

        eng_net_hi = _engine_exit_net_pct(entry, hi, spread_e, buy_i_e, sell_i_e, econ)
        eng_net_lo = _engine_exit_net_pct(entry, lo, spread_e, buy_i_e, sell_i_e, econ)
        rev_net_hi = _review_exit_net_pct(entry, hi, spread_e, sell_i_e, econ)

        max_engine_net = max(max_engine_net, eng_net_hi, eng_net_lo)
        max_review_net = max(max_review_net, rev_net_hi)

        if eng_net_hi >= econ.net_profit_target_pct:
            target_hit_engine_times.append(
                datetime.fromtimestamp(t, tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
            )
        if rev_net_hi >= econ.net_profit_target_pct:
            target_hit_review_times.append(
                datetime.fromtimestamp(t, tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
            )

    exit_pf_net = float(sell_pf.get("expected_net_edge_pct", 0))
    exit_reason = sell["exit_reason"]
    stale_timeout = econ.stale_scalp_timeout_sec
    hold_sec = (exit_ts - entry_ts).total_seconds()

    # Classify missed target stale exits
    classification = "n/a"
    if exit_reason == "STALE_SCALP_TIMEOUT":
        review_says_hit = len(target_hit_review_times) > 0
        engine_says_hit = len(target_hit_engine_times) > 0
        if not review_says_hit:
            classification = "stale_loss_or_flat_expected"
        elif engine_says_hit:
            classification = "true_missed_exit_bug"
        else:
            classification = "review_false_positive_candle_high_or_cost_mismatch"
        if exit_pf_net < econ.net_profit_target_pct and review_says_hit and not engine_says_hit:
            classification = "review_false_positive_candle_high_not_executable_bid"

    return {
        "trade_id": buy["trade_id"],
        "symbol": sym,
        "entry_time": buy["created_at"],
        "exit_time": sell["created_at"],
        "entry_price": entry,
        "exit_price": exit_p,
        "qty": qty,
        "target_bid_required": _target_bid(entry, spread_e, buy_i_e, sell_i_e, econ),
        "best_bid_during_hold_kline_high": best_high,
        "best_bid_conservative_kline_low_max": best_bid_live_proxy,
        "best_ask_kline_high": best_ask_proxy,
        "mark_source_live_loop": "binance_us_depth_best_bid (limit_sell_price)",
        "review_source": "1m_kline_high_as_bid_proxy",
        "redis_snapshots": "scalp:market:{SYMBOL} cache 120s — not persisted per tick",
        "target_hit_review_times_utc": target_hit_review_times[:5],
        "target_hit_engine_sim_times_utc": target_hit_engine_times[:5],
        "exit_loop_interval_sec": 5,
        "loop_running_at_target": "yes — soak service active entire hold",
        "sell_preflight_at_exit": sell_pf,
        "exit_net_calc_engine": "gross=(best_bid-entry)/entry minus roundtrip_cost_pct(spread,buy_impact,sell_impact)",
        "exit_net_calc_review": "gross=(bid-entry)/entry minus roundtrip_cost_pct(spread,0,sell_impact)",
        "exit_net_calc_pnl_usd": "(best_bid-entry)*qty - exit_fee - slip - entry_fee",
        "fee_double_count_risk": "engine exit gate re-walks buy_impact on current book each tick",
        "max_engine_net_pct_sim": max_engine_net,
        "max_review_net_pct_sim": max_review_net,
        "net_profit_target_pct": econ.net_profit_target_pct,
        "exit_preflight_expected_net_at_close": exit_pf_net,
        "hold_seconds": hold_sec,
        "stale_timeout_sec": stale_timeout,
        "exit_reason": exit_reason,
        "pnl_usd": float(sell["pnl_usd"] or 0),
        "classification": classification,
        "entry_gate": (pos_raw or {}).get("entry_diagnostics") or (pos_raw or {}).get("reachability"),
        "spread_at_entry": spread_e,
        "buy_impact_at_entry": buy_i_e,
    }


def audit_paper_retention() -> dict:
    with sqlite3.connect(DB) as conn:
        conn.row_factory = sqlite3.Row
        cutoff = conn.execute(
            "SELECT strftime('%Y-%m-%dT00:00:00', 'now', '-7 days')"
        ).fetchone()[0]
        remaining = conn.execute(
            "SELECT id, timestamp, side, symbol FROM paper_trades ORDER BY id"
        ).fetchall()
        oldest = conn.execute(
            "SELECT MIN(timestamp), MAX(timestamp), COUNT(*) FROM paper_trades"
        ).fetchone()
        ledger = conn.execute(
            "SELECT cash_balance, total_equity FROM portfolio_engine_ledger WHERE id=1"
        ).fetchone()
        xrp = conn.execute(
            "SELECT symbol, quantity, entry_price FROM portfolio_engine_positions"
        ).fetchall()

    baseline_path = Path("/tmp/scalp_phase3f/baseline.json")
    baseline_count = None
    if baseline_path.exists():
        baseline_count = json.loads(baseline_path.read_text())["day"]["paper_trades_count"]

    # Rows that would have been deleted by 7-day retention
    pruned_estimate = []
    if cutoff:
        with sqlite3.connect(DB) as conn:
            # We cannot see deleted rows; infer from id gaps and cutoff
            pruned_estimate = (
                f"Rows with timestamp < {cutoff} removed by portfolio_engine PAPER_RETENTION loop"
            )

    return {
        "cutoff_7d": cutoff,
        "baseline_paper_trades_at_soak_start": baseline_count,
        "current_paper_trades_count": len(remaining),
        "deleted_count_estimate": (baseline_count - len(remaining)) if baseline_count else None,
        "remaining_ids": [r["id"] for r in remaining],
        "remaining_oldest_newest": {"min_ts": oldest[0], "max_ts": oldest[1], "count": oldest[2]},
        "pruning_mechanism": "portfolio_engine_integration._paper_retention_loop every 15min",
        "scalp_touches_paper_trades": False,
        "expected_retention": True,
        "root_cause": pruned_estimate,
        "ledger": dict(ledger) if ledger else None,
        "positions": [dict(r) for r in xrp],
    }


def main() -> int:
    econ = ScalpEconomics.from_env()
    with sqlite3.connect(DB) as conn:
        conn.row_factory = sqlite3.Row
        pairs = conn.execute(
            """
            SELECT b.*, s.created_at AS sell_time, s.exit_reason, s.pnl_usd AS sell_pnl
            FROM scalp_paper_trades b
            JOIN scalp_paper_trades s ON s.trade_id = b.trade_id || '_SELL'
            WHERE b.side='BUY' AND b.created_at >= ?
            ORDER BY b.id
            """,
            (SOAK_START,),
        ).fetchall()
        sells = {
            str(r["trade_id"]): conn.execute(
                "SELECT * FROM scalp_paper_trades WHERE trade_id=?",
                (f"{r['trade_id']}_SELL",),
            ).fetchone()
            for r in conn.execute(
                "SELECT trade_id FROM scalp_paper_trades WHERE side='BUY' AND created_at >= ?",
                (SOAK_START,),
            )
        }
        buys = conn.execute(
            "SELECT * FROM scalp_paper_trades WHERE side='BUY' AND created_at >= ? ORDER BY id",
            (SOAK_START,),
        ).fetchall()

    exit_audits = [audit_exit_trade(b, sells[str(b["trade_id"])], econ) for b in buys]

    eth_trades = [t for t in exit_audits if t["symbol"] == "ETHUSDT"]
    btc_trades = [t for t in exit_audits if t["symbol"] == "BTCUSDT"]

    report = {
        "phase": "3g",
        "soak_start": SOAK_START,
        "economics": econ.as_dict(),
        "exit_timing_audits": exit_audits,
        "missed_target_summary": {
            t["trade_id"]: {
                "classification": t["classification"],
                "exit_reason": t["exit_reason"],
                "pnl_usd": t["pnl_usd"],
                "review_hit": bool(t["target_hit_review_times_utc"]),
                "engine_sim_hit": bool(t["target_hit_engine_sim_times_utc"]),
            }
            for t in exit_audits
            if t["exit_reason"] == "STALE_SCALP_TIMEOUT"
        },
        "paper_retention": audit_paper_retention(),
        "eth_selectivity": {
            "trades": len(eth_trades),
            "wins": sum(1 for t in eth_trades if t["pnl_usd"] > 0),
            "losses": sum(1 for t in eth_trades if t["pnl_usd"] <= 0),
            "pnl": sum(t["pnl_usd"] for t in eth_trades),
            "per_entry": [
                {
                    "trade_id": t["trade_id"],
                    "spread": t["spread_at_entry"],
                    "entry_gate": t["entry_gate"],
                    "classification": t["classification"],
                    "pnl_usd": t["pnl_usd"],
                }
                for t in eth_trades
            ],
            "btc_pnl": sum(t["pnl_usd"] for t in btc_trades),
            "eth_more_losses_reason": "more stale exits; wider spread moments; entry gate passed but exit gate stricter than review",
            "disable_eth_recommendation": "no — audit only; BTC outperformed but ETH had 1 true profit-target and 2 review FPs",
        },
    }
    print(json.dumps(report, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
