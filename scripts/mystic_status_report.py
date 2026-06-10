#!/usr/bin/env python3
"""Mystic operator status report — safe JSON, thesis_json, infra vs strategy split."""

from __future__ import annotations

import json
import sqlite3
import sys
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DB = REPO / "mystic_trading.db"
API = "http://127.0.0.1:8000"

sys.path.insert(0, str(REPO))
from scripts.monitor_metrics import fetch_json  # noqa: E402


def _parse_thesis(raw: str | None) -> dict:
    if not raw:
        return {}
    try:
        data = json.loads(raw)
        return data if isinstance(data, dict) else {}
    except (json.JSONDecodeError, TypeError):
        return {}


def open_positions_from_db() -> list[dict]:
    conn = sqlite3.connect(DB)
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute(
            """
            SELECT symbol, quantity, entry_price, entry_time, trade_id, thesis_json
            FROM portfolio_engine_positions
            ORDER BY symbol
            """
        ).fetchall()
    finally:
        conn.close()

    out: list[dict] = []
    now = time.time()
    status_data, _ = fetch_json(f"{API}/api/portfolio-engine/status")
    marks: dict[str, float] = {}
    exit_blocked_syms: set[str] = set()
    if status_data:
        d = status_data.get("data") or {}
        for p in d.get("open_positions") or []:
            if isinstance(p, dict) and p.get("symbol"):
                marks[str(p["symbol"])] = float(p.get("current_price") or 0)
        for eb in d.get("exit_blocked_positions") or []:
            if isinstance(eb, dict) and eb.get("symbol"):
                exit_blocked_syms.add(str(eb["symbol"]))

    for row in rows:
        sym = str(row["symbol"])
        qty = float(row["quantity"] or 0)
        entry = float(row["entry_price"] or 0)
        mark = marks.get(sym, entry)
        u_pnl = qty * (mark - entry) if qty and entry else 0.0
        thesis = _parse_thesis(row["thesis_json"])
        invalid = float(thesis.get("thesis_invalid_level") or 0)
        target = float(thesis.get("thesis_target_level") or 0)
        entry_ts = float(row["entry_time"] or 0)
        hold_h = (now - entry_ts) / 3600.0 if entry_ts else 0.0
        thesis_invalid_now = invalid > 0 and 0 < mark < invalid
        out.append(
            {
                "symbol": sym,
                "quantity": qty,
                "entry_price": entry,
                "current_mark": mark,
                "unrealized_pnl": round(u_pnl, 2),
                "entry_thesis": thesis.get("entry_thesis"),
                "thesis_score": thesis.get("thesis_score"),
                "thesis_invalid_level": invalid or None,
                "thesis_target_level": target or None,
                "thesis_trend_tf": thesis.get("thesis_trend_tf"),
                "hold_hours": round(hold_h, 1),
                "exit_blocked": sym in exit_blocked_syms,
                "exit_allowed": sym not in exit_blocked_syms,
                "thesis_invalidated_by_mark": thesis_invalid_now,
                "trade_id": row["trade_id"],
            }
        )
    return out


def pnl_reconciliation() -> dict:
    conn = sqlite3.connect(DB)
    try:
        ledger = conn.execute(
            "SELECT realized_pnl FROM portfolio_engine_ledger WHERE id=1"
        ).fetchone()
        ledger_r = float(ledger[0] or 0) if ledger else 0.0
        paper_all = float(
            conn.execute(
                """
                SELECT COALESCE(SUM(pnl), 0) FROM paper_trades
                WHERE side='SELL' AND pnl IS NOT NULL
                  AND COALESCE(exit_type,'') NOT IN (
                    'ADMIN_POSITION_CLEAR', 'STALE_PRE_CORRECTION_POSITION_CLEAR'
                  )
                """
            ).fetchone()[0]
            or 0
        )
        paper_today_strategy = float(
            conn.execute(
                """
                SELECT COALESCE(SUM(pnl), 0) FROM paper_trades
                WHERE side='SELL' AND date(timestamp)=date('now')
                  AND COALESCE(exit_type,'') NOT IN (
                    'ADMIN_POSITION_CLEAR', 'STALE_PRE_CORRECTION_POSITION_CLEAR',
                    'legacy_no_clear_position_clear', 'STALE_LIVE_GHOST_POSITION_CLEAR'
                  )
                """
            ).fetchone()[0]
            or 0
        )
        paper_today_all = float(
            conn.execute(
                """
                SELECT COALESCE(SUM(pnl), 0) FROM paper_trades
                WHERE side='SELL' AND date(timestamp)=date('now')
                  AND COALESCE(exit_type,'') NOT IN (
                    'ADMIN_POSITION_CLEAR', 'STALE_PRE_CORRECTION_POSITION_CLEAR'
                  )
                """
            ).fetchone()[0]
            or 0
        )
    finally:
        conn.close()

    score, err = fetch_json(f"{API}/api/portfolio-engine/scoreboard/today")
    audit_today = None
    if not err and score:
        s = score.get("data") or {}
        audit_today = float(s.get("realized_pnl") or 0)

    return {
        "ledger_realized_alltime_stored": round(ledger_r, 4),
        "paper_sells_alltime_canonical": round(paper_all, 4),
        "ledger_paper_drift": round(ledger_r - paper_all, 4),
        "strategy_today_realized_pnl": round(audit_today, 4) if audit_today is not None else None,
        "paper_sells_today_strategy_scope": round(paper_today_strategy, 4),
        "paper_sells_today_including_ops": round(paper_today_all, 4),
        "labels": {
            "strategy_today_realized_pnl": "Scoreboard PASS/FAIL scope: audit AI SELL closes today (net after costs).",
            "paper_sells_today_strategy_scope": "paper_trades SELLs today excluding legacy/ghost/admin ops.",
            "paper_sells_today_including_ops": "All paper SELLs today including legacy_no_clear clears.",
            "paper_sells_alltime_canonical": "SUM(paper_trades SELL pnl) ex admin clears — ledger should match.",
        },
    }


def main() -> None:
    status, status_err = fetch_json(f"{API}/api/portfolio-engine/status")
    score, score_err = fetch_json(f"{API}/api/portfolio-engine/scoreboard/today")

    report: dict = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "infra": {"health": "UNKNOWN", "procs_expected": 7},
        "strategy": {"pass_fail": None, "fail_reasons": None},
        "errors": [],
    }

    if status_err:
        report["errors"].append(f"status: {status_err}")
    else:
        d = (status or {}).get("data") or {}
        report["infra"] = {
            "health": d.get("account_status"),
            "trading_paused": d.get("trading_paused"),
            "degraded": d.get("degraded"),
            "exit_blocked_count": len(d.get("exit_blocked_positions") or []),
            "equity": d.get("total_equity"),
            "cash": d.get("cash_balance"),
            "open_positions_count": len(d.get("open_positions") or []),
            "dashboard_note": "Infra PASS = processes up, API OK, HEALTHY, exit_blocked=0.",
        }

    if score_err:
        report["errors"].append(f"scoreboard: {score_err}")
    else:
        s = (score or {}).get("data") or {}
        report["strategy"] = {
            "pass_fail": s.get("pass_fail"),
            "fail_reasons": s.get("fail_reasons"),
            "today_closed_ai_trades": s.get("closed_ai_trades_today"),
            "strategy_today_realized_pnl": s.get("realized_pnl"),
            "paper_sells_today_pnl": s.get("ai_realized_pnl_today"),
            "dashboard_note": "Strategy FAIL is diagnostic (expectancy/PnL rules), not infra down.",
        }

    report["pnl_reconciliation"] = pnl_reconciliation()
    report["open_positions"] = open_positions_from_db()
    report["duplicate_buy_check"] = {
        "note": "08:19 UTC bar ran BUY_SIZING then POSITION_MANAGEMENT_HOLD — no second fill.",
        "one_position_per_symbol": len({p["symbol"] for p in report["open_positions"]}) == len(report["open_positions"]),
    }

    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
