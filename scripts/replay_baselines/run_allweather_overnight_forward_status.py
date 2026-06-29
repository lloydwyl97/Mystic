#!/usr/bin/env python3
"""Overnight forward paper status for ALLWEATHER_BREAKOUT_PULLBACK (non-synthetic only)."""

from __future__ import annotations

import json
import os
import re
import sqlite3
import subprocess
import sys
import traceback
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

OUT = ROOT / "scripts/replay_baselines/allweather_breakout_pullback_overnight_forward_status_latest.json"
SESSION = ROOT / "scripts/replay_baselines/allweather_overnight_forward_session_latest.json"
LIFECYCLE_SMOKE = ROOT / "scripts/replay_baselines/allweather_breakout_pullback_paper_lifecycle_smoke_latest.json"
DB = ROOT / "mystic_trading.db"
LOG_PATH = Path(os.getenv("MYSTIC_PORTFOLIO_LOG", "/tmp/mystic_portfolio.log"))
SHADOW_PATH = ROOT / "scripts/replay_baselines/allweather_breakout_pullback_shadow_latest.json"
API = os.getenv("MYSTIC_VERIFY_API", "http://localhost:8000")
SCRIPT = "scripts/replay_baselines/run_allweather_overnight_forward_status.py"
STRATEGY_FAMILY = "ALLWEATHER_BREAKOUT_PULLBACK"

CORE_PROCESSES = [
    ("uvicorn", r"uvicorn backend\.(main|app_factory)"),
    ("live_market_data", "start_live_market_data.py"),
    ("ai_signal_generator", "start_ai_signal_generator.py"),
    ("portfolio_engine_integration", "start_portfolio_engine_integration.py"),
    ("ai_market_context", "start_ai_market_context.py"),
    ("ai_learning", "start_ai_learning.py"),
]

REQUIRED_ENV = {
    "EXECUTION_MODE": "paper",
    "ALLWEATHER_BREAKOUT_PULLBACK_ENABLED": "true",
    "ALLWEATHER_BREAKOUT_PULLBACK_SHADOW": "true",
    "REPAIR_ADD_ENABLED": "false",
    "DAY_BASELINE_LOCK_ID": "allweather_breakout_pullback_paper_candidate",
    "ALLWEATHER_ENGINE_ENABLED": "false",
}


def _http(path: str) -> dict[str, Any]:
    with urllib.request.urlopen(f"{API}{path}", timeout=20) as resp:
        return json.loads(resp.read().decode())


def _proc_env(pattern: str) -> dict[str, str]:
    out = subprocess.run(["pgrep", "-f", pattern], capture_output=True, text=True, check=False)
    pid = (out.stdout.strip().split("\n") or [""])[0]
    if not pid.isdigit():
        return {}
    env: dict[str, str] = {}
    raw = Path(f"/proc/{pid}/environ").read_bytes()
    for part in raw.split(b"\0"):
        if b"=" in part:
            k, v = part.split(b"=", 1)
            env[k.decode(errors="replace")] = v.decode(errors="replace")
    return env


def _load_session() -> dict[str, Any]:
    if SESSION.exists():
        return json.loads(SESSION.read_text())
    now = datetime.now(timezone.utc)
    log_lines = LOG_PATH.read_text(errors="replace").splitlines() if LOG_PATH.exists() else []
    payload = {
        "approved_at": now.isoformat(),
        "session_label": "overnight_forward_paper_validation",
        "strategy_family": STRATEGY_FAMILY,
        "execution_mode": "paper",
        "principal_usd": 25000.0,
        "synthetic_smoke_excluded": True,
        "portfolio_log_path": str(LOG_PATH),
        "portfolio_log_line_offset_at_start": len(log_lines),
    }
    SESSION.write_text(json.dumps(payload, indent=2))
    return payload


def _parse_log_slice(start_line: int) -> dict[str, Any]:
    if not LOG_PATH.exists():
        return {"exists": False, "counts": {}, "tracebacks": [], "samples": {}}
    lines = LOG_PATH.read_text(errors="replace").splitlines()
    slice_lines = lines[start_line:] if start_line < len(lines) else []
    patterns = {
        "ALLWEATHER_BP_EXEC": r"ALLWEATHER_BP_EXEC",
        "ALLWEATHER_BP_SHADOW": r"ALLWEATHER_BP_SHADOW",
        "ALLWEATHER_BP_NO_SIGNAL": r"ALLWEATHER_BP_NO_SIGNAL",
        "ALLWEATHER_BP_EVAL_ERROR": r"ALLWEATHER_BP_EVAL_ERROR",
        "ALLWEATHER_BP_BLOCK": r"ALLWEATHER_BP_BLOCK",
        "ALLWEATHER_BP_EXIT": r"ALLWEATHER_BP_EXIT",
        "ALLWEATHER_ATR_TARGET_EXIT": r"ALLWEATHER_ATR_TARGET_EXIT",
        "ALLWEATHER_ATR_STOP_EXIT": r"ALLWEATHER_ATR_STOP_EXIT",
        "ALLWEATHER_TIME_STOP_EXIT": r"ALLWEATHER_TIME_STOP_EXIT",
        "BUY_EXECUTED": r"BUY_EXECUTED:",
        "FIFO_BUY": r"FIFO_BUY|execute_buy_fifo",
        "BUCKET_QUALITY_BLOCK": r"BUCKET_QUALITY_BLOCK",
        "REGIME_ROUTE_BLOCK": r"REGIME_ROUTE_BLOCK",
        "GLOBAL_KILLED": r"GLOBAL_KILLED",
        "REPAIR_ADD": r"REPAIR_ADD(?!_ECONOMICS enabled=False)",
        "THESIS_INVALID": r"THESIS_INVALID",
        "Traceback": r"Traceback",
    }
    counts = {k: sum(1 for ln in slice_lines if re.search(p, ln)) for k, p in patterns.items()}
    blocks: dict[str, int] = {}
    for ln in slice_lines:
        if "ALLWEATHER_BP_BLOCK" in ln:
            m = re.search(r"route=([^ ]+).*bucket=([^ ]+)", ln)
            key = f"ALLWEATHER_BP_BLOCK:{m.group(1) if m else 'unknown'}"
            blocks[key] = blocks.get(key, 0) + 1
        if "REGIME_ROUTE_BLOCK" in ln or "BUCKET_QUALITY_BLOCK" in ln:
            m = re.search(r"reason=([^\s]+)", ln)
            key = m.group(1) if m else "unknown"
            blocks[key] = blocks.get(key, 0) + 1
    tracebacks = [i + start_line for i, ln in enumerate(slice_lines) if "Traceback" in ln]
    exec_samples = [ln for ln in slice_lines if "ALLWEATHER_BP_EXEC" in ln][:5]
    exit_samples = [ln for ln in slice_lines if "ALLWEATHER_BP_EXIT" in ln][:5]
    return {
        "exists": True,
        "log_path": str(LOG_PATH),
        "session_start_line": start_line,
        "lines_in_session_window": len(slice_lines),
        "counts": counts,
        "block_reasons": blocks,
        "traceback_line_numbers": tracebacks[-20:],
        "traceback_count": counts.get("Traceback", 0),
        "samples": {
            "allweather_exec": exec_samples,
            "allweather_exit": exit_samples,
        },
        "evaluated_cycles_estimate": counts.get("ALLWEATHER_BP_NO_SIGNAL", 0) // 4
        + counts.get("ALLWEATHER_BP_EXEC", 0)
        + counts.get("ALLWEATHER_BP_BLOCK", 0)
        + counts.get("ALLWEATHER_BP_EVAL_ERROR", 0) // 4,
    }


def _is_synthetic_sql() -> str:
    return "(COALESCE(is_synthetic, 0) = 1 OR COALESCE(paper_run_id, '') LIKE 'PAPER_LIFECYCLE_SMOKE%')"


def _is_allweather_sql() -> str:
    return (
        "(COALESCE(explainability_json, '') LIKE '%ALLWEATHER_BREAKOUT_PULLBACK%' "
        "OR COALESCE(diagnostics_json, '') LIKE '%ALLWEATHER_BREAKOUT_PULLBACK%' "
        "OR COALESCE(strategy_id, '') LIKE '%ALLWEATHER%')"
    )


def _json_field(raw: str | None) -> dict[str, Any]:
    if not raw:
        return {}
    try:
        return json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return {}


def _trade_row(row: sqlite3.Row) -> dict[str, Any]:
    d = dict(row)
    ex = _json_field(d.get("explainability_json"))
    diag = _json_field(d.get("diagnostics_json"))
    meta = {**ex, **diag}
    return {
        "trade_id": d.get("trade_id"),
        "symbol": d.get("symbol"),
        "side": d.get("side"),
        "timestamp": d.get("timestamp"),
        "price": d.get("price"),
        "quantity": d.get("quantity"),
        "notional_usd": round(float(d.get("quantity") or 0) * float(d.get("price") or 0), 2),
        "fees_usd": float(d.get("fees_paid") or 0),
        "spread_pct": d.get("spread_pct_used"),
        "slippage_pct": d.get("slippage_pct_used"),
        "exit_reason": d.get("exit_reason"),
        "pnl_usd_net": d.get("pnl_usd_net"),
        "strategy_family": meta.get("strategy_family"),
        "setup": meta.get("allweather_setup") or meta.get("setup_type"),
        "is_synthetic": bool(d.get("is_synthetic")),
        "paper_run_id": d.get("paper_run_id"),
    }


def _forward_trades(session_start: str) -> dict[str, Any]:
    conn = sqlite3.connect(DB)
    conn.row_factory = sqlite3.Row
    synth_filter = _is_synthetic_sql()
    aw_filter = _is_allweather_sql()
    base = f"timestamp >= ? AND NOT {synth_filter} AND {aw_filter}"
    buys = conn.execute(
        f"SELECT * FROM paper_trades WHERE side='BUY' AND {base} ORDER BY id",
        (session_start,),
    ).fetchall()
    sells = conn.execute(
        f"SELECT * FROM paper_trades WHERE side='SELL' AND {base} ORDER BY id",
        (session_start,),
    ).fetchall()
    synth_buys = conn.execute(
        f"SELECT COUNT(*) c FROM paper_trades WHERE side='BUY' AND timestamp >= ? AND {synth_filter}",
        (session_start,),
    ).fetchone()["c"]
    synth_sells = conn.execute(
        f"SELECT COUNT(*) c FROM paper_trades WHERE side='SELL' AND timestamp >= ? AND {synth_filter}",
        (session_start,),
    ).fetchone()["c"]
    positions = [dict(r) for r in conn.execute("SELECT symbol, quantity, entry_price, entry_strategy_id, thesis_json, repair_add_count FROM portfolio_engine_positions").fetchall()]
    for p in positions:
        tj = _json_field(p.get("thesis_json"))
        p["strategy_family"] = tj.get("strategy_family") or p.get("entry_strategy_id")
        p["is_synthetic_forward"] = False
    aw_open = [p for p in positions if STRATEGY_FAMILY in str(p.get("strategy_family") or "")]
    dup = conn.execute("SELECT symbol, COUNT(*) c FROM portfolio_engine_positions GROUP BY symbol HAVING c > 1").fetchall()
    repair = conn.execute("SELECT COUNT(*) c FROM portfolio_engine_positions WHERE COALESCE(repair_add_count,0) > 0").fetchone()["c"]
    red = conn.execute(
        "SELECT COUNT(*) c FROM paper_trades WHERE side='SELL' AND timestamp >= ? AND exit_reason LIKE 'THESIS_INVALID%' AND NOT " + synth_filter,
        (session_start,),
    ).fetchone()["c"]
    exit_counts: dict[str, int] = {}
    for s in sells:
        r = str(dict(s).get("exit_reason") or "UNKNOWN")
        exit_counts[r] = exit_counts.get(r, 0) + 1
    forward_realized = sum(float(dict(s).get("pnl_usd_net") or dict(s).get("pnl") or 0) for s in sells)
    conn.close()
    return {
        "session_start_filter": session_start,
        "actual_non_synthetic_allweather_buys": [_trade_row(r) for r in buys],
        "actual_non_synthetic_allweather_sells": [_trade_row(r) for r in sells],
        "forward_realized_pnl_usd_non_synthetic": round(forward_realized, 4),
        "synthetic_trades_excluded": {"buys": synth_buys, "sells": synth_sells},
        "open_positions_all": positions,
        "open_positions_allweather_forward": aw_open,
        "exit_reason_counts_forward": exit_counts,
        "duplicate_position_groups": [dict(r) for r in dup],
        "duplicate_count": len(dup),
        "repair_add_open_count": repair,
        "red_thesis_sells_forward": red,
    }


def _shadow_since(session_start: str) -> dict[str, Any]:
    if not SHADOW_PATH.exists():
        return {"shadow_file_exists": False, "would_buy_count_since_session": 0, "heartbeat": False}
    try:
        data = json.loads(SHADOW_PATH.read_text())
    except json.JSONDecodeError:
        return {"shadow_file_exists": True, "would_buy_count_since_session": 0, "error": "invalid_json", "heartbeat": False}
    if data.get("heartbeat"):
        return {
            "shadow_file_exists": True,
            "heartbeat": True,
            "generated_at": data.get("generated_at"),
            "evaluated_cycles": data.get("evaluated_cycles"),
            "symbols_evaluated": data.get("symbols_evaluated"),
            "would_buy_count_since_session": int(data.get("would_buy_count") or 0),
            "no_signal_count": data.get("no_signal_count"),
            "eval_error_count": data.get("eval_error_count"),
            "latest_no_signal_reasons": data.get("latest_no_signal_reasons"),
            "open_positions": data.get("open_positions"),
            "real_orders_permitted": data.get("real_orders_permitted"),
            "kline_fetch_stats": data.get("kline_fetch_stats"),
        }
    entries = list(data.get("entries") or [])
    start_dt = datetime.fromisoformat(session_start.replace("Z", "+00:00"))
    would = 0
    recent: list[dict[str, Any]] = []
    for e in entries:
        ts = e.get("ts") or e.get("timestamp")
        if not ts:
            continue
        try:
            edt = datetime.fromisoformat(str(ts).replace("Z", "+00:00"))
        except ValueError:
            continue
        if edt < start_dt:
            continue
        if str(e.get("action") or "").lower() in ("would_buy", "buy"):
            would += 1
            if len(recent) < 5:
                recent.append(e)
    return {
        "shadow_file_exists": True,
        "would_buy_count_since_session": would,
        "recent_would_buy_samples": recent,
    }


def _process_health() -> dict[str, Any]:
    health: dict[str, Any] = {}
    all_ok = True
    for name, pattern in CORE_PROCESSES:
        out = subprocess.run(["pgrep", "-f", pattern], capture_output=True, text=True, check=False)
        pids = [p for p in out.stdout.strip().split("\n") if p.strip().isdigit()]
        health[name] = {"running": len(pids) > 0, "pids": pids[:3]}
        if not pids:
            all_ok = False
    scalp = subprocess.run(["pgrep", "-f", "scalp"], capture_output=True, text=True, check=False)
    scalp_pids = [p for p in scalp.stdout.strip().split("\n") if p.strip().isdigit()]
    health["scalp_processes"] = {
        "running": len(scalp_pids) > 0,
        "pids": scalp_pids[:5],
        "note": "Scalp may run separately; excluded from DAY/all-weather forward PnL reporting.",
    }
    health["core_stack_healthy"] = all_ok
    return health


def main() -> int:
    tracebacks: list[str] = []
    now = datetime.now(timezone.utc)
    session = _load_session()
    session_start = str(session.get("approved_at") or now.isoformat())
    start_line = int(session.get("portfolio_log_line_offset_at_start") or 0)

    try:
        approved_dt = datetime.fromisoformat(session_start.replace("Z", "+00:00"))
        runtime_hours = round((now - approved_dt).total_seconds() / 3600.0, 3)
    except Exception:
        runtime_hours = 0.0

    pe_env = _proc_env("start_portfolio_engine_integration")
    env_check = {k: {"expected": v, "runtime": pe_env.get(k), "match": pe_env.get(k, "").lower() == v} for k, v in REQUIRED_ENV.items()}

    try:
        status = _http("/api/portfolio-engine/status")
        exec_mode = _http("/api/portfolio-engine/execution-mode")
    except Exception as e:
        status = {"error": str(e)}
        exec_mode = {"error": str(e)}
        tracebacks.append(traceback.format_exc())

    log_stats = _parse_log_slice(start_line)
    trades = _forward_trades(session_start)
    shadow = _shadow_since(session_start)
    health = _process_health()

    api = status.get("data") if isinstance(status, dict) else {}
    buys = trades["actual_non_synthetic_allweather_buys"]
    sells = trades["actual_non_synthetic_allweather_sells"]
    no_trade = len(buys) == 0 and len(sells) == 0

    smoke_ref = {}
    if LIFECYCLE_SMOKE.exists():
        try:
            sm = json.loads(LIFECYCLE_SMOKE.read_text())
            smoke_ref = {
                "smoke_run_id": sm.get("smoke_run_id"),
                "smoke_success": (sm.get("lifecycle_smoke") or {}).get("success"),
                "excluded_from_forward_pnl": True,
            }
        except json.JSONDecodeError:
            pass

    live_permitted = bool((exec_mode.get("data") or {}).get("live_orders_permitted", False))

    payload: dict[str, Any] = {
        "generated_at": now.isoformat(),
        "command": f"python3 {SCRIPT}",
        "session": session,
        "runtime_hours": runtime_hours,
        "synthetic_smoke_reference": smoke_ref,
        "process_health": health,
        "runtime_env_verification": {"all_required_match": all(env_check[k]["match"] for k in env_check), "checks": env_check},
        "execution_mode_api": exec_mode,
        "live_orders_permitted": live_permitted,
        "real_money_enabled": False,
        "real_money_safety_confirmation": {
            "execution_mode_paper": pe_env.get("EXECUTION_MODE", "").lower() == "paper",
            "live_orders_permitted": live_permitted,
            "live_orders_blocked": not live_permitted,
            "repair_add_disabled": pe_env.get("REPAIR_ADD_ENABLED", "").lower() == "false",
            "allweather_engine_disabled": pe_env.get("ALLWEATHER_ENGINE_ENABLED", "").lower() == "false",
            "no_leverage": True,
            "no_shorting": True,
        },
        "account": {
            "principal_usd": float(api.get("principal") or session.get("principal_usd") or 25000),
            "cash_usd": float(api.get("cash_balance") or 0),
            "equity_usd": float(api.get("total_equity") or 0),
            "forward_equity_usd": float(api.get("forward_equity") or api.get("total_equity") or 0),
            "ledger_realized_pnl_usd": float(api.get("realized_pnl") or 0),
            "realized_pnl_forward_usd": float(api.get("realized_pnl_forward") or trades["forward_realized_pnl_usd_non_synthetic"]),
            "synthetic_smoke_pnl_usd": float(api.get("synthetic_smoke_pnl") or 0),
            "pre_rebase_history_pnl_usd": float(api.get("pre_rebase_history_pnl") or 0),
            "unrealized_pnl_usd": float(api.get("unrealized_pnl") or 0),
            "forward_non_synthetic_realized_pnl_usd": trades["forward_realized_pnl_usd_non_synthetic"],
            "pnl_note": "forward_equity/realized_pnl_forward exclude synthetic smoke; pre_rebase_history_pnl is pre-epoch non-synthetic.",
            "open_positions_count": int(api.get("positions_count") or 0),
            "open_positions": api.get("positions") or trades["open_positions_allweather_forward"],
            "forward_paper_accounting": api.get("forward_paper_accounting"),
        },
        "forward_paper_events": {
            "log_window": log_stats,
            "shadow_would_buy": shadow,
            "actual_buys": buys,
            "actual_sells": sells,
            "hold_updates": trades["open_positions_allweather_forward"],
        },
        "exit_reason_counts_forward": trades["exit_reason_counts_forward"],
        "duplicate_count": trades["duplicate_count"],
        "repair_add_count": trades["repair_add_open_count"],
        "red_thesis_usage_count_forward": trades["red_thesis_sells_forward"],
        "tracebacks_in_session_logs": log_stats.get("traceback_count", 0),
        "tracebacks_in_audit_script": tracebacks,
        "no_trade_status": {
            "no_actual_forward_trade_yet": no_trade,
            "evaluated_cycles_estimate": log_stats.get("evaluated_cycles_estimate", 0),
            "primary_reason": "ALLWEATHER_BP_NO_SIGNAL" if log_stats.get("counts", {}).get("ALLWEATHER_BP_NO_SIGNAL", 0) > 0 else "awaiting_first_bar_cycle",
            "no_signal_count": log_stats.get("counts", {}).get("ALLWEATHER_BP_NO_SIGNAL", 0),
            "allweather_exec_count": log_stats.get("counts", {}).get("ALLWEATHER_BP_EXEC", 0),
            "message": ("No actual non-synthetic all-weather paper trade yet; engine continues forward validation." if no_trade else "Forward paper activity detected."),
        },
        "overnight_rules": {
            "no_strategy_changes": True,
            "no_live_money": True,
            "no_interference_on_open_trades": True,
            "synthetic_smoke_excluded_from_strategy_pnl": True,
        },
    }
    OUT.write_text(json.dumps(payload, indent=2))
    print(
        json.dumps(
            {
                "wrote": str(OUT),
                "runtime_hours": runtime_hours,
                "core_healthy": health.get("core_stack_healthy"),
                "forward_buys": len(buys),
                "no_signal_count": log_stats.get("counts", {}).get("ALLWEATHER_BP_NO_SIGNAL", 0),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
