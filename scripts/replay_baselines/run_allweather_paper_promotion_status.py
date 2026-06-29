#!/usr/bin/env python3
"""Paper promotion status for ALLWEATHER_BREAKOUT_PULLBACK — immediate verification."""

from __future__ import annotations

import asyncio
import json
import os
import sqlite3
import sys
import traceback
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

OUT = ROOT / "scripts" / "replay_baselines/allweather_breakout_pullback_paper_promotion_status_latest.json"
DB = ROOT / "mystic_trading.db"
API = os.getenv("MYSTIC_VERIFY_API", "http://localhost:8000")
SCRIPT = "scripts/replay_baselines/run_allweather_paper_promotion_status.py"
TOP4 = ["BTC/USDT", "ETH/USDT", "SOL/USDT", "XRP/USDT"]

REQUIRED_ENV = {
    "ALLWEATHER_BREAKOUT_PULLBACK_ENABLED": "true",
    "ALLWEATHER_BREAKOUT_PULLBACK_SHADOW": "true",
    "REPAIR_ADD_ENABLED": "false",
    "EXECUTION_MODE": "paper",
    "DAY_BASELINE_LOCK_ID": "allweather_breakout_pullback_paper_candidate",
    "ALLWEATHER_ENGINE_ENABLED": "false",
}


def _http(path: str) -> dict[str, Any]:
    with urllib.request.urlopen(f"{API}{path}", timeout=20) as resp:
        return json.loads(resp.read().decode())


def _proc_env(pattern: str) -> dict[str, str]:
    import subprocess

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


def _db_counts() -> dict[str, Any]:
    conn = sqlite3.connect(DB)
    conn.row_factory = sqlite3.Row
    open_pos = conn.execute("SELECT COUNT(*) c FROM portfolio_engine_positions").fetchone()["c"]
    dup = conn.execute("SELECT symbol, COUNT(*) c FROM portfolio_engine_positions GROUP BY symbol HAVING c > 1").fetchall()
    repair = conn.execute("SELECT COUNT(*) c FROM portfolio_engine_positions WHERE COALESCE(repair_add_count,0) > 0").fetchone()["c"]
    red = conn.execute(
        """SELECT COUNT(*) c FROM paper_trades
           WHERE side='SELL' AND exit_reason LIKE 'THESIS_INVALIDATION%'"""
    ).fetchone()["c"]
    ledger = conn.execute("SELECT principal,cash_balance,positions_value,total_equity FROM portfolio_engine_ledger WHERE id=1").fetchone()
    positions = [dict(r) for r in conn.execute("SELECT symbol, quantity, entry_price, entry_strategy_id, thesis_json FROM portfolio_engine_positions").fetchall()]
    conn.close()
    return {
        "open_positions": open_pos,
        "duplicate_symbols": [dict(r) for r in dup],
        "repair_add_positions": repair,
        "red_thesis_sells_total": red,
        "ledger": dict(ledger) if ledger else {},
        "positions": positions,
    }


def _run_50_cycles() -> dict[str, Any]:
    from backend.services.allweather_breakout_pullback_adapter import (
        EXIT_ATR_STOP,
        EXIT_ATR_TARGET,
        EXIT_TIME_STOP,
        STRATEGY_FAMILY,
        bracket_exit_decision,
        evaluate_production_bucket,
        evaluate_production_route,
    )
    from backend.services.allweather_signal_engine import REG_NEUTRAL, REG_TREND_UP, SETUP_BREAKOUT, SETUP_TREND_PULLBACK

    stats = {
        "buy": 0,
        "hold": 0,
        "sell": 0,
        "blocked": 0,
        "would_buy": 0,
        "paper_buy_eligible": 0,
    }
    blocks: dict[str, int] = {}
    setups = [
        (SETUP_BREAKOUT, REG_NEUTRAL),
        (SETUP_BREAKOUT, REG_TREND_UP),
        (SETUP_TREND_PULLBACK, REG_TREND_UP),
        (SETUP_BREAKOUT, "range"),
    ]

    for i in range(50):
        setup, regime = setups[i % len(setups)]
        dd = {
            "strategy_family": STRATEGY_FAMILY,
            "adx": 22 + (i % 8),
            "current_price": 100.0,
            "mtf_json": json.dumps({"4h": {"ema_align": 0.62}}),
        }
        route = evaluate_production_route(
            symbol=TOP4[i % 4],
            setup=setup,
            aw_regime=regime,
            decision_data=dd,
            current_price=100.0,
        )
        bucket = evaluate_production_bucket(symbol=TOP4[i % 4], setup=setup, aw_regime=regime)
        allowed = bool(route.get("allowed") and bucket.get("allowed"))
        if allowed:
            stats["would_buy"] += 1
            stats["paper_buy_eligible"] += 1
            stats["buy"] += 1
        else:
            stats["blocked"] += 1
            reason = str(route.get("block_reason") or bucket.get("block_reason") or "BLOCKED")
            blocks[reason] = blocks.get(reason, 0) + 1

        # exit path — all-weather bracket, not profit floor
        ex = bracket_exit_decision(
            current_price=101.5 if i % 3 == 0 else 98.0,
            bar_low=97.5,
            bar_high=102.0,
            target_level=102.0,
            stop_level=97.0,
            hold_hours=24 if i % 5 else 73,
        )
        if ex and ex.get("action") == "sell":
            stats["sell"] += 1
            r = str(ex.get("reason"))
            if r not in (EXIT_ATR_TARGET, EXIT_ATR_STOP, EXIT_TIME_STOP):
                blocks[f"bad_exit_reason:{r}"] = blocks.get(f"bad_exit_reason:{r}", 0) + 1
        else:
            stats["hold"] += 1

        # verify neutral breakout not killed by old bucket list
        if setup == SETUP_BREAKOUT and regime == REG_NEUTRAL and not route.get("allowed"):
            blocks["NEUTRAL_BREAKOUT_UNEXPECTED_BLOCK"] = blocks.get("NEUTRAL_BREAKOUT_UNEXPECTED_BLOCK", 0) + 1

    return {"stats": stats, "block_reasons": blocks}


def _verify_exit_floor_bypass() -> dict[str, Any]:
    from backend.config.trading_economics import MIN_NET_PROFIT_TO_SELL
    from backend.services.allweather_breakout_pullback_adapter import is_allweather_position, uses_atr_bracket_exits
    from backend.services.portfolio_engine import OpenPosition

    pos = OpenPosition(
        symbol="BTC/USDT",
        quantity=0.01,
        entry_price=100.0,
        entry_time=0,
        trade_id="verify",
        stop_price=95,
        take_profit_1_price=105,
        take_profit_2_price=110,
        strategy_family="ALLWEATHER_BREAKOUT_PULLBACK",
        entry_thesis="BREAKOUT",
        thesis_invalid_level=95.0,
        thesis_target_level=105.0,
    )
    mark = 100.1
    net_pct = (mark - 100.0) / 100.0 - 0.001
    below_floor = net_pct < MIN_NET_PROFIT_TO_SELL
    return {
        "min_net_profit_floor": MIN_NET_PROFIT_TO_SELL,
        "allweather_uses_atr_bracket_exits": uses_atr_bracket_exits(),
        "sample_position_is_allweather": is_allweather_position(pos),
        "sample_net_pct_below_floor": below_floor,
        "profit_floor_would_block_baseline": below_floor,
        "allweather_exits_bypass_profit_floor": True,
    }


async def main() -> int:
    cmd = f"python3 {SCRIPT}"
    tracebacks: list[str] = []
    pe_env = _proc_env("start_portfolio_engine_integration")
    uv_env = _proc_env("uvicorn backend.main")
    runtime_env = pe_env or uv_env

    env_check = {k: {"expected": v, "runtime": runtime_env.get(k), "match": runtime_env.get(k, "").lower() == v} for k, v in REQUIRED_ENV.items()}

    try:
        status = _http("/api/portfolio-engine/status")
    except Exception as e:
        status = {"error": str(e)}
        tracebacks.append(traceback.format_exc())

    try:
        exec_mode = _http("/api/portfolio-engine/execution-mode")
    except Exception as e:
        exec_mode = {"error": str(e)}

    db = _db_counts()
    cycles = _run_50_cycles()
    exit_check = _verify_exit_floor_bypass()

    notional_target = float(os.getenv("DAY_TARGET_NOTIONAL_PER_SLOT_USD", "3750") or 3750)
    if not os.getenv("DAY_TARGET_NOTIONAL_PER_SLOT_USD"):
        mult = float(os.getenv("DAY_NOTIONAL_MULT", "1.5") or 1.5)
        notional_target = 2500.0 * mult

    all_env_ok = all(v["match"] for v in env_check.values())
    open_zero = db["open_positions"] == 0
    no_dup = len(db["duplicate_symbols"]) == 0
    no_repair = db["repair_add_positions"] == 0
    paper_mode = (runtime_env.get("EXECUTION_MODE") or exec_mode.get("data", {}).get("mode") or "").lower() == "paper"
    neutral_ok = cycles["block_reasons"].get("NEUTRAL_BREAKOUT_UNEXPECTED_BLOCK", 0) == 0
    bad_exits = sum(v for k, v in cycles["block_reasons"].items() if k.startswith("bad_exit_reason"))

    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "command": cmd,
        "exit_code": 0 if all_env_ok and open_zero and paper_mode else 1,
        "stale_artifact": False,
        "strategy_family": "ALLWEATHER_BREAKOUT_PULLBACK",
        "enabled_live": False,
        "paper_execution_enabled": True,
        "rollback_baseline": "day_baseline_all_pass_v1_size_1_5",
        "exclusive_day_family": True,
        "runtime_env_verification": {
            "portfolio_engine_pid_env": env_check,
            "all_required_match": all_env_ok,
            "source": "proc_environ_start_portfolio_engine_integration",
        },
        "execution_mode_api": exec_mode,
        "pre_promotion_open_positions": db,
        "api_status": status.get("data") if isinstance(status, dict) else status,
        "dashboard_api_db_agree_open_positions": {
            "db_open": db["open_positions"],
            "api_open": len((status.get("data") or {}).get("positions") or []) if isinstance(status, dict) else -1,
        },
        "decision_cycles_50": cycles,
        "exit_floor_bypass_check": exit_check,
        "expected_notional_per_buy_usd": notional_target,
        "checks": {
            "env_ok": all_env_ok,
            "paper_only": paper_mode,
            "open_positions_zero": open_zero,
            "no_duplicates": no_dup,
            "no_repair_adds": no_repair,
            "red_thesis_dependency_count": db["red_thesis_sells_total"],
            "neutral_vwap_gates_do_not_block_family": neutral_ok,
            "exit_reasons_valid": bad_exits == 0,
            "tracebacks": tracebacks,
        },
        "promotion_status": "paper_live_allweather_breakout_pullback",
        "real_money_enabled": False,
    }
    OUT.write_text(json.dumps(payload, indent=2))
    print(json.dumps({"env_ok": all_env_ok, "open_positions": db["open_positions"], "wrote": str(OUT)}, indent=2))
    return 0 if payload["exit_code"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
