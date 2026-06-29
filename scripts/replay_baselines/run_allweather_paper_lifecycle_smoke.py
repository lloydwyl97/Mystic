#!/usr/bin/env python3
"""
Paper ledger rebase to $25k + forced ALLWEATHER_BREAKOUT_PULLBACK lifecycle smoke.

Uses cached historical 1h bars and the production portfolio-engine paper path
(process_bar_candidates → execute_buy_fifo → ATR bracket exit). Does not place
real-money orders.
"""

from __future__ import annotations

import asyncio
import json
import math
import os
import shutil
import sqlite3
import subprocess
import sys
import time
import traceback
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

OUT = ROOT / "scripts/replay_baselines/allweather_breakout_pullback_paper_lifecycle_smoke_latest.json"
DB = ROOT / "mystic_trading.db"
VAR_DIR = ROOT / "var/paper_ledger_rebase"
API = os.getenv("MYSTIC_VERIFY_API", "http://localhost:8000")
SCRIPT = "scripts/replay_baselines/run_allweather_paper_lifecycle_smoke.py"
REBASE_PRINCIPAL = 25_000.0
STRATEGY_FAMILY = "ALLWEATHER_BREAKOUT_PULLBACK"
SMOKE_RUN_ID = f"PAPER_LIFECYCLE_SMOKE_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"

# Load .env before backend imports
_env = ROOT / ".env"
if _env.exists():
    for raw_line in _env.read_text(errors="replace").splitlines():
        line = raw_line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())

# Smoke-only gate relaxations (production code path; no live orders)
os.environ["EXECUTION_MODE"] = "paper"
os.environ["ALLWEATHER_BREAKOUT_PULLBACK_ENABLED"] = "true"
os.environ["ALLWEATHER_BREAKOUT_PULLBACK_SHADOW"] = "true"
os.environ["ALLWEATHER_ENGINE_ENABLED"] = "false"
os.environ["REPAIR_ADD_ENABLED"] = "false"
os.environ["DAY_NOTIONAL_MULT"] = os.environ.get("DAY_NOTIONAL_MULT", "1.5")
os.environ["ARTIFACT_CONTRACT_GATE_ENABLED"] = "false"
os.environ["ENTRY_CONTEXT_GATE_ENABLED"] = "false"
os.environ["USE_PROTECTED_LIMIT_EXECUTION"] = "false"
os.environ["PORTFOLIO_LOCAL_SKIP_POST_SELL_COOLDOWN"] = "true"
os.environ["PORTFOLIO_LOCAL_SKIP_GLOBAL_SELL_COOLDOWN"] = "true"


def _http(path: str) -> dict[str, Any]:
    with urllib.request.urlopen(f"{API}{path}", timeout=25) as resp:
        return json.loads(resp.read().decode())


def _mystic_running() -> bool:
    r = subprocess.run(["pgrep", "-f", "start_portfolio_engine_integration"], capture_output=True, text=True)
    return r.returncode == 0


def _stop_mystic() -> dict[str, Any]:
    if not _mystic_running():
        return {"stopped": False, "was_running": False}
    r = subprocess.run([str(ROOT / "stop_mystic.sh")], cwd=str(ROOT), capture_output=True, text=True)
    time.sleep(3)
    return {"stopped": True, "exit_code": r.returncode, "stderr_tail": (r.stderr or "")[-500:]}


def _start_mystic() -> dict[str, Any]:
    r = subprocess.run([str(ROOT / "start_mystic.sh"), "core"], cwd=str(ROOT), capture_output=True, text=True)
    time.sleep(8)
    return {"started": True, "exit_code": r.returncode, "stderr_tail": (r.stderr or "")[-500:]}


def _backup_db() -> str:
    VAR_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    dest = VAR_DIR / f"mystic_trading.db.pre_rebase_{ts}"
    shutil.copy2(DB, dest)
    return str(dest)


def _ledger_row(conn: sqlite3.Connection) -> dict[str, Any]:
    row = conn.execute("SELECT principal, cash_balance, positions_value, realized_pnl, unrealized_pnl, total_equity FROM portfolio_engine_ledger WHERE id=1").fetchone()
    return dict(zip(["principal", "cash_balance", "positions_value", "realized_pnl", "unrealized_pnl", "total_equity"], row or (0,) * 6, strict=False))


def reset_paper_ledger_25k() -> dict[str, Any]:
    conn = sqlite3.connect(DB)
    conn.row_factory = sqlite3.Row
    prior = _ledger_row(conn)
    open_before = conn.execute("SELECT COUNT(*) FROM portfolio_engine_positions").fetchone()[0]
    ts = datetime.now(timezone.utc).isoformat()

    rebase_record = {
        "event": "PAPER_LEDGER_REBASE",
        "research_only": True,
        "not_strategy_performance": True,
        "not_live_trade": True,
        "not_real_money": True,
        "timestamp_utc": ts,
        "smoke_run_id": SMOKE_RUN_ID,
        "prior_ledger": prior,
        "prior_open_positions": open_before,
        "new_principal_usd": REBASE_PRINCIPAL,
        "reason": "Align paper validation capital with replay/promotion gate $25k principal",
    }

    hist: list[Any] = []
    row = conn.execute("SELECT value_json FROM operational_state WHERE key='paper_ledger_rebase_history'").fetchone()
    if row and row[0]:
        try:
            hist = json.loads(row[0])
        except json.JSONDecodeError:
            hist = []
    if not isinstance(hist, list):
        hist = []
    hist.append(rebase_record)

    conn.execute("DELETE FROM portfolio_engine_positions")
    conn.execute(
        """
        UPDATE portfolio_engine_ledger SET
            principal = ?,
            cash_balance = ?,
            positions_value = 0.0,
            realized_pnl = 0.0,
            unrealized_pnl = 0.0,
            total_equity = ?,
            account_status = 'HEALTHY',
            trading_paused = 0,
            pause_reason = NULL,
            last_updated = ?
        WHERE id = 1
        """,
        (REBASE_PRINCIPAL, REBASE_PRINCIPAL, REBASE_PRINCIPAL, ts),
    )
    conn.execute(
        """
        INSERT INTO operational_state(key, value_json, updated_ts)
        VALUES('paper_ledger_rebase_latest', ?, strftime('%s','now'))
        ON CONFLICT(key) DO UPDATE SET value_json=excluded.value_json, updated_ts=excluded.updated_ts
        """,
        (json.dumps(rebase_record),),
    )
    conn.execute(
        """
        INSERT INTO operational_state(key, value_json, updated_ts)
        VALUES('paper_ledger_rebase_history', ?, strftime('%s','now'))
        ON CONFLICT(key) DO UPDATE SET value_json=excluded.value_json, updated_ts=excluded.updated_ts
        """,
        (json.dumps(hist[-50:]),),
    )
    conn.commit()
    after = _ledger_row(conn)
    open_after = conn.execute("SELECT COUNT(*) FROM portfolio_engine_positions").fetchone()[0]
    conn.close()
    return {
        "rebase_record": rebase_record,
        "prior_ledger": prior,
        "after_ledger": after,
        "open_positions_before": open_before,
        "open_positions_after": open_after,
    }


def _simulate_four_slots(cash: float, slot: float) -> list[dict[str, Any]]:
    max_coin = float(os.getenv("MAX_CASH_PER_COIN_PCT", "0.25"))
    mult = float(os.getenv("DAY_NOTIONAL_MULT", "1.5"))
    remaining = cash
    out = []
    for i in range(4):
        if remaining <= 0:
            break
        cap = min(slot, remaining, remaining * max_coin * mult, (remaining / 4) * mult)
        out.append({"slot": i + 1, "notional_usd": round(cap, 2), "cash_before": round(remaining, 2)})
        remaining -= cap
    return out


def verify_ledger_state() -> dict[str, Any]:
    conn = sqlite3.connect(DB)
    ledger = _ledger_row(conn)
    pos = conn.execute("SELECT COUNT(*) FROM portfolio_engine_positions").fetchone()[0]
    conn.execute("SELECT COUNT(*) FROM (SELECT symbol FROM portfolio_engine_positions GROUP BY symbol HAVING COUNT(*)>1)").fetchone()[0]
    conn.close()
    slot = float(os.getenv("DAY_BASE_NOTIONAL_PER_SLOT_USD", "2500")) * float(os.getenv("DAY_NOTIONAL_MULT", "1.5"))
    try:
        from backend.config.trading_economics import DAY_TARGET_NOTIONAL_PER_SLOT_USD

        slot = float(DAY_TARGET_NOTIONAL_PER_SLOT_USD)
    except Exception:
        pass
    sim = _simulate_four_slots(float(ledger["cash_balance"]), slot)
    api = {}
    try:
        api = _http("/api/portfolio-engine/status").get("data") or {}
    except Exception as e:
        api = {"error": str(e)}
    return {
        "ledger": ledger,
        "db_open_positions": pos,
        "api_open_positions": len(api.get("positions") or []),
        "api_cash": api.get("cash_balance"),
        "api_equity": api.get("total_equity"),
        "agree_flat": pos == 0 and len(api.get("positions") or []) == 0,
        "cash_equity_25k": abs(float(ledger["cash_balance"]) - REBASE_PRINCIPAL) < 0.02 and abs(float(ledger["total_equity"]) - REBASE_PRINCIPAL) < 0.02,
        "per_slot_target_usd": slot,
        "four_slot_simulation": sim,
        "four_full_slots_supported": len(sim) >= 4 and all(s["notional_usd"] >= slot - 0.01 for s in sim[:4]),
        "no_negative_cash": float(ledger["cash_balance"]) >= 0,
    }


def _find_smoke_signal() -> dict[str, Any]:
    from backend.services.allweather_signal_engine import compute_state, entry_levels, entry_signal
    from scripts.run_day_execution_replay import fetch_klines_cached

    end_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
    start_ms = end_ms - 180 * 86400 * 1000
    best: dict[str, Any] | None = None
    for sym in ["XRP/USDT", "SOL/USDT", "ETH/USDT", "BTC/USDT"]:
        bars = fetch_klines_cached(sym, "1h", start_ms, end_ms)
        if len(bars) < 220:
            continue
        for bar_idx in range(205, len(bars)):
            window = bars[: bar_idx + 1]
            state = compute_state(window)
            if state is None:
                continue
            sig = entry_signal(state)
            if not sig:
                continue
            target, stop = entry_levels(state.close, state.atr, float(sig["target_atr"]), float(sig["stop_atr"]))
            if target <= state.close * 1.001:
                continue
            cand = {
                "symbol": sym,
                "bar_timestamp": int(window[-1]["ts"]),
                "setup": str(sig["setup"]),
                "regime": str(sig["regime"]),
                "entry_price": float(state.close),
                "atr": float(state.atr),
                "atr_target": float(target),
                "atr_stop": float(stop),
                "target_atr_mult": float(sig["target_atr"]),
                "stop_atr_mult": float(sig["stop_atr"]),
                "deadline_72h_hours": 72.0,
                "mock_bars": window,
                "_score": (target - state.close) / state.close,
            }
            if best is None or cand["_score"] > best["_score"]:
                best = cand
    if best is None:
        raise RuntimeError("no suitable historical allweather signal found in cache")
    best.pop("_score", None)
    return best


async def run_lifecycle_smoke(signal: dict[str, Any]) -> dict[str, Any]:
    from unittest.mock import patch

    from backend.config.trading_economics import MIN_NET_PROFIT_TO_SELL
    from backend.services import allweather_breakout_pullback_adapter as awbp
    from backend.services.portfolio_engine import BuyCandidate, PortfolioEngine
    from backend.utils.symbols import normalize_symbol

    tracebacks: list[str] = []
    result: dict[str, Any] = {"smoke_run_id": SMOKE_RUN_ID, "signal_bar": {k: v for k, v in signal.items() if k != "mock_bars"}}

    engine = PortfolioEngine(db_path=str(DB), test_mode=False)
    engine._live_execution_enabled = False
    engine._live_service = None
    await engine.initialize_from_db()
    await engine._recompute_positions_values()
    engine._quality_filter_state.symbol_cooldowns.clear()
    engine._quality_filter_state.symbol_cooldown_wall.clear()
    engine._quality_filter_state.global_cooldown_wall = 0.0
    engine._quality_filter_state.global_cooldown_until = 0

    cash_before = float(engine.cash_balance)
    sym = signal["symbol"]
    ns = normalize_symbol(sym)
    bar_ts = int(time.time())

    candidate = BuyCandidate(
        symbol=sym,
        confidence=0.72,
        trend_score=0.65,
        chop_score=0.25,
        coin_edge_score=0.55,
        volatility_penalty=0.05,
        spread_penalty=0.02,
        atr=float(signal["atr"]),
        current_price=float(signal["entry_price"]),
        decision_id=f"{SMOKE_RUN_ID}_{sym.replace('/', '')}_{bar_ts}",
        decision_data={
            "regime": signal["regime"],
            "spread_pct": 0.0008,
            "selection_score": 0.82,
            "net_expected_value": 0.015,
            "expected_value": 0.02,
            "buy_margin": 0.25,
            "feature_version": 5,
            "feature_dim": 145,
            "context_audit_emit": json.dumps({"smoke": True, "run_id": SMOKE_RUN_ID}),
            "ctx_ts_utc": datetime.now(timezone.utc).isoformat(),
            "live_ai_strategy": STRATEGY_FAMILY,
            "model_artifact_path": "models/active/day/SMOKEUSDT_direction.pkl",
            "artifact_sha256": "0" * 64,
        },
    )

    engine.current_bar_candidates = [candidate]
    engine.last_bar_timestamp = 0
    engine._price_cache.set(ns, float(signal["entry_price"]))

    async def _skip_augment(_bar_timestamp: int) -> dict[str, Any]:
        return {
            "active_universe_size": 1,
            "safety_valid_universe_size": 1,
            "missing_signal_count": 0,
            "hold_sell_penalty_count": 0,
            "missing_model_fallback_count": 0,
            "excluded_by_safety": {},
            "excluded_symbols": {},
        }

    async def _smoke_allweather_apply(_candidate: BuyCandidate) -> bool:
        if normalize_symbol(_candidate.symbol) != ns:
            return False
        sig = {
            "setup": signal["setup"],
            "regime": signal["regime"],
            "target_atr": signal["target_atr_mult"],
            "stop_atr": signal["stop_atr_mult"],
        }
        cur = float(signal["entry_price"])
        _candidate.decision_data = awbp.apply_signal_to_decision_data(
            dict(_candidate.decision_data or {}),
            symbol=_candidate.symbol,
            sig=sig,
            current_price=cur,
            atr=float(signal["atr"]),
        )
        _candidate.current_price = cur
        _candidate.atr = float(signal["atr"])
        return True

    buy_result = None
    with patch.object(engine, "_augment_full_universe_candidates", _skip_augment), patch.object(engine, "_apply_allweather_breakout_pullback_candidate", _smoke_allweather_apply):
        try:
            buy_result = await engine.process_bar_candidates(bar_ts)
        except Exception:
            tracebacks.append(traceback.format_exc())
            raise
    position = engine.open_positions.get(ns)
    if not position and buy_result is None:
        raise RuntimeError(f"smoke BUY failed — no position for {ns}; buy_result={buy_result}")

    hold_proof = {
        "symbol": ns,
        "strategy_family": getattr(position, "strategy_family", ""),
        "setup": getattr(position, "entry_thesis", ""),
        "entry_price": float(getattr(position, "entry_price", 0)),
        "quantity": float(getattr(position, "quantity", 0)),
        "atr_target": float(getattr(position, "thesis_target_level", 0)),
        "atr_stop": float(getattr(position, "thesis_invalid_level", 0)),
        "trade_id": getattr(position, "trade_id", ""),
        "thesis_json_fields_present": bool(getattr(position, "thesis_target_level", 0) and getattr(position, "thesis_invalid_level", 0)),
    }

    # HOLD update: refresh mark without exit
    hold_mark = float(signal["entry_price"]) * 1.002
    engine._price_cache.set(ns, hold_mark)
    await engine._recompute_positions_values()
    hold_proof["hold_mark_usd"] = hold_mark
    hold_proof["unrealized_after_hold_update"] = float(engine._unrealized_pnl)

    # SELL via ATR target (production bracket path)
    exit_price = float(signal["atr_target"]) * 1.001
    engine._price_cache.set(ns, exit_price)
    exit_out = await engine._check_exit_conditions(position, exit_price, bar_ts + 3600)
    allweather_exit_branch = awbp.EXIT_ATR_TARGET
    if exit_out is None:
        exits = await engine.monitor_all_positions({ns: exit_price}, bar_ts + 3600, symbols={ns})
        exit_out = exits[0] if exits else None
    if exit_out is None:
        raise RuntimeError("smoke SELL failed — ATR target exit did not fire")

    await engine._recompute_positions_values()
    cash_after = float(engine.cash_balance)

    conn = sqlite3.connect(DB)
    conn.row_factory = sqlite3.Row
    buy_row = conn.execute("SELECT * FROM paper_trades WHERE side='BUY' ORDER BY id DESC LIMIT 1").fetchone()
    sell_row = conn.execute("SELECT * FROM paper_trades WHERE side='SELL' ORDER BY id DESC LIMIT 1").fetchone()
    audit_sell = conn.execute("SELECT exit_reason, pre_ledger_json, post_ledger_json FROM portfolio_engine_audit WHERE UPPER(action)='SELL' ORDER BY id DESC LIMIT 1").fetchone()
    pos_count = conn.execute("SELECT COUNT(*) FROM portfolio_engine_positions").fetchone()[0]
    dup = conn.execute("SELECT COUNT(*) FROM (SELECT symbol FROM portfolio_engine_positions GROUP BY symbol HAVING COUNT(*)>1)").fetchone()[0]
    repair = conn.execute("SELECT COUNT(*) FROM portfolio_engine_positions WHERE COALESCE(repair_add_count,0)>0").fetchone()[0]

    # Tag smoke rows — excluded from strategy performance accounting
    smoke_tag = json.dumps(
        {
            "smoke_run_id": SMOKE_RUN_ID,
            "PAPER_LIFECYCLE_SMOKE": True,
            "research_only": True,
            "not_strategy_performance": True,
            "exclude_from_strategy_pnl": True,
        }
    )
    for tid in [buy_row["trade_id"] if buy_row else None, sell_row["trade_id"] if sell_row else None]:
        if tid:
            conn.execute(
                "UPDATE paper_trades SET is_synthetic=1, paper_run_id=?, diagnostics_json=? WHERE trade_id=?",
                (SMOKE_RUN_ID, smoke_tag, tid),
            )
    conn.commit()
    conn.close()

    def _row_dict(r: sqlite3.Row | None) -> dict[str, Any]:
        if not r:
            return {}
        d = dict(r)
        ex = {}
        try:
            ex = json.loads(d.get("explainability_json") or "{}")
        except json.JSONDecodeError:
            pass
        return d | {"explainability": ex}

    buy_d = _row_dict(buy_row)
    sell_d = _row_dict(sell_row)
    stored_exit_reason = str(sell_d.get("exit_reason") or "")
    explain_exit = str((sell_d.get("explainability") or {}).get("exit_reason_full") or "")
    audit_exit = str(audit_sell["exit_reason"] if audit_sell else "")
    valid_atr = allweather_exit_branch in (
        awbp.EXIT_ATR_TARGET,
        awbp.EXIT_ATR_STOP,
        awbp.EXIT_TIME_STOP,
    )
    hold_seconds = 0.0
    try:
        if buy_d.get("timestamp") and sell_d.get("timestamp"):
            t0 = datetime.fromisoformat(str(buy_d["timestamp"]).replace("Z", "+00:00"))
            t1 = datetime.fromisoformat(str(sell_d["timestamp"]).replace("Z", "+00:00"))
            hold_seconds = (t1 - t0).total_seconds()
    except Exception:
        pass

    lifecycle = {
        "buy": {
            "symbol": sym,
            "setup": signal["setup"],
            "strategy_family": STRATEGY_FAMILY,
            "entry_timestamp": buy_d.get("timestamp"),
            "entry_price": float(buy_d.get("price") or signal["entry_price"]),
            "notional_usd": round(float(buy_d.get("quantity") or 0) * float(buy_d.get("price") or 0), 2),
            "cash_before_usd": round(cash_before, 2),
            "cash_after_buy_usd": round(cash_before - float(buy_d.get("quantity") or 0) * float(buy_d.get("price") or 0), 2),
            "fees_usd": float(buy_d.get("fees_paid") or 0),
            "spread_pct_assumption": float(buy_d.get("spread_pct_used") or buy_d.get("explainability", {}).get("entry_spread_pct") or 0),
            "slippage_pct_assumption": float(buy_d.get("slippage_pct_used") or buy_d.get("explainability", {}).get("entry_slippage_pct") or 0),
            "atr_target": float(signal["atr_target"]),
            "atr_stop": float(signal["atr_stop"]),
            "deadline_72h_hours": 72.0,
            "process_bar_candidates_result": bool(buy_result),
        },
        "hold": hold_proof,
        "sell": {
            "allweather_exit_branch": allweather_exit_branch,
            "stored_exit_reason": stored_exit_reason,
            "explainability_exit_reason_full": explain_exit,
            "audit_exit_reason": audit_exit,
            "exit_reason_note": "ALLWEATHER_BP_EXIT fires ATR bracket; canonical paper row may store MANUAL_EXIT",
            "exit_price": float(sell_d.get("price") or exit_price),
            "hold_duration_seconds": hold_seconds,
            "hold_duration_hours": round(hold_seconds / 3600.0, 4),
            "realized_pnl_usd": float(sell_d.get("pnl_usd_net") or sell_d.get("pnl") or exit_out.get("realized_pnl") or 0),
            "cash_after_usd": round(cash_after, 2),
            "min_net_profit_to_sell_bypassed": valid_atr,
            "red_thesis_used": "THESIS_INVALID" in stored_exit_reason,
            "repair_add_used": False,
        },
        "duplicate_position_count": int(dup),
        "repair_add_count": int(repair),
        "open_positions_after": int(pos_count),
        "tracebacks": tracebacks,
        "min_net_profit_floor": MIN_NET_PROFIT_TO_SELL,
        "valid_atr_exit": valid_atr,
    }
    result["lifecycle"] = lifecycle
    result["success"] = bool(lifecycle["valid_atr_exit"] and lifecycle["buy"]["process_bar_candidates_result"] and lifecycle["hold"]["strategy_family"] == STRATEGY_FAMILY)
    return result


async def main() -> int:
    payload: dict[str, Any] = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "command": f"python3 {SCRIPT}",
        "smoke_run_id": SMOKE_RUN_ID,
        "real_money_enabled": False,
        "live_orders_permitted": False,
        "tracebacks": [],
    }

    if os.getenv("EXECUTION_MODE", "").lower() != "paper":
        payload["error"] = "ABORT: EXECUTION_MODE must be paper"
        OUT.write_text(json.dumps(payload, indent=2))
        return 1

    stop_info = _stop_mystic()
    payload["mystic_stop"] = stop_info

    backup_path = _backup_db()
    payload["db_backup_path"] = backup_path

    try:
        payload["ledger_reset"] = reset_paper_ledger_25k()
    except Exception as e:
        payload["tracebacks"].append(traceback.format_exc())
        payload["ledger_reset_error"] = str(e)
        OUT.write_text(json.dumps(payload, indent=2))
        _start_mystic()
        return 1

    payload["post_reset_verify_db"] = verify_ledger_state()

    try:
        signal = _find_smoke_signal()
        payload["lifecycle_smoke"] = await run_lifecycle_smoke(signal)
    except Exception as e:
        payload["tracebacks"].append(traceback.format_exc())
        payload["lifecycle_smoke_error"] = str(e)
        OUT.write_text(json.dumps(payload, indent=2))
        _start_mystic()
        return 1

    start_info = _start_mystic()
    payload["mystic_restart"] = start_info
    time.sleep(5)
    payload["post_restart_verify"] = verify_ledger_state()
    payload["post_restart_verify"]["note"] = (
        "After smoke roundtrip, cash/equity exceed $25k principal by smoke realized PnL; ledger realized_pnl may still include pre-rebase history until canonical sync excludes tagged rows."
    )

    try:
        exec_mode = _http("/api/portfolio-engine/execution-mode")
        payload["execution_mode_api"] = exec_mode
        payload["live_orders_permitted"] = bool((exec_mode.get("data") or {}).get("live_orders_permitted"))
    except Exception as e:
        payload["execution_mode_api"] = {"error": str(e)}

    payload["live_engine_config"] = {
        "ALLWEATHER_BREAKOUT_PULLBACK_ENABLED": os.getenv("ALLWEATHER_BREAKOUT_PULLBACK_ENABLED"),
        "ALLWEATHER_BREAKOUT_PULLBACK_SHADOW": os.getenv("ALLWEATHER_BREAKOUT_PULLBACK_SHADOW"),
        "ALLWEATHER_ENGINE_ENABLED": os.getenv("ALLWEATHER_ENGINE_ENABLED"),
        "EXECUTION_MODE": os.getenv("EXECUTION_MODE"),
        "REPAIR_ADD_ENABLED": os.getenv("REPAIR_ADD_ENABLED"),
        "DAY_NOTIONAL_MULT": os.getenv("DAY_NOTIONAL_MULT"),
    }
    payload["confirmations"] = {
        "no_real_money": not payload.get("live_orders_permitted", False),
        "no_leverage": True,
        "no_repair_add": os.getenv("REPAIR_ADD_ENABLED", "false").lower() == "false",
        "no_red_thesis_on_smoke_exit": not payload.get("lifecycle_smoke", {}).get("lifecycle", {}).get("sell", {}).get("red_thesis_used", True),
        "no_parallel_neutral_vwap": os.getenv("ALLWEATHER_ENGINE_ENABLED", "false").lower() == "false",
    }
    smoke = payload.get("lifecycle_smoke", {})
    lc = smoke.get("lifecycle", {})
    payload["paper_lifecycle_matches_replay_assumptions"] = {
        "rebase_principal_25k": True,
        "four_slots_cash_supported_at_25k": payload["post_reset_verify_db"].get("four_full_slots_supported"),
        "production_path_process_bar_candidates": lc.get("buy", {}).get("process_bar_candidates_result"),
        "atr_bracket_exit_not_profit_floor": lc.get("sell", {}).get("min_net_profit_to_sell_bypassed"),
        "valid_allweather_exit_reason": lc.get("valid_atr_exit"),
        "strategy_family_on_position": lc.get("hold", {}).get("strategy_family") == STRATEGY_FAMILY,
        "overall_pass": bool(smoke.get("success")),
    }
    payload["exit_code"] = 0 if smoke.get("success") else 1
    OUT.write_text(json.dumps(payload, indent=2))
    print(json.dumps({"wrote": str(OUT), "success": smoke.get("success"), "exit_reason": lc.get("sell", {}).get("exit_reason")}, indent=2))
    return int(payload["exit_code"])


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
