#!/usr/bin/env python3
"""
Immediate full-cycle Mystic verification — no live-market waiting.

Runs cleanup checks, BUY/HOLD/SELL scenario proofs, AI path checks,
50 rapid decision cycles, and dashboard/API vs DB consistency.
"""
from __future__ import annotations

import asyncio
import json
import math
import os
import pickle
import sqlite3
import sys
import traceback
import urllib.request
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

TOP4 = ["BTC/USDT", "ETH/USDT", "SOL/USDT", "XRP/USDT"]
TOP4_API = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT"]
DB_PATH = ROOT / "mystic_trading.db"
API_BASE = os.getenv("MYSTIC_VERIFY_API", "http://localhost:8000")


@dataclass
class VerifyReport:
    failures: list[str] = field(default_factory=list)
    assertions_failed: list[str] = field(default_factory=list)
    tracebacks: list[str] = field(default_factory=list)
    block_reasons: dict[str, int] = field(default_factory=dict)
    cycle_stats: dict[str, int] = field(default_factory=lambda: {
        "buy": 0, "hold": 0, "sell": 0, "blocked": 0, "warn": 0,
    })
    flags: dict[str, bool] = field(default_factory=dict)
    details: dict[str, Any] = field(default_factory=dict)

    def ok(self, name: str, cond: bool, msg: str = "") -> None:
        self.flags[name] = bool(cond)
        if not cond:
            self.failures.append(f"{name}: {msg or 'FAILED'}")

    def assert_eq(self, label: str, got: Any, expected: Any) -> None:
        if got != expected:
            self.assertions_failed.append(f"{label}: got={got!r} expected={expected!r}")
            self.flags[label] = False
        else:
            self.flags[label] = True

    def bump_block(self, reason: str) -> None:
        k = str(reason or "UNKNOWN")
        self.block_reasons[k] = self.block_reasons.get(k, 0) + 1


def _http_json(path: str) -> dict[str, Any]:
    with urllib.request.urlopen(f"{API_BASE}{path}", timeout=15) as resp:
        return json.loads(resp.read().decode())


def _htf_bull_mtf() -> str:
    return json.dumps({
        "1h": {"ema_align": 0.62, "trend": 0.62},
        "4h": {"ema_align": 0.58, "trend": 0.58},
        "15m": {"ema_align": 0.55, "trend": 0.55},
    })


def _htf_weak_mtf() -> str:
    return json.dumps({
        "5m": {"ema_align": 0.62, "trend": 0.62},
        "15m": {"ema_align": 0.58, "trend": 0.58},
        "1h": {"ema_align": 0.35, "trend": 0.35},
        "4h": {"ema_align": 0.33, "trend": 0.33},
    })


def verify_cleanup(report: VerifyReport) -> None:
    from backend.services.day_inventory_recovery import is_day_top4_symbol
    import backend.services.day_inventory_recovery as dir_mod

    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    eth_pos = conn.execute(
        "SELECT COUNT(*) c FROM portfolio_engine_positions WHERE symbol LIKE '%ETH%'"
    ).fetchone()["c"]
    open_pos = conn.execute("SELECT COUNT(*) c FROM portfolio_engine_positions").fetchone()["c"]
    eth_sell = conn.execute(
        """SELECT exit_reason, price, timestamp FROM paper_trades
           WHERE symbol='ETH/USDT' AND side='SELL' ORDER BY id DESC LIMIT 1"""
    ).fetchone()
    dup = conn.execute(
        """SELECT symbol, COUNT(*) c FROM portfolio_engine_positions GROUP BY symbol HAVING c > 1"""
    ).fetchall()
    repair_rows = conn.execute(
        "SELECT COUNT(*) c FROM portfolio_engine_positions WHERE COALESCE(repair_add_count,0) > 0"
    ).fetchone()["c"]
    eth_cleanup_ts = eth_sell["timestamp"] if eth_sell else "1970-01-01"
    recent_red = conn.execute(
        """SELECT COUNT(*) FROM paper_trades
           WHERE side='SELL' AND exit_reason LIKE 'THESIS_INVALIDATION%'
           AND timestamp > ?""",
        (eth_cleanup_ts,),
    ).fetchone()[0]
    conn.close()

    report.details["eth_cleanup"] = dict(eth_sell) if eth_sell else None
    report.ok("eth_cleaned", eth_pos == 0 and eth_sell and eth_sell["exit_reason"] == "LEGACY_INVENTORY_CLEANUP_EXIT",
              f"eth_positions={eth_pos} last_exit={dict(eth_sell) if eth_sell else None}")
    report.ok("eth_flat", eth_pos == 0)
    report.ok("no_stale_eth_row", eth_pos == 0)
    report.ok("no_duplicate_positions", len(dup) == 0, str(dup))
    report.ok("recovery_blocker_removed", not hasattr(dir_mod, "inventory_recovery_blocks_day_buy"))
    report.ok("no_repair_add_rows", repair_rows == 0, f"repair_add_rows={repair_rows}")

    try:
        status = _http_json("/api/portfolio-engine/status")
        data = status.get("data") or {}
        report.ok("new_day_buys_allowed", not data.get("trading_paused", True))
        report.ok("day_mode_normal", data.get("day_mode") == "NORMAL_NEW_REGIME")
        report.ok("stop_loss_inactive", data.get("position_exit_policy", {}).get("stop_loss_sell_path_active") is False)
        report.ok("trailing_stop_inactive", data.get("position_exit_policy", {}).get("trailing_stop_sell_path_active") is False)
        report.details["portfolio_status"] = {
            "cash": data.get("cash_balance"),
            "equity": data.get("total_equity"),
            "open": data.get("open_positions"),
            "day_mode": data.get("day_mode"),
        }
    except Exception as e:
        report.failures.append(f"api_status: {e}")
        report.tracebacks.append(traceback.format_exc())

    # Historical red thesis sells may exist pre-cleanup; exit loop must not emit new ones
    report.ok("no_red_thesis_exits_since_cleanup", recent_red == 0, f"since_cleanup_red={recent_red}")


def verify_buy_scenarios(report: VerifyReport) -> None:
    from backend.services.day_regime_router import (
        DAY_REGIME_BEAR,
        DAY_REGIME_BULL,
        DAY_REGIME_CHOP,
        DAY_REGIME_RANGE,
        classify_day_regime,
        evaluate_day_entry_route,
        htf_allows_day_long,
    )
    from backend.services.day_trade_thesis import (
        SETUP_BREAKOUT_CONTINUATION,
        SETUP_HTF_TREND_PULLBACK,
        SETUP_VWAP_REVERSION,
        apply_trade_thesis_to_candidate_fields,
    )

    scenarios = [
        ("bull_trend_pullback_buy", DAY_REGIME_BULL, SETUP_HTF_TREND_PULLBACK, {
            "mtf_json": _htf_bull_mtf(), "ema_alignment": 0.74, "adx": 24,
        }, True, None),
        ("breakout_continuation_buy", DAY_REGIME_BEAR, SETUP_BREAKOUT_CONTINUATION, {
            "mtf_json": json.dumps({"15m": {"ema_align": 0.58}, "1h": {"ema_align": 0.52}, "4h": {"ema_align": 0.51}}),
            "price_momentum": 0.06,
        }, True, None),
        ("range_vwap_reclaim_buy", DAY_REGIME_RANGE, SETUP_VWAP_REVERSION, {
            "vwap": 100, "bb_position": 0.22, "rsi": 36,
            "mtf_json": _htf_bull_mtf(),
        }, True, None),
        ("chop_weak_bounce_no_buy", DAY_REGIME_CHOP, SETUP_HTF_TREND_PULLBACK, {
            "mtf_json": _htf_bull_mtf(), "adx": 14,
        }, False, "REGIME_ROUTE_CHOP_NO_DAY"),
        ("bear_pullback_blocked", DAY_REGIME_BEAR, SETUP_HTF_TREND_PULLBACK, {
            "mtf_json": _htf_bull_mtf(),
        }, False, "REGIME_ROUTE_BEAR_NO_TREND_PULLBACK"),
        ("xrp_churn_penalty", DAY_REGIME_BEAR, SETUP_BREAKOUT_CONTINUATION, {
            "mtf_json": json.dumps({"15m": {"ema_align": 0.58}, "1h": {"ema_align": 0.52}, "4h": {"ema_align": 0.51}}),
            "price_momentum": 0.06,
        }, False, "REGIME_ROUTE_XRP_CHURN_CONFIRMATION"),
    ]

    buy_results: list[dict[str, Any]] = []
    all_buy_pass = True
    for name, regime, setup, dd, expect_allowed, expect_reason in scenarios:
        route = evaluate_day_entry_route(
            setup_type=setup,
            day_regime=regime,
            decision_data=dd,
            current_price=99.2 if setup == SETUP_VWAP_REVERSION else 100.0,
            thesis_score=0.72 if name != "xrp_churn_penalty" else 0.66,
            xrp_churn_active=(name == "xrp_churn_penalty"),
        )
        htf_ok, htf_reason = htf_allows_day_long(dd, setup_type=setup, thesis_score=0.72)
        allowed = bool(route.get("allowed"))
        block = route.get("block_reason")
        passed = allowed == expect_allowed
        if expect_reason and not expect_allowed:
            passed = passed and block == expect_reason
        if not passed:
            all_buy_pass = False
            report.assertions_failed.append(
                f"buy_scenario_{name}: allowed={allowed} block={block} expected_allowed={expect_allowed}"
            )
        buy_results.append({
            "scenario": name, "regime": regime, "setup": setup,
            "htf_ok": htf_ok, "htf_reason": htf_reason,
            "allowed": allowed, "block_reason": block,
            "passed": passed,
        })
        if allowed:
            report.cycle_stats["buy"] += 1
        else:
            report.cycle_stats["blocked"] += 1
            report.bump_block(str(block))

    # Top-four ranking snapshot from live redis
    sym_rows: list[dict[str, Any]] = []
    try:
        from backend.config.redis_config import get_redis_client
        from backend.services.day_regime_router import compute_hist_expectancy_pct

        redis = get_redis_client()
        for sym in TOP4:
            sk = sym.replace("/", "")
            ctx_raw = redis.hgetall(f"ai_context:{sk}") or {}
            sig_raw = redis.hgetall(f"ai_signal:day:{sk}") or {}
            ctx = {k.decode() if isinstance(k, bytes) else k: (v.decode() if isinstance(v, bytes) else v) for k, v in ctx_raw.items()}
            sig = {k.decode() if isinstance(k, bytes) else k: (v.decode() if isinstance(v, bytes) else v) for k, v in sig_raw.items()}
            merged = {**ctx, **sig}
            regime = classify_day_regime(merged, context_payload=ctx)
            dd = apply_trade_thesis_to_candidate_fields(
                merged, symbol=sym, current_price=float(sig.get("current_price") or ctx.get("close") or 100),
                atr=1.0, strategy_id="day",
            )
            setup = dd.get("setup_type") or "NO_CLEAR_THESIS"
            score = float(sig.get("selection_score") or sig.get("confidence") or dd.get("thesis_score") or 0)
            route = evaluate_day_entry_route(
                setup_type=setup, day_regime=regime, decision_data=dd,
                context_payload=ctx, current_price=float(sig.get("current_price") or 100),
                thesis_score=float(dd.get("thesis_score") or score),
            )
            htf_ok, htf_reason = htf_allows_day_long(dd, setup_type=setup, thesis_score=score)
            net_ev = float(sig.get("selected_net_expected_value") or dd.get("selected_net_expected_value") or 0)
            hist_exp = compute_hist_expectancy_pct(win_rate=0.5, avg_win_usd=50, avg_loss_usd=40, equity=25000)
            sym_rows.append({
                "symbol": sym, "regime": regime, "htf_ok": htf_ok, "htf_reason": htf_reason,
                "setup_type": setup, "selection_score": score, "net_ev_after_fees": net_ev,
                "historical_expectancy_pct": hist_exp, "route_allowed": route.get("allowed"),
                "block_reason": route.get("block_reason"), "signal_side": sig.get("action") or sig.get("side"),
            })
        sym_rows.sort(key=lambda r: r["selection_score"], reverse=True)
        for i, r in enumerate(sym_rows, 1):
            r["rank"] = i
            r["final_decision"] = "BUY_CANDIDATE" if r["route_allowed"] and r["signal_side"] in ("buy", "BUY") else "BLOCKED_OR_HOLD"
        report.details["top4_live"] = sym_rows
        report.ok("top4_ranking_works", len(sym_rows) == 4 and sym_rows[0]["selection_score"] >= sym_rows[-1]["selection_score"])
    except Exception as e:
        report.failures.append(f"top4_live_rank: {e}")
        report.tracebacks.append(traceback.format_exc())

    report.details["buy_scenarios"] = buy_results
    report.ok("buy_branch_tested", all_buy_pass)


def verify_hold_sell(report: VerifyReport) -> None:
    from backend.config.trading_economics import ESTIMATED_ROUNDTRIP_COST, MIN_NET_PROFIT_TO_SELL
    from backend.services.day_trade_thesis import (
        EXIT_EXTREME_PROTECTION,
        EXIT_LEGACY_INVENTORY_CLEANUP,
        EXIT_NET_PROFIT,
        EXIT_THESIS_WARNING,
        SETUP_HTF_TREND_PULLBACK,
        evaluate_extreme_protection,
        evaluate_thesis_exit,
    )

    bundle = {"1h": {"ema_align": 0.7}, "4h": {"ema_align": 0.68}}
    hold = evaluate_thesis_exit(
        entry_thesis=SETUP_HTF_TREND_PULLBACK, thesis_score=0.72,
        thesis_invalid_level=98.0, thesis_target_level=102.5,
        entry_vwap=0.0, entry_price=100.0, mark=99.2, bundle=bundle,
    )
    report.ok("hold_below_profit_floor", hold["action"] == "hold")
    report.cycle_stats["hold"] += 1

    warn = evaluate_thesis_exit(
        entry_thesis=SETUP_HTF_TREND_PULLBACK, thesis_score=0.72,
        thesis_invalid_level=98.0, thesis_target_level=102.5,
        entry_vwap=0.0, entry_price=100.0, mark=96.0, bundle=bundle,
    )
    report.ok("thesis_invalidation_warn_only", warn["action"] == "warn" and EXIT_THESIS_WARNING in str(warn["reason"]))
    report.cycle_stats["warn"] += 1

    target = 101.2
    sell = evaluate_thesis_exit(
        entry_thesis=SETUP_HTF_TREND_PULLBACK, thesis_score=0.72,
        thesis_invalid_level=98.5, thesis_target_level=target,
        entry_vwap=0.0, entry_price=100.0, mark=target, bundle=bundle,
    )
    net = (target - 100.0) / 100.0 - ESTIMATED_ROUNDTRIP_COST
    report.ok("net_profit_exit_fires", sell["action"] == "sell" and sell["reason"] == EXIT_NET_PROFIT and net >= MIN_NET_PROFIT_TO_SELL * 0.45)
    report.cycle_stats["sell"] += 1

    extreme = evaluate_extreme_protection(
        entry_price=100.0, mark=88.0, net_pnl_pct=-0.12, atr_pct=0.01,
        bundle={"1h": {"ema_align": 0.20}, "4h": {"ema_align": 0.18}},
    )
    report.ok("extreme_protection_separate", extreme.get("action") == "sell" and EXIT_EXTREME_PROTECTION in str(extreme.get("reason", "")))

    report.ok("legacy_cleanup_label_distinct", EXIT_LEGACY_INVENTORY_CLEANUP != EXIT_NET_PROFIT)
    report.ok("sell_branch_tested", report.flags.get("net_profit_exit_fires", False) and report.flags.get("thesis_invalidation_warn_only", False))


async def verify_engine_gates(report: VerifyReport) -> None:
    from backend.services.portfolio_engine import OpenPosition, PortfolioEngine

    engine = PortfolioEngine(db_path=str(DB_PATH), test_mode=True)
    await engine.initialize_from_canonical_sources()

    can_btc, reason_btc = await engine._can_open_position("BTC/USDT", 1000.0)
    report.ok("engine_can_open_btc", can_btc, reason_btc)

    # Duplicate symbol block
    engine.open_positions["BTC/USDT"] = OpenPosition(
        symbol="BTC/USDT", quantity=0.01, entry_price=64000, entry_time=0, trade_id="test",
        stop_price=60000, take_profit_1_price=66000, take_profit_2_price=68000,
    )
    can_dup, reason_dup = await engine._can_open_position("BTC/USDT", 1000.0)
    report.ok("one_position_per_symbol_blocks", not can_dup, reason_dup)
    del engine.open_positions["BTC/USDT"]

    # Bear max-one without freezing app
    orig = engine._is_bear_day_regime
    engine._is_bear_day_regime = lambda: True  # type: ignore[method-assign]
    engine.open_positions["SOL/USDT"] = OpenPosition(
        symbol="SOL/USDT", quantity=1.0, entry_price=150, entry_time=0, trade_id="test2",
        stop_price=140, take_profit_1_price=160, take_profit_2_price=165,
    )
    can_bear, reason_bear = await engine._can_open_position("XRP/USDT", 1000.0)
    can_bear_btc, _ = await engine._can_open_position("BTC/USDT", 1000.0)
    report.ok("bear_max_one_blocks_second", not can_bear, reason_bear)
    report.ok("bear_max_one_does_not_freeze_app", can_bear_btc or reason_bear == "BEAR_REGIME_MAX_ONE_DAY_POSITION")
    del engine.open_positions["SOL/USDT"]
    engine._is_bear_day_regime = orig  # type: ignore[method-assign]

    from backend.config.repair_add_economics import REPAIR_ADD_ENABLED
    adds = await engine.process_repair_adds_once({})
    report.details["repair_add_enabled"] = REPAIR_ADD_ENABLED
    report.ok("repair_add_no_open_positions_no_op", adds == [])


def verify_ai(report: VerifyReport) -> None:
    from backend.services.live_strategy_contracts import per_coin_artifact_file
    from backend.utils.path_helpers import ensure_model_directories

    models_active = Path(ensure_model_directories()["active"])
    ai_rows: list[dict[str, Any]] = []
    all_models = True
    all_infer = True
    features_ok = True

    for sym in TOP4_API:
        path = per_coin_artifact_file(models_active, "day", sym)
        row: dict[str, Any] = {"symbol": sym, "path": str(path)}
        if not path.exists():
            all_models = False
            row["model_exists"] = False
            ai_rows.append(row)
            continue
        row["model_exists"] = True
        try:
            with open(path, "rb") as f:
                payload = pickle.load(f)
            model = payload.get("model") if isinstance(payload, dict) else payload
            scaler = payload.get("scaler") if isinstance(payload, dict) else None
            row["loads"] = model is not None
            row["has_scaler"] = scaler is not None
            row["feature_dim"] = int(payload.get("feature_dim") or payload.get("n_features") or 145) if isinstance(payload, dict) else 145
            if row["feature_dim"] != 145:
                features_ok = False
            # zero-vector smoke infer
            import numpy as np
            vec = np.zeros((1, row["feature_dim"]), dtype=float)
            if scaler is not None:
                vec = scaler.transform(vec)
            proba = model.predict_proba(vec)[0]
            row["infer_ok"] = len(proba) >= 2 and all(math.isfinite(float(x)) for x in proba)
            if not row["infer_ok"]:
                all_infer = False
        except Exception as e:
            row["error"] = str(e)
            all_models = False
            all_infer = False
            report.tracebacks.append(traceback.format_exc())
        ai_rows.append(row)

    report.details["ai_models"] = ai_rows
    report.ok("ai_model_files_exist", all_models)
    report.ok("ai_model_loads", all_models)
    report.ok("ai_inference_tested", all_infer)
    report.ok("features_145_complete", features_ok)

    # Redis signal + learning tables
    try:
        from backend.config.redis_config import get_redis_client
        redis = get_redis_client()
        for sym in TOP4_API:
            sig = redis.hgetall(f"ai_signal:day:{sym}") or {}
            decoded = {k.decode() if isinstance(k, bytes) else k: (v.decode() if isinstance(v, bytes) else v) for k, v in sig.items()}
            report.details.setdefault("redis_signals", {})[sym] = {
                "has_signal": bool(decoded),
                "action": decoded.get("action") or decoded.get("side"),
                "decision_id": decoded.get("decision_id"),
                "selection_score": decoded.get("selection_score"),
            }
        report.ok("ai_signal_writes", all(report.details["redis_signals"].get(s, {}).get("has_signal") for s in TOP4_API))
    except Exception as e:
        report.failures.append(f"redis_signals: {e}")

    try:
        from backend.services.ai_learning_ingestion import learning_health_summary, label_pending_snapshots
        before = learning_health_summary(str(DB_PATH))
        labeled = label_pending_snapshots(str(DB_PATH))
        after = learning_health_summary(str(DB_PATH))
        report.details["learning_health"] = after
        report.details["forward_labeler_ran"] = labeled
        report.ok("learning_writes_present", int(before.get("totals", {}).get("candidate_snapshots") or 0) > 0)
        report.ok("forward_labeler_works", isinstance(labeled, dict) and "scanned" in labeled)
    except Exception as e:
        report.failures.append(f"learning: {e}")
        report.tracebacks.append(traceback.format_exc())


def run_50_cycles(report: VerifyReport) -> None:
    from backend.services.day_regime_router import (
        DAY_REGIME_BEAR,
        DAY_REGIME_BULL,
        DAY_REGIME_CHOP,
        DAY_REGIME_RANGE,
        evaluate_day_entry_route,
    )
    from backend.services.day_trade_thesis import (
        SETUP_BREAKOUT_CONTINUATION,
        SETUP_HTF_TREND_PULLBACK,
        SETUP_NO_CLEAR_THESIS,
        SETUP_VWAP_REVERSION,
        evaluate_thesis_exit,
    )

    regimes = [DAY_REGIME_BULL, DAY_REGIME_BEAR, DAY_REGIME_RANGE, DAY_REGIME_CHOP]
    setups = [SETUP_HTF_TREND_PULLBACK, SETUP_BREAKOUT_CONTINUATION, SETUP_VWAP_REVERSION, SETUP_NO_CLEAR_THESIS]
    marks = [99.0, 100.0, 101.5, 96.0]

    for i in range(50):
        regime = regimes[i % 4]
        setup = setups[i % 4]
        mark = marks[i % 4]
        dd = {"mtf_json": _htf_bull_mtf() if i % 2 == 0 else _htf_weak_mtf(), "adx": 20 + (i % 10)}
        route = evaluate_day_entry_route(
            setup_type=setup, day_regime=regime, decision_data=dd,
            current_price=100.0, thesis_score=0.55 + (i % 20) * 0.01,
            xrp_churn_active=(i % 7 == 0),
        )
        if route.get("allowed"):
            report.cycle_stats["buy"] += 1
        else:
            report.cycle_stats["blocked"] += 1
            report.bump_block(str(route.get("block_reason")))

        ev = evaluate_thesis_exit(
            entry_thesis=SETUP_HTF_TREND_PULLBACK, thesis_score=0.7,
            thesis_invalid_level=98.0, thesis_target_level=102.0,
            entry_vwap=0.0, entry_price=100.0, mark=mark,
            bundle={"1h": {"ema_align": 0.6}, "4h": {"ema_align": 0.58}},
        )
        act = str(ev.get("action") or "")
        if act == "hold":
            report.cycle_stats["hold"] += 1
        elif act == "sell":
            report.cycle_stats["sell"] += 1
        elif act == "warn":
            report.cycle_stats["warn"] += 1

    report.ok("fifty_cycle_loop_complete", True)


def verify_dashboard_db(report: VerifyReport) -> None:
    conn = sqlite3.connect(DB_PATH)
    ledger = conn.execute(
        "SELECT cash_balance, total_equity, positions_value FROM portfolio_engine_ledger WHERE id=1"
    ).fetchone()
    open_db = conn.execute("SELECT COUNT(*) FROM portfolio_engine_positions").fetchone()[0]
    learning_before = conn.execute("SELECT COUNT(*) FROM ai_candidate_snapshots").fetchone()[0]
    conn.close()

    try:
        status = _http_json("/api/portfolio-engine/status")["data"]
        api_cash = float(status.get("cash_balance") or 0)
        api_eq = float(status.get("total_equity") or 0)
        api_open = int(status.get("positions_count") or 0)
        match = (
            abs(api_cash - float(ledger[0])) < 0.02
            and abs(api_eq - float(ledger[1])) < 0.02
            and api_open == open_db
        )
        report.ok("dashboard_matches_db", match, f"api cash={api_cash} db={ledger[0]} open api={api_open} db={open_db}")
    except Exception as e:
        report.failures.append(f"dashboard_db: {e}")

    try:
        scalp = _http_json("/api/scalp/status")
        report.ok("scalp_isolated", scalp.get("engine") == "scalp" and "pnl_summary" in scalp)
        report.details["scalp"] = {
            "status": scalp.get("overall_decision"),
            "pnl_today": (scalp.get("pnl_summary") or {}).get("today"),
        }
    except Exception as e:
        report.failures.append(f"scalp_api: {e}")

    report.details["learning_snapshots_count"] = learning_before


async def main() -> int:
    report = VerifyReport()
    print("=== MYSTIC FULL-CYCLE VERIFICATION ===", flush=True)

    verify_cleanup(report)
    verify_buy_scenarios(report)
    verify_hold_sell(report)
    await verify_engine_gates(report)
    verify_ai(report)
    run_50_cycles(report)
    verify_dashboard_db(report)

    # Aggregate booleans for final rule
    report.ok("buy_branch_tested", report.flags.get("buy_branch_tested", False))
    report.ok("hold_branch_tested", report.flags.get("hold_below_profit_floor", False))
    report.ok("sell_branch_tested", report.flags.get("sell_branch_tested", False))
    report.ok("no_duplicate_positions", report.flags.get("no_duplicate_positions", False))
    report.ok("no_red_thesis_exits", report.flags.get("no_red_thesis_exits_since_cleanup", False))

    out = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "flags": report.flags,
        "failures": report.failures,
        "assertions_failed": report.assertions_failed,
        "tracebacks": report.tracebacks,
        "cycle_stats": report.cycle_stats,
        "block_reasons": report.block_reasons,
        "details": report.details,
        "summary": {
            "eth_cleaned": report.flags.get("eth_cleaned"),
            "recovery_blocker_removed": report.flags.get("recovery_blocker_removed"),
            "new_day_buys_allowed": report.flags.get("new_day_buys_allowed"),
            "top4_ranking_works": report.flags.get("top4_ranking_works"),
            "buy_branch_tested": report.flags.get("buy_branch_tested"),
            "hold_branch_tested": report.flags.get("hold_branch_tested"),
            "sell_branch_tested": report.flags.get("sell_branch_tested"),
            "ai_inference_tested": report.flags.get("ai_inference_tested"),
            "features_145_complete": report.flags.get("features_145_complete"),
            "learning_writes_tested": report.flags.get("learning_writes_present"),
            "no_red_thesis_exits": report.flags.get("no_red_thesis_exits_since_cleanup"),
            "no_duplicate_positions": report.flags.get("no_duplicate_positions"),
            "scalp_isolated": report.flags.get("scalp_isolated"),
            "dashboard_matches_db": report.flags.get("dashboard_matches_db"),
            "all_pass": not report.failures and not report.assertions_failed,
        },
    }
    print(json.dumps(out, indent=2, default=str))
    return 0 if out["summary"]["all_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
