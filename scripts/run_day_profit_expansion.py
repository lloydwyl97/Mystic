#!/usr/bin/env python3
"""
DAY profit-expansion sweeps — replay-only on locked positive buckets.

Does NOT modify live rules, revive killed buckets, or add blockers/modes.
"""
from __future__ import annotations

import json
import sys
import traceback
from copy import deepcopy
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
BASELINE_DIR = REPO / "scripts" / "replay_baselines"
CACHE_DIR = BASELINE_DIR / "cache"

from backend.config.trading_economics import MIN_NET_PROFIT_TO_SELL
from backend.services.day_bucket_quality import REPLAY_KILLED_BUCKETS
from backend.services.day_regime_router import DAY_REGIME_BEAR, DAY_REGIME_BULL
from backend.services.day_trade_thesis import SETUP_BREAKOUT_CONTINUATION, SETUP_HTF_TREND_PULLBACK, SETUP_VWAP_REVERSION
from scripts.run_day_execution_replay import (
    ALLOWED_POSITIVE_BUCKETS,
    ExecutionConfig,
    _build_fee_profiles,
    _run_stress_90d,
    fetch_klines_cached,
    run_execution_replay,
)
from scripts.run_day_strategy_replay import NOTIONAL_USD, PRINCIPAL, SYMBOLS, fetch_klines_1h
from scripts.run_day_bucket_discovery import _scan_opportunities

MAX_HOLD_HOURS_FAT_TAIL = 72.0
MAX_DD_PCT = 8.0


def _metrics(r: dict[str, Any]) -> dict[str, Any]:
    return {
        "net_pnl_usd": round(float(r.get("net_pnl_usd") or 0), 2),
        "gross_pnl_usd": round(float(r.get("gross_pnl_usd") or 0), 2),
        "max_drawdown_pct": r.get("max_drawdown_pct"),
        "worst_intrabar_mae_pct": r.get("worst_intrabar_mae_pct"),
        "avg_intrabar_mae_pct": r.get("avg_intrabar_mae_pct"),
        "avg_hold_hours": round(float(r.get("avg_hold_hours") or 0), 2),
        "longest_hold_hours": round(float(r.get("longest_hold_hours") or 0), 2),
        "expectancy_per_trade_usd": round(float(r.get("expectancy_per_trade_usd") or 0), 2),
        "total_trades": r.get("total_trades"),
        "red_thesis_sell_count": r.get("red_thesis_sell_count", 0),
        "duplicate_attempts": r.get("duplicate_attempts", 0),
    }


def _monthly_pnl(net_90d: float) -> float:
    return round(net_90d / 3.0, 2)


def _account_return_pct(net_90d: float) -> float:
    return round(100.0 * net_90d / PRINCIPAL, 4)


def _expansion_pass(w30: dict, w90: dict, wf_test: dict) -> dict[str, Any]:
    longest = float(w90.get("longest_hold_hours") or 0)
    checks = {
        "30d_net_positive": (w30.get("net_pnl_usd") or 0) > 0,
        "90d_net_positive": (w90.get("net_pnl_usd") or 0) > 0,
        "wf_test_positive": (wf_test.get("expectancy_per_trade_usd") or 0) > 0,
        "no_red_thesis_sells": w90.get("red_thesis_sell_count", 0) == 0,
        "no_duplicates": w90.get("duplicate_attempts", 0) == 0,
        "max_dd_under_cap": (w90.get("max_drawdown_pct") or 99) < MAX_DD_PCT,
        "no_fat_tail_holds": longest <= MAX_HOLD_HOURS_FAT_TAIL,
        "worst_mae_ok": (w90.get("worst_intrabar_mae_pct") or 0) > -0.20,
    }
    checks["all_pass"] = all(checks.values())
    return checks


def _base_config(name: str, profiles: dict[str, ExecutionConfig]) -> ExecutionConfig:
    p = profiles["binance_us_taker"]
    half = deepcopy(p.half_spread_by_symbol)
    return ExecutionConfig(
        name=name,
        execution_style=p.execution_style,
        maker_fee=p.maker_fee,
        taker_fee=p.taker_fee,
        slippage_buffer=p.slippage_buffer,
        platform_spread_one_way=0.0,
        half_spread_by_symbol=half,
        use_fill_based_exit_gate=True,
        allowed_buckets_only=True,
    )


def _run_suite(
    bars_1h: dict,
    bars_exec: dict,
    cfg: ExecutionConfig,
    *,
    start_ts: int,
    end_ts: int,
) -> dict[str, Any]:
    w30s = end_ts - 30 * 86400
    w90s = end_ts - 90 * 86400
    span = end_ts - start_ts
    t_end = start_ts + int(span * 0.50)
    v_end = start_ts + int(span * 0.75)
    w30 = run_execution_replay(
        bars_1h, bars_exec, window_days=30, start_ts=max(w30s, start_ts), end_ts=end_ts,
        config=cfg, exec_interval="15m",
    )
    w90 = run_execution_replay(
        bars_1h, bars_exec, window_days=90, start_ts=max(w90s, start_ts), end_ts=end_ts,
        config=cfg, exec_interval="15m",
    )
    test = run_execution_replay(
        bars_1h, bars_exec, window_days=int((end_ts - v_end) / 86400),
        start_ts=v_end, end_ts=end_ts, config=cfg, exec_interval="15m",
    )
    return {"30d": w30, "90d": w90, "wf_test": test}


def _summarize_run(label: str, suite: dict[str, Any]) -> dict[str, Any]:
    w30, w90, test = suite["30d"], suite["90d"], suite["wf_test"]
    net90 = float(w90.get("net_pnl_usd") or 0)
    return {
        "label": label,
        "metrics_90d": _metrics(w90),
        "metrics_30d": _metrics(w30),
        "wf_test_expectancy_usd": round(float(test.get("expectancy_per_trade_usd") or 0), 2),
        "monthly_pnl_usd_on_25k": _monthly_pnl(net90),
        "account_return_90d_pct": _account_return_pct(net90),
        "pass": _expansion_pass(w30, w90, test),
    }


def _load_bars(days: int) -> tuple[dict, dict, int, int, int]:
    end = datetime.now(timezone.utc)
    start = end - timedelta(days=days + 5)
    start_ms = int(start.timestamp() * 1000)
    end_ms = int(end.timestamp() * 1000)
    bars_1h = {sym: fetch_klines_cached(sym, "1h", start_ms, end_ms) for sym in SYMBOLS}
    bars_exec = {sym: fetch_klines_cached(sym, "15m", start_ms, end_ms) for sym in SYMBOLS}
    start_ts = bars_1h[SYMBOLS[0]][0]["ts"]
    end_ts = bars_1h[SYMBOLS[0]][-1]["ts"]
    cache_days = int((end_ts - start_ts) / 86400)
    return bars_1h, bars_exec, start_ts, end_ts, cache_days


def _scan_extended_buckets(bars_1h: dict, cache_days: int) -> dict[str, Any]:
    out: list[dict[str, Any]] = []
    for sym in SYMBOLS:
        for reg, thesis in (
            (DAY_REGIME_BULL, SETUP_HTF_TREND_PULLBACK),
            (DAY_REGIME_BULL, SETUP_BREAKOUT_CONTINUATION),
            (DAY_REGIME_BEAR, SETUP_VWAP_REVERSION),
            (DAY_REGIME_BEAR, SETUP_BREAKOUT_CONTINUATION),
            ("neutral", SETUP_BREAKOUT_CONTINUATION),
            ("range", SETUP_VWAP_REVERSION),
        ):
            scan = _scan_opportunities(bars_1h, symbol=sym, regime=reg, thesis=thesis)
            out.append({
                "bucket": f"{sym}/{reg}/{thesis}",
                "would_enter": scan.get("would_enter", 0),
                "in_positive_allowlist": (sym, reg, thesis) in ALLOWED_POSITIVE_BUCKETS,
                "killed": (sym, reg, thesis) in REPLAY_KILLED_BUCKETS,
            })
    return {"cache_days": cache_days, "candidates": out}


def _scalp_contribution() -> dict[str, Any]:
    import sqlite3
    from backend.database_schema import DATABASE_PATH

    scalp_monthly = 0.0
    scalp_90d = 0.0
    scalp_dd = 0.0
    note = "Scalp isolated from DAY ledger"
    try:
        conn = sqlite3.connect(DATABASE_PATH, timeout=3)
        row = conn.execute(
            """SELECT SUM(realized_pnl) FROM scalp_paper_trades
               WHERE side='SELL' AND timestamp >= datetime('now', '-90 days')"""
        ).fetchone()
        conn.close()
        if row and row[0]:
            scalp_90d = float(row[0])
            scalp_monthly = round(scalp_90d / 3.0, 2)
    except Exception:
        note = "scalp_paper_trades unavailable; replay report shows 0 trades in calibration mode"
    try:
        rep = json.loads((REPO / "scripts/scalp_strategy_replay_report.json").read_text())
        if rep.get("by_strategy"):
            scalp_90d = sum(float(v.get("net_pnl_usd") or 0) for v in rep["by_strategy"].values())
            scalp_monthly = round(scalp_90d / 3.0, 2) if scalp_90d else 0.0
            note = f"scalp replay ({rep.get('replay_hours', '?')}h calibration): {rep.get('enabled_strategies')}"
    except Exception:
        pass
    return {
        "scalp_90d_net_pnl_usd": round(scalp_90d, 2),
        "scalp_expected_monthly_usd": scalp_monthly,
        "scalp_max_drawdown_pct": scalp_dd,
        "note": note,
    }


def main() -> int:
    print("=== DAY PROFIT EXPANSION SWEEPS (positive buckets only) ===", flush=True)
    tracebacks: list[str] = []
    try:
        profiles = _build_fee_profiles()
        bars_1h, bars_exec, start_ts, end_ts, cache_days = _load_bars(95)
        print(f"  cache: {cache_days}d 1h bars", flush=True)

        baseline_cfg = _base_config("baseline", profiles)
        baseline_cfg.notional_mult = 1.0
        baseline_suite = _run_suite(bars_1h, bars_exec, baseline_cfg, start_ts=start_ts, end_ts=end_ts)
        baseline_sum = _summarize_run("baseline", baseline_suite)
        day_monthly = baseline_sum["monthly_pnl_usd_on_25k"]

        def _quick_90(cfg: ExecutionConfig) -> dict[str, Any]:
            w90 = _run_stress_90d(bars_1h, {"15m": bars_exec}, cfg)
            return _metrics(w90)

        # 1. Position size
        size_results: list[dict] = []
        for mult in (0.72, 1.00, 1.25, 1.50):
            cfg = _base_config(f"size_{mult}", profiles)
            cfg.notional_mult = mult
            print(f"  size sweep {mult}...", flush=True)
            m90 = _quick_90(cfg)
            net90 = m90["net_pnl_usd"]
            size_results.append({
                "label": f"notional_mult_{mult}",
                "metrics_90d": m90,
                "deployed_notional_per_slot_usd": round(NOTIONAL_USD * mult, 2),
                "max_deployed_usd_4_slots": round(NOTIONAL_USD * mult * 4, 2),
                "monthly_pnl_usd_on_25k": _monthly_pnl(net90),
                "account_return_90d_pct": _account_return_pct(net90),
            })

        # 2. Net-profit floor
        floor_results: list[dict] = []
        for pct in (0.0015, 0.0025, 0.004, 0.006, 0.008, 0.010):
            cfg = _base_config(f"floor_{pct}", profiles)
            cfg.min_net_profit_floor = pct
            print(f"  floor sweep {pct*100:.2f}%...", flush=True)
            m90 = _quick_90(cfg)
            net90 = m90["net_pnl_usd"]
            floor_results.append({
                "label": f"min_net_profit_{pct*100:.2f}pct",
                "metrics_90d": m90,
                "monthly_pnl_usd_on_25k": _monthly_pnl(net90),
                "account_return_90d_pct": _account_return_pct(net90),
                "exits_too_early_signal": net90 > baseline_sum["metrics_90d"]["net_pnl_usd"] and m90["longest_hold_hours"] > baseline_sum["metrics_90d"]["longest_hold_hours"],
            })

        # 3. Profit capture
        capture_results: list[dict] = []
        for mode, label in (("none", "immediate_exit"), ("vwap_continuation", "vwap_hold_after_floor")):
            cfg = _base_config(label, profiles)
            cfg.profit_capture_mode = mode
            print(f"  capture {label}...", flush=True)
            m90 = _quick_90(cfg)
            net90 = m90["net_pnl_usd"]
            capture_results.append({
                "label": label,
                "metrics_90d": m90,
                "monthly_pnl_usd_on_25k": _monthly_pnl(net90),
                "account_return_90d_pct": _account_return_pct(net90),
            })

        # 4. Maker execution
        exec_results: list[dict] = []
        styles = [
            ("binance_us_taker", "taker_taker"),
            ("maker_entry_taker_exit", "maker_entry_taker_exit"),
            ("maker_maker_calm", "maker_maker_when_calm"),
        ]
        for style, label in styles:
            cfg = _base_config(label, profiles)
            cfg.execution_style = style
            print(f"  execution {label}...", flush=True)
            m90 = _quick_90(cfg)
            net90 = m90["net_pnl_usd"]
            exec_results.append({
                "label": label,
                "metrics_90d": m90,
                "monthly_pnl_usd_on_25k": _monthly_pnl(net90),
                "account_return_90d_pct": _account_return_pct(net90),
            })

        # Full validation on baseline + top 90d candidates
        quick_all = size_results + floor_results + capture_results + exec_results
        top_candidates = sorted(quick_all, key=lambda x: x["monthly_pnl_usd_on_25k"], reverse=True)[:4]
        full_validation: dict[str, Any] = {"baseline": baseline_sum}
        for cand in top_candidates:
            lbl = cand["label"]
            cfg = _base_config(lbl, profiles)
            if lbl.startswith("notional_mult_"):
                cfg.notional_mult = float(lbl.replace("notional_mult_", ""))
            elif lbl.startswith("min_net_profit_"):
                pct_str = lbl.replace("min_net_profit_", "").replace("pct", "")
                cfg.min_net_profit_floor = float(pct_str) / 100.0
            elif lbl == "vwap_hold_after_floor":
                cfg.profit_capture_mode = "vwap_continuation"
            elif lbl in ("taker_taker", "maker_entry_taker_exit", "maker_maker_when_calm"):
                style_map = {
                    "taker_taker": "binance_us_taker",
                    "maker_entry_taker_exit": "maker_entry_taker_exit",
                    "maker_maker_when_calm": "maker_maker_calm",
                }
                cfg.execution_style = style_map[lbl]
            print(f"  full validation {lbl}...", flush=True)
            suite = _run_suite(bars_1h, bars_exec, cfg, start_ts=start_ts, end_ts=end_ts)
            full_validation[lbl] = _summarize_run(lbl, suite)

        for row in quick_all:
            fv = full_validation.get(row["label"])
            row["pass"] = fv.get("pass") if fv else {"all_pass": None, "note": "90d-only quick sweep"}

        passing = [r for r in quick_all if (full_validation.get(r["label"]) or {}).get("pass", {}).get("all_pass")]
        best = max(passing, key=lambda x: x["monthly_pnl_usd_on_25k"], default=None)
        ext_days_target = min(730, cache_days + 400)
        print(f"  extended history fetch target {ext_days_target}d...", flush=True)
        ext_end = datetime.now(timezone.utc)
        ext_start = ext_end - timedelta(days=ext_days_target + 5)
        ext_ms = int(ext_start.timestamp() * 1000)
        ext_end_ms = int(ext_end.timestamp() * 1000)
        bars_ext = {sym: fetch_klines_cached(sym, "1h", ext_ms, ext_end_ms) for sym in SYMBOLS}
        ext_cache_days = int((bars_ext[SYMBOLS[0]][-1]["ts"] - bars_ext[SYMBOLS[0]][0]["ts"]) / 86400)
        extended_scan = _scan_extended_buckets(bars_ext, ext_cache_days)
        ext_positive = [c for c in extended_scan["candidates"] if c["would_enter"] > 0 and c["in_positive_allowlist"]]
        ext_other = [c for c in extended_scan["candidates"] if c["would_enter"] > 0 and not c["in_positive_allowlist"]]

        # 6. Scalp
        scalp = _scalp_contribution()
        combined_monthly = round(day_monthly + float(scalp.get("scalp_expected_monthly_usd") or 0), 2)

        b90 = baseline_sum["metrics_90d"]
        pnl_math = {
            "principal_usd": PRINCIPAL,
            "notional_per_slot_usd": NOTIONAL_USD,
            "max_slots": 4,
            "max_deployed_capital_usd": NOTIONAL_USD * 4,
            "capital_utilization_note": (
                "DAY uses up to 4×$2500 = $10k deployed (40% of $25k). "
                "Expectancy ~$8/trade is on ~$2500 notional (~0.35%/trade), NOT on full $25k."
            ),
            "baseline_90d_net_usd": b90["net_pnl_usd"],
            "baseline_account_return_90d_pct": baseline_sum["account_return_90d_pct"],
            "baseline_monthly_on_full_25k_usd": day_monthly,
            "baseline_monthly_if_fully_deployed_10k_usd": round(b90["net_pnl_usd"] / 3.0 * (PRINCIPAL / (NOTIONAL_USD * 4)), 2),
            "trades_90d": b90["total_trades"],
            "why_not_8_dollars_on_25k": (
                f"20 trades × ${b90['expectancy_per_trade_usd']:.2f} ≈ ${b90['net_pnl_usd']:.0f} on $25k account "
                f"= {baseline_sum['account_return_90d_pct']}% / 90d ≈ ${day_monthly}/month. "
                "Low frequency + partial capital deployment, not broken math."
            ),
        }

        report = {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "active_buckets_only": [list(x) for x in sorted(ALLOWED_POSITIVE_BUCKETS)],
            "killed_buckets_unchanged": [list(k) for k in sorted(REPLAY_KILLED_BUCKETS)],
            "live_min_net_profit_floor": MIN_NET_PROFIT_TO_SELL,
            "pnl_math_explanation": pnl_math,
            "baseline": baseline_sum,
            "sweeps": {
                "position_size": size_results,
                "net_profit_floor": floor_results,
                "profit_capture": capture_results,
                "execution_style": exec_results,
            },
            "extended_history": {
                "days_available": ext_cache_days,
                "bull_bear_neutral_opportunities": ext_other,
                "positive_bucket_opportunities": ext_positive,
                "note": "Non-allowlist opportunities are informational only — not enabled",
            },
            "scalp_and_combined": {
                "day_baseline_monthly_usd": day_monthly,
                **scalp,
                "combined_expected_monthly_usd": combined_monthly,
                "drawdowns_separate": True,
            },
            "full_validation": full_validation,
            "best_passing_variant": best,
            "passing_variant_count": len(passing),
            "tracebacks": tracebacks,
        }

        out = BASELINE_DIR / "day_profit_expansion_latest.json"
        out.write_text(json.dumps(report, indent=2, default=str))
        print(json.dumps(report, indent=2, default=str))
        return 0
    except Exception:
        tracebacks.append(traceback.format_exc())
        print(json.dumps({"error": tracebacks}, indent=2))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
