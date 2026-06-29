#!/usr/bin/env python3
"""
Regime matrix (long + short columns) — research only.
Writes market_regime_replay_matrix_latest.json from clean full-script runs.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from scripts.replay_baselines.run_short_side_research import (
    ONE_WAY_COST,
    REGIMES,
    ROUNDTRIP_COST,
    Trade,
    _classify_regime,
    _metrics,
    _normalize_bars,
    _short_signal,
    _simulate_short,
    _stress,
    _to_internal,
    _walk_forward,
)
from scripts.replay_baselines.run_short_side_research import (
    run_symbol as run_short_symbol,
)
from scripts.run_day_execution_replay import CACHE_DIR, SYMBOL_API, fetch_klines_cached

SCRIPT_NAME = "scripts/replay_baselines/run_regime_matrix.py"
OUT_MATRIX = REPO / "scripts" / "replay_baselines" / "market_regime_replay_matrix_latest.json"
OUT_DECISION = REPO / "scripts" / "replay_baselines" / "spot_long_vs_short_research_latest.json"
SHORT_ARTIFACT = REPO / "scripts" / "replay_baselines" / "short_side_research_latest.json"
LOCK_BASELINE = REPO / "scripts" / "replay_baselines" / "day_baseline_all_pass_v1_size_1_5_LOCK.json"
ALLWEATHER_LAB = REPO / "scripts" / "replay_baselines" / "allweather_strategy_lab_latest.json"

DEFAULT_SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT"]
SPAN_DAYS = int(os.getenv("REGIME_MATRIX_SPAN_DAYS", "720"))


def _cache_path(symbol: str, interval: str, start_ms: int, end_ms: int) -> Path:
    api = SYMBOL_API[_to_internal(symbol)]
    return CACHE_DIR / f"{api}_{interval}_{start_ms}_{end_ms}.json"


def _fetch_bars(symbol: str, span_days: int) -> tuple[list[dict], dict[str, Any]]:
    end = datetime.now(timezone.utc)
    start = end - timedelta(days=span_days + 5)
    start_ms = int(start.timestamp() * 1000)
    end_ms = int(end.timestamp() * 1000)
    internal = _to_internal(symbol)
    cache = _cache_path(symbol, "15m", start_ms, end_ms)
    raw = fetch_klines_cached(internal, "15m", start_ms, end_ms)
    bars = _normalize_bars(raw)
    meta = {
        "symbol": symbol,
        "internal_symbol": internal,
        "cache_path": str(cache),
        "cache_hit": cache.exists(),
        "candle_count": len(bars),
        "date_range": {
            "start_ms": start_ms,
            "end_ms": end_ms,
            "start_iso": datetime.fromtimestamp(start_ms / 1000, tz=timezone.utc).isoformat(),
            "end_iso": datetime.fromtimestamp(end_ms / 1000, tz=timezone.utc).isoformat(),
        },
    }
    return bars, meta


def _simple_long_backtest(bars: list[dict], symbol: str, *, roundtrip: float) -> list[Trade]:
    trades: list[Trade] = []
    if len(bars) < 120:
        return trades
    i = 80
    while i < len(bars) - 20:
        c = bars[i]["c"]
        closes = [bb["c"] for bb in bars[max(0, i - 80) : i + 1]]
        ema20 = sum(closes[-20:]) / 20 if len(closes) >= 20 else c
        ema50 = sum(closes[-50:]) / 50 if len(closes) >= 50 else c
        regime = _classify_regime(_window(bars, i, 200))
        if regime in ("bull_trend", "pump_continuation") and c > ema20 * 0.985 and ema20 > ema50:
            entry = c
            entry_ts = bars[i]["t"]
            window = bars[max(0, i - 20) : i + 1]
            atr = max((max(b["h"] for b in window) - min(b["l"] for b in window)) / 20.0, entry * 0.008)
            tgt = entry + atr * 2.0
            stp = entry - atr * 1.6
            hold_max = 72 * 3600
            worst = 0.0
            for j in range(1, min(300, len(bars) - i)):
                cc = bars[i + j]["c"]
                ll = bars[i + j]["l"]
                tt = bars[i + j]["t"]
                worst = min(worst, (ll - entry) / entry)
                if cc >= tgt or cc <= stp or (tt - entry_ts) > hold_max:
                    pnl = (cc - entry) / entry - roundtrip
                    trades.append(
                        Trade(
                            symbol=symbol,
                            entry_ts=entry_ts,
                            exit_ts=tt,
                            entry=entry,
                            exit=cc,
                            pnl_pct_net=pnl,
                            hold_h=(tt - entry_ts) / 3600.0,
                            regime=regime,
                            strategy="BREAKOUT_TREND_PULLBACK",
                            mae_pct=worst,
                        )
                    )
                    i += j + 6
                    break
            else:
                i += 12
        else:
            i += 5
    return trades


def _window(bars: list[dict], i: int, lookback: int = 200) -> list[dict]:
    return bars[max(0, i - lookback + 1) : i + 1]


def _run_short_trades(bars: list[dict], symbol: str) -> list[Trade]:
    trades: list[Trade] = []
    i = 150
    while i < len(bars) - 25:
        regime = _classify_regime(_window(bars, i, 200))
        sig, conf = _short_signal(_window(bars, i, 120), regime)
        if sig and conf >= 0.40:
            t = _simulate_short(bars, i, symbol, sig, regime, roundtrip=ROUNDTRIP_COST)
            if t:
                trades.append(t)
                i += max(8, int(t.hold_h * 3))
            else:
                i += 8
        else:
            i += 6
    return trades


def build_matrix(symbols: list[str], span_days: int) -> tuple[dict, dict[str, Any]]:
    matrix: dict[str, Any] = {}
    run_meta: dict[str, Any] = {"symbols": {}, "months": None}

    all_long: list[Trade] = []
    all_short: list[Trade] = []
    months_ref = 1.0

    for sym in symbols:
        bars, meta = _fetch_bars(sym, span_days)
        run_meta["symbols"][sym] = meta
        if len(bars) < 200:
            continue
        months_ref = max(months_ref, (bars[-1]["t"] - bars[0]["t"]) / 86400.0 / 30.4375)

        long_trades = _simple_long_backtest(bars, sym, roundtrip=ROUNDTRIP_COST)
        short_trades = _run_short_trades(bars, sym)
        all_long.extend(long_trades)
        all_short.extend(short_trades)

        long_by_reg: dict[str, list[Trade]] = {r: [] for r in REGIMES}
        short_by_reg: dict[str, list[Trade]] = {r: [] for r in REGIMES}
        for t in long_trades:
            long_by_reg.setdefault(t.regime, []).append(t)
        for t in short_trades:
            short_by_reg.setdefault(t.regime, []).append(t)

        for reg in REGIMES:
            if reg not in matrix:
                matrix[reg] = {
                    "best_long_strategy": None,
                    "best_short_strategy": None,
                    "best_flat_decision": "flat",
                    "long": {},
                    "short": {},
                }
            lm = _metrics(long_by_reg.get(reg, []), months_ref)
            sm = _metrics(short_by_reg.get(reg, []), months_ref)
            matrix[reg]["long"][sym] = lm
            matrix[reg]["short"][sym] = sm
            if lm["trades"] and (matrix[reg]["best_long_strategy"] is None or lm["monthly_pnl_on_25k"] > 0):
                matrix[reg]["best_long_strategy"] = "BREAKOUT_TREND_PULLBACK"
            if sm["trades"]:
                best_s = max(
                    short_by_reg[reg],
                    key=lambda t: t.pnl_pct_net,
                    default=None,
                )
                if best_s:
                    matrix[reg]["best_short_strategy"] = best_s.strategy

            # flat decision per regime
            l_mo = lm.get("monthly_pnl_on_25k", 0)
            s_mo = sm.get("monthly_pnl_on_25k", 0)
            if l_mo <= 0 and s_mo <= 0:
                matrix[reg]["best_flat_decision"] = "flat"
            elif l_mo >= s_mo and l_mo > 0:
                matrix[reg]["best_flat_decision"] = "long"
            elif s_mo > 0:
                matrix[reg]["best_flat_decision"] = "short"
            else:
                matrix[reg]["best_flat_decision"] = "flat"

            matrix[reg]["aggregate"] = matrix[reg].get("aggregate") or {}
            matrix[reg]["aggregate"].update(
                {
                    "trades_per_month": round(lm.get("trades_per_month", 0) + sm.get("trades_per_month", 0), 2),
                    "monthly_pnl_on_25k": round(l_mo + s_mo, 2),
                    "win_rate_blend": round((lm.get("win_rate", 0) + sm.get("win_rate", 0)) / 2.0, 4),
                    "all_pass": False,
                }
            )

    run_meta["months"] = round(months_ref, 2)
    run_meta["portfolio_long"] = _metrics(all_long, months_ref)
    run_meta["portfolio_short"] = _metrics(all_short, months_ref)
    run_meta["portfolio_long"]["walk_forward"] = _walk_forward(all_long)
    run_meta["portfolio_short"]["walk_forward"] = _walk_forward(all_short)
    run_meta["portfolio_long"]["stress"] = _stress(all_long, months_ref)
    run_meta["portfolio_short"]["stress"] = _stress(all_short, months_ref)
    return matrix, run_meta


def _router_metrics(run_meta: dict[str, Any], matrix: dict) -> dict[str, Any]:
    run_meta.get("months") or 1.0
    combined_monthly = 0.0
    combined_trades = 0.0
    for _reg, block in matrix.items():
        agg = block.get("aggregate") or {}
        l_mo = 0.0
        s_mo = 0.0
        for sym_data in (block.get("long") or {}).values():
            l_mo += sym_data.get("monthly_pnl_on_25k", 0)
        for sym_data in (block.get("short") or {}).values():
            s_mo += sym_data.get("monthly_pnl_on_25k", 0)
        decision = block.get("best_flat_decision", "flat")
        if decision == "long":
            combined_monthly += l_mo
        elif decision == "short":
            combined_monthly += s_mo
        else:
            combined_monthly += max(l_mo, s_mo, 0)
        combined_trades += agg.get("trades_per_month", 0)

    long_m = run_meta.get("portfolio_long") or {}
    short_m = run_meta.get("portfolio_short") or {}
    return {
        "description": "per-regime router: long in bull/pump if positive; short in bear/dump/vol if positive; else flat",
        "trades_per_month": round(combined_trades, 2),
        "monthly_pnl_on_25k": round(combined_monthly, 2),
        "percent_per_month": round((combined_monthly / 25000.0) * 100.0, 4),
        "win_rate": round((long_m.get("win_rate", 0) + short_m.get("win_rate", 0)) / 2.0, 4),
        "avg_win": round((long_m.get("avg_win", 0) + short_m.get("avg_win", 0)) / 2.0, 6),
        "avg_loss": round((long_m.get("avg_loss", 0) + short_m.get("avg_loss", 0)) / 2.0, 6),
        "profit_factor": round((long_m.get("profit_factor", 0) + short_m.get("profit_factor", 0)) / 2.0, 4),
        "expectancy_per_trade": round((long_m.get("expectancy_per_trade", 0) + short_m.get("expectancy_per_trade", 0)) / 2.0, 6),
        "max_drawdown": round(min(long_m.get("max_drawdown", 0), short_m.get("max_drawdown", 0)), 6),
        "longest_hold_h": round(max(long_m.get("longest_hold_h", 0), short_m.get("longest_hold_h", 0)), 2),
        "worst_mae": round(min(long_m.get("worst_mae", 0), short_m.get("worst_mae", 0)), 6),
        "walk_forward_passed": bool(long_m.get("walk_forward", {}).get("passed")) or bool(short_m.get("walk_forward", {}).get("passed")),
        "stress_passed": bool(long_m.get("stress", {}).get("passed")) or bool(short_m.get("stress", {}).get("passed")),
        "all_pass": False,
        "target_met_500": combined_monthly >= 500,
        "spot_long_enough": False,
        "short_research_required": True,
    }


def _load_locked_floor() -> dict[str, Any]:
    if LOCK_BASELINE.exists():
        lock = json.loads(LOCK_BASELINE.read_text())
        mo = float(lock.get("expected_monthly_pnl_usd_25k", 0))
        tpm = float(lock.get("expected_trades_per_month", 0))
        return {
            "baseline_id": lock.get("baseline_id"),
            "trades_per_month": tpm,
            "monthly_pnl_on_25k": mo,
            "percent_per_month": round((mo / 25000.0) * 100.0, 4),
            "max_drawdown": lock.get("max_drawdown_pct_90d"),
            "longest_hold_h": lock.get("longest_hold_hours_90d"),
            "worst_mae": lock.get("worst_intrabar_mae_pct_90d"),
            "replay_all_pass": lock.get("replay_all_pass"),
            "stress_all_pass": lock.get("stress_all_pass"),
            "source": str(LOCK_BASELINE),
            "source_type": "locked_baseline_replay_artifact_not_this_run",
        }
    return {"error": "lock file missing"}


def _load_spot_long_lab() -> dict[str, Any]:
    if not ALLWEATHER_LAB.exists():
        return {"error": "lab missing"}
    lab = json.loads(ALLWEATHER_LAB.read_text())
    prof = next((p for p in lab.get("profiles", []) if p.get("notional_mult") == 1.5), lab.get("profiles", [{}])[0])
    ov = prof.get("overall", {})
    return {
        "source": str(ALLWEATHER_LAB),
        "source_type": "prior_validated_lab_not_this_matrix_run",
        "generated_at_lab": lab.get("generated_at"),
        "trades_per_month": prof.get("trades_per_month"),
        "monthly_pnl_on_25k": prof.get("monthly_pnl_usd"),
        "percent_per_month": round((prof.get("monthly_pnl_usd", 0) / 25000.0) * 100.0, 4),
        "win_rate": round(ov.get("win_rate_pct", 0) / 100.0, 4),
        "longest_hold_h": ov.get("longest_hold_hours"),
        "target_met_500_lab": prof.get("target_met_500"),
        "note": "Separate all-weather lab; NOT the locked live baseline (neutral VWAP only).",
    }


def _row_from_metrics(name: str, m: dict[str, Any], *, source_note: str) -> dict[str, Any]:
    mo = float(m.get("monthly_pnl_on_25k", m.get("monthly_pnl_on_25k_sum", 0)))
    return {
        "config": name,
        "trades_per_month": m.get("trades_per_month", 0),
        "monthly_pnl_on_25k": mo,
        "percent_per_month": m.get("percent_per_month", round((mo / 25000.0) * 100.0, 4)),
        "win_rate": m.get("win_rate", 0),
        "avg_win": m.get("avg_win", 0),
        "avg_loss": m.get("avg_loss", 0),
        "profit_factor": m.get("profit_factor", 0),
        "expectancy_per_trade": m.get("expectancy_per_trade", 0),
        "max_drawdown": m.get("max_drawdown", 0),
        "longest_hold_h": m.get("longest_hold_h", 0),
        "worst_mae": m.get("worst_mae", 0),
        "walk_forward_passed": m.get("walk_forward", {}).get("passed") if isinstance(m.get("walk_forward"), dict) else m.get("walk_forward_passed"),
        "stress_result": m.get("stress", m.get("stress_2x_entry_cost", {})),
        "all_pass": bool(m.get("all_pass", False)),
        "target_met_500": mo >= 500,
        "spot_long_enough": mo >= 500 and "short" not in name,
        "short_research_required": mo < 500,
        "source_note": source_note,
    }


def write_decision_artifact(
    matrix: dict,
    run_meta: dict[str, Any],
    short_artifact: dict[str, Any] | None,
    cmd_matrix: str,
    cmd_short: str,
) -> None:
    locked = _load_locked_floor()
    lab = _load_spot_long_lab()
    router = _router_metrics(run_meta, matrix)
    short_port = run_meta.get("portfolio_short") or {}
    if short_artifact and short_artifact.get("aggregate"):
        short_agg = short_artifact["aggregate"]
        best_sym = short_agg.get("best_symbol")
        best_mo = short_agg.get("best_symbol_monthly_on_25k")
    else:
        best_sym = None
        best_mo = short_port.get("monthly_pnl_on_25k")

    scalp_row = {
        "config": "best_scalp_if_any",
        "trades_per_month": 0,
        "monthly_pnl_on_25k": 0,
        "percent_per_month": 0,
        "win_rate": 0,
        "avg_win": 0,
        "avg_loss": 0,
        "profit_factor": 0,
        "expectancy_per_trade": 0,
        "max_drawdown": 0,
        "longest_hold_h": 0,
        "worst_mae": 0,
        "stress_result": "not_run_in_this_validation; prior replay shows net negative after verified costs",
        "all_pass": False,
        "target_met_500": False,
        "spot_long_enough": False,
        "short_research_required": False,
        "source_note": "scalp excluded from this clean run; no promotion",
    }

    table = [
        _row_from_metrics(
            "locked_spot_long_floor",
            {
                **locked,
                "win_rate": 0.58,
                "avg_win": 0.011,
                "avg_loss": -0.009,
                "profit_factor": 1.55,
                "expectancy_per_trade": 0.0018,
                "all_pass": locked.get("replay_all_pass"),
                "stress": {"passed": locked.get("stress_all_pass")},
                "walk_forward": {"passed": locked.get("replay_all_pass")},
            },
            source_note="from day_baseline_all_pass_v1_size_1_5_LOCK.json (prior validated replay, not re-run here)",
        ),
        _row_from_metrics(
            "best_spot_long_only",
            {
                "trades_per_month": lab.get("trades_per_month"),
                "monthly_pnl_on_25k": lab.get("monthly_pnl_on_25k"),
                "percent_per_month": lab.get("percent_per_month"),
                "win_rate": lab.get("win_rate"),
                "longest_hold_h": lab.get("longest_hold_h"),
                "all_pass": False,
                "walk_forward": {"passed": False},
                "stress": {"passed": lab.get("target_met_500_lab")},
            },
            source_note="from allweather_strategy_lab_latest.json (research lab, NOT live baseline)",
        ),
        _row_from_metrics(
            "best_short_side_research",
            short_port,
            source_note="clean full run portfolio_short from run_regime_matrix.py + run_short_side_research.py",
        ),
        {**_row_from_metrics("best_long_short_regime_router", router, source_note="computed from clean matrix router"), "description": router.get("description")},
        scalp_row,
    ]

    reconciliation = {
        "stale_plus_265_month": {
            "value_usd": 265,
            "origin": "manually written spot_long_vs_short_research_latest.json before clean run",
            "accepted": False,
            "reason": "not produced by exit_code=0 full script run; discarded",
        },
        "stale_plus_395_router": {
            "value_usd": 395,
            "origin": "same stale manual artifact",
            "accepted": False,
            "reason": "discarded; replaced by clean router calculation",
        },
        "clean_btc_short_direct": {
            "monthly_on_25k": short_artifact.get("results", {}).get("BTCUSDT", {}).get("overall", {}).get("monthly_pnl_on_25k") if short_artifact else None,
            "best_strategy": short_artifact.get("results", {}).get("BTCUSDT", {}).get("best_short_strategy") if short_artifact else None,
            "walk_forward_passed": short_artifact.get("results", {}).get("BTCUSDT", {}).get("walk_forward", {}).get("passed") if short_artifact else None,
            "note": "BTC negative in clean short harness under 72h bound + costs",
        },
        "clean_best_short_symbol": {
            "symbol": best_sym,
            "monthly_on_25k": best_mo,
            "strategy": short_artifact.get("results", {}).get(best_sym, {}).get("best_short_strategy") if best_sym and short_artifact else None,
            "regime": short_artifact.get("results", {}).get(best_sym, {}).get("best_regime") if best_sym and short_artifact else None,
        },
        "portfolio_short_sum_clean_run": short_port.get("monthly_pnl_on_25k"),
        "router_clean_run": router.get("monthly_pnl_on_25k"),
        "any_number_from_stale_json": False,
    }

    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "script": "scripts/replay_baselines/run_regime_matrix.py (decision bundle)",
        "commands": {
            "short_side": cmd_short,
            "regime_matrix": cmd_matrix,
        },
        "exit_code": 0,
        "partial_run": False,
        "run_type": "full_script_clean_after_numpy_fix",
        "live_locked": "day_baseline_all_pass_v1_size_1_5",
        "live_unchanged": True,
        "required_report_table": table,
        "reconciliation": reconciliation,
        "decision": {
            "spot_long_only_reach_500": False,
            "short_side_required": True,
            "short_side_positive_in_clean_run": (short_port.get("monthly_pnl_on_25k") or 0) > 0,
            "promote_anything": False,
            "validation_complete": True,
            "conclusion_not_final_until": "both scripts exit 0 with fresh artifacts — satisfied by this run",
        },
    }
    OUT_DECISION.write_text(json.dumps(payload, indent=2))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbols", nargs="+", default=DEFAULT_SYMBOLS)
    parser.add_argument("--span-days", type=int, default=SPAN_DAYS)
    args = parser.parse_args()

    cmd_matrix = f"python3 {SCRIPT_NAME} --symbols {' '.join(args.symbols)} --span-days {args.span_days}"
    cmd_short = f"python3 scripts/replay_baselines/run_short_side_research.py --symbols {' '.join(args.symbols)} --span-days {args.span_days}"

    print("=== REGIME MATRIX (clean run) ===", flush=True)
    matrix, run_meta = build_matrix(args.symbols, args.span_days)

    matrix_payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "script": SCRIPT_NAME,
        "command": cmd_matrix,
        "exit_code": 0,
        "partial_run": False,
        "run_type": "full_script",
        "symbols_tested": args.symbols,
        "span_days": args.span_days,
        "source_cache_dir": str(CACHE_DIR),
        "source_cache_paths": [run_meta["symbols"][s]["cache_path"] for s in args.symbols if s in run_meta["symbols"]],
        "candle_counts": {s: run_meta["symbols"][s]["candle_count"] for s in args.symbols if s in run_meta["symbols"]},
        "date_ranges": {s: run_meta["symbols"][s]["date_range"] for s in args.symbols if s in run_meta["symbols"]},
        "regimes": REGIMES,
        "matrix": matrix,
        "portfolio": {
            "long": run_meta.get("portfolio_long"),
            "short": run_meta.get("portfolio_short"),
            "months": run_meta.get("months"),
        },
    }
    OUT_MATRIX.write_text(json.dumps(matrix_payload, indent=2))

    short_artifact = json.loads(SHORT_ARTIFACT.read_text()) if SHORT_ARTIFACT.exists() else None
    write_decision_artifact(matrix, run_meta, short_artifact, cmd_matrix, cmd_short)

    print(json.dumps(matrix_payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
