#!/usr/bin/env python3
"""
Multi-year scalp regime validation — regime-separated, not blended.

Recent 30d failure is diagnostic only. This script tests whether any scalp
strategy is positive in its native regime across available history.

Paper/research only. No live enable. No DAY mixing.
"""

from __future__ import annotations

import json
import sys
import time
import traceback
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from backend.services.binance_scalp.scalp_regime_classifier import (
    ALL_SCALP_REGIMES,
    STRATEGY_NATIVE_REGIMES,
    build_regime_index,
    router_decision,
    summarize_regime_coverage,
)
from backend.services.binance_scalp.strategies import STRATEGY_NAMES
from scripts.replay_baselines.scalp_multiyear_lib import (
    TOP4,
    build_data_coverage,
    compute_trade_metrics,
    filter_trades_by_window,
    load_or_fetch_bars,
    promotion_checks,
    run_regime_filtered_replay,
    walk_forward_splits,
)

SCRIPT = "scripts/replay_baselines/run_scalp_multiyear_regime_validation.py"
OUT = REPO / "scripts/replay_baselines/scalp_multiyear_regime_validation_latest.json"
AW_REPLAY = REPO / "scripts/replay_baselines/allweather_breakout_pullback_portfolio_replay_latest.json"
AW_SHADOW = REPO / "scripts/replay_baselines/allweather_breakout_pullback_shadow_latest.json"
RECENT_30D = REPO / "scripts/replay_baselines/scalp_profitability_rebuild_latest.json"

IMPLEMENTED = list(STRATEGY_NAMES)
RESEARCH_ONLY = [
    "compression_breakout",
    "volume_impulse_continuation",
    "trend_pullback_micro",
    "failed_breakdown_reversal",
    "failed_breakout_reversal",
]

MAX_HOLD_MIN = (3, 5, 10, 15, 30)
TARGET_PCTS = (0.0012, 0.0018, 0.0025, 0.0035, 0.0050)


def _load_day_context() -> dict[str, Any]:
    ctx: dict[str, Any] = {}
    try:
        if AW_REPLAY.exists():
            aw = json.loads(AW_REPLAY.read_text())
            full = (aw.get("exact_candidate_mode") or {}).get("full_span_metrics") or {}
            ctx["day_trend_sleeve_replay_full_span_monthly_usd"] = full.get("monthly_pnl_usd_on_25k") or full.get("monthly_pnl_usd")
            walk = aw.get("walk_forward") or {}
            ctx["day_trend_sleeve_walk_forward"] = walk.get("summary") or walk.get("all_pass")
    except Exception:
        pass
    try:
        if AW_SHADOW.exists():
            sh = json.loads(AW_SHADOW.read_text())
            ctx["day_forward_idle"] = int(sh.get("would_buy_count") or 0) == 0
            ctx["day_forward_evals"] = sh.get("evaluated_cycles")
    except Exception:
        pass
    try:
        if RECENT_30D.exists():
            r = json.loads(RECENT_30D.read_text())
            ctx["recent_30d_scalp_diagnostic"] = {
                "note": "recent down/range window only — not permanent scalp verdict",
                "scalp_should_remain_disabled": r.get("combined_report", {}).get("scalp_should_remain_disabled"),
            }
    except Exception:
        pass
    return ctx


def main() -> int:
    print("=== SCALP MULTI-YEAR REGIME VALIDATION ===", flush=True)
    t0 = time.time()

    data_coverage = build_data_coverage(target_days_1m=180, target_days_1h=1100)
    regime_coverage: dict[str, Any] = {}
    regime_indices: dict[str, dict[int, str]] = {}
    bars_1m_by_sym: dict[str, list[dict]] = {}

    end = datetime.now(timezone.utc)
    end_ms = int(end.timestamp() * 1000)

    for sym in TOP4:
        meta_1h = data_coverage["symbols"][sym].get("1h", {})
        start_ms = meta_1h.get("start_ms") or int((end.timestamp() - 1100 * 86400) * 1000)
        end_ms_sym = meta_1h.get("end_ms") or end_ms
        bars_1h, _ = load_or_fetch_bars(sym, "1h", start_ms, end_ms_sym)
        regime_coverage[sym] = summarize_regime_coverage(bars_1h)
        regime_indices[sym] = build_regime_index(bars_1h)

        meta_1m = data_coverage["symbols"][sym].get("1m", {})
        s1 = meta_1m.get("start_ms") or start_ms
        e1 = meta_1m.get("end_ms") or end_ms
        bars_1m, _ = load_or_fetch_bars(sym, "1m", s1, e1)
        bars_1m_by_sym[sym] = bars_1m

    # Strategy x regime matrix
    matrix: dict[str, dict[str, Any]] = {}
    all_trades_by_strategy: dict[str, list] = defaultdict(list)
    wf_results: dict[str, Any] = {}
    grid_best: dict[str, Any] = {}

    for strat in IMPLEMENTED:
        print(f"--- replay {strat} ---", flush=True)
        strat_trades = []
        per_regime: dict[str, list] = defaultdict(list)
        per_symbol: dict[str, list] = defaultdict(list)

        for sym in TOP4:
            bars = bars_1m_by_sym.get(sym, [])
            if len(bars) < 100:
                continue
            try:
                trades = run_regime_filtered_replay(
                    sym,
                    bars,
                    regime_indices[sym],
                    strat,
                    allowed_regimes=STRATEGY_NATIVE_REGIMES.get(strat),
                )
            except Exception as exc:
                print(f"  {sym} error: {exc}", flush=True)
                continue
            strat_trades.extend(trades)
            per_symbol[sym].extend(trades)
            for t in trades:
                per_regime[t.regime].append(t)

        all_trades_by_strategy[strat] = strat_trades
        window_days = 0.0
        if strat_trades:
            window_days = max(1.0, (max(t.entry_epoch for t in strat_trades) - min(t.entry_epoch for t in strat_trades)) / 86400)

        regime_metrics = {reg: compute_trade_metrics(per_regime.get(reg, []), window_days=max(window_days, 30)) for reg in ALL_SCALP_REGIMES}
        symbol_metrics = {sym: compute_trade_metrics(per_symbol[sym], window_days=max(window_days, 30)) for sym in TOP4}

        overall = compute_trade_metrics(strat_trades, window_days=max(window_days, 30))
        matrix[strat] = {
            "native_regimes": sorted(STRATEGY_NATIVE_REGIMES.get(strat, frozenset())),
            "overall_in_native_regimes": overall,
            "by_regime": regime_metrics,
            "by_symbol": symbol_metrics,
            "best_regime": max(
                ((r, m) for r, m in regime_metrics.items() if m.get("trades", 0) > 0),
                key=lambda x: x[1].get("monthly_pnl_on_25k", -1e9),
                default=(None, {}),
            )[0],
            "best_symbol": max(
                ((s, m) for s, m in symbol_metrics.items() if m.get("trades", 0) > 0),
                key=lambda x: x[1].get("monthly_pnl_on_25k", -1e9),
                default=(None, {}),
            )[0],
        }

        if strat_trades:
            epochs = [int(t.entry_epoch) for t in strat_trades]
            splits = walk_forward_splits(min(epochs), max(epochs))
            wf = {}
            for name, window in splits.items():
                sub = filter_trades_by_window(strat_trades, window)
                wd = max(1.0, (window[1] - window[0]) / 86400)
                wf[name] = compute_trade_metrics(sub, window_days=wd)
            wf_results[strat] = {
                "splits": dict(splits.items()),
                "metrics": wf,
                "promotion": promotion_checks(wf.get("train", {}), wf.get("validation", {}), wf.get("test", {})),
            }

        # Small grid on 30m max hold + default target for best-regime candidate
        hold_grid = {}
        for hm in MAX_HOLD_MIN:
            try:
                sym0 = matrix[strat].get("best_symbol") or "BTC/USDT"
                bars = bars_1m_by_sym.get(sym0, [])
                if not bars:
                    continue
                tr = run_regime_filtered_replay(
                    sym0,
                    bars,
                    regime_indices[sym0],
                    strat,
                    max_hold_sec=hm * 60,
                )
                wd = max(30.0, (bars[-1]["ts"] - bars[0]["ts"]) / 86400)
                hold_grid[f"{hm}m"] = compute_trade_metrics(tr, window_days=wd)
            except Exception:
                hold_grid[f"{hm}m"] = {"error": True}
        target_grid = {}
        for tp in TARGET_PCTS:
            try:
                sym0 = matrix[strat].get("best_symbol") or "BTC/USDT"
                bars = bars_1m_by_sym.get(sym0, [])
                if not bars:
                    continue
                tr = run_regime_filtered_replay(
                    sym0,
                    bars,
                    regime_indices[sym0],
                    strat,
                    max_hold_sec=30 * 60,
                    target_pct=tp,
                )
                wd = max(30.0, (bars[-1]["ts"] - bars[0]["ts"]) / 86400)
                target_grid[f"{tp * 100:.2f}%"] = compute_trade_metrics(tr, window_days=wd)
            except Exception:
                target_grid[f"{tp * 100:.2f}%"] = {"error": True}
        grid_best[strat] = {"max_hold_grid": hold_grid, "target_grid_30m_hold": target_grid}

    # Best strategy per regime (any strategy with native match)
    best_per_regime: dict[str, Any] = {}
    for reg in ALL_SCALP_REGIMES:
        best_s = None
        best_m = -1e9
        for strat in IMPLEMENTED:
            m = matrix.get(strat, {}).get("by_regime", {}).get(reg, {})
            monthly = float(m.get("monthly_pnl_on_25k") or 0)
            if reg in STRATEGY_NATIVE_REGIMES.get(strat, frozenset()) and monthly > best_m:
                best_m = monthly
                best_s = strat
        best_per_regime[reg] = {
            "best_strategy": best_s,
            "monthly_pnl_on_25k": round(best_m, 2) if best_m > -1e8 else 0,
            "positive_in_native_regime": best_m > 0,
        }

    # Router snapshot (use BTC current regime from last 1h bar)
    current_regime = "range"
    try:
        btc_idx = regime_indices.get("BTC/USDT", {})
        if btc_idx:
            current_regime = list(btc_idx.values())[-1]
    except Exception:
        pass

    router_rows = []
    for strat in IMPLEMENTED:
        reg_m = matrix.get(strat, {}).get("by_regime", {}).get(current_regime, {})
        exp = float(reg_m.get("expectancy_per_trade") or 0)
        tpm = float(reg_m.get("trades_per_month") or 0)
        conf = "high" if reg_m.get("all_pass") else ("medium" if exp > 0 else "low")
        router_rows.append(
            router_decision(
                current_regime=current_regime,
                strategy=strat,
                expectancy=exp,
                trades_per_month=tpm,
                confidence=conf,
            )
        )

    any_promotion = any((wf_results.get(s, {}).get("promotion") or {}).get("promotion_ready") for s in IMPLEMENTED)
    best_strat = None
    best_monthly = -1e9
    for strat in IMPLEMENTED:
        m = float(matrix.get(strat, {}).get("overall_in_native_regimes", {}).get("monthly_pnl_on_25k") or 0)
        if m > best_monthly:
            best_monthly = m
            best_strat = strat

    day_ctx = _load_day_context()
    day_monthly = float(day_ctx.get("day_trend_sleeve_replay_full_span_monthly_usd") or day_ctx.get("day_trend_sleeve_replay_90d_monthly_usd") or 0)
    combined = day_monthly + max(best_monthly, 0)

    for rs in RESEARCH_ONLY:
        matrix[rs] = {
            "native_regimes": sorted(STRATEGY_NATIVE_REGIMES.get(rs, frozenset())),
            "status": "research_only_not_in_runner",
            "overall_in_native_regimes": compute_trade_metrics([], window_days=30),
            "note": "requires runner port before regime replay",
        }

    payload: dict[str, Any] = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "script": SCRIPT,
        "methodology": {
            "regime_separated": True,
            "recent_30d_not_final_verdict": True,
            "native_regime_only": True,
            "costs": "Binance.US taker + half-spread + slippage + depth walk (replay)",
            "walk_forward": "60/20/20 train/val/test on 1m replay window",
        },
        "data_coverage": data_coverage,
        "regime_coverage": regime_coverage,
        "research_only_not_in_runner": RESEARCH_ONLY,
        "implemented_strategies": IMPLEMENTED,
        "strategy_regime_matrix": matrix,
        "best_strategy_per_regime": best_per_regime,
        "walk_forward": wf_results,
        "max_hold_grids": grid_best,
        "target_grid_note": "full target/stop grid in scalp_profitability_rebuild; rerun per winning regime candidate",
        "scalp_regime_router": {
            "current_regime": current_regime,
            "strategies": router_rows,
            "allowed_now": [r["strategy"] for r in router_rows if r.get("allowed")],
            "blocked_now": [r["strategy"] for r in router_rows if r.get("blocked")],
            "flat_valid": len([r for r in router_rows if r.get("allowed")]) == 0,
        },
        "promotion_summary": {
            "any_strategy_promotion_ready": any_promotion,
            "promotion_ready": any_promotion,
            "all_pass": any_promotion,
            "live_enabled": False,
            "real_orders_permitted": False,
            "scalp_should_remain_disabled": not any_promotion,
            "reason": (
                "at least one strategy positive in native regime with walk-forward pass" if any_promotion else "no strategy passed native-regime walk-forward validation on available 1m history"
            ),
        },
        "combined_report": {
            "day_trend_sleeve_expected_pnl_replay_full_span_usd": day_monthly,
            "day_forward_status": day_ctx,
            "best_scalp_regime_router_strategy": best_strat,
            "best_scalp_native_regime_monthly_usd": round(best_monthly, 2),
            "combined_day_plus_scalp_monthly_usd": round(combined, 2),
            "target_500_per_month_met": combined >= 500,
            "scalp_should_remain_disabled": not any_promotion,
            "recent_30d_diagnostic_only": True,
        },
        "safety": {
            "live_enabled": False,
            "real_orders_permitted": False,
            "mixed_with_day_pnl": False,
            "repair_add": False,
            "no_gate_loosening": True,
        },
        "duration_sec": round(time.time() - t0, 1),
    }

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2, default=str))
    print(f"Wrote {OUT}", flush=True)
    print(
        f"promotion_ready={any_promotion} best_scalp={best_strat} best_monthly={best_monthly:.2f} combined={combined:.2f}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
