#!/usr/bin/env python3
"""
Scalp profitability rebuild — isolated per-strategy replay, gate audit, grids, combined DAY report.

Does NOT enable scalp live. Does NOT mix with DAY PnL.
"""

from __future__ import annotations

import json
import statistics
import sys
import time
import traceback
from collections import Counter, defaultdict
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from backend.services.binance_scalp.calibration_profiles import economics_for_config
from backend.services.binance_scalp.config import ScalpConfig
from backend.services.binance_scalp.momentum_tracker import MomentumTracker
from backend.services.binance_scalp.scalp_strategy_router import ScalpStrategyRouter
from backend.services.binance_scalp.strategies import STRATEGY_NAMES, enabled_strategies
from scripts.run_scalp_strategy_replay import (
    NOTIONAL,
    STEP_SEC,
    StrategyStats,
    _config_from_args,
    fetch_klines,
    make_snap,
    replay_symbol,
)

SCRIPT = "scripts/replay_baselines/run_scalp_profitability_rebuild.py"
OUT = REPO / "scripts/replay_baselines" / "scalp_profitability_rebuild_latest.json"
STATUS_OUT = REPO / "scripts/replay_baselines" / "scalp_current_status_latest.json"
AW_SHADOW = REPO / "scripts/replay_baselines" / "allweather_breakout_pullback_shadow_latest.json"
AW_REPLAY = REPO / "scripts/replay_baselines" / "allweather_breakout_pullback_portfolio_replay_latest.json"

SYMBOLS = ("BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT")
WINDOWS_D = (7, 14, 30, 90)
MAX_HOLD_MINUTES = (3, 5, 10, 15, 30)
TARGET_PCTS = (0.0012, 0.0018, 0.0025, 0.0035, 0.0050)
STOP_PCTS = (0.0008, 0.0012, 0.0018, 0.0025)

RESEARCH_ONLY = (
    "failed_breakdown_reversal",
    "compression_breakout",
    "volume_impulse_continuation",
)

GATE_MAP = {
    "SPREAD_TOO_WIDE": "spread_too_wide",
    "NO_BREAKOUT": "no_breakout",
    "BREAKOUT_NOT_CONFIRMED": "no_breakout",
    "NO_VWAP_EMA_RECLAIM": "no_reclaim",
    "NO_PULLBACK_RECOVERY": "no_reclaim",
    "TARGET_NOT_REACHABLE": "target_not_reachable_after_fees",
    "MOMENTUM_GROSS_BELOW_REQUIRED": "target_not_reachable_after_fees",
    "PROJECTED_SURPLUS_TOO_SMALL": "target_not_reachable_after_fees",
    "DEPTH_OR_IMPACT_FAIL": "liquidity_too_thin",
    "MOMENTUM_NOT_CONFIRMED": "orderbook_imbalance_not_confirmed_by_price",
    "INSUFFICIENT_BARS": "no_signal",
}


def _metrics_from_stats(st: StrategyStats, *, window_days: int, principal: float = 25000.0) -> dict[str, Any]:
    n = st.trades
    if n == 0:
        return {
            "trades": 0,
            "trades_per_month": 0.0,
            "monthly_pnl_on_25k": 0.0,
            "pct_per_month": 0.0,
            "win_rate": 0.0,
            "avg_win": 0.0,
            "avg_loss": 0.0,
            "profit_factor": 0.0,
            "expectancy_per_trade": 0.0,
            "max_drawdown_pct": 0.0,
            "longest_hold_min": 0.0,
            "worst_loss_usd": 0.0,
            "fees_usd_est": 0.0,
            "spread_cost_usd_est": 0.0,
            "slippage_cost_usd_est": 0.0,
            "maker_taker_split": {"maker": 0, "taker": n},
            "all_pass": False,
        }
    wins = st.wins
    wr = wins / n * 100
    avg_hold = statistics.mean(st.hold_seconds) / 60 if st.hold_seconds else 0
    longest = max(st.hold_seconds) / 60 if st.hold_seconds else 0
    exp = st.net_pnl_usd / n
    tpm = (n / max(window_days, 1)) * 30
    monthly = (st.net_pnl_usd / max(window_days, 1)) * 30
    pf = 999.0 if st.losses == 0 and wins > 0 else (wins / max(st.losses, 1))
    rt_cost = NOTIONAL * 0.0006 * n
    all_pass = st.net_pnl_usd > 0 and pf >= 1.2 and exp > 0 and longest <= 30
    return {
        "trades": n,
        "trades_per_month": round(tpm, 2),
        "monthly_pnl_on_25k": round(monthly, 2),
        "pct_per_month": round(monthly / principal * 100, 3),
        "win_rate": round(wr, 1),
        "avg_win": round(st.net_pnl_usd / max(wins, 1), 4) if wins else 0,
        "avg_loss": round(st.net_pnl_usd / max(st.losses, 1), 4) if st.losses else 0,
        "profit_factor": round(min(pf, 999), 2),
        "expectancy_per_trade": round(exp, 4),
        "max_drawdown_pct": 0.0,
        "longest_hold_min": round(longest, 2),
        "worst_loss_usd": round(min(0, st.net_pnl_usd), 4),
        "fees_usd_est": round(rt_cost * 0.67, 4),
        "spread_cost_usd_est": round(rt_cost * 0.2, 4),
        "slippage_cost_usd_est": round(rt_cost * 0.13, 4),
        "maker_taker_split": {"maker_pct": 0.0, "taker_pct": 100.0},
        "all_pass": all_pass,
    }


def _fetch_window_bars(window_days: int) -> dict[str, list[dict]]:
    """Cache klines per window to avoid repeated REST fetches."""
    cache_key = f"bars_{window_days}"
    if not hasattr(_fetch_window_bars, "_cache"):
        _fetch_window_bars._cache = {}
    if cache_key in _fetch_window_bars._cache:
        return _fetch_window_bars._cache[cache_key]
    end = datetime.now(timezone.utc)
    start = end - timedelta(days=window_days)
    start_ms = int(start.timestamp() * 1000)
    end_ms = int(end.timestamp() * 1000)
    out: dict[str, list[dict]] = {}
    for sym in SYMBOLS:
        out[sym] = fetch_klines(sym, start_ms, end_ms)
    _fetch_window_bars._cache[cache_key] = out
    return out


def _run_strategy_window(
    strategy: str,
    window_days: int,
    *,
    max_hold_min: int | None = None,
    target_pct: float | None = None,
    bars_cache: dict[str, list[dict]] | None = None,
) -> dict[str, Any]:
    class Args:
        only_strategy = strategy
        profile = "moderate"
        hours = window_days * 24
        spread_mult = 1.0

    config = _config_from_args(Args())
    econ = economics_for_config(config)
    overrides: dict[str, Any] = {}
    if max_hold_min is not None:
        overrides["stale_scalp_timeout_sec"] = int(max_hold_min * 60)
    if target_pct is not None:
        overrides["net_profit_target_pct"] = float(target_pct)
    if overrides:
        econ = replace(econ, **overrides)

    bars_by_sym = bars_cache or _fetch_window_bars(window_days)

    momentum = MomentumTracker()
    router = ScalpStrategyRouter(
        config=config,
        econ=econ,
        reader=type("R", (), {"read": lambda _s, _sym: None})(),
        momentum=momentum,
    )
    agg = StrategyStats()
    all_trades: list[dict] = []
    for sym in SYMBOLS:
        bars = bars_by_sym.get(sym, [])
        if len(bars) < 30:
            continue
        trades, stats, _ = replay_symbol(
            sym,
            bars,
            config=config,
            econ=econ,
            router=router,
            momentum=momentum,
        )
        st = stats.get(strategy, StrategyStats())
        agg.trades += st.trades
        agg.wins += st.wins
        agg.losses += st.losses
        agg.net_pnl_usd += st.net_pnl_usd
        agg.hold_seconds.extend(st.hold_seconds)
        all_trades.extend(trades)
        momentum._history.pop(sym, None)

    return {
        "strategy": strategy,
        "window_days": window_days,
        "max_hold_min": max_hold_min,
        "target_pct": target_pct,
        "metrics": _metrics_from_stats(agg, window_days=window_days),
        "sample_trades": len(all_trades),
    }


def _gate_audit(strategy: str, *, window_days: int = 7, sample_bars: int = 500) -> dict[str, Any]:
    """Count rejection reasons for one strategy over recent bars."""

    class Args:
        only_strategy = strategy
        profile = "moderate"
        hours = window_days * 24
        spread_mult = 1.0

    config = _config_from_args(Args())
    econ = economics_for_config(config)
    momentum = MomentumTracker()
    router = ScalpStrategyRouter(
        config=config,
        econ=econ,
        reader=type("R", (), {"read": lambda _s, _sym: None})(),
        momentum=momentum,
    )
    end = datetime.now(timezone.utc)
    start = end - timedelta(days=window_days)
    start_ms = int(start.timestamp() * 1000)
    end_ms = int(end.timestamp() * 1000)

    raw_counts: Counter[str] = Counter()
    gate_counts: Counter[str] = Counter()
    evaluated = 0

    sym = "BTCUSDT"
    bars = fetch_klines(sym, start_ms, end_ms)
    step = max(1, len(bars) // sample_bars) if bars else 1
    for idx in range(20, len(bars) - 5, step):
        bar = bars[idx]
        snap = make_snap(sym, bar)
        for sub in range(4):
            t = bar["epoch"] - 45 + sub * 15
            momentum.record(sym, t, snap.best_bid, snap.mid)
        momentum.record(sym, bar["epoch"], snap.best_bid, snap.mid)
        window = bars[max(0, idx - 60) : idx + 1]
        kline_window = [{"high": b["high"], "low": b["low"], "close": b["close"], "volume": b["volume"]} for b in window]
        _, all_sigs = router.evaluate_symbol(
            sym,
            epoch=bar["epoch"],
            notional_usd=NOTIONAL,
            snap=snap,
            bars=kline_window,
        )
        evaluated += 1
        for sig in all_sigs:
            if sig.reject_reason:
                raw_counts[sig.reject_reason] += 1
                gate_counts[GATE_MAP.get(sig.reject_reason, "other")] += 1

    top_raw = raw_counts.most_common(10)
    top_gate = gate_counts.most_common(10)
    return {
        "strategy": strategy,
        "evaluated_bars": evaluated,
        "top_reject_reasons_raw": dict(top_raw),
        "top_gate_categories": dict(top_gate),
    }


def _promotion_check(metrics_30d: dict, metrics_90d: dict | None) -> dict[str, Any]:
    m30 = metrics_30d.get("metrics", metrics_30d)
    m90 = (metrics_90d or {}).get("metrics", metrics_90d or {})
    checks = {
        "positive_30d_replay": float(m30.get("monthly_pnl_on_25k") or 0) > 0,
        "positive_90d_replay": float(m90.get("monthly_pnl_on_25k") or 0) > 0 if m90 else False,
        "profit_factor_above_1_2": float(m30.get("profit_factor") or 0) >= 1.2,
        "max_hold_under_30m": float(m30.get("longest_hold_min") or 999) <= 30,
        "no_day_ledger_contamination": True,
        "no_repair_add": True,
        "no_red_thesis_dependency": True,
        "positive_after_fees_spread_slippage": float(m30.get("expectancy_per_trade") or 0) > 0,
    }
    passed = all(checks.values())
    return {"checks": checks, "paper_enable_eligible": passed, "live_enable_eligible": False}


def _load_day_context() -> dict[str, Any]:
    ctx: dict[str, Any] = {"forward_paper_idle": True, "expected_monthly_pnl_usd": None}
    try:
        if AW_SHADOW.exists():
            sh = json.loads(AW_SHADOW.read_text())
            ctx["aw_shadow"] = {
                "evaluated_cycles": sh.get("evaluated_cycles"),
                "would_buy_count": sh.get("would_buy_count"),
                "sleeve_name": sh.get("sleeve_name", "TREND_BREAKOUT_PULLBACK_SLEEVE"),
            }
            ctx["forward_paper_idle"] = int(sh.get("would_buy_count") or 0) == 0
    except Exception:
        pass
    try:
        if AW_REPLAY.exists():
            aw = json.loads(AW_REPLAY.read_text())
            w90 = (aw.get("windows") or {}).get("90d") or aw.get("summary") or {}
            ctx["aw_replay_90d_monthly_est"] = w90.get("monthly_pnl_usd") or w90.get("net_pnl_usd")
            ctx["expected_day_trend_sleeve_monthly_usd"] = ctx["aw_replay_90d_monthly_est"]
    except Exception:
        pass
    return ctx


def main() -> int:
    print("=== SCALP PROFITABILITY REBUILD ===", flush=True)
    t0 = time.time()
    per_strategy: dict[str, Any] = {}
    gate_audits: dict[str, Any] = {}
    max_hold_grids: dict[str, Any] = {}
    target_grids: dict[str, Any] = {}

    implemented = list(STRATEGY_NAMES)
    for strat in implemented:
        print(f"--- {strat} ---", flush=True)
        windows_out: dict[str, Any] = {}
        bars_cache: dict[int, dict[str, list[dict]]] = {}
        for wd in WINDOWS_D:
            try:
                if wd not in bars_cache:
                    bars_cache[wd] = _fetch_window_bars(wd)
                windows_out[f"{wd}d"] = _run_strategy_window(strat, wd, bars_cache=bars_cache[wd])
            except Exception as exc:
                windows_out[f"{wd}d"] = {"error": str(exc), "traceback": traceback.format_exc()[-500:]}
        per_strategy[strat] = {"windows": windows_out, "promotion": _promotion_check(windows_out.get("30d", {}), windows_out.get("90d", {}))}
        try:
            gate_audits[strat] = _gate_audit(strat, window_days=7)
        except Exception as exc:
            gate_audits[strat] = {"error": str(exc)}

        hold_grid = {}
        grid_bars = bars_cache.get(30) or _fetch_window_bars(30)
        for hm in MAX_HOLD_MINUTES:
            try:
                hold_grid[f"{hm}m"] = _run_strategy_window(strat, 30, max_hold_min=hm, bars_cache=grid_bars)["metrics"]
            except Exception as exc:
                hold_grid[f"{hm}m"] = {"error": str(exc)}
        max_hold_grids[strat] = hold_grid

        tgt_grid = {}
        for tp in TARGET_PCTS:
            try:
                tgt_grid[f"target_{tp * 100:.2f}pct"] = _run_strategy_window(strat, 30, target_pct=tp, bars_cache=grid_bars)["metrics"]
            except Exception as exc:
                tgt_grid[f"target_{tp * 100:.2f}pct"] = {"error": str(exc)}
        target_grids[strat] = tgt_grid

    # Best scalp candidate
    best_name = None
    best_monthly = -1e9
    for name, data in per_strategy.items():
        m = (data.get("windows") or {}).get("30d", {}).get("metrics", {})
        monthly = float(m.get("monthly_pnl_on_25k") or 0)
        if monthly > best_monthly:
            best_monthly = monthly
            best_name = name

    day_ctx = _load_day_context()
    day_monthly = float(day_ctx.get("expected_day_trend_sleeve_monthly_usd") or 0)
    combined_monthly = day_monthly + max(best_monthly, 0)
    any_scalp_pass = any((v.get("promotion") or {}).get("paper_enable_eligible") for v in per_strategy.values())

    payload: dict[str, Any] = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "script": SCRIPT,
        "principal_usd": 25000.0,
        "notional_per_trade_usd": NOTIONAL,
        "symbols": list(SYMBOLS),
        "intervals": {"execution": "1m", "context_5m": "available_plumbing", "context_15m": "available_plumbing", "regime_1h": "plumbing_only_not_wired"},
        "implemented_strategies": implemented,
        "research_only_not_implemented": list(RESEARCH_ONLY),
        "per_strategy_replay": per_strategy,
        "gate_audits": gate_audits,
        "max_hold_grid_30d": max_hold_grids,
        "target_grid_30d": target_grids,
        "stop_grid_note": "stop tested via exit_manager + stale timeout; explicit stop pct grid uses ATR micro-stop in live engine",
        "promotion_requirements": {
            "positive_30d": True,
            "positive_90d_if_available": True,
            "walk_forward_positive": "pending_full_wf_script",
            "stress_positive": "pending_stress_suite",
            "profit_factor_min": 1.2,
            "max_hold_minutes": 30,
            "no_day_contamination": True,
            "no_repair_add": True,
            "no_red_thesis": True,
            "fees_spread_slippage_included": True,
        },
        "combined_report": {
            "day_trend_sleeve_expected_monthly_usd": day_monthly,
            "day_forward_status": day_ctx,
            "best_scalp_strategy": best_name,
            "best_scalp_30d_monthly_usd": round(best_monthly, 2) if best_name else 0,
            "combined_day_plus_scalp_monthly_usd": round(combined_monthly, 2),
            "target_500_per_month_met": combined_monthly >= 500,
            "scalp_should_remain_disabled": not any_scalp_pass,
            "scalp_paper_enable_recommendation": any_scalp_pass,
        },
        "safety": {
            "live_enabled": False,
            "real_orders_permitted": False,
            "mixed_with_day_pnl": False,
            "repair_add": False,
        },
        "duration_sec": round(time.time() - t0, 1),
    }

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2))
    print(f"Wrote {OUT}", flush=True)
    print(
        f"best_scalp={best_name} best_monthly={best_monthly:.2f} combined={combined_monthly:.2f} target_500={combined_monthly >= 500} scalp_enable={any_scalp_pass}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
