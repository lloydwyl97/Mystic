#!/usr/bin/env python3
"""
Frequency-first strategy rebuild — EXHAUSTED (diagnostic only).

Set MYSTIC_FORCE_EXHAUSTED_RESEARCH=1 to re-run.
Active path: scripts/run_topfour_profit_rebuild.py
"""
from __future__ import annotations

import json
import os
import statistics
import subprocess
import sys
import traceback
from collections import Counter
from copy import deepcopy
from dataclasses import replace
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
BASELINE_DIR = REPO / "scripts" / "replay_baselines"

os.environ.setdefault("SCALP_LIVE", "false")
os.environ.setdefault("SCALP_CALIBRATION_MODE", "true")
os.environ.setdefault("SCALP_PAPER_ENABLED", "true")
os.environ.setdefault("SCALP_FEE_MODEL_VERIFIED", "true")
os.environ.setdefault("SCALP_PRODUCTS", "BTCUSDT,ETHUSDT,SOLUSDT,XRPUSDT")

from backend.services.binance_scalp.exit_manager import DECISION_SELL, PositionTrack
from backend.services.binance_scalp.calibration_profiles import economics_for_config
from backend.services.binance_scalp.config import ScalpConfig
from backend.services.binance_scalp.momentum_tracker import MomentumTracker
from backend.services.binance_scalp.scalp_strategy_router import ScalpStrategyRouter
from backend.services.binance_scalp.strategies import STRATEGY_NAMES
from backend.services.day_regime_router import DAY_REGIME_BULL, DAY_REGIME_BEAR
from backend.services.day_trade_thesis import (
    SETUP_BREAKOUT_CONTINUATION,
    SETUP_HTF_TREND_PULLBACK,
    SETUP_VWAP_REVERSION,
)
from scripts.run_day_execution_replay import (
    ALLOWED_POSITIVE_BUCKETS,
    ExecutionConfig,
    _build_fee_profiles,
    _infer_context_regime_label,
    _run_stress_90d,
    _run_suite,
    fetch_klines_cached,
)
from scripts.run_day_profit_growth_search import (
    MAX_DD_CAP,
    MAX_HOLD_HOURS,
    TARGET_MONTHLY,
    _apply_variant,
    _base_cfg,
    _evaluate,
    _run_stress_battery,
)
from scripts.run_day_strategy_replay import PRINCIPAL, SYMBOLS
from scripts.run_mystic_frequency_diagnostic import (
    SCALP_HOURS,
    SCALP_STRATEGIES,
    _monthly_from_scalp_hours,
    _regime_taxonomy_audit,
    _replay_scalp_strategy,
    _scalp_profile_relaxation,
)
from scripts.run_scalp_strategy_replay import (
    LOOKAHEAD_BARS,
    NOTIONAL,
    OpenTrade,
    StrategyStats,
    evaluate_exit,
    fetch_klines,
    make_snap,
    net_pct_at_bid,
    spread_est,
)

TARGET_500 = TARGET_MONTHLY["2pct"]
LIVE_BASELINE_ID = "day_baseline_all_pass_v1_size_1_5"


def _profit_factor(wins: int, losses: int, avg_win: float, avg_loss: float) -> float:
    gw = avg_win * wins
    gl = abs(avg_loss) * losses
    if gl < 1e-9:
        return 999.0 if gw > 0 else 0.0
    return round(gw / gl, 3)


def _replay_scalp_isolated(
    strategy: str,
    *,
    profile: str = "moderate",
    hours: int = SCALP_HOURS,
    spread_mult: float = 1.0,
) -> dict[str, Any]:
    """Single-strategy replay with explicit disabled list (no env leakage)."""
    base = ScalpConfig.from_env()
    config = replace(
        base,
        disabled_strategies=frozenset(s for s in STRATEGY_NAMES if s != strategy),
        calibration_profile=profile,
        calibration_mode=True,
        scalp_live=False,
        scalp_paper_enabled=True,
    )
    econ = economics_for_config(config)
    momentum = MomentumTracker()
    router = ScalpStrategyRouter(
        config=config,
        econ=econ,
        reader=type("R", (), {"read": lambda _s, sym: None})(),
        momentum=momentum,
    )
    end = datetime.now(timezone.utc)
    start = end - timedelta(hours=hours + 1)
    start_ms = int(start.timestamp() * 1000)
    end_ms = int(end.timestamp() * 1000)

    all_trades: list[dict] = []
    reject_counts: Counter = Counter()
    stats = StrategyStats()
    missed = 0

    for sym in config.products:
        bars = fetch_klines(sym, start_ms, end_ms)
        momentum._history.pop(sym, None)
        open_pos: OpenTrade | None = None
        cooldown_until = 0.0

        for idx in range(20, len(bars) - LOOKAHEAD_BARS):
            bar = bars[idx]
            epoch = bar["epoch"]
            snap = make_snap(sym, bar, spread_mult=spread_mult)
            for sub in range(4):
                t = epoch - 45 + sub * 15
                price = bar["open"] + (bar["close"] - bar["open"]) * (sub + 1) / 4
                sp = spread_est(bar) * spread_mult
                bid = price * (1.0 - sp / 2)
                momentum.record(sym, t, bid, price)
            momentum.record(sym, epoch, snap.best_bid, snap.mid)
            window = bars[max(0, idx - 60) : idx + 1]
            kline_window = [
                {"high": b["high"], "low": b["low"], "close": b["close"], "volume": b["volume"]}
                for b in window
            ]

            if open_pos is not None:
                hold = epoch - open_pos.entry_epoch
                mom = momentum.diagnostics(sym, epoch, snap.best_bid, snap.mid)
                net_pct = net_pct_at_bid(
                    open_pos.entry_price, snap.best_bid, snap.spread_pct, open_pos.impact_pct, econ,
                )
                profit_hit = net_pct >= econ.net_profit_target_pct
                review = evaluate_exit(
                    track=open_pos.track,
                    snap=snap,
                    mom=mom,
                    econ=econ,
                    config=config,
                    trade_id="replay",
                    hold_sec=hold,
                    executable_net_pct=net_pct,
                    profit_hit=profit_hit,
                    exit_spread_ok=True,
                    perform_review=hold >= econ.stale_scalp_timeout_sec,
                )
                open_pos.track = review.updated_track

                if review.decision == DECISION_SELL and review.exit_reason:
                    qty = NOTIONAL / open_pos.entry_price
                    pnl = net_pct * open_pos.entry_price * qty
                    win = pnl > 0
                    stats.trades += 1
                    stats.net_pnl_usd += pnl
                    stats.hold_seconds.append(hold)
                    if win:
                        stats.wins += 1
                    else:
                        stats.losses += 1
                    all_trades.append({"pnl_usd": pnl, "hold_sec": hold, "win": win})
                    open_pos = None
                    cooldown_until = epoch + 120
                continue

            if epoch < cooldown_until:
                continue

            best, all_sigs = router.evaluate_symbol(
                sym, epoch=epoch, notional_usd=NOTIONAL, snap=snap, bars=kline_window,
            )
            for sig in all_sigs:
                if sig.setup_name == strategy and not sig.passed and sig.reject_reason:
                    reject_counts[sig.reject_reason] += 1

            if best is None or not best.passed or best.setup_name != strategy:
                chunk = bars[idx : idx + LOOKAHEAD_BARS]
                max_high = max(b["high"] for b in chunk)
                potential = (max_high - snap.best_ask) / snap.best_ask - econ.roundtrip_cost_pct(snap.spread_pct, 0, 0)
                if potential >= econ.net_profit_target_pct:
                    missed += 1
                continue

            track = PositionTrack(
                entry_price=best.limit_buy_price,
                state="OPEN",
                max_favorable_pct=0.0,
                max_adverse_pct=0.0,
                session_low_bid=best.limit_buy_price,
                stale_review_count=0,
                review_lows=(),
                setup_name=best.setup_name,
                setup_context=dict(best.setup_context),
            )
            open_pos = OpenTrade(
                symbol=sym,
                setup_name=best.setup_name,
                entry_price=best.limit_buy_price,
                entry_epoch=epoch,
                impact_pct=best.impact_pct,
                setup_context=dict(best.setup_context),
                track=track,
            )

    net = round(stats.net_pnl_usd, 4)
    n = stats.trades
    wins = stats.wins
    losses = stats.losses
    wr = round(100.0 * wins / max(1, wins + losses), 2)
    exp = round(net / max(1, n), 4)
    ext = _monthly_from_scalp_hours(net, hours, n)
    pnls = [t["pnl_usd"] for t in all_trades]
    wins_pnl = [p for p in pnls if p > 0]
    loss_pnl = [p for p in pnls if p <= 0]
    avg_win = round(statistics.mean(wins_pnl), 4) if wins_pnl else 0.0
    avg_loss = round(statistics.mean(loss_pnl), 4) if loss_pnl else 0.0
    pf = _profit_factor(wins, losses, avg_win, avg_loss)
    max_hold = round(max(stats.hold_seconds), 1) if stats.hold_seconds else 0.0
    eq = peak = max_dd = 0.0
    for p in pnls:
        eq += p
        peak = max(peak, eq)
        max_dd = max(max_dd, peak - eq)
    top_blockers = reject_counts.most_common(8)
    relax = _scalp_profile_relaxation(strategy)

    spread_rows = []
    if spread_mult == 1.0:
        for sm in (1.0, 1.5, 2.0):
            if sm == 1.0:
                spread_rows.append({"spread_mult": sm, "net_pnl_usd": net, "trades": n})
            else:
                sr = _replay_scalp_isolated(strategy, profile=profile, hours=hours, spread_mult=sm)
                spread_rows.append({
                    "spread_mult": sm,
                    "net_pnl_usd": sr["net_pnl_usd"],
                    "trades": sr["total_trades"],
                })

    loss_profile = next((x for x in relax.get("profiles", []) if x["profile"] == "fast"), {})
    loss_gate = "negative_expectancy_when_gates_loosened" if relax.get("relaxation_creates_losses") else "n/a"

    return {
        "strategy": strategy,
        "profile": profile,
        "spread_mult": spread_mult,
        "replay_hours": hours,
        "enabled_only": strategy,
        "total_trades": n,
        "win_rate_pct": wr,
        "net_pnl_usd": net,
        "expectancy_per_trade_usd": exp,
        "avg_win_usd": avg_win,
        "avg_loss_usd": avg_loss,
        "profit_factor": pf,
        "avg_hold_sec": round(statistics.mean(stats.hold_seconds), 1) if stats.hold_seconds else 0.0,
        "max_hold_sec": max_hold,
        "max_drawdown_usd": round(max_dd, 4),
        "spread_sensitivity": spread_rows,
        "missed_profitable_windows": missed,
        "dominant_entry_gate": top_blockers[0][0] if top_blockers else ("NO_TRADES" if n == 0 else "PASS"),
        "top_entry_gates": top_blockers,
        "profile_relaxation": relax,
        "loss_gate_when_loosened": loss_gate,
        "fast_profile_net": loss_profile.get("net_pnl_usd"),
        **ext,
        "contributing": n > 0 and net > 0,
        "status": "failed" if n == 0 or net <= 0 else "positive_replay_only",
    }


def _load_exec_bars() -> tuple[dict, dict]:
    end = datetime.now(timezone.utc)
    start = end - timedelta(days=95)
    start_ms = int(start.timestamp() * 1000)
    end_ms = int(end.timestamp() * 1000)
    bars_1h = {sym: fetch_klines_cached(sym, "1h", start_ms, end_ms) for sym in SYMBOLS}
    bars_exec: dict[str, dict] = {}
    for interval in ("1m", "5m", "15m"):
        bars_exec[interval] = {sym: fetch_klines_cached(sym, interval, start_ms, end_ms) for sym in SYMBOLS}
    return bars_1h, bars_exec


def _apply_replay_spec(cfg: ExecutionConfig, spec: dict[str, Any]) -> ExecutionConfig:
    cfg = _apply_variant(cfg, spec)
    if "replay_min_relative_volume" in spec:
        cfg.replay_min_relative_volume = float(spec["replay_min_relative_volume"])
    if spec.get("replay_symbols_allow"):
        cfg.replay_symbols_allow = spec["replay_symbols_allow"]
    if spec.get("regime_override_from_context"):
        cfg.regime_override_from_context = dict(spec["regime_override_from_context"])
    if spec.get("replay_entry_filter"):
        cfg.replay_entry_filter = str(spec["replay_entry_filter"])
    if spec.get("replay_setup_allow"):
        cfg.replay_setup_allow = spec["replay_setup_allow"]
    if "replay_min_thesis_score" in spec:
        cfg.replay_min_thesis_score = float(spec["replay_min_thesis_score"])
    if "replay_min_adx" in spec:
        cfg.replay_min_adx = float(spec["replay_min_adx"])
    return cfg


def _run_day_candidate(
    label: str,
    spec: dict[str, Any],
    bars_1h: dict,
    bars_exec: dict,
    profiles: dict,
) -> dict[str, Any]:
    try:
        cfg = _apply_replay_spec(_base_cfg(label, profiles), spec)
        cfg.controlled_exits_enabled = False
        suite = _run_suite(bars_1h, bars_exec, cfg)
        stress = _run_stress_battery(cfg, bars_1h, bars_exec)
        row = _evaluate(label, spec, suite, stress)
        w90 = suite["windows"]["90d"]
        row["win_rate_pct"] = round(float(w90.get("win_rate") or 0) * 100, 2)
        row["avg_win_usd"] = round(float(w90.get("average_win_usd") or 0), 2)
        row["avg_loss_usd"] = round(float(w90.get("average_loss_usd") or 0), 2)
        aw = float(w90.get("average_win_usd") or 0)
        al = float(w90.get("average_loss_usd") or 0)
        wins = int(w90.get("wins") or 0)
        losses = int(w90.get("losses") or 0)
        row["profit_factor"] = _profit_factor(wins, losses, aw, al)
        row["worst_mae_pct"] = w90.get("worst_intrabar_mae_pct")
        row["target_met_500"] = float(row.get("monthly_pnl_usd_on_25k") or 0) >= TARGET_500
        row["failed_profit_floor_count"] = w90.get("failed_profit_floor_count", 0)
        return row
    except Exception:
        return {
            "label": label,
            "error": traceback.format_exc(),
            "all_pass": False,
            "target_met_500": False,
            "accept_or_reject_reason": "replay_error",
        }


def _row_from_day(name: str, src: dict[str, Any]) -> dict[str, Any]:
    m = src.get("metrics_90d") or {}
    return {
        "strategy_name": name,
        "trades_per_month": src.get("trades_per_month"),
        "monthly_pnl_usd_on_25k": src.get("monthly_pnl_usd_on_25k"),
        "pct_per_month_on_25k": src.get("pct_per_month_on_25k"),
        "expectancy_per_trade_usd": src.get("expectancy_per_trade_usd") or m.get("expectancy_per_trade_usd"),
        "win_rate_pct": src.get("win_rate_pct"),
        "avg_win_usd": src.get("avg_win_usd"),
        "avg_loss_usd": src.get("avg_loss_usd"),
        "profit_factor": src.get("profit_factor"),
        "max_drawdown_pct": m.get("max_drawdown_pct"),
        "longest_hold_hours": m.get("longest_hold_hours"),
        "worst_mae_pct": src.get("worst_mae_pct") or m.get("worst_intrabar_mae_pct"),
        "all_pass": src.get("all_pass"),
        "target_met_500": src.get("target_met_500", False),
        "accept_or_reject_reason": src.get("accept_or_reject_reason") or src.get("verdict"),
    }


def _row_from_scalp(name: str, src: dict[str, Any]) -> dict[str, Any]:
    mo = float(src.get("extrapolated_monthly_pnl_usd") or 0)
    return {
        "strategy_name": name,
        "trades_per_month": src.get("extrapolated_trades_per_month"),
        "monthly_pnl_usd_on_25k": mo,
        "pct_per_month_on_25k": round(100.0 * mo / PRINCIPAL, 4),
        "expectancy_per_trade_usd": src.get("expectancy_per_trade_usd"),
        "win_rate_pct": src.get("win_rate_pct"),
        "avg_win_usd": src.get("avg_win_usd"),
        "avg_loss_usd": src.get("avg_loss_usd"),
        "profit_factor": src.get("profit_factor"),
        "max_drawdown_usd": src.get("max_drawdown_usd"),
        "longest_hold_sec": src.get("max_hold_sec"),
        "worst_mae_usd": src.get("avg_loss_usd"),
        "all_pass": src.get("contributing", False),
        "target_met_500": mo >= TARGET_500,
        "accept_or_reject_reason": (
            "positive_isolated_replay" if src.get("contributing")
            else ("no_trades" if src.get("total_trades", 0) == 0 else "negative_expectancy")
        ),
        "dominant_entry_gate": src.get("dominant_entry_gate"),
        "loss_gate_when_loosened": src.get("loss_gate_when_loosened"),
        "spread_sensitivity": src.get("spread_sensitivity"),
        "contributing": src.get("contributing", False),
    }


def main() -> int:
    from backend.services.replay_promotion_gate import block_exhausted_branch

    block_exhausted_branch("frequency_first_regime_remap")
    profiles = _build_fee_profiles()
    bull_pullback = frozenset((s, DAY_REGIME_BULL, SETUP_HTF_TREND_PULLBACK) for s in SYMBOLS)
    bull_breakout = frozenset((s, DAY_REGIME_BULL, SETUP_BREAKOUT_CONTINUATION) for s in SYMBOLS)
    bull_all = bull_pullback | bull_breakout

    print("  isolated scalp replay (4 strategies)...", flush=True)
    scalp_rows: list[dict] = []
    for strat in SCALP_STRATEGIES:
        print(f"    {strat}...", flush=True)
        scalp_rows.append(_replay_scalp_isolated(strat, profile="moderate", hours=SCALP_HOURS))
    best_scalp = max(scalp_rows, key=lambda x: float(x.get("extrapolated_monthly_pnl_usd") or -1e9))
    scalp_contributing = any(r.get("contributing") for r in scalp_rows)

    print("  load DAY bars...", flush=True)
    bars_1h, bars_exec = _load_exec_bars()
    regime_audit = _regime_taxonomy_audit(bars_1h)

    print("  lower-TF DAY entry candidates...", flush=True)
    ltf_candidates = [
        ("locked_live_baseline_1_5x", {"notional_mult": 1.5}),
        ("ltf_vwap_reclaim_15m", {
            "notional_mult": 1.5,
            "replay_entry_filter": "vwap_reclaim_15m",
            "replay_setup_allow": frozenset({SETUP_VWAP_REVERSION}),
        }),
        ("ltf_vwap_reclaim_30m", {
            "notional_mult": 1.5,
            "replay_entry_filter": "vwap_reclaim_30m",
            "replay_setup_allow": frozenset({SETUP_VWAP_REVERSION}),
        }),
        ("ltf_pullback_reclaim_5m15m", {
            "notional_mult": 1.5,
            "replay_entry_filter": "pullback_reclaim_5m15m",
            "replay_setup_allow": frozenset({SETUP_VWAP_REVERSION}),
        }),
        ("ltf_vol_compression_breakout", {
            "notional_mult": 1.5,
            "replay_entry_filter": "vol_compression_breakout",
            "extra_allowed_buckets": bull_breakout,
        }),
        ("ltf_range_low_reclaim", {
            "notional_mult": 1.5,
            "replay_entry_filter": "range_low_reclaim",
            "replay_setup_allow": frozenset({SETUP_VWAP_REVERSION}),
        }),
        ("ltf_high_relvol_reversal", {
            "notional_mult": 1.5,
            "replay_entry_filter": "high_relvol_reversal",
            "replay_setup_allow": frozenset({SETUP_VWAP_REVERSION}),
        }),
        ("ltf_eth_neutral_vwap_priority", {
            "notional_mult": 1.5,
            "replay_symbols_allow": frozenset({"ETH/USDT"}),
        }),
        ("ltf_sol_neutral_vwap", {
            "notional_mult": 1.5,
            "replay_symbols_allow": frozenset({"SOL/USDT"}),
        }),
        ("ltf_trend_continuation_confirmed", {
            "notional_mult": 1.5,
            "replay_entry_filter": "trend_continuation_confirmed",
            "extra_allowed_buckets": bull_all,
        }),
    ]
    ltf_rows = []
    for label, spec in ltf_candidates:
        print(f"    {label}...", flush=True)
        ltf_rows.append(_run_day_candidate(label, spec, bars_1h, bars_exec, profiles))

    locked = next(r for r in ltf_rows if r.get("label") == "locked_live_baseline_1_5x")
    ltf_only = [r for r in ltf_rows if r.get("label") != "locked_live_baseline_1_5x"]
    best_ltf = max(ltf_only, key=lambda x: float(x.get("monthly_pnl_usd_on_25k") or -1e9), default={})

    print("  safer regime-mapping candidates...", flush=True)
    regime_candidates = [
        ("map_trending_up_bull_pullback_only", {
            "notional_mult": 1.5,
            "regime_override_from_context": {"trending_up": DAY_REGIME_BULL},
            "extra_allowed_buckets": bull_pullback,
            "replay_min_adx": 22.0,
        }),
        ("map_trending_up_bull_eth_only", {
            "notional_mult": 1.5,
            "regime_override_from_context": {"trending_up": DAY_REGIME_BULL},
            "extra_allowed_buckets": bull_pullback,
            "replay_symbols_allow": frozenset({"ETH/USDT"}),
            "replay_min_thesis_score": 0.55,
        }),
        ("map_trending_up_bull_high_thesis", {
            "notional_mult": 1.5,
            "regime_override_from_context": {"trending_up": DAY_REGIME_BULL},
            "extra_allowed_buckets": bull_pullback,
            "replay_min_thesis_score": 0.65,
            "replay_min_adx": 25.0,
        }),
        ("map_trending_down_bear_block_longs", {
            "notional_mult": 1.5,
            "regime_override_from_context": {"trending_down": DAY_REGIME_BEAR},
        }),
        ("map_naive_trending_up_all_bull", {
            "notional_mult": 1.5,
            "regime_override_from_context": {"trending_up": DAY_REGIME_BULL},
            "extra_allowed_buckets": bull_all,
        }),
    ]
    regime_rows = []
    for label, spec in regime_candidates:
        print(f"    {label}...", flush=True)
        regime_rows.append(_run_day_candidate(label, spec, bars_1h, bars_exec, profiles))
    best_regime = max(regime_rows, key=lambda x: float(x.get("monthly_pnl_usd_on_25k") or -1e9), default={})

    day_mo = float(best_ltf.get("monthly_pnl_usd_on_25k") or locked.get("monthly_pnl_usd_on_25k") or 0)
    scalp_mo = float(best_scalp.get("extrapolated_monthly_pnl_usd") or 0) if scalp_contributing else 0.0
    combined_mo = round(
        float(locked.get("monthly_pnl_usd_on_25k") or 0)
        + (scalp_mo if scalp_contributing else 0),
        2,
    )

    summary_table = [
        _row_from_day("locked_live_DAY_baseline", locked),
        _row_from_scalp(f"best_isolated_scalp_{best_scalp.get('strategy')}", best_scalp),
        _row_from_day(f"best_lower_tf_DAY_{best_ltf.get('label', 'none')}", best_ltf),
        _row_from_day(f"best_regime_mapping_{best_regime.get('label', 'none')}", best_regime),
        {
            "strategy_name": "best_combined_DAY_plus_scalp",
            "trades_per_month": round(
                float(locked.get("trades_per_month") or 0)
                + (float(best_scalp.get("extrapolated_trades_per_month") or 0) if scalp_contributing else 0),
                2,
            ),
            "monthly_pnl_usd_on_25k": combined_mo,
            "pct_per_month_on_25k": round(100.0 * combined_mo / PRINCIPAL, 4),
            "expectancy_per_trade_usd": None,
            "win_rate_pct": None,
            "avg_win_usd": None,
            "avg_loss_usd": None,
            "profit_factor": None,
            "max_drawdown_pct": locked.get("metrics_90d", {}).get("max_drawdown_pct"),
            "longest_hold_hours": locked.get("metrics_90d", {}).get("longest_hold_hours"),
            "worst_mae_pct": locked.get("metrics_90d", {}).get("worst_intrabar_mae_pct"),
            "all_pass": bool(locked.get("all_pass")) and (scalp_contributing or scalp_mo == 0),
            "target_met_500": combined_mo >= TARGET_500,
            "accept_or_reject_reason": (
                "below_500_mo" if combined_mo < TARGET_500 else "pass"
            ),
            "day_component": locked.get("label"),
            "scalp_component": best_scalp.get("strategy") if scalp_contributing else "none_contributing",
        },
    ]

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "live_unchanged": LIVE_BASELINE_ID,
        "live_rules": {
            "notional_mult": 1.5,
            "min_net_profit_floor": 0.004,
            "buckets": "neutral VWAP only",
            "controlled_exits_live": False,
            "scalp_isolated": True,
        },
        "target_monthly_usd": TARGET_500,
        "scalp_marked_non_contributing": not scalp_contributing,
        "scalp_isolated_results": scalp_rows,
        "regime_taxonomy_audit": regime_audit,
        "lower_tf_day_candidates": ltf_rows,
        "regime_mapping_candidates": regime_rows,
        "summary_table": summary_table,
        "any_target_met_500": any(r.get("target_met_500") for r in summary_table),
        "next_step": (
            "entry_frequency_repair"
            if not any(r.get("target_met_500") for r in summary_table)
            else "validate_winner_before_live"
        ),
    }
    out = BASELINE_DIR / "frequency_first_rebuild_latest.json"
    out.write_text(json.dumps(report, indent=2, default=str))
    print(json.dumps({"summary_table": summary_table, "scalp_contributing": scalp_contributing, "out": str(out)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
