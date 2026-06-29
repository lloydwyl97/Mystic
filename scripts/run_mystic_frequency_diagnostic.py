#!/usr/bin/env python3
"""
Mystic frequency diagnostic — replay only. No live rule changes.

1. Per-strategy scalp gate audit
2. Regime/router taxonomy audit
3. DAY frequency expansion candidates
4. Summary growth table vs $500/mo target
"""

from __future__ import annotations

import json
import os
import statistics
import sys
import traceback
from collections import Counter, defaultdict
from copy import deepcopy
from dataclasses import dataclass, field, replace
from datetime import datetime, timedelta, timezone
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

from backend.services.binance_scalp.calibration_profiles import apply_profile, economics_for_config
from backend.services.binance_scalp.config import ScalpConfig
from backend.services.binance_scalp.momentum_tracker import MomentumTracker
from backend.services.binance_scalp.scalp_strategy_router import ScalpStrategyRouter
from backend.services.binance_scalp.strategies import ALL_STRATEGIES, STRATEGY_NAMES
from backend.services.day_bucket_quality import REPLAY_KILLED_BUCKETS, bucket_key
from backend.services.day_regime_router import (
    DAY_REGIME_BEAR,
    DAY_REGIME_BULL,
    DAY_REGIME_CHOP,
    DAY_REGIME_NEUTRAL,
    DAY_REGIME_RANGE,
    classify_day_regime,
    evaluate_day_entry_route,
)
from backend.services.day_trade_thesis import (
    SETUP_BREAKOUT_CONTINUATION,
    SETUP_HTF_TREND_PULLBACK,
    SETUP_NO_CLEAR_THESIS,
    SETUP_VWAP_REVERSION,
    apply_trade_thesis_to_candidate_fields,
)
from scripts.run_day_execution_replay import (
    ALLOWED_POSITIVE_BUCKETS,
    STRESS_SCENARIOS,
    ExecutionConfig,
    _apply_regime_override,
    _build_fee_profiles,
    _infer_context_regime_label,
    _run_stress_90d,
    _run_suite,
    fetch_klines_cached,
)
from scripts.run_day_profit_growth_search import (
    TARGET_MONTHLY,
    _apply_variant,
    _base_cfg,
    _evaluate,
    _run_stress_battery,
)
from scripts.run_day_strategy_replay import (
    NOTIONAL_USD,
    PRINCIPAL,
    SYMBOLS,
    _atr_pct,
    _resample_4h,
    build_decision_data,
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

SCALP_STRATEGIES = (
    "breakout_momentum",
    "orderbook_tape_scalp",
    "range_bounce_scalp",
    "vwap_ema_reclaim",
)
SCALP_HOURS = int(os.getenv("SCALP_REPLAY_HOURS", "168"))
TARGET_500 = TARGET_MONTHLY["2pct"]


def _monthly_from_90d(net90: float) -> float:
    return round(net90 / 3.0, 2)


def _monthly_from_scalp_hours(net: float, hours: float, trades: int) -> dict[str, float]:
    h = max(1.0, float(hours))
    return {
        "extrapolated_trades_per_month": round(trades * (730.0 / h), 2),
        "extrapolated_monthly_pnl_usd": round(net * (730.0 / h) / 3.0, 2),
    }


def _scalp_config_single(strategy: str, profile: str = "moderate") -> ScalpConfig:
    disabled = frozenset(s for s in SCALP_STRATEGIES if s != strategy)
    base = ScalpConfig.from_env()
    return replace(
        base,
        disabled_strategies=disabled,
        calibration_profile=profile,
        calibration_mode=True,
        scalp_live=False,
        scalp_paper_enabled=True,
    )


def _replay_scalp_strategy(
    strategy: str,
    *,
    profile: str = "moderate",
    hours: int = SCALP_HOURS,
) -> dict[str, Any]:
    config = _scalp_config_single(strategy, profile)
    econ = economics_for_config(config)
    momentum = MomentumTracker()
    router = ScalpStrategyRouter(
        config=config,
        econ=econ,
        reader=type("R", (), {"read": lambda _s, _sym: None})(),
        momentum=momentum,
    )
    end = datetime.now(timezone.utc)
    start = end - timedelta(hours=hours + 1)
    start_ms = int(start.timestamp() * 1000)
    end_ms = int(end.timestamp() * 1000)

    all_trades: list[dict] = []
    reject_counts: Counter = Counter()
    missed = 0
    stats = StrategyStats()

    for sym in config.products:
        bars = fetch_klines(sym, start_ms, end_ms)
        momentum._history.pop(sym, None)
        open_pos: OpenTrade | None = None
        cooldown_until = 0.0

        for idx in range(20, len(bars) - LOOKAHEAD_BARS):
            bar = bars[idx]
            epoch = bar["epoch"]
            snap = make_snap(sym, bar)
            for sub in range(4):
                t = epoch - 45 + sub * 15
                price = bar["open"] + (bar["close"] - bar["open"]) * (sub + 1) / 4
                sp = spread_est(bar)
                bid = price * (1.0 - sp / 2)
                momentum.record(sym, t, bid, price)
            momentum.record(sym, epoch, snap.best_bid, snap.mid)
            window = bars[max(0, idx - 60) : idx + 1]
            kline_window = [{"high": b["high"], "low": b["low"], "close": b["close"], "volume": b["volume"]} for b in window]

            if open_pos is not None:
                hold = epoch - open_pos.entry_epoch
                mom = momentum.diagnostics(sym, epoch, snap.best_bid, snap.mid)
                net_pct = net_pct_at_bid(
                    open_pos.entry_price,
                    snap.best_bid,
                    snap.spread_pct,
                    open_pos.impact_pct,
                    econ,
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
                from backend.services.binance_scalp.exit_manager import DECISION_SELL, PositionTrack

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
                    all_trades.append(
                        {
                            "symbol": sym,
                            "pnl_usd": pnl,
                            "hold_sec": hold,
                            "win": win,
                            "exit_reason": review.exit_reason,
                        }
                    )
                    open_pos = None
                    cooldown_until = epoch + 120
                continue

            if epoch < cooldown_until:
                continue

            best, all_sigs = router.evaluate_symbol(
                sym,
                epoch=epoch,
                notional_usd=NOTIONAL,
                snap=snap,
                bars=kline_window,
            )
            for sig in all_sigs:
                if not sig.passed and sig.reject_reason:
                    reject_counts[sig.reject_reason] += 1

            if best is None or not best.passed:
                chunk = bars[idx : idx + LOOKAHEAD_BARS]
                max_high = max(b["high"] for b in chunk)
                potential = (max_high - snap.best_ask) / snap.best_ask - econ.roundtrip_cost_pct(snap.spread_pct, 0, 0)
                if potential >= econ.net_profit_target_pct:
                    missed += 1
                continue

            from backend.services.binance_scalp.exit_manager import PositionTrack

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
    worst = round(min(pnls), 4) if pnls else 0.0
    avg_hold = round(statistics.mean(stats.hold_seconds), 1) if stats.hold_seconds else 0.0
    eq = 0.0
    peak = 0.0
    max_dd = 0.0
    for p in pnls:
        eq += p
        peak = max(peak, eq)
        max_dd = max(max_dd, peak - eq)

    top_blockers = reject_counts.most_common(8)
    dominant_gate = top_blockers[0][0] if top_blockers else "NONE"

    return {
        "strategy": strategy,
        "profile": profile,
        "replay_hours": hours,
        "total_trades": n,
        "win_rate_pct": wr,
        "net_pnl_usd": net,
        "expectancy_per_trade_usd": exp,
        "avg_hold_sec": avg_hold,
        "worst_loss_usd": worst,
        "max_drawdown_usd": round(max_dd, 4),
        **ext,
        "missed_profitable_windows": missed,
        "blocker_reasons": dict(reject_counts),
        "dominant_gate": dominant_gate,
        "top_blockers": top_blockers,
        "status": "dead" if n == 0 and missed > 0 else ("active" if n > 0 else "no_activity"),
    }


def _scalp_profile_relaxation(strategy: str) -> dict[str, Any]:
    rows = []
    for profile in ("strict", "moderate", "fast"):
        r = _replay_scalp_strategy(strategy, profile=profile, hours=SCALP_HOURS)
        rows.append(
            {
                "profile": profile,
                "trades": r["total_trades"],
                "net_pnl_usd": r["net_pnl_usd"],
                "extrapolated_monthly_pnl_usd": r["extrapolated_monthly_pnl_usd"],
                "dominant_gate": r["dominant_gate"],
            }
        )
    best = max(rows, key=lambda x: (x["net_pnl_usd"], x["trades"]))
    strict = next(x for x in rows if x["profile"] == "strict")
    improves = best["net_pnl_usd"] > strict["net_pnl_usd"] and best["trades"] >= strict["trades"]
    creates_losses = best["net_pnl_usd"] < 0 and best["trades"] > 0
    return {
        "strategy": strategy,
        "profiles": rows,
        "relaxation_improves_replay": improves,
        "relaxation_creates_losses": creates_losses,
        "note": (f"Gate relaxation via calibration profile only (replay). Best profile: {best['profile']} trades={best['trades']} net={best['net_pnl_usd']}"),
    }


def _regime_taxonomy_audit(bars_1h: dict[str, list]) -> dict[str, Any]:
    end_ts = bars_1h[SYMBOLS[0]][-1]["ts"]
    start_ts = end_ts - 90 * 86400
    ctx_counts: Counter = Counter()
    router_counts: Counter = Counter()
    pair_counts: Counter = Counter()
    mismatch_bull_pullback = 0
    mismatch_breakout = 0
    label_mismatch_only = 0
    bars_total = 0
    per_symbol: dict[str, Any] = {}

    for sym in SYMBOLS:
        sym_ctx: Counter = Counter()
        sym_router: Counter = Counter()
        idx = 0
        bars = bars_1h[sym]
        while idx < len(bars) and bars[idx]["ts"] < start_ts:
            idx += 1
        for i in range(max(idx, 80), len(bars)):
            if bars[i]["ts"] > end_ts:
                break
            slice_1h = bars[: i + 1]
            slice_4h = _resample_4h(slice_1h)
            dd = build_decision_data(sym, slice_1h, slice_4h)
            mark = dd["current_price"]
            atr = _atr_pct(slice_1h) * mark
            chop = 0.65 if dd["adx"] < 18 else 0.45
            ps = dd["price_structure_regime"]
            dd = apply_trade_thesis_to_candidate_fields(
                dd,
                symbol=sym,
                current_price=mark,
                atr=atr,
                strategy_id="day",
                price_structure_regime=ps,
            )
            ctx = _infer_context_regime_label(dd)
            router = classify_day_regime(
                dd,
                context_payload=None,
                chop_score=chop,
                atr_ratio=_atr_pct(slice_1h),
                price_structure_regime=ps,
            )
            setup = str(dd.get("setup_type") or SETUP_NO_CLEAR_THESIS)
            bars_total += 1
            ctx_counts[ctx] += 1
            router_counts[router] += 1
            pair_counts[f"{ctx}|{router}"] += 1
            sym_ctx[ctx] += 1
            sym_router[router] += 1

            if ctx in ("trending_up", "bull", "bullish") and router != DAY_REGIME_BULL:
                label_mismatch_only += 1
                if setup == SETUP_HTF_TREND_PULLBACK:
                    route_neutral = evaluate_day_entry_route(
                        setup_type=setup,
                        day_regime=router,
                        decision_data=dd,
                        context_payload=None,
                        current_price=mark,
                        thesis_score=float(dd.get("thesis_score") or 0),
                    )
                    route_bull = evaluate_day_entry_route(
                        setup_type=setup,
                        day_regime=DAY_REGIME_BULL,
                        decision_data=dd,
                        context_payload=None,
                        current_price=mark,
                        thesis_score=float(dd.get("thesis_score") or 0),
                    )
                    if not route_neutral.get("allowed") and route_bull.get("allowed"):
                        mismatch_bull_pullback += 1
                if setup == SETUP_BREAKOUT_CONTINUATION:
                    mismatch_breakout += 1

        per_symbol[sym] = {
            "context_regime_bars": dict(sym_ctx),
            "router_regime_bars": dict(sym_router),
        }

    mapping_table = [
        {"context_label": "trending_up", "router_label": "bull", "live_mapping": "none", "replay_test": "trending_up→bull"},
        {"context_label": "trending_down", "router_label": "bear", "live_mapping": "partial (bear in mr string only)", "replay_test": "trending_down→bear"},
        {"context_label": "range", "router_label": "range", "live_mapping": "adx/structure", "replay_test": "aligned"},
        {"context_label": "neutral", "router_label": "neutral", "live_mapping": "default fallback", "replay_test": "aligned"},
        {"context_label": "trending_up", "router_label": "neutral", "live_mapping": "MISMATCH — h1/h4/ema bull thresholds not met", "replay_test": "audit only"},
    ]

    return {
        "bars_90d_total": bars_total,
        "context_regime_distribution": dict(ctx_counts),
        "router_regime_distribution": dict(router_counts),
        "context_router_pairs": dict(pair_counts),
        "per_symbol": per_symbol,
        "mapping_table": mapping_table,
        "trending_up_labeled_but_not_bull": label_mismatch_only,
        "would_unlock_bull_pullback_if_mapped": mismatch_bull_pullback,
        "would_unlock_breakout_if_mapped": mismatch_breakout,
        "note": ("Live ai_context uses ctx_market_regime (trending_up/down). Router uses bull/bear/neutral/range/chop from HTF alignment + ADX — not 1:1 with context labels."),
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


def _apply_variant_ext(cfg: ExecutionConfig, spec: dict[str, Any]) -> ExecutionConfig:
    cfg = _apply_variant(cfg, spec)
    if "replay_min_relative_volume" in spec:
        cfg.replay_min_relative_volume = float(spec["replay_min_relative_volume"])
    if spec.get("replay_symbols_allow"):
        cfg.replay_symbols_allow = spec["replay_symbols_allow"]
    if spec.get("regime_override_from_context"):
        cfg.regime_override_from_context = dict(spec["regime_override_from_context"])
    return cfg


def _run_day_candidate(
    label: str,
    spec: dict[str, Any],
    bars_1h: dict,
    bars_exec: dict,
    profiles: dict,
) -> dict[str, Any]:
    try:
        cfg = _apply_variant_ext(_base_cfg(label, profiles), spec)
        suite = _run_suite(bars_1h, bars_exec, cfg)
        stress = _run_stress_battery(cfg, bars_1h, bars_exec)
        row = _evaluate(label, spec, suite, stress)
        row["target_met_500"] = float(row.get("monthly_pnl_usd_on_25k") or 0) >= TARGET_500
        return row
    except Exception:
        return {"label": label, "error": traceback.format_exc(), "all_pass": False, "target_met_500": False}


def main() -> int:
    print("=== MYSTIC FREQUENCY DIAGNOSTIC (replay-only) ===", flush=True)
    tracebacks: list[str] = []
    report: dict[str, Any] = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "mystic_finished": False,
        "live_unchanged": "day_baseline_all_pass_v1_size_1_5",
        "target_monthly_usd": TARGET_500,
    }

    try:
        print("  scalp per-strategy audit...", flush=True)
        scalp_rows = []
        for strat in SCALP_STRATEGIES:
            try:
                row = _replay_scalp_strategy(strat, profile="moderate", hours=SCALP_HOURS)
                row["profile_relaxation"] = _scalp_profile_relaxation(strat)
                scalp_rows.append(row)
            except Exception:
                tracebacks.append(traceback.format_exc())
                scalp_rows.append({"strategy": strat, "error": traceback.format_exc()})

        report["scalp_strategy_audit"] = scalp_rows
        best_scalp = max(
            scalp_rows,
            key=lambda x: float(x.get("extrapolated_monthly_pnl_usd") or 0),
            default={},
        )

        print("  regime taxonomy audit...", flush=True)
        bars_1h, bars_exec = _load_exec_bars()
        report["regime_taxonomy_audit"] = _regime_taxonomy_audit(bars_1h)

        print("  DAY frequency candidates...", flush=True)
        bull_buckets = frozenset((sym, DAY_REGIME_BULL, SETUP_HTF_TREND_PULLBACK) for sym in SYMBOLS) | frozenset((sym, DAY_REGIME_BULL, SETUP_BREAKOUT_CONTINUATION) for sym in SYMBOLS)
        profiles = _build_fee_profiles()
        day_candidates = [
            ("locked_baseline_1_5x", {"notional_mult": 1.5}),
            ("growth_candidate_2_5x_floor_0_60", {"notional_mult": 2.5, "min_net_profit_floor": 0.006}),
            (
                "high_relvol_neutral_vwap",
                {
                    "notional_mult": 1.5,
                    "replay_min_relative_volume": 1.0,
                },
            ),
            (
                "eth_priority_neutral_vwap",
                {
                    "notional_mult": 1.5,
                    "replay_symbols_allow": frozenset({"ETH/USDT"}),
                },
            ),
            (
                "sol_only_neutral_vwap",
                {
                    "notional_mult": 1.5,
                    "replay_symbols_allow": frozenset({"SOL/USDT"}),
                },
            ),
            (
                "trending_up_as_bull_pullback",
                {
                    "notional_mult": 1.5,
                    "regime_override_from_context": {"trending_up": DAY_REGIME_BULL},
                    "extra_allowed_buckets": bull_buckets,
                },
            ),
            (
                "trending_up_bull_breakout",
                {
                    "notional_mult": 1.5,
                    "regime_override_from_context": {"trending_up": DAY_REGIME_BULL},
                    "extra_allowed_buckets": frozenset((sym, DAY_REGIME_BULL, SETUP_BREAKOUT_CONTINUATION) for sym in SYMBOLS),
                },
            ),
        ]
        day_rows = []
        for label, spec in day_candidates:
            print(f"    {label}...", flush=True)
            day_rows.append(_run_day_candidate(label, spec, bars_1h, bars_exec, profiles))
        report["day_frequency_candidates"] = day_rows

        locked = next((r for r in day_rows if r.get("label") == "locked_baseline_1_5x"), {})
        best_day = max(day_rows, key=lambda x: float(x.get("monthly_pnl_usd_on_25k") or 0), default={})
        best_growth = next(
            (r for r in day_rows if r.get("label") == "growth_candidate_2_5x_floor_0_60"),
            best_day,
        )
        best_bucket = max(
            (
                r
                for r in day_rows
                if r.get("label")
                not in (
                    "locked_baseline_1_5x",
                    "growth_candidate_2_5x_floor_0_60",
                )
            ),
            key=lambda x: float(x.get("monthly_pnl_usd_on_25k") or 0),
            default={},
        )

        scalp_mo = float(best_scalp.get("extrapolated_monthly_pnl_usd") or 0)
        day_mo = float(best_day.get("monthly_pnl_usd_on_25k") or 0)
        combined_mo = round(day_mo + scalp_mo, 2)

        def _summary_row(name: str, src: dict, is_scalp: bool = False) -> dict[str, Any]:
            if is_scalp:
                return {
                    "row": name,
                    "trades_per_month": src.get("extrapolated_trades_per_month"),
                    "monthly_pnl_usd_on_25k": src.get("extrapolated_monthly_pnl_usd"),
                    "pct_per_month_on_25k": round(100.0 * float(src.get("extrapolated_monthly_pnl_usd") or 0) / PRINCIPAL, 4),
                    "max_drawdown": src.get("max_drawdown_usd"),
                    "longest_hold": src.get("avg_hold_sec"),
                    "worst_mae": src.get("worst_loss_usd"),
                    "all_pass": src.get("total_trades", 0) > 0 and float(src.get("net_pnl_usd") or 0) >= 0,
                    "target_met_500": float(src.get("extrapolated_monthly_pnl_usd") or 0) >= TARGET_500,
                    "label": src.get("strategy"),
                    "status": src.get("status"),
                    "dominant_gate": src.get("dominant_gate"),
                }
            m = src.get("metrics_90d") or {}
            return {
                "row": name,
                "label": src.get("label"),
                "trades_per_month": src.get("trades_per_month"),
                "monthly_pnl_usd_on_25k": src.get("monthly_pnl_usd_on_25k"),
                "pct_per_month_on_25k": src.get("pct_per_month_on_25k"),
                "max_drawdown_pct": m.get("max_drawdown_pct"),
                "longest_hold_hours": m.get("longest_hold_hours"),
                "worst_mae_pct": m.get("worst_intrabar_mae_pct"),
                "all_pass": src.get("all_pass"),
                "target_met_500": src.get("target_met_500", False),
                "verdict": src.get("verdict"),
                "accept_or_reject_reason": src.get("accept_or_reject_reason"),
            }

        report["summary_table"] = [
            _summary_row("locked_DAY_baseline", locked),
            _summary_row("best_DAY_only_growth_candidate", best_growth),
            _summary_row("best_scalp_candidate", best_scalp, is_scalp=True),
            {
                "row": "best_DAY_plus_scalp_combined",
                "day_component": best_day.get("label"),
                "scalp_component": best_scalp.get("strategy"),
                "trades_per_month": round(
                    float(best_day.get("trades_per_month") or 0) + float(best_scalp.get("extrapolated_trades_per_month") or 0),
                    2,
                ),
                "monthly_pnl_usd_on_25k": combined_mo,
                "pct_per_month_on_25k": round(100.0 * combined_mo / PRINCIPAL, 4),
                "all_pass": bool(best_day.get("all_pass")) and (best_scalp.get("total_trades", 0) == 0 or float(best_scalp.get("net_pnl_usd") or 0) >= 0),
                "target_met_500": combined_mo >= TARGET_500,
            },
            _summary_row("best_frequency_expansion_candidate", best_bucket),
        ]
        report["any_target_met_500"] = any(r.get("target_met_500") for r in report["summary_table"] if isinstance(r, dict))
        report["tracebacks"] = tracebacks

        out = BASELINE_DIR / "mystic_frequency_diagnostic_latest.json"
        out.write_text(json.dumps(report, indent=2, default=str))
        print(json.dumps({"summary_table": report["summary_table"], "out": str(out)}, indent=2))
        return 0
    except Exception:
        tracebacks.append(traceback.format_exc())
        print(json.dumps({"error": tracebacks}, indent=2))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
