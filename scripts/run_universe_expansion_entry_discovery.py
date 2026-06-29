#!/usr/bin/env python3
"""
Universe-expansion entry discovery — research only, no live promotion.

Scans all Binance.US liquid spot pairs, builds labeled dataset, mines symbol-specific
buckets, ranks symbols, tests scalp on liquid subset. Live top-four floor unchanged.
"""

from __future__ import annotations

import json
import os
import pickle
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
BASELINE_DIR = REPO / "scripts" / "replay_baselines"
CACHE_DIR = BASELINE_DIR / "universe_cache"
UNIVERSE_CACHE = BASELINE_DIR / "universe_eligibility_cache.json"
DATASET_CACHE = BASELINE_DIR / "universe_expansion_dataset_cache.pkl"
OUT_PATH = BASELINE_DIR / "universe_expansion_entry_discovery_latest.json"

from backend.services.ai_outcome_dataset import FEATURE_NAMES_145
from backend.services.replay_promotion_gate import RESEARCH_EXCLUDED_SYMBOLS
from backend.services.universe_eligibility import scan_eligible_universe
from backend.services.universe_outcome_dataset import (
    UNIVERSE_EXTRA_NAMES,
    build_universe_dataset_rows,
    compute_symbol_ranks,
    load_symbol_bars,
)

TARGET_MONTHLY = 500.0
PRINCIPAL = 25_000.0
DAYS = 90
TRAIN_DAYS = 54
VAL_DAYS = 18
TEST_DAYS = 18
NOTIONAL_DAY = PRINCIPAL / 4
NOTIONAL_SCALP = 25.0
MAX_DATASET_SYMBOLS = int(os.getenv("UNIVERSE_RESEARCH_MAX_SYMBOLS", "35"))
MAX_SCALP_SYMBOLS = int(os.getenv("UNIVERSE_SCALP_MAX_SYMBOLS", "4"))
SCALP_SAMPLE_SEC = 1800
TOP_FOUR_CCXT = frozenset({"BTC/USDT", "ETH/USDT", "SOL/USDT", "XRP/USDT"})
FEATURE_NAMES = tuple(FEATURE_NAMES_145) + UNIVERSE_EXTRA_NAMES

try:
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.metrics import roc_auc_score
    from sklearn.preprocessing import StandardScaler

    SKLEARN_OK = True
except ImportError:
    SKLEARN_OK = False


def _realized_net_pct(lab: dict, scalp: bool = False) -> float:
    if lab.get("hit_40bp_before_40bp"):
        return 0.004
    mae = float(lab.get("mae_72h" if not scalp else "mae_30m") or lab.get("mae_24h") or 0)
    if mae <= -0.004:
        return -0.004
    return float(lab.get("expected_net_pnl_pct") or 0)


def _metrics(rows: list[dict], notional: float = NOTIONAL_DAY, scalp: bool = False) -> dict[str, Any]:
    if not rows:
        return {"trades": 0, "monthly_pnl_usd": 0.0, "expectancy_per_trade": 0.0, "win_rate_pct": 0}
    pnls, holds, wins, maes = [], [], [], []
    equity, peak, max_dd = 0.0, 0.0, 0.0
    for r in rows:
        lab = r.get("labels") or {}
        net = _realized_net_pct(lab, scalp=scalp)
        pnl = net * notional
        pnls.append(pnl)
        wins.append(pnl > 0)
        holds.append(float(lab.get("time_to_profit_sec") or lab.get("best_net_hold_sec") or 0))
        maes.append(float(lab.get("mae_72h") or lab.get("mae_30m") or 0))
        equity += pnl
        peak = max(peak, equity)
        max_dd = max(max_dd, peak - equity)
    n = len(pnls)
    net = sum(pnls)
    months = max(DAYS / 30.0, 1)
    gw = sum(p for p in pnls if p > 0)
    gl = abs(sum(p for p in pnls if p <= 0))
    return {
        "trades": n,
        "trades_per_month": round(n / months, 2),
        "net_pnl_usd": round(net, 2),
        "monthly_pnl_usd": round(net / months, 2),
        "pct_per_month": round(100.0 * net / months / PRINCIPAL, 4),
        "expectancy_per_trade": round(net / n, 4),
        "win_rate_pct": round(100.0 * sum(wins) / n, 2),
        "avg_win_usd": round(gw / max(sum(wins), 1), 2),
        "avg_loss_usd": round(-gl / max(n - sum(wins), 1), 2),
        "profit_factor": round(gw / gl, 3) if gl > 1e-9 else (999.0 if gw > 0 else 0),
        "longest_hold_hours": round(max(holds) / 3600.0, 2) if holds else 0,
        "worst_mae_pct": round(min(maes) * 100, 4) if maes else 0,
        "max_drawdown_usd": round(max_dd, 2),
        "max_drawdown_pct": round(100.0 * max_dd / PRINCIPAL, 4),
    }


def _split(rows: list[dict], start_ts: int) -> tuple[list, list, list]:
    te = start_ts + TRAIN_DAYS * 86400
    ve = te + VAL_DAYS * 86400
    return (
        [r for r in rows if r["timestamp"] < te],
        [r for r in rows if te <= r["timestamp"] < ve],
        [r for r in rows if r["timestamp"] >= ve],
    )


def _train_per_symbol(train: list, val: list, dim: int) -> dict[str, Any]:
    if not SKLEARN_OK:
        return {}
    out: dict[str, Any] = {}
    for sym in sorted({r["symbol"] for r in train}):
        tr = [r for r in train if r["symbol"] == sym]
        va = [r for r in val if r["symbol"] == sym]
        if len(tr) < 60:
            continue
        Xtr = np.array([r["features"] for r in tr], dtype=np.float64)
        ytr = np.array([1 if (r.get("labels") or {}).get("hit_40bp_before_40bp") else 0 for r in tr])
        Xva = np.array([r["features"] for r in va], dtype=np.float64) if va else Xtr[:0]
        yva = np.array([1 if (r.get("labels") or {}).get("hit_40bp_before_40bp") else 0 for r in va]) if va else ytr[:0]
        if len(set(ytr)) < 2:
            continue
        sc = StandardScaler()
        Xt = sc.fit_transform(Xtr)
        model = GradientBoostingClassifier(n_estimators=80, max_depth=4, random_state=42)
        model.fit(Xt, ytr)
        imp = model.feature_importances_
        top = sorted(range(len(FEATURE_NAMES)), key=lambda i: -imp[i])[:8]
        auc = 0.5
        if len(Xva) and len(set(yva)) > 1:
            auc = float(roc_auc_score(yva, model.predict_proba(sc.transform(Xva))[:, 1]))
        out[sym] = {
            "val_auc": round(auc, 4),
            "top_features": [FEATURE_NAMES[i] for i in top if i < len(FEATURE_NAMES)],
            "importance": imp,
            "model": model,
            "scaler": sc,
        }
    return out


def _discover_symbol_buckets(
    symbol: str,
    train: list[dict],
    val: list[dict],
    test: list[dict],
    model_info: dict | None,
) -> list[dict]:
    buckets: list[dict] = []
    imp = model_info.get("importance") if model_info else None
    if imp is None or len(train) < 40:
        return buckets

    meta_keys = [
        ("regime", lambda r: r.get("meta", {}).get("regime")),
        ("relvol_band", lambda r: "high" if float(r.get("meta", {}).get("relative_volume") or 0) >= 1.2 else "low"),
        ("vwap_side", lambda r: "above" if float(r.get("meta", {}).get("vwap_distance_pct") or 0) >= 0 else "below"),
    ]
    top_fi = int(np.argmax(imp))

    filters: list[tuple[str, Any]] = []
    for name, fn in meta_keys:
        filters.append((name, fn))
    lo, hi = np.quantile([r["features"][top_fi] for r in train], [0.33, 0.66])
    filters.append((FEATURE_NAMES[top_fi], lambda r, lo=lo, hi=hi, fi=top_fi: lo <= r["features"][fi] <= hi))

    for fname, fn in filters:
        tr = [r for r in train if r["symbol"] == symbol and fn(r)]
        va = [r for r in val if r["symbol"] == symbol and fn(r)]
        te = [r for r in test if r["symbol"] == symbol and fn(r)]
        if len(tr) < 15:
            continue
        tm, vm, tem = _metrics(tr), _metrics(va), _metrics(te)
        if tm.get("expectancy_per_trade", 0) <= 0:
            continue
        setup = f"{symbol}_{fname}"
        stress_rows = [{**r, "labels": {**(r.get("labels") or {}), **{"expected_net_pnl_pct": _realized_net_pct(r.get("labels") or {}) - 0.0015}}} for r in te]
        stress_m = _metrics(stress_rows)
        wf_val_pos = vm.get("expectancy_per_trade", 0) > 0
        wf_test_pos = tem.get("expectancy_per_trade", 0) > 0
        hold_ok = tem.get("longest_hold_hours", 999) <= 72
        reasons = []
        if tem.get("monthly_pnl_usd", 0) < TARGET_MONTHLY:
            reasons.append("below_500_mo")
        if not wf_test_pos:
            reasons.append("walk_forward_test_fail")
        if not wf_val_pos:
            reasons.append("walk_forward_val_fail")
        if not stress_m.get("expectancy_per_trade", 0) > 0:
            reasons.append("stress_fail")
        if not hold_ok:
            reasons.append("fat_tail_hold")
        buckets.append(
            {
                "symbol": symbol,
                "setup_name": setup,
                "filter": fname,
                "train_metrics": tm,
                "validation_metrics": vm,
                "test_metrics": tem,
                "stress_metrics": stress_m,
                "walk_forward_validation_pass": wf_val_pos,
                "walk_forward_test_pass": wf_test_pos,
                "spread_stress_pass": stress_m.get("expectancy_per_trade", 0) > 0,
                "all_pass": len(reasons) == 0,
                "target_met_500": tem.get("monthly_pnl_usd", 0) >= TARGET_MONTHLY,
                "accept_or_reject_reason": reasons if reasons else "pass",
            }
        )
    buckets.sort(key=lambda b: float((b.get("test_metrics") or {}).get("monthly_pnl_usd") or -1e9), reverse=True)
    return buckets[:5]


def _rank_symbols(
    accepted: list[dict],
    per_sym_models: dict,
    all_buckets: list[dict],
    rows: list[dict],
    start_ts: int,
) -> list[dict]:
    _, _, test = _split(rows, start_ts)
    ranked: list[dict] = []
    for m in accepted:
        sym = m["ccxt_symbol"]
        sym_test = [r for r in test if r["symbol"] == sym]
        sym_m = _metrics(sym_test)
        best_b = max(
            (b for b in all_buckets if b["symbol"] == sym),
            key=lambda b: float((b.get("test_metrics") or {}).get("monthly_pnl_usd") or -1e9),
            default={},
        )
        mi = per_sym_models.get(sym, {})
        ranked.append(
            {
                "symbol": sym,
                "daily_volume_usd": m.get("daily_volume_usd"),
                "half_spread_pct": m.get("half_spread_pct"),
                "recent_test_expectancy": sym_m.get("expectancy_per_trade"),
                "walk_forward_test_monthly_pnl": (best_b.get("test_metrics") or {}).get("monthly_pnl_usd"),
                "liquidity_score": round(min(1.0, __import__("math").log10(max(float(m.get("daily_volume_usd") or 1), 1)) / 7.0), 4),
                "spread_score": round(1.0 - min(1.0, float(m.get("half_spread_pct") or 0) / 0.0015), 4),
                "model_val_auc": mi.get("val_auc"),
                "best_setup": best_b.get("setup_name"),
                "fat_tail_hold_hours": sym_m.get("longest_hold_hours"),
                "trade_frequency_per_month": sym_m.get("trades_per_month"),
            }
        )
    ranked.sort(
        key=lambda x: float(x.get("walk_forward_test_monthly_pnl") or x.get("recent_test_expectancy") or -1e9),
        reverse=True,
    )
    for i, r in enumerate(ranked):
        r["rank"] = i + 1
    return ranked


def _scalp_candidates(accepted: list[dict]) -> list[dict]:
    cands = [a for a in accepted if a.get("ccxt_symbol") in TOP_FOUR_CCXT and float(a.get("half_spread_pct") or 1) <= 0.001 and float(a.get("daily_volume_usd") or 0) >= 100_000]
    cands.sort(key=lambda x: (-float(x.get("daily_volume_usd") or 0), float(x.get("half_spread_pct") or 0)))
    return cands[:MAX_SCALP_SYMBOLS]


def _reject_combined(metrics: dict, wf_test: bool, stress: bool, hold_h: float, scalp: bool = False) -> tuple[bool, list[str]]:
    reasons = []
    if metrics.get("monthly_pnl_usd", 0) < TARGET_MONTHLY:
        reasons.append("below_500_mo")
    if metrics.get("expectancy_per_trade", 0) <= 0:
        reasons.append("negative_expectancy")
    if not wf_test:
        reasons.append("walk_forward_test_fail")
    if not stress:
        reasons.append("stress_fail")
    limit = 0.5 if scalp else 72
    if hold_h > limit:
        reasons.append("fat_tail_hold")
    return len(reasons) == 0, reasons


def main() -> int:
    print("=== UNIVERSE EXPANSION ENTRY DISCOVERY (research-only) ===", flush=True)
    if not SKLEARN_OK:
        print("sklearn required", file=sys.stderr)
        return 1

    end = datetime.now(timezone.utc)
    start = end - timedelta(days=DAYS + 5)
    start_ms = int(start.timestamp() * 1000)
    end_ms = int(end.timestamp() * 1000)
    start_ts = int((end - timedelta(days=DAYS)).timestamp())
    end_ts = int(end.timestamp())
    scalp_start = int((end - timedelta(days=30)).timestamp())

    print("  scanning Binance.US universe...", flush=True)
    if UNIVERSE_CACHE.exists():
        universe = json.loads(UNIVERSE_CACHE.read_text())
        print(f"    loaded universe cache accepted={universe.get('pairs_accepted')}", flush=True)
    else:
        universe = scan_eligible_universe(cache_dir=CACHE_DIR, max_coverage_checks=80)
        UNIVERSE_CACHE.write_text(json.dumps(universe, indent=2))
        print(f"    scanned={universe['total_pairs_scanned']} accepted={universe['pairs_accepted']}", flush=True)

    accepted = [a for a in (universe.get("accepted") or []) if a.get("ccxt_symbol") not in RESEARCH_EXCLUDED_SYMBOLS]
    universe["pairs_accepted"] = len(accepted)
    universe["pairs_rejected"] = int(universe.get("pairs_rejected") or 0) + len(RESEARCH_EXCLUDED_SYMBOLS)
    research_syms = accepted[:MAX_DATASET_SYMBOLS]
    ccxt_list = [m["ccxt_symbol"] for m in research_syms]

    rows: list[dict] = []
    done_syms: set[str] = set()
    scalp_meta_by_ccxt_pre = {m["ccxt_symbol"]: m for m in _scalp_candidates(accepted)}

    def _symbol_complete(ccxt: str, existing: list[dict]) -> bool:
        has_day = any(r["symbol"] == ccxt and r.get("timeframe") != "1m" for r in existing)
        if not has_day:
            return False
        if ccxt not in scalp_meta_by_ccxt_pre:
            return True
        return any(r["symbol"] == ccxt and r.get("timeframe") == "1m" for r in existing)

    if DATASET_CACHE.exists():
        try:
            cached = pickle.loads(DATASET_CACHE.read_bytes())
            if cached.get("version") == 1 and cached.get("rows"):
                rows = cached["rows"]
                done_syms = set(cached.get("done_symbols") or [])
                if not done_syms:
                    done_syms = {s for s in ccxt_list if _symbol_complete(s, rows)}
                print(f"    loaded cache rows={len(rows)} done_symbols={len(done_syms)}", flush=True)
        except Exception:
            pass

    if len(done_syms) < len(research_syms):
        print(f"  building dataset for {len(research_syms)} symbols...", flush=True)
        bars_1h_by_sym: dict[str, list] = {}
        for m in research_syms:
            api, ccxt = m["api_symbol"], m["ccxt_symbol"]
            bars_1h_by_sym[ccxt] = load_symbol_bars(api, ccxt, start_ms, end_ms, CACHE_DIR).get("1h", [])

        btc_meta = next((a for a in accepted if a["api_symbol"] == "BTCUSDT"), accepted[0] if accepted else None)
        btc_1h = bars_1h_by_sym.get("BTC/USDT", [])
        if not btc_1h and btc_meta:
            btc_1h = load_symbol_bars(btc_meta["api_symbol"], "BTC/USDT", start_ms, end_ms, CACHE_DIR).get("1h", [])

        strength_ranks, vol_ranks = compute_symbol_ranks(research_syms, bars_1h_by_sym)
        scalp_meta_by_ccxt = {m["ccxt_symbol"]: m for m in _scalp_candidates(accepted)}

        for m in research_syms:
            api, ccxt = m["api_symbol"], m["ccxt_symbol"]
            if ccxt in done_syms:
                print(f"    skip {ccxt} (cached)", flush=True)
                continue
            hs = float(m.get("half_spread_pct") or 0.00006)
            print(f"    DAY {ccxt}...", flush=True)
            bars = load_symbol_bars(api, ccxt, start_ms, end_ms, CACHE_DIR)
            umeta = {"daily_volume_usd": m.get("daily_volume_usd"), "half_spread_pct": hs}
            rows.extend(
                build_universe_dataset_rows(
                    ccxt,
                    bars,
                    start_ts,
                    end_ts,
                    sample_sec=3600,
                    half_spread=hs,
                    universe_meta=umeta,
                    btc_1h=btc_1h,
                    strength_ranks=strength_ranks,
                    vol_ranks=vol_ranks,
                    scalp=False,
                )
            )
            if ccxt in scalp_meta_by_ccxt:
                print(f"    SCALP {ccxt}...", flush=True)
                rows.extend(
                    build_universe_dataset_rows(
                        ccxt,
                        bars,
                        scalp_start,
                        end_ts,
                        sample_sec=SCALP_SAMPLE_SEC,
                        half_spread=hs,
                        universe_meta=umeta,
                        btc_1h=btc_1h,
                        strength_ranks=strength_ranks,
                        vol_ranks=vol_ranks,
                        scalp=True,
                    )
                )
            done_syms.add(ccxt)
            DATASET_CACHE.write_bytes(
                pickle.dumps(
                    {
                        "version": 1,
                        "rows": rows,
                        "symbols": ccxt_list,
                        "done_symbols": sorted(done_syms),
                    }
                )
            )

        DATASET_CACHE.write_bytes(pickle.dumps({"version": 1, "rows": rows, "symbols": ccxt_list}))

    day_rows = [r for r in rows if r.get("timeframe") != "1m"]
    scalp_rows = [r for r in rows if r.get("timeframe") == "1m"]
    train, val, test = _split(day_rows, start_ts)
    _, _, scalp_test = _split(scalp_rows, start_ts)

    print(f"  rows={len(rows)} day={len(day_rows)} scalp={len(scalp_rows)} features={len(FEATURE_NAMES)}", flush=True)
    print("  training per-symbol models...", flush=True)
    per_sym = _train_per_symbol(train, val, len(FEATURE_NAMES))

    all_buckets: list[dict] = []
    for sym in sorted({r["symbol"] for r in train}):
        all_buckets.extend(_discover_symbol_buckets(sym, train, val, test, per_sym.get(sym)))

    symbol_rankings = _rank_symbols(accepted, per_sym, all_buckets, day_rows, start_ts)

    best_day = max(all_buckets, key=lambda b: float((b.get("test_metrics") or {}).get("monthly_pnl_usd") or -1e9), default={})
    best_day_m = best_day.get("test_metrics") or _metrics(test)
    scalp_m = _metrics(scalp_test, NOTIONAL_SCALP, scalp=True) if scalp_test else {"trades": 0, "monthly_pnl_usd": 0}
    best_scalp = {"test_metrics": scalp_m, "setup_name": "expanded_liquid_scalp_pool", "symbol": "multi"}

    combined_rows = test[:]
    if best_day:
        sym = best_day.get("symbol")
        combined_rows = [r for r in test if r["symbol"] == sym] if sym else test
    combined_m = _metrics(combined_rows)
    wf_test = float(best_day_m.get("expectancy_per_trade") or 0) > 0
    stress_ok = bool((best_day.get("stress_metrics") or {}).get("expectancy_per_trade", 0) > 0)
    accepted_flag, reject_reasons = _reject_combined(
        best_day_m,
        wf_test,
        stress_ok,
        float(best_day_m.get("longest_hold_hours") or 0),
    )

    reject_summary: dict[str, int] = {}
    for r in universe.get("rejected") or []:
        for reason in r.get("reject_reasons") or []:
            reject_summary[reason] = reject_summary.get(reason, 0) + 1

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "discovery_type": "universe_expansion_research",
        "live_unchanged": "day_baseline_all_pass_v1_size_1_5",
        "live_promoted": False,
        "research_branch": "expanded_binance_us_spot",
        "target_monthly_usd": TARGET_MONTHLY,
        "universe_scan": {
            "total_pairs_scanned": universe.get("total_pairs_scanned"),
            "pairs_accepted": universe.get("pairs_accepted"),
            "pairs_rejected": universe.get("pairs_rejected"),
            "rejection_reason_counts": reject_summary,
            "summary": universe.get("summary"),
            "dataset_symbols_used": len(research_syms),
            "max_dataset_symbols_cap": MAX_DATASET_SYMBOLS,
        },
        "dataset": {
            "row_count": len(rows),
            "day_rows": len(day_rows),
            "scalp_rows": len(scalp_rows),
            "base_features": 145,
            "extra_universe_features": list(UNIVERSE_EXTRA_NAMES),
            "total_feature_dim": len(FEATURE_NAMES),
            "train_window_days": TRAIN_DAYS,
            "validation_window_days": VAL_DAYS,
            "test_window_days": TEST_DAYS,
            "train_rows": len(train),
            "validation_rows": len(val),
            "test_rows": len(test),
            "label_primary": "hit_40bp_before_40bp",
        },
        "symbol_rankings": symbol_rankings[:25],
        "discovered_buckets": all_buckets[:20],
        "best_expanded_day_candidate": best_day,
        "best_expanded_scalp_candidate": best_scalp,
        "best_combined_result": {
            **combined_m,
            "all_pass": accepted_flag,
            "target_met_500": combined_m.get("monthly_pnl_usd", 0) >= TARGET_MONTHLY,
            "accept_or_reject_reason": reject_reasons if reject_reasons else "pass",
        },
        "summary_table": [
            {
                "row": "locked_top_four_live_floor",
                "monthly_pnl_usd_on_25k": 87.0,
                "trades_per_month": 6.7,
                "target_met_500": False,
                "note": "unchanged safety floor — top-four neutral VWAP 1.5x",
            },
            {
                "row": "best_expanded_universe_DAY_candidate",
                "symbol": best_day.get("symbol"),
                "setup_name": best_day.get("setup_name"),
                **best_day_m,
                "all_pass": best_day.get("all_pass", False),
                "target_met_500": best_day.get("target_met_500", False),
                "accept_or_reject_reason": best_day.get("accept_or_reject_reason", reject_reasons),
            },
            {
                "row": "best_expanded_universe_scalp_candidate",
                **scalp_m,
                "setup_name": best_scalp.get("setup_name"),
                "target_met_500": scalp_m.get("monthly_pnl_usd", 0) >= TARGET_MONTHLY,
                "all_pass": False,
            },
            {
                "row": "best_combined_result",
                **combined_m,
                "symbols_scanned": universe.get("total_pairs_scanned"),
                "symbols_accepted": universe.get("pairs_accepted"),
                "symbols_rejected": universe.get("pairs_rejected"),
                "all_pass": accepted_flag,
                "target_met_500": combined_m.get("monthly_pnl_usd", 0) >= TARGET_MONTHLY,
                "accept_or_reject_reason": reject_reasons if reject_reasons else "pass",
            },
        ],
        "target_met_500": combined_m.get("monthly_pnl_usd", 0) >= TARGET_MONTHLY,
        "any_pattern_promoted": False,
        "conclusion": ("top_four_exhausted; universe_expansion_research_complete_no_promotable_edge" if not accepted_flag else "candidate_requires_full_execution_replay_validation"),
    }

    OUT_PATH.write_text(json.dumps(report, indent=2, default=str))
    print(
        json.dumps(
            {
                "scanned": universe.get("total_pairs_scanned"),
                "accepted": universe.get("pairs_accepted"),
                "rows": len(rows),
                "best_day_monthly": best_day_m.get("monthly_pnl_usd"),
                "target_met_500": report["target_met_500"],
                "out": str(OUT_PATH),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
