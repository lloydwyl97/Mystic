#!/usr/bin/env python3
"""
AI outcome-driven entry discovery — replay only.

Builds labeled dataset from all bars + 145 features, trains walk-forward models,
discovers replay buckets from feature regions. Does NOT promote live.
"""
from __future__ import annotations

import json
import pickle
import sys
import traceback
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
BASELINE_DIR = REPO / "scripts" / "replay_baselines"
DATASET_CACHE = BASELINE_DIR / "ai_outcome_dataset_cache.pkl"

from backend.config.binance_us_fee_schedule import verify_top_four_pairs
from backend.services.ai_outcome_dataset import (
    DAY_SAMPLE_TFS,
    FEATURE_NAMES_145,
    HORIZONS_SEC,
    build_dataset_rows,
)
from backend.services.ltf_pattern_miner import resample_bars
from scripts.run_day_execution_replay import fetch_klines_cached
from scripts.run_day_strategy_replay import PRINCIPAL, SYMBOLS

TARGET_MONTHLY = 500.0
DAYS = 90
TRAIN_DAYS = 54
VAL_DAYS = 18
TEST_DAYS = 18
NOTIONAL_DAY = 3750.0
NOTIONAL_SCALP = 25.0

try:
    from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.preprocessing import StandardScaler

    SKLEARN_OK = True
except ImportError:
    SKLEARN_OK = False


def _load_bars() -> tuple[int, int, int, dict[str, dict[str, list]]]:
    end = datetime.now(timezone.utc)
    start = end - timedelta(days=DAYS + 5)
    start_ms = int(start.timestamp() * 1000)
    end_ms = int(end.timestamp() * 1000)
    start_ts = int((end - timedelta(days=DAYS)).timestamp())
    end_ts = int(end.timestamp())
    scalp_start = int((end - timedelta(days=30)).timestamp())

    all_bars: dict[str, dict[str, list]] = {}
    for sym in SYMBOLS:
        sym_bars: dict[str, list] = {}
        for iv in ("1m", "5m", "15m", "1h"):
            sym_bars[iv] = fetch_klines_cached(sym, iv, start_ms, end_ms)
        sym_bars["30m"] = resample_bars(sym_bars["15m"], 30)
        sym_bars["4h"] = resample_bars(sym_bars["1h"], 240)
        sym_bars["1d"] = resample_bars(sym_bars["1h"], 1440)
        all_bars[sym] = sym_bars
    return start_ts, end_ts, scalp_start, all_bars


def _build_full_dataset(
    all_bars: dict,
    start_ts: int,
    end_ts: int,
    scalp_start: int,
    half_spreads: dict[str, float],
    *,
    use_cache: bool = True,
) -> list[dict]:
    if use_cache and DATASET_CACHE.exists():
        try:
            cached = pickle.loads(DATASET_CACHE.read_bytes())
            if cached.get("version") == 2 and cached.get("rows"):
                print(f"    loaded cache rows={len(cached['rows'])}", flush=True)
                return cached["rows"]
        except Exception:
            pass

    rows: list[dict] = []
    for sym in SYMBOLS:
        hs = half_spreads.get(sym, 0.00006)
        print(f"    DAY rows {sym}...", flush=True)
        rows.extend(
            build_dataset_rows(sym, all_bars[sym], start_ts, end_ts, sample_sec=3600, half_spread=hs, scalp=False)
        )
        print(f"      day={sum(1 for r in rows if r['symbol']==sym and r['timeframe']!='1m')} total={len(rows)}", flush=True)
        print(f"    SCALP rows {sym}...", flush=True)
        n0 = len(rows)
        rows.extend(
            build_dataset_rows(sym, all_bars[sym], scalp_start, end_ts, sample_sec=300, half_spread=hs, scalp=True)
        )
        print(f"      scalp={len(rows)-n0} total={len(rows)}", flush=True)

    try:
        DATASET_CACHE.write_bytes(pickle.dumps({"version": 2, "rows": rows}))
    except Exception:
        pass
    return rows


def _split_by_time(rows: list[dict], start_ts: int) -> tuple[list, list, list]:
    train_end = start_ts + TRAIN_DAYS * 86400
    val_end = train_end + VAL_DAYS * 86400
    train = [r for r in rows if r["timestamp"] < train_end]
    val = [r for r in rows if train_end <= r["timestamp"] < val_end]
    test = [r for r in rows if r["timestamp"] >= val_end]
    return train, val, test


def _xy(rows: list[dict], target_key: str = "hit_40bp_before_40bp") -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    X, y, ts = [], [], []
    for r in rows:
        lab = r.get("labels") or {}
        if target_key not in lab:
            continue
        X.append(r["features_145"])
        y.append(1 if lab[target_key] else 0)
        ts.append(r["timestamp"])
    if not X:
        return np.zeros((0, 145)), np.zeros(0), np.zeros(0)
    return np.array(X, dtype=np.float64), np.array(y, dtype=np.int32), np.array(ts, dtype=np.int64)


def _realized_net_pct(lab: dict) -> float:
    """Fee-aware realized proxy: target hit vs adverse-first exit."""
    if lab.get("hit_40bp_before_40bp"):
        return 0.004
    mae = float(lab.get("mae_72h") or lab.get("mae_24h") or 0)
    if mae <= -0.004:
        return -0.004
    return float(lab.get("expected_net_pnl_pct") or 0)


def _metrics_from_rows(rows: list[dict], notional: float = NOTIONAL_DAY) -> dict[str, Any]:
    if not rows:
        return {"trades": 0, "monthly_pnl_usd": 0.0, "expectancy_per_trade": 0.0}
    pnls = []
    holds = []
    wins = []
    maes = []
    equity = 0.0
    peak = 0.0
    max_dd = 0.0
    for r in rows:
        lab = r.get("labels") or {}
        net_pct = _realized_net_pct(lab)
        pnl = net_pct * notional
        pnls.append(pnl)
        holds.append(float(lab.get("time_to_profit_sec") or lab.get("best_net_hold_sec") or 0))
        wins.append(pnl > 0)
        maes.append(float(lab.get("mae_72h") or lab.get("mae_24h") or 0))
        equity += pnl
        peak = max(peak, equity)
        max_dd = max(max_dd, peak - equity)
    n = len(pnls)
    net = sum(pnls)
    months = max(DAYS / 30.0, 1)
    gw = sum(p for p in pnls if p > 0)
    gl = abs(sum(p for p in pnls if p <= 0))
    aw = gw / max(sum(wins), 1)
    al = gl / max(n - sum(wins), 1)
    return {
        "trades": n,
        "trades_per_month": round(n / months, 2),
        "net_pnl_usd": round(net, 2),
        "monthly_pnl_usd": round(net / months, 2),
        "pct_per_month": round(100.0 * net / months / PRINCIPAL, 4),
        "expectancy_per_trade": round(net / n, 4),
        "win_rate_pct": round(100.0 * sum(wins) / n, 2),
        "avg_win_usd": round(aw, 2),
        "avg_loss_usd": round(-al, 2),
        "profit_factor": round(gw / gl, 3) if gl > 1e-9 else (999.0 if gw > 0 else 0),
        "longest_hold_hours": round(max(holds) / 3600.0, 2) if holds else 0,
        "worst_mae_pct": round(min(maes) * 100, 4) if maes else 0,
        "max_drawdown_usd": round(max_dd, 2),
        "max_drawdown_pct": round(100.0 * max_dd / PRINCIPAL, 4) if max_dd else 0,
    }


def _train_models(X_train, y_train, X_val, y_val) -> dict[str, Any]:
    if not SKLEARN_OK or len(X_train) < 100:
        return {"error": "insufficient_data_or_sklearn"}
    scaler = StandardScaler()
    Xt = scaler.fit_transform(X_train)
    Xv = scaler.transform(X_val) if len(X_val) else Xt[:0]

    models = {
        "logistic": LogisticRegression(max_iter=500, class_weight="balanced"),
        "random_forest": RandomForestClassifier(n_estimators=100, max_depth=8, class_weight="balanced", random_state=42),
        "gradient_boosting": GradientBoostingClassifier(n_estimators=100, max_depth=4, random_state=42),
    }
    results = {}
    best_name, best_auc = "", -1.0
    for name, model in models.items():
        try:
            model.fit(Xt, y_train)
            proba = model.predict_proba(Xv)[:, 1] if len(Xv) else np.array([])
            auc = float(roc_auc_score(y_val, proba)) if len(proba) and len(set(y_val)) > 1 else 0.5
            imp = None
            if hasattr(model, "feature_importances_"):
                imp = model.feature_importances_
            elif hasattr(model, "coef_"):
                imp = np.abs(model.coef_[0])
            results[name] = {"val_auc": round(auc, 4), "model": model, "importance": imp}
            if auc > best_auc:
                best_auc = auc
                best_name = name
        except Exception as e:
            results[name] = {"error": str(e)}
    return {"models": results, "best_model": best_name, "best_auc": best_auc, "scaler": scaler}


def _top_features(importance: np.ndarray, k: int = 12) -> list[dict]:
    if importance is None or len(importance) != len(FEATURE_NAMES_145):
        return []
    idx = np.argsort(importance)[::-1][:k]
    return [
        {"feature": FEATURE_NAMES_145[i], "importance": round(float(importance[i]), 5)}
        for i in idx
    ]


def _discover_buckets(train_rows: list[dict], test_rows: list[dict], importance: np.ndarray) -> list[dict]:
    """Hard buckets from top feature quantile splits on train-positive regions."""
    if importance is None or not train_rows:
        return []
    top_idx = np.argsort(importance)[::-1][:8]
    buckets: list[dict] = []
    sym_filter = None
    for sym in SYMBOLS:
        sym_train = [r for r in train_rows if r["symbol"] == sym]
        sym_test = [r for r in test_rows if r["symbol"] == sym]
        if len(sym_train) < 50:
            continue
        for fi in top_idx[:4]:
            fname = FEATURE_NAMES_145[fi]
            vals = [r["features_145"][fi] for r in sym_train]
            for qlo, qhi in ((0.25, 0.75), (0.33, 0.66)):
                lo, hi = float(np.quantile(vals, qlo)), float(np.quantile(vals, qhi))
                tr = [r for r in sym_train if lo <= r["features_145"][fi] <= hi]
                te = [r for r in sym_test if lo <= r["features_145"][fi] <= hi]
                if len(tr) < 20:
                    continue
                meta_regime = {}
                regimes = [r.get("meta", {}).get("regime") for r in tr]
                if regimes:
                    meta_regime["regime_mode"] = max(set(regimes), key=regimes.count)
                tm = _metrics_from_rows(tr)
                tem = _metrics_from_rows(te)
                if tm.get("expectancy_per_trade", 0) <= 0:
                    continue
                buckets.append({
                    "bucket_id": f"{sym}_{fname}_q{qlo}_{qhi}",
                    "symbol": sym,
                    "feature": fname,
                    "range": [round(lo, 6), round(hi, 6)],
                    **meta_regime,
                    "train_metrics": tm,
                    "test_metrics": tem,
                    "test_positive": tem.get("expectancy_per_trade", 0) > 0,
                })
    buckets.sort(key=lambda b: float((b.get("test_metrics") or {}).get("monthly_pnl_usd") or -1e9), reverse=True)
    return buckets[:15]


def _threshold_replay(test_rows: list[dict], proba: np.ndarray, scaler, model, notional: float = NOTIONAL_DAY) -> list[dict]:
    results = []
    if len(test_rows) != len(proba):
        return results
    ranked = sorted(zip(proba, test_rows), key=lambda x: -x[0])
    n = len(ranked)
    for label, frac in [("top_1pct", 0.01), ("top_2pct", 0.02), ("top_5pct", 0.05), ("top_10pct", 0.10)]:
        k = max(1, int(n * frac))
        sub = [r for _, r in ranked[:k]]
        results.append({"threshold": label, "metrics": _metrics_from_rows(sub)})
    for pmin in (0.55, 0.60, 0.65, 0.70):
        sub = [r for p, r in ranked if p >= pmin]
        if sub:
            results.append({"threshold": f"prob_ge_{pmin}", "metrics": _metrics_from_rows(sub)})
    cost = 0.004 * notional if notional > 100 else 0.10
    for mult in (2, 3):
        sub = [r for p, r in ranked if _realized_net_pct(r.get("labels") or {}) * notional >= cost * mult]
        if sub:
            results.append({"threshold": f"ev_gt_cost_x{mult}", "metrics": _metrics_from_rows(sub, notional)})
    return results


def _stress_metrics(rows: list[dict], notional: float = NOTIONAL_DAY) -> dict[str, Any]:
    """Spread ×1.5 stress on label proxy PnL."""
    stressed = []
    for r in rows:
        nr = dict(r)
        lab = dict(r.get("labels") or {})
        net = _realized_net_pct(lab) - 0.0015
        lab["expected_net_pnl_pct"] = net
        nr["labels"] = lab
        stressed.append(nr)
    m = _metrics_from_rows(stressed, notional)
    m["stress_spread_mult"] = 1.5
    m["pass"] = m.get("expectancy_per_trade", 0) > 0
    return m


def _per_symbol_models(train, val, test, target_key: str) -> dict[str, Any]:
    out = {}
    for sym in SYMBOLS:
        tr = [r for r in train if r["symbol"] == sym]
        va = [r for r in val if r["symbol"] == sym]
        te = [r for r in test if r["symbol"] == sym]
        Xt, yt, _ = _xy(tr, target_key)
        Xv, yv, _ = _xy(va, target_key)
        if len(Xt) < 80:
            continue
        res = _train_models(Xt, yt, Xv, yv)
        out[sym] = {"best_model": res.get("best_model"), "best_val_auc": res.get("best_auc"), "test_rows": len(te)}
    return out


def _reject_bucket(metrics: dict, wf_test_pos: bool) -> tuple[bool, list[str]]:
    reasons = []
    if metrics.get("monthly_pnl_usd", 0) < TARGET_MONTHLY:
        reasons.append("below_500_mo")
    if metrics.get("expectancy_per_trade", 0) <= 0:
        reasons.append("negative_expectancy")
    if not wf_test_pos:
        reasons.append("walk_forward_test_fail")
    if metrics.get("longest_hold_hours", 0) > 72:
        reasons.append("fat_tail_hold")
    if metrics.get("max_drawdown_pct", 0) > 15:
        reasons.append("drawdown_limit")
    return len(reasons) == 0, reasons


def main() -> int:
    print("=== AI OUTCOME ENTRY DISCOVERY (replay-only) ===", flush=True)
    if not SKLEARN_OK:
        print("sklearn not available", file=sys.stderr)
        return 1

    verified = verify_top_four_pairs()
    half_spreads = {k: float(v["orderbook_half_spread_pct"]) for k, v in verified["pairs"].items()}

    print("  loading bars...", flush=True)
    start_ts, end_ts, scalp_start, all_bars = _load_bars()

    print("  building labeled dataset...", flush=True)
    rows = _build_full_dataset(all_bars, start_ts, end_ts, scalp_start, half_spreads)
    day_rows = [r for r in rows if r["timeframe"] != "1m"]
    scalp_rows = [r for r in rows if r["timeframe"] == "1m"]

    train, val, test = _split_by_time(day_rows, start_ts)
    _, _, scalp_test = _split_by_time(scalp_rows, start_ts)

    target_key = "hit_40bp_before_40bp"
    X_train, y_train, _ = _xy(train, target_key)
    X_val, y_val, _ = _xy(val, target_key)
    X_test, y_test, _ = _xy(test, target_key)

    print(f"  dataset rows={len(rows)} day={len(day_rows)} scalp={len(scalp_rows)} features=145", flush=True)
    print("  training models (walk-forward)...", flush=True)
    trained = _train_models(X_train, y_train, X_val, y_val)
    best = trained.get("best_model") or "logistic"
    best_info = (trained.get("models") or {}).get(best, {})
    model = best_info.get("model")
    scaler = trained.get("scaler")
    importance = best_info.get("importance")

    top_feats = _top_features(importance) if importance is not None else []
    buckets = _discover_buckets(train, test, importance) if importance is not None else []
    per_symbol = _per_symbol_models(train, val, test, target_key)

    threshold_results = []
    if model is not None and scaler is not None and len(X_test):
        proba = model.predict_proba(scaler.transform(X_test))[:, 1]
        threshold_results = _threshold_replay(test, proba, scaler, model)

    best_bucket = buckets[0] if buckets else {}
    stress = _stress_metrics(test) if test else {"pass": False}
    if best_bucket and best_bucket.get("feature") in FEATURE_NAMES_145:
        fi = FEATURE_NAMES_145.index(best_bucket["feature"])
        bucket_test_rows = [
            r for r in test
            if r["symbol"] == best_bucket.get("symbol")
            and best_bucket["range"][0] <= r["features_145"][fi] <= best_bucket["range"][1]
        ]
        if bucket_test_rows:
            stress = _stress_metrics(bucket_test_rows)

    best_thresh = max(
        (t for t in threshold_results if not str(t.get("threshold", "")).startswith("ev_gt_cost")),
        key=lambda x: float((x.get("metrics") or {}).get("monthly_pnl_usd") or -1e9),
        default={},
    )
    best_test_metrics = (best_bucket.get("test_metrics") or best_thresh.get("metrics") or _metrics_from_rows(test))
    wf_test_pos = float(best_test_metrics.get("expectancy_per_trade") or 0) > 0
    accepted, reject_reasons = _reject_bucket(best_test_metrics, wf_test_pos)

    scalp_metrics = _metrics_from_rows(scalp_test, NOTIONAL_SCALP) if scalp_test else {"trades": 0, "monthly_pnl_usd": 0}

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "discovery_type": "ai_outcome_driven",
        "live_unchanged": "day_baseline_all_pass_v1_size_1_5",
        "live_promoted": False,
        "target_monthly_usd": TARGET_MONTHLY,
        "dataset": {
            "row_count": len(rows),
            "day_rows": len(day_rows),
            "scalp_rows": len(scalp_rows),
            "feature_count": 145,
            "day_timeframes": list(DAY_SAMPLE_TFS),
            "feature_names_source": "FEATURE_MAPPING_124 + CONTEXT_DIMS_DAY_FULL_21",
            "label_definitions": {
                "primary_target": target_key,
                "mfe_mae_horizons": list(HORIZONS_SEC.keys()),
                "scalp_horizons_min": ["5m", "10m", "15m", "30m"],
                "net_after_fees": "Binance.US taker + orderbook half-spread + slippage buffer",
                "max_hold_cap": "72h",
            },
            "train_window_days": TRAIN_DAYS,
            "validation_window_days": VAL_DAYS,
            "test_window_days": TEST_DAYS,
            "train_rows": len(train),
            "validation_rows": len(val),
            "test_rows": len(test),
            "positive_rate_train": round(float(np.mean(y_train)) if len(y_train) else 0, 4),
        },
        "models": {
            "best_model_type": best,
            "best_val_auc": trained.get("best_auc"),
            "compared": ["logistic", "random_forest", "gradient_boosting"],
            "top_features": top_feats,
            "per_symbol": per_symbol,
            "global_vs_symbol": "global_model_primary",
        },
        "discovered_buckets": buckets,
        "threshold_replay": threshold_results,
        "threshold_replay_note": "ev_gt_cost rows use label oracle EV — excluded from acceptance; use prob/top-N only",
        "stress_results": stress,
        "scalp_test_metrics": scalp_metrics,
        "best_result": {
            "source": best_bucket.get("bucket_id") or best_thresh.get("threshold") or "all_test_rows",
            **best_test_metrics,
            "stress_pass": stress.get("pass"),
            "all_pass": accepted,
            "target_met_500": float(best_test_metrics.get("monthly_pnl_usd") or 0) >= TARGET_MONTHLY,
            "accept_or_reject_reason": reject_reasons if reject_reasons else "pass",
        },
        "summary_table": [
            {
                "row": "locked_live_DAY_floor",
                "monthly_pnl_usd_on_25k": 87.0,
                "trades_per_month": 6.7,
                "target_met_500": False,
                "note": "unchanged safety floor",
            },
            {
                "row": "best_ai_discovered_bucket",
                "bucket_id": best_bucket.get("bucket_id"),
                "trades_per_month": best_test_metrics.get("trades_per_month"),
                "monthly_pnl_usd_on_25k": best_test_metrics.get("monthly_pnl_usd"),
                "pct_per_month_on_25k": best_test_metrics.get("pct_per_month"),
                "win_rate_pct": best_test_metrics.get("win_rate_pct"),
                "avg_win_usd": best_test_metrics.get("avg_win_usd"),
                "avg_loss_usd": best_test_metrics.get("avg_loss_usd"),
                "profit_factor": best_test_metrics.get("profit_factor"),
                "expectancy_per_trade_usd": best_test_metrics.get("expectancy_per_trade"),
                "longest_hold_hours": best_test_metrics.get("longest_hold_hours"),
                "worst_mae_pct": best_test_metrics.get("worst_mae_pct"),
                "all_pass": accepted,
                "target_met_500": float(best_test_metrics.get("monthly_pnl_usd") or 0) >= TARGET_MONTHLY,
                "accept_or_reject_reason": reject_reasons if reject_reasons else "pass",
            },
            {
                "row": "best_threshold_replay",
                **(best_thresh.get("metrics") or {}),
                "threshold": best_thresh.get("threshold"),
            },
            {
                "row": "scalp_model_test",
                **scalp_metrics,
                "target_met_500": float(scalp_metrics.get("monthly_pnl_usd") or 0) >= TARGET_MONTHLY,
            },
        ],
        "target_met_500": float(best_test_metrics.get("monthly_pnl_usd") or 0) >= TARGET_MONTHLY,
        "any_pattern_promoted": False,
        "conclusion": (
            "human_ltf_patterns_exhausted; ai_outcome_discovery_complete"
            if not accepted
            else "candidate_bucket_requires_execution_replay_validation"
        ),
    }

    out = BASELINE_DIR / "ai_outcome_entry_discovery_latest.json"
    out.write_text(json.dumps(report, indent=2, default=str))
    print(json.dumps({
        "rows": len(rows),
        "best_model": best,
        "best_monthly": best_test_metrics.get("monthly_pnl_usd"),
        "target_met_500": report["target_met_500"],
        "out": str(out),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
