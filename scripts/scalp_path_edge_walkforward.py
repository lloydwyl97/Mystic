#!/usr/bin/env python3
"""Path-target walk-forward. Does not deploy. Does not use rejected 5m terminal model."""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backend.services.binance_scalp.forward_net_predictor import (
    GAP_BARS,
    WINDOW_BARS,
    chronological_folds,
    effective_sample_report,
    predict_linear,
    predict_logistic,
)
from backend.services.binance_scalp.historical_forensic import _ohlcv_symbol, load_ohlcv
from backend.services.binance_scalp.path_outcomes import DEFAULT_COST, HORIZONS_MIN, all_horizon_path_labels
from backend.services.binance_scalp.reconstructable_features import FEATURE_GROUPS, FEATURE_KEYS, reconstructable_features
from backend.services.binance_scalp.scalp_setup_measurements import measure_all_setups
from backend.services.binance_scalp.strategy_module_replay import LOOKBACK, _mom_from_bars, _snapshot

SYMBOLS = ("BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT")
REJECTED_BASELINE = {
    "version": "scalp_forward_net_v1",
    "accepted": False,
    "linear_oos_ev_corr": -0.014,
    "tree_rank_corr": 0.115,
    "projected_move_corr": 0.103,
    "rank_always_buy_wr": 0.2054,
    "rank_always_buy_expectancy": -0.000616,
    "hold_expectancy": 0.0,
}


def _corr(xs: np.ndarray, ys: np.ndarray) -> float | None:
    if len(xs) < 10 or float(np.std(xs)) < 1e-15 or float(np.std(ys)) < 1e-15:
        return None
    return round(float(np.corrcoef(xs, ys)[0, 1]), 4)


def _econ(realized: np.ndarray) -> dict[str, Any]:
    n = len(realized)
    if n == 0:
        return {"n": 0, "wr": None, "net": 0.0, "expectancy": None, "pf": None}
    wins = realized[realized > 0]
    losses = realized[realized <= 0]
    loss_sum = float(np.abs(losses).sum()) if len(losses) else 0.0
    win_sum = float(wins.sum()) if len(wins) else 0.0
    pf = (win_sum / loss_sum) if loss_sum > 0 else (None if len(wins) == 0 else 99.0)
    return {
        "n": n,
        "wr": round(float((realized > 0).mean()), 4),
        "net": round(float(realized.sum()), 6),
        "expectancy": round(float(realized.mean()), 6),
        "pf": None if pf is None else round(float(pf), 4),
        "positive_net_freq": round(float((realized > 0).mean()), 4),
    }


def _quartile(pred: np.ndarray, realized: np.ndarray) -> dict[str, Any] | None:
    if len(pred) < 20:
        return None
    order = np.argsort(pred)
    q = len(pred) // 4
    d = max(1, len(pred) // 10)
    bot, top = realized[order[:q]], realized[order[-q:]]
    bot_d, top_d = realized[order[:d]], realized[order[-d:]]
    return {
        "bottom_quartile": {"n": len(bot), "mean": round(float(bot.mean()), 6), "wr": round(float((bot > 0).mean()), 4)},
        "top_quartile": {"n": len(top), "mean": round(float(top.mean()), 6), "wr": round(float((top > 0).mean()), 4)},
        "bottom_decile": {"n": len(bot_d), "mean": round(float(bot_d.mean()), 6)},
        "top_decile": {"n": len(top_d), "mean": round(float(top_d.mean()), 6)},
        "top_minus_bottom_q": round(float(top.mean() - bot.mean()), 6),
    }


def _fit_predict(x_tr: np.ndarray, y_tr: np.ndarray, x_te: np.ndarray) -> np.ndarray:
    if len(x_tr) < 20 or float(np.std(y_tr)) < 1e-15:
        return np.zeros(len(x_te))
    mean = x_tr.mean(axis=0)
    scale = np.where(x_tr.std(axis=0) < 1e-12, 1.0, x_tr.std(axis=0))
    design = np.column_stack([(x_tr - mean) / scale, np.ones(len(x_tr))])
    coef, *_ = np.linalg.lstsq(design, y_tr, rcond=None)
    return predict_linear(x_te, mean, scale, coef[:-1], float(coef[-1]))


def _tree(x_tr, y_tr, x_te):
    from sklearn.tree import DecisionTreeRegressor

    if len(x_tr) < 30:
        return np.zeros(len(x_te))
    m = DecisionTreeRegressor(max_depth=3, min_samples_leaf=40, random_state=0)
    m.fit(x_tr, y_tr)
    return m.predict(x_te)


def _rf(x_tr, y_tr, x_te):
    try:
        from sklearn.ensemble import RandomForestRegressor
    except Exception:
        return None
    if len(x_tr) < 40:
        return np.zeros(len(x_te))
    m = RandomForestRegressor(n_estimators=40, max_depth=4, min_samples_leaf=40, random_state=0, n_jobs=1)
    m.fit(x_tr, y_tr)
    return m.predict(x_te)


def _gb(x_tr, y_tr, x_te):
    try:
        from sklearn.ensemble import GradientBoostingRegressor
    except Exception:
        return None
    if len(x_tr) < 40:
        return np.zeros(len(x_te))
    m = GradientBoostingRegressor(n_estimators=40, max_depth=2, min_samples_leaf=40, random_state=0)
    m.fit(x_tr, y_tr)
    return m.predict(x_te)


def _policy(rows, pred, y_key: str) -> dict[str, Any]:
    groups: dict[int, list[int]] = defaultdict(list)
    for i, row in enumerate(rows):
        groups[int(row["epoch"] // 60)].append(i)
    model_nets = []
    buys = 0
    holds = 0
    by_sym = {s: [] for s in SYMBOLS}
    for idxs in groups.values():
        best = max(idxs, key=lambda i: float(pred[i]))
        if float(pred[best]) > 0:
            net = float(rows[best]["path"][y_key] or 0.0)
            model_nets.append(net)
            buys += 1
            by_sym[rows[best]["symbol"]].append(net)
        else:
            model_nets.append(0.0)
            holds += 1
    arr = np.asarray(model_nets)
    return {
        "cycles": len(groups),
        "buy_count": buys,
        "hold_count": holds,
        "buy_pct": round(buys / len(groups), 4) if groups else None,
        "econ": _econ(arr),
        "beats_hold": bool(float(arr.sum()) > 0),
        "by_symbol": {s: _econ(np.asarray(v, dtype=float)) for s, v in by_sym.items()},
    }


def build_rows(db_path: str, *, step: int = 1) -> list[dict[str, Any]]:
    from dataclasses import replace

    from backend.services.binance_scalp.config import ScalpConfig
    from backend.services.binance_scalp.economics import ScalpEconomics
    from backend.services.binance_scalp.strategies.base import StrategyMarketContext

    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    try:
        bars_by = load_ohlcv(conn)
    finally:
        conn.close()
    btc = bars_by.get(_ohlcv_symbol("BTCUSDT"), [])
    btc_by_epoch = {}
    for b in btc:
        ts = b["ts"]
        btc_by_epoch[int(ts.timestamp() // 60)] = float(b["close"])
    rows = []
    for sym in SYMBOLS:
        raw = bars_by.get(_ohlcv_symbol(sym), [])
        if len(raw) < LOOKBACK + 21:
            continue
        for i in range(LOOKBACK, len(raw) - 20, max(1, step)):
            window = raw[i - LOOKBACK : i + 1]
            mid = float(window[-1]["close"])
            if mid <= 0:
                continue
            ts = window[-1]["ts"]
            epoch = ts.timestamp() if hasattr(ts, "timestamp") else float(ts)
            minute = int(epoch // 60)
            btc_now = btc_by_epoch.get(minute)
            btc_prev = btc_by_epoch.get(minute - 5)
            btc_ret = ((btc_now - btc_prev) / btc_prev) if btc_now and btc_prev and btc_prev > 0 else 0.0
            snap = replace(_snapshot(sym, mid, window), order_book_imbalance=0.0, book_source="replay_no_book")
            ctx = StrategyMarketContext(
                symbol=sym,
                snap=snap,
                mom=_mom_from_bars(window),
                bars_1m=window,
                econ=ScalpEconomics.from_env(),
                config=ScalpConfig.from_env(),
                notional_usd=50.0,
            )
            meas = measure_all_setups(ctx)
            proj = 0.0
            for block in meas.values():
                if isinstance(block, dict) and block.get("projected_move"):
                    proj = max(proj, float(block["projected_move"]))
            feats = reconstructable_features(
                window,
                btc_ret_5=btc_ret,
                market_vol_5=abs(btc_ret),
                ts=ts if hasattr(ts, "hour") else None,
                projected_move=proj,
            )
            paths = all_horizon_path_labels(mid, raw[i + 1 : i + 21], cost_pct=DEFAULT_COST)
            rows.append(
                {
                    "symbol": sym,
                    "epoch": epoch,
                    "features": feats,
                    "path": {h: paths[h] for h in HORIZONS_MIN},
                    "projected_move": proj,
                }
            )
    rows.sort(key=lambda r: (r["epoch"], r["symbol"]))
    return rows


def _downsample(rows):
    picked = {}
    for row in rows:
        key = (row["symbol"], int(row["epoch"] // (WINDOW_BARS * 60)))
        picked.setdefault(key, row)
    return sorted(picked.values(), key=lambda r: (r["epoch"], r["symbol"]))


def _xy(rows, names, horizon, target):
    x = np.asarray([[float(r["features"].get(n) or 0.0) for n in names] for r in rows], dtype=float)
    y = np.asarray([float(r["path"][horizon].get(target) or 0.0) for r in rows], dtype=float)
    return x, y


def eval_target(train, valid, test, names, horizon, target) -> dict[str, Any]:
    x_tr, y_tr = _xy(train, names, horizon, target)
    x_va, y_va = _xy(valid, names, horizon, target)
    x_te, y_te = _xy(test, names, horizon, target)
    lin = _fit_predict(x_tr, y_tr, x_te)
    lin_va = _fit_predict(x_tr, y_tr, x_va)
    tree = _tree(x_tr, y_tr, x_te)
    rf = _rf(x_tr, y_tr, x_te)
    gb = _gb(x_tr, y_tr, x_te)
    proj = np.asarray([r["projected_move"] for r in test], dtype=float)
    models = {
        "linear": {"corr": _corr(lin, y_te), "valid_corr": _corr(lin_va, y_va), "quartiles": _quartile(lin, y_te)},
        "tree": {"corr": _corr(tree, y_te), "quartiles": _quartile(tree, y_te)},
        "projected_move": {"corr": _corr(proj, y_te), "quartiles": _quartile(proj, y_te)},
    }
    if rf is not None:
        models["random_forest"] = {"corr": _corr(rf, y_te), "quartiles": _quartile(rf, y_te)}
    if gb is not None:
        models["gradient_boosting"] = {"corr": _corr(gb, y_te), "quartiles": _quartile(gb, y_te)}
    # policy uses linear predicted target as BUY EV vs HOLD=0
    policy_rows = []
    for r, _p in zip(test, lin, strict=False):
        item = dict(r)
        item["path"] = {target: r["path"][horizon].get("target_d_net" if target == "target_d_net" else "terminal_net")}
        policy_rows.append(item)
    # For MFE targets, policy realized should be that target, not terminal.
    realized_key = "terminal_net" if target.startswith("terminal") else target
    pol_rows = []
    for r in test:
        pol_rows.append({"epoch": r["epoch"], "symbol": r["symbol"], "path": {realized_key: r["path"][horizon].get(realized_key)}})
    policy = _policy(pol_rows, lin, realized_key)
    by_sym = {}
    for sym in SYMBOLS:
        idx = [i for i, r in enumerate(test) if r["symbol"] == sym]
        if len(idx) < 8:
            by_sym[sym] = {"n": len(idx)}
            continue
        by_sym[sym] = {"n": len(idx), "corr": _corr(lin[idx], y_te[idx]), "quartiles": _quartile(lin[idx], y_te[idx])}
    # time stability: split test in half
    mid = len(test) // 2
    stability = {
        "test_first_half_corr": _corr(lin[:mid], y_te[:mid]) if mid > 10 else None,
        "test_second_half_corr": _corr(lin[mid:], y_te[mid:]) if len(test) - mid > 10 else None,
        "train_corr": _corr(_fit_predict(x_tr, y_tr, x_tr), y_tr),
        "valid_corr": _corr(lin_va, y_va),
        "test_corr": _corr(lin, y_te),
    }
    return {
        "n_train": len(train),
        "n_valid": len(valid),
        "n_test": len(test),
        "mean_realized": round(float(y_te.mean()), 6),
        "positive_freq": round(float((y_te > 0).mean()), 4),
        "models": models,
        "policy_linear_vs_hold": policy,
        "by_symbol": by_sym,
        "stability": stability,
        "path_stats": {
            "mean_mfe": round(float(np.mean([r["path"][horizon]["mfe"] for r in test])), 6),
            "mean_mae": round(float(np.mean([r["path"][horizon]["mae"] for r in test])), 6),
            "mean_time_to_mfe": round(float(np.nanmean([r["path"][horizon]["time_to_mfe"] or np.nan for r in test])), 3),
            "mean_time_to_mae": round(float(np.nanmean([r["path"][horizon]["time_to_mae"] or np.nan for r in test])), 3),
            "mfe_first_rate": round(float(np.mean([1.0 if r["path"][horizon]["path_order"] == "MFE_FIRST" else 0.0 for r in test])), 4),
            "mae_first_rate": round(float(np.mean([1.0 if r["path"][horizon]["path_order"] == "MAE_FIRST" else 0.0 for r in test])), 4),
            "exec_profit_rate": round(float(np.mean([1.0 if r["path"][horizon]["executable_profit_occurred"] else 0.0 for r in test])), 4),
            "profit_before_adverse_rate": round(float(np.mean([1.0 if r["path"][horizon]["profit_before_adverse"] else 0.0 for r in test])), 4),
        },
    }


def ablate(train, test, names, horizon, target) -> dict[str, Any]:
    x_tr, y_tr = _xy(train, names, horizon, target)
    x_te, y_te = _xy(test, names, horizon, target)
    full = _corr(_fit_predict(x_tr, y_tr, x_te), y_te)
    out = {"full": full}
    for group, keys in FEATURE_GROUPS.items():
        keep = tuple(k for k in names if k not in keys)
        if len(keep) < 3:
            continue
        c = _corr(_fit_predict(*_xy(train, keep, horizon, target), _xy(test, keep, horizon, target)[0]), y_te)
        out[group] = {"corr_without": c, "delta_vs_full": None if full is None or c is None else round(full - c, 4)}
    return out


def event_analysis(test, horizon) -> dict[str, Any]:
    y_term = np.asarray([r["path"][horizon]["terminal_net"] for r in test], dtype=float)
    y_mfe = np.asarray([r["path"][horizon]["executable_mfe_net"] for r in test], dtype=float)
    out = {}
    for key in FEATURE_GROUPS["event_strength"]:
        xs = np.asarray([r["features"].get(key) or 0.0 for r in test], dtype=float)
        out[key] = {"corr_terminal": _corr(xs, y_term), "corr_exec_mfe": _corr(xs, y_mfe)}
    return out


def accept(ev: dict[str, Any]) -> tuple[bool, str]:
    pol = ev.get("policy_linear_vs_hold") or {}
    corr = ((ev.get("models") or {}).get("linear") or {}).get("corr")
    q = ((ev.get("models") or {}).get("linear") or {}).get("quartiles") or {}
    top = (q.get("top_quartile") or {}).get("mean")
    bot = (q.get("bottom_quartile") or {}).get("mean")
    if corr is None or corr <= 0:
        return False, "oos prediction vs realized correlation is not positive"
    if top is None or bot is None or top <= bot:
        return False, "top quartile does not exceed bottom quartile"
    if not pol.get("beats_hold"):
        return False, "policy does not beat HOLD"
    if (pol.get("buy_count") or 0) < 30:
        return False, f"BUY coverage too small ({pol.get('buy_count')})"
    if (pol.get("econ") or {}).get("expectancy", 0) <= 0:
        return False, "selected BUY expectancy is not positive"
    return True, "beats HOLD with ordered OOS economics and coverage"


def run(args) -> dict[str, Any]:
    raw = build_rows(args.ohlcv_db, step=args.step)
    rows = _downsample(raw)
    ess = effective_sample_report(
        [r["path"][5]["terminal_net"] for r in raw],
        [r["epoch"] for r in raw],
        [r["symbol"] for r in raw],
    )
    folds = chronological_folds(len(rows), gap=GAP_BARS)
    if not folds:
        return {"error": "insufficient rows"}
    tr, va, te = folds[0]
    train, valid, test = [rows[i] for i in tr], [rows[i] for i in va], [rows[i] for i in te]
    windows = {
        "train": {
            "n": len(train),
            "start": datetime.fromtimestamp(train[0]["epoch"], tz=timezone.utc).isoformat(),
            "end": datetime.fromtimestamp(train[-1]["epoch"], tz=timezone.utc).isoformat(),
        },
        "valid": {
            "n": len(valid),
            "start": datetime.fromtimestamp(valid[0]["epoch"], tz=timezone.utc).isoformat(),
            "end": datetime.fromtimestamp(valid[-1]["epoch"], tz=timezone.utc).isoformat(),
        },
        "test": {
            "n": len(test),
            "start": datetime.fromtimestamp(test[0]["epoch"], tz=timezone.utc).isoformat(),
            "end": datetime.fromtimestamp(test[-1]["epoch"], tz=timezone.utc).isoformat(),
        },
        "gap_bars": GAP_BARS,
    }
    report: dict[str, Any] = {
        "rejected_5m_terminal_baseline": REJECTED_BASELINE,
        "dataset": {
            "raw": len(raw),
            "windows": len(rows),
            "ess": ess,
            "features": list(FEATURE_KEYS),
            "live_only_excluded": ["orderbook_imbalance", "fabricated_replay_book"],
        },
        "windows": windows,
        "leakage_controls": [
            "features from bars at or before entry close only",
            "path MFE/MAE/time-to-target are labels only",
            "chronological split with 20-bar gap",
            "no fabricated book features",
        ],
    }
    comparisons = {}
    for h in (5, 10, 20):
        comparisons[f"{h}m"] = {
            "terminal_net": eval_target(train, valid, test, FEATURE_KEYS, h, "terminal_net"),
            "executable_mfe_net": eval_target(train, valid, test, FEATURE_KEYS, h, "executable_mfe_net"),
            "profit_before_adverse": eval_target(train, valid, test, FEATURE_KEYS, h, "profit_before_adverse"),
            "target_c": eval_target(train, valid, test, FEATURE_KEYS, h, "target_c"),
            "target_d_net": eval_target(train, valid, test, FEATURE_KEYS, h, "target_d_net"),
        }
    report["horizon_results"] = comparisons
    report["ablation_20m_exec_mfe"] = ablate(train, test, FEATURE_KEYS, 20, "executable_mfe_net")
    report["ablation_20m_terminal"] = ablate(train, test, FEATURE_KEYS, 20, "terminal_net")
    report["event_strength_test"] = {str(h): event_analysis(test, h) for h in (5, 10, 20)}
    # choose best linear policy among investigated targets
    candidates = []
    for h, block in comparisons.items():
        for tname, ev in block.items():
            ok, reason = accept(ev)
            candidates.append(
                {
                    "horizon": h,
                    "target": tname,
                    "accepted": ok,
                    "reason": reason,
                    "corr": ev["models"]["linear"]["corr"],
                    "buys": ev["policy_linear_vs_hold"]["buy_count"],
                    "expectancy": ev["policy_linear_vs_hold"]["econ"]["expectancy"],
                }
            )
    report["acceptance_candidates"] = candidates
    accepted = [c for c in candidates if c["accepted"]]
    report["model_accepted"] = bool(accepted)
    report["why"] = accepted[0] if accepted else "no target/horizon beat HOLD with ordered OOS economics and n>=30 BUYs"
    report["ocean_deployment"] = "not deployed — keep HOLD-as-action; rejected scalp_forward_net_v1 stays rejected"
    return report


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--ohlcv-db", default="/tmp/ocean_forward_net.db")
    p.add_argument("--step", type=int, default=1)
    p.add_argument("--out", default="/tmp/scalp_path_edge_report.json")
    args = p.parse_args()
    report = run(args)
    text = json.dumps(report, indent=2, default=str)
    Path(args.out).write_text(text)
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
