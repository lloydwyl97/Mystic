#!/usr/bin/env python3
"""Walk-forward SCALP forward-net predictor. Chronological only. No shuffle.

Trains on bar-reconstructable measurements + actual forward net after costs.
Does not train on BUY/HOLD/passed/rank labels as targets.
Does not deploy unless the model beats HOLD + current rank out of sample.
"""

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
    DEFAULT_COST,
    FEATURE_GROUPS,
    FEATURE_KEYS,
    GAP_BARS,
    MODEL_VERSION,
    PRIMARY_HORIZON,
    WINDOW_BARS,
    ForwardNetArtifact,
    chronological_folds,
    effective_sample_report,
    fit_artifact,
    flatten_measurements,
    path_labels,
    predict_linear,
    predict_logistic,
    save_artifact,
    vector_from_features,
)
from backend.services.binance_scalp.historical_forensic import _ohlcv_symbol, load_ohlcv
from backend.services.binance_scalp.scalp_setup_measurements import measure_all_setups
from backend.services.binance_scalp.strategy_module_replay import LOOKBACK, _mom_from_bars, _snapshot

SYMBOLS = ("BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT")
NOTIONAL = 50.0


def _corr(xs: np.ndarray, ys: np.ndarray) -> float | None:
    if len(xs) < 10:
        return None
    if float(np.std(xs)) < 1e-15 or float(np.std(ys)) < 1e-15:
        return None
    return round(float(np.corrcoef(xs, ys)[0, 1]), 4)


def _quartile(pred: np.ndarray, realized: np.ndarray) -> dict[str, Any] | None:
    if len(pred) < 20:
        return None
    order = np.argsort(pred)
    q = len(pred) // 4
    bot = realized[order[:q]]
    top = realized[order[-q:]]
    d = max(1, len(pred) // 10)
    bot_d = realized[order[:d]]
    top_d = realized[order[-d:]]
    return {
        "bottom_quartile": {"n": int(len(bot)), "mean_net": round(float(bot.mean()), 6), "wr": round(float((bot > 0).mean()), 4)},
        "top_quartile": {"n": int(len(top)), "mean_net": round(float(top.mean()), 6), "wr": round(float((top > 0).mean()), 4)},
        "bottom_decile": {"n": int(len(bot_d)), "mean_net": round(float(bot_d.mean()), 6), "wr": round(float((bot_d > 0).mean()), 4)},
        "top_decile": {"n": int(len(top_d)), "mean_net": round(float(top_d.mean()), 6), "wr": round(float((top_d > 0).mean()), 4)},
        "top_minus_bottom_q": round(float(top.mean() - bot.mean()), 6),
    }


def _buckets(pred: np.ndarray, realized: np.ndarray, n_buckets: int = 5) -> list[dict[str, Any]]:
    if len(pred) < n_buckets * 4:
        return []
    order = np.argsort(pred)
    edges = np.array_split(order, n_buckets)
    out = []
    means = []
    for i, idx in enumerate(edges):
        ys = realized[idx]
        means.append(float(ys.mean()) if len(ys) else 0.0)
        out.append(
            {
                "bucket": i + 1,
                "n": int(len(ys)),
                "mean_pred": round(float(pred[idx].mean()), 6) if len(idx) else None,
                "mean_realized": round(float(ys.mean()), 6) if len(ys) else None,
                "wr": round(float((ys > 0).mean()), 4) if len(ys) else None,
            }
        )
    mono = all(means[i] <= means[i + 1] + 1e-12 for i in range(len(means) - 1))
    return [{"monotonic_increasing": mono}, *out]


def _econ(realized: np.ndarray) -> dict[str, Any]:
    n = int(len(realized))
    if n == 0:
        return {"n": 0, "wr": None, "net": 0.0, "expectancy": None, "pf": None, "avg_win": None, "avg_loss": None, "mae": None}
    wins = realized[realized > 0]
    losses = realized[realized <= 0]
    win_sum = float(wins.sum()) if len(wins) else 0.0
    loss_sum = float(np.abs(losses).sum()) if len(losses) else 0.0
    pf = (win_sum / loss_sum) if loss_sum > 0 else (None if len(wins) == 0 else 99.0)
    return {
        "n": n,
        "wr": round(float((realized > 0).mean()), 4),
        "net": round(float(realized.sum()), 6),
        "expectancy": round(float(realized.mean()), 6),
        "pf": None if pf is None else round(float(pf), 4),
        "avg_win": None if len(wins) == 0 else round(float(wins.mean()), 6),
        "avg_loss": None if len(losses) == 0 else round(float(losses.mean()), 6),
        "mae": round(float(realized.min()), 6),
    }


def _calib(pred: np.ndarray, realized: np.ndarray, n_bins: int = 5) -> list[dict[str, Any]]:
    if len(pred) < 20:
        return []
    lo, hi = float(pred.min()), float(pred.max())
    if hi - lo < 1e-12:
        return [{"n": int(len(pred)), "mean_pred": round(lo, 6), "mean_realized": round(float(realized.mean()), 6)}]
    bins = np.linspace(lo, hi, n_bins + 1)
    out = []
    for i in range(n_bins):
        mask = (pred >= bins[i]) & (pred < bins[i + 1] if i < n_bins - 1 else pred <= bins[i + 1])
        if not mask.any():
            continue
        out.append(
            {
                "lo": round(float(bins[i]), 6),
                "hi": round(float(bins[i + 1]), 6),
                "n": int(mask.sum()),
                "mean_pred": round(float(pred[mask].mean()), 6),
                "mean_realized": round(float(realized[mask].mean()), 6),
            }
        )
    return out


def _prob_calib(prob: np.ndarray, realized: np.ndarray) -> list[dict[str, Any]]:
    if len(prob) < 20:
        return []
    edges = [0.0, 0.2, 0.4, 0.5, 0.6, 0.8, 1.01]
    out = []
    hit = (realized > 0).astype(float)
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (prob >= lo) & (prob < hi)
        if not mask.any():
            continue
        out.append(
            {
                "lo": lo,
                "hi": hi,
                "n": int(mask.sum()),
                "mean_pred": round(float(prob[mask].mean()), 4),
                "realized_rate": round(float(hit[mask].mean()), 4),
            }
        )
    return out


def _drawdown(rets: np.ndarray) -> float:
    if len(rets) == 0:
        return 0.0
    eq = np.cumsum(rets)
    peak = np.maximum.accumulate(eq)
    return round(float((eq - peak).min()), 6)


def _ctx_for_bar(symbol: str, window: list[dict[str, Any]]):
    from backend.services.binance_scalp.config import ScalpConfig
    from backend.services.binance_scalp.economics import ScalpEconomics
    from backend.services.binance_scalp.strategies.base import StrategyMarketContext

    from dataclasses import replace

    mid = float(window[-1]["close"])
    snap = replace(
        _snapshot(symbol, mid, window),
        order_book_imbalance=0.0,
        book_source="replay_no_book",
    )
    mom = _mom_from_bars(window)
    return StrategyMarketContext(
        symbol=symbol,
        snap=snap,
        mom=mom,
        bars_1m=window,
        econ=ScalpEconomics.from_env(),
        config=ScalpConfig.from_env(),
        notional_usd=NOTIONAL,
    )


def build_replay_rows(db_path: str, *, step: int = 1) -> list[dict[str, Any]]:
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    try:
        bars_by_sym = load_ohlcv(conn)
    finally:
        conn.close()
    rows: list[dict[str, Any]] = []
    for sym in SYMBOLS:
        raw = bars_by_sym.get(_ohlcv_symbol(sym), [])
        if len(raw) < LOOKBACK + 21:
            continue
        for i in range(LOOKBACK, len(raw) - 20, max(1, step)):
            window = raw[i - LOOKBACK : i + 1]
            mid = float(window[-1]["close"])
            if mid <= 0:
                continue
            ts = window[-1]["ts"]
            epoch = ts.timestamp() if hasattr(ts, "timestamp") else float(ts)
            ctx = _ctx_for_bar(sym, window)
            meas = measure_all_setups(ctx)
            feats = flatten_measurements(meas, live_book=False)
            labels = path_labels(mid, raw[i + 1 : i + 21], cost_pct=DEFAULT_COST)
            if f"net_{PRIMARY_HORIZON}m" not in labels:
                continue
            rows.append(
                {
                    "source": "replay_1m",
                    "symbol": sym,
                    "epoch": epoch,
                    "mid": mid,
                    "features": feats,
                    "rank_score": float(feats.get("projected_move") or 0.0),
                    "projected_move": float(feats.get("projected_move") or 0.0),
                    "orderbook_imbalance": 0.0,
                    "labels": labels,
                }
            )
    rows.sort(key=lambda r: (r["epoch"], r["symbol"]))
    return rows


def load_live_opportunity_rows(db_path: str) -> list[dict[str, Any]]:
    """Ocean/live snapshots. measurements_json only. Never parse broken signals_json."""
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    try:
        tables = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        if "scalp_opportunity_snapshots" not in tables:
            return []
        raw = list(
            conn.execute(
                """
                SELECT epoch, symbol, mid, rank_score, measurements_json,
                       plus_30s_net, plus_60s_net, plus_180s_net, plus_300s_net,
                       plus_600s_net, plus_1200s_net,
                       plus_300s_mfe, plus_300s_mae
                FROM scalp_opportunity_snapshots
                WHERE plus_300s_net IS NOT NULL
                ORDER BY epoch
                """
            )
        )
    finally:
        conn.close()
    rows: list[dict[str, Any]] = []
    for rec in raw:
        meas: dict[str, Any] = {}
        try:
            parsed = json.loads(rec["measurements_json"] or "{}")
            if isinstance(parsed, dict):
                meas = parsed
        except json.JSONDecodeError:
            meas = {}
        feats = flatten_measurements(meas, live_book=True)
        labels = {
            "cost_pct": DEFAULT_COST,
            "net_1m": rec["plus_60s_net"],
            "net_3m": rec["plus_180s_net"],
            "net_5m": rec["plus_300s_net"],
            "net_10m": rec["plus_600s_net"],
            "net_20m": rec["plus_1200s_net"],
            "mfe_5m": rec["plus_300s_mfe"],
            "mae_5m": rec["plus_300s_mae"],
        }
        if labels["net_5m"] is None:
            continue
        imb = feats.get("orderbook_imbalance", 0.0)
        rows.append(
            {
                "source": "live_opportunity",
                "symbol": str(rec["symbol"] or ""),
                "epoch": float(rec["epoch"] or 0),
                "mid": float(rec["mid"] or 0),
                "features": feats,
                "rank_score": float(rec["rank_score"] or 0),
                "projected_move": float(feats.get("projected_move") or 0.0),
                "orderbook_imbalance": float(imb),
                "labels": labels,
            }
        )
    return rows


def _matrix(rows: list[dict[str, Any]], names: tuple[str, ...], horizon: int) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray([vector_from_features(r["features"], names) for r in rows], dtype=float)
    y = np.asarray([float(r["labels"].get(f"net_{horizon}m") or 0.0) for r in rows], dtype=float)
    return x, y


def _downsample(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    picked: dict[tuple[str, int], dict[str, Any]] = {}
    for row in rows:
        key = (row["symbol"], int(row["epoch"] // (WINDOW_BARS * 60)))
        picked.setdefault(key, row)
    return sorted(picked.values(), key=lambda r: (r["epoch"], r["symbol"]))


def _tree_predict(x_train: np.ndarray, y_train: np.ndarray, x_test: np.ndarray) -> np.ndarray:
    from sklearn.tree import DecisionTreeRegressor

    if len(x_train) < 20:
        return np.zeros(len(x_test))
    model = DecisionTreeRegressor(max_depth=3, min_samples_leaf=40, random_state=0)
    model.fit(x_train, y_train)
    return model.predict(x_test)


def _policy_cycles(rows: list[dict[str, Any]], pred: np.ndarray, rank: np.ndarray) -> dict[str, Any]:
    """Group same-minute opportunities into HOLD-as-action cycles."""
    groups: dict[int, list[int]] = defaultdict(list)
    for i, row in enumerate(rows):
        groups[int(row["epoch"] // 60)].append(i)
    hold_nets: list[float] = []
    rank_nets: list[float] = []
    model_nets: list[float] = []
    model_buys = 0
    hold_wins = 0
    by_sym = {s: [] for s in SYMBOLS}
    for idxs in groups.values():
        hold_nets.append(0.0)
        best_rank = max(idxs, key=lambda i: float(rank[i]))
        rank_nets.append(float(rows[best_rank]["labels"].get(f"net_{PRIMARY_HORIZON}m") or 0.0))
        best_model = max(idxs, key=lambda i: float(pred[i]))
        if float(pred[best_model]) > 0.0:
            net = float(rows[best_model]["labels"].get(f"net_{PRIMARY_HORIZON}m") or 0.0)
            model_nets.append(net)
            model_buys += 1
            by_sym[rows[best_model]["symbol"]].append(net)
        else:
            model_nets.append(0.0)
            hold_wins += 1
    model_arr = np.asarray(model_nets)
    rank_arr = np.asarray(rank_nets)
    return {
        "cycles": len(groups),
        "hold_always": _econ(np.zeros(len(groups))),
        "rank_always_buy_best": _econ(rank_arr),
        "forward_net_vs_hold": _econ(model_arr),
        "buy_count": model_buys,
        "hold_count": hold_wins,
        "buy_pct_of_cycles": round(model_buys / len(groups), 4) if groups else None,
        "hold_pct_of_cycles": round(hold_wins / len(groups), 4) if groups else None,
        "beats_hold": bool(float(model_arr.sum()) > 0.0),
        "beats_rank": bool(float(model_arr.sum()) > float(rank_arr.sum())),
        "drawdown": _drawdown(model_arr),
        "by_symbol": {s: _econ(np.asarray(v, dtype=float)) for s, v in by_sym.items()},
    }


def _feature_hyps(rows: list[dict[str, Any]], y: np.ndarray) -> dict[str, Any]:
    out = {}
    for name in ("projected_move", "orderbook_imbalance", "reclaim_strength", "breakout_strength", "compression_score", "volume_impulse_strength", "pullback_depth", "momentum_60s"):
        xs = np.asarray([float(r["features"].get(name) or r.get(name) or 0.0) for r in rows], dtype=float)
        out[name] = _corr(xs, y)
    return out


def evaluate_split(
    train: list[dict[str, Any]],
    valid: list[dict[str, Any]],
    test: list[dict[str, Any]],
    *,
    names: tuple[str, ...],
    horizon: int,
) -> dict[str, Any]:
    x_tr, y_tr = _matrix(train, names, horizon)
    x_va, y_va = _matrix(valid, names, horizon) if valid else (np.zeros((0, len(names))), np.zeros(0))
    x_te, y_te = _matrix(test, names, horizon)
    mean = x_tr.mean(axis=0)
    scale = np.where(x_tr.std(axis=0) < 1e-12, 1.0, x_tr.std(axis=0))
    design = np.column_stack([(x_tr - mean) / scale, np.ones(len(x_tr))])
    coef, *_ = np.linalg.lstsq(design, y_tr, rcond=None)
    lin_te = predict_linear(x_te, mean, scale, coef[:-1], float(coef[-1]))
    lin_va = predict_linear(x_va, mean, scale, coef[:-1], float(coef[-1])) if len(x_va) else np.zeros(0)
    y_bin = (y_tr > 0).astype(int)
    from sklearn.linear_model import LogisticRegression

    if len(set(y_bin.tolist())) >= 2:
        clf = LogisticRegression(max_iter=200)
        clf.fit((x_tr - mean) / scale, y_bin)
        log_coef = clf.coef_[0]
        log_int = float(clf.intercept_[0])
    else:
        log_coef = np.zeros(x_tr.shape[1])
        log_int = -10.0
    prob_te = predict_logistic(x_te, mean, scale, log_coef, log_int)
    tree_te = _tree_predict(x_tr, y_tr, x_te)
    proj_te = np.asarray([float(r["projected_move"] or 0.0) for r in test], dtype=float)
    imb_te = np.asarray([float(r.get("orderbook_imbalance") or 0.0) for r in test], dtype=float)
    rank_te = np.asarray([float(r["rank_score"] or 0.0) for r in test], dtype=float)
    current_40 = np.zeros(len(test))  # live 40-feat model emits 0 expected move on clean entries

    baselines = {
        "hold_always": {"corr": None, "econ_if_always_buy": _econ(np.zeros(len(y_te))), "note": "EV=0 every cycle"},
        "rank_score": {"corr": _corr(rank_te, y_te), "quartiles": _quartile(rank_te, y_te)},
        "projected_move": {"corr": _corr(proj_te, y_te), "quartiles": _quartile(proj_te, y_te)},
        "orderbook_imbalance": {"corr": _corr(imb_te, y_te), "quartiles": _quartile(imb_te, y_te)},
        "current_40feat_zero": {"corr": _corr(current_40, y_te), "note": "artifact outputs 0 expected move"},
        "linear": {"corr": _corr(lin_te, y_te), "quartiles": _quartile(lin_te, y_te), "valid_corr": _corr(lin_va, y_va) if len(y_va) else None},
        "tree_depth3": {"corr": _corr(tree_te, y_te), "quartiles": _quartile(tree_te, y_te)},
        "logistic_p_pos": {"corr_vs_hit": _corr(prob_te, (y_te > 0).astype(float))},
    }
    ablations = {}
    full_corr = baselines["linear"]["corr"]
    for group, keys in FEATURE_GROUPS.items():
        keep = tuple(k for k in names if k not in keys)
        if not keep:
            continue
        xa_tr, ya_tr = _matrix(train, keep, horizon)
        xa_te, ya_te = _matrix(test, keep, horizon)
        ma = xa_tr.mean(axis=0)
        sa = np.where(xa_tr.std(axis=0) < 1e-12, 1.0, xa_tr.std(axis=0))
        da = np.column_stack([(xa_tr - ma) / sa, np.ones(len(xa_tr))])
        ca, *_ = np.linalg.lstsq(da, ya_tr, rcond=None)
        pred_a = predict_linear(xa_te, ma, sa, ca[:-1], float(ca[-1]))
        c = _corr(pred_a, ya_te)
        ablations[group] = {
            "corr_without": c,
            "delta_vs_full": None if full_corr is None or c is None else round(full_corr - c, 4),
        }

    policy = _policy_cycles(test, lin_te, rank_te)
    by_sym = {}
    for sym in SYMBOLS:
        idx = [i for i, r in enumerate(test) if r["symbol"] == sym]
        if len(idx) < 8:
            by_sym[sym] = {"n": len(idx)}
            continue
        p = lin_te[idx]
        y = y_te[idx]
        by_sym[sym] = {
            "n": len(idx),
            "corr": _corr(p, y),
            "quartiles": _quartile(p, y),
            "econ_if_buy_when_ev_gt_0": _econ(y[p > 0]) if (p > 0).any() else _econ(np.zeros(0)),
        }

    mfe = np.asarray([float(r["labels"].get("mfe_5m") or 0.0) for r in test], dtype=float)
    mae = np.asarray([float(r["labels"].get("mae_5m") or 0.0) for r in test], dtype=float)
    mfe_pred = lin_te  # same linear features; report correlation only
    return {
        "n_train": len(train),
        "n_valid": len(valid),
        "n_test": len(test),
        "feature_hyps_test": _feature_hyps(test, y_te),
        "baselines": baselines,
        "ablations": ablations,
        "linear_test": {
            "corr": baselines["linear"]["corr"],
            "ev_calibration": _calib(lin_te, y_te),
            "prob_calibration": _prob_calib(prob_te, y_te),
            "quartiles": baselines["linear"]["quartiles"],
            "buckets": _buckets(lin_te, y_te),
            "buy_when_ev_gt_0": _econ(y_te[lin_te > 0]) if (lin_te > 0).any() else _econ(np.zeros(0)),
        },
        "mfe_corr_with_pred_ev": _corr(mfe_pred, mfe),
        "mae_corr_with_pred_ev": _corr(mfe_pred, mae),
        "policy": policy,
        "by_symbol": by_sym,
        "artifact_parts": {
            "mean": mean.tolist(),
            "scale": scale.tolist(),
            "coef": coef[:-1].tolist(),
            "intercept": float(coef[-1]),
            "log_coef": log_coef.tolist(),
            "log_intercept": log_int,
        },
    }


def accept_model(eval_row: dict[str, Any]) -> tuple[bool, str]:
    policy = eval_row.get("policy") or {}
    lin = eval_row.get("linear_test") or {}
    corr = lin.get("corr")
    q = lin.get("quartiles") or {}
    top_q = (q.get("top_quartile") or {}).get("mean_net")
    bot_q = (q.get("bottom_quartile") or {}).get("mean_net")
    buy = lin.get("buy_when_ev_gt_0") or {}
    if corr is None or corr <= 0:
        return False, "oos predicted-EV vs realized-net correlation is not positive"
    if top_q is None or bot_q is None or top_q <= bot_q:
        return False, "oos top quartile realized net does not exceed bottom quartile"
    if not policy.get("beats_hold"):
        return False, "HOLD-as-action policy does not beat always-HOLD out of sample"
    if not policy.get("beats_rank"):
        return False, "HOLD-as-action policy does not beat current rank baseline out of sample"
    if (buy.get("n") or 0) < 30:
        return False, f"BUY-over-HOLD sample too small ({buy.get('n')})"
    if (buy.get("expectancy") or 0) <= 0:
        return False, "BUY-when-EV>0 expectancy is not positive after costs"
    return True, "beats HOLD and rank with positive OOS EV correlation and quartile separation"


def multi_horizon(train, valid, test, names) -> dict[str, Any]:
    out = {}
    for h in (1, 3, 5, 10, 20):
        usable_te = [r for r in test if r["labels"].get(f"net_{h}m") is not None]
        usable_tr = [r for r in train if r["labels"].get(f"net_{h}m") is not None]
        if len(usable_tr) < 40 or len(usable_te) < 20:
            out[f"{h}m"] = {"n_test": len(usable_te), "skipped": True}
            continue
        ev = evaluate_split(usable_tr, valid, usable_te, names=names, horizon=h)
        out[f"{h}m"] = {
            "n_test": ev["n_test"],
            "linear_corr": ev["linear_test"]["corr"],
            "projected_move_corr": ev["baselines"]["projected_move"]["corr"],
            "policy_expectancy": (ev["policy"]["forward_net_vs_hold"] or {}).get("expectancy"),
            "buy_count": ev["policy"]["buy_count"],
            "hold_count": ev["policy"]["hold_count"],
        }
    return out


def window_report(rows: list[dict[str, Any]], idx: range) -> dict[str, Any]:
    if not rows or not idx:
        return {}
    sl = [rows[i] for i in idx]
    return {
        "n": len(sl),
        "start": datetime.fromtimestamp(sl[0]["epoch"], tz=timezone.utc).isoformat(),
        "end": datetime.fromtimestamp(sl[-1]["epoch"], tz=timezone.utc).isoformat(),
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    replay = build_replay_rows(args.ohlcv_db, step=args.step)
    live = load_live_opportunity_rows(args.opp_db) if args.opp_db else []
    raw_nets = [float(r["labels"].get(f"net_{PRIMARY_HORIZON}m") or 0.0) for r in replay]
    ess = effective_sample_report(
        raw_nets,
        [r["epoch"] for r in replay],
        [r["symbol"] for r in replay],
    )
    model_rows = _downsample(replay) if args.downsample else replay
    folds = chronological_folds(len(model_rows), gap=GAP_BARS)
    if not folds:
        return {"error": "insufficient rows for chronological walk-forward", "raw": len(replay), "model_rows": len(model_rows)}
    train_i, valid_i, test_i = folds[0]
    train = [model_rows[i] for i in train_i]
    valid = [model_rows[i] for i in valid_i]
    test = [model_rows[i] for i in test_i]
    ev5 = evaluate_split(train, valid, test, names=FEATURE_KEYS, horizon=PRIMARY_HORIZON)
    accepted, reason = accept_model(ev5)
    art = fit_artifact(
        _matrix(train + valid, FEATURE_KEYS, PRIMARY_HORIZON)[0],
        _matrix(train + valid, FEATURE_KEYS, PRIMARY_HORIZON)[1],
        list(FEATURE_KEYS),
        accepted=accepted,
        reason=reason,
    )
    live_report = None
    if live:
        live_y = np.asarray([float(r["labels"]["net_5m"]) for r in live], dtype=float)
        live_report = {
            "n": len(live),
            "ess": effective_sample_report(live_y.tolist(), [r["epoch"] for r in live], [r["symbol"] for r in live]),
            "feature_hyps": _feature_hyps(live, live_y),
            "rank_corr": _corr(np.asarray([r["rank_score"] for r in live], dtype=float), live_y),
            "note": "tiny live window; hypothesis only; not used to accept the model",
        }
    report = {
        "model_version": MODEL_VERSION,
        "accepted": accepted,
        "reject_reason": reason,
        "dataset": {
            "raw_replay_rows": len(replay),
            "model_rows_after_window_group": len(model_rows),
            "downsampled": bool(args.downsample),
            "live_labeled_opportunities": len(live),
            "effective_sample": ess,
            "symbols": list(SYMBOLS),
            "cost_pct": DEFAULT_COST,
            "primary_horizon_min": PRIMARY_HORIZON,
            "feature_set": list(FEATURE_KEYS),
            "excluded_from_features": ["symbol", "passed", "rank_score", "signal_score", "signal_confidence", "evidence_rank_delta", "orderbook_imbalance_replay_fabricated"],
        },
        "windows": {
            "train": window_report(model_rows, train_i),
            "valid": window_report(model_rows, valid_i),
            "test": window_report(model_rows, test_i),
            "gap_bars": GAP_BARS,
            "leakage_controls": [
                "chronological split only",
                "no shuffle of adjacent snapshots",
                f"{GAP_BARS} bar gap between train/valid/test to block overlapping {PRIMARY_HORIZON}m-{20}m labels",
                "replay orderbook imbalance zeroed (fabricated in strategy_module_replay)",
                "broken signals_json never parsed",
                "coin identity not a feature",
            ],
        },
        "primary_5m": ev5,
        "multi_horizon": multi_horizon(train, valid, test, FEATURE_KEYS),
        "live_opportunity_hypotheses": live_report,
        "ocean_deployment": "not deployed — model rejected" if not accepted else "candidate accepted; deploy only after review",
    }
    if args.write_artifact:
        dest = save_artifact(art, Path(args.artifact) if args.artifact else None)
        report["artifact_path"] = str(dest)
        report["artifact_accepted"] = art.accepted
    return report


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--ohlcv-db", default=str(ROOT / "mystic_trading.db"))
    p.add_argument("--opp-db", default="")
    p.add_argument("--step", type=int, default=1)
    p.add_argument("--downsample", action="store_true", default=True)
    p.add_argument("--no-downsample", action="store_false", dest="downsample")
    p.add_argument("--write-artifact", action="store_true")
    p.add_argument("--artifact", default="")
    p.add_argument("--out", default="")
    args = p.parse_args()
    report = run(args)
    text = json.dumps(report, indent=2, default=str)
    if args.out:
        Path(args.out).write_text(text)
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
