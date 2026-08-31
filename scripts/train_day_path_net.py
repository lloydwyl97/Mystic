#!/usr/bin/env python3
"""Fit DAY Target-D path predictor on 60/120/180m. Accept only if OOS beats HOLD."""

from __future__ import annotations

import json
import sqlite3
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backend.config.trading_economics import ESTIMATED_ROUNDTRIP_COST
from backend.services.binance_scalp.forward_net_predictor import (
    chronological_folds,
    fit_artifact,
    save_artifact,
)
from backend.services.binance_scalp.historical_forensic import _ohlcv_symbol, load_ohlcv
from backend.services.binance_scalp.path_outcomes import path_labels_for_horizon
from backend.services.binance_scalp.reconstructable_features import FEATURE_KEYS, reconstructable_features
from backend.services.day_path_net import DAY_HORIZONS_MIN, DAY_PATH_MODEL_VERSION, LOOKBACK_BARS
from scripts.scalp_path_edge_walkforward import _corr, _downsample, _econ, _fit_predict, _policy, accept

SYMBOLS = ("BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT")
STEP = 5
GAP = 180
TARGET = "target_d_net"


def build_rows(db: str) -> list[dict]:
    conn = sqlite3.connect(db)
    bars_by_sym = load_ohlcv(conn)
    conn.close()
    btc = bars_by_sym.get(_ohlcv_symbol("BTCUSDT"), [])
    btc_close = [float(b["close"]) for b in btc]
    rows = []
    max_h = max(DAY_HORIZONS_MIN)
    for sym in SYMBOLS:
        raw = bars_by_sym.get(_ohlcv_symbol(sym), [])
        if len(raw) < LOOKBACK_BARS + max_h + 2:
            continue
        for i in range(LOOKBACK_BARS, len(raw) - max_h, STEP):
            window = raw[i - LOOKBACK_BARS : i]
            mid = float(window[-1]["close"] or 0)
            if mid <= 0:
                continue
            ts = window[-1]["ts"]
            epoch = ts.timestamp() if hasattr(ts, "timestamp") else float(i)
            btc_ret = 0.0
            if btc_close and i >= 5 and i < len(btc_close) and btc_close[i - 5] > 0:
                btc_ret = (btc_close[i] - btc_close[i - 5]) / btc_close[i - 5]
            feats = reconstructable_features(
                window,
                btc_ret_5=btc_ret,
                market_vol_5=abs(btc_ret),
                ts=ts if hasattr(ts, "hour") else None,
            )
            future = raw[i : i + max_h]
            path = {
                h: path_labels_for_horizon(
                    mid,
                    future,
                    horizon_min=h,
                    cost_pct=float(ESTIMATED_ROUNDTRIP_COST),
                    target_pct=0.004,
                )
                for h in DAY_HORIZONS_MIN
            }
            rows.append({"symbol": sym, "epoch": epoch, "features": feats, "path": path})
    rows.sort(key=lambda r: (r["epoch"], r["symbol"]))
    return rows


def _xy(rows, names, horizon, target):
    x = np.asarray([[float(r["features"].get(n) or 0.0) for n in names] for r in rows], dtype=float)
    y = np.asarray([float(r["path"][horizon].get(target) or 0.0) for r in rows], dtype=float)
    return x, y


def eval_one(train, valid, test, horizon: int) -> dict:
    x_tr, y_tr = _xy(train, FEATURE_KEYS, horizon, TARGET)
    x_va, y_va = _xy(valid, FEATURE_KEYS, horizon, TARGET)
    x_te, y_te = _xy(test, FEATURE_KEYS, horizon, TARGET)
    lin = _fit_predict(x_tr, y_tr, x_te)
    lin_va = _fit_predict(x_tr, y_tr, x_va)
    pol_rows = [{"epoch": r["epoch"], "symbol": r["symbol"], "path": {TARGET: r["path"][horizon].get(TARGET)}} for r in test]
    policy = _policy(pol_rows, lin, TARGET)
    ev = {
        "models": {
            "linear": {
                "corr": _corr(lin, y_te),
                "valid_corr": _corr(lin_va, y_va),
                "quartiles": __import__("scripts.scalp_path_edge_walkforward", fromlist=["_quartile"])._quartile(lin, y_te),
            }
        },
        "policy_linear_vs_hold": policy,
        "stability": {"valid_corr": _corr(lin_va, y_va), "test_corr": _corr(lin, y_te)},
    }
    ok, reason = accept(ev)
    valid_corr = ev["stability"]["valid_corr"]
    if ok and (valid_corr is None or valid_corr <= 0):
        ok, reason = False, "valid corr is not positive"
    by_sym = {}
    for sym in SYMBOLS:
        idx = [i for i, r in enumerate(test) if r["symbol"] == sym]
        if not idx:
            continue
        realized = np.asarray([float(test[i]["path"][horizon].get(TARGET) or 0.0) for i in idx])
        chosen = realized[lin[idx] > 0]
        by_sym[sym] = {
            "n": len(idx),
            "buy_n": len(chosen),
            "buy_wr": None if len(chosen) == 0 else round(float((chosen > 0).mean()), 4),
            "buy_exp": None if len(chosen) == 0 else round(float(chosen.mean()), 6),
        }
    return {"accepted": ok, "reason": reason, "horizon_min": horizon, "ev": ev, "by_symbol": by_sym, "lin": lin, "y_te": y_te}


def main() -> int:
    db = "/tmp/ocean_forward_net.db"
    if not Path(db).exists():
        db = str(ROOT / "mystic_trading.db")
    raw = build_rows(db)
    rows = _downsample(raw)
    folds = chronological_folds(len(rows), gap=GAP)
    if not folds:
        print(json.dumps({"accepted": False, "reason": "insufficient rows"}))
        return 1
    tr, va, te = folds[0]
    train = [rows[i] for i in tr]
    valid = [rows[i] for i in va]
    test = [rows[i] for i in te]
    scored = [eval_one(train, valid, test, h) for h in DAY_HORIZONS_MIN]
    scored.sort(
        key=lambda s: (
            not s["accepted"],
            -float(((s["ev"].get("policy_linear_vs_hold") or {}).get("econ") or {}).get("expectancy") or -9),
        )
    )
    best = scored[0]
    x_fit, y_fit = _xy(train + valid, FEATURE_KEYS, best["horizon_min"], TARGET)
    art = fit_artifact(x_fit, y_fit, list(FEATURE_KEYS), accepted=bool(best["accepted"]), reason=best["reason"])
    art.version = DAY_PATH_MODEL_VERSION
    art.primary_horizon_min = int(best["horizon_min"])
    dest = ROOT / "models" / "day_path_net_v1.json"
    if not best["accepted"]:
        dest = ROOT / "models" / "rejected" / "day_path_net_v1_rejected.json"
    save_artifact(art, dest)
    pol = best["ev"]["policy_linear_vs_hold"]
    report = {
        "artifact": str(dest),
        "accepted": best["accepted"],
        "reason": best["reason"],
        "horizon_min": best["horizon_min"],
        "target": TARGET,
        "cost_pct": float(ESTIMATED_ROUNDTRIP_COST),
        "n_train": len(train),
        "n_valid": len(valid),
        "n_test": len(test),
        "corr": best["ev"]["models"]["linear"]["corr"],
        "valid_corr": best["ev"]["stability"]["valid_corr"],
        "buy_count": pol.get("buy_count"),
        "hold_count": pol.get("hold_count"),
        "expectancy": (pol.get("econ") or {}).get("expectancy"),
        "pf": (pol.get("econ") or {}).get("pf"),
        "by_symbol": best["by_symbol"],
        "all_horizons": [
            {
                "horizon_min": s["horizon_min"],
                "accepted": s["accepted"],
                "reason": s["reason"],
                "corr": s["ev"]["models"]["linear"]["corr"],
                "valid_corr": s["ev"]["stability"]["valid_corr"],
                "expectancy": ((s["ev"].get("policy_linear_vs_hold") or {}).get("econ") or {}).get("expectancy"),
            }
            for s in scored
        ],
    }
    print(json.dumps(report, indent=2))
    return 0 if best["accepted"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
