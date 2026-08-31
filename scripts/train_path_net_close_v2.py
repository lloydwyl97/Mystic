#!/usr/bin/env python3
"""Train close-to-close path-net v2. Does not overwrite live v1 artifacts.

Label is horizon close minus assumed cost. No wick MFE. Drops evt_vol_accel.
Live loaders still read day_path_net_v1 / scalp_path_net_v1.
"""

from __future__ import annotations

import json
import sqlite3
import sys
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
from backend.services.day_path_net import DAY_HORIZONS_MIN, LOOKBACK_BARS
from scripts.scalp_path_edge_walkforward import _corr, _downsample, _econ, _fit_predict, _policy, accept

SYMBOLS = ("BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT")
TARGET = "target_close_net"
FEATURE_KEYS_V2 = tuple(k for k in FEATURE_KEYS if k != "evt_vol_accel")
DAY_VERSION = "day_path_net_v2"
SCALP_VERSION = "scalp_path_net_v2"
SCALP_HORIZON = 20
SCALP_COST = 0.0006
DAY_STEP = 5
SCALP_STEP = 5  # matches 5m downsample; same labels, less wasted 1m clones


def _wr(arr: np.ndarray) -> float | None:
    if len(arr) == 0:
        return None
    return round(float((arr > 0).mean()), 4)


def build_rows(db: str, *, horizons: tuple[int, ...], cost: float, step: int, target_pct: float) -> list[dict]:
    conn = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    bars_by_sym = load_ohlcv(conn)
    conn.close()
    btc = bars_by_sym.get(_ohlcv_symbol("BTCUSDT"), [])
    btc_close = [float(b["close"]) for b in btc]
    rows = []
    max_h = max(horizons)
    for sym in SYMBOLS:
        raw = bars_by_sym.get(_ohlcv_symbol(sym), [])
        if len(raw) < LOOKBACK_BARS + max_h + 2:
            continue
        for i in range(LOOKBACK_BARS, len(raw) - max_h, step):
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
            path = {h: path_labels_for_horizon(mid, future, horizon_min=h, cost_pct=cost, target_pct=target_pct) for h in horizons}
            rows.append({"symbol": sym, "epoch": epoch, "features": feats, "path": path})
    rows.sort(key=lambda r: (r["epoch"], r["symbol"]))
    return rows


def _xy(rows, names, horizon, target):
    x = np.asarray([[float(r["features"].get(n) or 0.0) for n in names] for r in rows], dtype=float)
    y = np.asarray([float(r["path"][horizon].get(target) or 0.0) for r in rows], dtype=float)
    return x, y


def eval_one(train, valid, test, horizon: int, names: tuple[str, ...]) -> dict:
    x_tr, y_tr = _xy(train, names, horizon, TARGET)
    x_va, y_va = _xy(valid, names, horizon, TARGET)
    x_te, y_te = _xy(test, names, horizon, TARGET)
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
    buy_realized = np.asarray([float(r["path"][horizon].get(TARGET) or 0.0) for r, p in zip(test, lin, strict=False) if float(p) > 0])
    return {
        "accepted": ok,
        "reason": reason,
        "horizon_min": horizon,
        "ev": ev,
        "lin": lin,
        "y_te": y_te,
        "buy_wr": _wr(buy_realized),
        "buy_n": len(buy_realized),
        "all_action_wr": _wr(np.asarray([float(r["path"][horizon].get(TARGET) or 0.0) if float(p) > 0 else 0.0 for r, p in zip(test, lin, strict=False)])),
    }


def _coef_report(art) -> dict:
    rows = []
    for n, c in zip(art.feature_names, art.coef, strict=False):
        rows.append({"feature": n, "coef": c, "one_sigma_bps": round(float(c) * 1e4, 4)})
    rows.sort(key=lambda r: abs(r["coef"]), reverse=True)
    return {
        "intercept": art.intercept,
        "intercept_bps": round(float(art.intercept) * 1e4, 4),
        "buy_at_mean_features": bool(float(art.intercept) > 0),
        "top": rows[:8],
        "volume_accel_bps": next((r["one_sigma_bps"] for r in rows if r["feature"] == "volume_accel"), None),
        "evt_vol_accel_present": "evt_vol_accel" in art.feature_names,
        "vwap_dist_bps": next((r["one_sigma_bps"] for r in rows if r["feature"] == "vwap_dist"), None),
        "pullback_recovery_bps": next((r["one_sigma_bps"] for r in rows if r["feature"] == "pullback_recovery"), None),
    }


def train_engine(*, db: str, out_dir: Path, engine: str) -> dict:
    if engine == "day":
        horizons = DAY_HORIZONS_MIN
        cost = float(ESTIMATED_ROUNDTRIP_COST)
        step = DAY_STEP
        gap = 180
        version = DAY_VERSION
        target_pct = 0.004
        dest_name = "day_path_net_v2.json"
    else:
        horizons = (SCALP_HORIZON,)
        cost = SCALP_COST
        step = SCALP_STEP
        gap = 20
        version = SCALP_VERSION
        target_pct = 0.0025
        dest_name = "scalp_path_net_v2.json"
    raw = build_rows(db, horizons=horizons, cost=cost, step=step, target_pct=target_pct)
    rows = _downsample(raw)
    folds = chronological_folds(len(rows), gap=gap)
    if not folds:
        return {"engine": engine, "accepted": False, "reason": "insufficient rows"}
    tr, va, te = folds[0]
    train = [rows[i] for i in tr]
    valid = [rows[i] for i in va]
    test = [rows[i] for i in te]
    scored = [eval_one(train, valid, test, h, FEATURE_KEYS_V2) for h in horizons]
    scored.sort(
        key=lambda s: (
            not s["accepted"],
            -float(((s["ev"].get("policy_linear_vs_hold") or {}).get("econ") or {}).get("expectancy") or -9),
        )
    )
    best = scored[0]
    x_fit, y_fit = _xy(train + valid, FEATURE_KEYS_V2, best["horizon_min"], TARGET)
    art = fit_artifact(x_fit, y_fit, list(FEATURE_KEYS_V2), accepted=bool(best["accepted"]), reason=best["reason"])
    art.version = version
    art.primary_horizon_min = int(best["horizon_min"])
    dest = out_dir / dest_name
    if not best["accepted"]:
        dest = out_dir / "rejected" / dest_name.replace(".json", "_rejected.json")
        dest.parent.mkdir(parents=True, exist_ok=True)
    save_artifact(art, dest)
    pol = best["ev"]["policy_linear_vs_hold"]
    wr = best["buy_wr"]
    exp = (pol.get("econ") or {}).get("expectancy")
    authority_pass = bool(best["accepted"] and wr is not None and wr >= 0.60 and (pol.get("econ") or {}).get("net", 0) > 0 and (exp or 0) > 0 and (best["buy_n"] or 0) >= 30)
    return {
        "engine": engine,
        "artifact": str(dest),
        "live_overwrite": False,
        "target": TARGET,
        "features": list(FEATURE_KEYS_V2),
        "accepted_by_train_script": best["accepted"],
        "reason": best["reason"],
        "authority_pass_60_wr_profit_exp": authority_pass,
        "horizon_min": best["horizon_min"],
        "cost_pct": cost,
        "n_train": len(train),
        "n_valid": len(valid),
        "n_test": len(test),
        "corr": best["ev"]["models"]["linear"]["corr"],
        "valid_corr": best["ev"]["stability"]["valid_corr"],
        "buy_count": pol.get("buy_count"),
        "hold_count": pol.get("hold_count"),
        "buy_wr": wr,
        "net": (pol.get("econ") or {}).get("net"),
        "expectancy": exp,
        "pf": (pol.get("econ") or {}).get("pf"),
        "coefficients": _coef_report(art),
    }


def main() -> int:
    db = str(ROOT / "mystic_trading.db")
    if len(sys.argv) > 1:
        db = sys.argv[1]
    out_dir = ROOT / "models"
    day = train_engine(db=db, out_dir=out_dir, engine="day")
    scalp = train_engine(db=db, out_dir=out_dir, engine="scalp")
    report = {"day": day, "scalp": scalp, "v1_untouched": True}
    print(json.dumps(report, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
