#!/usr/bin/env python3
"""Ablate close-to-close v2 feature groups on the 90d sidecar. Does not deploy."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backend.services.binance_scalp.forward_net_predictor import chronological_folds
from backend.services.binance_scalp.reconstructable_features import FEATURE_GROUPS
from scripts.scalp_path_edge_walkforward import _corr, _downsample, _fit_predict, _policy
from scripts.train_path_net_close_v2 import (
    DAY_STEP,
    FEATURE_KEYS_V2,
    SCALP_COST,
    SCALP_HORIZON,
    SCALP_STEP,
    TARGET,
    _xy,
    build_rows,
)
from backend.config.trading_economics import ESTIMATED_ROUNDTRIP_COST
from backend.services.day_path_net import DAY_HORIZONS_MIN

GROUPS = {
    name: tuple(k for k in keys if k in FEATURE_KEYS_V2)
    for name, keys in FEATURE_GROUPS.items()
    if any(k in FEATURE_KEYS_V2 for k in keys)
}


def _score(train, valid, test, names: tuple[str, ...], horizon: int) -> dict:
    if len(names) < 1:
        return {"corr": None, "valid_corr": None}
    x_tr, y_tr = _xy(train, names, horizon, TARGET)
    x_va, y_va = _xy(valid, names, horizon, TARGET)
    x_te, y_te = _xy(test, names, horizon, TARGET)
    pred = _fit_predict(x_tr, y_tr, x_te)
    pred_va = _fit_predict(x_tr, y_tr, x_va)
    pol_rows = [{"epoch": r["epoch"], "symbol": r["symbol"], "path": {TARGET: r["path"][horizon].get(TARGET)}} for r in test]
    pol = _policy(pol_rows, pred, TARGET)
    buy = np.asarray([float(r["path"][horizon].get(TARGET) or 0.0) for r, p in zip(test, pred) if float(p) > 0])
    wr = None if len(buy) == 0 else round(float((buy > 0).mean()), 4)
    exp = (pol.get("econ") or {}).get("expectancy")
    net = (pol.get("econ") or {}).get("net")
    return {
        "n_features": len(names),
        "corr": _corr(pred, y_te),
        "valid_corr": _corr(pred_va, y_va),
        "buy_count": pol.get("buy_count"),
        "hold_count": pol.get("hold_count"),
        "buy_wr": wr,
        "net": net,
        "expectancy": exp,
        "authority_pass": bool(
            wr is not None and wr >= 0.60 and (net or 0) > 0 and (exp or 0) > 0 and (pol.get("buy_count") or 0) >= 30
        ),
    }


def run_engine(*, db: str, engine: str) -> dict:
    if engine == "day":
        horizons = DAY_HORIZONS_MIN
        cost = float(ESTIMATED_ROUNDTRIP_COST)
        step = DAY_STEP
        gap = 180
        target_pct = 0.004
        horizon = 120
    else:
        horizons = (SCALP_HORIZON,)
        cost = SCALP_COST
        step = SCALP_STEP
        gap = 20
        target_pct = 0.0025
        horizon = SCALP_HORIZON
    raw = build_rows(db, horizons=horizons, cost=cost, step=step, target_pct=target_pct)
    rows = _downsample(raw)
    folds = chronological_folds(len(rows), gap=gap)
    if not folds:
        return {"engine": engine, "error": "insufficient rows"}
    tr, va, te = folds[0]
    train = [rows[i] for i in tr]
    valid = [rows[i] for i in va]
    test = [rows[i] for i in te]
    full = _score(train, valid, test, FEATURE_KEYS_V2, horizon)
    leave_one_out = {}
    group_only = {}
    for name, keys in GROUPS.items():
        without = tuple(k for k in FEATURE_KEYS_V2 if k not in keys)
        leave_one_out[name] = _score(train, valid, test, without, horizon)
        leave_one_out[name]["dropped"] = list(keys)
        if full["corr"] is not None and leave_one_out[name]["corr"] is not None:
            leave_one_out[name]["delta_vs_full"] = round(full["corr"] - leave_one_out[name]["corr"], 4)
        group_only[name] = _score(train, valid, test, keys, horizon)
        group_only[name]["kept"] = list(keys)
    return {
        "engine": engine,
        "horizon_min": horizon,
        "target": TARGET,
        "n_train": len(train),
        "n_valid": len(valid),
        "n_test": len(test),
        "full": full,
        "leave_one_group_out": leave_one_out,
        "group_only": group_only,
        "live_overwrite": False,
    }


def main() -> int:
    db = str(ROOT / "data" / "sidecar_ohlcv_90d.db")
    if len(sys.argv) > 1:
        db = sys.argv[1]
    report = {"day": run_engine(db=db, engine="day"), "scalp": run_engine(db=db, engine="scalp"), "v1_untouched": True}
    print(json.dumps(report, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
