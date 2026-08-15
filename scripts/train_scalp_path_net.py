#!/usr/bin/env python3
"""Fit Target-D path predictor. Writes accepted artifact only if OOS economics hold."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backend.services.binance_scalp.forward_net_predictor import (
    PATH_MODEL_VERSION,
    chronological_folds,
    fit_artifact,
    save_artifact,
)
from backend.services.binance_scalp.reconstructable_features import FEATURE_KEYS
from scripts.scalp_path_edge_walkforward import (
    _downsample,
    _fit_predict,
    _policy,
    _xy,
    build_rows,
    accept,
    eval_target,
)

HORIZON = 20
TARGET = "target_d_net"


def main() -> int:
    db = "/tmp/ocean_forward_net.db"
    if not Path(db).exists():
        db = str(ROOT / "mystic_trading.db")
    raw = build_rows(db, step=1)
    rows = _downsample(raw)
    folds = chronological_folds(len(rows), gap=20)
    if not folds:
        print(json.dumps({"accepted": False, "reason": "insufficient rows"}))
        return 1
    tr, va, te = folds[0]
    train = [rows[i] for i in tr]
    valid = [rows[i] for i in va]
    test = [rows[i] for i in te]
    ev = eval_target(train, valid, test, FEATURE_KEYS, HORIZON, TARGET)
    ok, reason = accept(ev)
    # Boolean-style accept is fine here because Target D is a net label.
    x_fit, y_fit = _xy(train + valid, FEATURE_KEYS, HORIZON, TARGET)
    art = fit_artifact(x_fit, y_fit, list(FEATURE_KEYS), accepted=ok, reason=reason)
    art.version = PATH_MODEL_VERSION
    art.primary_horizon_min = HORIZON
    dest = ROOT / "models" / "scalp_path_net_v1.json"
    save_artifact(art, dest)
    lin = ev["models"]["linear"]
    pol = ev["policy_linear_vs_hold"]
    report = {
        "artifact": str(dest),
        "accepted": ok,
        "reason": reason,
        "horizon_min": HORIZON,
        "target": TARGET,
        "corr": lin["corr"],
        "valid_corr": ev["stability"]["valid_corr"],
        "buy_count": pol["buy_count"],
        "hold_count": pol["hold_count"],
        "expectancy": pol["econ"]["expectancy"],
        "pf": pol["econ"]["pf"],
        "by_symbol": pol["by_symbol"],
    }
    print(json.dumps(report, indent=2))
    return 0 if ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
