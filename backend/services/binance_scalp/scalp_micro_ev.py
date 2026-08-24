"""Heuristic multi-horizon executable-net EV from microstructure.

Until markout N is large enough for a trained calibrated model, this uses
the existing 40-dim path-net predictor (when present) plus a documented
microstructure tilt. Outputs stay ranking/size inputs, never gates.

Calibration: if a bucket has < MIN_CALIB_N observations, report
INCONCLUSIVE rather than inventing a fitted calibrator.
"""

from __future__ import annotations

import math
from typing import Any

from backend.services.binance_scalp.scalp_micro_contract import EV_HORIZONS_SEC, MODEL_VERSION, version_stamps

MIN_CALIB_N = 80


def _f(d: dict[str, Any], k: str, default: float = 0.0) -> float:
    try:
        return float(d.get(k) if d.get(k) is not None else default)
    except (TypeError, ValueError):
        return default


def heuristic_horizon_ev(feats: dict[str, Any] | None, horizon_sec: int) -> float:
    """Signed expected executable net return (fraction) at ``horizon_sec``."""
    f = feats or {}
    ofi = math.tanh(_f(f, "ofi_5s") / 5.0)
    flow = _f(f, "agg_flow_imbalance_5s")
    mp = math.tanh(_f(f, "microprice_pressure") * 500.0)
    obi = _f(f, "obi_l5")
    absorp = _f(f, "bid_absorption_score") - _f(f, "ask_absorption_score")
    adverse = _f(f, "adverse_selection_score")
    spread = max(0.0, _f(f, "spread_pct"))
    signal = 0.28 * ofi + 0.22 * flow + 0.20 * mp + 0.15 * obi + 0.15 * absorp - 0.35 * adverse
    decay = {1: 0.35, 5: 0.70, 10: 1.00, 30: 0.85, 60: 0.65}.get(int(horizon_sec), 0.70)
    # Scale to typical SCALP net (few bp), then subtract half-spread as cost drag.
    raw = signal * 0.00035 * decay
    return float(raw - 0.5 * spread)


def multi_horizon_ev(feats: dict[str, Any] | None, base_ev: float | None = None) -> dict[str, Any]:
    evs: dict[str, float] = {}
    for h in EV_HORIZONS_SEC:
        evs[f"EV_{h}s"] = round(heuristic_horizon_ev(feats, h), 8)
    if base_ev is not None:
        # Blend existing path-net EV into 30s/60s where hold times overlap.
        evs["EV_30s"] = round(0.55 * evs["EV_30s"] + 0.45 * float(base_ev), 8)
        evs["EV_60s"] = round(0.45 * evs["EV_60s"] + 0.55 * float(base_ev), 8)
    p_pos = {h: round(max(0.05, min(0.95, 0.5 + 400.0 * evs[f"EV_{h}s"])), 4) for h in EV_HORIZONS_SEC}
    adverse = _f(feats or {}, "adverse_selection_score")
    out = {
        **evs,
        "p_positive_executable_net_5s": p_pos[5],
        "p_positive_executable_net_10s": p_pos[10],
        "p_positive_executable_net_30s": p_pos[30],
        "p_adverse_move": round(adverse, 4),
        "selection_micro_score": round(
            1.2 * evs["EV_5s"] + 1.4 * evs["EV_10s"] + 0.8 * evs["EV_30s"] - 0.0004 * adverse,
            8,
        ),
        "model_version": MODEL_VERSION,
        "calibration_status": "INCONCLUSIVE",
        **version_stamps(),
    }
    return out


def calibration_report(predicted: list[float], realized_positive: list[int], realized_net: list[float]) -> dict[str, Any]:
    n = min(len(predicted), len(realized_positive), len(realized_net))
    if n < MIN_CALIB_N:
        return {"n": n, "status": "INCONCLUSIVE", "reason": f"n<{MIN_CALIB_N}"}
    brier = sum((p - y) ** 2 for p, y in zip(predicted[:n], realized_positive[:n], strict=False)) / n
    # 10 equal-width buckets
    buckets = []
    for i in range(10):
        lo, hi = i / 10.0, (i + 1) / 10.0
        idx = [j for j, p in enumerate(predicted[:n]) if lo <= p < hi or (i == 9 and p == 1.0)]
        if not idx:
            continue
        obs = sum(realized_positive[j] for j in idx) / len(idx)
        pred = sum(predicted[j] for j in idx) / len(idx)
        buckets.append({"lo": lo, "hi": hi, "n": len(idx), "predicted": round(pred, 4), "observed": round(obs, 4)})
    ece = sum(abs(b["predicted"] - b["observed"]) * b["n"] for b in buckets) / n if buckets else 0.0
    exp_net = sum(predicted[:n]) / n
    real_net = sum(realized_net[:n]) / n
    return {
        "n": n,
        "status": "HEALTHY" if ece < 0.12 else "WEAK",
        "brier": round(brier, 6),
        "ece": round(ece, 6),
        "expected_net": round(exp_net, 8),
        "realized_net": round(real_net, 8),
        "buckets": buckets,
    }


__all__ = [
    "MIN_CALIB_N",
    "calibration_report",
    "heuristic_horizon_ev",
    "multi_horizon_ev",
]
