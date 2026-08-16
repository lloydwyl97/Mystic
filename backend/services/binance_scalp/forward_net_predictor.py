"""Forward-net predictor: expected executable net after costs.

Does not gate. HOLD-as-action consumes predicted BUY EV.
Bar-reconstructable features only unless live measurements are present.
Broken signals_json is never parsed as truth.
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

MODEL_VERSION = "scalp_forward_net_v1"
PATH_MODEL_VERSION = "scalp_path_net_v1"
DEFAULT_COST = 0.0006
HORIZONS_MIN = (1, 3, 5, 10, 20)
PRIMARY_HORIZON = 5
TARGET_PCT = 0.0025
GAP_BARS = 20  # keep train/test horizons from overlapping
WINDOW_BARS = 5  # validation downsample: one row per 5m window

# Bar-trustworthy continuous measurements. No symbol identity. No passed/rank labels.
FEATURE_KEYS: tuple[str, ...] = (
    "spread_pct",
    "volatility_state",
    "momentum_15s",
    "momentum_30s",
    "momentum_60s",
    "momentum_acceleration",
    "reclaim_strength",
    "breakout_strength",
    "reversal_strength",
    "momentum_flip_strength",
    "compression_score",
    "volume_impulse_strength",
    "pullback_depth",
    "projected_move",
    "ema_relationship",
    "range_width",
)

LIVE_ONLY_KEYS: tuple[str, ...] = ("orderbook_imbalance",)

FEATURE_GROUPS: dict[str, tuple[str, ...]] = {
    "momentum": ("momentum_15s", "momentum_30s", "momentum_60s", "momentum_acceleration"),
    "setup_strength": (
        "reclaim_strength",
        "breakout_strength",
        "reversal_strength",
        "momentum_flip_strength",
        "compression_score",
        "volume_impulse_strength",
        "pullback_depth",
    ),
    "projected_move": ("projected_move",),
    "execution": ("spread_pct", "volatility_state", "range_width", "ema_relationship"),
}


def _num(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        out = float(value)
        if not math.isfinite(out):
            return default
        return out
    except (TypeError, ValueError):
        return default


def flatten_measurements(measurements: dict[str, Any] | None, *, live_book: bool = False) -> dict[str, float]:
    """Reduce 9-strategy measurements to the shared continuous feature set."""
    out = {k: 0.0 for k in FEATURE_KEYS}
    if live_book:
        out["orderbook_imbalance"] = 0.0
    if not isinstance(measurements, dict):
        return out
    common = None
    for feats in measurements.values():
        if isinstance(feats, dict) and feats:
            common = feats
            break
    if isinstance(common, dict):
        for key in ("spread_pct", "volatility_state", "momentum_15s", "momentum_30s", "momentum_60s", "momentum_acceleration"):
            if key in common:
                out[key] = _num(common.get(key))
        if live_book and "orderbook_imbalance" in common:
            out["orderbook_imbalance"] = _num(common.get("orderbook_imbalance"))
    pick = {
        "reclaim_strength": ("vwap_ema_reclaim", "failed_breakdown_reversal", "failed_breakout_reversal"),
        "breakout_strength": ("breakout_momentum",),
        "reversal_strength": ("range_bounce_scalp", "failed_breakdown_reversal", "failed_breakout_reversal"),
        "momentum_flip_strength": ("range_bounce_scalp",),
        "compression_score": ("compression_breakout",),
        "volume_impulse_strength": ("volume_impulse_continuation", "failed_breakdown_reversal"),
        "pullback_depth": ("trend_pullback_micro", "vwap_ema_reclaim"),
        "projected_move": ("vwap_ema_reclaim", "range_bounce_scalp", "breakout_momentum"),
        "ema_relationship": ("vwap_ema_reclaim", "trend_pullback_micro"),
        "range_width": ("range_bounce_scalp", "compression_breakout"),
    }
    for feat, setups in pick.items():
        vals = []
        for name in setups:
            block = measurements.get(name) or {}
            if isinstance(block, dict) and feat in block:
                vals.append(_num(block.get(feat)))
        if vals:
            out[feat] = max(vals, key=abs) if feat != "projected_move" else max(vals)
    return out


def vector_from_features(feats: dict[str, float], names: tuple[str, ...]) -> list[float]:
    return [_num(feats.get(name)) for name in names]


def path_labels(mid0: float, future: list[dict[str, Any]], *, cost_pct: float, target_pct: float = TARGET_PCT) -> dict[str, Any]:
    if mid0 <= 0 or not future:
        return {}
    out: dict[str, Any] = {"cost_pct": cost_pct}
    highs: list[float] = []
    lows: list[float] = []
    t_mfe = None
    t_mae = None
    t_target = None
    for i, bar in enumerate(future, start=1):
        high = _num(bar.get("high"))
        low = _num(bar.get("low"))
        close = _num(bar.get("close"))
        if high > 0:
            highs.append(high)
        if low > 0:
            lows.append(low)
        if t_mfe is None and highs and (max(highs) - mid0) / mid0 >= target_pct:
            t_mfe = i
        if t_mae is None and lows and (min(lows) - mid0) / mid0 <= -target_pct:
            t_mae = i
        if t_target is None and high > 0 and (high - mid0) / mid0 >= target_pct:
            t_target = i
        if i in HORIZONS_MIN and close > 0:
            gross = (close - mid0) / mid0
            mfe = max(((h - mid0) / mid0) for h in highs) if highs else 0.0
            mae = min(((lo - mid0) / mid0) for lo in lows) if lows else 0.0
            out[f"net_{i}m"] = gross - cost_pct
            out[f"gross_{i}m"] = gross
            out[f"mfe_{i}m"] = mfe
            out[f"mae_{i}m"] = mae
            out[f"hit_{i}m"] = bool(mfe >= target_pct)
    out["time_to_mfe"] = t_mfe
    out["time_to_mae"] = t_mae
    out["time_to_target"] = t_target
    out["hit_target"] = bool(out.get("mfe_5m", 0.0) >= target_pct)
    return out


def chronological_folds(n: int, *, gap: int = GAP_BARS) -> list[tuple[range, range, range]]:
    """Train / validate / test index ranges. Time order only. Gap blocks horizon leak."""
    if n < 80:
        return []
    train_end = int(n * 0.60)
    valid_end = int(n * 0.78)
    test_start = min(n - 1, valid_end + gap)
    if train_end < 30 or valid_end - train_end - gap < 10 or n - test_start < 15:
        return []
    return [
        (
            range(0, train_end),
            range(min(n, train_end + gap), valid_end),
            range(test_start, n),
        )
    ]


def effective_sample_report(net: list[float], epochs: list[float], symbols: list[str]) -> dict[str, Any]:
    n = len(net)
    windows = set()
    for ep, sym in zip(epochs, symbols):
        windows.add((sym, int(ep // (WINDOW_BARS * 60))))
    xs = np.asarray(net, dtype=float)
    ac = None
    if n > 20:
        a = xs[:-1] - xs[:-1].mean()
        b = xs[1:] - xs[1:].mean()
        den = float(np.sqrt((a * a).sum() * (b * b).sum()))
        ac = float(a.dot(b) / den) if den else None
    ess = n
    if ac is not None and abs(ac) < 0.999:
        ess = int(round(n * (1.0 - abs(ac)) / (1.0 + abs(ac))))
    return {
        "raw_snapshots": n,
        "unique_market_windows": len(windows),
        "lag1_autocorr_net": None if ac is None else round(ac, 4),
        "effective_sample_size": max(1, ess),
    }


@dataclass
class ForwardNetArtifact:
    version: str
    accepted: bool
    feature_names: list[str]
    mean: list[float]
    scale: list[float]
    coef: list[float]
    intercept: float
    log_coef: list[float]
    log_intercept: float
    primary_horizon_min: int
    trained_at: str
    reject_reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "accepted": self.accepted,
            "feature_names": self.feature_names,
            "mean": self.mean,
            "scale": self.scale,
            "coef": self.coef,
            "intercept": self.intercept,
            "log_coef": self.log_coef,
            "log_intercept": self.log_intercept,
            "primary_horizon_min": self.primary_horizon_min,
            "trained_at": self.trained_at,
            "reject_reason": self.reject_reason,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ForwardNetArtifact":
        return cls(
            version=str(payload.get("version") or MODEL_VERSION),
            accepted=bool(payload.get("accepted")),
            feature_names=list(payload.get("feature_names") or FEATURE_KEYS),
            mean=[float(x) for x in payload.get("mean") or []],
            scale=[float(x) for x in payload.get("scale") or []],
            coef=[float(x) for x in payload.get("coef") or []],
            intercept=float(payload.get("intercept") or 0.0),
            log_coef=[float(x) for x in payload.get("log_coef") or []],
            log_intercept=float(payload.get("log_intercept") or 0.0),
            primary_horizon_min=int(payload.get("primary_horizon_min") or PRIMARY_HORIZON),
            trained_at=str(payload.get("trained_at") or ""),
            reject_reason=str(payload.get("reject_reason") or ""),
        )


def _standardize_fit(x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = x.mean(axis=0)
    scale = x.std(axis=0)
    scale = np.where(scale < 1e-12, 1.0, scale)
    return (x - mean) / scale, mean, scale


def _standardize_apply(x: np.ndarray, mean: np.ndarray, scale: np.ndarray) -> np.ndarray:
    return (x - mean) / np.where(scale < 1e-12, 1.0, scale)


def fit_linear(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, float]:
    xs, mean, scale = _standardize_fit(x)
    xs = np.column_stack([xs, np.ones(len(xs))])
    coef, *_ = np.linalg.lstsq(xs, y, rcond=None)
    return coef, float(coef[-1])


def predict_linear(x: np.ndarray, mean: np.ndarray, scale: np.ndarray, coef: np.ndarray, intercept: float) -> np.ndarray:
    xs = _standardize_apply(x, mean, scale)
    if len(coef) == xs.shape[1] + 1:
        return xs.dot(coef[:-1]) + coef[-1]
    return xs.dot(coef) + intercept


def fit_logistic(x: np.ndarray, y_bin: np.ndarray) -> tuple[np.ndarray, float]:
    from sklearn.linear_model import LogisticRegression

    xs, mean, scale = _standardize_fit(x)
    if len(set(y_bin.tolist())) < 2:
        return np.zeros(x.shape[1]), -10.0
    clf = LogisticRegression(max_iter=200, solver="lbfgs")
    clf.fit(xs, y_bin)
    return clf.coef_[0], float(clf.intercept_[0])


def predict_logistic(x: np.ndarray, mean: np.ndarray, scale: np.ndarray, coef: np.ndarray, intercept: float) -> np.ndarray:
    z = _standardize_apply(x, mean, scale).dot(coef) + intercept
    return 1.0 / (1.0 + np.exp(-np.clip(z, -20, 20)))


def fit_artifact(x: np.ndarray, y_net: np.ndarray, names: list[str], *, accepted: bool, reason: str = "") -> ForwardNetArtifact:
    xs, mean, scale = _standardize_fit(x)
    design = np.column_stack([xs, np.ones(len(xs))])
    coef, *_ = np.linalg.lstsq(design, y_net, rcond=None)
    y_bin = (y_net > 0).astype(int)
    log_coef, log_int = fit_logistic(x, y_bin)
    return ForwardNetArtifact(
        version=MODEL_VERSION,
        accepted=accepted,
        feature_names=names,
        mean=mean.tolist(),
        scale=scale.tolist(),
        coef=coef[:-1].tolist(),
        intercept=float(coef[-1]),
        log_coef=log_coef.tolist(),
        log_intercept=log_int,
        primary_horizon_min=PRIMARY_HORIZON,
        trained_at=datetime.now(timezone.utc).isoformat(),
        reject_reason=reason,
    )


def predict_artifact(art: ForwardNetArtifact, feats: dict[str, float]) -> dict[str, float]:
    names = tuple(art.feature_names)
    x = np.asarray([vector_from_features(feats, names)], dtype=float)
    mean = np.asarray(art.mean, dtype=float)
    scale = np.asarray(art.scale, dtype=float)
    ev = float(predict_linear(x, mean, scale, np.asarray(art.coef, dtype=float), art.intercept)[0])
    prob = float(predict_logistic(x, mean, scale, np.asarray(art.log_coef, dtype=float), art.log_intercept)[0])
    return {
        "predicted_net_ev": ev,
        "predicted_prob_positive_net": prob,
        "model_version": 1.0,
    }


_LOADED: ForwardNetArtifact | None = None
_LOAD_ATTEMPTED = False


def reset_artifact_cache() -> None:
    global _LOADED, _LOAD_ATTEMPTED
    _LOADED = None
    _LOAD_ATTEMPTED = False


def artifact_path() -> Path:
    raw = os.getenv("SCALP_FORWARD_NET_ARTIFACT", "")
    if raw:
        return Path(raw)
    root = Path(__file__).resolve().parents[3] / "models"
    path_model = root / "scalp_path_net_v1.json"
    if path_model.exists():
        return path_model
    return root / "scalp_forward_net_v1.json"


def load_accepted_artifact() -> ForwardNetArtifact | None:
    global _LOADED, _LOAD_ATTEMPTED
    if _LOAD_ATTEMPTED:
        return _LOADED if _LOADED is not None and _LOADED.accepted else None
    _LOAD_ATTEMPTED = True
    path = artifact_path()
    if not path.exists():
        return None
    try:
        art = ForwardNetArtifact.from_dict(json.loads(path.read_text()))
    except Exception:
        return None
    _LOADED = art
    return art if art.accepted else None


def _measurement_source(row: dict[str, Any]) -> dict[str, Any]:
    meta = row.get("rank_meta") or {}
    meas = meta.get("setup_measurements") or row.get("setup_measurements") or {}
    if not meas:
        sig = row.get("signal")
        ctx = getattr(sig, "setup_context", None) if sig is not None else None
        if isinstance(ctx, dict):
            meas = {"best": ctx.get("features") or {}}
    return meas if isinstance(meas, dict) else {}


def _has_usable_measurements(meas: dict[str, Any]) -> bool:
    for block in meas.values():
        if not isinstance(block, dict):
            continue
        for key, val in block.items():
            if key in ("passed", "hard_block", "entry_eligible"):
                continue
            try:
                if abs(float(val)) > 0.0:
                    return True
            except (TypeError, ValueError):
                continue
    return False


def _runtime_features_present(row: dict[str, Any], names: list[str]) -> bool:
    """True only when the artifact has the market/features it was trained on."""
    meta = row.get("rank_meta") or {}
    bars = meta.get("bars_1m") or row.get("bars_1m") or []
    uses_bars = any(str(n).startswith("ret_") or str(n).startswith("evt_") for n in names)
    if uses_bars:
        return isinstance(bars, list) and len(bars) >= 8
    return _has_usable_measurements(_measurement_source(row))


def _features_for_runtime_row(row: dict[str, Any], names: list[str]) -> dict[str, float]:
    meas = _measurement_source(row)
    bars = (row.get("rank_meta") or {}).get("bars_1m") or row.get("bars_1m") or []
    if bars and any(n.startswith("ret_") or n.startswith("evt_") for n in names):
        from backend.services.binance_scalp.reconstructable_features import reconstructable_features

        proj = 0.0
        if isinstance(meas, dict):
            for block in meas.values():
                if isinstance(block, dict) and block.get("projected_move"):
                    proj = max(proj, float(block["projected_move"] or 0.0))
        ts = None
        last = bars[-1] if bars else {}
        raw_ts = last.get("ts") if isinstance(last, dict) else None
        if raw_ts is not None:
            try:
                ts = raw_ts if hasattr(raw_ts, "hour") else datetime.fromtimestamp(float(raw_ts), tz=timezone.utc)
            except Exception:
                ts = None
        return reconstructable_features(bars, projected_move=proj, ts=ts)
    return flatten_measurements(meas, live_book=True)


def predict_row_expected_net(row: dict[str, Any]) -> float | None:
    """Runtime BUY EV from the accepted artifact, or None if none accepted.

    Missing/empty features are HOLD (0), not the intercept. Same fail-closed
    rule as DAY resolve_day_path_ev.
    """
    art = load_accepted_artifact()
    if art is None:
        return None
    if not _runtime_features_present(row, art.feature_names):
        return 0.0
    feats = _features_for_runtime_row(row, art.feature_names)
    pred = predict_artifact(art, feats)
    return float(pred["predicted_net_ev"])


def save_artifact(art: ForwardNetArtifact, path: Path | None = None) -> Path:
    dest = path or artifact_path()
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(art.to_dict(), indent=2))
    return dest
