"""SCALP microstructure feature contract — parallel to the 40-dim scalp vector.

Does not replace ``scalp_feature_contract.SCALP_FEATURE_DIM=40``.
Old and new outcomes stay separable via version stamps.
"""

from __future__ import annotations

from typing import Any

# Existing 40-dim contract stays SCALP_FEATURE_VERSION=1.
SCALP_FEATURE_VERSION = 1
MICROSTRUCTURE_VERSION = "scalp_micro_v1"
SELECTION_VERSION = "scalp_micro_select_v2"
SELECTION_VERSION_V1 = "scalp_micro_select_v1"
MODEL_VERSION = "scalp_micro_ev_v1"
FEATURE_SET_VERSION = "scalp_micro_features_v1"

MARKOUT_HORIZONS_SEC: tuple[int, ...] = (1, 2, 5, 10, 20, 30, 60, 120)
EV_HORIZONS_SEC: tuple[int, ...] = (1, 5, 10, 30, 60)

# NAME, SOURCE, WINDOW, NORMALIZATION, EXPECTED DIRECTION (+ long), MISSING
MICRO_FEATURE_SPEC: tuple[dict[str, str], ...] = (
    {"name": "obi_l1", "source": "local_l2", "window": "snapshot", "norm": "signed[-1,1]", "dir": "+", "missing": "0"},
    {"name": "obi_l5", "source": "local_l2", "window": "snapshot", "norm": "signed[-1,1]", "dir": "+", "missing": "0"},
    {"name": "obi_l10", "source": "local_l2", "window": "snapshot", "norm": "signed[-1,1]", "dir": "+", "missing": "0"},
    {"name": "obi_l20", "source": "local_l2", "window": "snapshot", "norm": "signed[-1,1]", "dir": "+", "missing": "0"},
    {"name": "weighted_depth_imbalance", "source": "local_l2", "window": "snapshot", "norm": "signed[-1,1]", "dir": "+", "missing": "0"},
    {"name": "l1_liquidity_ratio", "source": "local_l2", "window": "snapshot", "norm": "bid/ask", "dir": "+", "missing": "0"},
    {"name": "spread_pct", "source": "local_l2", "window": "snapshot", "norm": "frac", "dir": "-", "missing": "omit"},
    {"name": "microprice_pressure", "source": "local_l2", "window": "snapshot", "norm": "frac vs mid", "dir": "+", "missing": "0"},
    {"name": "microprice_accel", "source": "local_l2", "window": "5s", "norm": "delta frac", "dir": "+", "missing": "0"},
    {"name": "ofi_100ms", "source": "snapshot_ofi", "window": "100ms", "norm": "depth-rel", "dir": "+", "missing": "0"},
    {"name": "ofi_1s", "source": "snapshot_ofi", "window": "1s", "norm": "depth-rel", "dir": "+", "missing": "0"},
    {"name": "ofi_3s", "source": "snapshot_ofi", "window": "3s", "norm": "depth-rel", "dir": "+", "missing": "0"},
    {"name": "ofi_5s", "source": "snapshot_ofi", "window": "5s", "norm": "depth-rel", "dir": "+", "missing": "0"},
    {"name": "ofi_15s", "source": "snapshot_ofi", "window": "15s", "norm": "depth-rel", "dir": "+", "missing": "0"},
    {"name": "ofi_30s", "source": "snapshot_ofi", "window": "30s", "norm": "depth-rel", "dir": "+", "missing": "0"},
    {"name": "agg_flow_imbalance_1s", "source": "aggTrade", "window": "1s", "norm": "signed[-1,1]", "dir": "+", "missing": "0"},
    {"name": "agg_flow_imbalance_5s", "source": "aggTrade", "window": "5s", "norm": "signed[-1,1]", "dir": "+", "missing": "0"},
    {"name": "agg_flow_imbalance_15s", "source": "aggTrade", "window": "15s", "norm": "signed[-1,1]", "dir": "+", "missing": "0"},
    {"name": "signed_volume_5s", "source": "aggTrade", "window": "5s", "norm": "qty", "dir": "+", "missing": "0"},
    {"name": "trade_count_5s", "source": "aggTrade", "window": "5s", "norm": "count", "dir": "intensity", "missing": "0"},
    {"name": "avg_trade_size_5s", "source": "aggTrade", "window": "5s", "norm": "qty", "dir": "intensity", "missing": "0"},
    {"name": "flow_acceleration", "source": "aggTrade", "window": "1s-5s", "norm": "delta imb", "dir": "+", "missing": "0"},
    {"name": "bid_cancelled_5s", "source": "snapshot_delta", "window": "5s", "norm": "qty", "dir": "-", "missing": "0"},
    {"name": "ask_cancelled_5s", "source": "snapshot_delta", "window": "5s", "norm": "qty", "dir": "+", "missing": "0"},
    {"name": "bid_replenished_5s", "source": "snapshot_delta", "window": "5s", "norm": "qty", "dir": "+", "missing": "0"},
    {"name": "ask_replenished_5s", "source": "snapshot_delta", "window": "5s", "norm": "qty", "dir": "-", "missing": "0"},
    {"name": "cancel_imbalance_5s", "source": "snapshot_delta", "window": "5s", "norm": "signed[-1,1]", "dir": "+", "missing": "0"},
    {"name": "bid_absorption_score", "source": "flow+book", "window": "5s", "norm": "[0,1]", "dir": "+", "missing": "0"},
    {"name": "ask_absorption_score", "source": "flow+book", "window": "5s", "norm": "[0,1]", "dir": "-", "missing": "0"},
    {"name": "depth_fragility", "source": "flow+book", "window": "5s", "norm": "[0,1]", "dir": "-", "missing": "0"},
    {"name": "near_touch_depth_loss", "source": "local_l2", "window": "5s", "norm": "[0,1]", "dir": "-", "missing": "0"},
    {"name": "adverse_selection_score", "source": "composite", "window": "5s", "norm": "[0,1]", "dir": "-", "missing": "0"},
    {"name": "p_adverse_move", "source": "composite", "window": "5s", "norm": "[0,1]", "dir": "-", "missing": "0"},
    {"name": "cross_venue_dislocation_bps", "source": "coinbase_rest", "window": "cache30s", "norm": "bps", "dir": "mean-revert", "missing": "unavailable"},
    {"name": "spot_perp_basis_bps", "source": "binance_futures_rest", "window": "cache30s", "norm": "bps", "dir": "lead/lag", "missing": "unavailable"},
)

MICRO_FEATURE_NAMES: tuple[str, ...] = tuple(s["name"] for s in MICRO_FEATURE_SPEC)


def extract_micro_vector(feats: dict[str, Any] | None) -> list[float]:
    src = feats or {}
    out: list[float] = []
    for name in MICRO_FEATURE_NAMES:
        raw = src.get(name)
        try:
            out.append(float(raw) if raw is not None else 0.0)
        except (TypeError, ValueError):
            out.append(0.0)
    return out


def version_stamps() -> dict[str, str]:
    return {
        "feature_version": str(SCALP_FEATURE_VERSION),
        "microstructure_version": MICROSTRUCTURE_VERSION,
        "selection_version": SELECTION_VERSION,
        "model_version": MODEL_VERSION,
        "feature_set_version": FEATURE_SET_VERSION,
    }


MARKOUT_FEATURE_KEYS: tuple[str, ...] = (
    "ofi_1s",
    "ofi_3s",
    "ofi_5s",
    "ofi_15s",
    "ofi_30s",
    "obi_l1",
    "obi_l5",
    "obi_l10",
    "obi_l20",
    "microprice_pressure",
    "agg_flow_imbalance_1s",
    "agg_flow_imbalance_5s",
    "trade_count_1s",
    "trade_count_5s",
    "flow_acceleration",
    "signed_volume_5s",
    "bid_cancelled_5s",
    "ask_cancelled_5s",
    "bid_replenished_5s",
    "ask_replenished_5s",
    "bid_absorption_score",
    "ask_absorption_score",
    "depth_fragility",
    "adverse_selection_score",
    "p_adverse_move",
    "spread_pct",
    "ranking_delta",
    "cross_venue_dislocation_bps",
    "spot_perp_basis_bps",
)


def _f(raw: Any, default: float = 0.0) -> float:
    try:
        return float(raw)
    except (TypeError, ValueError):
        return default


def label_ofi_bucket(ofi: float) -> str:
    if ofi >= 1.0:
        return "strongly_positive"
    if ofi >= 0.15:
        return "positive"
    if ofi > -0.15:
        return "neutral"
    if ofi > -1.0:
        return "negative"
    return "strongly_negative"


def label_obi_bucket(obi: float) -> str:
    if obi >= 0.50:
        return "strongly_positive"
    if obi >= 0.15:
        return "positive"
    if obi > -0.15:
        return "neutral"
    if obi > -0.50:
        return "negative"
    return "strongly_negative"


def label_microprice_bucket(pressure: float) -> str:
    if pressure > 1e-6:
        return "favorable"
    if pressure < -1e-6:
        return "adverse"
    return "neutral"


def label_adverse_bucket(score: float) -> str:
    if score >= 0.45:
        return "high"
    if score >= 0.20:
        return "medium"
    return "low"


def label_agg_flow_bucket(flow: float) -> str:
    if flow > 0.15:
        return "buying"
    if flow < -0.15:
        return "selling"
    return "neutral"


def label_absorption_bucket(bid_abs: float, ask_abs: float) -> str:
    net = bid_abs - ask_abs
    if net > 0.15:
        return "supportive"
    if net < -0.15:
        return "adverse"
    return "neutral"


def feature_context_extra(feats: dict[str, Any] | None, extra: dict[str, Any] | None = None) -> dict[str, Any]:
    """Dataset labels + raw features for markouts. Never a permission gate."""
    src = dict(feats or {})
    out = dict(extra or {})
    out.update(version_stamps())
    for key in MARKOUT_FEATURE_KEYS:
        if key in src and src[key] is not None:
            out[key] = src[key]
    ofi = _f(out.get("ofi_5s") if out.get("ofi_5s") is not None else src.get("ofi_5s"))
    obi = _f(out.get("obi_l5") if out.get("obi_l5") is not None else src.get("obi_l5") or src.get("obi_l1"))
    pressure = _f(out.get("microprice_pressure") if out.get("microprice_pressure") is not None else src.get("microprice_pressure"))
    adverse = _f(out.get("adverse_selection_score") if out.get("adverse_selection_score") is not None else src.get("adverse_selection_score"))
    flow = _f(out.get("agg_flow_imbalance_5s") if out.get("agg_flow_imbalance_5s") is not None else src.get("agg_flow_imbalance_5s"))
    out["ofi_bucket"] = label_ofi_bucket(ofi)
    out["obi_bucket"] = label_obi_bucket(obi)
    out["microprice_bucket"] = label_microprice_bucket(pressure)
    out["adverse_bucket"] = label_adverse_bucket(adverse)
    out["agg_flow_bucket"] = label_agg_flow_bucket(flow)
    out["absorption_bucket"] = label_absorption_bucket(
        _f(src.get("bid_absorption_score")),
        _f(src.get("ask_absorption_score")),
    )
    if src.get("symbol") is not None:
        out.setdefault("symbol", src.get("symbol"))
    if src.get("ts") is not None:
        out.setdefault("decision_ts", src.get("ts"))
    return out


def buy_microstructure_invariant_violations(diag: dict[str, Any] | None) -> list[str]:
    """Return missing/wrong BUY stamp fields. Empty list means the invariant holds."""
    d = dict(diag or {})
    setup = (d.get("setup") or {}).get("setup_context") or d.get("setup_context") or {}
    violations: list[str] = []
    micro_ver = d.get("microstructure_version") or setup.get("microstructure_version")
    feature_set = d.get("feature_set_version") or setup.get("feature_set_version")
    model = d.get("model_version") or setup.get("model_version")
    feats = d.get("microstructure_features") or setup.get("microstructure_features") or {}
    if micro_ver != MICROSTRUCTURE_VERSION:
        violations.append(f"microstructure_version={micro_ver!r}")
    if feature_set != FEATURE_SET_VERSION:
        violations.append(f"feature_set_version={feature_set!r}")
    if model != MODEL_VERSION:
        violations.append(f"model_version={model!r}")
    if not isinstance(feats, dict) or not feats:
        violations.append("microstructure_features_empty")
    ev10 = d.get("EV_10s") if d.get("EV_10s") is not None else setup.get("EV_10s")
    if ev10 is None:
        violations.append("EV_10s_missing")
    return violations


__all__ = [
    "EV_HORIZONS_SEC",
    "FEATURE_SET_VERSION",
    "MARKOUT_FEATURE_KEYS",
    "MARKOUT_HORIZONS_SEC",
    "MICROSTRUCTURE_VERSION",
    "MICRO_FEATURE_NAMES",
    "MICRO_FEATURE_SPEC",
    "MODEL_VERSION",
    "SCALP_FEATURE_VERSION",
    "SELECTION_VERSION",
    "SELECTION_VERSION_V1",
    "buy_microstructure_invariant_violations",
    "extract_micro_vector",
    "feature_context_extra",
    "label_absorption_bucket",
    "label_adverse_bucket",
    "label_agg_flow_bucket",
    "label_microprice_bucket",
    "label_obi_bucket",
    "label_ofi_bucket",
    "version_stamps",
]
