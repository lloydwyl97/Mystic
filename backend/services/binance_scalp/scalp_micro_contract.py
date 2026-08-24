"""SCALP microstructure feature contract — parallel to the 40-dim scalp vector.

Does not replace ``scalp_feature_contract.SCALP_FEATURE_DIM=40``.
Old and new outcomes stay separable via version stamps.
"""

from __future__ import annotations

from typing import Any

# Existing 40-dim contract stays SCALP_FEATURE_VERSION=1.
SCALP_FEATURE_VERSION = 1
MICROSTRUCTURE_VERSION = "scalp_micro_v1"
SELECTION_VERSION = "scalp_micro_select_v1"
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


__all__ = [
    "EV_HORIZONS_SEC",
    "FEATURE_SET_VERSION",
    "MARKOUT_HORIZONS_SEC",
    "MICROSTRUCTURE_VERSION",
    "MICRO_FEATURE_NAMES",
    "MICRO_FEATURE_SPEC",
    "MODEL_VERSION",
    "SCALP_FEATURE_VERSION",
    "SELECTION_VERSION",
    "extract_micro_vector",
    "version_stamps",
]
