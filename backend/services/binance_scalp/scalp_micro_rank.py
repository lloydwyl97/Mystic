"""Deterministic SCALP ranking repair (select_v2, authoritative).

PRIMARY SORT = decision-time EV_10s.
Static/setup/intel are tie-break only when EV_10s is numerically tied.
Never blend static back into EV. Never a permission gate.

DAY ranking delta is intentionally unchanged.
"""

from __future__ import annotations

from typing import Any

from backend.services.binance_scalp.scalp_micro_contract import SELECTION_VERSION
from backend.services.binance_scalp.scalp_micro_ev import heuristic_horizon_ev

# No additive blend. Observed 0.25 bp EV gaps (~2.5e-6) were reversed by 1e-5 * static.
TIEBREAK_SCALE = 0.0
EV_TIE_TOLERANCE = 1e-12
RANK_PRIMARY = "EV_10s"
AUTHORITATIVE_SELECTION = True


def _f(d: dict[str, Any] | None, key: str, default: float = 0.0) -> float:
    src = d or {}
    raw = src.get(key)
    if raw is None:
        return default
    try:
        return float(raw)
    except (TypeError, ValueError):
        return default


def ev_scores_tied(a: float, b: float, *, tol: float = EV_TIE_TOLERANCE) -> bool:
    return abs(float(a) - float(b)) <= float(tol)


def repaired_primary_score(feats: dict[str, Any] | None, micro_ev: dict[str, Any] | None = None) -> float:
    """Decision-time primary rank. Existing heuristic EV_10s only. No markouts."""
    ev = None
    if micro_ev and micro_ev.get("EV_10s") is not None:
        try:
            ev = float(micro_ev["EV_10s"])
        except (TypeError, ValueError):
            ev = None
    if ev is None:
        ev = float(heuristic_horizon_ev(feats, 10))
    return ev


def rank_components(feats: dict[str, Any] | None, *, ev10: float, static_rank: float, tiebreak: float) -> dict[str, Any]:
    f = feats or {}
    return {
        "selection_version": SELECTION_VERSION,
        "primary": RANK_PRIMARY,
        "authoritative_selection": AUTHORITATIVE_SELECTION,
        "EV_10s": round(float(ev10), 8),
        "agg_flow_imbalance_5s": _f(f, "agg_flow_imbalance_5s"),
        "ofi_5s": _f(f, "ofi_5s"),
        "microprice_pressure": _f(f, "microprice_pressure"),
        "obi_l5": _f(f, "obi_l5"),
        "adverse_selection_score": _f(f, "adverse_selection_score"),
        "net_absorption": _f(f, "bid_absorption_score") - _f(f, "ask_absorption_score"),
        "depth_fragility": _f(f, "depth_fragility"),
        "static_rank": round(float(static_rank), 6),
        "static_tiebreak": round(float(tiebreak), 12),
        "obi_standalone": "neutralized",
        "absorption_standalone": "neutralized",
        "fragility_standalone": "neutralized",
        "eligibility_effect": False,
    }


def apply_repaired_rank(
    *,
    static_rank: float,
    feats: dict[str, Any] | None,
    live_ctx_adj: float = 0.0,
    learned_adj: float = 0.0,
    micro_learn_adj: float = 0.0,
    feature_adj: float = 0.0,
    micro_ev: dict[str, Any] | None = None,
) -> tuple[float, float, dict[str, Any]]:
    """Return (final_rank, micro_adj_diagnostic, components).

    final_rank is EV_10s only. Residual is recorded, never added.
    Does not change eligibility.
    """
    primary = repaired_primary_score(feats, micro_ev)
    residual = float(static_rank) + float(live_ctx_adj) + float(learned_adj) + float(micro_learn_adj) + float(feature_adj)
    tie = TIEBREAK_SCALE * residual
    comps = rank_components(feats, ev10=primary, static_rank=static_rank, tiebreak=tie)
    return round(float(primary), 8), round(float(primary), 8), comps


__all__ = [
    "AUTHORITATIVE_SELECTION",
    "EV_TIE_TOLERANCE",
    "RANK_PRIMARY",
    "TIEBREAK_SCALE",
    "apply_repaired_rank",
    "ev_scores_tied",
    "rank_components",
    "repaired_primary_score",
]
