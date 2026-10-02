"""SCALP canonical executable net edge.

One number decides SCALP entry eligibility, rank and size. It is the adaptive
learned edge (realized trade net blended with cost-adjusted candidate
markouts, plus the bounded microstructure tilt) re-priced at the live book:

    edge_before_cost = learned edge after micro + cost basis the labels are net of
    edge_after_cost  = edge_before_cost - canonical cost(live spread, buy impact)

``edge_after_cost <= 0`` is hard economic safety (NO_EXECUTABLE_NET_EDGE).
Anything above zero stays eligible. Confidence shrinks the learned edge toward
its prior and therefore moves size and rank; it is never a permission input,
and there is no sample-count floor.

The strategy's own gross move claim (``raw_expected_move_pct``) is recorded for
diagnostics. It is not a second admission hurdle.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from typing import Any

from backend.config.trading_economics import canonical_roundtrip_cost_pct

REJECT_THRESHOLD_PCT: float = 0.0


def learned_edge_cost_basis_pct() -> float:
    """Flat round-trip cost the learned SCALP edge is already net of.

    Candidate markouts subtract the canonical cost at record time and realized
    trade net is fill-to-fill after fees and slippage, so the learned edge sits
    on the flat canonical basis (taker fees, slippage, default exit half-spread).
    """
    return canonical_roundtrip_cost_pct()


@dataclass(frozen=True)
class ExecutableEdge:
    raw_expected_move_pct: float
    canonical_cost_pct: float
    cost_basis_pct: float
    micro_tilt_pct: float
    edge_before_micro_pct: float
    edge_after_micro_pct: float
    edge_before_cost_pct: float
    edge_after_cost_pct: float
    edge_prior_pct: float
    confidence: float
    target_pct: float
    hold_min: float
    risk_estimate_pct: float
    size_mult: float
    spread_pct: float
    impact_pct: float
    reject_threshold_pct: float = REJECT_THRESHOLD_PCT

    @property
    def final_executable_net_edge_pct(self) -> float:
        return self.edge_after_cost_pct

    @property
    def eligible(self) -> bool:
        return self.edge_after_cost_pct > self.reject_threshold_pct

    @property
    def deficit_to_zero_pct(self) -> float:
        return max(0.0, self.reject_threshold_pct - self.edge_after_cost_pct)

    def as_dict(self) -> dict[str, Any]:
        out = asdict(self)
        out["final_executable_net_edge_pct"] = self.final_executable_net_edge_pct
        out["deficit_to_zero_pct"] = self.deficit_to_zero_pct
        out["reject_deficit_pct"] = self.deficit_to_zero_pct
        out["eligible"] = self.eligible
        return out


def _f(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


def scalp_size_mult(edge_pct: float, prior_pct: float) -> float:
    """Adaptive size from the canonical edge. Same bounds and tilt as the learned view."""
    from backend.services.adaptive_learning import SCALP_ENGINE, SIZE_BOUNDS

    lo, hi = SIZE_BOUNDS[SCALP_ENGINE]
    tilt = math.tanh((edge_pct - prior_pct) / abs(prior_pct)) if prior_pct else 0.0
    return max(lo, min(hi, 1.0 + 0.40 * tilt))


def scalp_executable_edge(
    view: dict[str, Any],
    *,
    raw_expected_move_pct: float,
    spread_pct: float | None,
    impact_pct: float,
) -> ExecutableEdge:
    """Canonical SCALP executable net edge from the adaptive view at the live book.

    ``view`` is ``adaptive_learning.scalp_decision(...)`` computed with the live
    microstructure features, so ``expected_edge`` already includes the tilt.
    """
    edge_after_micro = _f(view.get("expected_edge"))
    micro = _f(view.get("micro_tilt"))
    basis = learned_edge_cost_basis_pct()
    cost = canonical_roundtrip_cost_pct(spread_pct=spread_pct, buy_impact_pct=max(0.0, _f(impact_pct)), sell_impact_pct=0.0)
    edge_before_cost = edge_after_micro + basis
    edge_after_cost = edge_before_cost - cost
    prior = _f(view.get("edge_prior"))
    return ExecutableEdge(
        raw_expected_move_pct=_f(raw_expected_move_pct),
        canonical_cost_pct=cost,
        cost_basis_pct=basis,
        micro_tilt_pct=micro,
        edge_before_micro_pct=edge_after_micro - micro,
        edge_after_micro_pct=edge_after_micro,
        edge_before_cost_pct=edge_before_cost,
        edge_after_cost_pct=edge_after_cost,
        edge_prior_pct=prior,
        confidence=_f(view.get("confidence")),
        target_pct=_f(view.get("target_pct")),
        hold_min=_f(view.get("hold_min")),
        risk_estimate_pct=_f(view.get("risk_estimate")),
        size_mult=scalp_size_mult(edge_after_cost, prior),
        spread_pct=_f(spread_pct),
        impact_pct=max(0.0, _f(impact_pct)),
    )


def stamp_view(view: dict[str, Any], edge: ExecutableEdge) -> dict[str, Any]:
    """Adaptive view as stamped on the order: size and edge are the canonical values."""
    out = dict(view)
    out["size_mult_learned"] = out.get("size_mult")
    out["size_mult"] = edge.size_mult
    out["executable_net_edge"] = edge.edge_after_cost_pct
    out["executable_edge"] = edge.as_dict()
    return out


_DETAIL_KEYS: tuple[tuple[str, str], ...] = (
    ("raw_expected_move", "raw_expected_move_pct"),
    ("canonical_cost", "canonical_cost_pct"),
    ("micro_tilt", "micro_tilt_pct"),
    ("edge_before_micro", "edge_before_micro_pct"),
    ("edge_after_micro", "edge_after_micro_pct"),
    ("confidence", "confidence"),
    ("target", "target_pct"),
    ("hold", "hold_min"),
    ("risk_estimate", "risk_estimate_pct"),
    ("size_mult", "size_mult"),
    ("edge_before_cost", "edge_before_cost_pct"),
    ("cost", "canonical_cost_pct"),
    ("edge_after_cost", "edge_after_cost_pct"),
    ("final_executable_net_edge", "final_executable_net_edge_pct"),
    ("reject_threshold", "reject_threshold_pct"),
    ("reject_deficit", "deficit_to_zero_pct"),
    ("deficit_to_zero", "deficit_to_zero_pct"),
)


def decision_detail(row: dict[str, Any] | None, **extra: Any) -> str:
    """Compact JSON for scalp_v2_decisions.detail. Never empty."""
    payload: dict[str, Any] = {}
    row = row if isinstance(row, dict) else {}
    edge = row.get("executable_edge")
    if isinstance(edge, dict) and edge:
        for key, src in _DETAIL_KEYS:
            val = edge.get(src)
            payload[key] = round(float(val), 8) if isinstance(val, (int, float)) and not isinstance(val, bool) else val
    else:
        meta = row.get("rank_meta") or {}
        payload["edge_computed"] = False
        payload["raw_expected_move"] = round(_f(meta.get("expected_move_pct")), 8)
        payload["hard_block"] = str(row.get("hard_block") or meta.get("hard_block") or "")
    payload["setup"] = str(row.get("best_setup") or "")
    payload["regime"] = str(row.get("adaptive_regime") or "")
    payload.update({k: v for k, v in extra.items() if v is not None})
    return json.dumps(payload, separators=(",", ":"), default=str)


__all__ = [
    "REJECT_THRESHOLD_PCT",
    "ExecutableEdge",
    "decision_detail",
    "learned_edge_cost_basis_pct",
    "scalp_executable_edge",
    "scalp_size_mult",
    "stamp_view",
]
