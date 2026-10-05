"""SCALP canonical executable net edge.

One number decides SCALP entry eligibility, rank and size. It starts from the
current candidate and the live book:

    calibrated_move       = setup gross mean + claim_capture x (raw projection - projection mean)
    base_executable_edge  = calibrated_move - live cost (fees, slippage, spread, impact)
    final_executable_edge = base_executable_edge + adaptive_residual + micro_residual

The raw projection is the strategy's structural target distance. It is a
feature, never the expected move: the gross mean and ``claim_capture`` are
learned from realized gross moves of claims (admitted or not) with zero priors,
so geometry, ATR or a floor alone produces an expected move of 0 and a
candidate priced at minus its cost. ``adaptive_residual`` is the key's gross
mean against its setup's, bounded. ``micro_residual`` is the bounded
microstructure tilt times its learned weight in [0, 1]; it may lower a
candidate but never lifts one that is not already positive. All learned terms
are read from state of the current economic version, pooled hierarchically.

``final_executable_edge <= 0`` is hard economic safety (NO_EXECUTABLE_NET_EDGE).
Anything above zero stays eligible. Confidence, risk and the final edge set
size; confidence is never a permission input and there is no sample-count floor.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from typing import Any

from backend.config.trading_economics import canonical_roundtrip_cost_pct

REJECT_THRESHOLD_PCT: float = 0.0
_RISK_FLOOR_PCT: float = 0.0005


@dataclass(frozen=True)
class ExecutableEdge:
    raw_expected_move_pct: float
    raw_move_source: str
    live_cost_pct: float
    base_executable_edge_pct: float
    adaptive_residual_pct: float
    micro_residual_model_pct: float
    micro_residual_pct: float
    final_executable_edge_pct: float
    confidence: float
    n_residual: float
    target_pct: float
    hold_min: float
    risk_estimate_pct: float
    size_mult: float
    spread_pct: float
    impact_pct: float
    reject_threshold_pct: float = REJECT_THRESHOLD_PCT
    claim_capture: float = 0.0
    calibrated_move_pct: float = 0.0
    micro_weight: float = 1.0
    micro_residual_unweighted_pct: float = 0.0
    uncertainty_pct: float = 0.0
    economic_version: str = ""

    @property
    def eligible(self) -> bool:
        return self.final_executable_edge_pct > self.reject_threshold_pct

    @property
    def pre_micro_edge_pct(self) -> float:
        return self.final_executable_edge_pct - self.micro_residual_pct

    @property
    def expected_move_pct(self) -> float:
        """Calibrated expected directional move before micro: setup level plus key residual."""
        return self.calibrated_move_pct + self.adaptive_residual_pct

    def economic(self) -> dict[str, Any]:
        """Expected net edge after costs, in the language DAY uses as well."""
        return {
            "engine_id": "SCALP_V2",
            "economic_version": self.economic_version,
            "raw_claim": self.raw_expected_move_pct,
            "calibrated_move": self.calibrated_move_pct,
            "calibration_adjustment": self.expected_move_pct - self.raw_expected_move_pct,
            "claim_capture": self.claim_capture,
            "adaptive_residual": self.adaptive_residual_pct,
            "setup_expected_edge": self.base_executable_edge_pct,
            "expected_gross": self.edge_before_cost_pct,
            "expected_cost": self.live_cost_pct,
            "adaptive_correction": self.edge_before_cost_pct - self.raw_expected_move_pct,
            "micro_residual": self.micro_residual_pct,
            "micro_weight": self.micro_weight,
            "micro_unweighted": self.micro_residual_unweighted_pct,
            "pre_micro_edge": self.pre_micro_edge_pct,
            "uncertainty": self.uncertainty_pct,
            "expected_net_edge": self.final_executable_edge_pct,
            "size_effect": self.size_mult - 1.0,
            "eligible": self.eligible,
        }

    @property
    def deficit_to_zero_pct(self) -> float:
        return max(0.0, self.reject_threshold_pct - self.final_executable_edge_pct)

    @property
    def edge_before_cost_pct(self) -> float:
        return self.final_executable_edge_pct + self.live_cost_pct

    def as_dict(self) -> dict[str, Any]:
        out = asdict(self)
        out["expected_move_pct"] = self.expected_move_pct
        out["edge_before_cost_pct"] = self.edge_before_cost_pct
        out["deficit_to_zero_pct"] = self.deficit_to_zero_pct
        out["reject_deficit_pct"] = self.deficit_to_zero_pct
        out["eligible"] = self.eligible
        out["economic"] = self.economic()
        return out


def _f(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


def scalp_size_mult(final_edge_pct: float, confidence: float, risk_estimate_pct: float, live_cost_pct: float) -> float:
    """Size from the final edge per unit of adverse risk, scaled down by low confidence.

    Same SCALP bounds as before. Low confidence reduces size; it never blocks.
    """
    from backend.services.adaptive_learning import SCALP_ENGINE, SIZE_BOUNDS

    lo, hi = SIZE_BOUNDS[SCALP_ENGINE]
    risk = max(_RISK_FLOOR_PCT, _f(risk_estimate_pct) + _f(live_cost_pct))
    tilt = math.tanh(_f(final_edge_pct) / risk)
    conf = max(0.0, min(1.0, _f(confidence)))
    return max(lo, min(hi, (1.0 + 0.40 * tilt) * (0.75 + 0.25 * conf)))


def scalp_executable_edge(
    view: dict[str, Any],
    *,
    raw_expected_move_pct: float,
    spread_pct: float | None,
    impact_pct: float,
    edge_source: str,
) -> ExecutableEdge:
    """Canonical SCALP executable net edge for the current candidate at the live book.

    ``raw_expected_move_pct`` is the strategy's structural projection
    (``raw_move_source`` must be a directional claim; a volatility magnitude is
    refused). It enters only through the learned calibration. ``view`` is
    ``adaptive_learning.scalp_decision(...)`` computed with the live
    microstructure features; it supplies the calibration, micro residual,
    confidence and risk.
    """
    from backend.services.adaptive_learning import scalp_expected_gross
    from backend.services.scalp_v2.raw_move_source import is_directional, normalize_raw_move_source

    source = normalize_raw_move_source(edge_source)
    if not is_directional(source):
        raise ValueError(f"raw move source {source} is not a directional claim")
    raw = max(0.0, _f(raw_expected_move_pct))
    cost = canonical_roundtrip_cost_pct(spread_pct=spread_pct, buy_impact_pct=max(0.0, _f(impact_pct)), sell_impact_pct=0.0)
    move = scalp_expected_gross(view, raw)
    base = move["calibrated"] - cost
    pre_micro = base + move["adaptive"]
    micro_model = _f(view.get("micro_residual"))
    micro = micro_model if (pre_micro > REJECT_THRESHOLD_PCT or micro_model < 0) else 0.0
    final = pre_micro + micro
    confidence = _f(view.get("confidence"))
    risk = _f(view.get("risk_estimate"))
    return ExecutableEdge(
        raw_expected_move_pct=raw,
        raw_move_source=source,
        live_cost_pct=cost,
        base_executable_edge_pct=base,
        adaptive_residual_pct=move["adaptive"],
        micro_residual_model_pct=micro_model,
        micro_residual_pct=micro,
        final_executable_edge_pct=final,
        confidence=confidence,
        n_residual=_f(view.get("n_claim")),
        target_pct=_f(view.get("target_pct")),
        hold_min=_f(view.get("hold_min")),
        risk_estimate_pct=risk,
        size_mult=scalp_size_mult(final, confidence, risk, cost),
        spread_pct=_f(spread_pct),
        impact_pct=max(0.0, _f(impact_pct)),
        claim_capture=move["capture"],
        calibrated_move_pct=move["calibrated"],
        micro_weight=max(0.0, min(1.0, _f(view.get("micro_weight"), 1.0))),
        micro_residual_unweighted_pct=_f(view.get("micro_residual_unweighted"), micro_model),
        uncertainty_pct=_f(view.get("claim_uncertainty")),
        economic_version=str(view.get("economic_version") or ""),
    )


def stamp_view(view: dict[str, Any], edge: ExecutableEdge) -> dict[str, Any]:
    """Adaptive view as stamped on the order: size and edge are the canonical values."""
    out = dict(view)
    out["size_mult"] = edge.size_mult
    out["final_executable_edge"] = edge.final_executable_edge_pct
    out["executable_edge"] = edge.as_dict()
    return out


_DETAIL_KEYS: tuple[tuple[str, str], ...] = (
    ("raw_expected_move", "raw_expected_move_pct"),
    ("raw_move_source", "raw_move_source"),
    ("live_cost", "live_cost_pct"),
    ("base_executable_edge", "base_executable_edge_pct"),
    ("adaptive_residual", "adaptive_residual_pct"),
    ("claim_capture", "claim_capture"),
    ("calibrated_move", "calibrated_move_pct"),
    ("expected_move", "expected_move_pct"),
    ("micro_residual", "micro_residual_pct"),
    ("micro_residual_model", "micro_residual_model_pct"),
    ("micro_weight", "micro_weight"),
    ("micro_residual_unweighted", "micro_residual_unweighted_pct"),
    ("uncertainty", "uncertainty_pct"),
    ("economic_version", "economic_version"),
    ("final_executable_edge", "final_executable_edge_pct"),
    ("confidence", "confidence"),
    ("n_residual", "n_residual"),
    ("target", "target_pct"),
    ("hold", "hold_min"),
    ("risk_estimate", "risk_estimate_pct"),
    ("size_mult", "size_mult"),
    ("edge_before_cost", "edge_before_cost_pct"),
    ("cost", "live_cost_pct"),
    ("edge_after_cost", "final_executable_edge_pct"),
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
        payload["raw_move_source"] = str(meta.get("edge_source") or "NONE")
        payload["hard_block"] = str(row.get("hard_block") or meta.get("hard_block") or "")
    vol = (row.get("rank_meta") or {}).get("volatility_move_pct")
    if isinstance(vol, (int, float)):
        payload["volatility_move"] = round(float(vol), 8)
    payload["setup"] = str(row.get("best_setup") or "")
    payload["regime"] = str(row.get("adaptive_regime") or "")
    payload.update({k: v for k, v in extra.items() if v is not None})
    return json.dumps(payload, separators=(",", ":"), default=str)


__all__ = [
    "REJECT_THRESHOLD_PCT",
    "ExecutableEdge",
    "decision_detail",
    "scalp_executable_edge",
    "scalp_size_mult",
    "stamp_view",
]
