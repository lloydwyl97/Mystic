"""SCALP candidate explanation snapshots (rank telemetry only)."""

from __future__ import annotations

import json
from typing import Any


def build_scalp_candidate_explanation(data: dict[str, Any], *, symbol: str = "") -> dict[str, Any]:
    dd = dict(data or {})
    sym = symbol or str(dd.get("symbol") or "")
    setup = str(dd.get("scalp_setup") or dd.get("setup_name") or "UNKNOWN")
    regime = str(dd.get("micro_regime") or "unknown")
    blocks = {k: dd.get(k) for k in dd if k.startswith("scalp_") and k.endswith("_score")}
    snap = {
        "symbol": sym,
        "model_probabilities": {
            "signal_score": dd.get("signal_score"),
            "signal_confidence": dd.get("signal_confidence"),
            "required_target_pct": dd.get("required_target_pct"),
        },
        "micro_regime": regime,
        "scalp_setup": setup,
        "setup_score": dd.get("setup_score"),
        "block_scores": blocks,
        "feature_health_score": dd.get("scalp_feature_health_score") or dd.get("feature_health_score"),
        "relative_strength_rank": dd.get("scalp_rs_rank") or dd.get("relative_strength_rank"),
        "execution_quality_score": dd.get("scalp_execution_quality_score"),
        "spread_score": dd.get("scalp_spread_score"),
        "depth_score": dd.get("scalp_depth_score"),
        "price_impact_score": dd.get("scalp_price_impact_score"),
        "adaptive_learning": {
            "scalp_adaptive_rank_delta": dd.get("scalp_adaptive_rank_delta"),
            "learning_bucket": dd.get("scalp_learning_bucket"),
        },
        "final_scalp_selection_score": dd.get("final_scalp_selection_score"),
        "rank_deltas": {
            "block_score_rank_delta": dd.get("block_score_rank_delta"),
            "setup_score_rank_delta": dd.get("setup_score_rank_delta"),
            "execution_rank_delta": dd.get("execution_rank_delta"),
            "regime_transition_rank_delta": dd.get("regime_transition_rank_delta"),
            "memory_rank_delta": dd.get("memory_rank_delta"),
        },
        "skipped_reason": dd.get("skipped_reason") or "",
        "why_selected": dd.get("why_selected") or "",
    }
    snap["narrative"] = format_scalp_narrative(snap)
    return snap


def format_scalp_narrative(snap: dict[str, Any]) -> str:
    sym = snap.get("symbol") or "?"
    rank = snap.get("relative_strength_rank") or "?"
    setup = snap.get("scalp_setup") or "unknown"
    regime = snap.get("micro_regime") or "unknown"
    exec_q = snap.get("execution_quality_score")
    exec_s = f"{float(exec_q)*100:.0f}%" if exec_q is not None else "n/a"
    final = snap.get("final_scalp_selection_score")
    final_s = f"{float(final):.4f}" if final is not None else "n/a"
    skip = snap.get("skipped_reason") or ""
    if skip:
        return f"{sym} skipped for {setup}: execution too expensive ({skip})."
    return (
        f"{sym} ranked #{rank} for {setup} in {regime} micro-regime; "
        f"execution {exec_s}, final_scalp_selection_score {final_s}."
    )


def explanation_json(data: dict[str, Any], *, symbol: str = "") -> str:
    return json.dumps(build_scalp_candidate_explanation(data, symbol=symbol), separators=(",", ":"))


__all__ = ["build_scalp_candidate_explanation", "explanation_json", "format_scalp_narrative"]
