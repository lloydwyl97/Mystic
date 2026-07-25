"""Orchestrator: enrich DAY candidate decision_data with AI intelligence rank inputs."""

from __future__ import annotations

import json
import logging
from typing import Any

from backend.services.day_block_scores import block_scores_rank_delta, compute_block_scores_from_decision_data
from backend.services.day_candidate_explanation import build_candidate_explanation, explanation_json
from backend.services.day_chart_pattern_detector import chart_pattern_rank_delta, get_chart_pattern_signal
from backend.services.day_cross_sectional_ranking import (
    combined_attractiveness_score,
    cross_sectional_rank_delta,
    publish_cross_sectional_score,
    read_peer_scores,
)
from backend.services.day_execution_ranking import compute_execution_ranking_scores, execution_rank_delta
from backend.services.day_market_memory import load_market_memory, memory_rank_delta, update_market_memory_on_candidate
from backend.services.day_regime_transition import compute_regime_transition_scores, regime_transition_rank_delta
from backend.services.day_setup_scores import compute_all_setup_scores, compute_setup_score, setup_score_rank_delta

logger = logging.getLogger(__name__)

INTELLIGENCE_DELTA_CAP = 0.10


def enrich_day_candidate_decision_data(
    decision_data: dict[str, Any],
    *,
    symbol: str,
    current_price: float | None = None,
    memory: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Stamp block/setup/execution/transition scores and bounded rank deltas. No gates."""
    dd = dict(decision_data or {})
    if current_price is not None:
        dd["current_price"] = float(current_price)

    block_scores = compute_block_scores_from_decision_data(dd)
    for k, v in block_scores.items():
        dd[k] = v

    setup = str(dd.get("setup_type") or dd.get("entry_thesis") or "NO_CLEAR_THESIS")
    setup_scores = compute_all_setup_scores(dd, block_scores)
    dd["setup_scores_json"] = json.dumps(setup_scores, separators=(",", ":"))
    dd["setup_score"] = compute_setup_score(setup, dd, block_scores)

    mem = memory or {}
    if mem.get("previous_regime"):
        dd["previous_regime"] = mem.get("previous_regime")
    transition = compute_regime_transition_scores(dd, mem)
    dd.update(transition)

    exec_scores = compute_execution_ranking_scores(dd)
    dd.update(exec_scores)

    try:
        pattern_info = get_chart_pattern_signal(symbol)
    except Exception as exc:
        logger.debug("chart pattern detection skipped %s: %s", symbol, exc)
        pattern_info = {"chart_pattern_score": 0.0, "chart_pattern_label": ""}
    dd.update(pattern_info)

    try:
        own_score = combined_attractiveness_score(dd)
        publish_cross_sectional_score(symbol, own_score)
        peer_scores = read_peer_scores(symbol)
        dd["cross_sectional_own_score"] = round(own_score, 4)
        dd["cross_sectional_peer_scores_json"] = json.dumps(peer_scores, separators=(",", ":"))
        cross_sectional_delta = cross_sectional_rank_delta(own_score, peer_scores)
    except Exception as exc:
        logger.debug("cross sectional ranking skipped %s: %s", symbol, exc)
        cross_sectional_delta = 0.0
    dd["cross_sectional_rank_delta"] = cross_sectional_delta

    dd["block_score_rank_delta"] = block_scores_rank_delta(block_scores)
    dd["setup_score_rank_delta"] = setup_score_rank_delta(setup, setup_scores)
    dd["execution_rank_delta"] = execution_rank_delta(exec_scores)
    dd["regime_transition_rank_delta"] = regime_transition_rank_delta(transition, setup)
    dd["memory_rank_delta"] = memory_rank_delta(mem, setup)
    dd["chart_pattern_rank_delta"] = chart_pattern_rank_delta(pattern_info)

    intel_delta = (
        float(dd.get("block_score_rank_delta") or 0.0)
        + float(dd.get("setup_score_rank_delta") or 0.0)
        + float(dd.get("execution_rank_delta") or 0.0)
        + float(dd.get("regime_transition_rank_delta") or 0.0)
        + float(dd.get("memory_rank_delta") or 0.0)
        + float(dd.get("chart_pattern_rank_delta") or 0.0)
        + float(dd.get("cross_sectional_rank_delta") or 0.0)
    )
    dd["intelligence_rank_delta"] = round(max(-INTELLIGENCE_DELTA_CAP, min(INTELLIGENCE_DELTA_CAP, intel_delta)), 4)

    try:
        snap = build_candidate_explanation(dd, symbol=symbol)
        dd["candidate_explanation_json"] = explanation_json(dd, symbol=symbol)
        dd["candidate_explanation_narrative"] = snap.get("narrative") or ""
    except Exception as exc:
        logger.debug("candidate explanation skipped %s: %s", symbol, exc)

    return dd


async def enrich_day_candidate_async(
    redis_client: Any,
    decision_data: dict[str, Any],
    *,
    symbol: str,
    current_price: float | None = None,
) -> dict[str, Any]:
    mem = await load_market_memory(redis_client, symbol)
    dd = enrich_day_candidate_decision_data(decision_data, symbol=symbol, current_price=current_price, memory=mem)
    if redis_client:
        try:
            await update_market_memory_on_candidate(redis_client, symbol, dd)
        except Exception:
            pass
    return dd


def apply_intelligence_rank_delta_to_candidate(candidate: Any) -> None:
    dd = dict(getattr(candidate, "decision_data", None) or {})
    delta = float(dd.get("intelligence_rank_delta") or 0.0)
    existing = float(dd.get("thesis_rank_delta") or 0.0)
    dd["thesis_rank_delta"] = round(existing + delta, 4)
    setattr(candidate, "decision_data", dd)


__all__ = [
    "apply_intelligence_rank_delta_to_candidate",
    "enrich_day_candidate_async",
    "enrich_day_candidate_decision_data",
]
