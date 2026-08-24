"""Orchestrator: enrich SCALP candidate intelligence (rank/size only — no gates)."""

from __future__ import annotations

import json
import logging
import time
from typing import Any

from backend.services.scalp_block_scores import block_scores_rank_delta, compute_block_scores_from_intelligence
from backend.services.scalp_candidate_explanation import build_scalp_candidate_explanation, explanation_json
from backend.services.scalp_execution_ranking import compute_scalp_execution_scores, execution_rank_delta
from backend.services.scalp_feature_audit import build_symbol_scalp_audit
from backend.services.scalp_feature_contract import STRATEGY_TO_SCALP_SETUP, build_scalp_feature_vector
from backend.services.scalp_feature_health import stamp_feature_health
from backend.services.scalp_market_memory import load_scalp_market_memory_sync, memory_rank_delta, update_scalp_market_memory_on_candidate
from backend.services.scalp_regime_transition import compute_scalp_regime_transition_scores, regime_transition_rank_delta
from backend.services.scalp_setup_scores import compute_all_setup_scores, compute_setup_score, setup_score_rank_delta
from backend.services.scalp_strategy_score_weight_writer import scalp_adaptive_rank_delta

logger = logging.getLogger(__name__)

INTELLIGENCE_DELTA_CAP = 0.16


def _flatten_momentum(mom: Any) -> dict[str, Any]:
    if hasattr(mom, "as_dict"):
        return mom.as_dict()
    return dict(mom) if isinstance(mom, dict) else {}


def build_scalp_intelligence(
    *,
    symbol: str,
    snap: Any,
    mom: Any,
    signal: Any | None = None,
    bars_1m: list[dict] | None = None,
    micro_regime: str = "",
    redis_client: Any = None,
    gross: dict[str, Any] | None = None,
    audit: dict[str, Any] | None = None,
) -> dict[str, Any]:
    mem = load_scalp_market_memory_sync(redis_client, symbol) if redis_client else {}
    mom_d = _flatten_momentum(mom)
    sig_d = signal.as_dict() if hasattr(signal, "as_dict") else (signal if isinstance(signal, dict) else {})
    setup_name = STRATEGY_TO_SCALP_SETUP.get(str(sig_d.get("setup_name") or ""), str(sig_d.get("setup_name") or "UNKNOWN"))

    audit = audit or build_symbol_scalp_audit(symbol, snap=snap, mom_diag=mom)
    data: dict[str, Any] = {
        "symbol": symbol,
        "engine": "binance_scalp_paper",
        "setup_name": sig_d.get("setup_name"),
        "scalp_setup": setup_name,
        "micro_regime": micro_regime or audit.get("micro_regime") or "",
        "spread_pct": float(getattr(snap, "spread_pct", 0.0) or 0.0),
        "order_book_imbalance": float(getattr(snap, "order_book_imbalance", 0.0) or 0.0),
        "orderbook_age_sec": float(getattr(snap, "orderbook_age_sec", 0.0) or 0.0),
        "impact_pct": float(sig_d.get("impact_pct") or 0.0),
        "signal_score": float(sig_d.get("score") or 0.0),
        "signal_confidence": float(sig_d.get("confidence") or 0.0),
        "required_target_pct": float(sig_d.get("required_target_pct") or 0.0),
        "slippage_estimate": float(sig_d.get("impact_pct") or 0.0) * 0.5,
        **mom_d,
        **{f["name"]: f["value"] for f in (audit.get("features") or [])},
        "same_scalp_setup_today_count": mem.get("same_scalp_setup_today_count"),
        "recent_scalp_win_rate": mem.get("recent_scalp_win_rate"),
    }
    if getattr(snap, "orderbook_age_sec", None) and float(snap.orderbook_age_sec) > 45:
        data["freshness_trust_modifier"] = 0.25
    else:
        data["freshness_trust_modifier"] = 1.0

    data = stamp_feature_health(data, audit)
    blocks = compute_block_scores_from_intelligence(data)
    data.update(blocks)

    setup_scores = compute_all_setup_scores(data, blocks)
    data["setup_scores_json"] = json.dumps(setup_scores, separators=(",", ":"))
    data["setup_score"] = compute_setup_score(setup_name, data, blocks)

    transition = compute_scalp_regime_transition_scores(data, mem)
    data.update(transition)

    exec_scores = compute_scalp_execution_scores(data)
    data.update(exec_scores)

    data["block_score_rank_delta"] = block_scores_rank_delta(blocks)
    data["setup_score_rank_delta"] = setup_score_rank_delta(setup_name, setup_scores)
    data["execution_rank_delta"] = execution_rank_delta(exec_scores)
    data["regime_transition_rank_delta"] = regime_transition_rank_delta(transition, setup_name)
    data["memory_rank_delta"] = memory_rank_delta(mem, setup_name)
    data["scalp_adaptive_rank_delta"] = scalp_adaptive_rank_delta(data, symbol)
    data["scalp_learning_bucket"] = f"{data.get('micro_regime')}::{setup_name}"

    base = float(sig_d.get("score") or 0.0)
    intel = (
        float(data.get("block_score_rank_delta") or 0)
        + float(data.get("setup_score_rank_delta") or 0)
        + float(data.get("execution_rank_delta") or 0)
        + float(data.get("regime_transition_rank_delta") or 0)
        + float(data.get("memory_rank_delta") or 0)
        + float(data.get("scalp_adaptive_rank_delta") or 0)
    )
    data["intelligence_rank_delta"] = round(max(-INTELLIGENCE_DELTA_CAP, min(INTELLIGENCE_DELTA_CAP, intel)), 4)
    data["final_scalp_selection_score"] = round(
        max(0.0, min(1.0, base * 0.7 + float(data.get("setup_score") or 0) * 0.2 + float(data.get("scalp_execution_quality_score") or 0) * 0.1 + data["intelligence_rank_delta"])), 4
    )

    try:
        snap_ex = build_scalp_candidate_explanation(data, symbol=symbol)
        data["scalp_candidate_explanation_json"] = explanation_json(data, symbol=symbol)
        data["scalp_candidate_explanation_narrative"] = snap_ex.get("narrative") or ""
    except Exception as exc:
        logger.debug("scalp explanation skipped: %s", exc)

    if redis_client:
        update_scalp_market_memory_on_candidate(redis_client, symbol, data, memory=mem)
    return data


def enrich_scalp_ranked_candidates(
    ranked: list[dict[str, Any]],
    *,
    redis_client: Any = None,
) -> list[dict[str, Any]]:
    enriched: list[dict[str, Any]] = []
    audit_cache: dict[str, dict[str, Any]] = {}
    for row in ranked:
        sym = str(row.get("symbol") or "")
        if sym not in audit_cache:
            audit_cache[sym] = build_symbol_scalp_audit(sym, snap=row.get("snap"), mom_diag=row.get("mom"))
        intel = build_scalp_intelligence(
            symbol=sym,
            snap=row.get("snap"),
            mom=row.get("mom"),
            signal=row.get("signal"),
            micro_regime=row.get("micro_regime", ""),
            redis_client=redis_client,
            audit=audit_cache[sym],
        )
        merged = dict(row)
        merged["intelligence"] = intel
        merged["final_scalp_selection_score"] = intel.get("final_scalp_selection_score")
        # Apply intelligence into the score pick_best_global_candidate actually uses.
        # select_v2 primary is EV_10s (~1e-4). A raw ±0.16 intel add would
        # dominate and re-invert the repaired order — keep intel as tie-break.
        base_rank = float(merged.get("rank_score") or 0.0)
        intel_delta = float(intel.get("intelligence_rank_delta") or 0.0)
        sel_ver = str(merged.get("selection_version") or (merged.get("rank_components") or {}).get("selection_version") or "")
        if sel_ver == "scalp_micro_select_v2" or (merged.get("rank_components") or {}).get("primary") == "EV_10s":
            from backend.services.binance_scalp.scalp_micro_rank import TIEBREAK_SCALE

            intel_delta = intel_delta * TIEBREAK_SCALE
        merged["rank_score_raw"] = base_rank
        merged["rank_score"] = round(base_rank + intel_delta, 8)
        enriched.append(merged)
    enriched.sort(
        key=lambda r: (
            -float(r.get("rank_score") or 0),
            -float((r.get("intelligence") or {}).get("final_scalp_selection_score") or 0),
            float(getattr(r.get("snap"), "spread_pct", 1.0) or 1.0),
        )
    )
    return enriched


__all__ = ["build_scalp_intelligence", "enrich_scalp_ranked_candidates"]
