"""Human-readable DAY candidate explanation snapshots (rank telemetry only)."""

from __future__ import annotations

import json
from typing import Any


def build_candidate_explanation(decision_data: dict[str, Any], *, symbol: str = "") -> dict[str, Any]:
    dd = dict(decision_data or {})
    sym = symbol or str(dd.get("symbol") or "")
    setup = str(dd.get("setup_type") or dd.get("entry_thesis") or "NO_CLEAR_THESIS")
    regime = str(dd.get("day_route_regime") or dd.get("regime") or "unknown")
    blocks = {
        k: dd.get(k)
        for k in (
            "trend_block_score",
            "momentum_block_score",
            "volatility_block_score",
            "volume_block_score",
            "sentiment_block_score",
            "orderbook_block_score",
            "context_block_score",
            "time_block_score",
            "feature_health_score",
        )
        if dd.get(k) is not None
    }
    snap = {
        "symbol": sym,
        "model_probabilities": {
            "prob_buy": dd.get("prob_buy"),
            "prob_hold": dd.get("prob_hold"),
            "prob_sell": dd.get("prob_sell"),
            "confidence": dd.get("winner_probability") or dd.get("confidence"),
            "buy_margin": dd.get("buy_margin"),
        },
        "regime": regime,
        "setup_thesis": setup,
        "setup_score": dd.get("setup_score"),
        "block_scores": blocks,
        "feature_health_score": dd.get("feature_health_score"),
        "feature_health_pass": dd.get("feature_health_pass"),
        "relative_strength_rank": dd.get("relative_strength_rank"),
        "execution_quality_score": dd.get("execution_quality_score"),
        "regime_transition_scores": {
            k: dd.get(k)
            for k in dd
            if k.endswith("_score") and k.startswith(("trend_to_", "range_to_", "bear_", "compression_", "bull_", "panic_", "liquidity_", "regime_transition"))
        },
        "adaptive_learning": {
            "adaptive_score_delta": dd.get("adaptive_score_delta"),
            "adaptive_regime": dd.get("adaptive_regime"),
            "intelligence_rank_delta": dd.get("intelligence_rank_delta"),
        },
        "final_selection_score": dd.get("final_selection_score"),
        "rank_deltas": {
            "block_score_rank_delta": dd.get("block_score_rank_delta"),
            "setup_score_rank_delta": dd.get("setup_score_rank_delta"),
            "execution_rank_delta": dd.get("execution_rank_delta"),
            "regime_transition_rank_delta": dd.get("regime_transition_rank_delta"),
            "memory_rank_delta": dd.get("memory_rank_delta"),
            "basket_rs_rank_delta": dd.get("basket_rs_rank_delta"),
        },
        "why_selected": dd.get("why_selected") or "",
        "selected_over_symbol": dd.get("selected_over_symbol") or "",
    }
    snap["narrative"] = format_explanation_narrative(snap)
    return snap


def format_explanation_narrative(snap: dict[str, Any]) -> str:
    sym = snap.get("symbol") or "?"
    rank = snap.get("relative_strength_rank") or "?"
    fh = snap.get("feature_health_score")
    fh_pct = f"{float(fh)*100:.0f}%" if fh is not None else "n/a"
    setup = snap.get("setup_thesis") or "unknown"
    regime = snap.get("regime") or "unknown"
    exec_q = snap.get("execution_quality_score")
    exec_s = f"{float(exec_q)*100:.0f}%" if exec_q is not None else "n/a"
    final = snap.get("final_selection_score")
    final_s = f"{float(final):.4f}" if final is not None else "n/a"
    over = snap.get("selected_over_symbol") or ""
    why = snap.get("why_selected") or ""
    base = (
        f"{sym} ranked #{rank} with setup {setup} in {regime} regime; "
        f"feature health {fh_pct}, execution {exec_s}, final_selection_score {final_s}."
    )
    if over and why:
        return f"{base} Selected over {over}: {why}."
    return base


def explanation_json(decision_data: dict[str, Any], *, symbol: str = "") -> str:
    return json.dumps(build_candidate_explanation(decision_data, symbol=symbol), separators=(",", ":"))


__all__ = ["build_candidate_explanation", "explanation_json", "format_explanation_narrative"]
