"""Read-only AI market understanding diagnostics endpoints."""

from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, HTTPException

from backend.database_schema import DATABASE_PATH
from backend.services.ai_feature_freshness_diagnostics import (
    build_feature_freshness_report,
    build_feature_importance_by_block,
)
from backend.services.ai_model_behavior_diagnostics import build_model_behavior_report
from backend.services.ai_market_diagnostics import (
    build_feature_completeness_report,
    build_feature_importance_report,
    build_full_ai_diagnostics_report,
    build_model_freshness_report,
    build_outcome_quality_audit,
    build_regime_performance_report,
    build_sentiment_slot_status,
)
from backend.services.ai_missed_opportunity_observer import get_missed_opportunity_report
from backend.services.ai_post_trade_feature_review import get_post_trade_feature_review_report

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/ai-diagnostics", tags=["ai-diagnostics"])


@router.get("/feature-completeness")
async def feature_completeness() -> dict[str, Any]:
    try:
        data = await build_feature_completeness_report()
        return {"success": True, "data": data}
    except Exception as exc:
        logger.exception("feature-completeness failed: %s", exc)
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@router.get("/learning-health")
async def learning_health() -> dict[str, Any]:
    """Learning ingestion health: closed outcomes, candidate snapshots,
    heartbeats, forward-label progress, per-symbol promotion readiness."""
    try:
        import asyncio

        from backend.services.ai_learning_ingestion import learning_health_summary

        data = await asyncio.to_thread(learning_health_summary)
        per_sym = data.get("per_symbol") or {}
        for sym, stats in per_sym.items():
            closed = int(stats.get("closed_outcomes") or 0)
            labeled = int(stats.get("labeled_snapshots") or 0)
            stats["promotion_ready"] = closed >= 20
            stats["tiered_fallback_eligible"] = labeled >= 40
        return {"success": True, "data": data}
    except Exception as exc:
        logger.exception("learning-health failed: %s", exc)
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@router.get("/feature-freshness")
async def feature_freshness() -> dict[str, Any]:
    try:
        data = await build_feature_freshness_report()
        return {"success": True, "data": data}
    except Exception as exc:
        logger.exception("feature-freshness failed: %s", exc)
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@router.get("/feature-importance-by-block")
async def feature_importance_by_block() -> dict[str, Any]:
    try:
        return {"success": True, "data": build_feature_importance_by_block()}
    except Exception as exc:
        logger.exception("feature-importance-by-block failed: %s", exc)
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@router.get("/post-trade-reviews")
async def post_trade_reviews(limit: int = 50) -> dict[str, Any]:
    try:
        return {"success": True, "data": get_post_trade_feature_review_report(limit=limit, db_path=DATABASE_PATH)}
    except Exception as exc:
        logger.exception("post-trade-reviews failed: %s", exc)
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@router.get("/feature-importance")
async def feature_importance() -> dict[str, Any]:
    try:
        return {"success": True, "data": build_feature_importance_report()}
    except Exception as exc:
        logger.exception("feature-importance failed: %s", exc)
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@router.get("/model-freshness")
async def model_freshness() -> dict[str, Any]:
    try:
        freshness = build_model_freshness_report(DATABASE_PATH)
        behavior = build_model_behavior_report(DATABASE_PATH)
        return {
            "success": True,
            "data": {
                **freshness,
                "model_behavior": behavior,
            },
        }
    except Exception as exc:
        logger.exception("model-freshness failed: %s", exc)
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@router.get("/model-behavior")
async def model_behavior() -> dict[str, Any]:
    try:
        return {"success": True, "data": build_model_behavior_report(DATABASE_PATH)}
    except Exception as exc:
        logger.exception("model-behavior failed: %s", exc)
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@router.get("/regime-performance")
async def regime_performance() -> dict[str, Any]:
    try:
        return {"success": True, "data": build_regime_performance_report(DATABASE_PATH)}
    except Exception as exc:
        logger.exception("regime-performance failed: %s", exc)
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@router.get("/outcome-quality")
async def outcome_quality(limit: int = 50) -> dict[str, Any]:
    try:
        return {"success": True, "data": build_outcome_quality_audit(DATABASE_PATH, limit=limit)}
    except Exception as exc:
        logger.exception("outcome-quality failed: %s", exc)
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@router.get("/missed-opportunities")
async def missed_opportunities(limit: int = 50) -> dict[str, Any]:
    try:
        return {"success": True, "data": get_missed_opportunity_report(limit=limit, db_path=DATABASE_PATH)}
    except Exception as exc:
        logger.exception("missed-opportunities failed: %s", exc)
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@router.get("/sentiment-status")
async def sentiment_status() -> dict[str, Any]:
    return {"success": True, "data": build_sentiment_slot_status()}


@router.get("/full")
async def full_diagnostics() -> dict[str, Any]:
    try:
        data = await build_full_ai_diagnostics_report(DATABASE_PATH)
        return {"success": True, "data": data}
    except Exception as exc:
        logger.exception("full ai diagnostics failed: %s", exc)
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@router.get("/summary")
async def diagnostics_summary() -> dict[str, Any]:
    """Alias for /full — aggregated AI diagnostics snapshot."""
    return await full_diagnostics()


@router.get("/market-readiness")
async def market_readiness_diagnostics() -> dict[str, Any]:
    """Alias for portfolio market-data readiness probe."""
    try:
        from backend.services.market_data_readiness_probe import probe_market_data_readiness

        data = await probe_market_data_readiness()
        return {"success": True, "data": data}
    except Exception as exc:
        logger.exception("market-readiness diagnostics failed: %s", exc)
        raise HTTPException(status_code=503, detail=str(exc)) from exc


__all__ = ["router"]
