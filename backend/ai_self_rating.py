import logging
from typing import Any

from ai_auto_learner import AIAutoLearner
from simulation_logger import SimulationLogger

# Import from single source of truth
try:
    from backend.config.trading_universe import EXCHANGE_ID
except (ImportError, ModuleNotFoundError, AttributeError, ValueError, TypeError, RuntimeError) as e:
    msg = f"Failed to import EXCHANGE_ID from trading_universe: {e}"
    raise RuntimeError(msg) from e

logger = logging.getLogger(__name__)


def _safe_summary(d: Any) -> dict[str, Any]:
    if not isinstance(d, dict):
        return {"avg_profit": 0.0, "total_trades": 0}
    try:
        avg_profit = float(d.get("avg_profit", 0.0) or 0.0)
    except (TypeError, ValueError):
        avg_profit = 0.0
    try:
        total_trades = int(d.get("total_trades", 0) or 0)
    except (TypeError, ValueError):
        total_trades = 0
    return {"avg_profit": avg_profit, "total_trades": total_trades}


def _safe_state(d: Any) -> dict[str, Any]:
    if not isinstance(d, dict):
        return {"confidence_threshold": 0.0, "adjustment_count": 0}
    try:
        confidence_threshold = float(d.get("confidence_threshold", 0.0) or 0.0)
    except (TypeError, ValueError):
        confidence_threshold = 0.0
    try:
        adjustment_count = int(d.get("adjustment_count", 0) or 0)
    except (TypeError, ValueError):
        adjustment_count = 0
    return {
        "confidence_threshold": confidence_threshold,
        "adjustment_count": adjustment_count,
    }


def _compute_score(avg_profit: float, confidence_threshold: float, adjustment_count: int) -> int:
    base = 50.0
    s = base + (avg_profit * 10.0) + (confidence_threshold * 25.0) - float(adjustment_count)
    s = round(s)
    s = max(s, 0)
    s = min(s, 100)
    return int(s)


def _rank(score: int) -> str:
    if score >= 80:
        return "A+ (Excellent)"
    if score >= 60:
        return "B (Good)"
    if score >= 40:
        return "C (Needs Improvement)"
    return "D (Unstable)"


def get_ai_rating() -> dict[str, Any]:
    try:
        sim_logger = SimulationLogger()
        learner = AIAutoLearner()
        summary = _safe_summary(sim_logger.get_summary())
        state = _safe_state(getattr(learner, "state", {}) or {})
        score = _compute_score(
            float(summary.get("avg_profit", 0.0) or 0.0),
            float(state.get("confidence_threshold", 0.0) or 0.0),
            int(state.get("adjustment_count", 0) or 0),
        )
        return {
            "ai_score": score,
            "rating": _rank(score),
            "adjustments": int(state.get("adjustment_count", 0) or 0),
            "avg_profit": float(summary.get("avg_profit", 0.0) or 0.0),
            "confidence_threshold": float(state.get("confidence_threshold", 0.0) or 0.0),
        }
    except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
        logger.exception(f"get_ai_rating failed: {e}")
        return {
            "ai_score": 0,
            "rating": "D (Unstable)",
            "adjustments": 0,
            "avg_profit": 0.0,
            "confidence_threshold": 0.0,
        }


def get_ai_health_report() -> dict[str, Any]:
    try:
        rating = get_ai_rating()
        sim_logger = SimulationLogger()
        summary = _safe_summary(sim_logger.get_summary())
        health_indicators = {
            "performance": "good" if float(summary.get("avg_profit", 0.0) or 0.0) > 0.0 else "poor",
            "stability": "stable" if int(rating.get("adjustments", 0) or 0) < 5 else "unstable",
            "confidence": "high" if float(rating.get("confidence_threshold", 0.0) or 0.0) > 0.8 else "low",
            "activity": "active" if int(summary.get("total_trades", 0) or 0) > 10 else "inactive",
        }
        return {
            "rating": rating,
            "health_indicators": health_indicators,
            "recommendations": generate_recommendations(rating, summary),
        }
    except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
        logger.exception(f"get_ai_health_report failed: {e}")
        return {
            "rating": {
                "ai_score": 0,
                "rating": "D (Unstable)",
                "adjustments": 0,
                "avg_profit": 0.0,
                "confidence_threshold": 0.0,
            },
            "health_indicators": {
                "performance": "poor",
                "stability": "unstable",
                "confidence": "low",
                "activity": "inactive",
            },
            "recommendations": ["Need more trading data for accurate assessment"],
        }


def generate_recommendations(rating: dict[str, Any], summary: dict[str, Any]) -> list[str]:
    rec: list[str] = []
    try:
        avg_profit = float(rating.get("avg_profit", 0.0) or 0.0)
    except (TypeError, ValueError):
        avg_profit = 0.0
    try:
        adjustments = int(rating.get("adjustments", 0) or 0)
    except (TypeError, ValueError):
        adjustments = 0
    try:
        total_trades = int(summary.get("total_trades", 0) or 0)
    except (TypeError, ValueError):
        total_trades = 0

    if avg_profit < 0.0:
        rec.append("Consider lowering confidence threshold")
    if adjustments > 10:
        rec.append("AI may be over-adjusting - consider reset")
    if total_trades < 5:
        rec.append("Need more trading data for accurate assessment")
    if not rec:
        rec.append("AI performing well - continue current strategy")
    return rec
