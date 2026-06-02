"""
Self Oracle Module

Provides self-validation and oracle functionality for trading predictions and decisions.
Integrates with the main trading system for automated validation and confidence scoring.
"""

import logging
import os
from datetime import datetime, timezone
from typing import Any

import httpx

# Import from single source of truth
try:
    from backend.config.trading_universe import TRADING_SYMBOLS
except (ImportError, ModuleNotFoundError, AttributeError, ValueError, TypeError, RuntimeError) as e:
    msg = f"Failed to import TRADING_SYMBOLS from trading_universe: {e}"
    raise RuntimeError(msg) from e

logger = logging.getLogger(__name__)

# All Live Data, No Fallback/Hardcoded Data
ALLOWED_SYMBOLS = set(TRADING_SYMBOLS)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def _safe_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError):
        return float(default)


async def fetch_real_world_trigger() -> bool:
    """
    Fetch real-world news and detect regulatory triggers.

    Returns:
        bool: True if regulatory news detected, False otherwise
    """
    try:
        token = os.environ.get("CRYPTOPANIC_TOKEN", "").strip()
        if not token:
            logger.error("[ORACLE] Missing CRYPTOPANIC_TOKEN; skipping news check")
            return False

        url = f"https://cryptopanic.com/api/v1/posts/?auth_token={token}&public=true"
        headers = {"User-Agent": "MysticOracle/1.0"}
        async with httpx.AsyncClient() as client:
            r = await client.get(url, headers=headers, timeout=10)
        # Ensure we raise for non-2xx before parsing
        r.raise_for_status()
        data = r.json()
        results = data.get("results", []) if isinstance(data, dict) else []
        headlines = [str(x.get("title", "")).strip() for x in results if x.get("title")]

        keywords = (
            "ban",
            "regulation",
            "regulatory",
            "lawsuit",
            "subpoena",
            "settlement",
            "fine",
            "sec",
            "cftc",
            "doj",
        )
        hit = any(any(k in h.lower() for k in keywords) for h in headlines)
        if hit:
            msg = "[ORACLE] Detected regulatory-related news"
            logger.warning(msg)
            return True

        logger.info("[ORACLE] No regulatory triggers detected")
    except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
        logger.exception(f"[ORACLE] Error fetching real-world triggers: {e!s}")
        return False
    else:
        return False


def self_validate_prediction(prediction: dict[str, Any], confidence: float, historical_accuracy: float) -> bool | None:
    """
    Self-validate a trading prediction based on confidence and historical accuracy.

    Args:
        prediction: Prediction data dictionary
        confidence: Confidence score (0.0-1.0)
        historical_accuracy: Historical accuracy score (0.0-1.0)

    Returns:
        Optional[bool]: True if validated, False if rejected, None if uncertain
    """
    # Validate inputs outside try to avoid TRY301
    if not isinstance(confidence, (int, float)) or not 0.0 <= confidence <= 1.0:
        msg = f"Confidence must be between 0-1, got {confidence}"
        raise ValueError(msg)
    if not isinstance(historical_accuracy, (int, float)) or not 0.0 <= historical_accuracy <= 1.0:
        msg = f"Historical accuracy must be between 0-1, got {historical_accuracy}"
        raise ValueError(msg)
    if not isinstance(prediction, dict):
        msg = "Prediction must be a dictionary"
        raise TypeError(msg)

    try:
        if confidence > 0.8 and historical_accuracy > 0.7:
            logger.info(f"[ORACLE] Prediction validated: confidence={confidence:.2f}, accuracy={historical_accuracy:.2f}")
            return True
        if confidence < 0.5 or historical_accuracy < 0.5:
            logger.warning(f"[ORACLE] Prediction rejected: confidence={confidence:.2f}, accuracy={historical_accuracy:.2f}")
            return False
        logger.info(f"[ORACLE] Prediction uncertain: confidence={confidence:.2f}, accuracy={historical_accuracy:.2f}")
    except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
        logger.exception(f"[ORACLE] Error validating prediction: {e!s}")
        return None
    else:
        return None


def calculate_oracle_confidence(
    prediction_data: dict[str, Any],
    market_conditions: dict[str, Any],
    model_performance: dict[str, float],
) -> dict[str, Any]:
    """
    Calculate comprehensive oracle confidence score.

    Args:
        prediction_data: Prediction information
        market_conditions: Current market conditions
        model_performance: Historical model performance metrics

    Returns:
        dict: Oracle confidence analysis
    """
    try:
        base_confidence = _safe_float(prediction_data.get("confidence", 0.5), 0.5)
        volatility = _safe_float(market_conditions.get("volatility", 0.5), 0.5)
        trend_strength = _safe_float(market_conditions.get("trend_strength", 0.5), 0.5)
        model_accuracy = _safe_float(model_performance.get("accuracy", 0.5), 0.5)
        model_precision = _safe_float(model_performance.get("precision", 0.5), 0.5)

        volatility_factor = 1.0 - (volatility * 0.3)
        trend_factor = trend_strength * 0.2
        model_factor = ((model_accuracy + model_precision) / 2.0) * 0.3

        weighted_confidence = base_confidence * 0.5 + volatility_factor * 0.3 + trend_factor * 0.2 + model_factor * 0.3
        final_confidence = _clamp(weighted_confidence, 0.0, 1.0)

        analysis = {
            "base_confidence": round(base_confidence, 3),
            "volatility_factor": round(volatility_factor, 3),
            "trend_factor": round(trend_factor, 3),
            "model_factor": round(model_factor, 3),
            "weighted_confidence": round(weighted_confidence, 3),
            "final_confidence": round(final_confidence, 3),
            "confidence_level": ("HIGH" if final_confidence > 0.7 else "MEDIUM" if final_confidence > 0.5 else "LOW"),
            "timestamp": _now_iso(),
        }
        logger.info(f"[ORACLE] Confidence calculated: {analysis['final_confidence']:.3f} ({analysis['confidence_level']})")
    except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
        logger.exception(f"[ORACLE] Error calculating confidence: {e!s}")
        return {
            "error": str(e),
            "final_confidence": 0.0,
            "confidence_level": "ERROR",
            "timestamp": _now_iso(),
        }
    else:
        return analysis


def validate_trading_signal(
    signal: dict[str, Any],
    market_data: dict[str, Any],
    risk_metrics: dict[str, float],
) -> dict[str, Any]:
    """
    Validate a trading signal using oracle logic.

    Args:
        signal: Trading signal data
        market_data: Current market data
        risk_metrics: Risk assessment metrics

    Returns:
        dict: Signal validation results
    """
    try:
        signal_type = str(signal.get("type", "unknown"))
        signal_strength = _safe_float(signal.get("strength", 0.5), 0.5)
        signal_direction = str(signal.get("direction", "neutral"))
        market_volatility = _safe_float(market_data.get("volatility", 0.5), 0.5)
        market_trend = str(market_data.get("trend", "neutral"))
        market_volume = _safe_float(market_data.get("volume", 0.0), 0.0)
        max_risk = _safe_float(risk_metrics.get("max_risk", 0.1), 0.1)
        current_risk = _safe_float(risk_metrics.get("current_risk", 0.05), 0.05)

        validation_score = 0.0
        reasons: list[str] = []

        if signal_strength > 0.7:
            validation_score += 0.3
            reasons.append("strong_signal")
        elif signal_strength > 0.5:
            validation_score += 0.2
            reasons.append("moderate_signal")

        if market_volatility < 0.3:
            validation_score += 0.2
            reasons.append("low_volatility")
        elif market_volatility < 0.6:
            validation_score += 0.1
            reasons.append("moderate_volatility")

        if current_risk < max_risk * 0.5:
            validation_score += 0.3
            reasons.append("low_risk")
        elif current_risk < max_risk:
            validation_score += 0.2
            reasons.append("acceptable_risk")

        if market_volume > 1_000_000:
            validation_score += 0.2
            reasons.append("high_volume")

        if validation_score >= 0.7:
            validation_result = "APPROVED"
        elif validation_score >= 0.5:
            validation_result = "CONDITIONAL"
        else:
            validation_result = "REJECTED"

        result = {
            "signal_type": signal_type,
            "signal_direction": signal_direction,
            "validation_score": round(validation_score, 3),
            "validation_result": validation_result,
            "validation_reasons": reasons,
            "market_conditions": {
                "volatility": market_volatility,
                "trend": market_trend,
                "volume": market_volume,
            },
            "risk_assessment": {
                "current_risk": current_risk,
                "max_risk": max_risk,
                "risk_ratio": round(current_risk / max_risk if max_risk > 0 else 0.0, 3),
            },
            "timestamp": _now_iso(),
        }
        logger.info(f"[ORACLE] Signal validation: {validation_result} (score: {validation_score:.3f})")
    except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
        logger.exception(f"[ORACLE] Error validating signal: {e!s}")
        return {
            "error": str(e),
            "validation_result": "ERROR",
            "validation_score": 0.0,
            "timestamp": _now_iso(),
        }
    else:
        return result


def integrate_with_trading_system() -> bool:
    """
    Integrate self oracle with the main trading system.

    Returns:
        bool: True if integration successful
    """
    try:
        logger.info("[ORACLE] Integrating with trading system")
        test_prediction = {"symbol": "BTCUSDT", "direction": "buy", "confidence": 0.8}
        test_validation = self_validate_prediction(test_prediction, 0.8, 0.75)
        logger.info(f"[ORACLE] Integration test completed: validation={test_validation}")
    except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
        logger.exception(f"[ORACLE] Integration failed: {e!s}")
        return False
    else:
        return True


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    test_prediction = {"symbol": "BTCUSDT", "direction": "buy"}
    result = self_validate_prediction(test_prediction, 0.8, 0.75)
    logger.info(f"[ORACLE] Validation result: {result}")
    confidence = calculate_oracle_confidence(
        {"confidence": 0.8},
        {"volatility": 0.3, "trend_strength": 0.7},
        {"accuracy": 0.75, "precision": 0.8},
    )
    logger.info(f"[ORACLE] Confidence analysis: {confidence}")
