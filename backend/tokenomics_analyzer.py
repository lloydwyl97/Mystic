"""
Tokenomics Analyzer Module

Provides utilities for analyzing token economics and calculating risk scores.
Integrates with the main trading system for token evaluation and risk assessment.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable
from datetime import datetime, timezone
from typing import Any

# Configure logging
logger = logging.getLogger(__name__)

# ----------------------------
# Helpers
# ----------------------------


def _as_float(x: Any, name: str) -> float:
    """Coerce to float with explicit error for better debuggability."""
    try:
        return float(x)
    except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
        msg = f"{name} must be a number; got {x!r}"
        raise ValueError(msg) from e


def _as_float_list(xs: Iterable[Any], name: str) -> list[float]:
    try:
        out = [float(v) for v in xs]
    except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
        msg = f"{name} must be a list of numbers; got {xs!r}"
        raise ValueError(msg) from e
    return out


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


# ----------------------------
# Core scoring
# ----------------------------


def evaluate_tokenomics(
    emission_rate: float,
    unlock_schedule: list[float],
    supply_cap: float,
    current_supply: float,
) -> float:
    """
    Evaluate tokenomics and calculate a risk score.

    Args:
        emission_rate: Annual token emission amount (units of token per year).
        unlock_schedule: List of future unlock amounts (units of token).
        supply_cap: Maximum total supply (units of token).
        current_supply: Current circulating supply (units of token).

    Returns:
        float: Risk score (0-100, higher is better)

    Raises:
        ValueError: If inputs are invalid
    """
    try:
        # Coerce & validate
        emission_rate = _as_float(emission_rate, "emission_rate")
        supply_cap = _as_float(supply_cap, "supply_cap")
        current_supply = _as_float(current_supply, "current_supply")
        unlock_schedule = _as_float_list(unlock_schedule, "unlock_schedule")
    except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
        logger.exception("Error coercing tokenomics parameters: %s", e)
        raise

    if emission_rate < 0:
        msg = "Emission rate must be non-negative"
        raise ValueError(msg)
    if not unlock_schedule or any(x < 0 for x in unlock_schedule):
        msg = "Unlock schedule must be non-empty and contain non-negative values"
        raise ValueError(msg)
    if supply_cap <= 0:
        msg = "Supply cap must be positive"
        raise ValueError(msg)
    if current_supply <= 0 or current_supply > supply_cap:
        msg = "Current supply must be positive and not exceed supply cap"
        raise ValueError(msg)

    try:
        # Original logic (with safe floats)
        inflation = (emission_rate / current_supply) * 100.0
        unlock_risk = sum(unlock_schedule) / supply_cap
        score = 100.0 - (inflation * 2.0 + unlock_risk * 100.0)

        # Clamp to [0, 100]
        final_score = round(max(0.0, min(100.0, score)), 2)

        logger.info(f"[TOKENOMICS] Calculated risk score: {final_score}/100")
        logger.debug(f"[TOKENOMICS] Inflation: {inflation:.4f}%, Unlock risk: {unlock_risk:.6f}")
    except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
        error_msg = f"[TOKENOMICS] Error evaluating tokenomics: {e!s}"
        logger.exception(error_msg)
        raise
    else:
        return final_score


def analyze_token_metrics(
    token_symbol: str,
    emission_rate: float,
    unlock_schedule: list[float],
    supply_cap: float,
    current_supply: float,
    market_cap: Any = None,
) -> dict[str, Any]:
    """
    Comprehensive tokenomics analysis.

    Args:
        token_symbol: Token symbol/name
        emission_rate: Annual token emission amount
        unlock_schedule: List of scheduled token unlocks
        supply_cap: Maximum total supply
        current_supply: Current circulating supply
        market_cap: Current market capitalization (optional)

    Returns:
        dict: Comprehensive tokenomics analysis
    """
    try:
        logger.info(f"[TOKENOMICS] Analyzing tokenomics for {token_symbol}")

        # Calculate risk score (includes validation & coercion)
        risk_score = evaluate_tokenomics(emission_rate, unlock_schedule, supply_cap, current_supply)

        # Recompute numeric metrics (safe casts)
        emission_rate_f = _as_float(emission_rate, "emission_rate")
        supply_cap_f = _as_float(supply_cap, "supply_cap")
        current_supply_f = _as_float(current_supply, "current_supply")
        unlock_schedule_f = _as_float_list(unlock_schedule, "unlock_schedule")
        market_cap_f = None if market_cap is None else _as_float(market_cap, "market_cap")

        inflation_rate = (emission_rate_f / current_supply_f) * 100.0
        unlock_risk = sum(unlock_schedule_f) / supply_cap_f
        circulating_ratio = current_supply_f / supply_cap_f

        # Determine risk level
        if risk_score >= 80:
            risk_level = "LOW"
        elif risk_score >= 60:
            risk_level = "MEDIUM"
        elif risk_score >= 40:
            risk_level = "HIGH"
        else:
            risk_level = "CRITICAL"

        analysis: dict[str, Any] = {
            "token_symbol": token_symbol,
            "risk_score": risk_score,
            "risk_level": risk_level,
            "inflation_rate": round(inflation_rate, 4),
            "unlock_risk": round(unlock_risk, 6),
            "circulating_ratio": round(circulating_ratio, 6),
            "emission_rate": emission_rate_f,
            "supply_cap": supply_cap_f,
            "current_supply": current_supply_f,
            "market_cap": market_cap_f,
            "analysis_timestamp": _now_iso(),
        }

        logger.info(f"[TOKENOMICS] Analysis completed for {token_symbol}: {risk_level} risk")
    except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
        error_msg = f"[TOKENOMICS] Error analyzing {token_symbol}: {e!s}"
        logger.exception(error_msg)
        return {
            "token_symbol": token_symbol,
            "error": str(e),
            "risk_score": 0.0,
            "risk_level": "ERROR",
        }
    else:
        return analysis


def compare_tokenomics(tokens: list[dict[str, Any]]) -> dict[str, Any]:
    """
    Compare tokenomics across multiple tokens.

    Args:
        tokens: List of token analysis dictionaries (from analyze_token_metrics)

    Returns:
        dict: Comparison results
    """
    try:
        logger.info(f"[TOKENOMICS] Comparing {len(tokens)} tokens")

        if not tokens:
            return {"error": "No tokens provided for comparison"}

        # Filter out clearly bad entries
        valid = []
        for t in tokens:
            try:
                rs = _as_float(t.get("risk_score", 0.0), "risk_score")
                inf = _as_float(t.get("inflation_rate", 0.0), "inflation_rate")
                valid.append({**t, "risk_score": rs, "inflation_rate": inf})
            except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError):
                # Skip tokens that don't have numeric fields
                continue

        if not valid:
            return {"error": "No valid token analyses to compare"}

        # Sort by risk score (best first)
        sorted_tokens = sorted(valid, key=lambda x: x.get("risk_score", 0.0), reverse=True)

        # Calculate averages
        avg_risk_score = sum(t.get("risk_score", 0.0) for t in valid) / len(valid)
        avg_inflation = sum(t.get("inflation_rate", 0.0) for t in valid) / len(valid)

        comparison = {
            "total_tokens": len(valid),
            "average_risk_score": round(avg_risk_score, 2),
            "average_inflation_rate": round(avg_inflation, 4),
            "best_token": sorted_tokens[0] if sorted_tokens else None,
            "worst_token": sorted_tokens[-1] if sorted_tokens else None,
            "ranked_tokens": sorted_tokens,
            "comparison_timestamp": _now_iso(),
        }

        best_sym = comparison["best_token"].get("token_symbol") if comparison["best_token"] else "N/A"
        logger.info(f"[TOKENOMICS] Comparison completed. Best: {best_sym}")
    except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
        error_msg = f"[TOKENOMICS] Error comparing tokens: {e!s}"
        logger.exception(error_msg)
        return {"error": str(e)}
    else:
        return comparison


# ----------------------------
# Integration with main trading system
# ----------------------------


def integrate_with_trading_system() -> bool:
    """
    Integrate tokenomics analyzer with the main trading system.

    Returns:
        bool: True if integration successful
    """
    try:
        logger.info("[TOKENOMICS] Integrating with trading system...")

        # Smoke test the analyzer
        test_analysis = analyze_token_metrics(
            "TEST",
            emission_rate=1_000_000,
            unlock_schedule=[1e6, 2e6, 2e6],
            supply_cap=1e9,
            current_supply=5e8,
        )

        logger.info(f"[TOKENOMICS] Integration test completed: {test_analysis.get('risk_score', 0)}/100")
    except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
        logger.exception(f"[TOKENOMICS] Integration failed: {e!s}")
        return False
    else:
        return True


# ----------------------------
# Standalone test harness
# ----------------------------

if __name__ == "__main__":
    # Configure logging for standalone execution
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # Test the tokenomics analyzer
    score = evaluate_tokenomics(
        emission_rate=1_000_000,
        unlock_schedule=[1e6, 2e6, 2e6],
        supply_cap=1e9,
        current_supply=5e8,
    )
    logger.info(f"[TOKENOMICS] Risk Score: {score}/100")

    # Test comprehensive analysis
    analysis = analyze_token_metrics(
        "TEST_TOKEN",
        emission_rate=1_000_000,
        unlock_schedule=[1e6, 2e6, 2e6],
        supply_cap=1e9,
        current_supply=5e8,
    )
    logger.info(f"[TOKENOMICS] Analysis: {analysis}")
