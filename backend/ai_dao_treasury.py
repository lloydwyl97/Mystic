from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

try:
    from capital_allocator import allocate_capital  # type: ignore[import-not-found]
except Exception as e:  # pragma: no cover
    logger.warning("capital_allocator not available: %s", e)
    allocate_capital: Any = None  # type: ignore[assignment]

try:
    from strategy_leaderboard import get_strategy_leaderboard  # type: ignore[import-not-found]
except Exception as e:  # pragma: no cover
    logger.warning("strategy_leaderboard not available: %s", e)
    get_strategy_leaderboard: Any = None  # type: ignore[assignment]


def simulate_treasury_growth(treasury_usdt: float, days: int, growth_rate_per_day: float = 0.015) -> float:
    """Simulate compounding growth of a USDT treasury."""
    try:
        days_int = int(days)
    except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError):
        days_int = 0

    balance = float(treasury_usdt)
    if days_int <= 0:
        return balance
    rate = float(growth_rate_per_day)
    for day in range(days_int):
        balance *= 1.0 + rate
        logger.info("Day %d: $%.2f", day + 1, balance)
    return balance


def allocate_dao_treasury(treasury_usdt: float) -> dict[str, float]:
    """Allocate DAO treasury across strategies based on recent performance."""
    leaderboard = None
    if get_strategy_leaderboard is not None:
        try:
            # Prefer calling with hours_back if supported
            try:
                leaderboard = get_strategy_leaderboard(hours_back=24)  # type: ignore[call-overload]
            except TypeError:
                # Fallback to calling without arguments if signature differs
                leaderboard = get_strategy_leaderboard()  # type: ignore[call-overload]
            if leaderboard is not None:
                try:
                    length = len(leaderboard)  # type: ignore[arg-type]
                except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError):
                    length = 1
                logger.info("Fetched strategy leaderboard for last 24h (%d entries)", length)
        except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
            logger.exception("Failed to fetch strategy leaderboard: %s", e)

    if allocate_capital is None:
        logger.error("Allocation module unavailable; returning empty allocation")
        return {}

    try:
        allocation_result = allocate_capital(float(treasury_usdt))  # type: ignore[call-overload]
        if not isinstance(allocation_result, dict):
            logger.error("allocate_capital did not return a dict; returning empty allocation")
            return {}

        allocation: dict[str, float] = allocation_result
        logger.info("DAO allocation computed for $%.2f", float(treasury_usdt))

        # Optionally log a short preview without printing sensitive details
        preview = dict(list(allocation.items())[:5])
        logger.debug("Allocation preview (first 5): %s", preview)
    except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
        logger.exception("Failed to allocate DAO treasury: %s", e)
        return {}
    else:
        return allocation


def manage_dao_governance() -> dict[str, Any]:
    """Process governance operations (placeholder for on-chain/off-chain logic)."""
    logger.info("Processing governance proposals")
    return {"status": "active", "proposals": []}


def execute_dao_decision(decision_type: str, amount: float) -> dict[str, Any]:
    """Execute a governance decision."""
    logger.info("Executing decision '%s' amount=$%.2f", decision_type, float(amount))
    return {"executed": True, "type": decision_type, "amount": float(amount)}
