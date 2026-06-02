import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

load_dotenv(dotenv_path=str(Path(__file__).parent.parent / ".env"))

logger = logging.getLogger(__name__)


class MicroAccountManager:
    """
    Micro Account Manager - manages trading budget with milestone-based scaling.

    CRITICAL FIX: Uses shared Redis connection pool from redis_config to prevent
    "max number of clients reached" errors. Redis client is lazily initialized
    on first use, not at import time.
    """

    def __init__(self, starting_budget: float = 100.0) -> None:
        # All Live Data, No Fallback/Hardcoded Data
        # Redis connection uses shared pool - lazy initialization on first use
        self._redis_client = None  # Lazy initialization - prevents import-time connection
        self.starting_budget = starting_budget
        self._current_budget = None  # Lazy load budget from Redis
        self.min_trade_size = 1.0
        self.max_position_pct = 0.15
        self.risk_per_trade_pct = 0.02
        self.emergency_stop_pct = 0.20
        self.milestones = {
            100: {"strategies": ["basic_rsi", "sma_cross"], "max_positions": 2},
            500: {
                "strategies": ["basic_rsi", "sma_cross", "breakout"],
                "max_positions": 3,
            },
            1000: {
                "strategies": ["basic_rsi", "sma_cross", "breakout", "macd"],
                "max_positions": 4,
            },
            5000: {
                "strategies": ["all_basic", "advanced_patterns"],
                "max_positions": 6,
            },
            10000: {
                "strategies": ["all_strategies", "leverage_1.5x"],
                "max_positions": 8,
            },
        }
        logger.info("MicroAccountManager initialized (Redis lazy-loaded)")

    @property
    def redis_client(self):
        """Lazy-load Redis client from shared connection pool."""
        if self._redis_client is None:
            try:
                from backend.config.redis_config import get_redis_client

                self._redis_client = get_redis_client()
                logger.debug("MicroAccountManager: Redis client obtained from shared pool")
            except (ImportError, RuntimeError) as e:
                logger.exception(f"MicroAccountManager: Failed to get Redis client: {e}")
                raise
        return self._redis_client

    @property
    def current_budget(self) -> float:
        """Lazy-load current budget from Redis."""
        if self._current_budget is None:
            self._current_budget = self.get_current_budget()
        return self._current_budget

    @current_budget.setter
    def current_budget(self, value: float) -> None:
        """Set current budget."""
        self._current_budget = value

    def get_current_budget(self) -> float:
        try:
            budget_str = self.redis_client.get("current_trading_budget")
            if budget_str is not None:
                result = float(budget_str)
            else:
                self.redis_client.set("current_trading_budget", str(self.starting_budget))
                result = self.starting_budget
        except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
            logger.exception(f"Error getting current budget: {e}")
            return self.starting_budget
        else:
            return result

    def update_budget(self, new_budget: float) -> dict[str, Any]:
        if not isinstance(new_budget, (int, float)) or new_budget <= 0:
            msg = "new_budget must be a positive number"
            raise ValueError(msg)
        try:
            old_budget = self.current_budget
            self.current_budget = float(new_budget)
            try:
                self.redis_client.set("current_trading_budget", str(self.current_budget))
            except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
                logger.warning(f"Could not persist budget to Redis: {e}")
            growth_factor = self.current_budget / old_budget if old_budget > 0 else 1.0
            scaled_params = self.get_scaled_parameters()
            try:
                self.redis_client.set("micro_account_params", json.dumps(scaled_params))
            except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
                logger.warning(f"Could not persist scaled parameters to Redis: {e}")
            milestone_info = self.check_milestones()
            logger.info(f"Budget updated from {old_budget} to {self.current_budget} (growth {growth_factor:.2f}x)")
            return {
                "old_budget": old_budget,
                "new_budget": self.current_budget,
                "growth_factor": growth_factor,
                "scaled_parameters": scaled_params,
                "milestone_info": milestone_info,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
        except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
            logger.exception(f"Error updating budget: {e}")
            return {"error": str(e)}

    def get_scaled_parameters(self) -> dict[str, Any]:
        return {
            "max_position_size": self.current_budget * self.max_position_pct,
            "min_trade_size": max(self.min_trade_size, self.current_budget * 0.01),
            "risk_per_trade": self.current_budget * self.risk_per_trade_pct,
            "max_order_size": self.current_budget * 0.50,
            "daily_loss_limit": self.current_budget * 0.05,
            "emergency_stop_loss": self.current_budget * self.emergency_stop_pct,
            "max_concurrent_positions": self.get_max_positions(),
            "available_strategies": self.get_available_strategies(),
            "stop_loss_pct": 0.03,
            "take_profit_pct": 0.06,
            "confidence_threshold": 0.70,
            "current_budget": self.current_budget,
            "budget_tier": self.get_budget_tier(),
            "growth_from_start": (self.current_budget / self.starting_budget) if self.starting_budget > 0 else 0.0,
        }

    def calculate_position_size(self, symbol: str, confidence: float, current_price: float) -> dict[str, Any]:
        if not symbol or not isinstance(symbol, str):
            msg = "symbol must be a non-empty string"
            raise ValueError(msg)
        try:
            from backend.services.confidence_normalizer import ConfidenceNormalizer

            confidence = ConfidenceNormalizer.normalize(float(confidence))
        except Exception:
            confidence = float(confidence)
        if not isinstance(confidence, (int, float)) or confidence < 0 or confidence > 1:
            msg = "confidence must be between 0 and 1"
            raise ValueError(msg)
        if not isinstance(current_price, (int, float)) or current_price <= 0:
            msg = "current_price must be a positive number"
            raise ValueError(msg)
        try:
            base_position_value = self.current_budget * self.max_position_pct
            confidence_multiplier = 0.5 + confidence
            adjusted_position_value = base_position_value * confidence_multiplier
            min_position_value = max(self.min_trade_size, self.current_budget * 0.01)
            final_position_value = max(adjusted_position_value, min_position_value)
            position_quantity = final_position_value / current_price
            risk_amount = final_position_value * 0.03
            return {
                "symbol": symbol,
                "position_value": round(final_position_value, 2),
                "position_quantity": round(position_quantity, 6),
                "risk_amount": round(risk_amount, 2),
                "confidence": float(confidence),
                "confidence_multiplier": round(confidence_multiplier, 2),
                "stop_loss_price": round(current_price * 0.97, 6),
                "take_profit_price": round(current_price * 1.06, 6),
                "is_viable": final_position_value >= min_position_value,
                "budget_percentage": round((final_position_value / self.current_budget) * 100, 2) if self.current_budget > 0 else 0.0,
            }
        except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
            logger.exception(f"Error calculating position size: {e}")
            return {"error": str(e)}

    def check_milestones(self) -> dict[str, Any]:
        current_tier = self.get_budget_tier()
        milestone_info = self.milestones.get(current_tier, {})
        next_milestone = None
        for milestone in sorted(self.milestones.keys()):
            if self.current_budget < milestone:
                next_milestone = milestone
                break
        progress_to_next = 100.0
        if next_milestone and next_milestone > 0:
            progress_to_next = round(min(100.0, (self.current_budget / next_milestone) * 100), 2)
        return {
            "current_tier": current_tier,
            "available_strategies": milestone_info.get("strategies", []),
            "max_positions": milestone_info.get("max_positions", 2),
            "next_milestone": next_milestone,
            "progress_to_next": progress_to_next,
        }

    def get_budget_tier(self) -> int:
        for milestone in sorted(self.milestones.keys(), reverse=True):
            if self.current_budget >= milestone:
                return milestone
        return 100

    def get_max_positions(self) -> int:
        tier = self.get_budget_tier()
        return int(self.milestones.get(tier, {}).get("max_positions", 2))

    def get_available_strategies(self) -> list[str]:
        tier = self.get_budget_tier()
        strategies = self.milestones.get(tier, {}).get("strategies", ["basic_rsi"])
        return list(strategies) if isinstance(strategies, list) else ["basic_rsi"]

    def validate_trade(self, trade_data: dict[str, Any]) -> dict[str, Any]:
        try:
            position_value = float(trade_data.get("position_value", 0) or 0)
            current_positions = self.get_current_position_count()
            validation = {"is_valid": True, "reasons": [], "warnings": []}
            if position_value > self.current_budget * self.max_position_pct:
                validation["is_valid"] = False
                validation["reasons"].append(f"Position too large: {position_value} > {self.current_budget * self.max_position_pct}")
            if position_value < self.min_trade_size:
                validation["is_valid"] = False
                validation["reasons"].append(f"Position too small: {position_value} < {self.min_trade_size}")
            max_positions = self.get_max_positions()
            if current_positions >= max_positions:
                validation["is_valid"] = False
                validation["reasons"].append(f"Too many positions: {current_positions} >= {max_positions}")
            if self.current_budget <= self.starting_budget * (1 - self.emergency_stop_pct):
                validation["is_valid"] = False
                validation["reasons"].append("Emergency stop activated due to drawdown")
            if position_value > self.current_budget * 0.10:
                validation["warnings"].append("Large position relative to account size")
        except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
            logger.exception(f"Error validating trade: {e}")
            return {"is_valid": False, "reasons": [str(e)]}
        else:
            return validation

    def get_current_position_count(self) -> int:
        try:
            positions_str = self.redis_client.get("current_positions")
            if positions_str:
                positions = json.loads(positions_str)
                result = len(positions) if isinstance(positions, list) else 0
            else:
                result = 0
        except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
            logger.warning(f"Could not get current positions: {e}")
            return 0
        else:
            return result

    def get_account_status(self) -> dict[str, Any]:
        try:
            scaled_params = self.get_scaled_parameters()
            milestone_info = self.check_milestones()
            total_growth = ((self.current_budget - self.starting_budget) / self.starting_budget) * 100 if self.starting_budget > 0 else 0.0
            return {
                "account_info": {
                    "current_budget": self.current_budget,
                    "starting_budget": self.starting_budget,
                    "total_growth_pct": round(total_growth, 2),
                    "budget_tier": self.get_budget_tier(),
                },
                "trading_limits": {
                    "max_position_size": scaled_params["max_position_size"],
                    "min_trade_size": scaled_params["min_trade_size"],
                    "max_concurrent_positions": scaled_params["max_concurrent_positions"],
                    "daily_loss_limit": scaled_params["daily_loss_limit"],
                },
                "milestone_info": milestone_info,
                "risk_management": {
                    "risk_per_trade": scaled_params["risk_per_trade"],
                    "stop_loss_pct": scaled_params["stop_loss_pct"],
                    "take_profit_pct": scaled_params["take_profit_pct"],
                    "emergency_stop_threshold": self.starting_budget * (1 - self.emergency_stop_pct),
                },
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
        except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
            logger.exception(f"Error getting account status: {e}")
            return {"error": str(e)}


# CRITICAL FIX: Removed module-level instantiation that caused Redis pool exhaustion
# Previous: micro_account_manager = MicroAccountManager() - created connection at import time
# Now using lazy initialization pattern via get_micro_account_manager()

# Module-level state for lazy singleton initialization
_micro_account_manager_state: dict[str, MicroAccountManager | None] = {"instance": None}


def get_micro_account_manager() -> MicroAccountManager:
    """
    Get the global MicroAccountManager instance using lazy initialization.

    CRITICAL: This prevents Redis connection pool exhaustion by deferring
    connection creation until first actual use, not at import time.

    Returns:
        MicroAccountManager: Singleton instance with shared Redis pool connection
    """
    if _micro_account_manager_state["instance"] is None:
        _micro_account_manager_state["instance"] = MicroAccountManager()
        logger.info("MicroAccountManager singleton created via lazy initialization")
    return _micro_account_manager_state["instance"]


# For backward compatibility - property that lazily returns the instance
# WARNING: Direct use of this variable will trigger lazy initialization
class _LazyMicroAccountManager:
    """Lazy proxy for backward compatibility with existing imports."""

    def __getattr__(self, name: str):
        return getattr(get_micro_account_manager(), name)

    def __setattr__(self, name: str, value):
        if name.startswith("_"):
            object.__setattr__(self, name, value)
        else:
            setattr(get_micro_account_manager(), name, value)


# Backward-compatible module-level instance (lazy proxy)
micro_account_manager = _LazyMicroAccountManager()

__all__ = ["MicroAccountManager", "get_micro_account_manager", "micro_account_manager"]
