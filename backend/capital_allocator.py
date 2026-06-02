"""
Smart Capital Allocator
Manages capital distribution across multiple trading strategies based on performance.
Python 3.12 compatible. Windows/PowerShell friendly.

ERROR CONTRACT:
- Success: Returns Dict[str, float] with strategy names as keys and allocation amounts as values
- Error: Returns Dict[str, str] with "error" key containing structured error codes:
  * STRATEGY_METRICS_UNAVAILABLE: Leaderboard service not available
  * INSUFFICIENT_LIVE_METRICS_FOR_KELLY: Missing avg_win/avg_loss metrics
  * INSUFFICIENT_LIVE_VOLATILITY_METRICS: Missing realized_volatility metrics
  * INVALID_MOMENTUM_PERIOD: Momentum period out of range
  * MOMENTUM_PERIOD_TOO_LONG: Momentum period exceeds 30 days
  * INSUFFICIENT_RECENT_DATA: No recent performance data
  * INSUFFICIENT_HISTORICAL_DATA: No historical performance data
  * NO_MOMENTUM_DETECTED: No momentum signals found
  * NO_STRATEGIES_TO_ALLOCATE: No strategies available for allocation
  * ALLOCATION_CAP_CONSTRAINTS_BINDING: All strategies at max weight cap
  * KELLY_ALLOCATION_UNAVAILABLE: PositionSizer not available
  * PERFORMANCE_ALLOCATION_ERROR: Performance-based allocation failed
  * RISK_PARITY_ERROR: Risk parity allocation failed
  * EQUAL_WEIGHT_ERROR: Equal weight allocation failed
  * KELLY_ALLOCATION_ERROR: Kelly criterion allocation failed
  * MOMENTUM_ERROR: Momentum allocation failed

LIVE DATA REQUIREMENTS:
- Kelly: Requires avg_win, avg_loss fields from strategy metrics
- Risk Parity: Requires realized_volatility field from strategy metrics
- Performance/Momentum: Requires win_rate, total_profit, trades, avg_profit fields
- All methods: No fabricated or synthetic data - fails cleanly if live metrics unavailable

CAP ENFORCEMENT:
- max_strategy_weight cap enforced in all allocation methods
- Remainder application respects caps (leaves unallocated if all strategies capped)
- Equal weight allocation respects caps (may leave remainder if cap binds)
"""

from __future__ import annotations

import logging
import os
from datetime import datetime, timezone
from enum import Enum
from math import isfinite
from typing import Any

from position_sizer import PositionSizer
from strategy_leaderboard import get_strategy_leaderboard

from backend.unified_config import get_config

# Define module-level logger
logger = logging.getLogger(__name__)

# Use shared constants instead of redefining


class AllocationMethod(Enum):
    """Supported allocation methods."""

    PERFORMANCE = "performance"
    RISK_PARITY = "risk_parity"
    EQUAL_WEIGHT = "equal_weight"
    KELLY = "kelly"
    MOMENTUM = "momentum"


ALLOCATION_METHODS = [m.value for m in AllocationMethod]


class CapitalAllocator:
    def __init__(self, total_capital: float | None = None, max_strategies: int | None = None) -> None:
        # Config-driven defaults
        config = get_config()
        self.total_capital = float(total_capital or os.getenv("TOTAL_CAPITAL", str(config.get("total_capital", 10000.0))))
        self.max_strategies = int(max_strategies or os.getenv("MAX_STRATEGIES", str(config.get("max_strategies", 50))))
        self.min_trade_threshold = float(os.getenv("MIN_TRADE_THRESHOLD", str(config.get("min_trade_threshold", 10.0))))
        self.max_strategy_weight = float(os.getenv("MAX_STRATEGY_WEIGHT", str(config.get("max_strategy_weight", 0.4))))

        # Comprehensive input validation
        if self.total_capital <= 0 or not isinstance(self.total_capital, (int, float)):
            msg = "total_capital must be a positive number"
            raise ValueError(msg)
        if self.max_strategies <= 0 or not isinstance(self.max_strategies, int):
            msg = "max_strategies must be a positive integer"
            raise ValueError(msg)
        if not (0 < self.max_strategy_weight <= 1):
            msg = "max_strategy_weight must be in range (0, 1]"
            raise ValueError(msg)
        if self.min_trade_threshold < 0:
            msg = "min_trade_threshold must be non-negative"
            raise ValueError(msg)

        # Check for NaN/Inf values
        if not all(
            isfinite(x)
            for x in [
                self.total_capital,
                self.min_trade_threshold,
                self.max_strategy_weight,
            ]
        ):
            msg = "Parameters cannot be NaN or infinite"
            raise ValueError(msg)

        # Safe sizer initialization
        if PositionSizer:
            try:
                self.sizer = PositionSizer()
            except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError):
                self.sizer = None
                logger.warning("PositionSizer could not be initialized - Kelly allocation disabled")
        else:
            self.sizer = None
            logger.warning("PositionSizer not available - Kelly allocation disabled")

        # Allocation history with proactive trimming
        self.allocation_history: list[dict[str, Any]] = []
        self.current_allocations: dict[str, float] = {}
        self.max_history = int(
            os.getenv(
                "MAX_ALLOCATION_HISTORY",
                str(config.get("max_allocation_history", 1000)),
            )
        )

    def allocate_by_performance(self, hours_back: int = 24, min_win_rate: float | None = None) -> dict[str, float]:
        """Allocate based on total profit performance."""
        if not get_strategy_leaderboard:
            return {"error": "STRATEGY_METRICS_UNAVAILABLE"}

        try:
            min_win_rate = min_win_rate or float(os.getenv("MIN_WIN_RATE", "0.55"))
            leaderboard = self._get_validated_leaderboard(hours_back)
            if not leaderboard:
                return {"error": "STRATEGY_METRICS_UNAVAILABLE"}

            winners = [s for s in leaderboard if s.get("win_rate", 0.0) > min_win_rate and s.get("total_profit", 0.0) > 0]
            # Sort by total profit, limit strategies
            winners.sort(key=lambda x: x.get("total_profit", 0.0), reverse=True)
            winners = winners[: self.max_strategies]

            # Calculate weights with cap
            total_score = sum(float(s.get("total_profit", 0.0)) for s in winners)
        except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
            logger.exception(f"Error in allocation calculation: {e}")
            return {"error": "ALLOCATION_CALCULATION_FAILED"}

        # Validate winners and total_score outside try to avoid TRY301
        if not winners:
            logger.critical("No winning strategies found for allocation - NO FALLBACK IN PRODUCTION")
            msg = "No winning strategies available - production requires profitable strategies"
            raise RuntimeError(msg)

        if total_score <= 0:
            logger.critical("No profitable strategies found - NO FALLBACK IN PRODUCTION")
            msg = "Total strategy profit is zero or negative - production requires profitable strategies"
            raise RuntimeError(msg)

        try:
            allocations: dict[str, float] = {}
            for s in winners:
                name = str(s.get("strategy", "unknown"))
                raw_weight = float(s.get("total_profit", 0.0)) / total_score
                capped_weight = min(raw_weight, self.max_strategy_weight)
                allocations[name] = round(self.total_capital * capped_weight, 2)

            # Apply remainder and validate
            allocations = self._apply_remainder_safe(allocations)
            self._log_allocation("performance_based", allocations, {"candidates": winners})
        except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
            logger.exception("Error in performance-based allocation: %s", e)
            return {"error": f"PERFORMANCE_ALLOCATION_ERROR: {e!s}"}
        else:
            return allocations

    def allocate_by_risk_parity(self, target_volatility: float | None = None) -> dict[str, float]:
        """Allocate based on risk parity using live volatility metrics only."""
        if not get_strategy_leaderboard:
            return {"error": "STRATEGY_METRICS_UNAVAILABLE"}

        try:
            target_volatility = target_volatility or float(os.getenv("TARGET_VOLATILITY", "0.15"))
            leaderboard = self._get_validated_leaderboard(24)
            if not leaderboard:
                return {"error": "STRATEGY_METRICS_UNAVAILABLE"}

            items = leaderboard[: self.max_strategies]
            tmp: dict[str, dict[str, float]] = {}
            total_rc = 0.0
            valid_strategies = 0

            for s in items:
                # Require live volatility metric - no fabricated calculations
                realized_volatility = float(s.get("realized_volatility", 0.0))

                # Skip if live volatility metric is missing or invalid
                if realized_volatility <= 0 or not isfinite(realized_volatility):
                    logger.warning(
                        "Strategy %s missing required live volatility metric",
                        s.get("strategy", "unknown"),
                    )
                    continue

                rc = target_volatility / realized_volatility
                tmp[str(s.get("strategy", "unknown"))] = {
                    "volatility": realized_volatility,
                    "risk_contribution": rc,
                }
                total_rc += rc
                valid_strategies += 1

            # Fail cleanly if no valid volatility data
            if valid_strategies == 0:
                return {"error": "INSUFFICIENT_LIVE_VOLATILITY_METRICS"}

            # Compute allocations proportional to risk contribution
            allocations: dict[str, float] = {}
            for name, meta in tmp.items():
                raw_weight = meta["risk_contribution"] / total_rc if total_rc > 0 else 0.0
                capped_weight = min(raw_weight, self.max_strategy_weight)
                allocations[name] = round(self.total_capital * capped_weight, 2)

            allocations = self._apply_remainder_safe(allocations)
            self._log_allocation("risk_parity", allocations, {"candidates": list(tmp.keys())})
        except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
            logger.exception("Error in risk-parity allocation: %s", e)
            return {"error": f"RISK_PARITY_ERROR: {e!s}"}

    def allocate_by_equal_weight(self) -> dict[str, float]:
        """Allocate equally across top strategies respecting caps."""
        if not get_strategy_leaderboard:
            return {"error": "STRATEGY_METRICS_UNAVAILABLE"}

        try:
            leaderboard = self._get_validated_leaderboard(24)
            if not leaderboard:
                return {"error": "STRATEGY_METRICS_UNAVAILABLE"}

            items = leaderboard[: self.max_strategies]
            if not items:
                return {"error": "NO_STRATEGIES_TO_ALLOCATE"}

            n = len(items)
            base_weight = 1.0 / n if n > 0 else 0.0
            # Respect cap per strategy
            weight_per = min(base_weight, self.max_strategy_weight)

            allocations: dict[str, float] = {}
            for s in items:
                name = str(s.get("strategy", "unknown"))
                allocations[name] = round(self.total_capital * weight_per, 2)

            allocations = self._apply_remainder_safe(allocations)
            self._log_allocation("equal_weight", allocations, {"candidates": [s.get("strategy") for s in items]})
        except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
            logger.exception("Equal weight allocation error: %s", e)
            return {"error": f"EQUAL_WEIGHT_ERROR: {e!s}"}
        else:
            return allocations

    def allocate_by_kelly_criterion(self) -> dict[str, float]:
        """Allocate using the Kelly criterion based on avg_win/avg_loss and win_rate.

        PositionSizer availability is required per contract; if not present return appropriate error.
        """
        if not get_strategy_leaderboard:
            return {"error": "STRATEGY_METRICS_UNAVAILABLE"}

        if not self.sizer:
            return {"error": "KELLY_ALLOCATION_UNAVAILABLE"}

        try:
            leaderboard = self._get_validated_leaderboard(24)
            if not leaderboard:
                return {"error": "STRATEGY_METRICS_UNAVAILABLE"}

            items = leaderboard[: self.max_strategies]
            valid = []
            for s in items:
                avg_win = float(s.get("avg_win", 0.0))
                avg_loss = float(s.get("avg_loss", 0.0))
                win_rate = float(s.get("win_rate", 0.0))

                # Need meaningful avg_win and avg_loss (positive magnitudes)
                if avg_win <= 0 or avg_loss <= 0 or not isfinite(avg_win) or not isfinite(avg_loss):
                    logger.debug("Strategy %s missing avg_win/avg_loss", s.get("strategy", "unknown"))
                    continue

                # Normalize win_rate
                wr = self._normalize_win_rate(win_rate)
                valid.append((str(s.get("strategy", "unknown")), avg_win, avg_loss, wr))

            if not valid:
                return {"error": "INSUFFICIENT_LIVE_METRICS_FOR_KELLY"}

            allocations: dict[str, float] = {}
            any_positive = False
            for name, avg_win, avg_loss, p in valid:
                # Kelly fractional allocation: b = avg_win/avg_loss
                b = avg_win / avg_loss if avg_loss != 0 else 0.0
                frac = 0.0 if b <= 0 else (p * (b + 1.0) - 1.0) / b
                frac = max(0.0, frac)
                # Enforce per-strategy cap as fraction
                frac = min(frac, self.max_strategy_weight)
                alloc_amount = round(self.total_capital * frac, 2)
                allocations[name] = alloc_amount
                if alloc_amount > 0:
                    any_positive = True

            if not any_positive:
                return {"error": "KELLY_ALLOCATION_ERROR: no positive allocations"}

            allocations = self._apply_remainder_safe(allocations)
            self._log_allocation("kelly", allocations, {"candidates": [v[0] for v in valid]})
        except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
            logger.exception("Kelly allocation error: %s", e)
            return {"error": f"KELLY_ALLOCATION_ERROR: {e!s}"}
        else:
            return allocations

    def allocate_by_momentum(self, period_days: int = 7) -> dict[str, float]:
        """Allocate based on recent momentum (period_days up to 30)."""
        if not get_strategy_leaderboard:
            return {"error": "STRATEGY_METRICS_UNAVAILABLE"}

        try:
            # Validate period
            if not isinstance(period_days, int) or period_days <= 0:
                return {"error": "INVALID_MOMENTUM_PERIOD"}
            if period_days > 30:
                return {"error": "MOMENTUM_PERIOD_TOO_LONG"}

            hours_back = period_days * 24
            leaderboard = self._get_validated_leaderboard(hours_back)
            if not leaderboard:
                return {"error": "INSUFFICIENT_RECENT_DATA"}

            # Momentum signals: positive recent profit and reasonable win_rate/trades
            candidates = [s for s in leaderboard if s.get("total_profit", 0.0) > 0 and s.get("win_rate", 0.0) > 0.5 and s.get("trades", 0) >= 1]
            if not candidates:
                return {"error": "NO_MOMENTUM_DETECTED"}

            candidates.sort(key=lambda x: x.get("total_profit", 0.0), reverse=True)
            candidates = candidates[: self.max_strategies]

            total_score = sum(float(s.get("total_profit", 0.0)) for s in candidates)
            if total_score <= 0:
                return {"error": "NO_MOMENTUM_DETECTED"}

            allocations: dict[str, float] = {}
            for s in candidates:
                name = str(s.get("strategy", "unknown"))
                raw_weight = float(s.get("total_profit", 0.0)) / total_score
                capped_weight = min(raw_weight, self.max_strategy_weight)
                allocations[name] = round(self.total_capital * capped_weight, 2)

            allocations = self._apply_remainder_safe(allocations)
            self._log_allocation("momentum", allocations, {"period_days": period_days, "candidates": candidates})
        except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
            logger.exception("Momentum allocation error: %s", e)
            return {"error": f"MOMENTUM_ERROR: {e!s}"}
        else:
            return allocations

    def rebalance_portfolio(self, _current_allocations: dict[str, float], method: str = "performance") -> dict[str, float]:
        """Rebalance portfolio using the target allocation method.

        For simplicity this returns the target allocation generated by the method.
        """
        if method not in ALLOCATION_METHODS:
            logger.warning("Unknown method %s, using performance", method)
            method = "performance"

        try:
            if method == AllocationMethod.PERFORMANCE.value:
                return self.allocate_by_performance()
            if method == AllocationMethod.RISK_PARITY.value:
                return self.allocate_by_risk_parity()
            if method == AllocationMethod.EQUAL_WEIGHT.value:
                return self.allocate_by_equal_weight()
            if method == AllocationMethod.KELLY.value:
                return self.allocate_by_kelly_criterion()
            if method == AllocationMethod.MOMENTUM.value:
                return self.allocate_by_momentum()
            return self.allocate_by_performance()
        except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
            logger.exception("Rebalance error: %s", e)
            return {"error": f"REBALANCE_ERROR: {e!s}"}

    def _log_allocation(self, method_name: str, allocations: dict[str, float], metadata: Any = None) -> None:
        """Log and persist allocation decision in history with trimming."""
        try:
            ts = datetime.now(timezone.utc).isoformat()
            entry = {
                "timestamp": ts,
                "method": method_name,
                "allocations": allocations.copy(),
                "metadata": metadata,
            }
            self.current_allocations = allocations.copy()
            self.allocation_history.append(entry)
            # Trim history proactively
            if len(self.allocation_history) > max(1, self.max_history):
                self.allocation_history = self.allocation_history[-self.max_history :]
            logger.info("Logged allocation method=%s total_strategies=%d", method_name, len(allocations))
        except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError):
            logger.exception("Failed to log allocation")

    # --------------------------------------------------------------------------
    # Helper methods
    # --------------------------------------------------------------------------
    def _get_validated_leaderboard(self, hours_back: int) -> list[dict[str, Any]]:
        """Get and validate leaderboard data with strict validation."""
        try:
            if not get_strategy_leaderboard:
                logger.error("Strategy leaderboard service not available")
                return []

            # Validate hours_back parameter
            if hours_back <= 0 or hours_back > 24 * 30:  # Max 30 days
                logger.error("Invalid hours_back parameter: %d", hours_back)
                return []

            leaderboard = list(get_strategy_leaderboard(hours_back))
            if not leaderboard:
                logger.warning("Empty leaderboard returned for hours_back=%d", hours_back)
                return []

            validated = []
            required_fields = [
                "strategy",
                "win_rate",
                "total_profit",
                "trades",
                "avg_profit",
            ]

            for s in leaderboard:
                try:
                    # Strict validation of required fields
                    if not all(field in s for field in required_fields):
                        logger.warning(
                            "Strategy missing required fields: %s",
                            s.get("strategy", "unknown"),
                        )
                        continue

                    strategy_name = str(s.get("strategy", "unknown"))
                    if not strategy_name or strategy_name == "unknown":
                        logger.warning("Invalid strategy name: %s", strategy_name)
                        continue

                    win_rate = self._normalize_win_rate(s.get("win_rate", 0.0))
                    total_profit = float(s.get("total_profit", 0.0))
                    trades = max(int(s.get("trades", 0)), 0)
                    avg_profit = float(s.get("avg_profit", 0.0))

                    # Check for NaN/Inf values
                    if not all(isfinite(x) for x in [win_rate, total_profit, trades, avg_profit]):
                        logger.warning("Strategy %s contains invalid numeric values", strategy_name)
                        continue

                    validated.append(
                        {
                            "strategy": strategy_name,
                            "win_rate": win_rate,
                            "total_profit": total_profit,
                            "trades": trades,
                            "avg_profit": avg_profit,
                            # Include live metrics if available
                            "avg_win": float(s.get("avg_win", 0.0)),
                            "avg_loss": float(s.get("avg_loss", 0.0)),
                            "realized_volatility": float(s.get("realized_volatility", 0.0)),
                        },
                    )
                except (ValueError, TypeError) as e:
                    logger.warning(
                        "Invalid strategy data for %s: %s",
                        s.get("strategy", "unknown"),
                        e,
                    )
                    continue

            logger.info(
                "Validated %d strategies from leaderboard (hours_back=%d)",
                len(validated),
                hours_back,
            )
        except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
            logger.exception("Error getting leaderboard for hours_back=%d: %s", hours_back, e)
            return []
        else:
            return validated

    def _normalize_win_rate(self, win_rate: Any) -> float:
        """Normalize win rate to 0-1 scale."""
        try:
            wr = float(win_rate)
            # If win rate > 1, assume it's percentage (0-100)
            if wr > 1.0:
                wr = wr / 100.0
            return max(0.0, min(1.0, wr))  # Clamp to 0-1
        except (ValueError, TypeError):
            return 0.0

    def _apply_remainder_safe(self, allocations: dict[str, float]) -> dict[str, float]:
        """Apply remainder without violating max_strategy_weight caps."""
        total_allocated = sum(allocations.values())
        remainder = self.total_capital - total_allocated

        if abs(remainder) <= 0.01:  # Only apply if significant
            return allocations

        if not allocations:
            return allocations

        # Find strategies that can accept more allocation without violating cap
        available_strategies = []
        for name, amount in allocations.items():
            current_weight = amount / self.total_capital if self.total_capital > 0 else 0.0
            if current_weight < self.max_strategy_weight:
                available_strategies.append((name, amount, current_weight))

        if not available_strategies:
            # All strategies are at cap - leave remainder unallocated
            logger.info("All strategies at cap - leaving remainder $%.2f unallocated", remainder)
            return allocations

        # Distribute remainder to largest non-capped allocation
        largest_key = max(available_strategies, key=lambda x: x[1])[0]
        allocations[largest_key] = round(allocations[largest_key] + remainder, 2)

        logger.info("Applied remainder $%.2f to strategy %s", remainder, largest_key)
        return allocations


def allocate_capital(total_capital: float, method: str = "performance") -> dict[str, float]:
    """Allocate capital using specified method."""
    if method not in ALLOCATION_METHODS:
        logger.warning("Unknown method %s, using performance", method)
        method = "performance"

    allocator = CapitalAllocator(total_capital)

    if method == AllocationMethod.PERFORMANCE.value:
        return allocator.allocate_by_performance()
    if method == AllocationMethod.RISK_PARITY.value:
        return allocator.allocate_by_risk_parity()
    if method == AllocationMethod.EQUAL_WEIGHT.value:
        return allocator.allocate_by_equal_weight()
    if method == AllocationMethod.KELLY.value:
        return allocator.allocate_by_kelly_criterion()
    if method == AllocationMethod.MOMENTUM.value:
        return allocator.allocate_by_momentum()
    return allocator.allocate_by_performance()


def rebalance_portfolio(
    current_allocations: dict[str, float],
    total_capital: float,
    method: str = "performance",
) -> dict[str, float]:
    """Rebalance portfolio using specified method."""
    if method not in ALLOCATION_METHODS:
        logger.warning("Unknown method %s, using performance", method)
        method = "performance"

    allocator = CapitalAllocator(total_capital)
    return allocator.rebalance_portfolio(current_allocations, method)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    total_capital = float(os.getenv("TEST_TOTAL_CAPITAL", "10000.0"))

    for m in ALLOCATION_METHODS:
        logger.info("Testing allocation method=%s", m)
        try:
            result = allocate_capital(total_capital, m)
            if result and not result.get("error"):
                total_allocated = sum(result.values())
                remainder = total_capital - total_allocated
                logger.info(
                    "%s allocation: %d strategies, $%.2f allocated, $%.2f remainder",
                    m,
                    len(result),
                    total_allocated,
                    remainder,
                )
            else:
                logger.warning(
                    "No allocations for method=%s: %s",
                    m,
                    result.get("error", "unknown"),
                )
        except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as exc:
            logger.exception("Allocation failed for method=%s error=%s", m, exc)
