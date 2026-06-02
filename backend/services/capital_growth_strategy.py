"""
Capital Growth Strategy Service
Manages progression from starting capital to $1/minute profit target
Uses stable coin parking and progressive reinvestment strategy
"""

import logging
import os
from datetime import datetime, timezone
from typing import Any

logger = logging.getLogger(__name__)

# Get starting capital from environment (default to 250 if not set)
DEFAULT_STARTING_CAPITAL = float(os.getenv("STARTING_CAPITAL", "250.0"))


class CapitalGrowthStrategy:
    """
    Manages capital growth to $1/minute profit target

    Strategy:
    1. Start with configured capital (from environment)
    2. Keep majority in stable coins (USDT)
    3. Use small percentage for trading
    4. Build profits incrementally
    5. Reinvest profits when threshold reached
    6. Increase trading capital as confidence grows
    7. Always maintain large reserve in stable coins
    """

    def __init__(self, starting_capital: float | None = None) -> None:
        # Starting capital from parameter, env, or default
        self.starting_capital = starting_capital or DEFAULT_STARTING_CAPITAL
        self.target_profit_per_minute = 1.0

        # Current state
        self.total_capital = self.starting_capital
        self.stable_coin_balance = self.starting_capital  # Start 100% in stable coins
        self.trading_balance = 0.0
        self.total_profit = 0.0

        # Risk management parameters
        self.stable_coin_min_percentage = 80.0  # Always keep at least 80% in stable coins
        self.trading_percentage_start = 5.0  # Start with only 5% for trading
        self.trading_percentage_max = 20.0  # Never exceed 20% in active trades
        self.profit_threshold_for_reinvestment = 5.0  # Reinvest when profit reaches $5

        # Progressive growth parameters
        self.confidence_level = 0.0  # 0-1, grows with winning trades
        self.win_streak = 0
        self.total_trades = 0
        self.winning_trades = 0

        # Performance tracking
        self.performance_history: list[dict[str, Any]] = []
        self.last_update = datetime.now(timezone.utc)

        logger.info(f"Capital Growth Strategy initialized: ${self.starting_capital} starting capital")

    def calculate_trading_allocation(self) -> float:
        """
        Calculate how much capital should be allocated to trading
        Starts at 5%, increases with confidence and performance
        """
        # Base allocation
        base_allocation = self.trading_percentage_start

        # Increase allocation based on confidence (up to max)
        confidence_bonus = self.confidence_level * (self.trading_percentage_max - self.trading_percentage_start)
        total_allocation_pct = min(base_allocation + confidence_bonus, self.trading_percentage_max)

        # Calculate dollar amount
        return (self.total_capital * total_allocation_pct) / 100.0

    def update_confidence(self) -> None:
        """Update confidence level based on trading performance"""
        if self.total_trades == 0:
            self.confidence_level = 0.0
            return

        # Win rate component (0-0.5)
        win_rate = self.winning_trades / self.total_trades
        win_rate_score = win_rate * 0.5

        # Win streak component (0-0.3)
        streak_score = min(self.win_streak / 10.0, 0.3)

        # Profit component (0-0.2)
        profit_ratio = self.total_profit / self.starting_capital if self.starting_capital > 0 else 0
        profit_score = min(profit_ratio, 0.2)

        # Total confidence (0-1)
        self.confidence_level = min(win_rate_score + streak_score + profit_score, 1.0)

        logger.info(f"Confidence updated: {self.confidence_level:.2%} (Win rate: {win_rate:.2%}, Streak: {self.win_streak})")

    async def record_trade_result(self, profit: float, was_win: bool) -> dict[str, Any]:
        """
        Record a trade result and update strategy
        """
        self.total_trades += 1

        if was_win:
            self.winning_trades += 1
            self.win_streak += 1
        else:
            self.win_streak = 0

        # Update profit
        self.total_profit += profit
        self.total_capital += profit

        # Update confidence
        self.update_confidence()

        # Check if we should reinvest profits
        if self.total_profit >= self.profit_threshold_for_reinvestment:
            await self._reinvest_profits()

        # Rebalance between stable coins and trading
        await self._rebalance_allocation()

        # Track performance
        performance_record = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "total_capital": self.total_capital,
            "stable_coin_balance": self.stable_coin_balance,
            "trading_balance": self.trading_balance,
            "total_profit": self.total_profit,
            "confidence_level": self.confidence_level,
            "win_rate": self.winning_trades / self.total_trades if self.total_trades > 0 else 0,
            "trades_count": self.total_trades,
        }
        self.performance_history.append(performance_record)

        return performance_record

    async def _reinvest_profits(self) -> None:
        """
        Reinvest accumulated profits strategically
        Keep majority in stable coins, gradually increase trading capital
        """
        logger.info(f"Reinvesting profits: ${self.total_profit:.2f}")

        # Reinvestment strategy:
        # - 90% to stable coins for safety
        # - 10% to increase trading capital
        stable_coin_addition = self.total_profit * 0.9
        trading_capital_addition = self.total_profit * 0.1

        self.stable_coin_balance += stable_coin_addition

        # Reset profit counter (it's now part of capital)
        self.total_profit = 0.0

        logger.info(f"Reinvestment: ${stable_coin_addition:.2f} to stable, ${trading_capital_addition:.2f} available for trading")

    async def _rebalance_allocation(self) -> None:
        """
        Rebalance capital between stable coins and trading
        Always maintain minimum stable coin percentage
        """
        # Calculate target allocation
        trading_allocation = self.calculate_trading_allocation()
        stable_allocation = self.total_capital - trading_allocation

        # Ensure minimum stable coin percentage
        min_stable = (self.total_capital * self.stable_coin_min_percentage) / 100.0
        if stable_allocation < min_stable:
            stable_allocation = min_stable
            trading_allocation = self.total_capital - stable_allocation

        self.stable_coin_balance = stable_allocation
        self.trading_balance = trading_allocation

        logger.debug(
            f"Rebalanced: ${self.stable_coin_balance:.2f} stable "
            f"({self.stable_coin_balance / self.total_capital * 100:.1f}%), "
            f"${self.trading_balance:.2f} trading "
            f"({self.trading_balance / self.total_capital * 100:.1f}%)"
        )

    def get_max_position_size(self) -> float:
        """
        Calculate maximum position size for a single trade
        Conservative: 2-5% of trading balance depending on confidence
        """
        base_position_pct = 2.0  # Start conservative
        max_position_pct = 5.0

        # Scale position size with confidence
        position_pct = base_position_pct + (self.confidence_level * (max_position_pct - base_position_pct))

        return (self.trading_balance * position_pct) / 100.0

    def should_take_trade(self, signal_confidence: float, current_exposure: float) -> tuple[bool, str]:
        """
        Decide if a trade should be taken based on capital growth strategy

        Returns: (should_trade, reason)
        """
        # Check if we have trading capital available
        if self.trading_balance <= 0:
            return False, "No trading capital available"

        # Check signal confidence threshold
        min_confidence = 0.7 - (self.confidence_level * 0.2)  # Lower threshold as AI improves
        if signal_confidence < min_confidence:
            return False, f"Signal confidence too low: {signal_confidence:.2%} < {min_confidence:.2%}"

        # Check current exposure
        max_exposure = self.trading_balance * 0.5  # Never use more than 50% of trading balance at once
        if current_exposure >= max_exposure:
            return False, f"Maximum exposure reached: ${current_exposure:.2f}"

        # Check stable coin minimum is maintained
        stable_pct = (self.stable_coin_balance / self.total_capital) * 100.0
        if stable_pct < self.stable_coin_min_percentage:
            return False, f"Stable coin percentage too low: {stable_pct:.1f}% < {self.stable_coin_min_percentage:.1f}%"

        return True, "Trade approved"

    async def get_status(self) -> dict[str, Any]:
        """Get current capital growth strategy status"""
        win_rate = (self.winning_trades / self.total_trades * 100.0) if self.total_trades > 0 else 0.0

        # Calculate progress to $1/min goal
        # At $1/min = $60/hour = $1,440/day
        # With $500 starting capital, need to generate 240% return to achieve this sustainably
        daily_target = 1440.0
        progress_to_goal = min((self.total_capital / daily_target) * 100.0, 100.0)

        return {
            "starting_capital": self.starting_capital,
            "total_capital": self.total_capital,
            "stable_coin_balance": self.stable_coin_balance,
            "stable_coin_percentage": (self.stable_coin_balance / self.total_capital * 100.0) if self.total_capital > 0 else 0,
            "trading_balance": self.trading_balance,
            "trading_percentage": (self.trading_balance / self.total_capital * 100.0) if self.total_capital > 0 else 0,
            "total_profit": self.total_profit,
            "profit_percentage": ((self.total_capital - self.starting_capital) / self.starting_capital * 100.0) if self.starting_capital > 0 else 0,
            "confidence_level": self.confidence_level,
            "win_rate": win_rate,
            "win_streak": self.win_streak,
            "total_trades": self.total_trades,
            "winning_trades": self.winning_trades,
            "progress_to_1_per_min_goal": progress_to_goal,
            "next_reinvestment_at": self.profit_threshold_for_reinvestment - self.total_profit,
            "max_position_size": self.get_max_position_size(),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }


# Singleton instance
capital_growth_strategy = CapitalGrowthStrategy()
