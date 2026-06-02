"""
Capital Allocation Engine for $1/Minute Profit Maximization

This engine implements intelligent capital allocation with:
- Starting capital: $250
- Profit parking in stable coins
- 100% profit threshold for reinvestment
- Dynamic position sizing based on confidence and volatility
- Automatic scaling as profits grow
"""

import logging
import os
import time
from dataclasses import dataclass
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)


class AllocationStrategy(Enum):
    CONSERVATIVE = "conservative"  # 5-10% allocation
    MODERATE = "moderate"  # 10-25% allocation
    AGGRESSIVE = "aggressive"  # 25-50% allocation
    ULTRA_AGGRESSIVE = "ultra_aggressive"  # 50-100% allocation (for proven strategies)


@dataclass
class AllocationConfig:
    """Configuration for capital allocation"""

    # Use env for starting capital: STARTING_CAPITAL for live, defaults to 2500 for paper
    starting_capital: float = float(os.getenv("STARTING_CAPITAL", "2500.0"))
    base_allocation_percent: float = 0.10  # 10% base allocation
    max_allocation_percent: float = 0.50  # Max 50% of capital per trade
    profit_reinvestment_threshold: float = 1.0  # 100% profit increase
    profit_parking_threshold: float = 0.7  # Park profits at 70% of target
    min_confidence_threshold: float = float(os.getenv("MIN_CONFIDENCE", "0.50"))  # Align with engine canonical
    max_volatility_multiplier: float = 2.0  # Max volatility adjustment
    stable_coins: list[str] = None

    def __post_init__(self):
        if self.stable_coins is None:
            self.stable_coins = ["USDT", "USDC", "BUSD", "DAI"]


@dataclass
class PositionAllocation:
    """Position allocation details"""

    symbol: str
    position_size: float
    confidence: float
    volatility: float
    allocation_strategy: AllocationStrategy
    profit_multiplier: float
    risk_multiplier: float
    timestamp: float
    reasoning: str


class CapitalAllocationEngine:
    """
    Intelligent capital allocation engine that scales with profits
    Implements the $1/minute profit maximization strategy
    """

    def __init__(self, config: AllocationConfig | None = None):
        self.config = config or AllocationConfig()
        self.logger = logging.getLogger(__name__)

        # Portfolio state
        self.starting_capital = self.config.starting_capital
        self.current_capital = self.config.starting_capital
        self.available_capital = self.config.starting_capital
        self.parked_profits = 0.0

        # Performance tracking
        self.total_trades = 0
        self.winning_trades = 0
        self.total_pnl = 0.0
        self.largest_win = 0.0
        self.largest_loss = 0.0

        # Allocation history
        self.allocation_history: list[PositionAllocation] = []
        self.performance_history: list[dict[str, Any]] = []

        # Strategy progression
        self.current_strategy = AllocationStrategy.CONSERVATIVE
        self.profit_milestones_achieved = 0
        self.last_reinvestment_threshold = self.config.profit_reinvestment_threshold

        # Risk management
        self.daily_loss_limit = self.starting_capital * 0.10  # 10% daily loss limit
        self.daily_pnl = 0.0
        self.daily_reset_time = time.time()

        self.logger.info(f"Capital Allocation Engine initialized with ${self.starting_capital:.2f}")

    async def calculate_position_size(self, symbol: str, signal_confidence: float, market_volatility: float, current_price: float, available_balance: float | None = None) -> PositionAllocation:
        """
        Calculate optimal position size based on multiple factors

        Args:
            symbol: Trading symbol
            signal_confidence: AI signal confidence (0.0-1.0)
            market_volatility: Market volatility measure (0.0-1.0)
            current_price: Current asset price
            available_balance: Available balance override

        Returns:
            PositionAllocation with calculated size and reasoning
        """

        # Check daily loss limit
        if not await self._check_daily_loss_limit():
            return PositionAllocation(
                symbol=symbol,
                position_size=0.0,
                confidence=signal_confidence,
                volatility=market_volatility,
                allocation_strategy=AllocationStrategy.CONSERVATIVE,
                profit_multiplier=1.0,
                risk_multiplier=0.0,
                timestamp=time.time(),
                reasoning="Daily loss limit exceeded - no new positions",
            )

        # Check minimum confidence
        if signal_confidence < self.config.min_confidence_threshold:
            return PositionAllocation(
                symbol=symbol,
                position_size=0.0,
                confidence=signal_confidence,
                volatility=market_volatility,
                allocation_strategy=AllocationStrategy.CONSERVATIVE,
                profit_multiplier=1.0,
                risk_multiplier=0.0,
                timestamp=time.time(),
                reasoning=f"Signal confidence {signal_confidence:.2f} below threshold {self.config.min_confidence_threshold}",
            )

        # Use available balance or current available capital
        capital_for_trading = available_balance or self.available_capital

        # Calculate profit multiplier (scales with accumulated profits)
        profit_multiplier = max(1.0, self.current_capital / self.starting_capital)

        # Calculate confidence multiplier
        confidence_multiplier = signal_confidence**2  # Square for stronger effect

        # Calculate volatility multiplier (reduce size in high volatility)
        volatility_multiplier = 1.0 / (1.0 + market_volatility)
        volatility_multiplier = min(volatility_multiplier, self.config.max_volatility_multiplier)

        # Get base allocation percentage based on strategy
        base_allocation_percent = self._get_base_allocation_percent()

        # Calculate position size
        base_position_value = capital_for_trading * base_allocation_percent
        position_size = base_position_value * profit_multiplier * confidence_multiplier * volatility_multiplier

        # Apply maximum allocation limit
        max_position_value = capital_for_trading * self.config.max_allocation_percent
        position_size = min(position_size, max_position_value)

        # Validate position size
        if current_price <= 0:
            position_size = 0.0

        # Determine allocation strategy
        allocation_strategy = self._determine_strategy(position_size, capital_for_trading)

        # Create reasoning
        reasoning = self._generate_allocation_reasoning(
            symbol, signal_confidence, market_volatility, profit_multiplier, confidence_multiplier, volatility_multiplier, base_allocation_percent, allocation_strategy
        )

        allocation = PositionAllocation(
            symbol=symbol,
            position_size=position_size,
            confidence=signal_confidence,
            volatility=market_volatility,
            allocation_strategy=allocation_strategy,
            profit_multiplier=profit_multiplier,
            risk_multiplier=volatility_multiplier,
            timestamp=time.time(),
            reasoning=reasoning,
        )

        # Store allocation history
        self.allocation_history.append(allocation)

        # Keep only recent history (last 1000 allocations)
        if len(self.allocation_history) > 1000:
            self.allocation_history = self.allocation_history[-1000:]

        return allocation

    def _get_base_allocation_percent(self) -> float:
        """Get base allocation percentage based on current strategy"""
        strategy_multipliers = {
            AllocationStrategy.CONSERVATIVE: 0.05,  # 5%
            AllocationStrategy.MODERATE: 0.10,  # 10%
            AllocationStrategy.AGGRESSIVE: 0.25,  # 25%
            AllocationStrategy.ULTRA_AGGRESSIVE: 0.50,  # 50%
        }
        return strategy_multipliers.get(self.current_strategy, 0.10)

    def _determine_strategy(self, position_size: float, capital_for_trading: float) -> AllocationStrategy:
        """Determine allocation strategy based on position size relative to capital"""
        allocation_ratio = position_size / capital_for_trading

        if allocation_ratio >= 0.40:
            return AllocationStrategy.ULTRA_AGGRESSIVE
        if allocation_ratio >= 0.20:
            return AllocationStrategy.AGGRESSIVE
        if allocation_ratio >= 0.15:
            return AllocationStrategy.MODERATE
        return AllocationStrategy.CONSERVATIVE

    def _generate_allocation_reasoning(
        self, symbol: str, confidence: float, volatility: float, profit_mult: float, conf_mult: float, vol_mult: float, base_alloc: float, strategy: AllocationStrategy
    ) -> str:
        """Generate detailed reasoning for allocation decision"""
        return (
            f"Symbol: {symbol} | Confidence: {confidence:.2f} | Volatility: {volatility:.2f} | "
            f"Profit Mult: {profit_mult:.2f}x | Conf Mult: {conf_mult:.2f}x | Vol Mult: {vol_mult:.2f}x | "
            f"Base Alloc: {base_alloc:.1%} | Strategy: {strategy.value}"
        )

    async def update_portfolio_performance(self, pnl: float, trade_count: int = 1) -> dict[str, Any]:
        """
        Update portfolio performance after trade execution

        Args:
            pnl: Profit/Loss from the trade
            trade_count: Number of trades (for batch updates)

        Returns:
            Updated performance metrics
        """

        # Update basic metrics
        self.total_trades += trade_count
        self.total_pnl += pnl
        self.daily_pnl += pnl
        self.current_capital += pnl

        # Update win/loss tracking
        if pnl > 0:
            self.winning_trades += trade_count
            self.largest_win = max(self.largest_win, pnl)
        else:
            self.largest_loss = min(self.largest_loss, pnl)

        # Check for profit milestones and strategy evolution
        await self._check_profit_milestones()

        # Check daily loss limit
        await self._check_daily_loss_limit()

        # Calculate performance metrics
        performance = self._calculate_performance_metrics()

        # Store performance history
        performance_entry = {
            "timestamp": time.time(),
            "capital": self.current_capital,
            "total_pnl": self.total_pnl,
            "daily_pnl": self.daily_pnl,
            "total_trades": self.total_trades,
            "win_rate": self.winning_trades / max(self.total_trades, 1),
            "dollar_per_minute": performance.get("dollar_per_minute", 0.0),
            "strategy": self.current_strategy.value,
        }
        self.performance_history.append(performance_entry)

        # Keep only recent history (last 1000 entries)
        if len(self.performance_history) > 1000:
            self.performance_history = self.performance_history[-1000:]

        self.logger.info(f"[STATS] Portfolio updated: ${self.current_capital:.2f} | P&L: ${pnl:.2f} | Strategy: {self.current_strategy.value}")

        return performance

    async def record_trade(
        self,
        _symbol: str,
        _entry_price: float,
        _exit_price: float,
        _quantity: float,
        pnl: float,
        _trade_type: str = "paper",
    ) -> dict[str, Any]:
        """
        Record a completed trade for capital allocation tracking

        Args:
            _symbol: Trading symbol
            _entry_price: Entry price
            _exit_price: Exit price
            _quantity: Trade quantity
            pnl: Profit/Loss from the trade
            _trade_type: Type of trade (paper/live)

        Returns:
            Updated performance metrics
        """
        # Delegate to update_portfolio_performance for unified tracking
        return await self.update_portfolio_performance(pnl, trade_count=1)

    async def _check_profit_milestones(self) -> None:
        """Check and handle profit milestones for strategy evolution"""
        profit_ratio = self.current_capital / self.starting_capital

        # Check for reinvestment threshold (100% profit increase)
        if profit_ratio >= self.last_reinvestment_threshold:
            await self._trigger_profit_reinvestment(profit_ratio)

        # Check for parking threshold (70% of target)
        elif profit_ratio >= self.config.profit_parking_threshold * self.last_reinvestment_threshold:
            await self._park_profits_partial(profit_ratio)

    async def _trigger_profit_reinvestment(self, profit_ratio: float) -> None:
        """Handle full profit reinvestment milestone"""
        self.profit_milestones_achieved += 1
        self.last_reinvestment_threshold += 0.5  # Increase threshold for next milestone

        # Evolve strategy based on performance
        await self._evolve_strategy()

        # Increase base allocation
        old_allocation = self.config.base_allocation_percent
        self.config.base_allocation_percent = min(0.25, self.config.base_allocation_percent * 1.5)

        self.logger.info(
            f"PROFIT MILESTONE ACHIEVED! ${self.current_capital:.2f} ({profit_ratio:.1f}x) | "
            f"Increased allocation: {old_allocation:.1%} → {self.config.base_allocation_percent:.1%} | "
            f"New Strategy: {self.current_strategy.value}"
        )

    async def _park_profits_partial(self, _profit_ratio: float) -> None:
        """Park portion of profits in stable coins"""
        # Calculate profits to park (excess over starting capital)
        excess_profits = self.current_capital - self.starting_capital
        parking_amount = excess_profits * 0.3  # Park 30% of excess profits

        if parking_amount > 10.0:  # Minimum parking threshold
            self.parked_profits += parking_amount
            self.available_capital -= parking_amount

            self.logger.info(f" PARKED PROFITS: ${parking_amount:.2f} in stable coins | Available: ${self.available_capital:.2f} | Parked: ${self.parked_profits:.2f}")

    async def _evolve_strategy(self) -> None:
        """Evolve allocation strategy based on performance"""
        win_rate = self.winning_trades / max(self.total_trades, 1)

        # Strategy evolution logic
        if self.profit_milestones_achieved >= 3 and win_rate >= 0.65:
            self.current_strategy = AllocationStrategy.ULTRA_AGGRESSIVE
        elif self.profit_milestones_achieved >= 2 and win_rate >= 0.60:
            self.current_strategy = AllocationStrategy.AGGRESSIVE
        elif self.profit_milestones_achieved >= 1 and win_rate >= 0.55:
            self.current_strategy = AllocationStrategy.MODERATE
        else:
            self.current_strategy = AllocationStrategy.CONSERVATIVE

    async def _check_daily_loss_limit(self) -> bool:
        """Check if daily loss limit has been exceeded"""
        # Reset daily tracking if it's a new day
        current_time = time.time()
        if current_time - self.daily_reset_time >= 86400:  # 24 hours
            self.daily_pnl = 0.0
            self.daily_reset_time = current_time

        # Check if loss limit exceeded
        if self.daily_pnl <= -self.daily_loss_limit:
            self.logger.warning(f"DAILY LOSS LIMIT EXCEEDED: ${self.daily_pnl:.2f} | Limit: ${self.daily_loss_limit:.2f}")
            return False

        return True

    def _calculate_performance_metrics(self) -> dict[str, Any]:
        """Calculate comprehensive performance metrics"""
        profit_ratio = self.current_capital / self.starting_capital
        win_rate = self.winning_trades / max(self.total_trades, 1)

        # Calculate dollar per minute (requires recent performance data)
        dollar_per_minute = 0.0
        if len(self.performance_history) >= 2:
            recent_entries = self.performance_history[-60:]  # Last hour equivalent
            if len(recent_entries) >= 2:
                time_diff = recent_entries[-1]["timestamp"] - recent_entries[0]["timestamp"]
                pnl_diff = recent_entries[-1]["total_pnl"] - recent_entries[0]["total_pnl"]

                if time_diff > 0:
                    dollar_per_minute = (pnl_diff / time_diff) * 60

        return {
            "current_capital": self.current_capital,
            "total_pnl": self.total_pnl,
            "profit_ratio": profit_ratio,
            "win_rate": win_rate,
            "total_trades": self.total_trades,
            "largest_win": self.largest_win,
            "largest_loss": self.largest_loss,
            "dollar_per_minute": dollar_per_minute,
            "dollar_per_hour": dollar_per_minute * 60,
            "dollar_per_day": dollar_per_minute * 1440,
            "dollar_per_year": dollar_per_minute * 525600,
            "current_strategy": self.current_strategy.value,
            "available_capital": self.available_capital,
            "parked_profits": self.parked_profits,
            "profit_milestones": self.profit_milestones_achieved,
            "target_achievement": dollar_per_minute >= 1.0,
        }

    def get_allocation_summary(self) -> dict[str, Any]:
        """Get comprehensive allocation and performance summary"""
        performance = self._calculate_performance_metrics()

        return {
            "allocation_engine": {
                "starting_capital": self.starting_capital,
                "current_capital": self.current_capital,
                "available_capital": self.available_capital,
                "parked_profits": self.parked_profits,
                "current_strategy": self.current_strategy.value,
                "base_allocation_percent": self.config.base_allocation_percent,
                "profit_reinvestment_threshold": self.last_reinvestment_threshold,
            },
            "performance": performance,
            "recent_allocations": [
                {"symbol": alloc.symbol, "position_size": alloc.position_size, "confidence": alloc.confidence, "strategy": alloc.allocation_strategy.value, "timestamp": alloc.timestamp}
                for alloc in self.allocation_history[-10:]  # Last 10 allocations
            ],
            "target_progress": {
                "current_dollar_per_minute": performance["dollar_per_minute"],
                "target_dollar_per_minute": 1.0,
                "progress_percent": min(100.0, (performance["dollar_per_minute"] / 1.0) * 100),
                "estimated_months_to_target": self._estimate_time_to_target(performance),
            },
        }

    def _estimate_time_to_target(self, performance: dict[str, Any]) -> float:
        """Estimate months to reach $1/minute target"""
        current_dpm = performance["dollar_per_minute"]
        target_dpm = 1.0

        if current_dpm <= 0:
            return float("inf")

        # Assume 200% monthly growth (conservative estimate)
        monthly_growth_rate = 2.0
        months_needed = 0

        while current_dpm < target_dpm and months_needed < 24:  # Max 2 years
            current_dpm *= monthly_growth_rate
            months_needed += 1

        return months_needed if months_needed < 24 else float("inf")

    async def reset_daily_limits(self) -> None:
        """Manually reset daily loss limits (for testing/emergency)"""
        self.daily_pnl = 0.0
        self.daily_reset_time = time.time()
        self.logger.info("[RELOAD] Daily limits reset manually")

    async def emergency_stop(self) -> None:
        """Emergency stop - park all capital in stable coins"""
        parking_amount = self.available_capital
        self.parked_profits += parking_amount
        self.available_capital = 0.0

        self.logger.warning(f"EMERGENCY STOP: Parked ${parking_amount:.2f} | Available: $0.00")

    def get_strategy_recommendations(self) -> list[str]:
        """Get AI recommendations for strategy improvement"""
        recommendations = []

        performance = self._calculate_performance_metrics()
        win_rate = performance["win_rate"]
        profit_ratio = performance["profit_ratio"]

        if win_rate < 0.50:
            recommendations.append("[WARN] Win rate below 50% - consider more conservative allocation")
        elif win_rate > 0.70:
            recommendations.append("[OK] Excellent win rate - can increase allocation confidence")

        if profit_ratio < 1.5:
            recommendations.append("[UP] Still in early growth phase - focus on consistent wins")
        elif profit_ratio > 3.0:
            recommendations.append("[START] Strong growth achieved - consider stabilizing profits")

        if performance["dollar_per_minute"] < 0.1:
            recommendations.append("Far from $1/minute target - focus on high-confidence signals")
        elif performance["dollar_per_minute"] > 0.5:
            recommendations.append("Close to target - optimize for consistency over aggression")

        if self.profit_milestones_achieved == 0:
            recommendations.append(" No profit milestones achieved - build consistent track record first")
        elif self.profit_milestones_achieved >= 3:
            recommendations.append("Multiple milestones achieved - ready for ultra-aggressive scaling")

        return recommendations if recommendations else ["[OK] Strategy performing optimally - continue current approach"]


# Global singleton for live operations (used by endpoints and services)
capital_allocation_engine = CapitalAllocationEngine()


def get_capital_allocation_engine() -> CapitalAllocationEngine:
    """Return the global capital allocation engine instance."""
    return capital_allocation_engine
