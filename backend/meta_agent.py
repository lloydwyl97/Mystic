import logging
import time
from typing import Any

from capital_allocator import allocate_capital
from position_sizer import calculate_position_size
from strategy_leaderboard import get_strategy_leaderboard

logger = logging.getLogger(__name__)


class MetaAgent:
    """Meta-agent that coordinates all trading strategies"""

    def __init__(self, total_capital: float = 10000) -> None:
        self.total_capital: float = total_capital
        self.active_strategies: dict[str, float] = {}
        self.performance_history: list[dict[str, Any]] = []
        logger.info("MetaAgent initialized with total_capital=%s", total_capital)

    def pick_best_strategy(self) -> str | None:
        """Select the best performing strategy"""
        try:
            leaderboard = list(get_strategy_leaderboard() or [])
            if not leaderboard:
                return None
            best = max(
                leaderboard,
                key=lambda s: (
                    float(s.get("total_profit", 0.0)),
                    float(s.get("win_rate", 0.0)),
                ),
            )
            return str(best.get("strategy")) if best.get("strategy") is not None else None
        except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
            logger.exception("pick_best_strategy failed: %s", e)
            return None

    def allocate_capital_to_strategies(self) -> dict[str, float]:
        """Allocate capital across strategies"""
        try:
            allocation = allocate_capital(self.total_capital)
            if isinstance(allocation, dict):
                self.active_strategies = allocation
                result = allocation
            else:
                logger.warning("allocate_capital returned non-dict result; clearing active_strategies")
                self.active_strategies = {}
                result = {}
        except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
            logger.exception("allocate_capital_to_strategies failed: %s", e)
            self.active_strategies = {}
            return {}
        else:
            return result

    def calculate_optimal_position_size(self, strategy_name: str, win_rate: float) -> float:
        """Calculate optimal position size for a strategy"""
        try:
            return float(calculate_position_size(self.total_capital, float(win_rate)))
        except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
            logger.exception("calculate_optimal_position_size failed for %s: %s", strategy_name, e)
            return 0.0

    def monitor_system_health(self) -> dict[str, Any]:
        """Monitor overall system health"""
        try:
            leaderboard = list(get_strategy_leaderboard() or [])
            total_profit = float(sum(float(s.get("total_profit", 0.0)) for s in leaderboard))
            avg_win_rate = float((sum(float(s.get("win_rate", 0.0)) for s in leaderboard) / len(leaderboard)) if leaderboard else 0.0)
            health_score = (total_profit / 1000.0) + (avg_win_rate * 100.0)
            return {
                "total_profit": total_profit,
                "avg_win_rate": avg_win_rate,
                "health_score": float(health_score),
                "active_strategies": len(self.active_strategies),
            }
        except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
            logger.exception("monitor_system_health failed: %s", e)
            return {
                "total_profit": 0.0,
                "avg_win_rate": 0.0,
                "health_score": 0.0,
                "active_strategies": 0,
            }

    def execute_meta_decision(self) -> dict[str, Any]:
        """Execute a meta-level decision"""
        logger.info("[META] Executing meta-level decision")
        health = self.monitor_system_health()
        allocation = self.allocate_capital_to_strategies()
        best_strategy = self.pick_best_strategy()
        action = "continue" if float(health.get("health_score", 0.0)) > 50.0 else "rebalance"
        decision = {
            "timestamp": float(time.time()),
            "health": health,
            "allocation": allocation,
            "best_strategy": best_strategy,
            "action": action,
        }
        self.performance_history.append(decision)
        logger.info(
            "[META] Decision completed: action=%s best_strategy=%s",
            action,
            best_strategy,
        )
        return decision

    def get_system_summary(self) -> dict[str, Any]:
        """Get complete system summary"""
        health = self.monitor_system_health()
        return {
            "meta_agent_status": "active",
            "total_capital": float(self.total_capital),
            "system_health": health,
            "active_strategies": dict(self.active_strategies),
            "performance_history_count": len(self.performance_history),
        }


def pick_best_strategy() -> str | None:
    """Simple function to pick the best strategy"""
    try:
        leaderboard = list(get_strategy_leaderboard() or [])
        if not leaderboard:
            return None
        leaderboard.sort(
            key=lambda x: (
                float(x.get("total_profit", 0.0)),
                float(x.get("win_rate", 0.0)),
            ),
            reverse=True,
        )
        top = leaderboard[0]
        return str(top.get("strategy")) if top.get("strategy") is not None else None
    except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
        logger.exception("pick_best_strategy (module-level) failed: %s", e)
        return None


def run_meta_agent() -> MetaAgent:
    """Run the meta-agent system"""
    agent = MetaAgent(total_capital=10000.0)
    logger.info("[META] Starting meta-agent")
    decision = agent.execute_meta_decision()
    logger.info(
        "[META] action=%s best_strategy=%s health_score=%.2f",
        decision.get("action"),
        decision.get("best_strategy"),
        float(decision.get("health", {}).get("health_score", 0.0)),
    )
    return agent


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    run_meta_agent()
