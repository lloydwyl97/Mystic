from __future__ import annotations

import logging
import signal
import sys
import time
import uuid
from datetime import datetime, timezone
from typing import Any

from db_logger import get_session, register_strategy
from reward_engine import get_top_performers

from models import Strategy

# Import from single source of truth
try:
    from backend.config.trading_universe import EXCHANGE_ID
except (ImportError, ModuleNotFoundError, AttributeError, ValueError, TypeError, RuntimeError) as e:
    msg = f"Failed to import EXCHANGE_ID from trading_universe: {e}"
    raise RuntimeError(msg) from e

logger = logging.getLogger(__name__)


def mutate_top_strategies(top_n: int = 3, mutation_count: int = 2) -> list[dict[str, Any]]:
    top_strategies = get_top_performers(top_n=top_n, min_trades=5)
    mutations_created: list[dict[str, Any]] = []

    logger.info("Selected %d top strategies for mutation", len(top_strategies))

    for strat in top_strategies:
        for _ in range(mutation_count):
            mutation_name = f"{strat['name']}_mut_{uuid.uuid4().hex[:4]}"
            mutation_desc = generate_mutation_description(strat["name"], strat)
            strategy_id = register_strategy(mutation_name, mutation_desc)
            if strategy_id:
                mutation_info = {
                    "id": strategy_id,
                    "name": mutation_name,
                    "description": mutation_desc,
                    "parent_strategy": strat["name"],
                    "parent_win_rate": strat.get("win_rate", 0.0),
                    "parent_avg_profit": strat.get("avg_profit", 0.0),
                }
                mutations_created.append(mutation_info)
                logger.info("Created mutation: %s from %s", mutation_name, strat["name"])

    logger.info("Created %d strategy mutations", len(mutations_created))
    return mutations_created


def generate_mutation_description(parent_name: str, parent_stats: dict[str, Any]) -> str:
    mutations = [
        f"Enhanced version of {parent_name} with improved entry timing",
        f"Optimized {parent_name} with reduced risk parameters",
        f"Advanced {parent_name} with volume confirmation",
        f"Refined {parent_name} with momentum filters",
        f"Evolved {parent_name} with trend strength indicators",
        f"Improved {parent_name} with volatility adjustments",
        f"Enhanced {parent_name} with support/resistance levels",
        f"Optimized {parent_name} with RSI divergence",
        f"Advanced {parent_name} with MACD crossovers",
        f"Refined {parent_name} with Bollinger Band signals",
    ]

    base_mutation = mutations[0]  # Use first mutation instead of random choice
    win_rate = float(parent_stats.get("win_rate", 0.0) or 0.0)
    avg_profit = float(parent_stats.get("avg_profit", 0.0) or 0.0)

    if win_rate > 0.6:
        performance_note = f" Based on high-performing parent (Win Rate: {win_rate:.1%}, Avg Profit: {avg_profit:.2f})"
    elif win_rate > 0.4:
        performance_note = f" Based on moderate-performing parent (Win Rate: {win_rate:.1%}, Avg Profit: {avg_profit:.2f})"
    else:
        performance_note = f" Based on parent strategy (Win Rate: {win_rate:.1%}, Avg Profit: {avg_profit:.2f})"

    return base_mutation + performance_note


def crossover_strategies(strategy1_id: int, strategy2_id: int) -> int | None:
    session = get_session()
    try:
        strat1 = session.query(Strategy).filter_by(id=strategy1_id).first()
        strat2 = session.query(Strategy).filter_by(id=strategy2_id).first()

        if not strat1 or not strat2:
            logger.error("One or both strategies not found for crossover")
            return None

        # Safely convert potential None values to floats for formatting
        strat1_win = float(getattr(strat1, "win_rate", 0.0) or 0.0)
        strat1_avg = float(getattr(strat1, "avg_profit", 0.0) or 0.0)
        strat2_win = float(getattr(strat2, "win_rate", 0.0) or 0.0)
        strat2_avg = float(getattr(strat2, "avg_profit", 0.0) or 0.0)

        crossover_name = f"Cross_{strat1.name}_{strat2.name}_{uuid.uuid4().hex[:4]}"
        crossover_desc = (
            f"Hybrid strategy combining {strat1.name} and {strat2.name}. "
            f"Takes best elements from both strategies: "
            f"{strat1.name} (Win Rate: {strat1_win:.1%}, Avg Profit: {strat1_avg:.2f}) and "
            f"{strat2.name} (Win Rate: {strat2_win:.1%}, Avg Profit: {strat2_avg:.2f})"
        )

        strategy_id = register_strategy(crossover_name, crossover_desc)
        if strategy_id:
            logger.info("Created crossover strategy: %s", crossover_name)
    except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
        logger.exception("Failed to create crossover strategy: %s", e)
        return None
    else:
        return strategy_id
    finally:
        session.close()


def create_random_strategy() -> int | None:
    strategy_templates = [
        {
            "name": f"Random_EMA_{uuid.uuid4().hex[:4]}",
            "description": "Random EMA crossover strategy with dynamic timeframes",
        },
        {
            "name": f"Random_RSI_{uuid.uuid4().hex[:4]}",
            "description": "Random RSI strategy with custom overbought/oversold levels",
        },
        {
            "name": f"Random_BB_{uuid.uuid4().hex[:4]}",
            "description": "Random Bollinger Bands strategy with volume confirmation",
        },
        {
            "name": f"Random_MACD_{uuid.uuid4().hex[:4]}",
            "description": "Random MACD strategy with signal line crossovers",
        },
        {
            "name": f"Random_Vol_{uuid.uuid4().hex[:4]}",
            "description": "Random volatility breakout strategy with ATR filters",
        },
    ]

    template = strategy_templates[0]  # Use first template instead of random choice
    strategy_id = register_strategy(template["name"], template["description"])
    if strategy_id:
        logger.info("Created random strategy: %s", template["name"])
    return strategy_id


def evolve_strategy_population(
    mutation_rate: float = 0.3,
    crossover_rate: float = 0.2,
    random_rate: float = 0.1,
) -> dict[str, Any]:
    evolution_results: dict[str, Any] = {
        "mutations_created": 0,
        "crossovers_created": 0,
        "random_strategies_created": 0,
        "total_new_strategies": 0,
        "details": [],
    }

    # Use deterministic evolution instead of random
    if mutation_rate > 0.5:  # Use threshold instead of random
        mutations = mutate_top_strategies(top_n=3, mutation_count=2)
        evolution_results["mutations_created"] = len(mutations)
        evolution_results["details"].extend([{"type": "mutation", "info": m} for m in mutations])

    if crossover_rate > 0.3:  # Use threshold instead of random
        top_strategies = get_top_performers(top_n=5, min_trades=5)
        if len(top_strategies) >= 2:
            for i in range(min(2, len(top_strategies) // 2)):
                strat1 = top_strategies[i]  # Use deterministic selection
                strat2 = top_strategies[(i + 1) % len(top_strategies)]  # Use next strategy
                if strat1["id"] != strat2["id"]:
                    crossover_id = crossover_strategies(strat1["id"], strat2["id"])
                    if crossover_id:
                        evolution_results["crossovers_created"] += 1
                        evolution_results["details"].append(
                            {
                                "type": "crossover",
                                "info": {
                                    "id": crossover_id,
                                    "parent1": strat1["name"],
                                    "parent2": strat2["name"],
                                },
                            },
                        )

    if random_rate > 0.2:  # Use threshold instead of random
        for _ in range(1):  # Use fixed count instead of random
            random_id = create_random_strategy()
            if random_id:
                evolution_results["random_strategies_created"] += 1
                evolution_results["details"].append({"type": "random", "info": {"id": random_id}})

    evolution_results["total_new_strategies"] = evolution_results["mutations_created"] + evolution_results["crossovers_created"] + evolution_results["random_strategies_created"]

    logger.info(
        "Evolution completed: %d new strategies created",
        evolution_results["total_new_strategies"],
    )
    return evolution_results


def cleanup_poor_strategies(max_strategies: int = 50, min_win_rate: float = 0.3) -> list[dict[str, Any]]:
    session = get_session()
    try:
        strategies = session.query(Strategy).filter_by(is_active=True).order_by(Strategy.win_rate.desc(), Strategy.avg_profit.desc()).all()

        deactivated: list[dict[str, Any]] = []

        for i, strat in enumerate(strategies):
            should_deactivate = False
            if i >= max_strategies or (strat.trades_executed >= 10 and float(strat.win_rate or 0.0) < min_win_rate and float(strat.avg_profit or 0.0) < 0):
                should_deactivate = True

            if should_deactivate:
                strat.is_active = False
                strat.updated_at = datetime.now(timezone.utc)
                deactivated.append(
                    {
                        "id": strat.id,
                        "name": strat.name,
                        "win_rate": float(strat.win_rate or 0.0),
                        "avg_profit": float(strat.avg_profit or 0.0),
                        "trades_executed": int(strat.trades_executed or 0),
                        "reason": "population_limit" if i >= max_strategies else "poor_performance",
                    },
                )

        session.commit()
        logger.info("Deactivated %d poor performing strategies", len(deactivated))
    except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
        session.rollback()
        logger.exception("Failed to cleanup poor strategies: %s", e)
        return []
    else:
        return deactivated
    finally:
        session.close()


def run_evolution_cycle() -> dict[str, Any]:
    logger.info("Starting evolution cycle")
    evolution_results = evolve_strategy_population()
    deactivated = cleanup_poor_strategies()
    session = get_session()
    try:
        total_strategies = session.query(Strategy).count()
        active_strategies = session.query(Strategy).filter_by(is_active=True).count()
        evolution_results["population_stats"] = {
            "total_strategies": int(total_strategies),
            "active_strategies": int(active_strategies),
            "deactivated_strategies": len(deactivated),
        }
        evolution_results["deactivated_strategies"] = deactivated
    finally:
        session.close()
    logger.info(
        "Evolution cycle completed. Active strategies: %d",
        evolution_results["population_stats"]["active_strategies"],
    )
    return evolution_results


def _signal_handler(_signum, _frame):
    logger.info("Received shutdown signal, stopping strategy mutator")
    sys.exit(0)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    logger.info("Strategy Mutator Service Starting")

    signal.signal(signal.SIGINT, _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)

    try:
        logger.info("Strategy Mutator Service is running")
        logger.info("Will run evolution cycles every 30 minutes")
        cycle_count = 0
        while True:
            try:
                cycle_count += 1
                logger.info("Starting evolution cycle #%d", cycle_count)
                results = run_evolution_cycle()
                logger.info("Evolution cycle #%d completed", cycle_count)
                logger.info("New strategies created: %d", results["total_new_strategies"])
                logger.info(
                    "Active strategies: %d",
                    results["population_stats"]["active_strategies"],
                )
                logger.info("Strategies deactivated: %d", len(results["deactivated_strategies"]))
                logger.info("Waiting 30 minutes before next evolution cycle")
                time.sleep(1800)
            except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
                logger.exception("Evolution cycle #%d failed: %s", cycle_count, e)
                logger.info("Waiting 5 minutes before retrying")
                time.sleep(300)
    except KeyboardInterrupt:
        logger.info("Strategy Mutator Service stopped by user")
    except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
        logger.exception("Strategy Mutator Service failed: %s", e)
        sys.exit(1)
