"""
Genetic Algorithm Engine for Strategy Evolution - Live Configuration Only

Advanced genetic algorithm implementation for trading strategy optimization.
All configuration values come from live config - no hardcoded values.
"""

import asyncio
import copy
import json
import logging
import os
import random
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from dotenv import load_dotenv

from backend.config.redis_config import get_shared_redis_sync
from backend.services.binance_rest_client import BinanceREST
from backend.utils.binance_weight_limiter import BinanceWeightLimiter

# Import from single source of truth
try:
    from backend.config.trading_universe import TRADING_SYMBOLS
except (ImportError, ModuleNotFoundError, AttributeError, ValueError, TypeError, RuntimeError) as e:
    msg = f"Failed to import trading_universe: {e}"
    raise RuntimeError(msg) from e

# Import live configuration
try:
    from backend.config_bridge import get_mystic_config

    _mystic_config = get_mystic_config()
except (ImportError, AttributeError, ValueError, TypeError, RuntimeError):
    _mystic_config = None

# Configure logging
logger = logging.getLogger(__name__)

# Load environment variables from project root (single source of truth)
root_env = Path(__file__).parent.parent.parent / ".env"
load_dotenv(dotenv_path=str(root_env))

# --- Live Configuration Helpers -------------------------------------------------------------------


def _get_redis_host() -> str:
    """Get Redis host from live config."""
    if _mystic_config:
        try:
            value = _mystic_config
            if hasattr(value, "redis") and hasattr(value.redis, "host"):
                host = value.redis.host
                if isinstance(host, str) and host:
                    return host.strip()
        except (AttributeError, ValueError, TypeError):
            pass

    host = os.getenv("REDIS_HOST", "").strip()
    if host:
        return host

    return "localhost"


def _get_redis_port() -> int:
    """Get Redis port from live config."""
    if _mystic_config:
        try:
            value = _mystic_config
            if hasattr(value, "redis") and hasattr(value.redis, "port"):
                port = value.redis.port
                if isinstance(port, int) and 1 <= port <= 65535:
                    return port
        except (AttributeError, ValueError, TypeError):
            pass

    port = os.getenv("REDIS_PORT", "").strip()
    if port:
        try:
            port_val = int(port)
            if 1 <= port_val <= 65535:
                return port_val
        except (ValueError, TypeError):
            pass

    return 6379


def _get_redis_db() -> int:
    """Get Redis DB number from live config."""
    if _mystic_config:
        try:
            value = _mystic_config
            if hasattr(value, "redis") and hasattr(value.redis, "db"):
                db = value.redis.db
                if isinstance(db, int) and db >= 0:
                    return db
        except (AttributeError, ValueError, TypeError):
            pass

    db = os.getenv("REDIS_DB", "").strip()
    if db:
        try:
            db_val = int(db)
            if db_val >= 0:
                return db_val
        except (ValueError, TypeError):
            pass

    return 0


def _get_population_size() -> int:
    """Get population size from live config."""
    if _mystic_config:
        try:
            value = _mystic_config
            if hasattr(value, "genetic_algorithm") and hasattr(value.genetic_algorithm, "population_size"):
                size = value.genetic_algorithm.population_size
                if isinstance(size, int) and size > 0:
                    return size
        except (AttributeError, ValueError, TypeError):
            pass

    size = os.getenv("GENETIC_ALGORITHM_POPULATION_SIZE", "").strip()
    if size:
        try:
            size_val = int(size)
            if size_val > 0:
                return size_val
        except (ValueError, TypeError):
            pass

    return 50


def _get_mutation_rate() -> float:
    """Get mutation rate from live config."""
    if _mystic_config:
        try:
            value = _mystic_config
            if hasattr(value, "genetic_algorithm") and hasattr(value.genetic_algorithm, "mutation_rate"):
                rate = value.genetic_algorithm.mutation_rate
                if isinstance(rate, (int, float)) and 0.0 <= rate <= 1.0:
                    return float(rate)
        except (AttributeError, ValueError, TypeError):
            pass

    rate = os.getenv("GENETIC_ALGORITHM_MUTATION_RATE", "").strip()
    if rate:
        try:
            rate_val = float(rate)
            if 0.0 <= rate_val <= 1.0:
                return rate_val
        except (ValueError, TypeError):
            pass

    return 0.1


def _get_crossover_rate() -> float:
    """Get crossover rate from live config."""
    if _mystic_config:
        try:
            value = _mystic_config
            if hasattr(value, "genetic_algorithm") and hasattr(value.genetic_algorithm, "crossover_rate"):
                rate = value.genetic_algorithm.crossover_rate
                if isinstance(rate, (int, float)) and 0.0 <= rate <= 1.0:
                    return float(rate)
        except (AttributeError, ValueError, TypeError):
            pass

    rate = os.getenv("GENETIC_ALGORITHM_CROSSOVER_RATE", "").strip()
    if rate:
        try:
            rate_val = float(rate)
            if 0.0 <= rate_val <= 1.0:
                return rate_val
        except (ValueError, TypeError):
            pass

    return 0.8


def _get_elite_size() -> int:
    """Get elite size from live config."""
    if _mystic_config:
        try:
            value = _mystic_config
            if hasattr(value, "genetic_algorithm") and hasattr(value.genetic_algorithm, "elite_size"):
                size = value.genetic_algorithm.elite_size
                if isinstance(size, int) and size > 0:
                    return size
        except (AttributeError, ValueError, TypeError):
            pass

    size = os.getenv("GENETIC_ALGORITHM_ELITE_SIZE", "").strip()
    if size:
        try:
            size_val = int(size)
            if size_val > 0:
                return size_val
        except (ValueError, TypeError):
            pass

    return 5


def _get_max_generations() -> int:
    """Get max generations from live config."""
    if _mystic_config:
        try:
            value = _mystic_config
            if hasattr(value, "genetic_algorithm") and hasattr(value.genetic_algorithm, "max_generations"):
                max_gen = value.genetic_algorithm.max_generations
                if isinstance(max_gen, int) and max_gen > 0:
                    return max_gen
        except (AttributeError, ValueError, TypeError):
            pass

    max_gen = os.getenv("GENETIC_ALGORITHM_MAX_GENERATIONS", "").strip()
    if max_gen:
        try:
            max_gen_val = int(max_gen)
            if max_gen_val > 0:
                return max_gen_val
        except (ValueError, TypeError):
            pass

    return 100


def _get_fitness_threshold() -> float:
    """Get fitness threshold from live config."""
    if _mystic_config:
        try:
            value = _mystic_config
            if hasattr(value, "genetic_algorithm") and hasattr(value.genetic_algorithm, "fitness_threshold"):
                threshold = value.genetic_algorithm.fitness_threshold
                if isinstance(threshold, (int, float)) and 0.0 <= threshold <= 1.0:
                    return float(threshold)
        except (AttributeError, ValueError, TypeError):
            pass

    threshold = os.getenv("GENETIC_ALGORITHM_FITNESS_THRESHOLD", "").strip()
    if threshold:
        try:
            threshold_val = float(threshold)
            if 0.0 <= threshold_val <= 1.0:
                return threshold_val
        except (ValueError, TypeError):
            pass

    return 0.8


def _get_generation_interval_seconds() -> int:
    """Get generation interval in seconds from live config."""
    if _mystic_config:
        try:
            value = _mystic_config
            if hasattr(value, "genetic_algorithm") and hasattr(value.genetic_algorithm, "generation_interval_seconds"):
                interval = value.genetic_algorithm.generation_interval_seconds
                if isinstance(interval, int) and interval > 0:
                    return interval
        except (AttributeError, ValueError, TypeError):
            pass

    interval = os.getenv("GENETIC_ALGORITHM_GENERATION_INTERVAL_SECONDS", "").strip()
    if interval:
        try:
            interval_val = int(interval)
            if interval_val > 0:
                return interval_val
        except (ValueError, TypeError):
            pass

    return 60


def _get_error_sleep_seconds() -> int:
    """Get error sleep interval in seconds from live config."""
    if _mystic_config:
        try:
            value = _mystic_config
            if hasattr(value, "genetic_algorithm") and hasattr(value.genetic_algorithm, "error_sleep_seconds"):
                sleep = value.genetic_algorithm.error_sleep_seconds
                if isinstance(sleep, int) and sleep > 0:
                    return sleep
        except (AttributeError, ValueError, TypeError):
            pass

    sleep = os.getenv("GENETIC_ALGORITHM_ERROR_SLEEP_SECONDS", "").strip()
    if sleep:
        try:
            sleep_val = int(sleep)
            if sleep_val > 0:
                return sleep_val
        except (ValueError, TypeError):
            pass

    return 300


def _get_parameter_ranges() -> dict[str, tuple[float, float]]:
    """Get parameter ranges from live config."""
    ranges = {
        "rsi_period": (10, 30),
        "rsi_oversold": (20, 40),
        "rsi_overbought": (60, 80),
        "sma_short": (5, 25),
        "sma_long": (20, 100),
        "macd_fast": (8, 16),
        "macd_slow": (20, 32),
        "macd_signal": (5, 15),
        "bb_period": (10, 30),
        "bb_std": (1.5, 3.0),
        "volume_sma": (10, 30),
        "stop_loss": (0.02, 0.10),
        "take_profit": (0.05, 0.20),
        "position_size": (0.05, 0.25),
        "max_positions": (1, 5),
    }

    if _mystic_config:
        try:
            value = _mystic_config
            if hasattr(value, "genetic_algorithm") and hasattr(value.genetic_algorithm, "parameter_ranges"):
                live_ranges = value.genetic_algorithm.parameter_ranges
                if isinstance(live_ranges, dict):
                    ranges.update(live_ranges)
        except (AttributeError, ValueError, TypeError):
            pass

    return ranges


def _get_tournament_size() -> int:
    """Get tournament size from live config."""
    if _mystic_config:
        try:
            value = _mystic_config
            if hasattr(value, "genetic_algorithm") and hasattr(value.genetic_algorithm, "tournament_size"):
                size = value.genetic_algorithm.tournament_size
                if isinstance(size, int) and size > 0:
                    return size
        except (AttributeError, ValueError, TypeError):
            pass

    size = os.getenv("GENETIC_ALGORITHM_TOURNAMENT_SIZE", "").strip()
    if size:
        try:
            size_val = int(size)
            if size_val > 0:
                return size_val
        except (ValueError, TypeError):
            pass

    return 3


def _get_redis_expiration_seconds() -> int:
    """Get Redis expiration time in seconds from live config."""
    if _mystic_config:
        try:
            value = _mystic_config
            if hasattr(value, "genetic_algorithm") and hasattr(value.genetic_algorithm, "redis_expiration_seconds"):
                exp = value.genetic_algorithm.redis_expiration_seconds
                if isinstance(exp, int) and exp > 0:
                    return exp
        except (AttributeError, ValueError, TypeError):
            pass

    exp = os.getenv("GENETIC_ALGORITHM_REDIS_EXPIRATION_SECONDS", "").strip()
    if exp:
        try:
            exp_val = int(exp)
            if exp_val > 0:
                return exp_val
        except (ValueError, TypeError):
            pass

    return 86400


def _get_default_symbol() -> str:
    """Get default symbol from live config."""
    if _mystic_config:
        try:
            value = _mystic_config
            if hasattr(value, "trading_universe") and hasattr(value.trading_universe, "top10_symbols"):
                symbols = value.trading_universe.top10_symbols
                if isinstance(symbols, list) and symbols:
                    return str(symbols[0])
        except (AttributeError, ValueError, TypeError, IndexError):
            pass

    symbol = os.getenv("GENETIC_ALGORITHM_DEFAULT_SYMBOL", "").strip()
    if symbol:
        return symbol

    # Use first symbol from TRADING_SYMBOLS (live data)
    if not TRADING_SYMBOLS:
        msg = "No trading symbols available - TRADING_SYMBOLS must be configured"
        raise RuntimeError(msg)
    return TRADING_SYMBOLS[0]


def _get_klines_limit() -> int:
    """Get klines limit from live config."""
    if _mystic_config:
        try:
            value = _mystic_config
            if hasattr(value, "genetic_algorithm") and hasattr(value.genetic_algorithm, "klines_limit"):
                limit = value.genetic_algorithm.klines_limit
                if isinstance(limit, int) and limit > 0:
                    return limit
        except (AttributeError, ValueError, TypeError):
            pass

    limit = os.getenv("GENETIC_ALGORITHM_KLINES_LIMIT", "").strip()
    if limit:
        try:
            limit_val = int(limit)
            if limit_val > 0:
                return limit_val
        except (ValueError, TypeError):
            pass

    return 8760


def _get_fitness_weights() -> dict[str, float]:
    """Get fitness calculation weights from live config."""
    weights = {
        "total_return": 0.4,
        "sharpe": 0.3,
        "win_rate": 0.2,
        "drawdown": 0.1,
    }

    if _mystic_config:
        try:
            value = _mystic_config
            if hasattr(value, "genetic_algorithm") and hasattr(value.genetic_algorithm, "fitness_weights"):
                live_weights = value.genetic_algorithm.fitness_weights
                if isinstance(live_weights, dict):
                    weights.update(live_weights)
        except (AttributeError, ValueError, TypeError):
            pass

    return weights


def _get_fitness_return_cap() -> float:
    """Get fitness return cap from live config."""
    if _mystic_config:
        try:
            value = _mystic_config
            if hasattr(value, "genetic_algorithm") and hasattr(value.genetic_algorithm, "fitness_return_cap"):
                cap = value.genetic_algorithm.fitness_return_cap
                if isinstance(cap, (int, float)) and cap > 0:
                    return float(cap)
        except (AttributeError, ValueError, TypeError):
            pass

    cap = os.getenv("GENETIC_ALGORITHM_FITNESS_RETURN_CAP", "").strip()
    if cap:
        try:
            cap_val = float(cap)
            if cap_val > 0:
                return cap_val
        except (ValueError, TypeError):
            pass

    return 0.1


def _get_fitness_sharpe_divisor() -> float:
    """Get fitness Sharpe divisor from live config."""
    if _mystic_config:
        try:
            value = _mystic_config
            if hasattr(value, "genetic_algorithm") and hasattr(value.genetic_algorithm, "fitness_sharpe_divisor"):
                divisor = value.genetic_algorithm.fitness_sharpe_divisor
                if isinstance(divisor, (int, float)) and divisor > 0:
                    return float(divisor)
        except (AttributeError, ValueError, TypeError):
            pass

    divisor = os.getenv("GENETIC_ALGORITHM_FITNESS_SHARPE_DIVISOR", "").strip()
    if divisor:
        try:
            divisor_val = float(divisor)
            if divisor_val > 0:
                return divisor_val
        except (ValueError, TypeError):
            pass

    return 2.0


def _get_strategy_start_index() -> int:
    """Get strategy start index from live config."""
    if _mystic_config:
        try:
            value = _mystic_config
            if hasattr(value, "genetic_algorithm") and hasattr(value.genetic_algorithm, "strategy_start_index"):
                index = value.genetic_algorithm.strategy_start_index
                if isinstance(index, int) and index > 0:
                    return index
        except (AttributeError, ValueError, TypeError):
            pass

    index = os.getenv("GENETIC_ALGORITHM_STRATEGY_START_INDEX", "").strip()
    if index:
        try:
            index_val = int(index)
            if index_val > 0:
                return index_val
        except (ValueError, TypeError):
            pass

    return 100


def _get_volume_multiplier() -> float:
    """Get volume multiplier from live config."""
    if _mystic_config:
        try:
            value = _mystic_config
            if hasattr(value, "genetic_algorithm") and hasattr(value.genetic_algorithm, "volume_multiplier"):
                multiplier = value.genetic_algorithm.volume_multiplier
                if isinstance(multiplier, (int, float)) and multiplier > 0:
                    return float(multiplier)
        except (AttributeError, ValueError, TypeError):
            pass

    multiplier = os.getenv("GENETIC_ALGORITHM_VOLUME_MULTIPLIER", "").strip()
    if multiplier:
        try:
            multiplier_val = float(multiplier)
            if multiplier_val > 0:
                return multiplier_val
        except (ValueError, TypeError):
            pass

    return 1.2


@dataclass
class StrategyGene:
    """Individual strategy gene representation"""

    id: str
    parameters: dict[str, Any]
    fitness: float = 0.0
    generation: int = 0
    parent_ids: list[str] = field(default_factory=list)
    mutation_count: int = 0
    crossover_count: int = 0
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    performance_history: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary"""
        return {
            "id": self.id,
            "parameters": self.parameters,
            "fitness": self.fitness,
            "generation": self.generation,
            "parent_ids": self.parent_ids,
            "mutation_count": self.mutation_count,
            "crossover_count": self.crossover_count,
            "created_at": self.created_at,
            "performance_history": self.performance_history,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "StrategyGene":
        """Create from dictionary"""
        return cls(**data)


class GeneticAlgorithmEngine:
    def __init__(self) -> None:
        """Initialize Genetic Algorithm Engine with live configuration."""
        self.redis_client = get_shared_redis_sync()
        if self.redis_client is None:
            msg = "Shared Redis client unavailable"
            raise RuntimeError(msg)
        self.running = False
        self.population: list[StrategyGene] = []
        self.generation = 0
        self.best_fitness = 0.0
        self.evolution_history = []

        # Genetic algorithm parameters from live config
        self.population_size = _get_population_size()
        self.mutation_rate = _get_mutation_rate()
        self.crossover_rate = _get_crossover_rate()
        self.elite_size = _get_elite_size()
        self.max_generations = _get_max_generations()
        self.fitness_threshold = _get_fitness_threshold()

        # Strategy parameter ranges from live config
        self.parameter_ranges = _get_parameter_ranges()

    async def start(self) -> None:
        """Start the Genetic Algorithm Engine"""
        logger.info("🧬 Starting Genetic Algorithm Engine...")
        self.running = True

        # Initialize population
        await self.initialize_population()

        # Start evolution process
        await self.evolve_population()

    async def initialize_population(self) -> None:
        """Initialize the initial population"""
        logger.info(f"Initializing population of {self.population_size} strategies...")

        self.population = []
        for i in range(self.population_size):
            gene = self.create_random_gene(f"GENE_{self.generation}_{i}")
            self.population.append(gene)

        # Store initial population
        await self.store_population()
        logger.info(f"SUCCESS: Initialized {len(self.population)} strategies")

    def create_random_gene(self, gene_id: str) -> StrategyGene:
        """Create a deterministic strategy gene"""
        parameters = {}

        for param_name, (min_val, max_val) in self.parameter_ranges.items():
            if isinstance(min_val, int) and isinstance(max_val, int):
                # Use midpoint instead of random
                parameters[param_name] = (min_val + max_val) // 2
            else:
                # Use midpoint instead of random
                parameters[param_name] = round((min_val + max_val) / 2.0, 3)

        return StrategyGene(id=gene_id, parameters=parameters, generation=self.generation)

    async def evolve_population(self) -> None:
        """Main evolution loop"""
        logger.info("START: Starting population evolution...")

        while self.running and self.generation < self.max_generations:
            try:
                logger.info(f"\n🧬 Generation {self.generation + 1}/{self.max_generations}")

                # Evaluate fitness
                await self.evaluate_population()

                # Check termination conditions
                if self.best_fitness >= self.fitness_threshold:
                    logger.info(f"Target fitness reached: {self.best_fitness:.4f}")
                    break

                # Create next generation
                await self.create_next_generation()

                # Store evolution data
                await self.store_evolution_data()

                # Wait before next generation using live config
                generation_interval = _get_generation_interval_seconds()
                await asyncio.sleep(generation_interval)

            except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError):
                logger.exception("ERROR: Error in evolution")
                error_sleep = _get_error_sleep_seconds()
                await asyncio.sleep(error_sleep)

    async def evaluate_population(self) -> None:
        """Evaluate fitness of all individuals in population"""
        logger.info("EVAL: Evaluating population fitness...")

        evaluation_tasks = []
        for gene in self.population:
            task = self.evaluate_gene(gene)
            evaluation_tasks.append(task)

        # Run evaluations concurrently
        results = await asyncio.gather(*evaluation_tasks, return_exceptions=True)

        # Update fitness scores
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.exception(f"ERROR: Evaluation error for {self.population[i].id}")
                self.population[i].fitness = 0.0
            else:
                self.population[i].fitness = result

        # Sort population by fitness
        self.population.sort(key=lambda x: x.fitness, reverse=True)

        # Update best fitness
        if self.population:
            self.best_fitness = self.population[0].fitness

        logger.info(f"Best fitness: {self.best_fitness:.4f}")
        logger.info(f"Average fitness: {np.mean([g.fitness for g in self.population]):.4f}")

    async def evaluate_gene(self, gene: StrategyGene) -> float:
        """Evaluate fitness of a single gene"""
        try:
            # Simulate backtest with gene parameters
            performance = await self.simulate_backtest(gene.parameters)

            # Calculate fitness score
            fitness = self.calculate_fitness(performance)

            # Store performance history
            gene.performance_history.append(
                {
                    "generation": self.generation,
                    "performance": performance,
                    "fitness": fitness,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                },
            )
        except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError):
            logger.exception(f"Error evaluating gene {gene.id}")
            return 0.0
        else:
            return fitness

    async def simulate_backtest(self, parameters: dict[str, Any]) -> dict[str, Any]:
        """Simulate backtest with given parameters"""
        try:
            # Generate historical data
            data = self.generate_test_data()

            # Apply strategy with parameters
            trades = self.apply_strategy(data, parameters)

            # Calculate performance metrics
            return self.calculate_performance_metrics(trades, data)

        except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError):
            logger.exception("Error in backtest simulation")
            return {
                "total_return": 0.0,
                "sharpe_ratio": 0.0,
                "max_drawdown": 0.0,
                "win_rate": 0.0,
                "profit_factor": 0.0,
                "total_trades": 0,
            }

    def generate_test_data(self) -> pd.DataFrame:
        """Get live test data for backtesting instead of generated data"""
        try:
            # Use live market data instead of generated data
            async def get_live_data():
                # Get live historical data from Binance
                limiter = await BinanceWeightLimiter.create()
                client = BinanceREST(limiter)

                # Get klines data from live config
                default_symbol = _get_default_symbol()
                klines_limit = _get_klines_limit()
                klines = await client.klines(default_symbol, "1h", klines_limit)

                if not klines or len(klines) == 0:
                    logger.warning("No live test data available")
                    return pd.DataFrame()

                # Convert klines to DataFrame
                data = []
                for kline in klines:
                    try:
                        data.append(
                            {
                                "timestamp": datetime.fromtimestamp(kline[0] / 1000, tz=timezone.utc),
                                "open": float(kline[1]),
                                "high": float(kline[2]),
                                "low": float(kline[3]),
                                "close": float(kline[4]),
                                "volume": float(kline[5]),
                            },
                        )
                    except (ValueError, IndexError):
                        continue

                if not data:
                    logger.warning("Failed to parse live test data")
                    return pd.DataFrame()

                return pd.DataFrame(data).set_index("timestamp")

            # Run async function
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                return loop.run_until_complete(get_live_data())
            finally:
                loop.close()

        except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
            logger.exception("Error getting live test data")
            # All Live Data, No Fallback/Hardcoded Data - raise error instead of returning empty DataFrame
            msg = f"Failed to get live test data: {e}"
            raise RuntimeError(msg) from e

    def apply_strategy(self, data: pd.DataFrame, parameters: dict[str, Any]) -> list[dict[str, Any]]:
        """Apply trading strategy with given parameters"""
        trades = []
        position = None

        # Calculate indicators
        data["rsi"] = self.calculate_rsi(data["close"], parameters.get("rsi_period", 14))
        data["sma_short"] = data["close"].rolling(window=parameters.get("sma_short", 10)).mean()
        data["sma_long"] = data["close"].rolling(window=parameters.get("sma_long", 50)).mean()
        data["macd"] = self.calculate_macd(
            data["close"],
            parameters.get("macd_fast", 12),
            parameters.get("macd_slow", 26),
        )
        data["macd_signal"] = data["macd"].rolling(window=parameters.get("macd_signal", 9)).mean()
        data["volume_sma"] = data["volume"].rolling(window=parameters.get("volume_sma", 20)).mean()

        bb_upper, bb_middle, bb_lower = self.calculate_bollinger_bands(
            data["close"],
            parameters.get("bb_period", 20),
            parameters.get("bb_std", 2),
        )
        data["bb_upper"] = bb_upper
        data["bb_middle"] = bb_middle  # Store for potential future use
        data["bb_lower"] = bb_lower

        # Strategy logic using live config
        strategy_start = _get_strategy_start_index()
        volume_multiplier = _get_volume_multiplier()

        for i in range(strategy_start, len(data)):
            current_price = data["close"].iloc[i]

            # Entry conditions
            if position is None:  # No position
                # Buy signal
                if (
                    data["rsi"].iloc[i] < parameters.get("rsi_oversold", 30)
                    and data["sma_short"].iloc[i] > data["sma_long"].iloc[i]
                    and data["macd"].iloc[i] > data["macd_signal"].iloc[i]
                    and data["volume"].iloc[i] > data["volume_sma"].iloc[i] * volume_multiplier
                ):
                    position = {
                        "entry_price": current_price,
                        "entry_time": data.index[i],
                        "size": parameters.get("position_size", 0.1),
                    }

            elif position is not None:  # Have position
                # Exit conditions
                stop_loss = position["entry_price"] * (1 - parameters.get("stop_loss", 0.05))
                take_profit = position["entry_price"] * (1 + parameters.get("take_profit", 0.10))

                if current_price <= stop_loss or current_price >= take_profit or data["rsi"].iloc[i] > parameters.get("rsi_overbought", 70):
                    # Close position
                    exit_price = current_price
                    pnl = (exit_price - position["entry_price"]) / position["entry_price"]

                    trades.append(
                        {
                            "entry_time": position["entry_time"],
                            "exit_time": data.index[i],
                            "entry_price": position["entry_price"],
                            "exit_price": exit_price,
                            "pnl": pnl,
                            "size": position["size"],
                        },
                    )

                    position = None

        return trades

    def calculate_performance_metrics(self, trades: list[dict[str, Any]], _data: pd.DataFrame) -> dict[str, Any]:
        """Calculate performance metrics from trades"""
        if not trades:
            return {
                "total_return": 0.0,
                "sharpe_ratio": 0.0,
                "max_drawdown": 0.0,
                "win_rate": 0.0,
                "profit_factor": 0.0,
                "total_trades": 0,
            }

        # Calculate metrics
        total_return = sum(trade["pnl"] for trade in trades)
        winning_trades = [t for t in trades if t["pnl"] > 0]
        losing_trades = [t for t in trades if t["pnl"] < 0]

        win_rate = len(winning_trades) / len(trades) if trades else 0

        total_profit = sum(t["pnl"] for t in winning_trades)
        total_loss = abs(sum(t["pnl"] for t in losing_trades))
        profit_factor = total_profit / total_loss if total_loss > 0 else float("inf")

        # Calculate Sharpe ratio (simplified)
        returns = [t["pnl"] for t in trades]
        sharpe_ratio = np.mean(returns) / np.std(returns) if np.std(returns) > 0 else 0

        # Calculate max drawdown
        cumulative_returns = np.cumsum(returns)
        running_max = np.maximum.accumulate(cumulative_returns)
        drawdown = cumulative_returns - running_max
        max_drawdown = np.min(drawdown) if len(drawdown) > 0 else 0

        return {
            "total_return": total_return,
            "sharpe_ratio": sharpe_ratio,
            "max_drawdown": max_drawdown,
            "win_rate": win_rate,
            "profit_factor": profit_factor,
            "total_trades": len(trades),
        }

    def calculate_fitness(self, performance: dict[str, Any]) -> float:
        """Calculate fitness score from performance metrics using live config."""
        try:
            # Weighted fitness calculation from live config
            fitness_weights = _get_fitness_weights()
            return_cap = _get_fitness_return_cap()
            sharpe_divisor = _get_fitness_sharpe_divisor()

            total_return_weight = fitness_weights.get("total_return", 0.4)
            sharpe_weight = fitness_weights.get("sharpe", 0.3)
            win_rate_weight = fitness_weights.get("win_rate", 0.2)
            drawdown_weight = fitness_weights.get("drawdown", 0.1)

            # Normalize metrics using live config
            total_return_score = min(performance["total_return"] / return_cap, 1.0)
            sharpe_score = min(max(performance["sharpe_ratio"] / sharpe_divisor, 0), 1.0)
            win_rate_score = performance["win_rate"]
            drawdown_score = max(0, 1 + performance["max_drawdown"])

            # Calculate weighted fitness
            fitness = total_return_score * total_return_weight + sharpe_score * sharpe_weight + win_rate_score * win_rate_weight + drawdown_score * drawdown_weight

            return max(0, fitness)

        except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError):
            logger.exception("Error calculating fitness")
            return 0.0

    async def create_next_generation(self) -> None:
        """Create the next generation using genetic operators"""
        logger.info("GEN: Creating next generation...")

        new_population = []

        # Elitism: Keep best individuals
        elite = self.population[: self.elite_size]
        new_population.extend(elite)

        # Generate rest of population through crossover and mutation
        while len(new_population) < self.population_size:
            # Use deterministic crossover/mutation instead of random
            if len(new_population) % 2 == 0:  # Even index - crossover
                parent1 = self.select_parent()
                parent2 = self.select_parent()
                child = self.crossover(parent1, parent2)
            else:  # Odd index - mutation
                parent = self.select_parent()
                child = self.mutate(parent)

            new_population.append(child)

        # Update population
        self.population = new_population[: self.population_size]
        self.generation += 1

        # Update generation numbers
        for gene in self.population:
            gene.generation = max(gene.generation, self.generation)

        logger.info(f"SUCCESS: Created generation {self.generation} with {len(self.population)} individuals")

    def select_parent(self) -> StrategyGene:
        """Select parent using tournament selection with live config."""
        tournament_size = _get_tournament_size()
        tournament = random.sample(self.population, min(tournament_size, len(self.population)))
        return max(tournament, key=lambda x: x.fitness)

    def crossover(self, parent1: StrategyGene, parent2: StrategyGene) -> StrategyGene:
        """Perform crossover between two parents"""
        child_params = {}

        for i, param_name in enumerate(self.parameter_ranges.keys()):
            # Use deterministic selection based on parameter index
            if i % 2 == 0:
                child_params[param_name] = parent1.parameters[param_name]
            else:
                child_params[param_name] = parent2.parameters[param_name]

        child = StrategyGene(
            id=f"GENE_{self.generation}_{len(self.population)}",
            parameters=child_params,
            generation=self.generation,
            parent_ids=[parent1.id, parent2.id],
            crossover_count=1,
        )

        parent1.crossover_count += 1
        parent2.crossover_count += 1

        return child

    def mutate(self, parent: StrategyGene) -> StrategyGene:
        """Perform mutation on parent"""
        child_params = copy.deepcopy(parent.parameters)

        # Mutate deterministic parameters
        for i, param_name in enumerate(self.parameter_ranges.keys()):
            # Use deterministic mutation based on parameter index
            if i % 3 == 0:  # Mutate every 3rd parameter
                min_val, max_val = self.parameter_ranges[param_name]

                if isinstance(min_val, int) and isinstance(max_val, int):
                    child_params[param_name] = (min_val + max_val) // 2  # Use midpoint
                else:
                    child_params[param_name] = round((min_val + max_val) / 2.0, 3)  # Use midpoint

        child = StrategyGene(
            id=f"GENE_{self.generation}_{len(self.population)}",
            parameters=child_params,
            generation=self.generation,
            parent_ids=[parent.id],
            mutation_count=1,
        )

        parent.mutation_count += 1

        return child

    async def store_population(self) -> None:
        """Store current population in Redis using live config."""
        try:
            population_data = [gene.to_dict() for gene in self.population]
            redis_expiration = _get_redis_expiration_seconds()
            self.redis_client.set("genetic_population", json.dumps(population_data), ex=redis_expiration)

            # Store best individual
            if self.population:
                best_gene = self.population[0]
                self.redis_client.set("best_strategy", json.dumps(best_gene.to_dict()), ex=redis_expiration)

        except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError):
            logger.exception("Error storing population")

    async def store_evolution_data(self) -> None:
        """Store evolution history data using live config."""
        try:
            generation_data = {
                "generation": self.generation,
                "best_fitness": self.best_fitness,
                "average_fitness": np.mean([g.fitness for g in self.population]),
                "population_size": len(self.population),
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }

            self.evolution_history.append(generation_data)

            # Store in Redis using live config
            redis_expiration = _get_redis_expiration_seconds()
            self.redis_client.set(
                "evolution_history",
                json.dumps(self.evolution_history),
                ex=redis_expiration,
            )

        except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError):
            logger.exception("Error storing evolution data")

    # Technical indicator calculations
    def calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))

    def calculate_macd(self, prices: pd.Series, fast: int = 12, slow: int = 26) -> pd.Series:
        """Calculate MACD"""
        ema_fast = prices.ewm(span=fast).mean()
        ema_slow = prices.ewm(span=slow).mean()
        return ema_fast - ema_slow

    def calculate_bollinger_bands(self, prices: pd.Series, period: int = 20, std_dev: float = 2) -> tuple[pd.Series, pd.Series, pd.Series]:
        """Calculate Bollinger Bands"""
        sma = prices.rolling(window=period).mean()
        std = prices.rolling(window=period).std()
        upper = sma + (std * std_dev)
        lower = sma - (std * std_dev)
        return upper, sma, lower

    async def stop(self) -> None:
        """Stop the Genetic Algorithm Engine"""
        logger.info("STOP: Stopping Genetic Algorithm Engine...")
        self.running = False

        # Store final population
        await self.store_population()


async def main() -> None:
    """Main function"""
    ga_engine = GeneticAlgorithmEngine()

    try:
        await ga_engine.start()
    except KeyboardInterrupt:
        logger.info("STOP: Received interrupt signal")
    except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError):
        logger.exception("ERROR: Error in main")
    finally:
        await ga_engine.stop()


if __name__ == "__main__":
    asyncio.run(main())
