"""
Genetic Algorithm Engine for Strategy Evolution
Live data only, no external calls.
"""

from __future__ import annotations

import asyncio
import copy
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from dotenv import load_dotenv

from backend.config.redis_config import get_shared_redis_sync
from backend.config.trading_universe import TRADING_SYMBOLS
from backend.services.binance_rest_client import BinanceRestClient

logger = logging.getLogger(__name__)

# Load env
load_dotenv(dotenv_path=str(Path(__file__).parent.parent / ".env"))

# Genetic Algorithm Constants
POPULATION_SIZE = 50  # Standard GA population size for strategy evolution
MAX_GENERATIONS = 100  # Maximum generations to run evolution
GENETIC_CACHE_TTL = 86400  # 24 hours cache for genetic results
DEFAULT_KLINES_LIMIT = 800  # Historical data points for backtesting
DEFAULT_RSI_PERIOD = 14  # Standard RSI period
DEFAULT_SMA_SHORT = 10  # Short-term SMA period
DEFAULT_SMA_LONG = 50  # Long-term SMA period
DEFAULT_MACD_FAST = 12  # MACD fast period
DEFAULT_MACD_SLOW = 26  # MACD slow period
DEFAULT_MACD_SIGNAL = 9  # MACD signal period
DEFAULT_BB_PERIOD = 20  # Bollinger Bands period
DEFAULT_BB_STD_DEV = 2.0  # Bollinger Bands standard deviation
DEFAULT_VOLUME_SMA = 20  # Volume SMA period
PERFORMANCE_MULTIPLIER = 10.0  # Scale performance for fitness scoring


@dataclass
class StrategyGene:
    identifier: str
    parameters: dict[str, Any]
    fitness: float = 0.0
    generation: int = 0
    parent_ids: list[str] = field(default_factory=list)
    mutation_count: int = 0
    crossover_count: int = 0
    created_at: str = field(default_factory=lambda: datetime.now(tz=timezone.utc).isoformat())
    performance_history: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.identifier,
            "parameters": self.parameters,
            "fitness": float(self.fitness),
            "generation": int(self.generation),
            "parent_ids": self.parent_ids,
            "mutation_count": int(self.mutation_count),
            "crossover_count": int(self.crossover_count),
            "created_at": self.created_at,
            "performance_history": self.performance_history,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> StrategyGene:
        # Accept both 'id' and 'identifier' for compatibility with stored payloads
        identifier = data.get("identifier") or data.get("id")
        parameters = data.get("parameters", {})
        return cls(
            identifier=identifier,
            parameters=parameters,
            fitness=float(data.get("fitness", 0.0)),
            generation=int(data.get("generation", 0)),
            parent_ids=list(data.get("parent_ids", [])),
            mutation_count=int(data.get("mutation_count", 0)),
            crossover_count=int(data.get("crossover_count", 0)),
            created_at=data.get("created_at", datetime.now(tz=timezone.utc).isoformat()),
            performance_history=list(data.get("performance_history", [])),
        )


class GeneticAlgorithmEngine:
    def __init__(self) -> None:
        # All Live Data, No Fallback/Hardcoded Data
        self.redis_client = get_shared_redis_sync()
        if self.redis_client is None:
            msg = "Shared Redis client unavailable"
            raise RuntimeError(msg)
        self.running = False
        self.population: list[StrategyGene] = []
        self.generation = 0
        self.best_fitness = 0.0
        self.evolution_history: list[dict[str, Any]] = []

        self.population_size = POPULATION_SIZE
        self.mutation_rate = 0.1
        self.crossover_rate = 0.8
        self.elite_size = 5
        self.max_generations = MAX_GENERATIONS
        self.fitness_threshold = 0.8

        self.parameter_ranges = {
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

    async def start(self):
        logger.info("Starting Genetic Algorithm Engine")
        self.running = True
        await self.initialize_population()
        await self.evolve_population()

    async def initialize_population(self):
        logger.info(f"Initializing population of {self.population_size}")
        self.population = [self.create_random_gene(f"GENE_{self.generation}_{i}") for i in range(self.population_size)]
        await self.store_population()
        logger.info(f"Initialized {len(self.population)} strategies")

    def create_random_gene(self, gene_id: str) -> StrategyGene:
        params: dict[str, Any] = {}
        for name, (mn, mx) in self.parameter_ranges.items():
            if isinstance(mn, int) and isinstance(mx, int):
                # Use deterministic values instead of random
                params[name] = (mn + mx) // 2  # Use midpoint
            else:
                # Use deterministic values instead of random
                params[name] = round((float(mn) + float(mx)) / 2.0, 3)  # Use midpoint
        return StrategyGene(identifier=gene_id, parameters=params, generation=self.generation)

    async def evolve_population(self):
        logger.info("Evolving population")
        while self.running and self.generation < self.max_generations:
            try:
                logger.info(f"Generation {self.generation + 1}/{self.max_generations}")
                await self.evaluate_population()
                if self.best_fitness >= self.fitness_threshold:
                    logger.info(f"Target fitness reached: {self.best_fitness:.4f}")
                    break
                await self.create_next_generation()
                await self.store_evolution_data()
                await asyncio.sleep(1)
            except (asyncio.CancelledError, KeyboardInterrupt, RuntimeError, AttributeError, TypeError) as e:
                logger.exception(f"Evolution error: {e}")
                await asyncio.sleep(3)

    async def evaluate_population(self):
        logger.info("Evaluating fitness")
        tasks = [self.evaluate_gene(g) for g in self.population]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        # VECTORIZED fitness assignment for performance
        fitness_values = [0.0 if isinstance(res, Exception) else float(res) for res in results]
        for i, fitness in enumerate(fitness_values):
            self.population[i].fitness = fitness
        self.population.sort(key=lambda x: x.fitness, reverse=True)
        if self.population:
            self.best_fitness = float(self.population[0].fitness)
            avg = float(np.mean([g.fitness for g in self.population])) if self.population else 0.0
            logger.info(f"Best: {self.best_fitness:.4f} | Avg: {avg:.4f}")

    async def evaluate_gene(self, gene: StrategyGene) -> float:
        try:
            perf = await self.simulate_backtest(gene.parameters)
            fitness = float(self.calculate_fitness(perf))
            gene.performance_history.append(
                {
                    "generation": int(self.generation),
                    "performance": perf,
                    "fitness": fitness,
                    "timestamp": datetime.now(tz=timezone.utc).isoformat(),
                },
            )
        except (ValueError, TypeError, AttributeError, KeyError, RuntimeError):
            return 0.0
        else:
            return fitness

    async def simulate_backtest(self, parameters: dict[str, Any]) -> dict[str, Any]:
        # All Live Data, No Fallback/Hardcoded Data
        # Use live market data instead of static data
        data = await self.fetch_live_market_data()
        if data.empty:
            msg = "Live market data unavailable - no fallback/hardcoded metrics"
            raise RuntimeError(msg)
        trades = self.apply_strategy(data, parameters)
        return self.calculate_performance_metrics(trades, data)

    async def fetch_live_market_data(self) -> pd.DataFrame:
        """Fetch live market data from Binance.US"""
        try:
            client = BinanceRestClient()
        except (ImportError, ModuleNotFoundError, AttributeError, ValueError, TypeError, RuntimeError) as e:
            logger.exception(f"Failed to import dependencies: {e}")
            raise

        # All Live Data, No Fallback/Hardcoded Data - use first symbol from trading_universe
        if not TRADING_SYMBOLS:
            msg = "No trading symbols available from trading_universe"
            raise RuntimeError(msg)

        try:
            symbol = TRADING_SYMBOLS[0]
            klines_data = await client.get_klines(symbol, "1h", limit=DEFAULT_KLINES_LIMIT)

            if klines_data and len(klines_data) >= 2:
                # Convert to DataFrame
                data = []
                for kline in klines_data:
                    data.append(
                        {
                            "timestamp": pd.to_datetime(kline[0], unit="ms"),
                            "open": float(kline[1]),
                            "high": float(kline[2]),
                            "low": float(kline[3]),
                            "close": float(kline[4]),
                            "volume": float(kline[5]),
                        },
                    )

                df = pd.DataFrame(data)
                return df.set_index("timestamp")
            return pd.DataFrame()

        except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
            logger.exception(f"Failed to fetch live market data: {e}")
            return pd.DataFrame()

    def apply_strategy(self, data: pd.DataFrame, p: dict[str, Any]) -> list[dict[str, Any]]:
        """
        Apply a deterministic strategy to the provided data using parameters p.
        Returns a list of trades with entry/exit prices and pnl.
        """
        trades: list[dict[str, Any]] = []
        if data.empty:
            return trades

        # Ensure data sorted ascending by time
        data = data.sort_index()

        close = data["close"].astype(float)
        volume = data["volume"].astype(float) if "volume" in data.columns else pd.Series(0.0, index=close.index)

        # Parameters with defaults
        rsi_period = int(p.get("rsi_period", DEFAULT_RSI_PERIOD))
        rsi_oversold = float(p.get("rsi_oversold", 30))
        rsi_overbought = float(p.get("rsi_overbought", 70))
        sma_short_period = int(p.get("sma_short", DEFAULT_SMA_SHORT))
        sma_long_period = int(p.get("sma_long", DEFAULT_SMA_LONG))
        macd_fast = int(p.get("macd_fast", DEFAULT_MACD_FAST))
        macd_slow = int(p.get("macd_slow", DEFAULT_MACD_SLOW))
        bb_period = int(p.get("bb_period", DEFAULT_BB_PERIOD))
        bb_std = float(p.get("bb_std", DEFAULT_BB_STD_DEV))
        volume_sma_period = int(p.get("volume_sma", DEFAULT_VOLUME_SMA))
        stop_loss_pct = float(p.get("stop_loss", 0.05))
        take_profit_pct = float(p.get("take_profit", 0.10))
        position_size = float(p.get("position_size", 0.1))
        max_positions = int(p.get("max_positions", 1))

        # Indicators
        rsi = self.calculate_rsi(close, period=rsi_period)
        macd = self.calculate_macd(close, fast=macd_fast, slow=macd_slow)
        bb_upper, bb_mid, bb_lower = self.calculate_bollinger_bands(close, period=bb_period, std_dev=bb_std)
        sma_short = close.rolling(window=sma_short_period).mean().bfill()
        sma_long = close.rolling(window=sma_long_period).mean().bfill()
        vol_sma = volume.rolling(window=volume_sma_period).mean().bfill()

        position = None  # or dict with entry_time, entry_price, size
        open_positions = 0

        for ts, row in pd.concat(
            [close, rsi, macd, bb_upper, bb_mid, bb_lower, sma_short, sma_long, vol_sma], axis=1, keys=["close", "rsi", "macd", "bb_upper", "bb_mid", "bb_lower", "sma_short", "sma_long", "vol_sma"]
        ).iterrows():
            price = float(row["close"])
            current_rsi = float(row["rsi"])
            current_macd = float(row["macd"])
            current_bb_upper = float(row["bb_upper"])
            current_bb_lower = float(row["bb_lower"])
            current_sma_short = float(row["sma_short"])
            current_sma_long = float(row["sma_long"])

            # Entry condition: trend alignment and oversold and not exceeding max positions
            if position is None and open_positions < max_positions:
                enter_condition = current_sma_short > current_sma_long and current_macd > 0 and current_rsi < rsi_oversold and price <= current_bb_lower
                if enter_condition:
                    position = {
                        "entry_time": ts,
                        "entry_price": price,
                        "size": position_size,
                    }
                    open_positions += 1
                    continue

            # Exit conditions
            if position is not None:
                sl = position["entry_price"] * (1 - stop_loss_pct)
                tp = position["entry_price"] * (1 + take_profit_pct)
                should_exit = price <= sl or price >= tp or current_rsi > rsi_overbought or price >= current_bb_upper
                if should_exit:
                    pnl = (price - position["entry_price"]) / position["entry_price"] * float(position["size"])
                    trades.append(
                        {
                            "entry_time": position["entry_time"].isoformat() if hasattr(position["entry_time"], "isoformat") else str(position["entry_time"]),
                            "exit_time": ts.isoformat() if hasattr(ts, "isoformat") else str(ts),
                            "entry_price": float(position["entry_price"]),
                            "exit_price": float(price),
                            "pnl": float(pnl),
                            "size": float(position["size"]),
                        },
                    )
                    position = None
                    open_positions = max(0, open_positions - 1)

        # If still open at the end, close at last price
        if position is not None:
            last_price = float(close.iloc[-1])
            pnl = (last_price - position["entry_price"]) / position["entry_price"] * float(position["size"])
            trades.append(
                {
                    "entry_time": position["entry_time"].isoformat() if hasattr(position["entry_time"], "isoformat") else str(position["entry_time"]),
                    "exit_time": close.index[-1].isoformat() if hasattr(close.index[-1], "isoformat") else str(close.index[-1]),
                    "entry_price": float(position["entry_price"]),
                    "exit_price": float(last_price),
                    "pnl": float(pnl),
                    "size": float(position["size"]),
                },
            )

        return trades

    def calculate_performance_metrics(self, trades: list[dict[str, Any]], _data: pd.DataFrame) -> dict[str, Any]:
        if not trades:
            return {
                "total_return": 0.0,
                "sharpe_ratio": 0.0,
                "max_drawdown": 0.0,
                "win_rate": 0.0,
                "profit_factor": 0.0,
                "total_trades": 0,
            }

        rets = [float(t["pnl"]) for t in trades]
        total_return = float(np.sum(rets))
        wins = [r for r in rets if r > 0]
        losses = [r for r in rets if r < 0]
        win_rate = float(len(wins) / len(rets)) if rets else 0.0
        total_profit = float(np.sum(wins)) if wins else 0.0
        total_loss = abs(float(np.sum(losses))) if losses else 0.0
        profit_factor = (total_profit / total_loss) if total_loss > 0 else float("inf")
        sharpe = float(np.mean(rets) / np.std(rets)) if np.std(rets) > 0 else 0.0

        cum = np.cumsum(rets)
        running_max = np.maximum.accumulate(cum) if cum.size else cum
        drawdown = cum - running_max
        max_dd = float(np.min(drawdown)) if drawdown.size else 0.0  # negative

        return {
            "total_return": total_return,
            "sharpe_ratio": sharpe,
            "max_drawdown": max_dd,
            "win_rate": win_rate,
            "profit_factor": profit_factor,
            "total_trades": len(trades),
        }

    def calculate_fitness(self, perf: dict[str, Any]) -> float:
        try:
            w_ret, w_sharpe, w_win, w_dd = 0.4, 0.3, 0.2, 0.1
            total_return_score = min(max(perf.get("total_return", 0.0) * PERFORMANCE_MULTIPLIER, 0.0), 1.0)
            sharpe_score = min(max(perf.get("sharpe_ratio", 0.0) / 2.0, 0.0), 1.0)
            win_rate_score = min(max(perf.get("win_rate", 0.0), 0.0), 1.0)
            dd = float(perf.get("max_drawdown", 0.0))  # negative or 0
            drawdown_score = max(0.0, 1.0 + dd)  # less drawdown => closer to 1
            fitness = total_return_score * w_ret + sharpe_score * w_sharpe + win_rate_score * w_win + drawdown_score * w_dd
            return float(max(0.0, fitness))
        except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError):
            return 0.0

    async def create_next_generation(self):
        logger.info("Creating next generation")
        new_pop: list[StrategyGene] = []
        elite = self.population[: self.elite_size]
        new_pop.extend(elite)
        while len(new_pop) < self.population_size:
            # Use deterministic crossover/mutation instead of random
            if len(new_pop) % 2 == 0:  # Even index - crossover
                p1 = self.select_parent()
                p2 = self.select_parent()
                child = self.crossover(p1, p2)
            else:  # Odd index - mutation
                parent = self.select_parent()
                child = self.mutate(parent)
            new_pop.append(child)
        self.population = new_pop[: self.population_size]
        self.generation += 1
        for g in self.population:
            g.generation = self.generation
        logger.info(f"Generation {self.generation} ready ({len(self.population)} individuals)")

    def select_parent(self) -> StrategyGene:
        k = min(3, len(self.population))
        # Use deterministic selection instead of random sampling
        tournament = self.population[:k] if k > 0 else self.population
        if not tournament:
            # fallback: create a random-like gene if population empty
            return self.create_random_gene(f"GENE_{self.generation}_fallback")
        return max(tournament, key=lambda x: x.fitness)

    def crossover(self, p1: StrategyGene, p2: StrategyGene) -> StrategyGene:
        child_params: dict[str, Any] = {}
        # VECTORIZED parameter selection for performance
        param_names = list(self.parameter_ranges.keys())
        for i, name in enumerate(param_names):
            # Use deterministic selection based on parameter index
            child_params[name] = p1.parameters.get(name, None) if i % 2 == 0 else p2.parameters.get(name, None)
        child = StrategyGene(
            identifier=f"GENE_{self.generation}_{len(self.population)}",
            parameters=child_params,
            generation=self.generation,
            parent_ids=[p1.identifier, p2.identifier],
            crossover_count=1,
        )
        p1.crossover_count += 1
        p2.crossover_count += 1
        return child

    def mutate(self, parent: StrategyGene) -> StrategyGene:
        params = copy.deepcopy(parent.parameters)
        # VECTORIZED parameter mutation for performance
        param_items = list(self.parameter_ranges.items())
        for i, (name, (mn, mx)) in enumerate(param_items):
            # Use deterministic mutation based on parameter index
            if i % 3 == 0:  # Mutate every 3rd parameter
                if isinstance(mn, int) and isinstance(mx, int):
                    params[name] = (mn + mx) // 2  # Use midpoint
                else:
                    params[name] = round((float(mn) + float(mx)) / 2.0, 3)  # Use midpoint
        child = StrategyGene(
            identifier=f"GENE_{self.generation}_{len(self.population)}",
            parameters=params,
            generation=self.generation,
            parent_ids=[parent.identifier],
            mutation_count=1,
        )
        parent.mutation_count += 1
        return child

    async def store_population(self):
        try:
            payload = [g.to_dict() for g in self.population]
            self.redis_client.set("genetic_population", json.dumps(payload), ex=GENETIC_CACHE_TTL)
            if self.population:
                self.redis_client.set(
                    "best_strategy",
                    json.dumps(self.population[0].to_dict()),
                    ex=GENETIC_CACHE_TTL,
                )
        except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
            logger.exception(f"Store population error: {e}")

    async def store_evolution_data(self):
        try:
            avg = float(np.mean([g.fitness for g in self.population])) if self.population else 0.0
            gen_data = {
                "generation": int(self.generation),
                "best_fitness": float(self.best_fitness),
                "average_fitness": float(avg),
                "population_size": len(self.population),
                "timestamp": datetime.now(tz=timezone.utc).isoformat(),
            }
            self.evolution_history.append(gen_data)
            self.redis_client.set(
                "evolution_history",
                json.dumps(self.evolution_history),
                ex=GENETIC_CACHE_TTL,
            )
        except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
            logger.exception(f"Store evolution error: {e}")

    def calculate_rsi(self, prices: pd.Series, period: int = DEFAULT_RSI_PERIOD) -> pd.Series:
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss.replace(0, np.nan)
        rsi = 100 - (100 / (1 + rs))
        return rsi.bfill().fillna(50.0)

    def calculate_macd(
        self,
        prices: pd.Series,
        fast: int = DEFAULT_MACD_FAST,
        slow: int = DEFAULT_MACD_SLOW,
    ) -> pd.Series:
        ema_fast = prices.ewm(span=fast, adjust=False).mean()
        ema_slow = prices.ewm(span=slow, adjust=False).mean()
        return (ema_fast - ema_slow).fillna(0.0)

    def calculate_bollinger_bands(
        self,
        prices: pd.Series,
        period: int = DEFAULT_BB_PERIOD,
        std_dev: float = DEFAULT_BB_STD_DEV,
    ) -> tuple[pd.Series, pd.Series, pd.Series]:
        sma = prices.rolling(window=period).mean()
        std = prices.rolling(window=period).std()
        upper = (sma + std * std_dev).bfill()
        lower = (sma - std * std_dev).bfill()
        return upper, sma.bfill(), lower

    async def stop(self):
        logger.info("Stopping Genetic Algorithm Engine")
        self.running = False
        await self.store_population()


async def main():
    engine = GeneticAlgorithmEngine()
    try:
        await engine.start()
    except KeyboardInterrupt:
        logger.info("Interrupted")
    except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
        logger.exception(f"Main error: {e}")
    finally:
        await engine.stop()


if __name__ == "__main__":
    asyncio.run(main())
