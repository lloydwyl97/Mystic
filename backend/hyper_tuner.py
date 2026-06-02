"""
Hyperparameter Optimization Engine for AI Trading Strategies
Auto-tunes strategy parameters to maximize profit, win rate, and Sharpe ratio.
Windows 11 Pro + Python 3.12. Binance US only implementation.
"""

import logging
import random
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from typing import Any

# Import from single source of truth
try:
    from backend.config.trading_universe import (
        EXCHANGE_ID,
        TOP10_COINS,
        TRADING_SYMBOLS,
    )
except (ImportError, ModuleNotFoundError, AttributeError, ValueError, TypeError, RuntimeError) as e:
    msg = f"Failed to import trading_universe: {e}"
    raise RuntimeError(msg) from e

try:
    from backend.modules.market.binance_data_fetcher import _to_ccxt_symbol
except (ImportError, ModuleNotFoundError, AttributeError, ValueError, TypeError, RuntimeError) as e:
    msg = f"Failed to import _to_ccxt_symbol from binance_data_fetcher: {e}"
    raise RuntimeError(msg) from e

from strat_versions import save_strategy_version

try:
    from backend.services.backtester import run_backtest  # type: ignore[import-not-found]
except ImportError:
    try:
        from backtester import run_backtest  # type: ignore[import-not-found]
    except ImportError:
        run_backtest = None  # type: ignore[assignment]

# Use trading_universe symbols (live data)
DEFAULT_BASE_SYMBOLS = list(TOP10_COINS)


def _resolve_backtester():
    return run_backtest


logger = logging.getLogger("hyper_tuner")
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


class HyperparameterTuner:
    def __init__(self, max_workers: int = 4) -> None:
        self.max_workers = max_workers
        self.best_configs: list[dict[str, Any]] = []
        self.optimization_history: list[dict[str, Any]] = []
        self._backtester = _resolve_backtester()

    def generate_random_config(self, strategy_type: str = "rsi_ema_breakout") -> dict[str, Any]:
        if strategy_type == "rsi_ema_breakout":
            return {
                "rsi_period": random.randint(5, 30),
                "rsi_oversold": random.randint(20, 35),
                "rsi_overbought": random.randint(65, 80),
                "ema_fast": random.randint(5, 20),
                "ema_slow": random.randint(21, 100),
                "breakout_threshold": round(random.uniform(0.01, 0.05), 4),
                "volume_multiplier": round(random.uniform(1.0, 3.0), 2),
                "stop_loss_pct": round(random.uniform(0.02, 0.08), 4),
                "take_profit_pct": round(random.uniform(0.03, 0.12), 4),
            }
        if strategy_type == "bollinger_bands":
            return {
                "bb_period": random.randint(10, 50),
                "bb_std": round(random.uniform(1.5, 3.0), 2),
                "volume_threshold": round(random.uniform(1.0, 2.5), 2),
                "rsi_period": random.randint(10, 25),
                "rsi_oversold": random.randint(25, 40),
                "rsi_overbought": random.randint(60, 75),
            }
        if strategy_type == "macd_crossover":
            return {
                "macd_fast": random.randint(8, 15),
                "macd_slow": random.randint(20, 35),
                "macd_signal": random.randint(5, 15),
                "volume_filter": round(random.uniform(1.0, 2.0), 2),
                "trend_strength": round(random.uniform(0.5, 1.5), 2),
            }
        return {
            "period_1": random.randint(5, 50),
            "period_2": random.randint(10, 100),
            "threshold": round(random.uniform(0.01, 0.10), 4),
            "multiplier": round(random.uniform(0.5, 3.0), 2),
        }

    def mutate_config(self, base_config: dict[str, Any], mutation_rate: float = 0.3) -> dict[str, Any]:
        mutated = dict(base_config)
        for k, v in list(mutated.items()):
            if random.random() < mutation_rate:
                if isinstance(v, int):
                    if "period" in k.lower():
                        mutated[k] = max(1, v + random.randint(-5, 5))
                    else:
                        mutated[k] = max(1, v + random.randint(-2, 2))
                elif isinstance(v, float):
                    if "threshold" in k.lower() or "pct" in k.lower():
                        mutated[k] = max(0.001, round(v * random.uniform(0.8, 1.2), 6))
                    else:
                        mutated[k] = max(0.1, round(v * random.uniform(0.7, 1.3), 6))
        return mutated

    def crossover_configs(self, config1: dict[str, Any], config2: dict[str, Any]) -> dict[str, Any]:
        child: dict[str, Any] = {}
        keys = set(config1.keys()) | set(config2.keys())
        for k in keys:
            chosen = (config1[k] if random.random() < 0.5 else config2[k]) if k in config1 and k in config2 else config1.get(k, config2.get(k))
            if isinstance(chosen, (int, float)) and random.random() < 0.2:
                chosen = max(1, chosen + random.randint(-1, 1)) if isinstance(chosen, int) else max(0.001, round(chosen * random.uniform(0.95, 1.05), 6))
            child[k] = chosen
        return child

    def evaluate_config(
        self,
        config: dict[str, Any],
        strategy_type: str = "rsi_ema_breakout",
        symbols: list[str] | None = None,
        timeframe: str = "1h",
        lookback_days: int = 14,
    ) -> dict[str, Any]:
        if not self._backtester:
            logger.error("Backtester module not available. Cannot evaluate configuration without live/backtest engine.")
            return {
                "config": config,
                "total_profit": -1.0,
                "win_rate": 0.0,
                "trades_count": 0,
                "sharpe_ratio": 0.0,
                "evaluation_time": datetime.now(timezone.utc).isoformat(),
                "error": "backtester_unavailable",
            }
        try:
            # All Live Data, No Fallback/Hardcoded Data
            # Use trading_universe symbols (live data)
            bases = symbols or list(TOP10_COINS)
            ccxt_symbols = [_to_ccxt_symbol(b) for b in bases]
            result = self._backtester(
                exchange_id=EXCHANGE_ID,
                strategy_type=strategy_type,
                params=config,
                symbols=ccxt_symbols,
                timeframe=timeframe,
                lookback_days=lookback_days,
            )
            profit = float(result.get("total_profit", 0.0))
            win_rate = float(result.get("win_rate", 0.0))
            trades = int(result.get("trades_count", 0))
            sharpe = float(result.get("sharpe_ratio", 0.0))
            return {
                "config": config,
                "total_profit": round(profit, 6),
                "win_rate": round(win_rate, 6),
                "trades_count": trades,
                "sharpe_ratio": round(sharpe, 6),
                "evaluation_time": datetime.now(timezone.utc).isoformat(),
            }
        except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
            logger.exception("Evaluation error: %s", e)
            return {
                "config": config,
                "total_profit": -1.0,
                "win_rate": 0.0,
                "trades_count": 0,
                "sharpe_ratio": 0.0,
                "evaluation_time": datetime.now(timezone.utc).isoformat(),
                "error": str(e),
            }

    def run_random_search(
        self,
        strategy_type: str = "rsi_ema_breakout",
        rounds: int = 100,
        save_best: bool = True,
        symbols: list[str] | None = None,
        timeframe: str = "1h",
        lookback_days: int = 14,
    ) -> list[dict[str, Any]]:
        logger.info("Starting random search for %s, rounds=%d", strategy_type, rounds)
        best_configs: list[dict[str, Any]] = []
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {
                executor.submit(
                    self.evaluate_config,
                    self.generate_random_config(strategy_type),
                    strategy_type,
                    symbols,
                    timeframe,
                    lookback_days,
                ): i
                for i in range(rounds)
            }
            for fut in as_completed(futures):
                res = fut.result()
                trial_num = futures[fut] + 1
                # Safe-access values for logging
                total_profit = float(res.get("total_profit", -1.0))
                win_rate = float(res.get("win_rate", 0.0))
                trades_count = int(res.get("trades_count", 0))
                sharpe = float(res.get("sharpe_ratio", 0.0))
                logger.info(
                    "Trial %03d | Profit=%10.4f | WinRate=%6.2f%% | Trades=%3d | Sharpe=%7.4f",
                    trial_num,
                    total_profit,
                    win_rate * 100.0,
                    trades_count,
                    sharpe,
                )
                if res.get("error"):
                    continue
                if res["total_profit"] > 0:
                    best_configs.append(res)
                    best_configs.sort(key=lambda x: x["total_profit"], reverse=True)
                    best_configs = best_configs[:10]
        logger.info("Random search complete. Profitable configs found: %d", len(best_configs))
        if best_configs and save_best:
            cfg = best_configs[0]
            name = f"{strategy_type}_optimized_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
            save_strategy_version(name, cfg)
            logger.info("Saved best configuration as %s", name)
        return best_configs

    def run_genetic_optimization(
        self,
        strategy_type: str = "rsi_ema_breakout",
        population_size: int = 20,
        generations: int = 10,
        mutation_rate: float = 0.3,
        save_best: bool = True,
        symbols: list[str] | None = None,
        timeframe: str = "1h",
        lookback_days: int = 14,
    ) -> list[dict[str, Any]]:
        logger.info(
            "Starting genetic optimization for %s | population=%d | generations=%d",
            strategy_type,
            population_size,
            generations,
        )
        population: list[dict[str, Any]] = []
        for _ in range(population_size):
            cfg = self.generate_random_config(strategy_type)
            res = self.evaluate_config(cfg, strategy_type, symbols, timeframe, lookback_days)
            population.append(res)
        best_configs: list[dict[str, Any]] = []
        for g in range(generations):
            population.sort(key=lambda x: x["total_profit"], reverse=True)
            best_configs.append(population[0])
            logger.info(
                "Generation %02d | Best Profit=%10.4f | WinRate=%6.2f%% | Sharpe=%7.4f",
                g + 1,
                population[0]["total_profit"],
                population[0]["win_rate"] * 100.0,
                population[0]["sharpe_ratio"],
            )
            elite_count = max(1, population_size // 5)
            new_population = list(population[:elite_count])
            while len(new_population) < population_size:
                p1 = random.choice(population[: max(2, population_size // 2)])
                p2 = random.choice(population[: max(2, population_size // 2)])
                child_cfg = self.crossover_configs(p1["config"], p2["config"])
                if random.random() < mutation_rate:
                    child_cfg = self.mutate_config(child_cfg, mutation_rate)
                child_res = self.evaluate_config(child_cfg, strategy_type, symbols, timeframe, lookback_days)
                new_population.append(child_res)
            population = new_population
        # Ensure returned best_configs is sorted with the best first
        best_configs = sorted(best_configs, key=lambda x: x["total_profit"], reverse=True) if best_configs else []
        if best_configs:
            logger.info(
                "Genetic optimization complete. Best Profit=%10.4f",
                best_configs[0]["total_profit"],
            )
            if save_best:
                cfg = best_configs[0]
                name = f"{strategy_type}_genetic_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
                save_strategy_version(name, cfg)
                logger.info("Saved best configuration as %s", name)
        else:
            logger.info("Genetic optimization complete. No profitable configurations found.")
        return best_configs

    def run_bayesian_optimization(
        self,
        strategy_type: str = "rsi_ema_breakout",
        rounds: int = 50,
        save_best: bool = True,
        symbols: list[str] | None = None,
        timeframe: str = "1h",
        lookback_days: int = 14,
    ) -> list[dict[str, Any]]:
        logger.info(
            "Starting Bayesian-like optimization for %s | rounds=%d",
            strategy_type,
            rounds,
        )
        best_configs: list[dict[str, Any]] = []
        explored: list[dict[str, Any]] = []
        for r in range(rounds):
            if r < rounds // 3 or not best_configs:
                cfg = self.generate_random_config(strategy_type)
            else:
                base = random.choice(best_configs[: min(3, len(best_configs))])["config"]
                cfg = self.mutate_config(base, mutation_rate=0.2)
            res = self.evaluate_config(cfg, strategy_type, symbols, timeframe, lookback_days)
            explored.append(res)
            logger.info(
                "Round %03d | Profit=%10.4f | WinRate=%6.2f%% | Sharpe=%7.4f",
                r + 1,
                res["total_profit"],
                res["win_rate"] * 100.0,
                res["sharpe_ratio"],
            )
            if res.get("error"):
                continue
            if res["total_profit"] > 0:
                best_configs.append(res)
                best_configs.sort(key=lambda x: x["total_profit"], reverse=True)
                best_configs = best_configs[:10]
        if best_configs:
            logger.info(
                "Bayesian optimization complete. Best Profit=%10.4f",
                best_configs[0]["total_profit"],
            )
            if save_best:
                cfg = best_configs[0]
                name = f"{strategy_type}_bayesian_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
                save_strategy_version(name, cfg)
                logger.info("Saved best configuration as %s", name)
        else:
            logger.warning("Bayesian optimization complete. No profitable configurations found.")
        return best_configs

    def optimize_strategy(
        self,
        strategy_type: str = "rsi_ema_breakout",
        method: str = "genetic",
        **kwargs,
    ) -> dict[str, Any] | None:
        start = time.time()
        if method == "random":
            best = self.run_random_search(strategy_type, **kwargs)
        elif method == "genetic":
            best = self.run_genetic_optimization(strategy_type, **kwargs)
        elif method == "bayesian":
            best = self.run_bayesian_optimization(strategy_type, **kwargs)
        else:
            msg = f"Unknown optimization method: {method}"
            raise ValueError(msg)
        elapsed = time.time() - start
        if not best:
            logger.error("No profitable configurations found. Duration=%.1fs", elapsed)
            return None
        top = best[0]
        logger.info(
            "Optimization complete in %.1fs | Best Profit=%10.4f | WinRate=%6.2f%% | Sharpe=%7.4f | Trades=%d",
            elapsed,
            top["total_profit"],
            top["win_rate"] * 100.0,
            top["sharpe_ratio"],
            top["trades_count"],
        )
        return top


def optimize_rsi_ema_breakout(method: str = "genetic", rounds: int = 50) -> dict[str, Any] | None:
    tuner = HyperparameterTuner()
    return tuner.optimize_strategy("rsi_ema_breakout", method, rounds=rounds)


def optimize_bollinger_bands(method: str = "genetic", rounds: int = 50) -> dict[str, Any] | None:
    tuner = HyperparameterTuner()
    return tuner.optimize_strategy("bollinger_bands", method, rounds=rounds)


def optimize_macd_crossover(method: str = "genetic", rounds: int = 50) -> dict[str, Any] | None:
    tuner = HyperparameterTuner()
    return tuner.optimize_strategy("macd_crossover", method, rounds=rounds)


if __name__ == "__main__":
    logging.getLogger().setLevel(logging.INFO)
    tuner = HyperparameterTuner()
    if not tuner._backtester:
        logger.error("Backtester not available. Exiting.")
        sys_exit_code = 2
        import sys

        sys.exit(sys_exit_code)
    res = optimize_rsi_ema_breakout(method="genetic", rounds=20)
    if not res:
        import sys

        sys.exit(1)
