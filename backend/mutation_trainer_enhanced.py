from __future__ import annotations

import ast
import hashlib
import json
import logging
import os
import sqlite3
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

# Import from single source of truth
try:
    from backend.config.trading_universe import EXCHANGE_ID
except (ImportError, ModuleNotFoundError, AttributeError, ValueError, TypeError, RuntimeError) as e:
    msg = f"Failed to import EXCHANGE_ID from trading_universe: {e}"
    raise RuntimeError(msg) from e


def _to_ccxt_symbol(base: str, quote: str) -> str:
    return f"{base.upper()}/{quote.upper()}"


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

TRAINER_DB = os.getenv("MUTATION_TRAINER_DB", "./data/mutation_trainer.db")
TRAINER_INTERVAL = 7200
POPULATION_SIZE = 50
GENERATIONS = 10
MUTATION_RATE = 0.1
CROSSOVER_RATE = 0.8
ELITE_SIZE = 5

Path("./data").mkdir(parents=True, exist_ok=True)
Path("./logs").mkdir(parents=True, exist_ok=True)
Path("./strategies").mkdir(parents=True, exist_ok=True)


class TrainerDatabase:
    def __init__(self, db_path: str) -> None:
        self.db_path = db_path
        self.init_database()

    def init_database(self):
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute(
                    """
                    CREATE TABLE IF NOT EXISTS genetic_population (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        generation INTEGER NOT NULL,
                        individual_id TEXT NOT NULL,
                        strategy_code TEXT NOT NULL,
                        fitness_score REAL NOT NULL,
                        mutation_history TEXT NOT NULL,
                        parent_ids TEXT,
                        created_at TEXT DEFAULT CURRENT_TIMESTAMP
                    )
                """,
                )
                cursor.execute(
                    """
                    CREATE TABLE IF NOT EXISTS evolution_progress (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        generation INTEGER NOT NULL,
                        avg_fitness REAL NOT NULL,
                        best_fitness REAL NOT NULL,
                        worst_fitness REAL NOT NULL,
                        diversity_score REAL NOT NULL,
                        convergence_rate REAL NOT NULL,
                        created_at TEXT DEFAULT CURRENT_TIMESTAMP
                    )
                """,
                )
                cursor.execute(
                    """
                    CREATE TABLE IF NOT EXISTS mutation_operations (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        timestamp TEXT NOT NULL,
                        operation_type TEXT NOT NULL,
                        parent_id TEXT NOT NULL,
                        child_id TEXT NOT NULL,
                        mutation_details TEXT NOT NULL,
                        fitness_improvement REAL,
                        created_at TEXT DEFAULT CURRENT_TIMESTAMP
                    )
                """,
                )
                conn.commit()
        except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
            logger.exception("Database initialization error: %s", e)

    def save_individual(self, individual: dict):
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute(
                    """
                    INSERT INTO genetic_population
                    (generation, individual_id, strategy_code, fitness_score, mutation_history, parent_ids)
                    VALUES (?, ?, ?, ?, ?, ?)
                """,
                    (
                        individual["generation"],
                        individual["individual_id"],
                        individual["strategy_code"],
                        float(individual["fitness_score"]),
                        json.dumps(individual["mutation_history"]),
                        json.dumps(individual.get("parent_ids", [])),
                    ),
                )
                conn.commit()
        except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
            logger.exception("Error saving individual: %s", e)

    def save_evolution_progress(self, progress: dict):
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute(
                    """
                    INSERT INTO evolution_progress
                    (generation, avg_fitness, best_fitness, worst_fitness, diversity_score, convergence_rate)
                    VALUES (?, ?, ?, ?, ?, ?)
                """,
                    (
                        int(progress["generation"]),
                        float(progress["avg_fitness"]),
                        float(progress["best_fitness"]),
                        float(progress["worst_fitness"]),
                        float(progress["diversity_score"]),
                        float(progress["convergence_rate"]),
                    ),
                )
                conn.commit()
        except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
            logger.exception("Error saving evolution progress: %s", e)

    def save_mutation_operation(self, operation: dict):
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute(
                    """
                    INSERT INTO mutation_operations
                    (timestamp, operation_type, parent_id, child_id, mutation_details, fitness_improvement)
                    VALUES (?, ?, ?, ?, ?, ?)
                """,
                    (
                        operation["timestamp"],
                        operation["operation_type"],
                        operation["parent_id"],
                        operation["child_id"],
                        json.dumps(operation["mutation_details"]),
                        float(operation.get("fitness_improvement", 0.0)),
                    ),
                )
                conn.commit()
        except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
            logger.exception("Error saving mutation operation: %s", e)


def generate_base_strategy() -> str:
    return """
def base_strategy(df):
    import pandas as pd

    df['sma_20'] = df['close'].rolling(window=20).mean()
    df['sma_50'] = df['close'].rolling(window=50).mean()

    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / (loss.replace(0, 1))
    df['rsi'] = 100 - (100 / (1 + rs))

    df['signal'] = 0
    df.loc[(df['close'] > df['sma_20']) & (df['rsi'] < 70), 'signal'] = 1
    df.loc[(df['close'] < df['sma_20']) & (df['rsi'] > 30), 'signal'] = -1

    return df
""".strip("\n")


def parse_strategy_code(code: str) -> dict[str, Any]:
    try:
        tree = ast.parse(code)
        components = {"imports": [], "indicators": [], "conditions": [], "signals": []}
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                components["imports"].extend([alias.name for alias in node.names])
            elif isinstance(node, ast.ImportFrom):
                components["imports"].append(node.module or "")
            elif isinstance(node, ast.Call):
                f = getattr(node, "func", None)
                if hasattr(f, "id"):
                    components["indicators"].append(f.id)
                elif hasattr(f, "attr"):
                    components["indicators"].append(f.attr)
    except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
        logger.exception("Error parsing strategy code: %s", e)
        return {"imports": [], "indicators": [], "conditions": [], "signals": []}
    else:
        return components


def mutate_strategy_code(code: str, mutation_type: str = "random") -> str:
    try:
        lines = code.split("\n")
        mutated_lines = lines.copy()

        # Default 'random' to 'parameter' deterministic mutation
        if mutation_type == "random":
            mutation_type = "parameter"

        if mutation_type == "parameter":
            for i, line in enumerate(mutated_lines):
                if "rolling(window=" in line:
                    try:
                        start = line.index("rolling(window=") + len("rolling(window=")
                        end = line.index(")", start)
                        old_window = int(line[start:end])
                        new_window = max(5, min(100, old_window + 5))  # deterministic adjustment
                        mutated_lines[i] = line[:start] + str(new_window) + line[end:]
                    except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError):
                        continue
                if "df['rsi']" in line and ("<" in line or ">" in line):
                    # Replace common RSI thresholds deterministically
                    try:
                        # replace "< 70" -> "< 75", "> 30" -> "> 35"
                        if "< 70" in line:
                            mutated_lines[i] = line.replace("< 70", "< 75")
                        if "> 30" in line:
                            mutated_lines[i] = line.replace("> 30", "> 35")
                    except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError):
                        continue

        elif mutation_type == "indicator":
            # Add a simple EMA indicator if not present
            has_ema = any("ewm(" in line or "ema_" in line for line in mutated_lines)
            if not has_ema:
                insert_idx = 0
                for idx, line in enumerate(mutated_lines):
                    if "sma_20" in line:
                        insert_idx = idx + 1
                        break
                ema_line = "    df['ema_10'] = df['close'].ewm(span=10, adjust=False).mean()"
                mutated_lines.insert(insert_idx, ema_line)

        elif mutation_type == "condition":
            # Flip comparison operators in signal conditions deterministically
            for i, line in enumerate(mutated_lines):
                if "df.loc" in line or ("signal" in line and ("&" in line or "|" in line)):
                    # swap '>' and '<' carefully
                    s = line
                    s = s.replace(" >= ", " __GE__ ")
                    s = s.replace(" <= ", " __LE__ ")
                    s = s.replace(" > ", " __GT__ ")
                    s = s.replace(" < ", " __LT__ ")
                    s = s.replace(" __GT__ ", " < ")
                    s = s.replace(" __LT__ ", " > ")
                    s = s.replace(" __GE__ ", " <= ")
                    s = s.replace(" __LE__ ", " >= ")
                    mutated_lines[i] = s

        # join back
        return "\n".join(mutated_lines)
    except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
        logger.exception("Mutation error: %s", e)
        return code


def crossover_strategies(parent1_code: str, parent2_code: str) -> tuple[str, str]:
    try:
        lines1 = parent1_code.split("\n")
        lines2 = parent2_code.split("\n")
        if not lines1 or not lines2:
            return parent1_code, parent2_code
        # Use middle point instead of random
        crossover_point = min(len(lines1), len(lines2)) // 2
        child1_lines = lines1[:crossover_point] + lines2[crossover_point:]
        child2_lines = lines2[:crossover_point] + lines1[crossover_point:]
        child1_code = fix_indentation("\n".join(child1_lines))
        child2_code = fix_indentation("\n".join(child2_lines))
    except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
        logger.exception("Crossover error: %s", e)
        return parent1_code, parent2_code
    else:
        return child1_code, child2_code


def fix_indentation(code: str) -> str:
    lines = code.split("\n")
    fixed_lines: list[str] = []
    for line in lines:
        s = line.rstrip("\r")
        if not s.strip():
            fixed_lines.append("")
        elif s.lstrip().startswith(("def ", "import ", "from ")):
            fixed_lines.append(s.strip())
        else:
            # Ensure a single level of indentation for function body lines
            fixed_lines.append("    " + s.strip())
    return "\n".join(fixed_lines)


def evaluate_fitness(strategy_code: str) -> float:
    try:
        components = parse_strategy_code(strategy_code)
        fitness_score = 0.0
        complexity_penalty = len(strategy_code.split("\n")) * 0.01
        fitness_score -= complexity_penalty
        unique_indicators = len(set(components["indicators"]))
        fitness_score += unique_indicators * 0.1
        try:
            ast.parse(strategy_code)
            fitness_score += 0.5
        except SyntaxError as e:
            logger.warning("Invalid syntax in strategy code: %s", e)
            fitness_score -= 1.0
        if "signal" in strategy_code:
            fitness_score += 0.3
        if "def " in strategy_code:
            fitness_score += 0.2
        performance_bonus = 0.25  # deterministic bonus
        fitness_score += performance_bonus
        return max(0.0, float(fitness_score))
    except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
        logger.exception("Fitness evaluation error: %s", e)
        return 0.0


def create_individual(generation: int, strategy_code: str | None = None) -> dict:
    code = strategy_code or generate_base_strategy()
    individual_id = hashlib.md5(code.encode("utf-8")).hexdigest()[:8]
    fitness_score = evaluate_fitness(code)
    return {
        "generation": int(generation),
        "individual_id": individual_id,
        "strategy_code": code,
        "fitness_score": float(fitness_score),
        "mutation_history": [],
        "parent_ids": [],
    }


def select_parents(population: list[dict], tournament_size: int = 3) -> tuple[dict, dict]:
    if len(population) < 2:
        # Return the best individual twice if there is only one
        best = max(population, key=lambda x: x["fitness_score"]) if population else {}
        return best, best
    # Use deterministic selection instead of random
    tournament1 = population[: min(tournament_size, len(population))]  # Use first N strategies
    tournament2 = population[: min(tournament_size, len(population))]  # Use first N strategies
    parent1 = max(tournament1, key=lambda x: x["fitness_score"])
    parent2 = max(tournament2, key=lambda x: x["fitness_score"])
    return parent1, parent2


def evolve_population(population: list[dict], generation: int) -> list[dict]:
    new_population: list[dict] = []
    sorted_population = sorted(population, key=lambda x: x["fitness_score"], reverse=True)
    elite = sorted_population[:ELITE_SIZE]
    new_population.extend(elite)

    while len(new_population) < POPULATION_SIZE:
        if len(new_population) % 2 == 0 and len(population) >= 2:  # deterministic condition
            parent1, parent2 = select_parents(population)
            child1_code, child2_code = crossover_strategies(parent1["strategy_code"], parent2["strategy_code"])
            child1 = create_individual(generation, child1_code)
            child2 = create_individual(generation, child2_code)
            child1["parent_ids"] = [parent1["individual_id"], parent2["individual_id"]]
            child2["parent_ids"] = [parent1["individual_id"], parent2["individual_id"]]
            new_population.extend([child1, child2])
        else:
            parent = population[len(new_population) % len(population)]  # deterministic selection
            mutation_types = ["parameter", "indicator", "condition"]
            mutation_type = mutation_types[len(new_population) % len(mutation_types)]  # deterministic
            mutated_code = mutate_strategy_code(parent["strategy_code"], mutation_type)
            child = create_individual(generation, mutated_code)
            child["parent_ids"] = [parent["individual_id"]]
            child["mutation_history"] = [*parent.get("mutation_history", []), mutation_type]
            new_population.append(child)

    return new_population[:POPULATION_SIZE]


def calculate_population_metrics(population: list[dict]) -> dict:
    if not population:
        return {
            "avg_fitness": 0.0,
            "best_fitness": 0.0,
            "worst_fitness": 0.0,
            "diversity_score": 0.0,
            "convergence_rate": 0.0,
        }
    fitness_scores = np.array([float(ind["fitness_score"]) for ind in population], dtype=float)
    avg = float(np.mean(fitness_scores))
    std = float(np.std(fitness_scores))
    return {
        "avg_fitness": avg,
        "best_fitness": float(np.max(fitness_scores)),
        "worst_fitness": float(np.min(fitness_scores)),
        "diversity_score": std,
        "convergence_rate": (1.0 - (std / avg)) if avg > 0 else 0.0,
    }


def train_mutations_enhanced():
    try:
        db = TrainerDatabase(TRAINER_DB)
        population: list[dict] = []
        for _ in range(POPULATION_SIZE):
            individual = create_individual(0)
            population.append(individual)
            db.save_individual(individual)
        logger.info("Initialized population of %d individuals", POPULATION_SIZE)

        for generation in range(GENERATIONS):
            logger.info("Starting generation %d/%d", generation + 1, GENERATIONS)
            metrics = calculate_population_metrics(population)
            metrics["generation"] = generation + 1
            db.save_evolution_progress(metrics)
            logger.info(
                "Generation %d metrics: avg=%.3f best=%.3f diversity=%.3f",
                generation + 1,
                metrics["avg_fitness"],
                metrics["best_fitness"],
                metrics["diversity_score"],
            )

            new_population = evolve_population(population, generation + 1)
            for individual in new_population:
                db.save_individual(individual)

            for individual in new_population:
                if individual.get("parent_ids"):
                    for parent_id in individual["parent_ids"]:
                        operation = {
                            "timestamp": datetime.now(timezone.utc).isoformat(),
                            "operation_type": "mutation" if len(individual["parent_ids"]) == 1 else "crossover",
                            "parent_id": parent_id,
                            "child_id": individual["individual_id"],
                            "mutation_details": {
                                "mutation_history": individual.get("mutation_history", []),
                                "fitness_score": float(individual["fitness_score"]),
                            },
                            "fitness_improvement": 0.0,
                        }
                        db.save_mutation_operation(operation)

            population = new_population
            if metrics["convergence_rate"] > 0.9:
                logger.info("Population converged at generation %d", generation + 1)
                break

        best_individual = max(population, key=lambda x: x["fitness_score"])
        logger.info("Training complete")
        logger.info("Best individual ID: %s", best_individual["individual_id"])
        logger.info("Best fitness score: %.3f", best_individual["fitness_score"])
        logger.info(
            "Best strategy code length: %d characters",
            len(best_individual["strategy_code"]),
        )

        file_path = Path(f"./strategies/best_evolved_strategy_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}.py")
        file_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            with file_path.open("w", encoding="utf-8") as f:
                f.write(best_individual["strategy_code"])
            logger.info("Best strategy saved to %s", file_path)
        except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
            logger.exception("Failed to save best strategy: %s", e)

    except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
        logger.exception("Enhanced training error: %s", e)


def _main_loop():
    while True:
        try:
            train_mutations_enhanced()
        except KeyboardInterrupt:
            logger.info("Shutdown requested by user")
            break
        except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
            logger.exception("Trainer loop error: %s", e)
        # Sleep for configured training interval - sync sleep OK for standalone trainer script
        time.sleep(TRAINER_INTERVAL)


if __name__ == "__main__":
    _main_loop()
