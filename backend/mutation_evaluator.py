import contextlib
import json
import logging
import os
import tempfile
import time
from collections.abc import Iterable
from pathlib import Path
from typing import Any

# Ensure log directory exists before configuring handlers
Path("logs").mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("logs/mutation_evaluator.log", encoding="utf-8"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger("mutation_evaluator")

MUTATION_FILE = "mutations.json"
LEADERBOARD_FILE = "mutation_leaderboard.json"
MIN_WIN_RATE = 0.55
MIN_PROFIT = 10.0
EVALUATION_INTERVAL_SEC = 300


def _atomic_json_write(path: str, data: Any) -> None:
    directory = Path(path).parent
    directory.mkdir(parents=True, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(prefix=".tmp_", dir=directory)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as tmp_file:
            json.dump(data, tmp_file, indent=2)
            tmp_file.flush()
            os.fsync(tmp_file.fileno())
        Path(tmp_path).replace(Path(path))
    except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError):
        with contextlib.suppress(ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError):
            Path(tmp_path).unlink()
        raise


def load_file(path: str) -> list:
    try:
        path_obj = Path(path)
        with path_obj.open(encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            # Only accept dict entries
            result = [x for x in data if isinstance(x, dict)]
        else:
            logger.warning("File %s did not contain a list; ignoring", path)
            result = []
    except (OSError, FileNotFoundError, PermissionError, json.JSONDecodeError) as e:
        logger.warning("Failed to load file %s: %s", path, e)
        return []
    else:
        return result


def save_file(path: str, data: list) -> None:
    _atomic_json_write(path, data)


def _valid_mutation(entry: dict) -> bool:
    try:
        _ = float(entry.get("win_rate", 0))
        _ = float(entry.get("profit", 0))
    except (TypeError, ValueError, AttributeError):
        return False
    else:
        return True


def _already_promoted(entry: dict) -> bool:
    return bool(entry.get("promoted"))


def _meets_criteria(entry: dict) -> bool:
    try:
        win_rate = float(entry.get("win_rate", 0))
        profit = float(entry.get("profit", 0))
    except (TypeError, ValueError, AttributeError):
        return False
    else:
        return (win_rate >= MIN_WIN_RATE) and (profit >= MIN_PROFIT)


def _dedup_by_id(items: Iterable) -> list:
    seen = set()
    deduped = []
    for x in items:
        if not isinstance(x, dict):
            continue
        sid = str(x.get("id", ""))
        key = sid if sid else json.dumps(x, sort_keys=True)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(x)
    return deduped


def evaluate_mutations() -> None:
    mutations = load_file(MUTATION_FILE)
    leaderboard = load_file(LEADERBOARD_FILE)

    if not mutations:
        logger.info("No mutations found to evaluate")
        return

    promoted_count = 0
    updated_mutations = []

    for m in mutations:
        if not isinstance(m, dict):
            logger.warning("Skipping non-dict mutation entry: %s", m)
            updated_mutations.append(m)
            continue

        if not _valid_mutation(m):
            logger.warning("Skipping invalid mutation entry: %s", m)
            updated_mutations.append(m)
            continue

        if _already_promoted(m):
            updated_mutations.append(m)
            continue

        if _meets_criteria(m):
            m["promoted"] = True
            leaderboard.append(m)
            promoted_count += 1
            try:
                win_rate_val = float(m.get("win_rate", 0))
            except (TypeError, ValueError, AttributeError):
                win_rate_val = 0.0
            try:
                profit_val = float(m.get("profit", 0))
            except (TypeError, ValueError, AttributeError):
                profit_val = 0.0
            logger.info(
                "Promoted strategy: %s (win_rate=%.4f, profit=%.4f)",
                m.get("id", "unknown"),
                win_rate_val,
                profit_val,
            )

        updated_mutations.append(m)

    if promoted_count > 0:
        leaderboard = _dedup_by_id(leaderboard)
        leaderboard.sort(
            key=lambda x: (
                float(x.get("profit", 0) or 0),
                float(x.get("win_rate", 0) or 0),
            ),
            reverse=True,
        )
        try:
            save_file(LEADERBOARD_FILE, leaderboard)
            save_file(MUTATION_FILE, updated_mutations)
            logger.info("Promoted %d strategies to leaderboard", promoted_count)
        except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
            logger.exception("Failed to persist leaderboard or mutation updates: %s", e)
    else:
        logger.info("No new strategies promoted")


def run_continuous_evaluation() -> None:
    logger.info(
        "Starting continuous mutation evaluation | min_win_rate=%.2f | min_profit=%.2f | interval=%ds",
        MIN_WIN_RATE,
        MIN_PROFIT,
        EVALUATION_INTERVAL_SEC,
    )
    while True:
        try:
            evaluate_mutations()
            logger.info("Sleeping for %d seconds", EVALUATION_INTERVAL_SEC)
            time.sleep(EVALUATION_INTERVAL_SEC)
        except KeyboardInterrupt:
            logger.info("Mutation evaluation stopped by user")
            break
        except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
            logger.exception("Error in mutation evaluation loop: %s", e)
            time.sleep(EVALUATION_INTERVAL_SEC)


if __name__ == "__main__":
    run_continuous_evaluation()
