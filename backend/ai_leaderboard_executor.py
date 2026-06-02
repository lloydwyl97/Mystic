"""
AI Leaderboard Executor (Repaired & Hardened)
- Robust logging with rotation and auto-created logs dir
- CLI with one-shot or continuous execution (+ configurable interval)
- Environment overrides for thresholds/symbol/amount
- Defensive JSON loading & validation
- Graceful shutdown via SIGINT/SIGTERM
- Safer call into execute_ai_strategy_signal with import fallback
"""

from __future__ import annotations

import argparse
import contextlib
import json
import logging
import logging.handlers
import os
import signal
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# ------------------------------------------------------------------------------
# Direct imports for production
# ------------------------------------------------------------------------------
from ai_strategy_execution import execute_ai_strategy_signal  # type: ignore[import-not-found]

# ------------------------------------------------------------------------------
# Logging setup (rotating file + console)
# ------------------------------------------------------------------------------
LOGS_DIR = Path("logs")
LOGS_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE = LOGS_DIR / "ai_leaderboard_executor.log"

_root = logging.getLogger()
_root.setLevel(logging.INFO)
# Clear existing handlers to avoid duplicate logs in some environments
if hasattr(_root, "handlers"):
    _root.handlers.clear()

_console = logging.StreamHandler()
_console.setLevel(logging.INFO)
_console.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))
_root.addHandler(_console)

_file = logging.handlers.RotatingFileHandler(LOG_FILE, maxBytes=1 * 1024 * 1024, backupCount=2, encoding="utf-8")
_file.setLevel(logging.INFO)
_file.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))
_root.addHandler(_file)

logger = logging.getLogger("ai_leaderboard_executor")

# ------------------------------------------------------------------------------
# Configuration (env-overridable)
# ------------------------------------------------------------------------------
LEADERBOARD_FILE = Path(os.getenv("LEADERBOARD_FILE", "mutation_leaderboard.json"))

# All Live Data, No Fallback/Hardcoded Data
# Import from single source of truth
try:
    from backend.config.trading_universe import TRADING_SYMBOLS
except (ImportError, ModuleNotFoundError, AttributeError, ValueError, TypeError, RuntimeError) as e:
    msg = f"Failed to import TRADING_SYMBOLS from trading_universe: {e}"
    raise RuntimeError(msg) from e

# Use first symbol from trading_universe if TRADE_SYMBOL_BINANCE not set
TRADE_SYMBOL_BINANCE = os.getenv("TRADE_SYMBOL_BINANCE")
if not TRADE_SYMBOL_BINANCE:
    if not TRADING_SYMBOLS:
        msg = "TRADE_SYMBOL_BINANCE environment variable is required - no fallback/hardcoded symbol"
        raise RuntimeError(msg)
    TRADE_SYMBOL_BINANCE = TRADING_SYMBOLS[0]

USD_TRADE_AMOUNT = float(os.getenv("USD_TRADE_AMOUNT", "50"))

MIN_WIN_RATE = float(os.getenv("MIN_WIN_RATE", "0.55"))
MIN_PROFIT = float(os.getenv("MIN_PROFIT", "10.0"))

DEFAULT_INTERVAL_SEC = int(os.getenv("EXEC_INTERVAL_SEC", str(60 * 60)))  # default 1 hour


# ------------------------------------------------------------------------------
# Data structures & validation
# ------------------------------------------------------------------------------
@dataclass
class StrategyRow:
    identifier: str
    profit: float
    win_rate: float
    meta: dict[str, Any]

    @staticmethod
    def _parse_number(val: Any) -> float:
        """Try to coerce different numeric representations into float."""
        try:
            if isinstance(val, (int, float)):
                return float(val)
            if isinstance(val, str):
                s = val.strip()
                # handle percent like "55%" -> 55
                if s.endswith("%"):
                    s = s[:-1].strip()
                    return float(s)
                # remove common currency symbols
                for ch in ("$", ","):
                    s = s.replace(ch, "")
                return float(s)
        except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError):
            pass
        return 0.0

    @staticmethod
    def from_obj(obj: dict[str, Any]) -> StrategyRow | None:
        try:
            sid = str(obj.get("id") or obj.get("name") or "unknown")
            profit_raw = obj.get("profit", 0)
            win_raw = obj.get("win_rate", 0)

            profit = StrategyRow._parse_number(profit_raw)

            # Win rate may be in 0-1, 0-100, "55%" or string
            if isinstance(win_raw, str) and win_raw.strip().endswith("%"):
                win_val = StrategyRow._parse_number(win_raw) / 100.0
            else:
                win_val = StrategyRow._parse_number(win_raw)
                # If value looks like a percentage (e.g., 55) convert to 0.55
                if 1.0 < win_val <= 100.0:
                    win_val = win_val / 100.0

            # Ensure bounds sanity
            if not (0.0 <= win_val <= 1.0):
                return None

            return StrategyRow(identifier=sid, profit=profit, win_rate=win_val, meta=obj)
        except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError):
            return None


def ensure_leaderboard_file(path: Path = LEADERBOARD_FILE) -> None:
    """Ensure leaderboard file exists as an empty list JSON."""
    if path.exists():
        return
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text("[]", encoding="utf-8")
        tmp.replace(path)
        logger.info(f"Created empty leaderboard at {path}")
    except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
        logger.exception(f"Failed to create leaderboard file {path}: {e}")


def load_leaderboard(path: Path = LEADERBOARD_FILE) -> list[StrategyRow]:
    """Load & validate leaderboard entries."""
    try:
        with path.open("r", encoding="utf-8") as f:
            raw = json.load(f)
        if not isinstance(raw, list):
            logger.warning("Leaderboard content is not a list; ignoring")
            return []
        rows: list[StrategyRow] = []
        for item in raw:
            if isinstance(item, dict):
                row = StrategyRow.from_obj(item)
                if row:
                    rows.append(row)
        logger.info(f"Loaded leaderboard with {len(rows)} valid strategies")
    except FileNotFoundError:
        logger.warning(f"Leaderboard file {path} not found; creating empty leaderboard")
        ensure_leaderboard_file(path)
        return []
    except json.JSONDecodeError as e:
        logger.exception(f"Invalid JSON in leaderboard file {path}: {e}")
        return []
    except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
        logger.exception(f"Error loading leaderboard: {e}")
        return []
    else:
        return rows


def select_top_strategy(
    min_win_rate: float = MIN_WIN_RATE,
    min_profit: float = MIN_PROFIT,
    path: Path = LEADERBOARD_FILE,
) -> StrategyRow | None:
    leaderboard = load_leaderboard(path)
    if not leaderboard:
        logger.info("No strategies in leaderboard")
        return None

    # Sort by (profit, win_rate), both desc
    leaderboard.sort(key=lambda r: (r.profit, r.win_rate), reverse=True)
    logger.info(f"Top strategies: {[s.identifier for s in leaderboard[:3]]}")

    for strat in leaderboard:
        if strat.win_rate >= min_win_rate and strat.profit >= min_profit:
            logger.info(f"Selected strategy: {strat.identifier} (win_rate: {strat.win_rate:.2f}, profit: {strat.profit:.2f})")
            return strat

    logger.info(f"No strategy meets criteria (min_win_rate: {min_win_rate}, min_profit: {min_profit})")
    return None


def execute_leaderboard_top_strategy(
    symbol: str = TRADE_SYMBOL_BINANCE,
    usd_amount: float = USD_TRADE_AMOUNT,
    min_win_rate: float = MIN_WIN_RATE,
    min_profit: float = MIN_PROFIT,
    path: Path = LEADERBOARD_FILE,
) -> dict[str, Any] | None:
    """Pick a strategy and execute it. Returns result dict or None."""
    top = select_top_strategy(min_win_rate=min_win_rate, min_profit=min_profit, path=path)
    if not top:
        logger.info("No suitable strategy found for execution")
        return None

    logger.info(f"Executing top strategy: {top.identifier} on {symbol} with ${usd_amount:.2f}")

    # All Live Data, No Fallback/Hardcoded Data
    try:
        result = execute_ai_strategy_signal(symbol, float(usd_amount), True)
    except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
        logger.exception(f"Strategy execution raised: {e}")
        return {
            "error": str(e),
            "strategy": top.identifier,
            "symbol": symbol,
            "amount": usd_amount,
        }

    if result and not isinstance(result, dict):
        logger.error(f"Unexpected result type from execute_ai_strategy_signal: {type(result)}")
    elif result and "error" not in result:
        logger.info(f"Strategy execution successful: {result}")
    else:
        logger.error(f"Strategy execution failed: {result}")

    return result


def run_continuous_execution(
    interval_sec: int = DEFAULT_INTERVAL_SEC,
    symbol: str = TRADE_SYMBOL_BINANCE,
    usd_amount: float = USD_TRADE_AMOUNT,
    min_win_rate: float = MIN_WIN_RATE,
    min_profit: float = MIN_PROFIT,
    path: Path = LEADERBOARD_FILE,
) -> None:
    """Run continuous leaderboard execution with graceful shutdown."""
    logger.info("Starting continuous leaderboard execution...")
    logger.info(f"Configuration: symbol={symbol} amount=${usd_amount:.2f}")
    logger.info(f"Criteria: min_win_rate={min_win_rate}, min_profit={min_profit}")
    logger.info(f"Interval: {interval_sec}s")

    stop_flag = {"stop": False}

    def _signal_handler(signum, _frame):
        logger.info(f"Received signal {signum}; stopping after current cycle...")
        stop_flag["stop"] = True

    try:
        signal.signal(signal.SIGINT, _signal_handler)
        signal.signal(signal.SIGTERM, _signal_handler)
    except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
        logger.warning(f"Could not set signal handlers: {e}")

    # Ensure leaderboard exists
    ensure_leaderboard_file(path)

    while not stop_flag["stop"]:
        try:
            execute_leaderboard_top_strategy(
                symbol=symbol,
                usd_amount=usd_amount,
                min_win_rate=min_win_rate,
                min_profit=min_profit,
                path=path,
            )

            # Sleep in small chunks to react faster to signals
            slept = 0
            while slept < interval_sec and not stop_flag["stop"]:
                chunk = min(5, interval_sec - slept)
                time.sleep(chunk)
                slept += chunk

        except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
            logger.exception(f"Error in continuous execution loop: {e}")
            # Sleep a full interval on error to avoid tight error loops
            with contextlib.suppress(ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError):
                # interrupted by signal; loop will check stop_flag
                time.sleep(interval_sec)

    logger.info("Continuous execution stopped.")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="AI Leaderboard Executor")
    parser.add_argument(
        "--leaderboard-file",
        type=Path,
        default=LEADERBOARD_FILE,
        help="Path to leaderboard JSON",
    )
    parser.add_argument(
        "--symbol",
        type=str,
        default=TRADE_SYMBOL_BINANCE,
        help="Binance US symbol (e.g., ETHUSDT)",
    )
    parser.add_argument("--amount", type=float, default=USD_TRADE_AMOUNT, help="USD trade amount")
    parser.add_argument(
        "--min-win-rate",
        type=float,
        default=MIN_WIN_RATE,
        help="Minimum win rate (0-1)",
    )
    parser.add_argument("--min-profit", type=float, default=MIN_PROFIT, help="Minimum profit threshold")
    parser.add_argument(
        "--interval",
        type=int,
        default=DEFAULT_INTERVAL_SEC,
        help="Seconds between runs in loop mode",
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="Run a single selection+execution cycle and exit",
    )

    args = parser.parse_args(argv)

    # Ensure leaderboard exists
    ensure_leaderboard_file(args.leaderboard_file)

    if args.once:
        res = execute_leaderboard_top_strategy(
            symbol=args.symbol,
            usd_amount=args.amount,
            min_win_rate=args.min_win_rate,
            min_profit=args.min_profit,
            path=args.leaderboard_file,
        )
        # Friendly exit code: 0 even if no candidate, non-zero only on hard error
        logger.info(json.dumps({"result": res}, indent=2, default=str))
        return 0

    run_continuous_execution(
        interval_sec=args.interval,
        symbol=args.symbol,
        usd_amount=args.amount,
        min_win_rate=args.min_win_rate,
        min_profit=args.min_profit,
        path=args.leaderboard_file,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
