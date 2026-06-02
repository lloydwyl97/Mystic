"""
Mutation Trainer - Fast Strategy Evolution System
Continuously trains and evolves trading strategies using backtesting.
Built for Windows 11 Home + PowerShell (no Docker required).

Notes:
- Data source: Binance.US ONLY (no Coinbase/Coingecko/Kraken/fin*).
- Windows-safe paths (no hardcoded /tmp or /data).
- Robust to missing 'signal' column from strategies.
- Simple, reproducible P&L + win rate logic.
"""

from __future__ import annotations

import asyncio
import importlib.util
import json
import logging
import os
import random
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# Direct imports for production
import anyio
import pandas as pd

from backend.services.canonical_http_client import get_json

# Import from single source of truth
try:
    from backend.config.trading_universe import TRADING_SYMBOLS
except (ImportError, ModuleNotFoundError, AttributeError, ValueError, TypeError, RuntimeError) as e:
    msg = f"Failed to import trading_universe: {e}"
    raise RuntimeError(msg) from e

# ---------- logging ----------
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("mutation_trainer")

# ---------- config (env-overridable) ----------
BASE_DIR = Path(__file__).resolve().parent

STRATEGY_DIR = Path(os.getenv("STRATEGY_DIR", str(BASE_DIR / "generated_modules")))
RESULTS_DIR = Path(os.getenv("RESULTS_DIR", str(BASE_DIR / "data" / "backtest_results")))
SUMMARY_PATH = Path(os.getenv("TRAINING_SUMMARY_PATH", str(BASE_DIR / "data" / "training_summary.json")))
PING_FILE = Path(
    os.getenv(
        "TRAINER_PING_FILE",
        str(Path(tempfile.gettempdir()) / "train_fast_mutations.ping"),
    )
)
LOCK_FILE = Path(
    os.getenv(
        "TRAINER_LOCK_FILE",
        str(Path(tempfile.gettempdir()) / "train_fast_mutations.lock"),
    )
)

TRAIN_INTERVAL = int(os.getenv("TRAIN_INTERVAL_SECONDS", "1800"))  # 30 min default
MAX_STRATEGIES = int(os.getenv("MAX_STRATEGIES", "50"))

# Training timing constants
KLINES_FETCH_BACKOFF_MAX = 8.0  # Maximum backoff for klines fetch retries
PARALLEL_TRAIN_STAGGER_DELAY = 0.5  # Delay between parallel training starts
TRAINING_ERROR_RECOVERY_DELAY = 60.0  # Delay after training errors before retry

# Binance.US only - Use TRADING_SYMBOLS from trading_universe (live data)
# Default featured symbols from trading_universe (first two symbols)
DEFAULT_FEATURED = ",".join(TRADING_SYMBOLS[:2]) if len(TRADING_SYMBOLS) >= 2 else (TRADING_SYMBOLS[0] if TRADING_SYMBOLS else "BTCUSDT")
FEATURED_SYMBOLS: list[str] = [s.strip() for s in os.getenv("FEATURED_SYMBOLS", DEFAULT_FEATURED).replace(" ", ",").split(",") if s.strip()]
SYMBOL = (FEATURED_SYMBOLS[0] if FEATURED_SYMBOLS else (TRADING_SYMBOLS[0] if TRADING_SYMBOLS else "BTCUSDT")).upper()
INTERVAL = os.getenv("BINANCE_KLINE_INTERVAL", "1h")
LIMIT = int(os.getenv("BINANCE_KLINE_LIMIT", "500"))
# Use TRADING_SYMBOLS from trading_universe (live data)
APPROVED_TOP10 = set(TRADING_SYMBOLS)


def _build_klines_url(symbol: str, interval: str, limit: int) -> str:
    return f"https://api.binance.us/api/v3/klines?symbol={symbol}&interval={interval}&limit={limit}"


REQUEST_TIMEOUT = float(os.getenv("REQUEST_TIMEOUT", "10"))

# ---------- filesystem setup ----------
for p in (STRATEGY_DIR, RESULTS_DIR, SUMMARY_PATH.parent):
    p.mkdir(parents=True, exist_ok=True)


def list_strategies() -> list[str]:
    """List available strategy .py files in STRATEGY_DIR (excluding non-modules)."""
    try:
        if not STRATEGY_DIR.exists():
            STRATEGY_DIR.mkdir(parents=True, exist_ok=True)
            return []
        files = []
        for f in STRATEGY_DIR.iterdir():
            if f.suffix == ".py" and f.name not in {"__init__.py"}:
                files.append(f.name)
    except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
        logger.exception(f"Error listing strategies: {e}")
        return []
    else:
        return files


def _load_strategy_module(strategy_file: str):
    """Dynamically import a strategy module from file."""
    try:
        strategy_path = STRATEGY_DIR / strategy_file
        module_name = f"strategy_mod_{Path(strategy_file).stem}_{int(time.time() * 1000)}"
        spec = importlib.util.spec_from_file_location(module_name, strategy_path)
        if spec is None or spec.loader is None:
            msg = f"Invalid spec for {strategy_file}"
            raise ImportError(msg)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
    except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
        msg = f"Failed to import {strategy_file}: {e}"
        raise ImportError(msg) from e


def _fetch_klines_df() -> pd.DataFrame:
    """Fetch klines from Binance.US and return a pandas DataFrame with a 'close' column."""
    headers = {"User-Agent": "mutation-trainer/1.0"}

    url = _build_klines_url(SYMBOL, INTERVAL, LIMIT)

    async def _fetch_once() -> Any:
        return await get_json(url, headers=headers, timeout=REQUEST_TIMEOUT)

    # Avoid nested event loops
    loop = None
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop is not None and loop.is_running():  # pragma: no cover
        msg = "_fetch_klines_df() must be called from sync context (no running event loop)"
        raise RuntimeError(msg)

    attempts = 0
    last_err: Exception | None = None
    while attempts < 3:
        try:
            klines = anyio.run(_fetch_once)
        except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
            last_err = e
            attempts += 1
            if attempts < 3:
                time.sleep(1)  # Fixed: Use time.sleep instead of await in sync function
                continue
            raise

        if not isinstance(klines, list) or len(klines) == 0:
            msg = "Empty or invalid klines payload from Binance.US"
            raise ValueError(msg)
        if not all(isinstance(k, (list, tuple)) and len(k) >= 6 for k in klines):
            msg = "Unexpected klines row format"
            raise ValueError(msg)
        try:
            closes = [float(k[4]) for k in klines]
            ts = [int(k[0]) for k in klines]
            df = pd.DataFrame({"close": closes, "ts": pd.to_datetime(ts, unit="ms", utc=True)})
        except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
            last_err = e
            attempts += 1
            if attempts < 3:
                time.sleep(1)  # Fixed: Use time.sleep instead of await in sync function
                continue
            raise

        if df.empty:
            msg = "Klines DataFrame is empty"
            raise ValueError(msg)
        return df

    msg = f"Failed to fetch klines after retries: {last_err}"
    raise RuntimeError(msg)


def _ensure_signal(df: pd.DataFrame, strategy_mod: Any) -> pd.DataFrame:
    """
    Ensure a 'signal' column exists:
      - If strategy module exposes `rsi_strategy(df)` or `generate_signals(df)`, use it.
      - Otherwise, provide a safe, flat neutral signal (no trades).
    Expected signal semantics: {1=long, 0=flat, -1=short}. We only use 1/0 for simple long-only backtest.
    """
    try:
        if hasattr(strategy_mod, "rsi_strategy"):
            out = strategy_mod.rsi_strategy(df.copy())
            # if out is a DataFrame with 'signal' column
            if hasattr(out, "columns") and "signal" in out.columns:
                return out
            # if out is a Series that looks like signal
            if isinstance(out, pd.Series) and (out.name == "signal" or len(out) == len(df)):
                df2 = df.copy()
                df2["signal"] = out.to_numpy()
                return df2
        if hasattr(strategy_mod, "generate_signals"):
            out = strategy_mod.generate_signals(df.copy())
            if hasattr(out, "columns") and "signal" in out.columns:
                return out
            if isinstance(out, pd.Series) and (out.name == "signal" or len(out) == len(df)):
                df2 = df.copy()
                df2["signal"] = out.to_numpy()
                return df2
    except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
        logger.warning(f"Strategy function error, falling back to neutral signal: {e}")

    # No synthetic neutral signals in live-only mode; return unsignaled df (caller will add neutral signal)
    return df.copy()


def _pair_trades_from_signal(signal: pd.Series) -> list[tuple[int, int]]:
    """
    Convert a binary long-only signal (1/0) into (buy_idx, sell_idx) pairs.
    Buy on rising edge 0->1; sell on falling edge 1->0. Ignore incomplete last open.
    """
    s = signal.fillna(0).astype(float)
    diff = s.diff().fillna(0)

    buys = list(diff[diff > 0.5].index)  # 0 -> 1
    sells = list(diff[diff < -0.5].index)  # 1 -> 0

    pairs: list[tuple[int, int]] = []
    si = 0
    for bi in buys:
        # find first sell index after this buy
        while si < len(sells) and sells[si] <= bi:
            si += 1
        if si < len(sells):
            pairs.append((int(bi), int(sells[si])))
            si += 1
        else:
            break
    return pairs


def simulate_backtest(strategy_file: str) -> dict[str, Any]:
    """Run backtest for a strategy using Binance.US historical data."""
    # Enforce approved Binance.US symbol set
    if SYMBOL not in APPROVED_TOP10:
        msg = "unsupported_symbol"
        raise RuntimeError(msg)
    try:
        df = _fetch_klines_df()

        # Load strategy and ensure 'signal'
        strategy_mod = _load_strategy_module(strategy_file)
        df = _ensure_signal(df, strategy_mod)

        # If strategy did not produce a 'signal' column, provide neutral (flat) signal
        if "signal" not in df.columns:
            df["signal"] = 0

    except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
        logger.exception(f"Error in backtest: {e}")
        raise

    # Require columns
    if "close" not in df.columns or "signal" not in df.columns:
        msg = "Dataframe missing required columns 'close' and/or 'signal'"
        raise ValueError(msg)

    try:
        # Validate/coerce signal semantics to {1,0,-1}
        sig = df["signal"].copy()
        try:
            sig = sig.astype(float)
        except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
            msg = "Signal column not numeric and cannot be coerced"
            raise ValueError(msg) from e
        sig = sig.apply(lambda x: -1.0 if x < -0.5 else (1.0 if x > 0.5 else 0.0))
        df["signal"] = sig

        # Make sure indices align and are monotonic
        df = df.reset_index(drop=True)

        # Build trade pairs
        pairs = _pair_trades_from_signal(df["signal"])
        total_trades = len(pairs)

        # Calculate simple P&L using 1 unit per trade; record % P&L context
        total_pl_abs = 0.0
        wins = 0
        for bi, si in pairs:
            buy = float(df.loc[bi, "close"])
            sell = float(df.loc[si, "close"])
            pl = sell - buy
            total_pl_abs += pl
            if pl > 0:
                wins += 1

        winrate = (wins / total_trades) if total_trades > 0 else 0.0

        denom = sum(float(df.loc[b, "close"]) for b, _ in pairs) / max(1, len(pairs))
        result = {
            "strategy": strategy_file,
            "symbol": SYMBOL,
            "interval": INTERVAL,
            "limit": LIMIT,
            "winrate": round(winrate, 4),
            "total_trades": total_trades,
            "profit_loss": round(total_pl_abs, 6),  # absolute, 1 unit per trade
            "notional_per_trade_units": 1,
            "profit_loss_pct": round((total_pl_abs / max(1e-9, denom)) * 100.0, 4),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
    except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
        if str(e) == "unsupported_symbol":
            logger.info(f"Skipping {strategy_file}: symbol {SYMBOL} not in approved Binance.US set")
        else:
            logger.exception(f"Error backtesting {strategy_file}: {e}")
        return {
            "strategy": strategy_file,
            "symbol": SYMBOL,
            "interval": INTERVAL,
            "error": str(e),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
    else:
        logger.info(f"Backtest {strategy_file} | {SYMBOL} {INTERVAL} | trades={total_trades} winrate={winrate:.1%} PnL={total_pl_abs:.4f}")
        return result


def save_backtest_result(result: dict[str, Any]) -> None:
    """Save backtest result to RESULTS_DIR."""
    try:
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        filename = f"backtest_{Path(result.get('strategy', 'strategy')).stem}_{int(time.time())}.json"
        filepath = RESULTS_DIR / filename
        with filepath.open("w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)
        logger.info(f"Saved backtest result: {filepath.name}")
    except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
        logger.exception(f"Error saving backtest result: {e}")


def initialize_live_strategy_training() -> None:
    """Ensure strategy dir exists and has a marker file (no sample strategies created)."""
    try:
        STRATEGY_DIR.mkdir(parents=True, exist_ok=True)
        marker = STRATEGY_DIR / "LIVE_TRAINING_READY.txt"
        if not marker.exists():
            with marker.open("w", encoding="utf-8") as f:
                f.write("Live strategy training initialized\n")
                f.write("Strategies will be generated by AI mutation system\n")
                f.write(f"Initialized: {datetime.now(timezone.utc).isoformat()}\n")
        logger.info("Strategy directory ready for live AI training")
    except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
        logger.exception(f"Error initializing strategy training: {e}")


def _acquire_single_process_lock() -> bool:
    """Best-effort single process guard using a temp lock file (Windows-friendly)."""
    try:
        if LOCK_FILE.exists():
            logger.error(f"Trainer appears to be running (lock exists at {LOCK_FILE}).")
            return False
        with LOCK_FILE.open("x", encoding="utf-8") as f:
            f.write(str(os.getpid()))
    except FileExistsError:
        logger.exception(f"Trainer appears to be running (lock exists at {LOCK_FILE}).")
        return False
    except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
        logger.exception(f"Failed to create lock file {LOCK_FILE}: {e}")
        return False
    else:
        return True


def _release_single_process_lock() -> None:
    try:
        if LOCK_FILE.exists():
            LOCK_FILE.unlink()
    except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError):
        pass


def run_training_cycle() -> dict[str, Any]:
    """Run a complete training cycle over a bounded set of strategies."""
    try:
        strategies = list_strategies()
        if not strategies:
            logger.info("No strategies found for training")
            return {"trained": 0, "total": 0}

        if len(strategies) > MAX_STRATEGIES:
            strategies = random.sample(strategies, MAX_STRATEGIES)

        logger.info(f"Starting training cycle for {len(strategies)} strategies")

        results = []
        success = 0
        skipped = 0
        errors = 0
        for strategy_file in strategies:
            res = simulate_backtest(strategy_file)
            # Skip saving if error or no trades
            if res.get("error"):
                errors += 1
            elif res.get("total_trades", 0) <= 0:
                skipped += 1
            else:
                results.append(res)
                save_backtest_result(res)
                success += 1
            # Brief delay to stagger parallel training - sync sleep OK for training script
            time.sleep(PARALLEL_TRAIN_STAGGER_DELAY)

        summary = {
            "cycle_timestamp": datetime.now(timezone.utc).isoformat(),
            "strategies_trained": success,
            "strategies_skipped": skipped,
            "strategies_errors": errors,
            "results": results,
            "cycle_status": {
                "success": success,
                "skipped": skipped,
                "errors": errors,
                "processed": len(strategies),
            },
            "throughput_per_min": round((success / max(1, TRAIN_INTERVAL)) * 60.0, 2),
        }
        try:
            with SUMMARY_PATH.open("w", encoding="utf-8") as f:
                json.dump(summary, f, indent=2)
        except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
            logger.exception(f"Error saving training summary: {e}")

        logger.info(f"Training cycle complete: processed={len(strategies)} success={success} skipped={skipped} errors={errors}")
        return {
            "trained": success,
            "total": len(strategies),
            "skipped": skipped,
            "errors": errors,
        }

    except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
        logger.exception(f"Error in training cycle: {e}")
        return {"trained": 0, "total": 0, "error": str(e)}


def main():
    """Main loop for continuous strategy training."""
    logger.info(f"Mutation Trainer starting (symbol={SYMBOL}, interval={INTERVAL}, limit={LIMIT})")
    initialize_live_strategy_training()
    if not _acquire_single_process_lock():
        return

    while True:
        try:
            summary = run_training_cycle()
            # ping only if at least one backtest saved successfully
            if summary.get("trained", 0) > 0:
                try:
                    with PING_FILE.open("w", encoding="utf-8") as f:
                        f.write(str(time.time()))
                except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
                    logger.debug(f"Ping write failed: {e}")

            logger.info(f"Next training cycle in {TRAIN_INTERVAL} seconds")
            # Sleep for configured training interval - sync sleep OK for standalone script
            time.sleep(TRAIN_INTERVAL)

        except KeyboardInterrupt:
            logger.info("Mutation Trainer stopped by user")
            break
        except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
            logger.exception(f"Main loop error: {e}")
            # Error recovery delay before retry
            time.sleep(TRAINING_ERROR_RECOVERY_DELAY)
    _release_single_process_lock()


if __name__ == "__main__":
    main()
