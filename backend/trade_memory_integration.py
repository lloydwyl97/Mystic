"""
Integration module for trade logging and strategy memory engine.
Hooks into the execution system to automatically log trades and track strategy performance.
"""

from __future__ import annotations

import contextlib
import datetime
import logging
from datetime import timezone
from typing import Any

from db_logger import (
    get_strategy_id,
    get_strategy_stats,
    log_trade,
    register_strategy,
    update_trade_exit,
)
from mutator import run_evolution_cycle
from reward_engine import evaluate_strategies

from alerts import alert_evolution_cycle, alert_strategy_mutation, alert_trade_execution

logger = logging.getLogger(__name__)


def _now_utc() -> datetime.datetime:
    return datetime.datetime.now(timezone.utc)


def _coerce_trade_id(result: Any) -> int | None:
    """
    Try to extract a DB trade id from various possible return shapes of `log_trade`:
      - int -> id
      - str of int -> id
      - dict -> result["id"] or result["trade_id"]
      - tuple/list -> look for first int-ish item
      - bool/None -> no id
    """
    try:
        if result is None:
            return None
        if isinstance(result, int):
            return result
        if isinstance(result, str) and result.isdigit():
            return int(result)
        if isinstance(result, dict):
            for k in ("id", "trade_id", "tradeId"):
                if k in result:
                    v = result[k]
                    if isinstance(v, int):
                        return v
                    if isinstance(v, str) and v.isdigit():
                        return int(v)
        if isinstance(result, (tuple, list)):
            for item in result:
                if isinstance(item, int):
                    return item
                if isinstance(item, str) and item.isdigit():
                    return int(item)
        # If it's a bare True/False, there is no id
    except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
        logger.warning(f"Could not coerce trade id from log_trade result ({type(result)}): {e}")
        return None
    else:
        return None


class TradeMemoryIntegration:
    """
    Hooks into the execution engine to log trades & manage strategy evolution.
    """

    def __init__(self) -> None:
        # Track open trades by DB id (preferred) or a local fallback id
        self.active_trades: dict[int, dict[str, Any]] = {}
        self.strategy_cache: dict[str, int] = {}  # strategy_name -> strategy_id
        self.evaluation_interval = 100  # Evaluate strategies every N trades
        self.evolution_interval = 500  # Run evolution every N trades
        self.trade_counter = 0

        # Local id fallback counter (used only if log_trade doesn't return an id)
        self._local_id_seq = 1
        self._local_to_db: dict[int, int] = {}  # map local->db id when we later learn it (defensive)

        self._initialize_default_strategies()

    # ---------- initialization ----------

    def _initialize_default_strategies(self) -> None:
        """Initialize default strategies if they don't already exist."""
        default_strategies = [
            ("Breakout_EMA", "EMA crossover with breakout detection"),
            ("RSI_Dip", "RSI oversold with volume confirmation"),
            ("MACD_Signal", "MACD signal line crossover strategy"),
            ("Bollinger_Bands", "Bollinger Bands mean reversion"),
            ("Volume_Spike", "Volume spike breakout strategy"),
            ("Trend_Following", "Trend following with momentum"),
            ("Mean_Reversion", "Mean reversion with support/resistance"),
            ("Volatility_Breakout", "Volatility breakout strategy"),
        ]
        for name, description in default_strategies:
            try:
                sid = get_strategy_id(name)
                if not sid:
                    sid = register_strategy(name, description)
                if sid:
                    self.strategy_cache[name] = sid
                    logger.info(f"Strategy ready: {name} (ID: {sid})")
                else:
                    logger.warning(f"Strategy setup failed (no ID): {name}")
            except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
                logger.warning(f"Error initializing strategy '{name}': {e}")

    # ---------- public API ----------

    def log_trade_entry(
        self,
        coin: str,
        strategy_name: str,
        entry_price: float,
        quantity: float = 1.0,
        entry_reason: str = "",
        trade_type: str = "spot",
        risk_level: str = "medium",
        tags: str = "",
    ) -> int | None:
        """
        Log a trade entry to the DB and the in-memory ledger.

        Returns:
            int: Trade ID (DB id if available; otherwise local fallback id), or None on failure.
        """
        try:
            # Resolve strategy id
            sid = self.strategy_cache.get(strategy_name)
            if not sid:
                sid = get_strategy_id(strategy_name)
                if not sid:
                    sid = register_strategy(strategy_name, f"Auto-registered strategy: {strategy_name}")
                if sid:
                    self.strategy_cache[strategy_name] = sid
            if not sid:
                logger.error(f"Could not get strategy ID for: {strategy_name}")
                return None

            # Persist entry
            result = log_trade(
                coin=coin,
                strategy_id=sid,
                entry_price=entry_price,
                exit_price=None,  # filled later
                quantity=quantity,
                duration_minutes=None,  # computed on exit
                entry_reason=entry_reason,
                exit_reason="",
                trade_type=trade_type,
                risk_level=risk_level,
                tags=tags,
            )

            db_id = _coerce_trade_id(result)

            # Choose the id we expose to the caller
            if db_id is not None:
                trade_id = db_id
            else:
                # Fallback to a local id to not break callers; warn loudly.
                trade_id = self._local_id_seq
                self._local_id_seq += 1
                logger.warning(
                    "log_trade did not return a DB trade id; using local id %s. update_trade_exit may require a DB id.",
                    trade_id,
                )

            # Track open trade
            self.active_trades[trade_id] = {
                "coin": coin,
                "strategy_name": strategy_name,
                "entry_price": float(entry_price),
                "quantity": float(quantity),
                "entry_time": _now_utc(),
                "db_id": db_id,  # None if unknown
            }

            self.trade_counter += 1
            logger.info(f"Logged trade ENTRY: {coin} | Strategy: {strategy_name} | Entry: {entry_price} | id={trade_id}")

            self._check_evaluation_triggers()
        except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
            logger.exception(f"Failed to log trade entry: {e}")
            return None
        else:
            return trade_id

    def log_trade_exit(self, trade_id: int, exit_price: float, exit_reason: str = "") -> bool:
        """
        Log a trade exit. Attempts to use the DB trade id if known; otherwise
        uses given id (for compatibility with environments where log_trade returns the id).
        """
        try:
            info = self.active_trades.get(trade_id)
            if not info:
                logger.error(f"Trade ID {trade_id} not found in active trades")
                return False

            entry_time: datetime.datetime = info["entry_time"]
            duration_minutes = (_now_utc() - entry_time).total_seconds() / 60.0

            # Prefer DB id if we have it
            db_id = info.get("db_id") or self._local_to_db.get(trade_id) or trade_id

            success = update_trade_exit(db_id, float(exit_price), exit_reason)
            if not success:
                logger.error(f"DB exit update failed for trade_id={db_id}")
                return False

            # Compute profit for alert
            profit = (float(exit_price) - float(info["entry_price"])) * float(info["quantity"])
            success_bool = profit > 0

            # Send alert
            alert_trade_execution(
                {
                    "coin": info["coin"],
                    "strategy_name": info["strategy_name"],
                    "entry_price": info["entry_price"],
                    "exit_price": float(exit_price),
                    "profit": profit,
                    "success": success_bool,
                    "duration_minutes": duration_minutes,
                }
            )

            # Remove from in-memory ledger
            with contextlib.suppress(KeyError):
                # Already removed? continue gracefully.
                del self.active_trades[trade_id]

            logger.info(
                "Logged trade EXIT: %s | PnL: %.4f | success=%s | duration=%.2f min | id=%s",
                info["coin"],
                profit,
                success_bool,
                duration_minutes,
                db_id,
            )
        except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
            logger.exception(f"Failed to log trade exit: {e}")
            return False
        else:
            return True

    def get_strategy_performance(self, strategy_name: str, days: int = 30) -> dict[str, Any]:
        """Get performance statistics for a strategy."""
        try:
            sid = self.strategy_cache.get(strategy_name) or get_strategy_id(strategy_name)
            if sid:
                self.strategy_cache[strategy_name] = sid
            if not sid:
                return {
                    "error": f"Strategy {strategy_name} not found",
                    "total_trades": 0,
                    "win_rate": 0.0,
                    "avg_profit": 0.0,
                    "total_profit": 0.0,
                }

            stats = get_strategy_stats(sid, days=days)
            if not stats:
                return {
                    "error": f"No stats available for strategy ID {sid}",
                }
            stats["strategy_name"] = strategy_name
        except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
            logger.exception(f"Error getting strategy performance: {e}")
            return {"error": str(e)}
        else:
            return stats

    def get_active_trades(self) -> list[dict[str, Any]]:
        """Get list of currently active trades with durations (minutes)."""
        now = _now_utc()
        return [
            {
                "trade_id": trade_id,
                "coin": info["coin"],
                "strategy_name": info["strategy_name"],
                "entry_price": info["entry_price"],
                "quantity": info["quantity"],
                "entry_time": info["entry_time"].isoformat(),
                "duration_minutes": (now - info["entry_time"]).total_seconds() / 60.0,
            }
            for trade_id, info in self.active_trades.items()
        ]

    def force_evaluation(self) -> dict[str, Any]:
        """Force run strategy evaluation."""
        try:
            logger.info("Forcing strategy evaluation...")
            return evaluate_strategies(min_trades=1, days=1)
        except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
            logger.exception(f"Error in forced evaluation: {e}")
            return {"error": str(e)}

    def force_evolution(self) -> dict[str, Any]:
        """Force run evolution cycle and alerts."""
        try:
            logger.info("Forcing evolution cycle...")
            results = run_evolution_cycle()
            alert_evolution_cycle(results)
            # Send alerts for any new mutations
            for detail in (results or {}).get("details", []):
                if detail.get("type") == "mutation":
                    alert_strategy_mutation(detail.get("info", {}))
        except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
            logger.exception(f"Error in forced evolution: {e}")
            return {"error": str(e)}
        else:
            return results

    # ---------- internal ----------

    def _check_evaluation_triggers(self) -> None:
        """Check if we should run evaluation or evolution cycles."""
        try:
            if self.trade_counter > 0 and self.trade_counter % self.evaluation_interval == 0:
                logger.info(f"Running strategy evaluation (trade #{self.trade_counter})")
                evaluation_results = evaluate_strategies(min_trades=3, days=7) or {}
                logger.info(
                    "Evaluation completed: %s strategies updated",
                    evaluation_results.get("updated_strategies", 0),
                )

            if self.trade_counter > 0 and self.trade_counter % self.evolution_interval == 0:
                logger.info(f"Running evolution cycle (trade #{self.trade_counter})")
                evolution_results = run_evolution_cycle() or {}
                alert_evolution_cycle(evolution_results)
                for detail in evolution_results.get("details", []):
                    if detail.get("type") == "mutation":
                        alert_strategy_mutation(detail.get("info", {}))
                logger.info(
                    "Evolution completed: %s new strategies created",
                    evolution_results.get("total_new_strategies", 0),
                )

        except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
            logger.exception(f"Error in evaluation triggers: {e}")


# Global instance
trade_memory = TradeMemoryIntegration()


# Convenience functions
def log_trade_entry(coin: str, strategy_name: str, entry_price: float, **kwargs) -> int | None:
    return trade_memory.log_trade_entry(coin, strategy_name, entry_price, **kwargs)


def log_trade_exit(trade_id: int, exit_price: float, exit_reason: str = "") -> bool:
    return trade_memory.log_trade_exit(trade_id, exit_price, exit_reason)


def get_strategy_performance(strategy_name: str, days: int = 30) -> dict[str, Any]:
    return trade_memory.get_strategy_performance(strategy_name, days)


def get_active_trades() -> list[dict[str, Any]]:
    return trade_memory.get_active_trades()


def force_evaluation() -> dict[str, Any]:
    return trade_memory.force_evaluation()


def force_evolution() -> dict[str, Any]:
    return trade_memory.force_evolution()
