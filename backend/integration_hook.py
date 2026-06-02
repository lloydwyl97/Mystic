"""
Simple integration hook for existing trading systems.

This file shows how to add trade logging to your existing trading bots
with minimal code changes.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Mapping
from typing import Any

from db_logger import get_strategy_stats
from trade_memory_integration import log_trade_entry, log_trade_exit

# Import from single source of truth
try:
    from backend.config.trading_universe import TOP10_COINS, TRADING_SYMBOLS
except (ImportError, ModuleNotFoundError, AttributeError, ValueError, TypeError, RuntimeError) as e:
    msg = f"Failed to import trading_universe: {e}"
    raise RuntimeError(msg) from e

logger = logging.getLogger(__name__)

# Use TOP10_COINS and TRADING_SYMBOLS from trading_universe (single source of truth)
_TOP10_COINS = tuple(TOP10_COINS)
_SUPPORTED_PAIRS = set(TRADING_SYMBOLS)


def _env_extra_pairs() -> set[str]:
    """
    Optional: comma-separated list of extra pairs allowed via env,
    e.g. MYSTIC_EXTRA_PAIRS="BNBUSDT,NEARUSDT"
    """
    raw = os.getenv("MYSTIC_EXTRA_PAIRS", "")
    if not raw:
        return set()
    return {p.strip().upper() for p in raw.split(",") if p.strip()}


def _normalize_symbol(symbol: str) -> str:
    """
    Normalize an input coin or pair to a USDT pair.
    Accepts "BTC" or "BTCUSDT" and returns "BTCUSDT".
    """
    s = (symbol or "").upper().strip()
    if not s:
        return s
    if s.endswith("USDT"):
        return s
    return f"{s}USDT"


def _validate_pair(pair: str) -> bool:
    allowed = _SUPPORTED_PAIRS | _env_extra_pairs()
    return pair in allowed


def _positive_float(name: str, value: Any) -> float | None:
    try:
        f = float(value)
        if f > 0:
            return f
    except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError):
        pass
    logger.warning("Invalid %s: %r", name, value)
    return None


class TradingHook:
    """
    Simple hook class that can be added to existing trading systems.
    Validates inputs, normalizes symbols, and delegates to trade logging.
    """

    def __init__(self) -> None:
        self.active_trades: dict[int, dict[str, Any]] = {}

    def on_trade_entry(
        self,
        coin: str,
        strategy_name: str,
        entry_price: float,
        quantity: float = 1.0,
        entry_reason: str = "",
        metadata: Mapping[str, Any] | None = None,
    ) -> int | None:
        """
        Call this when entering a trade.

        Args:
            coin: Trading symbol, e.g., "BTC" or "BTCUSDT"
            strategy_name: Name of the strategy used
            entry_price: Executed entry price (> 0)
            quantity: Trade quantity (> 0)
            entry_reason: Reason for entry
            metadata: Optional extra fields to persist

        Returns:
            trade_id on success, or None on failure
        """
        symbol = _normalize_symbol(coin)
        if not _validate_pair(symbol):
            logger.warning("Rejected trade entry for unsupported symbol: %s", symbol)
            return None

        ep = _positive_float("entry_price", entry_price)
        qty = _positive_float("quantity", quantity)
        if ep is None or qty is None:
            return None

        try:
            trade_id = log_trade_entry(
                coin=symbol,
                strategy_name=strategy_name,
                entry_price=ep,
                quantity=qty,
                entry_reason=entry_reason,
                metadata=dict(metadata or {}),
            )
            if trade_id:
                self.active_trades[trade_id] = {
                    "coin": symbol,
                    "strategy_name": strategy_name,
                    "entry_price": ep,
                    "quantity": qty,
                }
                logger.info(
                    "Trade entry logged | id=%s symbol=%s strategy=%s price=%.8f qty=%.8f",
                    trade_id,
                    symbol,
                    strategy_name,
                    ep,
                    qty,
                )
                result = trade_id
            else:
                logger.error(
                    "Trade entry logging returned no id for symbol=%s strategy=%s",
                    symbol,
                    strategy_name,
                )
                result = None
        except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
            logger.exception(
                "Failed to log trade entry | symbol=%s strategy=%s error=%s",
                symbol,
                strategy_name,
                e,
            )
            return None
        else:
            return result

    def on_trade_exit(
        self,
        trade_id: int,
        exit_price: float,
        exit_reason: str = "",
        metadata: Mapping[str, Any] | None = None,
    ) -> bool:
        """
        Call this when exiting a trade.

        Args:
            trade_id: Trade ID returned from on_trade_entry
            exit_price: Executed exit price (> 0)
            exit_reason: Reason for exit
            metadata: Optional extra fields to persist

        Returns:
            True if exit was logged, otherwise False
        """
        xp = _positive_float("exit_price", exit_price)
        if xp is None:
            return False

        try:
            success = log_trade_exit(trade_id, xp, exit_reason, metadata=dict(metadata or {}))
            if success:
                info = self.active_trades.pop(trade_id, None)
                if info:
                    pnl = (xp - float(info["entry_price"])) * float(info["quantity"])
                    logger.info(
                        "Trade exit logged | id=%s symbol=%s pnl=%.8f reason=%s",
                        trade_id,
                        info["coin"],
                        pnl,
                        exit_reason,
                    )
                else:
                    logger.info("Trade exit logged | id=%s (no local entry cache)", trade_id)
            else:
                logger.error("Trade exit failed | id=%s", trade_id)
            return bool(success)
        except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
            logger.exception("Failed to log trade exit | id=%s error=%s", trade_id, e)
            return False

    def get_strategy_performance(self, strategy_name: str) -> Mapping[str, Any] | None:
        """
        Retrieve performance stats for a strategy. Returns None on failure.
        """
        try:
            return get_strategy_stats(strategy_name)
        except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
            logger.exception(
                "Failed to get strategy performance | strategy=%s error=%s",
                strategy_name,
                e,
            )
            return None


# Global hook instance
trading_hook = TradingHook()


def simple_trade_logger(
    coin: str,
    strategy: str,
    entry_price: float,
    *,
    action: str = "entry",
    quantity: float = 1.0,
    trade_id: int | None = None,
    exit_price: float | None = None,
    reason: str = "",
    metadata: Mapping[str, Any] | None = None,
) -> int | bool | None:
    """
    Simple function for logging trades without prints or mock assumptions.

    Args:
        coin: Symbol like "BTC" or "BTCUSDT" (normalized to USDT pair)
        strategy: Strategy name
        entry_price: Entry price (required for action="entry")
        action: "entry" or "exit"
        quantity: Quantity for entry
        trade_id: Required for action="exit"
        exit_price: Required for action="exit"
        reason: Optional reason text
        metadata: Optional metadata dictionary

    Returns:
        trade_id on successful entry,
        True/False on exit,
        None on validation failure.
    """
    if action == "entry":
        return trading_hook.on_trade_entry(
            coin=coin,
            strategy_name=strategy,
            entry_price=entry_price,
            quantity=quantity,
            entry_reason=reason,
            metadata=metadata,
        )

    if action == "exit":
        if trade_id is None or exit_price is None:
            logger.warning("Exit requires trade_id and exit_price")
            return False
        return trading_hook.on_trade_exit(
            trade_id=trade_id,
            exit_price=exit_price,
            exit_reason=reason,
            metadata=metadata,
        )

    logger.warning("Unknown action: %s", action)
    return None


def log_trades(func):
    """
    Decorator to automatically log trades.
    The wrapped function must return a mapping with keys:
      - action: "entry"|"exit"
      - coin, strategy
      - entry_price (for entry)
      - quantity (optional, defaults to 1.0)
      - trade_id and exit_price (for exit)
      - reason (optional)
      - metadata (optional)
    """

    def wrapper(*args: Any, **kwargs: Any):
        result = func(*args, **kwargs)
        try:
            if isinstance(result, Mapping):
                action = str(result.get("action", "entry")).lower()
                coin = str(result.get("coin", ""))
                strategy = str(result.get("strategy", ""))
                metadata = result.get("metadata")

                if action == "entry":
                    entry_price = result.get("entry_price")
                    quantity = result.get("quantity", 1.0)
                    reason = result.get("reason", "")
                    return simple_trade_logger(
                        coin=coin,
                        strategy=strategy,
                        entry_price=float(entry_price),
                        action="entry",
                        quantity=float(quantity),
                        reason=reason,
                        metadata=metadata if isinstance(metadata, Mapping) else None,
                    )
                if action == "exit":
                    trade_id = result.get("trade_id")
                    exit_price = result.get("exit_price")
                    reason = result.get("reason", "")
                    return simple_trade_logger(
                        coin=coin,
                        strategy=strategy,
                        entry_price=0.0,  # unused for exit
                        action="exit",
                        trade_id=int(trade_id),
                        exit_price=float(exit_price),
                        reason=reason,
                        metadata=metadata if isinstance(metadata, Mapping) else None,
                    )
        except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
            logger.exception("log_trades decorator error: %s", e)
        return result

    return wrapper
