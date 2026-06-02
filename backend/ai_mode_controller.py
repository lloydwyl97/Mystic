"""
AI Trading Controller (Repaired & Hardened)

Key fixes & improvements:
- Clear drawdown logic using equity/peak tracking (instead of comparing a raw cumulative profit to a negative threshold)
- Safer mode switching (accepts str or AIMode; validates input)
- Structured logger (no global basicConfig side effects)
- Daily trade limit enforcement with explicit reset
- Rich status reporting, including equity, peak, and current drawdown
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)
# Avoid imposing global logging configuration on consumers of this module.
logger.addHandler(logging.NullHandler())


class AIMode(str, Enum):
    TRAINING = "training"
    LIVE = "live"
    OFF = "off"


@dataclass
class AITradingController:
    """
    Controls whether the AI should place trades based on mode, limits, and risk.

    Parameters
    ----------
    starting_equity : float
        Initial account equity used to track drawdowns.
    daily_trade_limit : int
        Max number of live trades allowed per (manually managed) day.
    max_total_drawdown : float
        Maximum allowed peak-to-trough equity drawdown (fraction, e.g. 0.10 = 10%).
    mode : AIMode
        Operating mode: TRAINING (no live trades), LIVE (enabled), OFF (disabled).
    """

    starting_equity: float = 10_000.0
    daily_trade_limit: int = 20
    max_total_drawdown: float = 0.10
    mode: AIMode = AIMode.TRAINING

    # Runtime state (auto-managed)
    trade_counter: int = 0
    equity: float | None = field(default=None)  # initialized in __post_init__
    peak_equity: float | None = field(default=None)
    total_pnl: float = 0.0  # cumulative P&L in currency since controller creation/reset
    current_day: date = field(default_factory=date.today)

    def __post_init__(self) -> None:
        if self.daily_trade_limit < 0:
            logger.warning("daily_trade_limit < 0 provided; clamping to 0")
            self.daily_trade_limit = 0
        if not (0 <= self.max_total_drawdown < 1):
            msg = "max_total_drawdown must be in [0, 1)."
            raise ValueError(msg)

        # Initialize equity/peak if not provided
        self.equity = self.starting_equity if self.equity is None else self.equity
        self.peak_equity = self.equity if self.peak_equity is None else self.peak_equity

    # -------------------------
    # Mode management
    # -------------------------
    def set_mode(self, mode: str | AIMode) -> None:
        """
        Set the AI operating mode.
        Accepts either AIMode or case-insensitive strings: "training", "live", "off".
        """
        if isinstance(mode, AIMode):
            self.mode = mode
            logger.info(f"AI mode switched to {self.mode.value}")
            return

        try:
            mode_clean = str(mode).strip().lower()
            self.mode = AIMode(mode_clean)  # validates
            logger.info(f"AI mode switched to {self.mode.value}")
        except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
            msg = "Invalid mode. Must be 'training', 'live', or 'off'"
            raise ValueError(msg) from e

    # -------------------------
    # Risk / limit checks
    # -------------------------
    @property
    def current_drawdown(self) -> float:
        """
        Current peak-to-trough drawdown as a fraction in [0, 1].
        Drawdown is (peak_equity - equity) / peak_equity, using tracked equity.
        """
        if self.peak_equity is None or self.peak_equity <= 0:
            return 0.0
        # equity should not be None here due to initialization, but guard anyway
        equity = self.equity if self.equity is not None else 0.0
        dd = (self.peak_equity - equity) / self.peak_equity
        return max(0.0, dd)

    def _enforce_daily_rollover(self) -> None:
        """
        Rollover helper: if the date changed, reset counters for the new day.
        Call this at the beginning of should_execute_trade if your process runs across days.
        """
        today = datetime.now(timezone.utc).date()
        if today != self.current_day:
            self.reset_daily_counter()
            self.current_day = today
            logger.info("New trading day detected; daily counters reset.")

    def should_execute_trade(self, _simulated_profit: float | None = None) -> bool:
        """
        Decide if a live trade should be placed *right now*.
        Note: `simulated_profit` is ignored here; keep it for API compatibility.

        Returns
        -------
        bool
            True if trading is allowed given mode, limits, and drawdown. False otherwise.
        """
        self._enforce_daily_rollover()

        if self.mode == AIMode.OFF:
            logger.debug("Trading blocked: mode=OFF")
            return False

        if self.mode == AIMode.TRAINING:
            logger.debug("Training mode: not executing live trades (simulation only).")
            return False

        # Limit checks (LIVE mode)
        if self.trade_counter >= self.daily_trade_limit:
            logger.warning("Trade blocked: daily trade limit reached.")
            return False

        if self.current_drawdown >= self.max_total_drawdown:
            logger.warning(f"Trade blocked: max total drawdown reached (dd={self.current_drawdown:.2%}, limit={self.max_total_drawdown:.2%})")
            return False

        return True

    # -------------------------
    # State updates
    # -------------------------
    def record_trade(self, profit: float) -> None:
        """
        Record a completed trade.

        Parameters
        ----------
        profit : float
            Trade P&L in account currency (e.g., USDT/USD). Positive for gains, negative for losses.
        """
        self.trade_counter += 1
        self.total_pnl += profit
        # ensure equity is initialized
        if self.equity is None:
            self.equity = self.starting_equity
        self.equity += profit
        # Update peak equity after profitable moves
        if self.peak_equity is None:
            self.peak_equity = self.equity
        else:
            self.peak_equity = max(self.peak_equity, self.equity)

        logger.info(
            "Trade recorded",
            extra={
                "extra_fields": {
                    "profit": profit,
                    "equity": self.equity,
                    "peak_equity": self.peak_equity,
                    "drawdown": round(self.current_drawdown, 6),
                    "count": self.trade_counter,
                    "total_pnl": self.total_pnl,
                },
            },
        )

    def reset_daily_counter(self) -> None:
        """Reset the daily trade counter."""
        self.trade_counter = 0
        logger.info("Daily trade counter reset")

    def reset_equity(self, new_starting_equity: float | None = None) -> None:
        """
        Reset equity and drawdown tracking (e.g., at the start of a new tracking period).
        """
        if new_starting_equity is not None:
            self.starting_equity = float(new_starting_equity)
        self.equity = self.starting_equity
        self.peak_equity = self.starting_equity
        self.total_pnl = 0.0
        logger.info(
            "Equity & drawdown tracking reset",
            extra={"extra_fields": {"starting_equity": self.starting_equity}},
        )

    # -------------------------
    # Status
    # -------------------------
    def get_status(self) -> dict[str, Any]:
        """
        Current controller status snapshot suitable for logging/serialization.
        """
        return {
            "mode": self.mode.value,
            "daily_trades": self.trade_counter,
            "daily_limit": self.daily_trade_limit,
            "equity": self.equity,
            "peak_equity": self.peak_equity,
            "current_drawdown": self.current_drawdown,  # fraction (0..1)
            "max_total_drawdown": self.max_total_drawdown,  # fraction (0..1)
            "total_pnl": self.total_pnl,
            "starting_equity": self.starting_equity,
            "current_day": self.current_day.isoformat(),
        }
