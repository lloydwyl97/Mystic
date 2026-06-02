"""
Single source of truth for Mystic trading economics.

Paper and live MUST read the same numbers from this module. There is no
paper-only profit threshold and no live-only profit threshold. The values
exposed here are the inputs to the one DAY trading brain (portfolio_engine)
and any execution adapter (paper or live) that wraps it.

Categories:
  * fee/cost model (TAKER_FEE, SLIPPAGE_BUFFER, ESTIMATED_ROUNDTRIP_COST)
  * sell threshold (MIN_NET_PROFIT_TO_SELL, MIN_PROFIT_AFTER_COSTS_USD)
  * cooldowns    (COOLDOWN_SECONDS_AFTER_SELL,
                  COOLDOWN_SECONDS_AFTER_HUMAN_SELL)

All values are env-overridable but the defaults are the canonical Mystic
DAY-only top-4 profile (BTCUSDT, ETHUSDT, SOLUSDT, XRPUSDT).
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Final

logger = logging.getLogger(__name__)


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name, "")
    if not raw:
        return float(default)
    try:
        return float(raw)
    except (TypeError, ValueError):
        logger.warning(
            "TRADING_ECONOMICS env var %s=%r is not a float; using default %s",
            name,
            raw,
            default,
        )
        return float(default)


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name, "")
    if not raw:
        return int(default)
    try:
        return int(raw)
    except (TypeError, ValueError):
        logger.warning(
            "TRADING_ECONOMICS env var %s=%r is not an int; using default %s",
            name,
            raw,
            default,
        )
        return int(default)


# -- Fee model ---------------------------------------------------------------
# Binance.US spot taker fee for one fill (fraction of notional).
BINANCE_US_TAKER_FEE_PCT: Final[float] = _env_float("BINANCE_US_TAKER_FEE_PCT", 0.0002)
# Paper/live execution taker model: defaults to Binance.US spot taker unless overridden.
TAKER_FEE: Final[float] = _env_float("TAKER_FEE", BINANCE_US_TAKER_FEE_PCT)
# One-sided slippage buffer beyond live bid/ask (fraction of notional per leg).
SLIPPAGE_BUFFER: Final[float] = _env_float("SLIPPAGE_BUFFER", 0.0001)
# Default round-trip before live spread: 2× taker + 2× one-sided slippage buffer.
_DEFAULT_ROUNDTRIP_NO_SPREAD: Final[float] = (2.0 * TAKER_FEE) + (2.0 * SLIPPAGE_BUFFER)
# Round-trip cost for the sell gate (`_check_exit_conditions`): operator sets from
# live Binance.US bid/ask (+fees+SLIPPAGE_BUFFER); default is fee+slippage only.
ESTIMATED_ROUNDTRIP_COST: Final[float] = _env_float(
    "ESTIMATED_ROUNDTRIP_COST",
    _DEFAULT_ROUNDTRIP_NO_SPREAD,
)
# Must match `ESTIMATED_ROUNDTRIP_COST` unless deliberately split for reporting.
ESTIMATED_ROUNDTRIP_COST_PCT: Final[float] = _env_float(
    "ESTIMATED_ROUNDTRIP_COST_PCT",
    ESTIMATED_ROUNDTRIP_COST,
)

# -- Sell thresholds ---------------------------------------------------------
# Real net profit floor (fraction of cost basis) required to take profit.
MIN_NET_PROFIT_TO_SELL: Final[float] = _env_float("MIN_NET_PROFIT_TO_SELL", 0.004)
# Optional absolute floor in USD (must clear both the percent floor and this).
# 0.0 disables; default 0.0 keeps backward compatibility.
MIN_PROFIT_AFTER_COSTS_USD: Final[float] = _env_float("MIN_PROFIT_AFTER_COSTS_USD", 0.0)

# -- Cooldowns ---------------------------------------------------------------
# Block re-entry on a symbol for this many seconds after Mystic closed it.
COOLDOWN_SECONDS_AFTER_SELL: Final[int] = _env_int(
    "COOLDOWN_SECONDS_AFTER_SELL",
    _env_int("POST_SELL_COOLDOWN_WALL_SEC", 2400),
)
# Block re-entry on a symbol after a HUMAN_MANUAL_SELL detected on the
# exchange. Same cadence as AI-triggered close by default.
COOLDOWN_SECONDS_AFTER_HUMAN_SELL: Final[int] = _env_int(
    "COOLDOWN_SECONDS_AFTER_HUMAN_SELL",
    COOLDOWN_SECONDS_AFTER_SELL,
)


@dataclass(frozen=True)
class TradingEconomicsSnapshot:
    taker_fee: float
    slippage_buffer: float
    estimated_roundtrip_cost: float
    min_net_profit_to_sell: float
    min_profit_after_costs_usd: float
    cooldown_seconds_after_sell: int
    cooldown_seconds_after_human_sell: int


def get_trading_economics() -> TradingEconomicsSnapshot:
    """Return the canonical economics snapshot shared by paper and live."""
    return TradingEconomicsSnapshot(
        taker_fee=TAKER_FEE,
        slippage_buffer=SLIPPAGE_BUFFER,
        estimated_roundtrip_cost=ESTIMATED_ROUNDTRIP_COST,
        min_net_profit_to_sell=MIN_NET_PROFIT_TO_SELL,
        min_profit_after_costs_usd=MIN_PROFIT_AFTER_COSTS_USD,
        cooldown_seconds_after_sell=COOLDOWN_SECONDS_AFTER_SELL,
        cooldown_seconds_after_human_sell=COOLDOWN_SECONDS_AFTER_HUMAN_SELL,
    )


def is_net_profit_acceptable(
    net_profit_pct: float,
    net_profit_usd: float,
) -> bool:
    """
    Centralized "should AI sell now?" check. Both PAPER and LIVE must call
    this same function to evaluate a candidate sell. Returns True only when
    the net profit clears both the percent floor and the (optional) USD floor.
    """
    if net_profit_pct < MIN_NET_PROFIT_TO_SELL:
        return False
    if MIN_PROFIT_AFTER_COSTS_USD > 0.0 and net_profit_usd < MIN_PROFIT_AFTER_COSTS_USD:
        return False
    return True


def log_trading_economics_at_startup() -> TradingEconomicsSnapshot:
    snap = get_trading_economics()
    logger.warning(
        "TRADING_ECONOMICS_RESOLVED taker_fee=%s slippage_buffer=%s roundtrip_cost=%s min_net_profit_to_sell=%s min_profit_after_costs_usd=%s cooldown_after_sell=%ss cooldown_after_human_sell=%ss",
        snap.taker_fee,
        snap.slippage_buffer,
        snap.estimated_roundtrip_cost,
        snap.min_net_profit_to_sell,
        snap.min_profit_after_costs_usd,
        snap.cooldown_seconds_after_sell,
        snap.cooldown_seconds_after_human_sell,
    )
    return snap
