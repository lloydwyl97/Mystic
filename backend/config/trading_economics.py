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
from typing import Any, Final

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


# -- Fee model (Binance.US Advanced Spot) ------------------------------------
# Apr 2026 universal Advanced Spot: 0% maker, 0.02% taker (all pairs incl. top-four USDT).
# Legacy Tier-0 subset: 0% maker, 0.01% taker — NOT used for Mystic top-four USDT pairs.
# See backend.config.binance_us_fee_schedule for verification sources.
BINANCE_US_MAKER_FEE_PCT: Final[float] = _env_float("BINANCE_US_MAKER_FEE_PCT", 0.0)
BINANCE_US_TAKER_FEE_PCT: Final[float] = _env_float("BINANCE_US_TAKER_FEE_PCT", 0.0002)
BINANCE_US_TIER0_TAKER_FEE_PCT: Final[float] = _env_float("BINANCE_US_TIER0_TAKER_FEE_PCT", 0.0001)
EXCHANGE_NAME: Final[str] = "Binance.US"
FEE_SCHEDULE_SOURCE_DATE: Final[str] = os.getenv("FEE_SCHEDULE_SOURCE_DATE", "2026-04-21")

MAKER_FEE: Final[float] = _env_float("MAKER_FEE", BINANCE_US_MAKER_FEE_PCT)
TAKER_FEE: Final[float] = _env_float("TAKER_FEE", BINANCE_US_TAKER_FEE_PCT)
# One-sided slippage buffer beyond order-book (Advanced Trading has no platform spread).
SLIPPAGE_BUFFER: Final[float] = _env_float("SLIPPAGE_BUFFER", 0.0001)
# Default order-book half-spread estimate (live measured via bookTicker; override via env).
ORDERBOOK_HALF_SPREAD_ESTIMATE: Final[float] = _env_float("ORDERBOOK_HALF_SPREAD_ESTIMATE", 0.00006)
# Round-trip for sell gate: 2x taker fee + 2x half-spread + 2x slippage buffer (no platform spread).
_DEFAULT_ROUNDTRIP_NO_SPREAD: Final[float] = (2.0 * TAKER_FEE) + (2.0 * ORDERBOOK_HALF_SPREAD_ESTIMATE) + (2.0 * SLIPPAGE_BUFFER)
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

# -- DAY sizing (replay-aligned per-slot notional) -----------------------------
# Baseline replay uses $2,500/slot; 1.5x candidate → $3,750/slot, $15k max (4 slots).
DAY_BASE_NOTIONAL_PER_SLOT_USD: Final[float] = _env_float("DAY_BASE_NOTIONAL_PER_SLOT_USD", 2500.0)
DAY_NOTIONAL_MULT: Final[float] = _env_float("DAY_NOTIONAL_MULT", 1.0)
DAY_TARGET_NOTIONAL_PER_SLOT_USD: Final[float] = _env_float(
    "DAY_TARGET_NOTIONAL_PER_SLOT_USD",
    DAY_BASE_NOTIONAL_PER_SLOT_USD * DAY_NOTIONAL_MULT,
)
DAY_MAX_DEPLOYED_USD: Final[float] = _env_float(
    "DAY_MAX_DEPLOYED_USD",
    DAY_TARGET_NOTIONAL_PER_SLOT_USD * 4.0,
)
DAY_MAX_OPEN_SLOTS: Final[int] = _env_int("DAY_MAX_OPEN_SLOTS", 4)


@dataclass(frozen=True)
class TradingEconomicsSnapshot:
    exchange: str
    maker_fee: float
    taker_fee: float
    slippage_buffer: float
    orderbook_half_spread_estimate: float
    estimated_roundtrip_cost: float
    fee_schedule_source_date: str
    min_net_profit_to_sell: float
    min_profit_after_costs_usd: float
    cooldown_seconds_after_sell: int
    cooldown_seconds_after_human_sell: int
    day_notional_mult: float
    day_target_notional_per_slot_usd: float
    day_max_deployed_usd: float


def get_trading_economics() -> TradingEconomicsSnapshot:
    """Return the canonical economics snapshot shared by paper and live."""
    return TradingEconomicsSnapshot(
        exchange=EXCHANGE_NAME,
        maker_fee=MAKER_FEE,
        taker_fee=TAKER_FEE,
        slippage_buffer=SLIPPAGE_BUFFER,
        orderbook_half_spread_estimate=ORDERBOOK_HALF_SPREAD_ESTIMATE,
        estimated_roundtrip_cost=ESTIMATED_ROUNDTRIP_COST,
        fee_schedule_source_date=FEE_SCHEDULE_SOURCE_DATE,
        min_net_profit_to_sell=MIN_NET_PROFIT_TO_SELL,
        min_profit_after_costs_usd=MIN_PROFIT_AFTER_COSTS_USD,
        cooldown_seconds_after_sell=COOLDOWN_SECONDS_AFTER_SELL,
        cooldown_seconds_after_human_sell=COOLDOWN_SECONDS_AFTER_HUMAN_SELL,
        day_notional_mult=DAY_NOTIONAL_MULT,
        day_target_notional_per_slot_usd=DAY_TARGET_NOTIONAL_PER_SLOT_USD,
        day_max_deployed_usd=DAY_MAX_DEPLOYED_USD,
    )


def get_trading_economics_display() -> dict[str, Any]:
    """Dashboard/API display payload for fee model."""
    snap = get_trading_economics()
    return {
        "exchange": snap.exchange,
        "maker_fee_pct": snap.maker_fee,
        "taker_fee_pct": snap.taker_fee,
        "maker_fee_bps": round(snap.maker_fee * 10000, 2),
        "taker_fee_bps": round(snap.taker_fee * 10000, 2),
        "slippage_buffer_pct": snap.slippage_buffer,
        "orderbook_half_spread_estimate_pct": snap.orderbook_half_spread_estimate,
        "orderbook_full_spread_estimate_pct": snap.orderbook_half_spread_estimate * 2,
        "roundtrip_estimated_cost_pct": snap.estimated_roundtrip_cost,
        "roundtrip_estimated_cost_bps": round(snap.estimated_roundtrip_cost * 10000, 2),
        "fee_schedule_source_date": snap.fee_schedule_source_date,
        "fee_schedule_note": ("Binance.US Advanced Spot: 0% maker / 0.02% taker universal (Apr 2026). No platform spread; order-book spread + slippage buffer only."),
        "min_net_profit_to_sell_pct": snap.min_net_profit_to_sell,
        "day_notional_mult": snap.day_notional_mult,
        "day_base_notional_per_slot_usd": DAY_BASE_NOTIONAL_PER_SLOT_USD,
        "day_target_notional_per_slot_usd": snap.day_target_notional_per_slot_usd,
        "day_max_deployed_usd": snap.day_max_deployed_usd,
        "day_max_open_slots": DAY_MAX_OPEN_SLOTS,
        "baseline_lock_id": os.getenv("DAY_BASELINE_LOCK_ID", "day_baseline_all_pass_v1_size_1_5"),
    }


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
    return not (MIN_PROFIT_AFTER_COSTS_USD > 0.0 and net_profit_usd < MIN_PROFIT_AFTER_COSTS_USD)


def log_trading_economics_at_startup() -> TradingEconomicsSnapshot:
    snap = get_trading_economics()
    logger.warning(
        "TRADING_ECONOMICS_RESOLVED exchange=%s maker_fee=%s taker_fee=%s slippage_buffer=%s "
        "orderbook_half_spread=%s roundtrip_cost=%s min_net_profit_to_sell=%s "
        "min_profit_after_costs_usd=%s cooldown_after_sell=%ss cooldown_after_human_sell=%ss fee_source=%s "
        "day_notional_mult=%s day_target_notional_per_slot=%s day_max_deployed=%s",
        snap.exchange,
        snap.maker_fee,
        snap.taker_fee,
        snap.slippage_buffer,
        snap.orderbook_half_spread_estimate,
        snap.estimated_roundtrip_cost,
        snap.min_net_profit_to_sell,
        snap.min_profit_after_costs_usd,
        snap.cooldown_seconds_after_sell,
        snap.cooldown_seconds_after_human_sell,
        snap.fee_schedule_source_date,
        snap.day_notional_mult,
        snap.day_target_notional_per_slot_usd,
        snap.day_max_deployed_usd,
    )
    return snap
