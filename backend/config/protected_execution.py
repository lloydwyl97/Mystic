"""
Protected limit execution configuration.

Controls order-book preflight and protected limit pricing for BUY/SELL.
Paper and live-capable paths share the same rules; live remains gated separately.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Final

logger = logging.getLogger(__name__)


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name, "")
    if not raw:
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on")


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name, "")
    if not raw:
        return float(default)
    try:
        return float(raw)
    except (TypeError, ValueError):
        logger.warning("PROTECTED_EXEC env %s=%r invalid; using %s", name, raw, default)
        return float(default)


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name, "")
    if not raw:
        return int(default)
    try:
        return int(raw)
    except (TypeError, ValueError):
        logger.warning("PROTECTED_EXEC env %s=%r invalid; using %s", name, raw, default)
        return int(default)


MAKER_FEE: Final[float] = _env_float("MAKER_FEE", 0.0)
BINANCE_US_MAKER_FEE_PCT: Final[float] = _env_float("BINANCE_US_MAKER_FEE_PCT", MAKER_FEE)

MAX_ORDERBOOK_SPREAD_PCT: Final[float] = _env_float("MAX_ORDERBOOK_SPREAD_PCT", 0.0005)


def day_paper_align_spread_with_bar_enabled() -> bool:
    """When true (paper sim only), preflight uses MAX_SPREAD_PCT bar ceiling instead of tight orderbook cap."""
    return _env_bool("DAY_PAPER_ALIGN_SPREAD_WITH_BAR", False)


def bar_rank_max_spread_fraction() -> float:
    """Same units as MAX_ORDERBOOK_SPREAD_PCT (fraction, not percent points)."""
    return _env_float("MAX_SPREAD_PCT", 1.0) / 100.0


def effective_max_orderbook_spread_pct(*, live_capable: bool) -> float:
    if not live_capable and day_paper_align_spread_with_bar_enabled():
        return max(MAX_ORDERBOOK_SPREAD_PCT, bar_rank_max_spread_fraction())
    return MAX_ORDERBOOK_SPREAD_PCT
MAX_ORDERBOOK_PRICE_IMPACT_PCT: Final[float] = _env_float("MAX_ORDERBOOK_PRICE_IMPACT_PCT", 0.0005)
PROTECTED_LIMIT_ORDER_TIMEOUT_SEC: Final[int] = _env_int("PROTECTED_LIMIT_ORDER_TIMEOUT_SEC", 8)
PROTECTED_LIMIT_ALLOW_PARTIAL: Final[bool] = _env_bool("PROTECTED_LIMIT_ALLOW_PARTIAL", False)
USE_PROTECTED_LIMIT_EXECUTION: Final[bool] = _env_bool("USE_PROTECTED_LIMIT_EXECUTION", True)
ORDERBOOK_MAX_AGE_SEC: Final[float] = _env_float("ORDERBOOK_MAX_AGE_SEC", 5.0)
ORDERBOOK_DEPTH_LIMIT: Final[int] = _env_int("ORDERBOOK_DEPTH_LIMIT", 100)

# Reject reason codes (stable strings for logs/API)
ORDERBOOK_STALE = "ORDERBOOK_STALE"
ORDERBOOK_MISSING = "ORDERBOOK_MISSING"
SPREAD_TOO_WIDE = "SPREAD_TOO_WIDE"
DEPTH_INSUFFICIENT = "DEPTH_INSUFFICIENT"
PRICE_IMPACT_TOO_HIGH = "PRICE_IMPACT_TOO_HIGH"
EXECUTABLE_NET_PROFIT_BELOW_FLOOR = "EXECUTABLE_NET_PROFIT_BELOW_FLOOR"
PROTECTED_FILL_NOT_PROFITABLE = "PROTECTED_FILL_NOT_PROFITABLE"


@dataclass
class ProtectedExecutionSnapshot:
    use_protected_limit_execution: bool
    maker_fee: float
    taker_fee: float
    max_orderbook_spread_pct: float
    max_orderbook_price_impact_pct: float
    protected_limit_order_timeout_sec: int
    protected_limit_allow_partial: bool


def get_protected_execution_snapshot(*, taker_fee: float) -> ProtectedExecutionSnapshot:
    return ProtectedExecutionSnapshot(
        use_protected_limit_execution=USE_PROTECTED_LIMIT_EXECUTION,
        maker_fee=MAKER_FEE,
        taker_fee=taker_fee,
        max_orderbook_spread_pct=MAX_ORDERBOOK_SPREAD_PCT,
        max_orderbook_price_impact_pct=MAX_ORDERBOOK_PRICE_IMPACT_PCT,
        protected_limit_order_timeout_sec=PROTECTED_LIMIT_ORDER_TIMEOUT_SEC,
        protected_limit_allow_partial=PROTECTED_LIMIT_ALLOW_PARTIAL,
    )


def log_protected_execution_at_startup(*, taker_fee: float) -> ProtectedExecutionSnapshot:
    snap = get_protected_execution_snapshot(taker_fee=taker_fee)
    logger.warning(
        "PROTECTED_EXECUTION enabled=%s maker_fee=%s taker_fee=%s max_spread=%s paper_effective_spread=%s align_with_bar=%s max_impact=%s timeout_sec=%s allow_partial=%s",
        snap.use_protected_limit_execution,
        snap.maker_fee,
        snap.taker_fee,
        snap.max_orderbook_spread_pct,
        effective_max_orderbook_spread_pct(live_capable=False),
        day_paper_align_spread_with_bar_enabled(),
        snap.max_orderbook_price_impact_pct,
        snap.protected_limit_order_timeout_sec,
        snap.protected_limit_allow_partial,
    )
    return snap
