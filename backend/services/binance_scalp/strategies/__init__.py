"""Paper-only scalp strategy modules."""

from __future__ import annotations

from typing import TYPE_CHECKING

from backend.services.binance_scalp.strategies.base import ScalpSetupSignal
from backend.services.binance_scalp.strategies.breakout_momentum import BreakoutMomentumStrategy
from backend.services.binance_scalp.strategies.orderbook_tape_scalp import OrderbookTapeScalpStrategy
from backend.services.binance_scalp.strategies.range_bounce_scalp import RangeBounceScalpStrategy
from backend.services.binance_scalp.strategies.vwap_ema_reclaim import VwapEmaReclaimStrategy

if TYPE_CHECKING:
    from backend.services.binance_scalp.config import ScalpConfig

ALL_STRATEGIES = (
    BreakoutMomentumStrategy(),
    VwapEmaReclaimStrategy(),
    OrderbookTapeScalpStrategy(),
    RangeBounceScalpStrategy(),
)

STRATEGY_NAMES = tuple(s.name for s in ALL_STRATEGIES)


def enabled_strategies(config: ScalpConfig) -> tuple:
    disabled = config.disabled_strategies
    return tuple(s for s in ALL_STRATEGIES if s.name not in disabled)


__all__ = [
    "ScalpSetupSignal",
    "ALL_STRATEGIES",
    "STRATEGY_NAMES",
    "enabled_strategies",
    "BreakoutMomentumStrategy",
    "VwapEmaReclaimStrategy",
    "OrderbookTapeScalpStrategy",
    "RangeBounceScalpStrategy",
]
