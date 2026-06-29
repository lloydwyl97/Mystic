"""Paper-only scalp strategy modules."""

from __future__ import annotations

from typing import TYPE_CHECKING

from backend.services.binance_scalp.strategies.base import ScalpSetupSignal
from backend.services.binance_scalp.strategies.breakout_momentum import BreakoutMomentumStrategy
from backend.services.binance_scalp.strategies.compression_breakout import CompressionBreakoutStrategy
from backend.services.binance_scalp.strategies.failed_breakdown_reversal import FailedBreakdownReversalStrategy
from backend.services.binance_scalp.strategies.failed_breakout_reversal import FailedBreakoutReversalStrategy
from backend.services.binance_scalp.strategies.orderbook_tape_scalp import OrderbookTapeScalpStrategy
from backend.services.binance_scalp.strategies.range_bounce_scalp import RangeBounceScalpStrategy
from backend.services.binance_scalp.strategies.trend_pullback_micro import TrendPullbackMicroStrategy
from backend.services.binance_scalp.strategies.volume_impulse_continuation import VolumeImpulseContinuationStrategy
from backend.services.binance_scalp.strategies.vwap_ema_reclaim import VwapEmaReclaimStrategy

if TYPE_CHECKING:
    from backend.services.binance_scalp.config import ScalpConfig

ALL_STRATEGIES = (
    BreakoutMomentumStrategy(),
    VwapEmaReclaimStrategy(),
    OrderbookTapeScalpStrategy(),
    RangeBounceScalpStrategy(),
    FailedBreakdownReversalStrategy(),
    CompressionBreakoutStrategy(),
    VolumeImpulseContinuationStrategy(),
    TrendPullbackMicroStrategy(),
    FailedBreakoutReversalStrategy(),
)

STRATEGY_NAMES = tuple(s.name for s in ALL_STRATEGIES)


def enabled_strategies(config: ScalpConfig) -> tuple:
    disabled = config.disabled_strategies
    return tuple(s for s in ALL_STRATEGIES if s.name not in disabled)


__all__ = [
    "ALL_STRATEGIES",
    "STRATEGY_NAMES",
    "BreakoutMomentumStrategy",
    "OrderbookTapeScalpStrategy",
    "RangeBounceScalpStrategy",
    "ScalpSetupSignal",
    "VwapEmaReclaimStrategy",
    "enabled_strategies",
]
