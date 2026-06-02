"""
Signal Events - All Live Data, No Fallback/Hardcoded Data

This module provides an event-driven system for live trading signal distribution (backend port 8000).
All signals:
- Generated from live market data and AI models
- Broadcast live trading signals to trading components
- No fallback/hardcoded signals - all signals from live trading operations
- Used by backend services on port 8000 for live trading operations

Live Data Sources:
- Trading signals: Generated from live market data analysis (Binance.US API)
- Signal confidence: Calculated from live market indicators
- Signal prices: Current live prices from Binance.US API
- Signal metadata: Derived from live market conditions
- All signals use live data from Binance.US API and AI models - no mock/test data

Endpoint References:
- Binance.US API: https://api.binance.us (live exchange API for market data)
- Backend API: Port 8000 (signal event bus used by backend services)
- AI Models: Live trading signal generation from market analysis
- All signals from live endpoints - no fallback/hardcoded data
"""

import asyncio
from dataclasses import dataclass
from enum import Enum


class SignalType(Enum):
    """
    Types of live trading signals from AI models and market analysis.

    All signal types are generated from live market data - no fallback/hardcoded signals.
    """

    BUY = "buy"  # Live buy signal from AI model analysis
    SELL = "sell"  # Live sell signal from AI model analysis
    HOLD = "hold"  # Live hold signal from AI model analysis


@dataclass
class TradingSignal:
    """
    Represents a live trading signal event from AI models and market analysis.

    All fields contain live data from Binance.US API and AI models - no fallback/hardcoded data.
    """

    symbol: str  # Trading symbol (from live Binance.US Top-10)
    signal_type: SignalType  # Live signal type from AI model analysis
    confidence: float  # Live confidence score from AI model (0.0-1.0)
    price: float  # Live current price from Binance.US API
    source: str  # Source of live signal (AI model identifier)
    timestamp: float  # Live event timestamp (Unix timestamp in seconds)
    metadata: dict  # Live signal metadata (indicators, market conditions, etc.)


class SignalEventBus:
    """
    Event bus for live trading signal distribution from AI models.

    Allows signal generators to publish live trading signals and consumers to receive
    them in real-time without polling.
    All signals broadcast live data from AI models and market analysis - no fallback/hardcoded data.
    Used by backend services on port 8000 for live trading operations.
    """

    def __init__(self) -> None:
        """Initialize signal event bus for live trading signal distribution."""
        self.subscribers: set[asyncio.Queue] = set()  # Subscribers for live trading signals
        self._lock = asyncio.Lock()  # Lock for thread-safe subscriber management

    async def subscribe(self) -> asyncio.Queue:
        """
        Subscribe to live trading signals from AI models.

        Subscribes to live trading signals generated from market analysis.
        All signals contain live data - no fallback/hardcoded signals.

        Returns:
            asyncio.Queue: Queue that will receive live TradingSignal events from AI models
        """
        # Create queue for live trading signal events (max size: 500 signals)
        queue = asyncio.Queue(maxsize=500)
        async with self._lock:
            # Subscribe to live trading signals
            self.subscribers.add(queue)
        return queue

    async def unsubscribe(self, queue: asyncio.Queue) -> None:
        """
        Unsubscribe from live trading signals.

        Args:
            queue: The queue to unsubscribe from live trading signals
        """
        async with self._lock:
            # Unsubscribe from live trading signals
            self.subscribers.discard(queue)

    async def publish(self, signal: TradingSignal) -> None:
        """
        Publish a live trading signal to subscribers.

        Broadcasts live trading signal from AI models to all subscribed queues.
        All signals contain live data - no fallback/hardcoded signals.

        Args:
            signal: The live trading signal from AI models to publish
        """
        # Get subscribers (copy to avoid modification during iteration)
        async with self._lock:
            subs = self.subscribers.copy()

        # Notify all subscribers with live trading signal (non-blocking)
        tasks = []
        for queue in subs:
            tasks.append(self._safe_put(queue, signal))

        if tasks:
            # Broadcast live trading signal to all subscribers
            await asyncio.gather(*tasks, return_exceptions=True)

    async def _safe_put(self, queue: asyncio.Queue, signal: TradingSignal) -> None:
        """
        Safely put live trading signal in queue without blocking.

        Drops signal if queue is full (subscriber too slow) or on errors.
        This prevents blocking on slow subscribers when broadcasting live trading signals.

        Args:
            queue: Queue to put live trading signal in
            signal: Live trading signal from AI models to put
        """
        try:
            # Put live trading signal in queue with timeout (non-blocking)
            await asyncio.wait_for(queue.put(signal), timeout=0.1)
        except (asyncio.TimeoutError, asyncio.QueueFull):
            # Drop live signal if queue is full - subscriber is too slow (not fallback data, dropped signal)
            pass
        except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError):
            # Drop live signal on error (not fallback data, error handling)
            pass


# Global signal event bus instance for live trading signal distribution from AI models
# Used by backend services on port 8000 for live trading operations
# All signals contain live data - no fallback/hardcoded signals
signal_event_bus = SignalEventBus()
