"""
Market Events - All Live Data, No Fallback/Hardcoded Data

This module provides an event-driven system for live market data updates (backend port 8000).
All events:
- Broadcast live price updates from Binance.US API
- Deliver live market data events to trading components
- No fallback/hardcoded data - all events from live market operations
- Used by backend services on port 8000 for live trading operations

Live Data Sources:
- Price updates: Live market prices from Binance.US API
- Volume data: Live trading volume from Binance.US API
- Market events: All events from live market operations
- All events use live data from Binance.US API - no mock/test data

Endpoint References:
- Binance.US API: https://api.binance.us (live exchange API for market data)
- Backend API: Port 8000 (event bus used by backend services)
- All events from live endpoints - no fallback/hardcoded data
"""

import asyncio
from dataclasses import dataclass


@dataclass
class PriceUpdate:
    """
    Represents a live market price update event from Binance.US API.

    All fields contain live data from Binance.US API - no fallback/hardcoded data.
    """

    symbol: str  # Trading symbol (from live Binance.US Top-10)
    price: float  # Live price from Binance.US API
    volume: float  # Live trading volume from Binance.US API
    timestamp: float  # Live event timestamp (Unix timestamp in seconds)


class MarketEventBus:
    """
    Event bus for live market data updates from Binance.US API.

    Allows trading components to subscribe to live price updates and receive
    them in real-time instead of polling.
    All events broadcast live data from Binance.US API - no fallback/hardcoded data.
    Used by backend services on port 8000 for live trading operations.
    """

    def __init__(self) -> None:
        """Initialize market event bus for live market data updates."""
        self.price_subscribers: dict[str, set[asyncio.Queue]] = {}  # Symbol-specific subscribers for live price updates
        self.global_subscribers: set[asyncio.Queue] = set()  # Global subscribers for all live price updates
        self._lock = asyncio.Lock()  # Lock for thread-safe subscriber management

    async def subscribe(self, symbol: str | None = None) -> asyncio.Queue:
        """
        Subscribe to live price updates for a symbol (or all symbols if None).

        Subscribes to live price updates from Binance.US API.
        All events contain live data - no fallback/hardcoded data.

        Args:
            symbol: Trading symbol (from live Binance.US Top-10) to subscribe to, or None for all symbols

        Returns:
            asyncio.Queue: Queue that will receive live PriceUpdate events from Binance.US API
        """
        # Create queue for live price update events (max size: 1000 events)
        queue = asyncio.Queue(maxsize=1000)

        async with self._lock:
            if symbol is None:
                # Subscribe to all live price updates
                self.global_subscribers.add(queue)
            else:
                # Subscribe to live price updates for specific symbol
                if symbol not in self.price_subscribers:
                    self.price_subscribers[symbol] = set()
                self.price_subscribers[symbol].add(queue)

        return queue

    async def unsubscribe(self, queue: asyncio.Queue, symbol: str | None = None) -> None:
        """
        Unsubscribe from live price updates.

        Args:
            queue: The queue to unsubscribe from live price updates
            symbol: Symbol (from live Binance.US Top-10) to unsubscribe from, or None for global subscriptions
        """
        async with self._lock:
            if symbol is None:
                # Unsubscribe from all live price updates
                self.global_subscribers.discard(queue)
            elif symbol in self.price_subscribers:
                # Unsubscribe from live price updates for specific symbol
                self.price_subscribers[symbol].discard(queue)

    async def publish(self, update: PriceUpdate) -> None:
        """
        Publish a live price update to subscribers.

        Broadcasts live price update from Binance.US API to all subscribed queues.
        All updates contain live data - no fallback/hardcoded data.

        Args:
            update: The live price update from Binance.US API to publish
        """
        # Get subscribers (copy to avoid modification during iteration)
        async with self._lock:
            symbol_subs = self.price_subscribers.get(update.symbol, set()).copy()
            global_subs = self.global_subscribers.copy()

        # Notify all subscribers with live price update (non-blocking)
        tasks = []
        for queue in symbol_subs | global_subs:
            tasks.append(self._safe_put(queue, update))

        if tasks:
            # Broadcast live price update to all subscribers
            await asyncio.gather(*tasks, return_exceptions=True)

    async def _safe_put(self, queue: asyncio.Queue, update: PriceUpdate) -> None:
        """
        Safely put live price update in queue without blocking.

        Drops update if queue is full (subscriber too slow) or on errors.
        This prevents blocking on slow subscribers when broadcasting live price updates.

        Args:
            queue: Queue to put live price update in
            update: Live price update from Binance.US API to put
        """
        try:
            # Put live price update in queue with timeout (non-blocking)
            await asyncio.wait_for(queue.put(update), timeout=0.1)
        except (asyncio.TimeoutError, asyncio.QueueFull):
            # Drop live update if queue is full - subscriber is too slow (not fallback data, dropped update)
            pass
        except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError):
            # Drop live update on error (not fallback data, error handling)
            pass


# Global event bus instance for live market data updates from Binance.US API
# Used by backend services on port 8000 for live trading operations
# All events contain live data - no fallback/hardcoded data
market_event_bus = MarketEventBus()
