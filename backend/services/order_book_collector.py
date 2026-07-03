"""
Order Book Data Collector
WebSocket client for Binance order book streams
Connects to Binance WebSocket and forwards order book updates to OrderBookService
All data from Binance US only - Production ready
"""

import asyncio
import json
import logging
import os
import socket
from typing import Any

# Force IPv4 only for Binance US (required)
_orig_getaddrinfo = socket.getaddrinfo


def _ipv4_getaddrinfo(host, port, family=0, type=0, proto=0, flags=0):
    return _orig_getaddrinfo(host, port, socket.AF_INET, type, proto, flags)


socket.getaddrinfo = _ipv4_getaddrinfo

import websockets

from backend.services.order_book_service import order_book_service
from backend.services.task_manager import task_manager

logger = logging.getLogger(__name__)


class OrderBookCollector:
    """
    Collects real-time order book data from Binance WebSocket
    Forwards to OrderBookService for feature calculation
    """

    def __init__(self) -> None:
        self.is_running = False
        self.ws_url = "wss://stream.binance.us:9443/stream"
        self.symbols = []
        self.websocket = None
        self._last_heartbeat_ts = 0.0

        # Top-4 Binance.US trading symbols only (Mystic day-trade scope)
        symbols_str = os.getenv("TRADING_SYMBOLS", "BTC,ETH,SOL,XRP")
        self.symbols = [s.strip().replace("USDT", "") for s in symbols_str.split(",")]

        # Stats
        self.stats = {
            "messages_received": 0,
            "order_books_processed": 0,
            "errors": 0,
            "reconnects": 0,
            "last_error": None,
        }

        logger.info(f"OrderBookCollector initialized for {len(self.symbols)} symbols")

    async def start(self) -> None:
        """Start collecting order book data from Binance WebSocket"""
        self.is_running = True

        # Start WebSocket connection in background
        self._ws_task = await task_manager.create_task(self._run_websocket(), name="order_book_collector:run_websocket")

        logger.info("OrderBookCollector started - connecting to Binance WebSocket")

    async def stop(self) -> None:
        """Stop the collector"""
        self.is_running = False
        if self.websocket:
            await self.websocket.close()
        logger.info("OrderBookCollector stopped")

    async def _run_websocket(self) -> None:
        """Main WebSocket loop with reconnection"""
        while self.is_running:
            try:
                await self._connect_and_listen()
            except Exception as e:
                logger.warning(f"WebSocket error: {e}, reconnecting in 5s...")
                self.stats["errors"] += 1
                self.stats["last_error"] = str(e)
                self.stats["reconnects"] += 1
                await asyncio.sleep(5)

    async def _connect_and_listen(self) -> None:
        """Connect to Binance WebSocket and listen for order book updates"""
        try:
            # Partial book depth snapshots (top 20) — not incremental deltas
            streams = [f"{symbol.lower()}usdt@depth20@100ms" for symbol in self.symbols]

            # Connect to WebSocket with explicit timeout
            async with websockets.connect(self.ws_url, open_timeout=30, ping_interval=20, ping_timeout=20) as websocket:
                self.websocket = websocket

                # Subscribe to streams
                subscribe_msg = {"method": "SUBSCRIBE", "params": streams, "id": 1}
                await websocket.send(json.dumps(subscribe_msg))
                logger.info(f"Subscribed to {len(streams)} order book streams")

                # Listen for messages
                async for message in websocket:
                    if not self.is_running:
                        break

                    try:
                        await self._process_message(message)
                    except Exception as e:
                        logger.debug(f"Error processing message: {e}")
                        self.stats["errors"] += 1

        except websockets.exceptions.WebSocketException as e:
            logger.warning(f"WebSocket connection error: {e}")
            raise
        except Exception as e:
            err_msg = str(e).strip() or type(e).__name__
            logger.warning("WebSocket error: %s, reconnecting", err_msg)
            raise

    async def _process_message(self, message: str) -> None:
        """
        Process order book update message from Binance

        Message format (partial book depth snapshot, e.g. btcusdt@depth20@100ms):
        {
            "stream": "btcusdt@depth20@100ms",
            "data": {
                "lastUpdateId": 160,
                "bids": [["9168.00", "1.00"], ...],  # full top-N bids
                "asks": [["9169.00", "1.00"], ...]   # full top-N asks
            }
        }

        Note: this is NOT the incremental diff-depth stream (which uses
        abbreviated "b"/"a" keys). Partial-depth snapshot streams always use
        the full "bids"/"asks" key names.
        """
        try:
            data = json.loads(message)

            # Skip subscription confirmation messages
            if "result" in data:
                return

            if "stream" not in data or "data" not in data:
                return

            stream = data["stream"]
            update = data["data"]

            # Extract symbol from stream name (btcusdt@depth20@100ms -> BTC)
            symbol = stream.split("@")[0].replace("usdt", "").upper()

            # Partial depth snapshot: full top-N bids/asks each tick
            bids = update.get("bids", [])
            asks = update.get("asks", [])

            if not bids or not asks:
                return

            top_bids = [[float(b[0]), float(b[1])] for b in bids]
            top_asks = [[float(a[0]), float(a[1])] for a in asks]

            # Forward to order book service for processing
            # Ensure service is started before processing
            if not order_book_service.is_running:
                logger.info(f"Starting OrderBookService for first message from {symbol}")
                await order_book_service.start()
            await order_book_service.process_order_book(symbol, top_bids, top_asks)

            self.stats["messages_received"] += 1
            self.stats["order_books_processed"] += 1
            await self._heartbeat_throttled(symbol)

            # Log first successful processing per symbol
            if self.stats["order_books_processed"] == 1:
                logger.info(f"First order book processed for {symbol}")

        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse WebSocket message: {e}")
            self.stats["errors"] += 1
        except Exception as e:
            logger.warning(f"Error processing order book message: {e}")
            self.stats["errors"] += 1

    async def get_stats(self) -> dict[str, Any]:
        """Get collector statistics"""
        return {
            "is_running": self.is_running,
            "symbols": self.symbols,
            "stats": self.stats,
        }

    async def _heartbeat_throttled(self, symbol: str) -> None:
        """Emit a task-health heartbeat at most once every 10s (avoid Redis spam at ~40 msg/s)."""
        now = asyncio.get_event_loop().time()
        if now - self._last_heartbeat_ts < 10.0:
            return
        self._last_heartbeat_ts = now
        try:
            from backend.config.redis_config import get_shared_redis_async
            from backend.services.task_health_monitor import beat

            await beat(
                "order_book_collector:ws_messages",
                get_shared_redis_async(),
                extra={"last_symbol": symbol, "messages_received": self.stats["messages_received"]},
            )
        except Exception:
            pass


# Singleton instance
order_book_collector = OrderBookCollector()
