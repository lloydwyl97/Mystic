"""
Order Book Service
Subscribes to Binance order book via WebSocket
Calculates bid/ask spread, imbalance, depth, liquidity score, price impact, market efficiency
All data from Binance US only - Production ready
"""

import contextlib
import logging
import os
from datetime import datetime, timezone
from typing import Any

import redis.asyncio as redis

from backend.services.redis_service import get_redis_service

logger = logging.getLogger(__name__)


def order_book_features_from_bids_asks(
    bids: list[list[float]],
    asks: list[list[float]],
    *,
    depth_levels: int = 20,
) -> dict[str, float]:
    """
    Same microstructure mapping as OrderBookService writes to Redis ``orderbook:{BASE}``.
    Shared so live inference can fall back when WebSocket hashes are cold.
    """
    features: dict[str, float] = {
        "bid_ask_spread": 0.0,
        "order_book_imbalance": 0.0,
        "market_depth": 0.0,
        "liquidity_score": 0.0,
        "price_impact": 0.0,
        "market_efficiency": 0.0,
    }
    try:
        if not bids or not asks:
            return features

        best_bid = float(bids[0][0]) if bids else 0.0
        best_ask = float(asks[0][0]) if asks else 0.0

        if best_bid <= 0 or best_ask <= 0:
            return features

        features["bid_ask_spread"] = (best_ask - best_bid) / best_bid

        bid_volume = sum(float(b[1]) for b in bids[:depth_levels])
        ask_volume = sum(float(a[1]) for a in asks[:depth_levels])
        total_volume = bid_volume + ask_volume
        if total_volume > 0:
            features["order_book_imbalance"] = (bid_volume - ask_volume) / total_volume

        features["market_depth"] = total_volume

        spread_score = max(0, 1 - (features["bid_ask_spread"] * 1000))
        volume_score = min(1.0, total_volume / 100.0)
        features["liquidity_score"] = (spread_score + volume_score) / 2.0

        impact_size = 1.0
        cumulative_volume = 0.0
        weighted_price = 0.0

        for ask_price, ask_qty in asks[:depth_levels]:
            ask_price_f = float(ask_price)
            ask_qty_f = float(ask_qty)

            if cumulative_volume + ask_qty_f >= impact_size:
                remaining = impact_size - cumulative_volume
                weighted_price += ask_price_f * remaining
                cumulative_volume += remaining
                break
            weighted_price += ask_price_f * ask_qty_f
            cumulative_volume += ask_qty_f

        if cumulative_volume > 0:
            avg_fill_price = weighted_price / cumulative_volume
            features["price_impact"] = (avg_fill_price - best_ask) / best_ask

        if features["bid_ask_spread"] > 0:
            features["market_efficiency"] = 1.0 / (1.0 + features["bid_ask_spread"] * 100)

    except Exception as e:
        logger.debug("order_book_features_from_bids_asks failed: %s", e)
        return features
    else:
        return features


async def fetch_order_book_features_live(ccxt_symbol: str, *, depth_levels: int | None = None) -> dict[str, float] | None:
    """Fetch L2 from exchange public API and return the same hash fields as Redis."""
    dl = depth_levels if depth_levels is not None else int(os.getenv("ORDER_BOOK_DEPTH", "20"))
    try:
        from backend.services.live_market_data import live_market_data_service

        if live_market_data_service is None:
            return None
        ob = await live_market_data_service.get_order_book(ccxt_symbol, limit=max(10, dl))
        if not ob:
            return None
        bids = ob.get("bids") or []
        asks = ob.get("asks") or []
        if not bids or not asks:
            return None
        return order_book_features_from_bids_asks(bids, asks, depth_levels=dl)
    except Exception as e:
        logger.debug("fetch_order_book_features_live failed for %s: %s", ccxt_symbol, e)
        return None


class OrderBookService:
    """
    Manages real-time order book data from Binance WebSocket
    Calculates market microstructure features for AI trading
    """

    def __init__(self) -> None:
        self.is_running = False
        self.redis: redis.Redis | None = None
        self.order_books: dict[str, dict[str, Any]] = {}
        self.last_update: dict[str, datetime] = {}

        # Configuration
        self.depth_levels = int(os.getenv("ORDER_BOOK_DEPTH", "20"))
        self.update_speed = os.getenv("ORDER_BOOK_UPDATE_SPEED", "100ms")

        # Stats
        self.stats = {
            "updates_processed": 0,
            "features_calculated": 0,
            "errors": 0,
            "last_error": None,
        }

        logger.info("OrderBookService initialized - market microstructure features ready")

    async def start(self) -> None:
        """Start the order book service"""
        try:
            # Connect to Redis
            redis_url = os.getenv("REDIS_URL")
            if not redis_url:
                redis_host = os.getenv("REDIS_HOST")
                if not redis_host:
                    logger.warning("No Redis configuration found for order book service")
                    return
                redis_port = os.getenv("REDIS_PORT", "6379")
                redis_db = os.getenv("REDIS_DB", "0")
                redis_url = f"redis://{redis_host}:{redis_port}/{redis_db}"

            self.redis = get_redis_service()

            self.is_running = True
            logger.info("OrderBookService started - ready to process Binance order book data")

        except Exception as e:
            logger.exception(f"Failed to start OrderBookService: {e}")
            self.stats["errors"] += 1
            self.stats["last_error"] = str(e)

    async def stop(self) -> None:
        """Stop the order book service"""
        self.is_running = False
        if self.redis:
            await self.redis.close()
        logger.info("OrderBookService stopped")

    async def process_order_book(self, symbol: str, bids: list[list[float]], asks: list[list[float]]) -> None:
        """
        Process order book update from Binance WebSocket

        Args:
            symbol: Trading symbol (e.g., "BTC")
            bids: List of [price, quantity] for bids
            asks: List of [price, quantity] for asks
        """
        try:
            # Ensure Redis connection exists
            if not self.redis:
                self.redis = get_redis_service()
                if not self.redis:
                    logger.debug("Order book service has no Redis connection")
                    return

            # Store raw order book
            self.order_books[symbol] = {
                "bids": bids,
                "asks": asks,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
            self.last_update[symbol] = datetime.now(timezone.utc)

            # Calculate features (all live from Binance)
            features = self._calculate_features(symbol, bids, asks)

            # Store in Redis (TTL: 5 seconds - very short-lived) - Convert mapping to individual hset calls for compatibility
            order_book_key = f"orderbook:{symbol}"
            for field, value in features.items():
                await self.redis.hset(order_book_key, field, str(value))
            await self.redis.expire(order_book_key, 5)

            self.stats["updates_processed"] += 1
            self.stats["features_calculated"] += 1

            logger.debug(f"Order book for {symbol}: spread={features.get('bid_ask_spread', 0):.4f}, imbalance={features.get('order_book_imbalance', 0):.3f}")

        except Exception as e:
            error_str = str(e)
            # Handle Redis buffer closed error by reconnecting
            if "Buffer is closed" in error_str or "Connection closed" in error_str:
                logger.debug(f"Redis connection lost for {symbol}, reconnecting...")
                with contextlib.suppress(Exception):
                    self.redis = get_redis_service()
            else:
                logger.warning(f"Failed to process order book for {symbol}: {e}")
            self.stats["errors"] += 1
            self.stats["last_error"] = str(e)

    def _calculate_features(self, symbol: str, bids: list[list[float]], asks: list[list[float]]) -> dict[str, float]:
        """
        Calculate all order book features from live Binance data

        Returns:
            Dictionary with 6 features: spread, imbalance, depth, liquidity, impact, efficiency
        """
        _ = symbol
        return order_book_features_from_bids_asks(bids, asks, depth_levels=self.depth_levels)

    async def get_order_book_features(self, symbol: str) -> dict[str, float] | None:
        """
        Get calculated order book features for a symbol from Redis

        Args:
            symbol: Trading symbol (e.g., "BTC")

        Returns:
            Dictionary of features or None if not available
        """
        try:
            if not self.redis:
                return None

            order_book_key = f"orderbook:{symbol}"
            data = await self.redis.hgetall(order_book_key)

            if not data:
                logger.debug(f"No order book data found for {symbol}")
                return None

            # Convert strings back to floats
            features = {
                "bid_ask_spread": float(data.get("bid_ask_spread", 0)),
                "order_book_imbalance": float(data.get("order_book_imbalance", 0)),
                "market_depth": float(data.get("market_depth", 0)),
                "liquidity_score": float(data.get("liquidity_score", 0)),
                "price_impact": float(data.get("price_impact", 0)),
                "market_efficiency": float(data.get("market_efficiency", 0)),
            }

        except Exception as e:
            logger.debug(f"Failed to get order book features for {symbol}: {e}")
            return None
        else:
            return features

    async def get_stats(self) -> dict[str, Any]:
        """Get service statistics"""
        return {
            "is_running": self.is_running,
            "symbols_tracked": len(self.order_books),
            "stats": self.stats,
            "config": {
                "depth_levels": self.depth_levels,
                "update_speed": self.update_speed,
            },
        }


# Singleton instance
order_book_service = OrderBookService()
