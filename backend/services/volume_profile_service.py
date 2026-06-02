"""
Volume Profile Service
Calculates Point of Control (POC), Value Area High (VAH), Value Area Low (VAL)
Uses Binance historical data already in Redis
All data from Binance US only - Production ready
"""

import asyncio
import json
import logging
import os
from datetime import datetime, timezone
from typing import Any

import redis.asyncio as redis

from backend.services.redis_service import get_redis_service
from backend.services.task_manager import task_manager

logger = logging.getLogger(__name__)


class VolumeProfileService:
    """
    Calculates volume profile from Binance historical data
    Provides POC, VAH, VAL for support/resistance levels
    """

    def __init__(self) -> None:
        self.is_running = False
        self.redis: redis.Redis | None = None
        self.volume_profiles: dict[str, dict[str, float]] = {}
        self.last_calculation: dict[str, datetime] = {}

        # Configuration
        self.lookback_hours = int(os.getenv("VOLUME_PROFILE_LOOKBACK_HOURS", "24"))
        self.update_interval = int(os.getenv("VOLUME_PROFILE_UPDATE_INTERVAL", "900"))  # 15 minutes
        self.price_bins = int(os.getenv("VOLUME_PROFILE_BINS", "100"))

        # Stats
        self.stats = {
            "profiles_calculated": 0,
            "errors": 0,
            "last_error": None,
        }

        logger.info(f"VolumeProfileService initialized - {self.lookback_hours}h lookback")

    async def start(self) -> None:
        """Start the volume profile service"""
        try:
            # Connect to Redis
            redis_url = os.getenv("REDIS_URL")
            if not redis_url:
                redis_host = os.getenv("REDIS_HOST")
                if not redis_host:
                    logger.warning("No Redis configuration found for volume profile service")
                    return
                redis_port = os.getenv("REDIS_PORT", "6379")
                redis_db = os.getenv("REDIS_DB", "0")
                redis_url = f"redis://{redis_host}:{redis_port}/{redis_db}"

            self.redis = get_redis_service()

            self.is_running = True

            # Start background calculation loop
            self._calc_task = await task_manager.create_task(self._calculation_loop(), name="volume_profile_service:calculation_loop")

            logger.info("VolumeProfileService started - calculating POC/VAH/VAL from Binance data")

        except Exception as e:
            logger.exception(f"Failed to start VolumeProfileService: {e}")
            self.stats["errors"] += 1
            self.stats["last_error"] = str(e)

    async def stop(self) -> None:
        """Stop the volume profile service"""
        self.is_running = False
        if self.redis:
            await self.redis.close()
        logger.info("VolumeProfileService stopped")

    async def _calculation_loop(self) -> None:
        """Background loop to calculate volume profiles periodically"""
        # Top-4 Binance.US trading symbols only (Mystic day-trade scope)
        symbols_str = os.getenv("TRADING_SYMBOLS", "BTC,ETH,SOL,XRP")
        symbols = [s.strip() for s in symbols_str.split(",")]

        while self.is_running:
            try:
                for symbol in symbols:
                    try:
                        await self.calculate_volume_profile(symbol)
                    except Exception as e:
                        logger.debug(f"Error calculating volume profile for {symbol}: {e}")

                # Wait before next update
                await asyncio.sleep(self.update_interval)

            except Exception as e:
                logger.warning(f"Error in volume profile calculation loop: {e}")
                self.stats["errors"] += 1
                await asyncio.sleep(60)

    async def calculate_volume_profile(self, symbol: str) -> dict[str, float]:
        """
        Calculate volume profile (POC, VAH, VAL) for a symbol

        Args:
            symbol: Trading symbol (e.g., "BTC")

        Returns:
            Dictionary with poc, vah, val prices
        """
        try:
            if not self.redis:
                return {"poc": 0.0, "vah": 0.0, "val": 0.0}

            # Get historical price/volume data from Redis
            historical_data = await self._get_historical_data(symbol)

            if not historical_data or len(historical_data) < 10:
                logger.debug(f"Insufficient historical data for {symbol}")
                return {"poc": 0.0, "vah": 0.0, "val": 0.0}

            # Build volume distribution by price level
            volume_distribution = self._build_volume_distribution(historical_data)

            # Calculate POC (Point of Control - highest volume price level)
            poc = self._calculate_poc(volume_distribution)

            # Calculate Value Area (70% of volume)
            vah, val = self._calculate_value_area(volume_distribution)

            features = {
                "poc": poc,
                "vah": vah,
                "val": val,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }

            # Store in Redis (TTL: 1 hour) - Convert mapping to individual hset calls for compatibility
            profile_key = f"volume_profile:{symbol}"
            for field, value in features.items():
                await self.redis.hset(profile_key, field, str(value))
            await self.redis.expire(profile_key, 3600)

            # Cache locally
            self.volume_profiles[symbol] = features
            self.last_calculation[symbol] = datetime.now(timezone.utc)

            self.stats["profiles_calculated"] += 1

            logger.debug(f"Volume profile for {symbol}: POC=${poc:.2f}, VAH=${vah:.2f}, VAL=${val:.2f}")

        except Exception as e:
            logger.debug(f"Failed to calculate volume profile for {symbol}: {e}")
            self.stats["errors"] += 1
            self.stats["last_error"] = str(e)
            return {"poc": 0.0, "vah": 0.0, "val": 0.0}
        else:
            return features

    async def _get_historical_data(self, symbol: str) -> list[dict[str, float]]:
        """
        Get historical price/volume data from Redis klines

        Returns:
            List of {price, volume} dicts
        """
        try:
            if not self.redis:
                return []

            # Try to get from klines stored by market data service
            # Klines format: [timestamp, open, high, low, close, volume, ...]
            # Try 1m timeframe first, then 5m, then 15m
            parsed_data = []

            for timeframe in ["1m", "5m", "15m"]:
                klines_key = f"klines:{symbol}USDT:{timeframe}"
                klines_raw = await self.redis.get(klines_key)

                if klines_raw:
                    try:
                        klines = json.loads(klines_raw)
                        for kline in klines:
                            # kline format: [timestamp, open, high, low, close, volume, ...]
                            if isinstance(kline, list) and len(kline) >= 6:
                                close_price = float(kline[4])
                                volume = float(kline[5])
                                if close_price > 0 and volume > 0:
                                    parsed_data.append({"price": close_price, "volume": volume})
                    except (json.JSONDecodeError, ValueError, TypeError):
                        continue

                if parsed_data:
                    break  # Found data, use this timeframe

            if not parsed_data:
                logger.debug(f"No klines data found for {symbol}")

        except Exception as e:
            logger.debug(f"Error getting historical data for {symbol}: {e}")
            return []
        else:
            return parsed_data

    def _build_volume_distribution(self, historical_data: list[dict[str, float]]) -> dict[float, float]:
        """
        Build volume distribution across price levels

        Returns:
            Dictionary mapping price level to total volume
        """
        if not historical_data:
            return {}

        # Get price range
        prices = [d["price"] for d in historical_data]
        min_price = min(prices)
        max_price = max(prices)

        if max_price <= min_price:
            return {}

        # Create price bins
        price_step = (max_price - min_price) / self.price_bins
        volume_dist: dict[float, float] = {}

        # Aggregate volume by price bin
        for data in historical_data:
            price = data["price"]
            volume = data["volume"]

            # Find price bin
            bin_idx = int((price - min_price) / price_step)
            bin_idx = min(bin_idx, self.price_bins - 1)  # Cap at max bin
            bin_price = min_price + (bin_idx * price_step) + (price_step / 2)  # Bin center

            # Add volume to bin
            volume_dist[bin_price] = volume_dist.get(bin_price, 0.0) + volume

        return volume_dist

    def _calculate_poc(self, volume_distribution: dict[float, float]) -> float:
        """
        Calculate Point of Control (highest volume price level)

        Returns:
            Price level with highest volume
        """
        if not volume_distribution:
            return 0.0

        # Find price level with maximum volume
        poc_price = max(volume_distribution.items(), key=lambda x: x[1])[0]
        return poc_price

    def _calculate_value_area(self, volume_distribution: dict[float, float]) -> tuple[float, float]:
        """
        Calculate Value Area High (VAH) and Value Area Low (VAL)
        Value area contains 70% of total volume, centered around POC

        Returns:
            Tuple of (VAH, VAL)
        """
        if not volume_distribution:
            return (0.0, 0.0)

        # Sort by price
        sorted_levels = sorted(volume_distribution.items(), key=lambda x: x[0])

        # Calculate total volume
        total_volume = sum(v for _, v in sorted_levels)
        target_volume = total_volume * 0.70  # 70% value area

        # Find POC index
        poc_price = self._calculate_poc(volume_distribution)
        poc_idx = next((i for i, (p, _) in enumerate(sorted_levels) if p == poc_price), len(sorted_levels) // 2)

        # Expand from POC until we reach 70% of volume
        accumulated_volume = sorted_levels[poc_idx][1]
        low_idx = poc_idx
        high_idx = poc_idx

        # Expand in both directions, prioritizing higher volume
        while accumulated_volume < target_volume and (low_idx > 0 or high_idx < len(sorted_levels) - 1):
            # Get volumes at boundaries
            low_volume = sorted_levels[low_idx - 1][1] if low_idx > 0 else 0
            high_volume = sorted_levels[high_idx + 1][1] if high_idx < len(sorted_levels) - 1 else 0

            # Expand toward higher volume
            if low_volume >= high_volume and low_idx > 0:
                low_idx -= 1
                accumulated_volume += sorted_levels[low_idx][1]
            elif high_idx < len(sorted_levels) - 1:
                high_idx += 1
                accumulated_volume += sorted_levels[high_idx][1]
            else:
                break

        # Get VAH and VAL prices
        val = sorted_levels[low_idx][0]
        vah = sorted_levels[high_idx][0]

        return (vah, val)

    async def get_volume_profile_features(self, symbol: str) -> dict[str, float] | None:
        """
        Get volume profile features for a symbol from Redis

        Args:
            symbol: Trading symbol (e.g., "BTC")

        Returns:
            Dictionary with poc, vah, val or None if not available
        """
        try:
            if not self.redis:
                return None

            profile_key = f"volume_profile:{symbol}"
            data = await self.redis.hgetall(profile_key)

            if not data:
                logger.debug(f"No volume profile data found for {symbol}")
                return None

            # Convert strings back to floats
            features = {
                "poc": float(data.get("poc", 0)),
                "vah": float(data.get("vah", 0)),
                "val": float(data.get("val", 0)),
            }

        except Exception as e:
            logger.debug(f"Failed to get volume profile features for {symbol}: {e}")
            return None
        else:
            return features

    async def get_stats(self) -> dict[str, Any]:
        """Get service statistics"""
        return {
            "is_running": self.is_running,
            "profiles_calculated": self.stats["profiles_calculated"],
            "errors": self.stats["errors"],
            "config": {
                "lookback_hours": self.lookback_hours,
                "update_interval": self.update_interval,
                "price_bins": self.price_bins,
            },
        }


# Singleton instance
volume_profile_service = VolumeProfileService()
