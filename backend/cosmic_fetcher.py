#!/usr/bin/env python3
"""
Tier 3: Mystic / Cosmic / Meta Signals
Handles trend confirmation and big-picture filters every 1 hour globally
Binance.US Top-10 universe only.

Top-10 symbols from trading_universe (live data, no hardcoded values).
"""

from __future__ import annotations

import asyncio
import json
import logging
import math
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Any

import httpx

logger = logging.getLogger(__name__)


@dataclass
class CosmicSignal:
    mystic_score: float | None
    solar_activity: float | None
    cosmic_timing_score: float | None
    earth_frequency_match: float | None
    timestamp: str


class CosmicFetcher:
    def __init__(self, redis_client: Any) -> None:
        self.redis_client = redis_client
        self.session: httpx.AsyncClient | None = None
        self.is_running = False
        self.config = {
            "mystic_fetch_interval": 3600,
            "solar_fetch_interval": 3600,
            "cache_ttl": 7200,
            "max_retries": 3,
            "retry_delay": 60,
        }
        self.last_fetch_times: dict[str, float] = {}
        self.noaa_base_url = "https://services.swpc.noaa.gov/json"
        logger.info("Cosmic Fetcher initialized for global signals")

    async def initialize(self) -> None:
        if not self.session:
            # provide a float timeout value
            self.session = httpx.AsyncClient(timeout=30.0)
        logger.info("Cosmic Fetcher initialized")

    async def close(self) -> None:
        if self.session:
            try:
                await self.session.aclose()
            except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
                logger.debug("Error closing HTTP session: %s", e)
            finally:
                self.session = None
        self.is_running = False
        logger.info("Cosmic Fetcher closed")

    def _should_fetch(self, signal_type: str) -> bool:
        now = time.time()
        last = self.last_fetch_times.get(signal_type, 0.0)
        interval = self.config.get(f"{signal_type}_fetch_interval", 3600)
        try:
            interval = float(interval)
        except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError):
            interval = 3600.0
        return (now - last) >= interval

    def _update_fetch_time(self, signal_type: str) -> None:
        self.last_fetch_times[signal_type] = time.time()

    async def fetch_mystic_score(self) -> float | None:
        if not self._should_fetch("mystic"):
            return None
        try:
            now = datetime.now(timezone.utc)
            lunar = self._calculate_lunar_phase(now)
            solar_pos = self._calculate_solar_position(now)
            align = self._calculate_planetary_alignment(now)
            time_energy = self._calculate_time_energy(now)
            score = (lunar * 0.25) + (solar_pos * 0.30) + (align * 0.25) + (time_energy * 0.20)
            # score expected 0..1, convert to percentage 0..100
            score = max(0.0, min(100.0, score * 100.0))
            val = round(score, 2)
            self._update_fetch_time("mystic")
        except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
            logger.exception("Error calculating mystic score: %s", e)
            return None
        else:
            return val

    async def fetch_solar_activity(self) -> float | None:
        if not self._should_fetch("solar"):
            return None
        try:
            if not self.session:
                # session not initialized
                logger.debug("HTTP session not initialized, cannot fetch solar activity")
                return None
            url = f"{self.noaa_base_url}/goes/primary/xrays-1-day.json"
            resp = await self.session.get(url)
            if resp.status_code != 200:
                logger.warning("NOAA X-ray endpoint returned HTTP %s", resp.status_code)
                return None
            # resp.json() is synchronous but acceptable here
            data = resp.json()
            if not isinstance(data, list) or not data:
                return None
            latest = data[-1]
            flux = latest.get("flux", 0) or 0
            try:
                flux_val = float(flux)
            except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError):
                flux_val = 0.0
            # Normalize to 0..10 range (approx)
            index = max(0.0, min(10.0, flux_val / 1000.0))
            val = round(index, 2)
            self._update_fetch_time("solar")
        except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
            logger.warning("NOAA API failure: %s", e)
            return None
        else:
            return val

    def _calculate_lunar_phase(self, dt: datetime) -> float:
        # approximate lunar phase between 0 and 1
        start_of_year = datetime(dt.year, 1, 1, tzinfo=timezone.utc)
        days_since_new = ((dt - start_of_year).days % 29) + (dt.day % 2) * 0.5
        phase = (days_since_new / 29.530588) * 2.0 * math.pi
        return abs(math.sin(phase))

    def _calculate_solar_position(self, dt: datetime) -> float:
        return max(0.0, 1.0 - abs(dt.hour - 12) / 12.0)

    def _calculate_planetary_alignment(self, dt: datetime) -> float:
        doy = dt.timetuple().tm_yday
        return (math.sin(doy * 0.017) + 1.0) / 2.0

    def _calculate_time_energy(self, dt: datetime) -> float:
        hour_energy = abs(math.sin(dt.hour * math.pi / 12.0))
        day_energy_map = {0: 0.8, 1: 0.9, 2: 0.7, 3: 0.6, 4: 0.5, 5: 0.4, 6: 0.3}
        day_energy = day_energy_map.get(dt.weekday(), 0.5)
        return (hour_energy + day_energy) / 2.0

    async def calculate_cosmic_timing_score(self, mystic_score: float | None, solar_activity: float | None) -> float | None:
        try:
            if mystic_score is None or solar_activity is None:
                return None
            mystic_norm = mystic_score / 100.0
            solar_norm = solar_activity / 10.0
            timing = (mystic_norm * 0.6) + (solar_norm * 0.4)
            return round(timing * 100.0, 2)
        except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
            logger.exception("Error calculating cosmic timing score: %s", e)
            return None

    async def calculate_earth_frequency_match(self) -> float | None:
        try:
            # Placeholder for future implementation; currently unknown
            return None
        except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
            logger.exception("Error calculating Earth frequency match: %s", e)
            return None

    async def _cache_cosmic_data(self, data: dict[str, Any]) -> None:
        if not self.redis_client:
            logger.debug("No redis client provided; skipping cache")
            return
        try:
            serialized = json.dumps(data)
            ttl = int(self.config.get("cache_ttl", 7200))
            setex = getattr(self.redis_client, "setex", None)
            if callable(setex):
                try:
                    result = setex("cosmic_signals", ttl, serialized)
                    # if the result is a coroutine, await it
                    if asyncio.iscoroutine(result):
                        await result
                except TypeError:
                    # some redis clients expect different arg order or keywords; try a fallback
                    try:
                        result = setex("cosmic_signals", serialized, ttl)
                        if asyncio.iscoroutine(result):
                            await result
                    except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
                        logger.debug("Fallback setex failed: %s", e)
            else:
                # try generic set with expire
                set_fn = getattr(self.redis_client, "set", None)
                expire_fn = getattr(self.redis_client, "expire", None)
                if callable(set_fn):
                    result = set_fn("cosmic_signals", serialized)
                    if asyncio.iscoroutine(result):
                        await result
                    if callable(expire_fn):
                        exp_res = expire_fn("cosmic_signals", ttl)
                        if asyncio.iscoroutine(exp_res):
                            await exp_res
        except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
            logger.exception("Error caching cosmic data: %s", e)

    async def fetch_all_tier3_signals(self) -> dict[str, Any]:
        results: dict[str, Any] = {
            "cosmic_signals": {},
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        try:
            mystic_score = await self.fetch_mystic_score()
            solar_activity = await self.fetch_solar_activity()
            cosmic_timing = await self.calculate_cosmic_timing_score(mystic_score, solar_activity)
            earth_frequency = await self.calculate_earth_frequency_match()

            cosmic_data = CosmicSignal(
                mystic_score=mystic_score,
                solar_activity=solar_activity,
                cosmic_timing_score=cosmic_timing,
                earth_frequency_match=earth_frequency,
                timestamp=datetime.now(timezone.utc).isoformat(),
            )
            results["cosmic_signals"] = asdict(cosmic_data)

            await self._cache_cosmic_data(results)
            logger.info(
                "Fetched cosmic signals: mystic=%s solar=%s timing=%s",
                mystic_score,
                solar_activity,
                cosmic_timing,
            )
        except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
            logger.exception("Error fetching Tier 3 signals: %s", e)
        return results

    async def run(self) -> None:
        logger.info("Starting Tier 3 Cosmic Fetcher (global signals)")
        self.is_running = True
        try:
            await self.initialize()
            while self.is_running:
                try:
                    signals = await self.fetch_all_tier3_signals()
                    if signals.get("cosmic_signals"):
                        logger.info("Updated global cosmic signals")
                    # Mystic fetch interval from config (typically hourly)
                    await asyncio.sleep(float(self.config.get("mystic_fetch_interval", 3600)))
                except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
                    logger.exception("Error in cosmic fetcher loop: %s", e)
                    # Retry delay from config before next attempt
                    await asyncio.sleep(float(self.config.get("retry_delay", 60)))
        except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
            logger.exception("Fatal error in cosmic fetcher: %s", e)
        finally:
            await self.close()

    def get_status(self) -> dict[str, Any]:
        return {
            "status": "running" if self.is_running else "stopped",
            "config": self.config,
            "last_fetch_times": self.last_fetch_times,
            "signal_type": "global",
        }
