#!/usr/bin/env python3
"""
Experimental Services Integration
Integrates quantum, blockchain, satellite, and 5G services with autobuy decisions
"""

from __future__ import annotations

import asyncio
import logging
import os
from datetime import datetime, timezone
from typing import Any

import httpx

# Import from single source of truth
try:
    from backend.config.trading_universe import TOP10_COINS
except (ImportError, ModuleNotFoundError, AttributeError, ValueError, TypeError, RuntimeError) as e:
    msg = f"Failed to import trading_universe: {e}"
    raise RuntimeError(msg) from e

logger = logging.getLogger(__name__)

# Use TOP10_COINS from trading_universe (live data)
TOP10 = set(TOP10_COINS)


class ExperimentalIntegration:
    def __init__(self) -> None:
        self.is_running: bool = False

        # Allow service endpoints to be overridden via environment variables
        self.service_endpoints: dict[str, str] = {
            "quantum": os.getenv("SERVICE_QUANTUM_URL", "http://quantum-trading-engine-new:8087"),
            "blockchain": os.getenv("SERVICE_BLOCKCHAIN_URL", "http://bitcoin-miner:8084"),
            "satellite": os.getenv("SERVICE_SATELLITE_URL", "http://satellite-analytics:8085"),
            "5g": os.getenv("SERVICE_5G_URL", "http://fiveg-core:8086"),
        }

        self.integration_weights: dict[str, float] = {
            "quantum": 0.25,
            "blockchain": 0.20,
            "satellite": 0.20,
            "5g": 0.15,
        }

        self.service_status: dict[str, dict[str, Any]] = {}
        self.last_integration: datetime | None = None

        self.integration_cache: dict[str, Any] = {}
        self.cache_ttl_seconds: int = int(os.getenv("EXPERIMENTAL_CACHE_TTL", "300"))

        logger.info("Experimental Services Integration initialized")

    async def start(self) -> None:
        self.is_running = True
        logger.info("Starting Experimental Services Integration")
        while self.is_running:
            try:
                await self.collect_experimental_data(force_refresh=False)
            except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
                logger.exception(f"Error in experimental integration loop: {e}")
            await asyncio.sleep(60)

    async def stop(self) -> None:
        self.is_running = False
        logger.info("Experimental Services Integration stopped")

    async def collect_experimental_data(self, force_refresh: bool = False) -> None:
        try:
            if not force_refresh and self._cache_fresh():
                return

            # Fetch in parallel with per-request retries
            (
                quantum_data,
                blockchain_data,
                satellite_data,
                g5_data,
            ) = await asyncio.gather(
                self._collect_with_retries(self._collect_quantum_data),
                self._collect_with_retries(self._collect_blockchain_data),
                self._collect_with_retries(self._collect_satellite_data),
                self._collect_with_retries(self._collect_5g_data),
            )

            current_time = datetime.now(timezone.utc)
            combined = await self._combine_experimental_signals(quantum_data, blockchain_data, satellite_data, g5_data)

            self.integration_cache = {
                "timestamp": current_time.isoformat(),
                "quantum": quantum_data,
                "blockchain": blockchain_data,
                "satellite": satellite_data,
                "5g": g5_data,
                "combined_signals": combined,
            }
            self.last_integration = current_time
        except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
            logger.exception(f"Error collecting experimental data: {e}")

    def _cache_fresh(self) -> bool:
        if not self.last_integration:
            return False
        age = datetime.now(timezone.utc) - self.last_integration
        return age.total_seconds() < self.cache_ttl_seconds

    async def _collect_with_retries(self, fn, retries: int = 2, backoff: float = 0.5) -> dict[str, Any]:
        last_exc: Exception | None = None
        for attempt in range(retries + 1):
            try:
                return await fn()
            except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
                last_exc = e
                await asyncio.sleep(backoff * (2**attempt))
        logger.warning(f"Service collection failed after retries: {last_exc}")
        return {}

    async def _collect_quantum_data(self) -> dict[str, Any]:
        url = f"{self.service_endpoints['quantum']}/status"
        return await self._fetch_json("quantum", url)

    async def _collect_blockchain_data(self) -> dict[str, Any]:
        url = f"{self.service_endpoints['blockchain']}/status"
        return await self._fetch_json("blockchain", url)

    async def _collect_satellite_data(self) -> dict[str, Any]:
        url = f"{self.service_endpoints['satellite']}/status"
        return await self._fetch_json("satellite", url)

    async def _collect_5g_data(self) -> dict[str, Any]:
        url = f"{self.service_endpoints['5g']}/status"
        return await self._fetch_json("5g", url)

    async def _fetch_json(self, key: str, url: str) -> dict[str, Any]:
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                r = await client.get(url)
            if r.status_code == 200:
                data = r.json()
                self.service_status[key] = {"status": "online", "data": data}
                result = data
            else:
                self.service_status[key] = {
                    "status": "offline",
                    "error": f"HTTP {r.status_code}",
                }
                result = {}
        except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
            self.service_status[key] = {"status": "offline", "error": str(e)}
            logger.warning(f"{key} service unavailable: {e}")
            return {}
        else:
            return result

    async def _combine_experimental_signals(
        self,
        quantum_data: dict[str, Any],
        blockchain_data: dict[str, Any],
        satellite_data: dict[str, Any],
        g5_data: dict[str, Any],
    ) -> dict[str, Any]:
        try:
            combined = {
                "overall_signal": "NEUTRAL",
                "confidence": 0.5,
                "strength": 0.5,
                "risk_level": "MEDIUM",
                "recommendation": "HOLD",
                "service_contributions": {},
            }

            total_weight = 0.0
            w_conf = 0.0
            w_str = 0.0
            risk_factors: list[str] = []

            if quantum_data and self.service_status.get("quantum", {}).get("status") == "online":
                q = self._process_quantum_signal(quantum_data)
                w = self.integration_weights.get("quantum", 0.0)
                total_weight += w
                w_conf += q.get("confidence", 0.5) * w
                w_str += q.get("strength", 0.5) * w
                risk_factors.extend(q.get("risk_factors", []))
                combined["service_contributions"]["quantum"] = q

            if blockchain_data and self.service_status.get("blockchain", {}).get("status") == "online":
                b = self._process_blockchain_signal(blockchain_data)
                w = self.integration_weights.get("blockchain", 0.0)
                total_weight += w
                w_conf += b.get("confidence", 0.5) * w
                w_str += b.get("strength", 0.5) * w
                risk_factors.extend(b.get("risk_factors", []))
                combined["service_contributions"]["blockchain"] = b

            if satellite_data and self.service_status.get("satellite", {}).get("status") == "online":
                s = self._process_satellite_signal(satellite_data)
                w = self.integration_weights.get("satellite", 0.0)
                total_weight += w
                w_conf += s.get("confidence", 0.5) * w
                w_str += s.get("strength", 0.5) * w
                risk_factors.extend(s.get("risk_factors", []))
                combined["service_contributions"]["satellite"] = s

            if g5_data and self.service_status.get("5g", {}).get("status") == "online":
                g = self._process_5g_signal(g5_data)
                w = self.integration_weights.get("5g", 0.0)
                total_weight += w
                w_conf += g.get("confidence", 0.5) * w
                w_str += g.get("strength", 0.5) * w
                risk_factors.extend(g.get("risk_factors", []))
                combined["service_contributions"]["5g"] = g

            # Compute aggregated metrics
            if total_weight > 0:
                agg_conf = max(0.0, min(1.0, w_conf / total_weight))
                agg_str = max(0.0, min(1.0, w_str / total_weight))
            else:
                agg_conf = 0.5
                agg_str = 0.5

            combined["confidence"] = agg_conf
            combined["strength"] = agg_str

            score = agg_conf * agg_str

            if score > 0.6:
                overall = "BUY"
                recommendation = "BUY"
            elif score < 0.35:
                overall = "SELL"
                recommendation = "SELL"
            else:
                overall = "HOLD"
                recommendation = "HOLD"

            combined["overall_signal"] = overall
            combined["recommendation"] = recommendation

            unique_risks = sorted(set(risk_factors))
            combined["risk_factors"] = unique_risks

            # Risk level heuristic
            if len(unique_risks) == 0:
                risk_level = "LOW"
            elif len(unique_risks) <= 2:
                risk_level = "MEDIUM"
            else:
                risk_level = "HIGH"
            combined["risk_level"] = risk_level

            active_services = len(combined["service_contributions"])
            combined["active_services"] = active_services
        except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
            logger.exception(f"Error combining experimental signals: {e}")
            return {
                "overall_signal": "NEUTRAL",
                "confidence": 0.5,
                "strength": 0.5,
                "risk_level": "MEDIUM",
                "recommendation": "HOLD",
                "service_contributions": {},
                "active_services": 0,
                "risk_factors": [],
            }
        else:
            return combined

    def _process_quantum_signal(self, data: dict[str, Any]) -> dict[str, Any]:
        try:
            # Example fields: qubit_count, circuit_depth, error_rate, quantum_advantage_score
            qubit_count = int(data.get("qubit_count", 0) or 0)
            circuit_depth = int(data.get("circuit_depth", 0) or 0)
            error_rate = float(data.get("error_rate", 0.0) or 0.0)
            quantum_advantage = float(data.get("quantum_advantage", 0.0) or 0.0)

            # Confidence increases with qubit count and advantage, decreases with error rate
            conf_qubits = min(1.0, qubit_count / 1000.0)
            conf_adv = min(1.0, quantum_advantage)
            confidence = max(0.0, min(1.0, (conf_qubits * 0.6) + (conf_adv * 0.4) - (error_rate * 0.5)))

            # Strength approximated by circuit depth normalized
            strength = max(0.0, min(1.0, circuit_depth / 1000.0))

            risks: list[str] = []
            if error_rate > 0.05:
                risks.append("high_error_rate")
            if qubit_count < 50:
                risks.append("low_qubit_count")
            if quantum_advantage < 0.1:
                risks.append("low_quantum_advantage")

            result = {
                "signal": "BUY" if confidence > 0.6 else "HOLD",
                "confidence": confidence,
                "strength": strength,
                "risk_factors": risks,
                "metrics": {
                    "qubit_count": qubit_count,
                    "circuit_depth": circuit_depth,
                    "error_rate": error_rate,
                    "quantum_advantage": quantum_advantage,
                },
            }
        except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
            logger.exception(f"Error processing quantum signal: {e}")
            return {
                "signal": "HOLD",
                "confidence": 0.5,
                "strength": 0.5,
                "risk_factors": ["processing_error"],
            }
        else:
            return result

    def _process_blockchain_signal(self, data: dict[str, Any]) -> dict[str, Any]:
        try:
            hash_rate = float(data.get("hash_rate", 0))
            difficulty = float(data.get("difficulty", 0))
            block_time = float(data.get("block_time", 0) or 1)

            max(0.0, min(1.0, (hash_rate / 1_000_000.0) * (difficulty / 1_000_000.0)))
            max(0.0, min(1.0, 600.0 / max(block_time, 1.0)))

            risks: list[str] = []
            if hash_rate < 100_000:
                risks.append("low_hash_rate")
            if block_time > 1200:
                risks.append("slow_block_time")

        except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
            logger.exception(f"Error processing blockchain signal: {e}")
            return {
                "signal": "HOLD",
                "confidence": 0.5,
                "strength": 0.5,
                "risk_factors": ["processing_error"],
            }

    def _process_satellite_signal(self, data: dict[str, Any]) -> dict[str, Any]:
        try:
            signal_strength = float(data.get("signal_strength", 0.0))
            data_quality = float(data.get("data_quality", 0.0))
            coverage_area = float(data.get("coverage_area", 0.0))

            max(0.0, min(1.0, (signal_strength + data_quality) / 2.0))
            max(0.0, min(1.0, coverage_area / 100.0))

            risks: list[str] = []
            if signal_strength < 0.5:
                risks.append("weak_signal")
            if data_quality < 0.7:
                risks.append("poor_data_quality")

        except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
            logger.exception(f"Error processing satellite signal: {e}")
            return {
                "signal": "HOLD",
                "confidence": 0.5,
                "strength": 0.5,
                "risk_factors": ["processing_error"],
            }

    def _process_5g_signal(self, data: dict[str, Any]) -> dict[str, Any]:
        try:
            bandwidth = float(data.get("bandwidth", 0))
            latency = float(data.get("latency", 1000) or 1)
            connection_count = float(data.get("connection_count", 0))

            max(0.0, min(1.0, (bandwidth / 1000.0) * (1000.0 / max(latency, 1.0))))
            max(0.0, min(1.0, connection_count / 10_000.0))

            risks: list[str] = []
            if latency > 50:
                risks.append("high_latency")
            if bandwidth < 100:
                risks.append("low_bandwidth")

        except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
            logger.exception(f"Error processing 5G signal: {e}")
            return {
                "signal": "HOLD",
                "confidence": 0.5,
                "strength": 0.5,
                "risk_factors": ["processing_error"],
            }

    async def get_experimental_influence(self, symbol: str) -> dict[str, Any]:
        try:
            # Enforce allowed coins (base symbol only)
            base = symbol.replace("-", "").replace("/", "")
            if base.endswith("USDT"):
                base = base[:-4]
            elif base.endswith("USD"):
                base = base[:-3]
            base = base.upper()
            if base not in TOP10:
                return {
                    "influence": 0.0,
                    "recommendation": "HOLD",
                    "reason": f"Symbol '{symbol}' not in supported list",
                }

            if not self.integration_cache or not self._cache_fresh():
                await self.collect_experimental_data(force_refresh=True)

            combined = self.integration_cache.get("combined_signals", {}) if self.integration_cache else {}
            raw_conf = float(combined.get("confidence", 0.5) or 0.5)
            try:
                from backend.services.confidence_normalizer import ConfidenceNormalizer

                conf = ConfidenceNormalizer.normalize(raw_conf)
            except Exception:
                conf = raw_conf
            influence = conf * float(combined.get("strength", 0.5) or 0.5)

            return {
                "symbol": symbol,
                "influence": influence,
                "overall_signal": combined.get("overall_signal", "NEUTRAL"),
                "recommendation": combined.get("recommendation", "HOLD"),
                "confidence": conf,
                "strength": combined.get("strength", 0.5),
                "risk_level": combined.get("risk_level", "MEDIUM"),
                "active_services": combined.get("active_services", 0),
                "service_contributions": combined.get("service_contributions", {}),
                "risk_factors": combined.get("risk_factors", []),
                "timestamp": self.integration_cache.get("timestamp"),
            }
        except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
            logger.exception(f"Error getting experimental influence: {e}")
            return {"influence": 0.0, "recommendation": "HOLD", "reason": str(e)}

    def get_service_status(self) -> dict[str, Any]:
        return {
            "services": self.service_status,
            "integration_weights": self.integration_weights,
            "last_integration": (self.last_integration.isoformat() if self.last_integration else None),
            "active_services": sum(1 for s in self.service_status.values() if s.get("status") == "online"),
            "total_services": len(self.service_status),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

    def get_status(self) -> dict[str, Any]:
        return {
            "is_running": self.is_running,
            "service_status": self.get_service_status(),
            "integration_weights": self.integration_weights,
            "cache_ttl_seconds": self.cache_ttl_seconds,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }


# Experimental integration state - using dict to avoid global keyword
_experimental_integration_state: dict[str, ExperimentalIntegration | None] = {"instance": None}


def get_experimental_integration() -> ExperimentalIntegration:
    if _experimental_integration_state["instance"] is None:
        _experimental_integration_state["instance"] = ExperimentalIntegration()
    return _experimental_integration_state["instance"]
