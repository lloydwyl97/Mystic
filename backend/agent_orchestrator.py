"""
Agent Orchestrator
Coordinates all AI agents in the Mystic AI Trading Platform.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from backend.config.redis_config import get_redis_client
from backend.services.task_manager import task_manager

# Minimal, clean logger
logger = logging.getLogger(__name__)

# Redis configuration - All Live Data, No Fallback/Hardcoded Data
REDIS_URL = os.getenv("REDIS_URL")
if not REDIS_URL:
    redis_host = os.getenv("REDIS_HOST")
    if not redis_host:
        # Redis not configured - will fail gracefully in _init_redis
        REDIS_URL = None
    else:
        redis_port = os.getenv("REDIS_PORT", "6379")
        redis_db = os.getenv("REDIS_DB", "0")
        REDIS_URL = f"redis://{redis_host}:{redis_port}/{redis_db}"

# Agent orchestrator timing constants
AGENT_PROCESS_LOOP_INTERVAL = 1.0  # Agent process loop check interval
AGENT_COORDINATION_INTERVAL = 1.0  # Agent coordination check interval
ORCHESTRATOR_HEALTH_CHECK_INTERVAL = 30.0  # Health check frequency
ORCHESTRATOR_ERROR_RECOVERY_DELAY = 60.0  # Delay after orchestrator errors

# Optional path add (kept minimal to avoid side effects when running as a module)
BASE_DIR = str(Path(__file__).resolve().parent.parent)
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

# Optional MemoryAgent import; provides a tiny fallback if not present
try:
    # Adjust if your MemoryAgent lives elsewhere
    from backend.agents.memory_agent import MemoryAgent as _ExternalMemoryAgent  # type: ignore[import-not-found]

    MemoryAgent = _ExternalMemoryAgent
except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError):

    class MemoryAgent:
        pass


# Lazy imports for optional Phase 5 agents (may not be available in all deployments)
try:
    from backend.agents.auranet_channel_interface import AuraNetChannelInterface  # type: ignore[import-not-found]
except (ImportError, ModuleNotFoundError, AttributeError, ValueError, TypeError, RuntimeError):
    AuraNetChannelInterface = None  # type: ignore[assignment, misc]

try:
    from backend.agents.cosmic_pattern_recognizer import CosmicPatternRecognizer  # type: ignore[import-not-found]
except (ImportError, ModuleNotFoundError, AttributeError, ValueError, TypeError, RuntimeError):
    CosmicPatternRecognizer = None  # type: ignore[assignment, misc]

try:
    from backend.agents.interdimensional_signal_decoder import InterdimensionalSignalDecoder  # type: ignore[import-not-found]
except (ImportError, ModuleNotFoundError, AttributeError, ValueError, TypeError, RuntimeError):
    InterdimensionalSignalDecoder = None  # type: ignore[assignment, misc]

try:
    from backend.agents.neuro_synchronization_engine import NeuroSynchronizationEngine  # type: ignore[import-not-found]
except (ImportError, ModuleNotFoundError, AttributeError, ValueError, TypeError, RuntimeError):
    NeuroSynchronizationEngine = None  # type: ignore[assignment, misc]


class MemoryAgent:
    def __init__(self, agent_id: str) -> None:
        self.id = agent_id
        self.memory: list[str] = []
        self.state = "idle"

    async def start(self) -> None:
        self.state = "running"

    async def stop(self) -> None:
        self.state = "stopped"

    async def process_loop(self) -> None:
        while self.state == "running":
            # Regular process loop check - allows clean shutdown
            await asyncio.sleep(AGENT_PROCESS_LOOP_INTERVAL)

    async def get_status(self) -> dict[str, Any]:
        return {"status": self.state, "memory_items": len(self.memory)}

    async def handle_message(self, message: dict[str, Any]) -> None:
        self.memory.append(str(message))


class AgentOrchestrator:
    """Main orchestrator for all AI agents."""

    def __init__(self) -> None:
        self.agents: dict[str, Any] = {}
        self.running: bool = False
        self.redis_client: Any = None
        self.agent_tasks: list[asyncio.Task] = []

    async def _init_redis(self) -> None:
        if self.redis_client is None:
            try:
                # Use shared Redis pool
                self.redis_client = get_redis_client()
                # ping is a coroutine and should be awaited
                await self.redis_client.ping()
                logger.info("Redis connection established")
            except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
                logger.warning("Redis not available: %s", e)
                self.redis_client = None

    async def initialize_agents(self) -> None:
        try:
            logger.info("Initializing AI agents")
            await self._init_redis()
            await self.initialize_phase5_agents()
            await self.initialize_other_agents()

            # Always register at least a simple memory agent
            if "memory_agent" not in self.agents:
                self.agents["memory_agent"] = MemoryAgent("memory_agent")

            logger.info("Initialized %d agents", len(self.agents))
        except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError):
            logger.exception("Error initializing agents")

    async def initialize_phase5_agents(self) -> None:
        try:
            # Phase 5 agents are optional; use if available
            if InterdimensionalSignalDecoder is not None:
                self.agents["interdimensional_signal_decoder"] = InterdimensionalSignalDecoder()  # type: ignore[misc]
            if NeuroSynchronizationEngine is not None:
                self.agents["neuro_synchronization_engine"] = NeuroSynchronizationEngine()  # type: ignore[misc]
            if CosmicPatternRecognizer is not None:
                self.agents["cosmic_pattern_recognizer"] = CosmicPatternRecognizer()  # type: ignore[misc]
            if AuraNetChannelInterface is not None:
                self.agents["auranet_channel_interface"] = AuraNetChannelInterface()  # type: ignore[misc]

            logger.info("Phase 5 agents initialized")
        except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
            logger.info("Phase 5 agents not available: %s", e)

    async def initialize_other_agents(self) -> None:
        try:
            # Optional modules; load if available
            agent_modules = [
                "agents.strategy_agent",
                "agents.market_sentiment_agent",
                "agents.news_sentiment_agent",
                "agents.social_media_agent",
                "agents.chart_pattern_agent",
                "agents.technical_indicator_agent",
                "agents.market_visualization_agent",
                "agents.deep_learning_agent",
                "agents.reinforcement_learning_agent",
                "agents.ai_model_manager",
                "agents.quantum_algorithm_engine",
                "agents.quantum_machine_learning_agent",
                "agents.quantum_optimization_agent",
            ]

            for module_name in agent_modules:
                try:
                    module = __import__(module_name, fromlist=["*"])
                    for attr_name in dir(module):
                        attr = getattr(module, attr_name)
                        bases = getattr(attr, "__bases__", ())
                        if bases and any(getattr(b, "__name__", "") == "BaseAgent" for b in bases):
                            try:
                                agent_instance = attr()
                                agent_id = getattr(agent_instance, "agent_id", attr.__name__.lower())
                                self.agents[agent_id] = agent_instance
                                logger.info("Loaded agent: %s", agent_id)
                                break
                            except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as inst_exc:
                                logger.info("Failed to instantiate agent class %s: %s", attr_name, inst_exc)
                except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
                    logger.info("Module not loaded %s: %s", module_name, e)
        except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
            logger.info("Other agents not available: %s", e)

    async def start_agents(self) -> None:
        try:
            logger.info("Starting all agents")
            self.running = True

            for agent_id, agent in self.agents.items():
                try:
                    if hasattr(agent, "start"):
                        await agent.start()
                    elif hasattr(agent, "initialize"):
                        await agent.initialize()

                    task = await task_manager.create_task(self.run_agent_loop(agent), name="agent_orchestrator:run_agent_loop")
                    self.agent_tasks.append(task)
                    logger.info("Started agent: %s", agent_id)
                except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError):
                    logger.exception("Failed to start agent %s", agent_id)

            logger.info("Started %d agent tasks", len(self.agent_tasks))
        except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError):
            logger.exception("Error starting agents")

    async def run_agent_loop(self, agent: Any) -> None:
        try:
            if hasattr(agent, "process_loop"):
                await agent.process_loop()
                return
            if hasattr(agent, "run"):
                await agent.run()
                return
            # Keep agent alive while orchestrator is running
            while self.running:
                await asyncio.sleep(AGENT_COORDINATION_INTERVAL)
        except asyncio.CancelledError:
            raise
        except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError):
            agent_id = getattr(agent, "agent_id", getattr(agent, "id", "unknown"))
            logger.exception("Agent loop error for %s", agent_id)

    async def stop_agents(self) -> None:
        try:
            logger.info("Stopping all agents")
            self.running = False

            for agent_id, agent in self.agents.items():
                try:
                    if hasattr(agent, "stop"):
                        await agent.stop()
                    logger.info("Stopped agent: %s", agent_id)
                except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError):
                    logger.exception("Failed to stop agent %s", agent_id)

            for task in self.agent_tasks:
                task.cancel()
            if self.agent_tasks:
                await asyncio.gather(*self.agent_tasks, return_exceptions=True)
            self.agent_tasks.clear()

            # Close redis connection if present
            try:
                if self.redis_client:
                    await self.redis_client.close()
            except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError):
                # Closures are best-effort, don't let them block shutdown
                pass

            logger.info("All agents stopped")
        except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError):
            logger.exception("Error stopping agents")

    async def get_agent_status(self) -> dict[str, Any]:
        try:
            status: dict[str, Any] = {}
            processed = 0
            for agent_id, agent in self.agents.items():
                try:
                    if hasattr(agent, "get_status"):
                        status[agent_id] = await agent.get_status()
                    elif hasattr(agent, "state"):
                        status[agent_id] = {
                            "status": "running" if self.running else "stopped",
                            "state": getattr(agent, "state", None),
                            "last_update": datetime.now(timezone.utc).isoformat(),
                        }
                    else:
                        status[agent_id] = {
                            "status": "running" if self.running else "stopped",
                            "last_update": datetime.now(timezone.utc).isoformat(),
                        }
                    processed += 1
                except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
                    status[agent_id] = {
                        "status": "error",
                        "error": str(e),
                        "last_update": datetime.now(timezone.utc).isoformat(),
                    }

            if processed != len(self.agents):
                status["processing_warning"] = f"Only {processed}/{len(self.agents)} agents processed"
        except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError):
            logger.exception("Error getting agent status")
            return {}
        else:
            return status

    async def broadcast_message(self, message: dict[str, Any]) -> None:
        try:
            for agent_id, agent in self.agents.items():
                try:
                    if hasattr(agent, "handle_message"):
                        await agent.handle_message(message)
                except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError):
                    logger.exception("Failed to send message to %s", agent_id)
        except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError):
            logger.exception("Error broadcasting message")

    async def run(self) -> None:
        try:
            logger.info("Agent Orchestrator starting")
            await self.initialize_agents()
            await self.start_agents()

            while self.running:
                try:
                    await self.update_orchestrator_status()
                    await self.check_agent_health()
                    # Regular health check interval
                    await asyncio.sleep(ORCHESTRATOR_HEALTH_CHECK_INTERVAL)
                except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError):
                    logger.exception("Orchestrator loop error")
                    # Wait longer after errors to avoid rapid failure loops
                    await asyncio.sleep(ORCHESTRATOR_ERROR_RECOVERY_DELAY)
        except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError):
            logger.exception("Orchestrator run error")
        finally:
            await self.stop_agents()

    async def update_orchestrator_status(self) -> None:
        try:
            if self.redis_client:
                status = {
                    "orchestrator_status": "running" if self.running else "stopped",
                    "agent_count": len(self.agents),
                    "running_tasks": len(self.agent_tasks),
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                }
                await self.redis_client.set("orchestrator_status", json.dumps(status), ex=300)
        except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError):
            logger.exception("Error updating orchestrator status")

    async def check_agent_health(self) -> None:
        try:
            for agent_id, agent in self.agents.items():
                try:
                    health = getattr(agent, "health_status", None)
                    if health == "error":
                        logger.warning("Agent %s has health issues", agent_id)
                except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError):
                    logger.exception("Health check failed for %s", agent_id)
        except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError):
            logger.exception("Error checking agent health")


# Global orchestrator instance
orchestrator = AgentOrchestrator()


async def main() -> None:
    try:
        await orchestrator.run()
    except KeyboardInterrupt:
        logger.info("Orchestrator interrupted by user")
        await orchestrator.stop_agents()
    except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError):
        logger.exception("Orchestrator main error")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(main())

# Quick test checklist:
# - ccxt calls only receive BASE/QUOTE (N/A here).
# - No binance/binanceus string leaks—only central constants elsewhere.
# - No unreachable code after returns.
# - Logging has no weird characters.
