#!/usr/bin/env python3
"""
AI Agent Orchestrator Service
Port 8006 - Orchestrates all AI agents in a single service
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import redis.asyncio as redis
import uvicorn
from fastapi import FastAPI, HTTPException

from backend.services.task_manager import task_manager

# Backend imports are handled through proper package structure

# import backend.ai as ai agents (optional; don't crash if missing)
try:
    from backend.agents.advanced_ai_orchestrator import AdvancedAIOrchestrator  # type: ignore[import-not-found]
except (ImportError, ModuleNotFoundError):
    AdvancedAIOrchestrator = None  # type: ignore[assignment]
try:
    from backend.agents.agent_orchestrator import AgentOrchestrator  # type: ignore[import-not-found]
except (ImportError, ModuleNotFoundError):
    AgentOrchestrator = None  # type: ignore[assignment]
try:
    from backend.agents.ai_model_manager import AIModelManager  # type: ignore[import-not-found]
except (ImportError, ModuleNotFoundError):
    AIModelManager = None  # type: ignore[assignment]

# Optional helper
try:
    from utils.redis_helpers import to_str
except (ImportError, ModuleNotFoundError):

    def to_str(v: Any) -> str | None:
        return None if v is None else (v if isinstance(v, str) else str(v))


# Logging setup
Path("logs").mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("logs/ai_agent_orchestrator.log", encoding="utf-8"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger("ai_agent_orchestrator_service")

# FastAPI app
app = FastAPI(
    title="AI Agent Orchestrator Service",
    description="Orchestrates all AI agents in a single service",
    version="1.0.0",
)

# Redis URL - All Live Data, No Fallback/Hardcoded Data
REDIS_URL = os.getenv("REDIS_URL")
if not REDIS_URL:
    redis_host = os.getenv("REDIS_HOST")
    if redis_host:
        redis_port = os.getenv("REDIS_PORT", "6379")
        redis_db = os.getenv("REDIS_DB", "0")
        REDIS_URL = f"redis://{redis_host}:{redis_port}/{redis_db}"
    else:
        # Redis not configured - will fail gracefully in initialize
        REDIS_URL = None


class AIAgentOrchestratorService:
    """AI Agent Orchestrator Service"""

    def __init__(self) -> None:
        self.running: bool = False
        self.agents: dict[str, Any] = {}
        self.agent_history: list[dict[str, Any]] = []
        self.redis_client: redis.Redis | None = None
        # Track background tasks for proper cleanup
        self._tasks: list[asyncio.Task[Any]] = []

    async def initialize(self) -> None:
        """Initialize dependencies and agents."""
        try:
            if not REDIS_URL:
                logger.warning("Redis URL not configured - Redis features disabled")
                self.redis_client = None
            else:
                # redis.from_url is not a coroutine; do not await it
                self.redis_client = redis.from_url(
                    REDIS_URL,
                    encoding="utf-8",
                    decode_responses=True,
                    protocol=2,  # CRITICAL: Use RESP2 to avoid CLIENT SETINFO issues on Windows Redis
                )  # type: ignore[attr-defined]
                # Ensure connection works
                try:
                    await self.redis_client.ping()
                    logger.info("Redis connection established")
                except (ConnectionError, OSError, AttributeError, TypeError) as e:
                    logger.warning("Redis not available after connect: %s", e)
                    self.redis_client = None
        except (ConnectionError, OSError, AttributeError, TypeError) as e:
            logger.warning("Redis not available: %s", e)
            self.redis_client = None

        # Initialize AI agents (only those we can import)
        if AIModelManager:
            try:
                self.agents["model_manager"] = AIModelManager()
                logger.info("[OK] AIModelManager initialized")
            except (AttributeError, TypeError, ValueError, RuntimeError) as e:
                logger.warning(f"AIModelManager unavailable: {e}")
        if AgentOrchestrator:
            try:
                self.agents["orchestrator"] = AgentOrchestrator()
                logger.info("[OK] AgentOrchestrator initialized")
            except (AttributeError, TypeError, ValueError, RuntimeError) as e:
                logger.warning(f"AgentOrchestrator unavailable: {e}")
        if AdvancedAIOrchestrator:
            try:
                self.agents["advanced_orchestrator"] = AdvancedAIOrchestrator()
                logger.info("[OK] AdvancedAIOrchestrator initialized")
            except (AttributeError, TypeError, ValueError, RuntimeError) as e:
                logger.warning(f"AdvancedAIOrchestrator unavailable: {e}")

        if not self.agents:
            logger.warning("No agents were initialized (optional modules missing). Service will still run.")
        else:
            logger.info("[OK] AI Agent Orchestrator Service initialized")

    async def _call_agent_method(self, agent: Any, method_name: str, *args, **kwargs) -> Any:
        """Call agent method and await if it returns a coroutine."""
        if not agent:
            msg = "Agent is not available"
            raise RuntimeError(msg)
        if not hasattr(agent, method_name):
            msg = f"Agent does not have method {method_name}"
            raise AttributeError(msg)
        method = getattr(agent, method_name)
        if not callable(method):
            msg = f"Agent attribute {method_name} is not callable"
            raise TypeError(msg)
        result = method(*args, **kwargs)
        if asyncio.iscoroutine(result):
            return await result
        return result

    async def start(self) -> None:
        """Start the service and agents."""
        logger.info("Starting AI Agent Orchestrator Service")
        self.running = True

        for agent_name, agent in self.agents.items():
            try:
                if hasattr(agent, "start"):
                    await self._call_agent_method(agent, "start")
                logger.info("Started agent: %s", agent_name)
            except (AttributeError, TypeError, ValueError, RuntimeError) as e:
                logger.exception("Error starting agent %s: %s", agent_name, e)

        task = await task_manager.create_task(self._agent_monitor_loop(), name="ai_agent_orchestrator_service:agent_monitor_loop")
        self._tasks.append(task)

    async def stop(self) -> None:
        """Stop the service and agents."""
        logger.info("Stopping AI Agent Orchestrator Service")
        self.running = False
        # Cancel all background tasks
        for task in self._tasks:
            if not task.done():
                task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await task
        self._tasks.clear()

        for agent_name, agent in self.agents.items():
            try:
                if hasattr(agent, "stop"):
                    await self._call_agent_method(agent, "stop")
                logger.info("Stopped agent: %s", agent_name)
            except (AttributeError, TypeError, ValueError, RuntimeError) as e:
                logger.exception("Error stopping agent %s: %s", agent_name, e)

    async def _agent_monitor_loop(self) -> None:
        """Monitor Redis queue for agent requests."""
        logger.info("Starting agent monitor loop")
        while self.running:
            try:
                if not self.redis_client:
                    await asyncio.sleep(10)
                    continue

                req_raw = await self.redis_client.lpop("ai_agent_queue")
                request = to_str(req_raw)

                if request:
                    try:
                        data = json.loads(request)
                    except (json.JSONDecodeError, ValueError, TypeError) as e:
                        logger.exception("Failed to parse queued request: %s", e)
                        await asyncio.sleep(1)
                        continue
                    await self.process_agent_request(data)
                else:
                    await asyncio.sleep(10)
            except (ConnectionError, OSError, AttributeError, TypeError, ValueError) as e:
                logger.exception("Error in agent monitor loop: %s", e)
                await asyncio.sleep(30)

    async def process_agent_request(self, request_data: dict[str, Any]) -> dict[str, Any]:
        """Process a single agent request."""
        agent_type = request_data.get("agent_type", "orchestrator")
        action = request_data.get("action", "process")
        data = request_data.get("data", {}) or {}
        request_id = request_data.get("request_id", f"req_{int(time.time())}")

        logger.info("Processing agent request: type=%s action=%s", agent_type, action)

        # Resolve agent safely
        agent = self.agents.get(agent_type) or self.agents.get("orchestrator") or (next(iter(self.agents.values()), None))
        if not agent:
            msg = "No suitable agent available to process the request"
            raise RuntimeError(msg)

        try:
            result = await self._call_agent_method(agent, "process_request", action, data)

            # Determine status safely
            status = "failed"
            status = ("failed" if result.get("error") else "success") if isinstance(result, dict) else "success" if result else "failed"

            record = {
                "request_id": request_id,
                "agent_type": agent_type,
                "action": action,
                "data": data,
                "result": result,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "status": status,
            }

            self.agent_history.append(record)

            if self.redis_client:
                try:
                    await self.redis_client.set(f"agent:{request_id}", json.dumps(record))
                    await self.redis_client.lpush("agent_results", json.dumps(record))
                except (ConnectionError, OSError, AttributeError, TypeError, ValueError) as e:
                    logger.warning("Failed to publish agent result: %s", e)

            logger.info("Agent request processed: %s", record["status"])
        except (AttributeError, TypeError, ValueError, RuntimeError, KeyError) as e:
            logger.exception("Error processing agent request: %s", e)
            record = {
                "request_id": request_data.get("request_id", f"req_{int(time.time())}"),
                "agent_type": request_data.get("agent_type", "unknown"),
                "error": str(e),
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "status": "failed",
            }
            self.agent_history.append(record)
            return record
        else:
            return record

    async def execute_agent_action(self, agent_type: str, action: str, data: dict[str, Any] | None = None) -> dict[str, Any]:
        """Execute an action on a specific agent."""
        data = data or {}
        logger.info("Executing agent action: type=%s action=%s", agent_type, action)

        agent = self.agents.get(agent_type) or self.agents.get("orchestrator") or (next(iter(self.agents.values()), None))
        if not agent:
            msg = "No suitable agent available to execute action"
            raise RuntimeError(msg)

        try:
            result = await self._call_agent_method(agent, "process_request", action, data)

            status = "failed"
            status = ("failed" if result.get("error") else "success") if isinstance(result, dict) else "success" if result else "failed"

            record = {
                "agent_type": agent_type,
                "action": action,
                "data": data,
                "result": result,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "status": status,
            }
            self.agent_history.append(record)
            logger.info("Agent action executed: %s", record["status"])
        except (AttributeError, TypeError, ValueError, RuntimeError, KeyError) as e:
            logger.exception("Error executing agent action: %s", e)
            raise
        else:
            return record

    async def get_agent_history(self, limit: int = 100) -> list[dict[str, Any]]:
        """Return recent agent execution history."""
        try:
            return self.agent_history[-limit:] if self.agent_history else []
        except (AttributeError, TypeError, ValueError, IndexError) as e:
            logger.exception("Error getting agent history: %s", e)
            return []

    async def get_agent_status(self) -> dict[str, Any]:
        """Return status for all agents."""
        try:
            # VECTORIZED agent status collection for performance
            agent_status: dict[str, Any] = {}
            for name, agent in self.agents.items():
                try:
                    if hasattr(agent, "get_status"):
                        status = await self._call_agent_method(agent, "get_status")
                    else:
                        status = {"status": "running" if self.running else "stopped"}
                    agent_status[name] = status
                except (AttributeError, TypeError, ValueError, RuntimeError) as e:
                    agent_status[name] = {"status": "error", "error": str(e)}

            return {
                "agents": agent_status,
                "total_agents": len(self.agents),
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
        except (AttributeError, TypeError, ValueError, RuntimeError) as e:
            logger.exception("Error getting agent status: %s", e)
            return {"error": str(e)}


# Global service instance
# Agent service state - using dict to avoid global keyword
_agent_service_state: dict[str, AIAgentOrchestratorService | None] = {"instance": None}


@app.on_event("startup")
async def startup_event() -> None:
    """Initialize and start service."""
    _agent_service_state["instance"] = AIAgentOrchestratorService()
    await _agent_service_state["instance"].initialize()
    await _agent_service_state["instance"].start()
    logger.info("AI Agent Orchestrator Service started")


@app.on_event("shutdown")
async def shutdown_event() -> None:
    """Stop service."""
    if _agent_service_state["instance"]:
        await _agent_service_state["instance"].stop()
        logger.info("AI Agent Orchestrator Service stopped")


# Health endpoint DELETED


@app.get("/status")
async def service_status():
    """Service status and agent summary."""
    if not _agent_service_state["instance"]:
        raise HTTPException(status_code=503, detail="Service not initialized")
    try:
        agent_status = await _agent_service_state["instance"].get_agent_status()
        redis_connected = False
        if _agent_service_state["instance"].redis_client:
            try:
                redis_connected = bool(await _agent_service_state["instance"].redis_client.ping())
            except (ConnectionError, OSError, AttributeError, TypeError):
                redis_connected = False
        return {
            "status": "running" if _agent_service_state["instance"].running else "stopped",
            "redis_connected": redis_connected,
            "agent_count": len(_agent_service_state["instance"].agents),
            "agent_history_count": len(_agent_service_state["instance"].agent_history),
            "agents": agent_status,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
    except (AttributeError, TypeError, ValueError, RuntimeError) as e:
        logger.exception("Error getting service status: %s", e)
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.post("/execute")
async def execute_agent_action(agent_type: str, action: str, data: dict[str, Any] | None = None):
    """Execute a specific agent action."""
    if not _agent_service_state["instance"]:
        raise HTTPException(status_code=503, detail="Service not initialized")
    try:
        record = await _agent_service_state["instance"].execute_agent_action(agent_type, action, data or {})
        return {
            "status": "success",
            "agent": record,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
    except (AttributeError, TypeError, ValueError, RuntimeError) as e:
        logger.exception("Error in execute endpoint: %s", e)
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.get("/agents")
async def get_agents():
    """Return agent statuses."""
    if not _agent_service_state["instance"]:
        raise HTTPException(status_code=503, detail="Service not initialized")
    try:
        return await _agent_service_state["instance"].get_agent_status()
    except (AttributeError, TypeError, ValueError, RuntimeError) as e:
        logger.exception("Error getting agents: %s", e)
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.get("/history")
async def get_agent_history(limit: int = 100):
    """Return agent execution history."""
    if not _agent_service_state["instance"]:
        raise HTTPException(status_code=503, detail="Service not initialized")
    try:
        history = await _agent_service_state["instance"].get_agent_history(limit)
        return {
            "history": history,
            "count": len(history),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
    except (AttributeError, TypeError, ValueError, RuntimeError) as e:
        logger.exception("Error getting agent history: %s", e)
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.post("/queue/process")
async def process_agent_queue():
    """Drain and process queued agent requests from Redis."""
    if not _agent_service_state["instance"]:
        raise HTTPException(status_code=503, detail="Service not initialized")
    try:
        if not _agent_service_state["instance"].redis_client:
            return {
                "status": "success",
                "processed": 0,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
        processed = 0
        while True:
            req_raw = await _agent_service_state["instance"].redis_client.lpop("ai_agent_queue")
            request = to_str(req_raw)
            if not request:
                break
            try:
                data = json.loads(request)
            except (json.JSONDecodeError, ValueError, TypeError) as e:
                logger.exception("Failed to parse queued request during manual process: %s", e)
                continue
            await _agent_service_state["instance"].process_agent_request(data)
            processed += 1
        return {
            "status": "success",
            "processed": processed,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
    except (AttributeError, TypeError, ValueError, RuntimeError) as e:
        logger.exception("Error processing agent queue: %s", e)
        raise HTTPException(status_code=500, detail=str(e)) from e


if __name__ == "__main__":
    port = int(os.getenv("SERVICE_PORT", "8006"))
    logger.info("Starting AI Agent Orchestrator Service on port %d", port)
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")
