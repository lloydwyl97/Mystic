#!/usr/bin/env python3
"""
AI Strategy Executor Service
Port 8003 - Standalone AI strategy execution service
"""

import asyncio
import contextlib
import json
import logging
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import uvicorn

# Add backend directory to path for imports
# import backend.ai as ai strategy execution
from ai_strategy_execution import execute_ai_strategy_signal
from fastapi import FastAPI, HTTPException

from backend.services.task_manager import task_manager

# Lazy import for optional redis helpers (may not be available in all deployments)
try:
    from utils.redis_helpers import to_str
except (ImportError, ModuleNotFoundError, AttributeError, ValueError, TypeError, RuntimeError):
    to_str = None  # type: ignore[assignment, misc]

# Ensure logs directory exists before creating FileHandler
Path("logs").mkdir(parents=True, exist_ok=True)

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("logs/ai_strategy_executor.log", encoding="utf-8"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger("ai_strategy_executor_service")

# Initialize FastAPI app
app = FastAPI(
    title="AI Strategy Executor Service",
    description="Standalone AI strategy execution service",
    version="1.0.0",
)


class AIStrategyExecutorService:
    """AI Strategy Executor Service"""

    def __init__(self) -> None:
        """Initialize the service"""
        self.running = False
        self.execution_history = []
        # Track background tasks for proper cleanup
        self._tasks: list[asyncio.Task[Any]] = []

        # Use shared Redis connection pool to prevent connection exhaustion
        from backend.config.redis_config import get_shared_redis_sync

        self.redis_client = get_shared_redis_sync()

        logger.info("AI Strategy Executor Service initialized")

    async def start(self):
        """Start the service"""
        logger.info("Starting AI Strategy Executor Service...")
        self.running = True

        # Start execution monitoring loop
        task = await task_manager.create_task(self.execution_monitor_loop(), name="ai_strategy_executor_service:execution_monitor_loop")
        self._tasks.append(task)

    async def stop(self):
        """Stop the service"""
        logger.info("Stopping AI Strategy Executor Service...")
        self.running = False
        # Cancel all background tasks
        for task in self._tasks:
            if not task.done():
                task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await task
        self._tasks.clear()

    async def execution_monitor_loop(self):
        """Monitor for strategy execution requests"""
        logger.info("Starting execution monitor loop...")

        while self.running:
            try:
                # Check for execution requests
                request = to_str(self.redis_client.lpop("strategy_execution_queue")) if to_str is not None else self.redis_client.lpop("strategy_execution_queue")

                if request:
                    request_data = json.loads(request)
                    await self.execute_strategy_request(request_data)

                # Wait before next check
                await asyncio.sleep(10)

            except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
                logger.exception(f"Error in execution monitor loop: {e}")
                await asyncio.sleep(30)

    async def execute_strategy_request(self, request_data: dict[str, Any]):
        """Execute a strategy request"""
        # All Live Data, No Fallback/Hardcoded Data
        # Validate required fields before entering try block
        symbol_binance = request_data.get("symbol_binance")
        if not symbol_binance:
            msg = "symbol_binance is required in request_data - no fallback/hardcoded symbol"
            raise ValueError(msg)
        usd_amount = request_data.get("usd_amount", 50)
        signal = request_data.get("signal", True)
        strategy_id = request_data.get("strategy_id", "unknown")

        try:
            logger.info(f"Executing strategy {strategy_id} for {symbol_binance}")

            # Execute the strategy
            result = execute_ai_strategy_signal(symbol_binance, usd_amount, signal)

            # Record execution
            execution_record = {
                "strategy_id": strategy_id,
                "symbol_binance": symbol_binance,
                "usd_amount": usd_amount,
                "signal": signal,
                "result": result,
                "timestamp": datetime.now(tz=timezone.utc).isoformat(),
                "status": ("success" if result and "error" not in result else "failed"),
            }

            # Store execution record
            self.execution_history.append(execution_record)
            self.redis_client.set(
                f"execution:{strategy_id}:{int(time.time())}",
                json.dumps(execution_record),
            )

            # Publish result
            self.redis_client.lpush("execution_results", json.dumps(execution_record))

            logger.info(f"Strategy {strategy_id} executed: {execution_record['status']}")
        except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
            logger.exception(f"Error executing strategy request: {e}")

            # Record failed execution
            execution_record = {
                "strategy_id": request_data.get("strategy_id", "unknown"),
                "error": str(e),
                "timestamp": datetime.now(tz=timezone.utc).isoformat(),
                "status": "failed",
            }

            self.execution_history.append(execution_record)
            return execution_record
        else:
            return execution_record

    async def execute_strategy(
        self,
        strategy_id: str,
        symbol_binance: str,
        # Coinbase removed - using Binance US only
        usd_amount: float,
        signal: bool,
    ) -> dict[str, Any]:
        """Execute a specific strategy"""
        try:
            logger.info(f"Executing strategy {strategy_id}")

            # Execute the strategy
            result = execute_ai_strategy_signal(symbol_binance, usd_amount, signal)

            # Record execution
            execution_record = {
                "strategy_id": strategy_id,
                "symbol_binance": symbol_binance,
                "usd_amount": usd_amount,
                "signal": signal,
                "result": result,
                "timestamp": datetime.now(tz=timezone.utc).isoformat(),
                "status": ("success" if result and "error" not in result else "failed"),
            }

            # Store execution record
            self.execution_history.append(execution_record)
            self.redis_client.set(
                f"execution:{strategy_id}:{int(time.time())}",
                json.dumps(execution_record),
            )

            logger.info(f"Strategy {strategy_id} executed: {execution_record['status']}")
        except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
            logger.exception(f"Error executing strategy: {e}")
            raise
        else:
            return execution_record

    async def get_execution_history(self, limit: int = 100) -> list[dict[str, Any]]:
        """Get execution history"""
        try:
            # Return recent executions
            return self.execution_history[-limit:] if self.execution_history else []
        except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
            logger.exception(f"Error getting execution history: {e}")
            return []

    async def get_strategy_executions(self, strategy_id: str) -> list[dict[str, Any]]:
        """Get executions for a specific strategy"""
        try:
            executions = []
            for key in self.redis_client.scan_iter(f"execution:{strategy_id}:*"):
                execution_data = self.redis_client.get(key)
                if execution_data:
                    executions.append(json.loads(execution_data))
        except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
            logger.exception(f"Error getting strategy executions: {e}")
            return []
        else:
            return executions


# Global service instance
# Executor service state - using dict to avoid global keyword
_executor_service_state: dict[str, AIStrategyExecutorService | None] = {"instance": None}


@app.on_event("startup")
async def startup_event():
    """Startup event - initialize service"""
    try:
        _executor_service_state["instance"] = AIStrategyExecutorService()
        await _executor_service_state["instance"].start()
        logger.info("AI Strategy Executor Service started")
    except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
        logger.exception(f"Failed to start AI Strategy Executor Service: {e}")
        raise


@app.on_event("shutdown")
async def shutdown_event():
    """Shutdown event - stop service"""
    if _executor_service_state["instance"]:
        await _executor_service_state["instance"].stop()
        logger.info("AI Strategy Executor Service stopped")


# Health endpoint DELETED


@app.get("/status")
async def service_status():
    """Get service status"""
    if not _executor_service_state["instance"]:
        raise HTTPException(status_code=503, detail="Service not initialized")

    try:
        redis_connected = _executor_service_state["instance"].redis_client.ping()
    except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError):
        redis_connected = False

    return {
        "status": "running" if _executor_service_state["instance"].running else "stopped",
        "redis_connected": redis_connected,
        "execution_count": len(_executor_service_state["instance"].execution_history),
        "timestamp": datetime.now(tz=timezone.utc).isoformat(),
    }


@app.post("/execute")
async def execute_strategy(
    strategy_id: str,
    symbol_binance: str,
    # All Live Data, No Fallback/Hardcoded Data - symbol_binance is required
    usd_amount: float = 50.0,
    signal: bool = True,
):
    """Execute a strategy"""
    if not _executor_service_state["instance"]:
        raise HTTPException(status_code=503, detail="Service not initialized")

    try:
        execution_record = await _executor_service_state["instance"].execute_strategy(strategy_id, symbol_binance, usd_amount, signal)

        return {
            "status": "success",
            "execution": execution_record,
            "timestamp": datetime.now(tz=timezone.utc).isoformat(),
        }
    except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
        logger.exception(f"Error in execute endpoint: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.get("/executions")
async def get_executions(limit: int = 100):
    """Get execution history"""
    if not _executor_service_state["instance"]:
        raise HTTPException(status_code=503, detail="Service not initialized")

    try:
        executions = await _executor_service_state["instance"].get_execution_history(limit)
        return {
            "executions": executions,
            "count": len(executions),
            "timestamp": datetime.now(tz=timezone.utc).isoformat(),
        }
    except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
        logger.exception(f"Error getting executions: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.get("/executions/{strategy_id}")
async def get_strategy_executions(strategy_id: str):
    """Get executions for a specific strategy"""
    if not _executor_service_state["instance"]:
        raise HTTPException(status_code=503, detail="Service not initialized")

    try:
        executions = await _executor_service_state["instance"].get_strategy_executions(strategy_id)
        return {
            "strategy_id": strategy_id,
            "executions": executions,
            "count": len(executions),
            "timestamp": datetime.now(tz=timezone.utc).isoformat(),
        }
    except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
        logger.exception(f"Error getting strategy executions: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.post("/queue/process")
async def process_execution_queue():
    """Process execution queue"""
    if not _executor_service_state["instance"]:
        raise HTTPException(status_code=503, detail="Service not initialized")

    try:
        processed = 0
        while True:
            if to_str is not None:
                request = to_str(_executor_service_state["instance"].redis_client.lpop("strategy_execution_queue"))
            else:
                request = _executor_service_state["instance"].redis_client.lpop("strategy_execution_queue")
            if not request:
                break

            request_data = json.loads(request)
            await _executor_service_state["instance"].execute_strategy_request(request_data)
            processed += 1

        return {
            "status": "success",
            "processed": processed,
            "timestamp": datetime.now(tz=timezone.utc).isoformat(),
        }
    except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
        logger.exception(f"Error processing execution queue: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e


if __name__ == "__main__":
    # Get port from environment
    port = int(os.getenv("SERVICE_PORT", "8003"))

    logger.info(f"Starting AI Strategy Executor Service on port {port}")

    # Start the FastAPI server
    uvicorn.run(
        "ai_strategy_executor_service:app",
        host="0.0.0.0",
        port=port,
        log_level="info",
        reload=False,
    )
