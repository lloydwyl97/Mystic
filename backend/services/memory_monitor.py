# backend/services/memory_monitor.py
"""Lightweight in-process memory monitor.

Periodically records the top-N growing object types using `objgraph` and
logs the table so we can spot memory leaks without attaching an external profiler.
Only enabled when objgraph is available; otherwise it is a no-op.
"""

from __future__ import annotations

import asyncio
import contextlib
import gc
import logging
import os
import tracemalloc
from typing import Any

try:
    import objgraph
except ImportError:  # pragma: no cover - objgraph optional
    objgraph = None  # type: ignore[assignment]

from backend.services.task_manager import task_manager

logger = logging.getLogger("backend.memory_monitor")


class _MemoryMonitor:
    """Singleton memory monitor service."""

    def __init__(self, interval_sec: int | None = None) -> None:
        if interval_sec is None:
            # Read from environment variable, default to 120 seconds
            env_interval = os.getenv("MEM_MONITOR_INTERVAL_SEC")
            interval_sec = int(env_interval) if env_interval else 120
        self._interval = max(10, interval_sec)  # safety floor
        self._running = False
        self._task: asyncio.Task[Any] | None = None

    async def _loop(self) -> None:  # pragma: no cover - runtime task
        tracemalloc.start()
        logger.info("Memory monitor started - reporting every %ss", self._interval)

        while self._running:
            try:
                await asyncio.sleep(self._interval)
                if objgraph is None:  # objgraph missing → just skip until installed
                    continue
                gc.collect()
                growth = objgraph.growth(limit=10)  # returns list instead of printing
                logger.info("MEM_GROWTH_TOP10: %s", growth)
            except asyncio.CancelledError:  # graceful shutdown
                break
            except Exception as exc:
                logger.exception("Memory monitor error: %s", exc)

    async def start(self) -> None:
        """Start the monitor if not already running."""
        if self._running or objgraph is None:
            return
        self._running = True
        self._task = await task_manager.create_task(self._loop(), name="memory_monitor:loop")

    async def stop(self) -> None:
        """Stop the monitor and cancel the background task."""
        if not self._running:
            return
        self._running = False
        if self._task and not self._task.done():
            self._task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._task
        self._task = None
        logger.info("Memory monitor stopped")


memory_monitor = _MemoryMonitor()
