"""
Process Memory Profiler - Monitors individual process memory usage and alerts on thresholds.
"""

import asyncio
import contextlib
import logging
import os
from datetime import datetime, timezone
from typing import Any

import psutil

from backend.services.task_manager import task_manager

logger = logging.getLogger("backend.process_memory_profiler")


class ProcessMemoryProfiler:
    """Monitor and alert on process memory usage."""

    def __init__(self, interval_sec: int | None = None, alert_threshold_mb: float = 1200.0) -> None:
        if interval_sec is None:
            env_interval = os.getenv("PROC_MEM_INTERVAL_SEC")
            interval_sec = int(env_interval) if env_interval else 60

        self._interval = max(10, interval_sec)
        self._alert_threshold_mb = float(os.getenv("PROC_MEM_ALERT_MB", str(alert_threshold_mb)))
        # Note: 1200 MB is appropriate for AI/ML workloads with model inference and real-time data processing
        self._running = False
        self._task: asyncio.Task[Any] | None = None
        self._process = psutil.Process()
        self._alerted_pids: dict[int, float] = {}  # PID -> last alert time

    async def _loop(self) -> None:
        """Main monitoring loop."""
        logger.info(
            "Process memory profiler started - reporting every %ss, alert threshold: %s MB",
            self._interval,
            self._alert_threshold_mb,
        )

        while self._running:
            try:
                await asyncio.sleep(self._interval)
                await self._check_all_processes()
            except asyncio.CancelledError:
                break
            except Exception as exc:
                logger.exception("Process memory profiler error: %s", exc)

    async def _check_all_processes(self) -> None:
        """Check all Python processes for memory usage."""
        try:
            for proc in psutil.process_iter(["pid", "name", "memory_info"]):
                try:
                    # Only check Python processes
                    if "python" not in proc.name().lower():
                        continue

                    rss_bytes = proc.memory_info().rss
                    rss_mb = rss_bytes / (1024 * 1024)

                    # Log all Python processes
                    logger.debug(f"Python process PID {proc.pid}: {rss_mb:.2f} MB")

                    # Alert if over threshold
                    if rss_mb > self._alert_threshold_mb:
                        now = datetime.now(timezone.utc).timestamp()
                        last_alert = self._alerted_pids.get(proc.pid, 0)

                        # Only alert once per 5 minutes per process
                        if now - last_alert > 300:
                            logger.warning(f"HIGH MEMORY ALERT: Python process PID {proc.pid} using {rss_mb:.2f} MB (threshold: {self._alert_threshold_mb} MB)")
                            self._alerted_pids[proc.pid] = now

                            # If extremely high (>1000 MB), log traceback info
                            if rss_mb > 1000:
                                logger.error(f"CRITICAL MEMORY: Process PID {proc.pid} exceeds 1000 MB with {rss_mb:.2f} MB!")

                except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                    pass

        except Exception as exc:
            logger.exception("Error checking process memory: %s", exc)

    async def start(self) -> None:
        """Start the profiler if not already running."""
        if self._running:
            return
        self._running = True
        self._task = await task_manager.create_task(self._loop(), name="process_memory_profiler:loop")
        logger.info("Process memory profiler started")

    async def stop(self) -> None:
        """Stop the profiler and cancel the background task."""
        if not self._running:
            return
        self._running = False
        if self._task and not self._task.done():
            self._task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._task
        self._task = None
        logger.info("Process memory profiler stopped")


# Global singleton instance
process_memory_profiler = ProcessMemoryProfiler()
