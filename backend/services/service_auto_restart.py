"""
Service Auto-Restart Manager - Automatically restarts services when memory exceeds threshold.

This service monitors process memory and automatically restarts services that exceed
the configured memory threshold, preventing system degradation without manual intervention.
"""

import asyncio
import contextlib
import logging
import os
from datetime import datetime, timezone
from typing import Any

import psutil

from backend.services.task_manager import task_manager

logger = logging.getLogger("backend.service_auto_restart")


class ServiceAutoRestartManager:
    """Monitor and auto-restart services based on memory usage."""

    def __init__(
        self,
        memory_threshold_mb: float = 600.0,
        interval_sec: int = 120,
        restart_cooldown_sec: int = 300,
    ) -> None:
        """
        Initialize the auto-restart manager.

        Args:
            memory_threshold_mb: Memory threshold before restart (default 600 MB)
            interval_sec: Check interval in seconds (default 120)
            restart_cooldown_sec: Cooldown between restarts of same service (default 300 sec)
        """
        self._memory_threshold_mb = float(os.getenv("SERVICE_RESTART_MEMORY_MB", str(memory_threshold_mb)))
        self._interval = max(10, int(interval_sec))
        self._cooldown = max(60, int(restart_cooldown_sec))
        self._running = False
        self._task: asyncio.Task[Any] | None = None

        # Track restart history: PID -> last restart timestamp
        self._restart_history: dict[int, float] = {}

        # Python process names that should be restarted
        self._critical_services = {
            "python.exe",
            "python",
        }

    async def _loop(self) -> None:
        """Main monitoring loop."""
        logger.info(
            "Service auto-restart manager started - threshold: %s MB, check interval: %s seconds",
            self._memory_threshold_mb,
            self._interval,
        )

        while self._running:
            try:
                await asyncio.sleep(self._interval)
                await self._check_and_restart()
            except asyncio.CancelledError:
                break
            except Exception as exc:
                logger.exception("Service auto-restart error: %s", exc)

    async def _check_and_restart(self) -> None:
        """Check all processes and restart if needed."""
        try:
            critical_processes = []

            for proc in psutil.process_iter(["pid", "name", "memory_info", "cmdline"]):
                try:
                    if proc.name().lower() not in self._critical_services:
                        continue

                    # Skip system python processes
                    cmdline = " ".join(proc.cmdline())
                    if any(skip in cmdline for skip in ["pip", "conda", "setuptools"]):
                        continue

                    rss_bytes = proc.memory_info().rss
                    rss_mb = rss_bytes / (1024 * 1024)

                    # Check if over threshold
                    if rss_mb > self._memory_threshold_mb:
                        critical_processes.append(
                            {
                                "pid": proc.pid,
                                "memory_mb": rss_mb,
                                "cmdline": cmdline[:100],  # First 100 chars
                            }
                        )

                except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                    pass

            # Process critical services
            for proc_info in critical_processes:
                pid = proc_info["pid"]
                memory_mb = proc_info["memory_mb"]
                cmdline = proc_info["cmdline"]

                # Check if enough time has passed since last restart
                now = datetime.now(timezone.utc).timestamp()
                last_restart = self._restart_history.get(pid, 0)
                time_since_restart = now - last_restart

                if time_since_restart < self._cooldown:
                    # Still in cooldown, just log
                    logger.warning(f"Service PID {pid} still over threshold ({memory_mb:.2f} MB) but in cooldown (restart in {self._cooldown - time_since_restart:.0f}s)")
                else:
                    # Time to restart
                    await self._restart_service(pid, memory_mb, cmdline)
                    self._restart_history[pid] = now

        except Exception as exc:
            logger.exception("Error in check and restart: %s", exc)

    async def _restart_service(self, pid: int, memory_mb: float, cmdline: str) -> None:
        """
        Restart a service that exceeded memory threshold.

        Args:
            pid: Process ID to restart
            memory_mb: Current memory usage in MB
            cmdline: Command line for logging
        """
        try:
            logger.error(f"CRITICAL: Restarting service due to high memory - PID {pid} using {memory_mb:.2f} MB (threshold: {self._memory_threshold_mb} MB)")

            # Log the command for debugging
            logger.info(f"Service cmdline: {cmdline}")

            # Try to terminate gracefully first
            try:
                proc = psutil.Process(pid)

                # Send SIGTERM for graceful shutdown (5 second timeout)
                logger.info(f"Sending SIGTERM to PID {pid}")
                proc.terminate()

                try:
                    proc.wait(timeout=5)
                    logger.info(f"Process PID {pid} terminated gracefully")
                except psutil.TimeoutExpired:
                    # If not terminated, force kill
                    logger.warning(f"Process PID {pid} did not terminate, forcing kill")
                    proc.kill()
                    proc.wait()
                    logger.info(f"Process PID {pid} killed forcefully")

            except psutil.NoSuchProcess:
                logger.info(f"Process PID {pid} already terminated")
            except psutil.AccessDenied:
                logger.warning(f"Access denied to restart PID {pid} - may need elevated privileges")

        except Exception as exc:
            logger.exception(f"Error restarting service PID {pid}: {exc}")

    async def start(self) -> None:
        """Start the auto-restart manager."""
        if self._running:
            return
        self._running = True
        self._task = await task_manager.create_task(self._loop(), name="service_auto_restart_manager:loop")
        logger.info("Service auto-restart manager started")

    async def stop(self) -> None:
        """Stop the auto-restart manager."""
        if not self._running:
            return
        self._running = False
        if self._task and not self._task.done():
            self._task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._task
        self._task = None
        logger.info("Service auto-restart manager stopped")

    async def get_status(self) -> dict[str, Any]:
        """Get current status of auto-restart manager."""
        return {
            "running": self._running,
            "memory_threshold_mb": self._memory_threshold_mb,
            "check_interval_seconds": self._interval,
            "restart_cooldown_seconds": self._cooldown,
            "restart_history_count": len(self._restart_history),
        }


# Global singleton instance
service_auto_restart = ServiceAutoRestartManager()
