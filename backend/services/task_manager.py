"""
Centralized Task Manager for proper asyncio task tracking and cleanup.

This module provides a centralized way to manage background tasks, ensuring
proper cleanup and preventing resource leaks.
"""

import asyncio
import logging
from typing import Any

logger = logging.getLogger(__name__)


class TaskManager:
    """Centralized task manager for tracking and cleaning up background tasks."""

    def __init__(self) -> None:
        self._tasks: set[asyncio.Task] = set()
        self._running = True  # Always running for singleton usage
        self._lock: asyncio.Lock | None = None  # Lazy init to avoid event loop issues

    def _get_lock(self) -> asyncio.Lock:
        """Get or create lock lazily to avoid event loop issues at import time."""
        if self._lock is None:
            self._lock = asyncio.Lock()
        return self._lock

    async def create_task(self, coro, name: str | None = None) -> asyncio.Task:
        """Create a tracked background task."""
        async with self._get_lock():
            if not self._running:
                msg = "TaskManager is not running"
                raise RuntimeError(msg)

            task = asyncio.create_task(coro, name=name)
            self._tasks.add(task)

            # Add callback to remove task when it completes
            task.add_done_callback(self._task_done_callback)

            logger.debug(f"Created tracked task: {name or 'unnamed'}")
            return task

    def create_task_sync(self, coro, name: str | None = None) -> asyncio.Task:
        """Create a tracked background task from a sync context."""
        # For sync contexts, we can't acquire the lock asynchronously,
        # so we use asyncio.create_task directly but still track the task
        task = asyncio.create_task(coro, name=name)
        self._tasks.add(task)

        # Add callback to remove task when it completes
        task.add_done_callback(self._task_done_callback)

        logger.debug(f"Created tracked task (sync): {name or 'unnamed'}")
        return task

    def _task_done_callback(self, task: asyncio.Task) -> None:
        """Callback to remove completed tasks from tracking."""
        self._tasks.discard(task)

        if task.cancelled():
            logger.debug(f"Task cancelled: {task.get_name() or 'unnamed'}")
        elif task.exception():
            try:
                exc = task.exception()
                logger.error(f"Task failed: {task.get_name() or 'unnamed'}: {exc}")
            except Exception as e:
                logger.exception(f"Task failed: {task.get_name() or 'unnamed'}: Unable to get exception details: {e}")
        else:
            logger.debug(f"Task completed: {task.get_name() or 'unnamed'}")

    async def cancel_all(self, timeout: float = 10.0) -> None:
        """Cancel all tracked tasks and wait for them to complete."""
        async with self._get_lock():
            if not self._tasks:
                logger.info("No tracked tasks to cancel")
                return

            logger.info(f"Cancelling {len(self._tasks)} tracked tasks")

            # MEMORY LEAK FIX: Take a snapshot to avoid race condition with done callbacks
            tasks_to_cancel = list(self._tasks)

            # Cancel all tasks
            for task in tasks_to_cancel:
                task_name = task.get_name() or "unnamed"
                logger.debug(f"Cancelling task: {task_name}")
                task.cancel()

            # Wait for all tasks to complete with timeout
            # Use the snapshot, not self._tasks which may be modified by callbacks
            if tasks_to_cancel:
                try:
                    await asyncio.wait_for(asyncio.gather(*tasks_to_cancel, return_exceptions=True), timeout=timeout)
                    logger.info("All tracked tasks cancelled successfully")
                except asyncio.TimeoutError:
                    logger.warning(f"Task cancellation timed out after {timeout} seconds")
                except Exception as e:
                    logger.exception(f"Error during task cancellation: {e}")

            # Clear any remaining tasks (in case some were added during cancel)
            self._tasks.clear()

    def get_task_count(self) -> int:
        """Get the number of currently tracked tasks."""
        return len(self._tasks)

    def get_task_info(self) -> dict[str, Any]:
        """Get information about tracked tasks."""
        task_info = []
        for task in self._tasks:
            task_info.append(
                {
                    "name": task.get_name() or "unnamed",
                    "done": task.done(),
                    "cancelled": task.cancelled(),
                }
            )
        return {
            "total_tasks": len(self._tasks),
            "tasks": task_info,
            "running": self._running,
        }


# Global task manager instance
task_manager = TaskManager()
