"""
Optimized async task management utilities for Mystic Trading Platform.
Connected to live configuration and integrated with TaskManager.
"""

import asyncio
import functools
import os
import time
from collections.abc import Coroutine
from enum import Enum
from typing import Any, Generic, TypeVar

T = TypeVar("T")

# Import live configuration
try:
    from backend.config_bridge import get_mystic_config

    _mystic_config = get_mystic_config()
except (ImportError, AttributeError, ValueError, TypeError, RuntimeError):
    _mystic_config = None

# Import TaskManager for task tracking integration
try:
    from backend.services.task_manager import get_task_manager

    _task_manager = None  # Will be initialized on first use
except (ImportError, AttributeError, ValueError, TypeError, RuntimeError):
    _task_manager = None


# Task priorities
class TaskPriority(Enum):
    HIGH = 0
    MEDIUM = 1
    LOW = 2


# Task statistics tracking (separate from TaskManager which tracks task lifecycle)
_task_stats: dict[str, dict[str, Any]] = {}


def _get_max_concurrent_requests() -> int:
    """Get max concurrent requests from live configuration."""
    if _mystic_config is not None:
        try:
            value = _mystic_config.system.max_concurrent_requests
            if isinstance(value, int) and value > 0:
                return value
        except (AttributeError, ValueError, TypeError):
            pass
    # Fallback to environment variable
    try:
        value = int(os.getenv("MAX_CONCURRENT_REQUESTS", "50"))
        return max(1, value)
    except (ValueError, TypeError):
        return 50


def _get_queue_high_water_mark() -> int:
    """Get queue high water mark from environment."""
    try:
        value = int(os.getenv("QUEUE_HIGH_WATER_MARK", "1000"))
        return max(100, value)
    except (ValueError, TypeError):
        return 1000


def _get_task_manager_instance():
    """Get or initialize TaskManager instance."""
    global _task_manager
    if _task_manager is None:
        try:
            _task_manager = get_task_manager()
        except (AttributeError, ValueError, TypeError, RuntimeError):
            _task_manager = None
    return _task_manager


# Semaphore for controlling overall concurrency - initialized from live config
_global_semaphore: asyncio.Semaphore | None = None


def _get_global_semaphore() -> asyncio.Semaphore:
    """Get or create global semaphore from live configuration."""
    global _global_semaphore
    if _global_semaphore is None:
        _global_semaphore = asyncio.Semaphore(_get_max_concurrent_requests())
    return _global_semaphore


class PriorityTaskQueue(Generic[T]):
    """Task queue with priority scheduling"""

    def __init__(self, high_water_mark: int | None = None):
        self.high_priority: asyncio.Queue[T] = asyncio.Queue()
        self.medium_priority: asyncio.Queue[T] = asyncio.Queue()
        self.low_priority: asyncio.Queue[T] = asyncio.Queue()
        self.high_water_mark = high_water_mark if high_water_mark is not None else _get_queue_high_water_mark()
        self._throttling = False

    @property
    def total_size(self) -> int:
        """Total number of items in all queues"""
        return self.high_priority.qsize() + self.medium_priority.qsize() + self.low_priority.qsize()

    @property
    def is_throttling(self) -> bool:
        """Check if queue is in throttling mode"""
        # Start throttling when queue size exceeds high water mark
        if not self._throttling and self.total_size > self.high_water_mark:
            self._throttling = True
        # Stop throttling when queue size drops below 75% of high water mark
        elif self._throttling and self.total_size < (self.high_water_mark * 0.75):
            self._throttling = False
        return self._throttling

    async def put(self, item: T, priority: TaskPriority = TaskPriority.MEDIUM) -> None:
        """Add item to the appropriate priority queue"""
        if priority == TaskPriority.HIGH:
            await self.high_priority.put(item)
        elif priority == TaskPriority.MEDIUM:
            await self.medium_priority.put(item)
        else:
            await self.low_priority.put(item)

    async def get(self) -> T:
        """Get next item based on priority"""
        # First check high priority queue
        if not self.high_priority.empty():
            return await self.high_priority.get()

        # Then check medium priority queue
        if not self.medium_priority.empty():
            return await self.medium_priority.get()

        # Finally check low priority queue
        if not self.low_priority.empty():
            return await self.low_priority.get()

        # If all queues are empty, wait for the next item in high priority
        return await self.high_priority.get()

    def task_done(self, priority: TaskPriority = TaskPriority.MEDIUM) -> None:
        """Mark a task as done in the appropriate queue"""
        if priority == TaskPriority.HIGH:
            self.high_priority.task_done()
        elif priority == TaskPriority.MEDIUM:
            self.medium_priority.task_done()
        else:
            self.low_priority.task_done()


# Create a global priority task queue (initialized from live config)
priority_task_queue = PriorityTaskQueue()


async def run_task_with_priority(coro: Coroutine[Any, Any, T], task_name: str, priority: TaskPriority = TaskPriority.MEDIUM, timeout: float | None = None) -> T:
    """Run a coroutine as a task with priority and controlled concurrency"""
    semaphore = _get_global_semaphore()
    async with semaphore:
        # Track task stats
        start_time = time.time()
        _task_stats[task_name] = {"start_time": start_time, "priority": priority.name, "status": "running"}

        # Run the task with optional timeout
        try:
            if timeout is not None:
                result = await asyncio.wait_for(coro, timeout=timeout)
            else:
                result = await coro
        except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError, asyncio.TimeoutError) as e:
            # Update stats on error
            _task_stats[task_name].update({"end_time": time.time(), "duration": time.time() - start_time, "status": "failed", "error": str(e)})
            raise
        else:
            # Update stats on completion
            _task_stats[task_name].update({"end_time": time.time(), "duration": time.time() - start_time, "status": "completed", "error": None})
            return result


async def create_task(coro: Coroutine[Any, Any, T], name: str | None = None, _priority: TaskPriority = TaskPriority.MEDIUM) -> asyncio.Task[T]:
    """Create and track a new task with the given priority using TaskManager"""
    # Note: priority parameter kept for API compatibility but TaskManager doesn't support priority tracking
    task_mgr = _get_task_manager_instance()
    if task_mgr is not None:
        try:
            # Use TaskManager if available and running
            task = await task_mgr.create_task(coro, name=name)
        except (RuntimeError, AttributeError, ValueError, TypeError):
            # Fallback to asyncio.create_task if TaskManager not running
            task = asyncio.create_task(coro, name=name)
    else:
        # Fallback if TaskManager not available
        task = asyncio.create_task(coro, name=name)
    return task


def prioritize(priority: TaskPriority = TaskPriority.MEDIUM):
    """Decorator to run a coroutine function with the specified priority"""

    def decorator(func):
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            # Use function name as task name
            task_name = func.__qualname__
            return await run_task_with_priority(func(*args, **kwargs), task_name=task_name, priority=priority)

        return wrapper

    return decorator


async def get_task_stats() -> dict[str, Any]:
    """Get statistics about all running and completed tasks"""
    task_mgr = _get_task_manager_instance()
    running_count = 0
    task_names: list[str] = []
    if task_mgr is not None:
        try:
            running_count = task_mgr.get_task_count()
            task_names = task_mgr.get_task_names()
        except (AttributeError, ValueError, TypeError, RuntimeError):
            pass

    return {
        "running_tasks": running_count,
        "task_names": task_names,
        "task_history": _task_stats.copy(),
        "throttling": priority_task_queue.is_throttling,
        "queue_sizes": {
            "high": priority_task_queue.high_priority.qsize(),
            "medium": priority_task_queue.medium_priority.qsize(),
            "low": priority_task_queue.low_priority.qsize(),
            "total": priority_task_queue.total_size,
        },
        "concurrency_limit": _get_max_concurrent_requests(),
        "queue_high_water_mark": _get_queue_high_water_mark(),
    }
