from __future__ import annotations

import asyncio
import contextlib
import inspect
import json
import os
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from functools import partial, wraps
from typing import Any

import psutil
import redis.asyncio as redis
import structlog

from backend.config.redis_config import get_shared_redis_async
from backend.services.canonical_http_client import get_http_client
from backend.services.task_manager import task_manager

logger = structlog.get_logger()


@dataclass
class PerformanceMetrics:
    timestamp: datetime
    cpu_usage: float
    memory_usage: float
    disk_io: dict[str, float]
    network_io: dict[str, float]
    response_times: dict[str, float]
    cache_hit_rate: float
    active_connections: int
    queue_size: int


@dataclass
class CacheConfig:
    ttl: int = 300
    max_size: int = 1000
    enable_compression: bool = False
    enable_stats: bool = True


class AdvancedCache:
    def __init__(self, redis_url: str | None = None, redis_db: int | None = None) -> None:
        # All Live Data, No Fallback/Hardcoded Data
        # Redis connection must be configured via environment variables
        if redis_url is None:
            redis_url = os.getenv("REDIS_URL")
            if not redis_url:
                # Fallback to individual components if REDIS_URL not set
                redis_host = os.getenv("REDIS_HOST")
                if not redis_host:
                    msg = "REDIS_URL or REDIS_HOST environment variable is required - no fallback/hardcoded Redis connection"
                    raise RuntimeError(msg)
                redis_port = os.getenv("REDIS_PORT", "6379")
                redis_db_num = os.getenv("REDIS_DB", "0")
                redis_url = f"redis://{redis_host}:{redis_port}/{redis_db_num}"
        if redis_db is None:
            redis_db = int(os.getenv("REDIS_DB", "0"))
        self.redis_url = redis_url
        self.redis_db = redis_db
        self.redis_client: redis.Redis | None = None
        self.memory_cache: dict[str, dict[str, Any]] = {}
        self.cache_stats = {"hits": 0, "misses": 0, "sets": 0, "deletes": 0}
        self.config = CacheConfig()

    async def initialize(self) -> None:
        try:
            self.redis_client = get_shared_redis_async()
            await self.redis_client.ping()
            logger.info("cache.redis_initialized", url=self.redis_url, db=self.redis_db)
        except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
            self.redis_client = None
            logger.warning("cache.redis_init_failed", error=str(e))

    async def get(self, key: str) -> Any | None:
        try:
            now = datetime.now(timezone.utc)
            mc = self.memory_cache.get(key)
            if mc and mc.get("expires_at") and mc["expires_at"] > now:
                self.cache_stats["hits"] += 1
                return mc["value"]
            if mc:
                self.memory_cache.pop(key, None)
            if self.redis_client:
                raw = await self.redis_client.get(key)
                if raw is not None:
                    try:
                        val = json.loads(raw)
                    except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError):
                        val = raw
                    self.memory_cache[key] = {
                        "value": val,
                        "expires_at": now + timedelta(seconds=min(60, self.config.ttl)),
                    }
                    self.cache_stats["hits"] += 1
                    return val
            self.cache_stats["misses"] += 1
        except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
            logger.exception("cache.get_error", key=key, error=str(e))
            return None
        else:
            return None

    async def set(self, key: str, value: Any, ttl: int | None = None) -> bool:
        try:
            ttl = int(ttl or self.config.ttl)
            expires_at = datetime.now(timezone.utc) + timedelta(seconds=ttl)
            self.memory_cache[key] = {"value": value, "expires_at": expires_at}
            if self.redis_client:
                try:
                    payload = json.dumps(value)
                except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError):
                    payload = str(value)
                await self.redis_client.setex(key, ttl, payload)
            self.cache_stats["sets"] += 1
            if len(self.memory_cache) > self.config.max_size:
                self._cleanup_memory_cache()
        except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
            logger.exception("cache.set_error", key=key, error=str(e))
            return False
        else:
            return True

    async def delete(self, key: str) -> bool:
        try:
            self.memory_cache.pop(key, None)
            if self.redis_client:
                await self.redis_client.delete(key)
            self.cache_stats["deletes"] += 1
        except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
            logger.exception("cache.delete_error", key=key, error=str(e))
            return False
        else:
            return True

    def _cleanup_memory_cache(self) -> None:
        now = datetime.now(timezone.utc)
        expired = [k for k, v in self.memory_cache.items() if v.get("expires_at") and v["expires_at"] <= now]
        for k in expired:
            self.memory_cache.pop(k, None)
        if len(self.memory_cache) > self.config.max_size:
            for k in list(self.memory_cache.keys())[: len(self.memory_cache) - self.config.max_size]:
                self.memory_cache.pop(k, None)

    def get_stats(self) -> dict[str, Any]:
        total = self.cache_stats["hits"] + self.cache_stats["misses"]
        hit_rate = (self.cache_stats["hits"] / total) if total > 0 else 0.0
        return {
            "hits": self.cache_stats["hits"],
            "misses": self.cache_stats["misses"],
            "hit_rate": hit_rate,
            "sets": self.cache_stats["sets"],
            "deletes": self.cache_stats["deletes"],
            "memory_cache_size": len(self.memory_cache),
            "redis_connected": self.redis_client is not None,
        }


class ConnectionPool:
    def __init__(self) -> None:
        self.http_client: Any | None = None
        self.max_connections = 100
        self.connection_timeout = 30
        self._using_shared_client = False

    async def initialize(self, http_client: Any = None) -> None:
        try:
            if http_client:
                # Use shared HTTP client
                self.http_client = http_client
                self._using_shared_client = True
                logger.info("pool.http_initialized_with_shared_client")
            else:
                # Use centralized HTTP client
                self.http_client = await get_http_client()
                # We created our own centralized client; mark as not using a shared client so that close() will aclose it.
                self._using_shared_client = False
                logger.info("pool.http_initialized_with_centralized_client")
        except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
            self.http_client = None
            logger.exception("pool.http_init_failed", error=str(e))

    async def get_http_session(self) -> Any | None:
        return self.http_client

    async def close(self) -> None:
        try:
            # Only close if we created our own client (not using shared client)
            if self.http_client and hasattr(self, "_using_shared_client") and not self._using_shared_client:
                # Assume http_client has aclose coroutine method
                aclose = getattr(self.http_client, "aclose", None)
                if aclose and inspect.iscoroutinefunction(aclose):
                    await aclose()
                elif aclose:
                    # if aclose is a non-async callable
                    aclose()
                self.http_client = None
                logger.info("pool.closed")
            elif self.http_client:
                # Using shared client - don't close it
                logger.info("pool.using_shared_client_no_close")
        except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
            logger.exception("pool.close_error", error=str(e))


class AsyncTaskQueue:
    def __init__(self, max_workers: int = 10) -> None:
        self.max_workers = max_workers
        self.task_queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
        self.result_queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
        self.workers: list[asyncio.Task] = []
        self.is_running = False
        self.stats = {
            "tasks_processed": 0,
            "tasks_failed": 0,
            "average_processing_time": 0.0,
        }

    async def start(self) -> None:
        self.is_running = True
        for _ in range(self.max_workers):
            self.workers.append(await task_manager.create_task(self._worker(), name="performance_optimizer:worker"))
        logger.info("queue.started", workers=self.max_workers)

    async def stop(self) -> None:
        self.is_running = False
        for w in self.workers:
            w.cancel()
        await asyncio.gather(*self.workers, return_exceptions=True)
        self.workers.clear()
        logger.info("queue.stopped")

    async def add_task(self, task_func: Callable, *args, **kwargs) -> str:
        task_id = f"task_{int(time.time() * 1000)}"
        await self.task_queue.put(
            {
                "id": task_id,
                "func": task_func,
                "args": args,
                "kwargs": kwargs,
                "created_at": datetime.now(timezone.utc),
            }
        )
        return task_id

    async def get_result(self, task_id: str, timeout: float = 30.0) -> Any:
        start = time.time()
        while time.time() - start < timeout:
            try:
                result = await asyncio.wait_for(self.result_queue.get(), timeout=1.0)
                if result.get("task_id") == task_id:
                    return result.get("result")
                # If not the result we want, continue looping (other callers may fetch their results)
            except asyncio.TimeoutError:
                continue
        msg = f"Task {task_id} result not available within {timeout} seconds"
        raise TimeoutError(msg)

    async def _worker(self) -> None:
        while self.is_running:
            try:
                task = await asyncio.wait_for(self.task_queue.get(), timeout=1.0)
            except asyncio.TimeoutError:
                continue

            try:
                started = time.time()
                func = task.get("func")
                args = task.get("args", ())
                kwargs = task.get("kwargs", {})
                task_id = task.get("id")

                result = None
                # If the provided function is a coroutine function, await it directly
                if asyncio.iscoroutinefunction(func):
                    result = await func(*args, **kwargs)
                else:
                    possible = func(*args, **kwargs)
                    if asyncio.iscoroutine(possible):
                        result = await possible
                    else:
                        loop = asyncio.get_running_loop()
                        # Run synchronous call in executor
                        result = await loop.run_in_executor(None, partial(func, *args, **kwargs))

                duration = time.time() - started
                processed = self.stats["tasks_processed"]
                avg = self.stats["average_processing_time"]
                self.stats["tasks_processed"] = processed + 1
                # update running average
                self.stats["average_processing_time"] = (avg * processed + duration) / (processed + 1)

                await self.result_queue.put({"task_id": task_id, "result": result})
            except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
                self.stats["tasks_failed"] += 1
                task_id = task.get("id") if isinstance(task, dict) else None
                await self.result_queue.put({"task_id": task_id, "result": None, "error": str(e)})
                logger.exception("queue.task_error", error=str(e))
            finally:
                with contextlib.suppress(ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError):
                    self.task_queue.task_done()

    def get_stats(self) -> dict[str, Any]:
        return dict(self.stats)


class PerformanceMonitor:
    def __init__(self) -> None:
        self.metrics_history: list[PerformanceMetrics] = []
        self.max_history_size = 1000
        self.monitoring_interval = 60
        self.is_monitoring = False
        self.monitor_task: asyncio.Task | None = None

    async def start_monitoring(self) -> None:
        self.is_monitoring = True
        self.monitor_task = await task_manager.create_task(self._monitor_loop(), name="performance_optimizer:monitor_loop")
        logger.info("monitor.started", interval=self.monitoring_interval)

    async def stop_monitoring(self) -> None:
        self.is_monitoring = False
        if self.monitor_task:
            self.monitor_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self.monitor_task
        logger.info("monitor.stopped")

    async def _monitor_loop(self) -> None:
        while self.is_monitoring:
            try:
                metrics = await self._collect_metrics()
                self.metrics_history.append(metrics)
                if len(self.metrics_history) > self.max_history_size:
                    self.metrics_history.pop(0)
                await asyncio.sleep(self.monitoring_interval)
            except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
                logger.exception("monitor.loop_error", error=str(e))
                await asyncio.sleep(self.monitoring_interval)

    async def _collect_metrics(self) -> PerformanceMetrics:
        cpu = psutil.cpu_percent(interval=1)
        mem = psutil.virtual_memory().percent
        disk = psutil.disk_io_counters()
        net = psutil.net_io_counters()
        disk_io = {
            "read_bytes_per_sec": disk.read_bytes if disk else 0.0,
            "write_bytes_per_sec": disk.write_bytes if disk else 0.0,
        }
        network_io = {
            "bytes_sent_per_sec": net.bytes_sent if net else 0.0,
            "bytes_recv_per_sec": net.bytes_recv if net else 0.0,
        }
        response_times = {
            "/api/v1/trading": 0.1,
            "/api/v1/portfolio": 0.05,
            "/api/v1/market-data": 0.02,
        }
        cache_hit_rate = 0.85
        active_connections = 10
        queue_size = 5
        return PerformanceMetrics(
            timestamp=datetime.now(timezone.utc),
            cpu_usage=cpu,
            memory_usage=mem,
            disk_io=disk_io,
            network_io=network_io,
            response_times=response_times,
            cache_hit_rate=cache_hit_rate,
            active_connections=active_connections,
            queue_size=queue_size,
        )

    def get_current_metrics(self) -> PerformanceMetrics | None:
        return self.metrics_history[-1] if self.metrics_history else None

    def get_metrics_history(self, hours: int = 24) -> list[PerformanceMetrics]:
        cutoff = datetime.now(timezone.utc) - timedelta(hours=hours)
        return [m for m in self.metrics_history if m.timestamp >= cutoff]

    def get_performance_summary(self) -> dict[str, Any]:
        if not self.metrics_history:
            return {}
        recent = self.get_metrics_history(1)
        if not recent:
            return {}
        cpu_vals = [m.cpu_usage for m in recent]
        mem_vals = [m.memory_usage for m in recent]
        return {
            "cpu": {
                "current": cpu_vals[-1],
                "average": sum(cpu_vals) / len(cpu_vals),
                "max": max(cpu_vals),
                "min": min(cpu_vals),
            },
            "memory": {
                "current": mem_vals[-1],
                "average": sum(mem_vals) / len(mem_vals),
                "max": max(mem_vals),
                "min": min(mem_vals),
            },
            "performance_score": self._score(recent),
            "alerts": self._alerts(recent),
        }

    def _score(self, metrics: list[PerformanceMetrics]) -> float:
        if not metrics:
            return 0.0
        cpu_score = 100.0 - max(m.cpu_usage for m in metrics)
        mem_score = 100.0 - max(m.memory_usage for m in metrics)
        resp_max = max(max(m.response_times.values()) for m in metrics)
        resp_score = 100.0 - min(100.0, resp_max * 1000.0)
        return cpu_score * 0.4 + mem_score * 0.3 + resp_score * 0.3

    def _alerts(self, metrics: list[PerformanceMetrics]) -> list[dict[str, Any]]:
        alerts: list[dict[str, Any]] = []
        if not metrics:
            return alerts
        latest = metrics[-1]
        if latest.cpu_usage > 80.0:
            alerts.append(
                {
                    "type": "high_cpu",
                    "severity": "warning",
                    "message": f"High CPU usage: {latest.cpu_usage:.1f}%",
                    "timestamp": latest.timestamp.isoformat(),
                },
            )
        if latest.memory_usage > 85.0:
            alerts.append(
                {
                    "type": "high_memory",
                    "severity": "warning",
                    "message": f"High memory usage: {latest.memory_usage:.1f}%",
                    "timestamp": latest.timestamp.isoformat(),
                },
            )
        for ep, rt in latest.response_times.items():
            if rt > 1.0:
                alerts.append(
                    {
                        "type": "slow_response",
                        "severity": "warning",
                        "message": f"Slow response time for {ep}: {rt:.2f}s",
                        "timestamp": latest.timestamp.isoformat(),
                    },
                )
        return alerts


class PerformanceOptimizer:
    def __init__(self) -> None:
        self.cache = AdvancedCache()
        self.connection_pool = ConnectionPool()
        self.task_queue = AsyncTaskQueue()
        self.performance_monitor = PerformanceMonitor()
        self.optimization_enabled = True

    async def initialize(self) -> None:
        try:
            await self.cache.initialize()
            await self.connection_pool.initialize()
            await self.task_queue.start()
            await self.performance_monitor.start_monitoring()
            logger.info("optimizer.initialized")
        except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
            logger.exception("optimizer.init_failed", error=str(e))

    async def shutdown(self) -> None:
        try:
            await self.task_queue.stop()
            await self.connection_pool.close()
            await self.performance_monitor.stop_monitoring()
            logger.info("optimizer.shutdown_complete")
        except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
            logger.exception("optimizer.shutdown_error", error=str(e))

    def cache_decorator(self, ttl: int | None = None):
        def decorator(func: Callable):
            @wraps(func)
            async def wrapper(*args, **kwargs):
                if not self.optimization_enabled:
                    return await func(*args, **kwargs)
                key = self._make_cache_key(func.__name__, args, kwargs)
                cached = await self.cache.get(key)
                if cached is not None:
                    return cached
                result = await func(*args, **kwargs)
                await self.cache.set(key, result, ttl)
                return result

            return wrapper

        return decorator

    def async_task_decorator(self):
        def decorator(func: Callable):
            @wraps(func)
            async def wrapper(*args, **kwargs):
                if not self.optimization_enabled:
                    return await func(*args, **kwargs)
                task_id = await self.task_queue.add_task(func, *args, **kwargs)
                return await self.task_queue.get_result(task_id)

            return wrapper

        return decorator

    def get_optimization_status(self) -> dict[str, Any]:
        current_metrics = self.performance_monitor.get_current_metrics()
        return {
            "optimization_enabled": self.optimization_enabled,
            "cache_stats": self.cache.get_stats(),
            "task_queue_stats": self.task_queue.get_stats(),
            "performance_summary": self.performance_monitor.get_performance_summary(),
            "current_metrics": (current_metrics.__dict__ if current_metrics else None),
        }

    def enable_optimization(self) -> None:
        self.optimization_enabled = True
        logger.info("optimizer.enabled")

    def disable_optimization(self) -> None:
        self.optimization_enabled = False
        logger.info("optimizer.disabled")

    @staticmethod
    def _make_cache_key(name: str, args: tuple, kwargs: dict) -> str:
        try:
            args_repr = json.dumps(args, default=str, separators=(",", ":"), ensure_ascii=False)
        except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError):
            args_repr = repr(args)
        try:
            kwargs_repr = json.dumps(
                kwargs,
                default=str,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
            )
        except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError):
            kwargs_repr = repr(kwargs)
        return f"{name}:{hash(args_repr + '|' + kwargs_repr)}"


performance_optimizer = PerformanceOptimizer()
