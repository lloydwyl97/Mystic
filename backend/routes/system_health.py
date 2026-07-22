"""
System Health API - Comprehensive health check endpoint for monitoring.

Provides detailed system status including memory, connections, services, etc.
"""

import asyncio
import contextlib
import logging
from datetime import datetime, timezone
from typing import Any

import psutil

logger = logging.getLogger(__name__)
from fastapi import APIRouter

# Try to import our monitoring services
try:
    from backend.services.process_memory_profiler import process_memory_profiler
except ImportError:
    process_memory_profiler = None

# Service auto-restart DISABLED (user request - not important)
# try:
#     from backend.services.service_auto_restart import service_auto_restart
# except ImportError:
#     service_auto_restart = None
service_auto_restart = None  # DISABLED

try:
    from backend.services.memory_leak_analyzer import memory_leak_analyzer
except ImportError:
    memory_leak_analyzer = None

try:
    from backend.config.redis_config import get_shared_redis_sync
except ImportError:
    get_shared_redis_sync = None

try:
    from backend.utils.redis_helpers import WRITER_ROLES, to_str
except ImportError:
    to_str = None
    WRITER_ROLES = None

try:
    from backend.config.redis_config import get_shared_redis_async
except ImportError:
    get_shared_redis_async = None

router = APIRouter(prefix="/api/system", tags=["system"])


def _safe_monitoring_block(
    process_memory_profiler: Any,
    service_auto_restart: Any,
    memory_growth: Any,
) -> dict[str, Any]:
    """Build monitoring dict without touching attributes that may not exist."""
    try:
        prof_running = False
        prof_threshold = 400
        if process_memory_profiler:
            prof_running = getattr(process_memory_profiler, "_running", False)
            prof_threshold = getattr(process_memory_profiler, "_alert_threshold_mb", 400)
        auto_running = False
        auto_threshold = 600
        if service_auto_restart:
            auto_running = getattr(service_auto_restart, "running", getattr(service_auto_restart, "_running", False))
            auto_threshold = getattr(service_auto_restart, "_memory_threshold_mb", 600)
        return {
            "memory_profiler": {
                "running": prof_running,
                "threshold_mb": prof_threshold,
            },
            "auto_restart": {
                "running": auto_running,
                "threshold_mb": auto_threshold,
            },
            "memory_analysis": memory_growth if memory_growth else {"status": "not_available"},
        }
    except Exception as ex:
        logger.debug("_safe_monitoring_block failed: %s", ex)
        return {
            "memory_profiler": {"running": False, "threshold_mb": 400},
            "auto_restart": {"running": False, "threshold_mb": 600},
            "memory_analysis": {"status": "not_available"},
        }


@router.get("/health/comprehensive")
async def get_comprehensive_health() -> dict[str, Any]:
    """Get comprehensive system health status."""

    try:
        current_process = psutil.Process()
        memory_info = current_process.memory_info()
        memory_mb = memory_info.rss / (1024 * 1024)

        # Get all Python processes
        python_processes = []
        total_python_memory = 0
        for proc in psutil.process_iter(["pid", "name", "memory_info"]):
            try:
                if "python" in proc.name().lower():
                    proc_memory_mb = proc.memory_info().rss / (1024 * 1024)
                    python_processes.append(
                        {
                            "pid": proc.pid,
                            "memory_mb": round(proc_memory_mb, 2),
                        }
                    )
                    total_python_memory += proc_memory_mb
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass

        # Check Redis (run sync client in thread to avoid blocking event loop)
        redis_status = "unknown"
        redis_connections = 0
        if get_shared_redis_sync:

            def _check_redis_sync():
                try:
                    redis_client = get_shared_redis_sync()
                    if redis_client:
                        redis_client.ping()
                        conns = 0
                        if hasattr(redis_client, "connection_pool") and hasattr(redis_client.connection_pool, "_available_connections"):
                            conns = len(redis_client.connection_pool._available_connections)
                        return "connected", conns
                except Exception as ex:
                    logger.debug("Redis health check failed: %s", ex)
                return "error", 0

            redis_status, redis_connections = await asyncio.to_thread(_check_redis_sync)

        # System resources
        cpu_percent = psutil.cpu_percent(interval=1)
        disk = psutil.disk_usage("/")

        # Memory growth analysis
        memory_growth = None
        if memory_leak_analyzer:
            with contextlib.suppress(Exception):
                memory_growth = await memory_leak_analyzer.get_memory_growth()

        vm = psutil.virtual_memory()
        health_status = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "overall_status": "healthy" if memory_mb < 2000 else "warning" if memory_mb < 2800 else "critical",
            "system": {
                "cpu_percent": round(cpu_percent, 1),
                "ram_total_gb": round(vm.total / (1024**3), 2),
                "ram_available_gb": round(vm.available / (1024**3), 2),
                "ram_used_percent": round(vm.percent, 1),
                "disk_used_percent": round(disk.percent, 1),
                "disk_free_gb": round(disk.free / (1024**3), 2),
            },
            "backend_api": {
                "memory_mb": round(memory_mb, 2),
                "memory_status": "healthy" if memory_mb < 300 else "warning" if memory_mb < 400 else "critical",
            },
            "python_processes": {
                "count": len(python_processes),
                "total_memory_mb": round(total_python_memory, 2),
                "processes": sorted(python_processes, key=lambda x: x["memory_mb"], reverse=True)[:5],
            },
            "monitoring": _safe_monitoring_block(process_memory_profiler, service_auto_restart, memory_growth),
            "redis": {
                "status": redis_status,
                "connections": redis_connections,
                "connection_limit": 10,
            },
            "recommendations": get_health_recommendations(memory_mb, cpu_percent),
        }

        return health_status

    except Exception as e:
        return {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "error": str(e),
            "status": "error",
        }


@router.get("/health/quick")
async def get_quick_health() -> dict[str, Any]:
    """Get quick system health check (minimal response)."""

    try:
        current_process = psutil.Process()
        memory_mb = current_process.memory_info().rss / (1024 * 1024)

        return {
            "status": "ok" if memory_mb < 2000 else "warning" if memory_mb < 2800 else "critical",
            "memory_mb": round(memory_mb, 2),
            "cpu_percent": psutil.cpu_percent(interval=0.1),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
    except Exception as e:
        return {
            "status": "error",
            "error": str(e),
        }


@router.get("/health/memory")
async def get_memory_health() -> dict[str, Any]:
    """Get detailed memory health information."""

    try:
        current_process = psutil.Process()
        memory_info = current_process.memory_info()

        memory_data = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "process": {
                "rss_mb": round(memory_info.rss / (1024 * 1024), 2),
                "vms_mb": round(memory_info.vms / (1024 * 1024), 2),
                "percent": current_process.memory_percent(),
            },
            "system": {
                "total_mb": round(psutil.virtual_memory().total / (1024 * 1024), 2),
                "available_mb": round(psutil.virtual_memory().available / (1024 * 1024), 2),
                "used_percent": psutil.virtual_memory().percent,
            },
        }

        # Add memory growth if available
        if memory_leak_analyzer:
            try:
                growth = await memory_leak_analyzer.get_memory_growth()
                memory_data["growth_analysis"] = growth
            except Exception as ex:
                logger.debug("Memory growth fetch failed: %s", ex)

        return memory_data

    except Exception as e:
        return {"error": str(e), "status": "error"}


def get_health_recommendations(memory_mb: float, cpu_percent: float) -> list:
    """Get recommendations based on current health."""

    recommendations = []

    if memory_mb > 600:
        recommendations.append({"level": "CRITICAL", "issue": "Memory usage exceeding restart threshold", "action": "Service will be auto-restarted"})
    elif memory_mb > 400:
        recommendations.append({"level": "WARNING", "issue": "Memory usage high", "action": "Monitor closely, investigate memory leak"})

    if cpu_percent > 80:
        recommendations.append({"level": "WARNING", "issue": "CPU usage high", "action": "Check for heavy computations or infinite loops"})

    if cpu_percent < 5 and memory_mb > 300:
        recommendations.append({"level": "INFO", "issue": "High memory with low CPU", "action": "Likely memory leak, not active computation"})

    if not recommendations:
        recommendations.append({"level": "OK", "issue": "System healthy", "action": "No action needed"})

    return recommendations


def _process_running(pattern: str) -> bool:
    import subprocess

    try:
        res = subprocess.run(
            ["pgrep", "-f", pattern],
            capture_output=True,
            timeout=5,
            check=False,
        )
        return res.returncode == 0
    except Exception:
        return False


@router.get("/process-health")
async def get_process_health() -> dict[str, Any]:
    """Read-only Mystic core process heartbeat (pgrep-based).

    Core processes match ``./start_mystic.sh core`` (7 processes).
    ``live_data_collector`` is retired/optional — OHLCV lives in live_market_data.
    """
    checks = {
        "uvicorn": _process_running("uvicorn backend.main:app") or _process_running("backend.main:app"),
        "live_market_data": _process_running("start_live_market_data.py"),
        "ai_signal_generator": _process_running("start_ai_signal_generator.py"),
        "portfolio_engine": _process_running("start_portfolio_engine_integration.py"),
        "ai_market_context": _process_running("start_ai_market_context.py"),
        "ai_learning": _process_running("start_ai_learning.py"),
        "scalp_runner": _process_running("backend.services.binance_scalp.runner"),
    }
    optional = {
        "live_data_collector": {
            "running": _process_running("live_data_collector.py"),
            "classification": "retired_optional",
            "note": "Not launched by start_mystic.sh core; OHLCV is provided by start_live_market_data.py.",
        },
    }
    redis_ok = False
    if get_shared_redis_sync:
        try:
            client = get_shared_redis_sync()
            redis_ok = client is not None and client.ping()
        except Exception:
            redis_ok = False
    core_ok = checks["uvicorn"] and checks["portfolio_engine"]
    all_ok = core_ok and checks["live_market_data"] and checks["ai_signal_generator"] and checks["ai_market_context"] and checks["ai_learning"] and checks["scalp_runner"]
    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "status": "healthy" if all_ok else ("degraded" if core_ok else "critical"),
        "redis": "ok" if redis_ok else "down",
        "processes": checks,
        "optional_processes": optional,
        "core_process_count_expected": 7,
    }


@router.get("/task-health")
async def get_task_health_report() -> dict[str, Any]:
    """Internal task-level heartbeats — catches a process that is technically
    running (pgrep-alive) but whose actual work loop has silently stalled.

    This is distinct from /api/system/process-health, which only checks OS
    process liveness. On 2026-07-02 the order_book_collector WebSocket loop
    stopped processing messages for ~8 hours with the OS process, socket
    connection, and pgrep check all showing healthy — only a heartbeat from
    inside the loop doing real work would have caught it.
    """
    if get_shared_redis_async is None:
        return {"overall_status": "UNKNOWN", "error": "redis client unavailable"}
    try:
        from backend.services.task_health_monitor import get_task_health

        return await get_task_health(get_shared_redis_async())
    except Exception as e:
        logger.warning("task-health report failed: %s", e)
        return {"overall_status": "ERROR", "error": str(e)}


@router.get("/topology")
async def get_topology_report() -> dict[str, Any]:
    """
    Get writer lock topology report for single-writer enforcement.

    Shows active writer locks, their status, and detects duplicates or missing writers.
    """

    if get_shared_redis_sync is None or WRITER_ROLES is None:
        return {"error": "Redis client or WRITER_ROLES not available", "status": "UNAVAILABLE"}

    def _fetch_topology_sync():
        redis_client = get_shared_redis_sync()
        if redis_client is None:
            return {"error": "Redis connection not available", "status": "NO_REDIS", "result": None}
        writer_keys = list(redis_client.scan_iter(match="writer:*", count=100))
        active_writers = {}
        expected_writers = list(WRITER_ROLES.values())
        found_writers = []
        for key in writer_keys:
            try:
                key_str = to_str(key) if to_str else (key.decode("utf-8") if isinstance(key, bytes) else str(key))
                value = redis_client.get(key_str)
                value_str = to_str(value) if to_str else (value.decode("utf-8") if isinstance(value, bytes) else str(value))
                if value_str and "|" in value_str:
                    writer_id, start_ts = value_str.split("|", 1)
                    role = key_str.replace("writer:", "")
                    ttl = redis_client.ttl(key_str)
                    active_writers[role] = {
                        "writer_id": writer_id,
                        "start_timestamp": int(start_ts),
                        "start_time": datetime.fromtimestamp(int(start_ts), timezone.utc).isoformat(),
                        "ttl_seconds": ttl,
                        "status": "ACTIVE" if ttl > 0 else "EXPIRED",
                    }
                    if role in expected_writers:
                        found_writers.append(role)
            except Exception as ex:
                logger.debug("Topology report: skip malformed key %s: %s", key, ex)
        missing_writers = [role for role in expected_writers if role not in found_writers]
        duplicate_count = len(active_writers) - len(set(active_writers.keys()))
        if duplicate_count > 0:
            status, reason = "BROKEN", f"Duplicate writers detected: {duplicate_count}"
        elif missing_writers:
            status, reason = "DEGRADED", f"Missing writers: {', '.join(missing_writers)}"
        elif len(found_writers) == len(expected_writers):
            status, reason = "OK", "All expected writers present and active"
        else:
            status, reason = "UNKNOWN", "Unexpected writer configuration"
        return {
            "result": {
                "report_generated_at": datetime.now(timezone.utc).isoformat(),
                "status": status,
                "status_reason": reason,
                "topology": {"expected_writers": expected_writers, "active_writers": active_writers, "missing_writers": missing_writers, "writer_count": len(active_writers)},
                "health_check": {
                    "total_locks": len(active_writers),
                    "expected_locks": len(expected_writers),
                    "expired_locks": sum(1 for w in active_writers.values() if w["status"] == "EXPIRED"),
                    "healthy_locks": sum(1 for w in active_writers.values() if w["status"] == "ACTIVE"),
                },
            },
        }

    try:
        out = await asyncio.to_thread(_fetch_topology_sync)
        if out.get("result") is None:
            return {"error": out.get("error", "unknown"), "status": out.get("status", "ERROR")}
        return out["result"]
    except Exception as e:
        logger.warning("Topology report failed: %s", e)
        return {"error": f"Failed to get topology report: {e!s}", "status": "ERROR", "report_generated_at": datetime.now(timezone.utc).isoformat()}
