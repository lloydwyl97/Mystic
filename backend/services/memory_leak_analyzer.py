"""
Memory Leak Analysis Tool - Identifies the source of memory leaks.

This utility analyzes memory growth patterns and identifies which objects
are consuming the most memory, helping pinpoint memory leak sources.
"""

import asyncio
import gc
import logging
import os
import tracemalloc
from datetime import datetime, timezone
from typing import Any

logger = logging.getLogger("backend.memory_leak_analyzer")


class MemoryLeakAnalyzer:
    """Analyze memory usage patterns to identify leaks."""

    def __init__(self, interval_minutes: int = 30) -> None:
        """
        Initialize the analyzer.

        Args:
            interval_minutes: How often to take memory snapshots (default 30 min)
        """
        self._interval = max(5, interval_minutes)
        self._snapshots: list[dict[str, Any]] = []
        self._running = False

    async def analyze_snapshot(self) -> dict[str, Any]:
        """Take a memory snapshot and analyze growth."""
        try:
            gc.collect()
            tracemalloc.start()

            # Get top memory allocations
            snapshot = tracemalloc.take_snapshot()
            top_stats = snapshot.statistics("lineno")

            timestamp = datetime.now(timezone.utc).isoformat()
            allocations = []

            for stat in top_stats[:20]:
                allocations.append(
                    {
                        "file": stat.traceback[0].filename if stat.traceback else "unknown",
                        "line": stat.traceback[0].lineno if stat.traceback else 0,
                        "size_mb": stat.size / (1024 * 1024),
                        "count": stat.count,
                    }
                )

            snapshot_data = {
                "timestamp": timestamp,
                "total_mb": sum(s["size_mb"] for s in allocations),
                "allocations": allocations,
            }

            self._snapshots.append(snapshot_data)

            # Keep only last 24 hours worth (50 snapshots @ 30min interval)
            if len(self._snapshots) > 50:
                self._snapshots = self._snapshots[-50:]

            return snapshot_data

        except Exception as exc:
            logger.exception(f"Error analyzing memory: {exc}")
            return {}

    async def get_memory_growth(self) -> dict[str, Any]:
        """Analyze memory growth over time."""
        if len(self._snapshots) < 2:
            return {"error": "Not enough snapshots yet"}

        first = self._snapshots[0]
        latest = self._snapshots[-1]

        growth_mb = latest["total_mb"] - first["total_mb"]
        growth_percent = (growth_mb / max(first["total_mb"], 1)) * 100
        time_span_hours = len(self._snapshots) * self._interval / 60

        # Identify top growing files
        file_growth = {}
        for latest_alloc in latest["allocations"]:
            file_key = f"{latest_alloc['file']}:{latest_alloc['line']}"
            file_growth[file_key] = latest_alloc["size_mb"]

        analysis = {
            "total_growth_mb": round(growth_mb, 2),
            "growth_percent": round(growth_percent, 1),
            "time_span_hours": round(time_span_hours, 1),
            "growth_rate_mb_per_hour": round(growth_mb / max(time_span_hours, 0.1), 2),
            "top_allocations": latest["allocations"][:10],
            "snapshots_count": len(self._snapshots),
        }

        return analysis

    async def get_status(self) -> dict[str, Any]:
        """Get current analysis status."""
        growth = await self.get_memory_growth()
        return {
            "running": self._running,
            "snapshots_collected": len(self._snapshots),
            "memory_growth": growth,
        }


# Global singleton instance
memory_leak_analyzer = MemoryLeakAnalyzer()


async def start_memory_analysis():
    """Start periodic memory analysis (for manual use)."""
    logger.info("Starting periodic memory analysis...")

    interval_minutes = int(os.getenv("MEM_ANALYSIS_INTERVAL_MIN", "30"))
    analyzer = MemoryLeakAnalyzer(interval_minutes)

    while True:
        try:
            logger.info("Taking memory snapshot for analysis...")
            snapshot = await analyzer.analyze_snapshot()

            if snapshot:
                logger.info(f"Memory snapshot - Total: {snapshot['total_mb']:.2f} MB, Top allocation: {snapshot['allocations'][0]['file'] if snapshot['allocations'] else 'N/A'}")

            # Show growth if we have enough data
            if len(analyzer._snapshots) >= 3:
                growth = await analyzer.get_memory_growth()
                logger.info(f"Memory growth analysis - Total: {growth['total_growth_mb']:.2f} MB, Rate: {growth['growth_rate_mb_per_hour']:.2f} MB/hour")

            await asyncio.sleep(interval_minutes * 60)

        except asyncio.CancelledError:
            # BUG #48 FIX: Clean exit on cancellation
            logger.info("Memory analysis shutting down")
            break
        except Exception as exc:
            logger.exception(f"Error in memory analysis loop: {exc}")
            await asyncio.sleep(60)


def get_top_memory_objects(limit: int = 20) -> list[dict[str, Any]]:
    """
    Get top memory-consuming objects (for debugging).

    Args:
        limit: Number of top objects to return

    Returns:
        List of objects with size and type information
    """
    try:
        gc.collect()

        # Get all objects
        objects = {}
        for obj in gc.get_objects():
            obj_type = type(obj).__name__
            obj_size = object.__sizeof__(obj)

            if obj_type not in objects:
                objects[obj_type] = {"size": 0, "count": 0}

            objects[obj_type]["size"] += obj_size
            objects[obj_type]["count"] += 1

        # Sort by size
        sorted_objects = sorted(objects.items(), key=lambda x: x[1]["size"], reverse=True)

        result = []
        for obj_type, stats in sorted_objects[:limit]:
            result.append(
                {
                    "type": obj_type,
                    "size_mb": stats["size"] / (1024 * 1024),
                    "count": stats["count"],
                }
            )

        return result

    except Exception as exc:
        logger.exception(f"Error getting top objects: {exc}")
        return []
