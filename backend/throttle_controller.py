#!/usr/bin/env python3
"""
Throttle Controller for Mystic Trading Platform

Simple interface to control API throttling and monitor performance.
Usage: python throttle_controller.py [command] [options]
"""

import argparse
import asyncio
import logging
import sys
import time
from typing import Any

# Optional imports - try at top level
try:
    from api_throttler import api_throttler
except (ImportError, ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError):
    api_throttler = None

from backend.services.task_manager import task_manager

try:
    from performance_monitor import performance_monitor
except (ImportError, ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError):
    performance_monitor = None

try:
    from backend.services.canonical_cache import canonical_cache
except (ImportError, ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError):
    canonical_cache = None

logger = logging.getLogger(__name__)

# Module-level task tracker for fire-and-forget tasks
_active_tasks: list[asyncio.Task[Any]] = []

# --- Small utilities ---------------------------------------------------------

# Logging functions are now properly handled by the logger


def _fmt_bool_icon(value: Any) -> str:
    return "[OK]" if bool(value) else "[NO]"


def _fmt_percent(value: Any) -> str:
    """Format success rate whether it comes as 0..1 or 0..100."""
    try:
        v = float(value)
    except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError):
        return "n/a"
    if 0.0 <= v <= 1.0:
        return f"{v:.2%}"
    return f"{v:.2f}%"


# --- Adapters to optional modules -------------------------------------------


def get_performance_dashboard() -> dict[str, Any]:
    """Get current performance dashboard (optional module)."""
    try:
        if performance_monitor is None:
            return {}
        return performance_monitor.get_performance_dashboard()
    except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError):
        logger.exception("Performance monitor not available")
        return {}


def get_api_stats() -> dict[str, Any]:
    """Get API throttling statistics (optional module)."""
    try:
        if api_throttler is None:
            return {}
        return api_throttler.get_performance_stats()
    except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError):
        logger.exception("API throttler not available")
        return {}


def increase_throttling() -> None:
    """Increase API throttling level (optional module)."""
    try:
        if api_throttler is None:
            logger.warning("API throttler not available")
            return
        api_throttler.increase_throttling()
        logger.info("Throttling increased")
    except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError):
        logger.exception("API throttler not available")


def decrease_throttling() -> None:
    """Decrease API throttling level (optional module)."""
    try:
        if api_throttler is None:
            logger.warning("API throttler not available")
            return
        api_throttler.decrease_throttling()
        logger.info("Throttling decreased")
    except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError):
        logger.exception("API throttler not available")


def optimize_system() -> None:
    """Run automatic system optimization (optional module)."""
    try:
        if performance_monitor is None:
            logger.warning("Performance monitor not available")
            return
        performance_monitor.optimize_system()
        logger.info("System optimization completed")
    except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError):
        logger.exception("Performance monitor not available")


def clear_caches() -> None:
    """Clear all caches (optional modules)."""
    try:
        if canonical_cache is None:
            logger.warning("Canonical cache service not available")
            return
    except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError):
        logger.exception("Canonical cache service not available")
        return

    # Clear canonical cache, handling sync and async clear methods
    clear_fn = getattr(canonical_cache, "clear", None)
    if clear_fn is None:
        msg = "canonical_cache has no clear() method"
        raise AttributeError(msg)
    try:
        result = clear_fn()
        if asyncio.iscoroutine(result):
            try:
                # If no running loop, run to completion
                asyncio.get_running_loop()  # Check if loop exists
            except RuntimeError:
                # No running loop; safe to run coroutine
                asyncio.run(result)
                logger.info("Canonical cache cleared")
            else:
                # Running loop present; schedule coroutine
                # Note: This is a standalone function, not a class method
                # Store task reference to track fire-and-forget tasks
                task = task_manager.create_task_sync(result, name="throttle_controller:clear_cache")
                _active_tasks.append(task)
                # Clean up completed tasks
                _active_tasks[:] = [t for t in _active_tasks if not t.done()]
                logger.info("Canonical cache clear scheduled")
        else:
            # Synchronous clear completed
            logger.info("Canonical cache cleared")
    except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
        logger.exception(f"Cache clear failed: {e}")

    logger.info("All caches cleared")


# --- UI commands -------------------------------------------------------------


def show_status() -> None:
    """Show current system status."""
    logger.info("\nSYSTEM STATUS")
    logger.info("=" * 50)

    # Performance dashboard
    dashboard = get_performance_dashboard()
    if dashboard:
        health = dashboard.get("system_health", {}) or {}
        logger.info(f"Overall Health: {health.get('overall', 'unknown')}")
        logger.info(f"Database: {_fmt_bool_icon(health.get('database'))}")
        logger.info(f"API:      {_fmt_bool_icon(health.get('api'))}")
        logger.info(f"Cache:    {_fmt_bool_icon(health.get('cache'))}")

        issues = health.get("issues") or []
        if isinstance(issues, (list, tuple)) and issues:
            logger.info(f"Issues: {', '.join(map(str, issues))}")

    # API stats
    api_stats = get_api_stats()
    if api_stats:
        avg_resp = api_stats.get("average_response_time", 0) or 0
        try:
            avg_resp = float(avg_resp)
        except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError):
            avg_resp = 0.0

        logger.info("\n[STATS] API STATISTICS")
        logger.info(f"Total Requests:        {api_stats.get('total_requests', 0)}")
        logger.info(f"Throttled Requests:    {api_stats.get('throttled_requests', 0)}")
        logger.info(f"Success Rate:          {_fmt_percent(api_stats.get('success_rate', 0))}")
        logger.info(f"Average Response Time: {avg_resp:.3f}s")
        logger.info(f"Current Level:         {api_stats.get('current_throttle_level', 'unknown')}")


def show_recommendations() -> None:
    """Show optimization recommendations."""
    logger.info("\nOPTIMIZATION RECOMMENDATIONS")
    logger.info("=" * 50)

    dashboard = get_performance_dashboard()
    if dashboard:
        recs = dashboard.get("optimization_recommendations", []) or []
        if recs:
            for i, rec in enumerate(recs, 1):
                logger.info(f"{i}. {rec}")
        else:
            logger.info("No recommendations at this time.")
    else:
        logger.info("No dashboard data available.")


def monitor_performance(duration: int = 60) -> None:
    """Monitor performance for specified duration."""
    logger.info(f"\n[STATS] MONITORING PERFORMANCE FOR {duration} SECONDS")
    logger.info("=" * 50)
    logger.info("Press Ctrl+C to stop early")

    try:
        start = time.time()
        while time.time() - start < duration:
            dashboard = get_performance_dashboard()
            if dashboard:
                health = dashboard.get("system_health", {}) or {}
                line = (
                    f"\r[{time.strftime('%H:%M:%S')}] "
                    f"Health: {health.get('overall', 'unknown')} | "
                    f"API: {_fmt_bool_icon(health.get('api'))} | "
                    f"DB:  {_fmt_bool_icon(health.get('database'))} | "
                    f"Cache: {_fmt_bool_icon(health.get('cache'))}"
                )
                print(line, end="", flush=True)
            time.sleep(5)
        logger.info("\n[OK] Monitoring completed")
    except KeyboardInterrupt:
        logger.info("\nMonitoring stopped by user")


# --- CLI --------------------------------------------------------------------


def main() -> None:
    """Main command line interface."""
    parser = argparse.ArgumentParser(description="Mystic Trading Platform Throttle Controller")
    parser.add_argument(
        "command",
        choices=[
            "status",
            "increase",
            "decrease",
            "optimize",
            "clear",
            "recommendations",
            "monitor",
        ],
        help="Command to execute",
    )
    parser.add_argument(
        "--duration",
        type=int,
        default=60,
        help="Duration for monitoring (default: 60 seconds)",
    )

    args = parser.parse_args()

    if args.command == "status":
        show_status()
    elif args.command == "increase":
        increase_throttling()
    elif args.command == "decrease":
        decrease_throttling()
    elif args.command == "optimize":
        optimize_system()
    elif args.command == "clear":
        clear_caches()
    elif args.command == "recommendations":
        show_recommendations()
    elif args.command == "monitor":
        monitor_performance(args.duration)


if __name__ == "__main__":
    if len(sys.argv) == 1:
        logger.info("[START] Mystic Trading Platform Throttle Controller")
        logger.info("=" * 50)
        logger.info("Available commands:")
        logger.info("  status          - Show current system status")
        logger.info("  increase        - Increase API throttling")
        logger.info("  decrease        - Decrease API throttling")
        logger.info("  optimize        - Run automatic optimization")
        logger.info("  clear           - Clear all caches")
        logger.info("  recommendations - Show optimization recommendations")
        logger.info("  monitor         - Monitor performance in real-time")
        logger.info("\nUsage: python throttle_controller.py [command]")
        logger.info("Example: python throttle_controller.py status")
    else:
        main()
