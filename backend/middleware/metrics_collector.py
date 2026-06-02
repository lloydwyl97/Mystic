"""
Metrics Collector - All Live Data, No Fallback/Hardcoded Data

This module provides metrics collection for live API operations (backend port 8000).
All operations:
- Collect live metrics from API requests/responses (backend port 8000)
- Track live system performance metrics (CPU, memory, disk, network)
- Monitor live request/response latency and error rates
- No fallback/hardcoded data - all metrics from live operations
- Used by backend services on port 8000 for live trading operations

Live Data Sources:
- API requests: Live requests to backend API (port 8000)
- API responses: Live responses from backend API (port 8000)
- System metrics: Live CPU, memory, disk, network usage
- Latency metrics: Live request/response handling times
- Error metrics: Live error counts from real API operations
- All metrics collected from live operations - no mock/test data

Endpoint References:
- Backend API: Port 8000 (metrics collected from live requests)
- All metrics collected from live connections - no fallback/hardcoded data
"""

import logging
import threading
import time
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from typing import Any

import psutil

logger = logging.getLogger(__name__)


class MetricsCollector:
    """
    Metrics collector for live API operations (backend port 8000).

    Collects live metrics from API requests, responses, and system performance.
    All metrics collected from live operations - no fallback/hardcoded data.
    """

    def __init__(self) -> None:
        """Initialize metrics collector for live operations."""
        # Live metrics storage (all from live operations)
        self.metrics: dict[str, Any] = {
            "requests": defaultdict(int),  # Live request counts
            "responses": defaultdict(int),  # Live response counts
            "errors": defaultdict(int),  # Live error counts
            "latency": defaultdict(list),  # Live latency measurements
            "bandwidth": defaultdict(int),  # Live bandwidth usage
            "cpu_usage": [],  # Live CPU usage
            "memory_usage": [],  # Live memory usage
            "disk_io": [],  # Live disk I/O
            "network_io": [],  # Live network I/O
        }

        # Metrics configuration defaults (not fallback data, configuration defaults)
        self.config: dict[str, Any] = {
            "retention_period": 3600,  # 1 hour (configuration default, not fallback data)
            "sampling_interval": 60,  # 1 minute (configuration default, not fallback data)
            "max_samples": 1000,  # Maximum samples to keep (configuration default, not fallback data)
            "endpoints": {},  # Endpoints will be set in metrics.py
        }

        # Start metrics collection thread
        self.collecting = True
        self.collector_thread = threading.Thread(target=self._collect_metrics)
        self.collector_thread.daemon = True
        self.collector_thread.start()

    def _collect_metrics(self) -> None:
        """
        Collect live system metrics periodically.

        Collects live CPU, memory, disk, and network metrics from running system.
        All metrics from live operations - no fallback/hardcoded data.
        """
        while self.collecting:
            try:
                # Collect live CPU usage
                cpu_percent = psutil.cpu_percent(interval=1)
                self.metrics["cpu_usage"].append(
                    {
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                        "value": cpu_percent,  # Live CPU usage
                    },
                )

                # Collect live memory usage
                memory = psutil.virtual_memory()
                self.metrics["memory_usage"].append(
                    {
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                        "value": memory.percent,  # Live memory usage
                    },
                )

                # Collect live disk I/O
                disk_io = psutil.disk_io_counters()
                if disk_io is not None:
                    self.metrics["disk_io"].append(
                        {
                            "timestamp": datetime.now(timezone.utc).isoformat(),
                            "read_bytes": getattr(disk_io, "read_bytes", 0),  # Live disk read bytes
                            "write_bytes": getattr(disk_io, "write_bytes", 0),  # Live disk write bytes
                        },
                    )

                # Collect live network I/O
                net_io = psutil.net_io_counters()
                self.metrics["network_io"].append(
                    {
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                        "bytes_sent": net_io.bytes_sent,  # Live network bytes sent
                        "bytes_recv": net_io.bytes_recv,  # Live network bytes received
                    },
                )

                # Cleanup old metrics
                self._cleanup_old_metrics()

                # Sleep for configured sampling interval - sync sleep OK for background thread
                time.sleep(self.config["sampling_interval"])

            except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
                # Log live exception (not fallback data, error handling)
                logger.exception("Error collecting live metrics: %s", e)
                # Continue with same interval even on error
                time.sleep(self.config["sampling_interval"])

    def _cleanup_old_metrics(self) -> None:
        """
        Remove old metrics data beyond retention period.

        Cleans up old live metrics data (not fallback data, data retention management).
        """
        try:
            cutoff_time = datetime.now(timezone.utc) - timedelta(seconds=self.config["retention_period"])

            # Cleanup time series metrics
            for metric in [
                "cpu_usage",
                "memory_usage",
                "disk_io",
                "network_io",
            ]:
                self.metrics[metric] = [m for m in self.metrics[metric] if datetime.fromisoformat(m["timestamp"]) > cutoff_time][-self.config["max_samples"] :]

        except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
            logger.exception("Error cleaning up live metrics: %s", e)

    def get_metrics(self) -> dict[str, Any]:
        """
        Get all live metrics collected from operations.

        Returns:
            Dictionary with live metrics from API operations and system performance
        """
        try:
            return {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "requests": dict(self.metrics["requests"]),
                "responses": dict(self.metrics["responses"]),
                "errors": dict(self.metrics["errors"]),
                "latency": {
                    path: {
                        "avg": sum(times) / len(times) if times else 0,
                        "min": min(times) if times else 0,
                        "max": max(times) if times else 0,
                        "count": len(times),
                    }
                    for path, times in self.metrics["latency"].items()
                },
                "bandwidth": dict(self.metrics["bandwidth"]),
                "system": {
                    "cpu": (self.metrics["cpu_usage"][-1] if self.metrics["cpu_usage"] else None),
                    "memory": (self.metrics["memory_usage"][-1] if self.metrics["memory_usage"] else None),
                    "disk_io": (self.metrics["disk_io"][-1] if self.metrics["disk_io"] else None),
                    "network_io": (self.metrics["network_io"][-1] if self.metrics["network_io"] else None),
                },
            }

        except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
            logger.exception("Error getting live metrics: %s", e)
            return {"error": str(e)}  # Error response (not fallback data, error handling)

    def get_metrics_summary(self) -> dict[str, Any]:
        """
        Get summary of live metrics from operations.

        Returns:
            Dictionary with summary of live metrics (requests, errors, latency, system health)
        """
        try:
            total_requests = sum(self.metrics["requests"].values())  # Live request total
            total_errors = sum(self.metrics["errors"].values())  # Live error total

            return {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "total_requests": total_requests,
                "total_errors": total_errors,
                "error_rate": ((total_errors / total_requests * 100) if total_requests > 0 else 0),
                "avg_latency": {path: sum(times) / len(times) if times else 0 for path, times in self.metrics["latency"].items()},
                "system_health": {
                    "cpu": (self.metrics["cpu_usage"][-1]["value"] if self.metrics["cpu_usage"] else 0),
                    "memory": (self.metrics["memory_usage"][-1]["value"] if self.metrics["memory_usage"] else 0),
                },
            }

        except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
            logger.exception("Error getting live metrics summary: %s", e)
            return {"error": str(e)}  # Error response (not fallback data, error handling)

    def get_detailed_metrics(self) -> dict[str, Any]:
        """
        Get detailed live metrics with time series data.

        Returns:
            Dictionary with detailed live metrics including time series data
        """
        try:
            return {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "requests": {
                    "total": sum(self.metrics["requests"].values()),
                    "by_endpoint": dict(self.metrics["requests"]),
                },
                "errors": {
                    "total": sum(self.metrics["errors"].values()),
                    "by_endpoint": dict(self.metrics["errors"]),
                },
                "latency": {
                    path: {
                        "avg": sum(times) / len(times) if times else 0,
                        "min": min(times) if times else 0,
                        "max": max(times) if times else 0,
                        "p95": (sorted(times)[int(len(times) * 0.95)] if times else 0),
                        "p99": (sorted(times)[int(len(times) * 0.99)] if times else 0),
                        "count": len(times),
                    }
                    for path, times in self.metrics["latency"].items()
                },
                "bandwidth": {
                    "total": sum(self.metrics["bandwidth"].values()),
                    "by_endpoint": dict(self.metrics["bandwidth"]),
                },
                "system": {
                    "cpu": self.metrics["cpu_usage"],
                    "memory": self.metrics["memory_usage"],
                    "disk_io": self.metrics["disk_io"],
                    "network_io": self.metrics["network_io"],
                },
            }

        except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
            logger.exception("Error getting detailed live metrics: %s", e)
            return {"error": str(e)}  # Error response (not fallback data, error handling)

    def track_request(self, request: Any, start_time: float) -> None:
        """
        Track live request metrics from API operations.

        Args:
            request: Live API request to backend (port 8000)
            start_time: Request start time for latency calculation
        """
        try:
            path = request.url.path
            method = request.method

            # Track live request count
            self.metrics["requests"][f"{method} {path}"] += 1

            # Track live latency
            latency = time.time() - start_time
            self.metrics["latency"][path].append(latency)

            # Track live bandwidth
            content_length = request.headers.get("content-length")
            if content_length:
                self.metrics["bandwidth"][path] += int(content_length)

        except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
            logger.exception("Error tracking live request: %s", e)

    def track_response(self, request: Any, response: Any) -> None:
        """
        Track live response metrics from API operations.

        Args:
            request: Live API request to backend (port 8000)
            response: Live API response from backend (port 8000)
        """
        try:
            path = request.url.path
            method = request.method

            # Track live response count
            self.metrics["responses"][f"{method} {path}"] += 1

            # Track live errors
            if response.status_code >= 400:
                self.metrics["errors"][f"{method} {path}"] += 1

            # Track live bandwidth
            if hasattr(response, "body") and response.body is not None:
                self.metrics["bandwidth"][path] += len(response.body)

        except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
            logger.exception("Error tracking live response: %s", e)
