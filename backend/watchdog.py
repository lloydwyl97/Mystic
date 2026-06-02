"""
Health Watchdog & Auto-Recovery System - Live Configuration Only

Monitors trading services and automatically restarts failed components.
All configuration values come from live config - no hardcoded values.
"""

from __future__ import annotations

import json
import logging
import os
import socket
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# Import live configuration
try:
    from backend.config_bridge import get_mystic_config

    _mystic_config = get_mystic_config()
except (ImportError, AttributeError, ValueError, TypeError, RuntimeError):
    _mystic_config = None

try:
    import psutil  # type: ignore[import-untyped]
except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
    msg = "psutil is required by watchdog. Please install it."
    raise RuntimeError(msg) from e
try:
    import httpx  # type: ignore[import-untyped]
except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError):
    httpx = None  # We'll skip HTTP checks if unavailable

logger = logging.getLogger("mystic.watchdog")


class TradingWatchdog:
    """
    Advanced watchdog system for trading services.
    All configuration values come from live config.

    Notes:
      - Uses process name matching by script substring (simple & effective for your layout).
      - Avoids blocking on child process pipes (redirects to DEVNULL).
      - Fixes undefined variable bug during restart logging.
      - Uses graceful process termination with fallback to kill.
    """

    def __init__(self, check_interval: int | None = None, max_restart_attempts: int | None = None) -> None:
        """
        Initialize watchdog.

        Args:
            check_interval: Seconds between health checks (overrides live config if provided).
            max_restart_attempts: Maximum restart attempts per service (overrides live config if provided).
        """
        # Load from live config if not provided
        if check_interval is None:
            check_interval = _get_check_interval()
        if max_restart_attempts is None:
            max_restart_attempts = _get_max_restart_attempts()

        self.check_interval = check_interval
        self.max_restart_attempts = max_restart_attempts
        self.service_status: dict[str, dict[str, Any]] = {}
        self.restart_history: list[dict[str, Any]] = []
        self.health_log: list[dict[str, Any]] = []

        # Define critical services via live config or env JSON
        # Each service entry may include: script (abs path), host, port, scheme, health_endpoint,
        # acceptable_status (list), restart_command, max_memory_mb, max_cpu_percent
        self.critical_services: dict[str, dict[str, Any]] = {}
        self._load_services_config()

    def _load_services_config(self) -> None:
        """Load services configuration from live config or environment variables."""
        # Try live config first
        if _mystic_config is not None:
            try:
                value = getattr(_mystic_config, "watchdog", None)
                if value and hasattr(value, "services_json"):
                    services_json = value.services_json
                    if isinstance(services_json, str) and services_json:
                        try:
                            self.critical_services = json.loads(services_json)
                        except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
                            logger.exception("Failed to parse services_json from live config: %s", e)
                        else:
                            return
            except (AttributeError, ValueError, TypeError):
                pass
        # Fallback to environment variable
        services_json = os.getenv("WATCHDOG_SERVICES_JSON")
        if services_json:
            try:
                self.critical_services = json.loads(services_json)
            except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
                logger.exception("Failed to parse WATCHDOG_SERVICES_JSON: %s", e)
                self.critical_services = {}
        if not self.critical_services:
            logger.warning("No services configured for watchdog (WATCHDOG_SERVICES_JSON not set).")

    def _reload_config(self) -> None:
        """Reload configuration values from live config."""
        self.check_interval = _get_check_interval()
        self.max_restart_attempts = _get_max_restart_attempts()
        self._load_services_config()

    # ---------------------------
    # Public checks
    # ---------------------------
    def check_service_health(self, service_name: str) -> dict[str, Any]:
        """Check health of a specific service."""
        if service_name not in self.critical_services:
            return {"status": "unknown", "error": "Service not configured"}

        cfg = self.critical_services[service_name]

        process_running = self._is_process_running(cfg.get("script", ""))
        port_healthy = True
        if cfg.get("port"):
            try:
                port_healthy = self._check_port_health(int(cfg.get("port")))
            except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError):
                port_healthy = False

        http_healthy = True
        http_check_skipped = False
        if cfg.get("health_endpoint"):
            if httpx is None:
                http_check_skipped = True
                http_healthy = True  # neutral
            else:
                default_host = _get_default_host()
                default_scheme = _get_default_scheme()
                default_http_timeout = _get_default_http_timeout()
                default_acceptable_status = _get_default_acceptable_status()
                http_healthy = self._check_http_health(
                    host=str(cfg.get("host", default_host)),
                    port=cfg.get("port"),
                    endpoint=str(cfg.get("health_endpoint")),
                    scheme=str(cfg.get("scheme", default_scheme)),
                    acceptable_statuses=list(cfg.get("acceptable_status", [])) or default_acceptable_status,
                    timeout=float(cfg.get("http_timeout", default_http_timeout)),
                )

        resource_usage = self._check_resource_usage(cfg.get("script", ""))

        # Resource policy is optional
        mem_limit = cfg.get("max_memory_mb")
        cpu_limit = cfg.get("max_cpu_percent")
        try:
            memory_ok = True if mem_limit is None else (resource_usage["memory_mb"] <= float(mem_limit))
        except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError):
            memory_ok = False
        try:
            cpu_ok = True if cpu_limit is None else (resource_usage["cpu_percent"] <= float(cpu_limit))
        except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError):
            cpu_ok = False

        # overall_healthy includes resource limits only when configured
        overall_healthy = process_running and port_healthy and http_healthy and memory_ok and cpu_ok

        # Build check breakdown
        checks = {
            "process": "ok" if process_running else "fail",
            "port": ("skipped" if cfg.get("port") in (None, "") else ("ok" if port_healthy else "fail")),
            "http": ("skipped" if http_check_skipped or not cfg.get("health_endpoint") else ("ok" if http_healthy else "fail")),
            "resources": ("skipped" if (mem_limit is None and cpu_limit is None) else ("ok" if (memory_ok and cpu_ok) else "fail")),
        }
        failures = []
        if not process_running:
            failures.append("process_not_running")
        if checks["port"] == "fail":
            failures.append("port_down")
        if checks["http"] == "fail":
            failures.append("http_unhealthy")
        if checks["resources"] == "fail":
            failures.append("resource_limits_exceeded")

        health_status = {
            "service": service_name,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "check_id": f"chk-{int(time.time() * 1000)}",
            "overall_healthy": overall_healthy,
            "process_running": process_running,
            "port_healthy": port_healthy,
            "http_healthy": http_healthy,
            "http_check_skipped": http_check_skipped,
            "memory_ok": memory_ok,
            "cpu_ok": cpu_ok,
            "resource_usage": resource_usage,
            "not_found": (resource_usage.get("pid") is None),
            "checks": checks,
            "status": ("healthy" if overall_healthy else ("degraded" if process_running else "down")),
            "failures": failures,
        }
        self.service_status[service_name] = health_status
        return health_status

    # ---------------------------
    # Internals: checks
    # ---------------------------
    def _is_process_running(self, script_name: str) -> bool:
        """Check if a Python script is running."""
        try:
            for proc in psutil.process_iter(["pid", "name", "cmdline"]):
                cmd = proc.info.get("cmdline") or []
                if cmd and script_name and script_name in " ".join(cmd):
                    return True
        except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
            logger.exception("Process check error: %s", e)
            return False
        else:
            return False

    def _check_port_health(self, port: int, host: str | None = None) -> bool:
        """Check if a TCP port is listening on host."""
        if host is None:
            host = _get_default_host()
        port_timeout = _get_port_check_timeout()
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
                sock.settimeout(port_timeout)
                return sock.connect_ex((host, port)) == 0
        except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
            logger.warning("Port %s check error: %s", port, e)
            return False

    def _check_http_health(
        self,
        *,
        host: str,
        port: int | None,
        endpoint: str,
        scheme: str | None = None,
        acceptable_statuses: list[int] | None = None,
        timeout: float | None = None,
    ) -> bool:
        """Check HTTP/HTTPS health endpoint synchronously; returns False if port is None or client missing."""
        if port is None or httpx is None:
            return False
        if scheme is None:
            scheme = _get_default_scheme()
        if timeout is None:
            timeout = _get_default_http_timeout()
        if acceptable_statuses is None:
            acceptable_statuses = _get_default_acceptable_status()
        try:
            url = f"{scheme}://{host}:{int(port)}{endpoint}"
            with httpx.Client() as client:
                r = client.get(url, timeout=timeout)
            ok_set = set(acceptable_statuses)
            return int(r.status_code) in ok_set
        except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
            logger.warning("HTTP health check error (%s): %s", endpoint, e)
            return False

    def _check_resource_usage(self, script_name: str) -> dict[str, float]:
        """Check resource usage for a process matching the script."""
        try:
            for proc in psutil.process_iter(["pid", "name", "cmdline"]):
                cmd = proc.info.get("cmdline") or []
                if cmd and script_name and script_name in " ".join(cmd):
                    p = psutil.Process(proc.info["pid"])
                    # Get a more meaningful CPU sample
                    cpu_interval = _get_cpu_check_interval()
                    cpu = p.cpu_percent(interval=cpu_interval)
                    mem_mb = p.memory_info().rss / (1024 * 1024)
                    return {
                        "memory_mb": round(mem_mb, 2),
                        "cpu_percent": round(cpu, 1),
                        "pid": proc.info["pid"],
                    }
        except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
            logger.exception("Resource usage check error: %s", e)
            return {"memory_mb": 0.0, "cpu_percent": 0.0, "pid": None}
        else:
            return {"memory_mb": 0.0, "cpu_percent": 0.0, "pid": None}

    # ---------------------------
    # Restart flow
    # ---------------------------
    def restart_service(self, service_name: str) -> dict[str, Any]:
        """Restart a failed service."""
        if service_name not in self.critical_services:
            return {"success": False, "error": "Service not configured"}

        cfg = self.critical_services[service_name]

        restart_count = self._get_restart_count(service_name)
        if restart_count >= self.max_restart_attempts:
            return {
                "success": False,
                "error": f"Max restart attempts ({self.max_restart_attempts}) exceeded",
            }

        try:
            self._kill_process(cfg.get("script", ""))
            # Brief delay to ensure process terminates cleanly
            restart_delay = _get_restart_delay_sec()
            time.sleep(restart_delay)  # Sync sleep OK for process management script

            # Avoid blocking on pipes; let the child inherit stdio or drop it
            # Resolve command: prefer provided, else use current interpreter + script
            cmd = cfg.get("restart_command")
            if not cmd:
                script = str(Path(str(cfg.get("script", ""))).resolve())
                cmd = [sys.executable, script]
            # Set cwd to script directory if not provided
            script_path = Path(str(cfg.get("script", ""))).resolve()
            cwd = cfg.get("cwd") or (str(script_path.parent) if script_path else None)
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                stdin=subprocess.DEVNULL,
                close_fds=True,
                cwd=cwd,
            )

            # post-restart health window with backoff retries
            backoff_total = _get_post_restart_window_sec()
            step = _get_health_check_retry_step_sec()  # Step for health check retries
            waited = 0.0
            health_check = {}
            while waited < backoff_total:
                # Incremental health check during backoff - sync sleep OK for monitoring script
                time.sleep(step)
                waited += step
                health_check = self.check_service_health(service_name)
                # consider http skipped as neutral
                if health_check.get("overall_healthy") or (health_check.get("http_check_skipped") and health_check.get("process_running") and health_check.get("port_healthy")):
                    break

            result = {
                "service": service_name,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "restart_attempt": restart_count + 1,
                "process_id": getattr(process, "pid", None),
                "success": bool(health_check.get("overall_healthy")),
                "health_status": health_check,
                "restart_id": f"rst-{int(time.time() * 1000)}",
            }
            self._log_restart(result)

            if result["success"]:
                logger.info("[OK] Restarted %s (pid=%s)", service_name, result.get("process_id"))
            else:
                logger.error("[X] Restart failed for %s", service_name)
        except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
            err = {
                "service": service_name,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "success": False,
                "error": str(e),
            }
            self._log_restart(err)
            logger.exception("Restart error for %s: %s", service_name, e)
            return err
        else:
            return result

    def _kill_process(self, script_name: str) -> None:
        """Terminate (then kill if needed) a process by script name."""
        try:
            for proc in psutil.process_iter(["pid", "name", "cmdline"]):
                cmd = proc.info.get("cmdline") or []
                if cmd and script_name and script_name in " ".join(cmd):
                    p = psutil.Process(proc.info["pid"])
                    logger.warning(
                        "Terminating process %s (pid=%s) for %s",
                        p.name(),
                        p.pid,
                        script_name,
                    )
                    p.terminate()
                    process_wait_timeout = _get_process_wait_timeout_sec()
                    try:
                        p.wait(timeout=process_wait_timeout)
                    except psutil.TimeoutExpired:
                        logger.warning("Terminate timed out; killing pid=%s", p.pid)
                        p.kill()
        except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
            logger.exception("Error killing process for %s: %s", script_name, e)

    def _get_restart_count(self, service_name: str) -> int:
        """Get number of restart attempts for a service (this run)."""
        return sum(1 for r in self.restart_history if r.get("service") == service_name)

    def _log_restart(self, restart_result: dict[str, Any]) -> None:
        """Log restart attempt to memory (append-only)."""
        self.restart_history.append(restart_result)

    # ---------------------------
    # Monitoring
    # ---------------------------
    def monitor_all_services(self) -> dict[str, Any]:
        """Monitor health of all critical services and attempt recovery."""
        logger.info("Checking health of %d services...", len(self.critical_services))

        results: dict[str, Any] = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "services_checked": len(self.critical_services),
            "healthy_services": 0,
            "unhealthy_services": 0,
            "restarts_performed": 0,
            "service_status": {},
        }

        for name in self.critical_services:
            status = self.check_service_health(name)
            results["service_status"][name] = status

            if status["overall_healthy"]:
                results["healthy_services"] += 1
                logger.info("[OK] %s: Healthy", name)
            else:
                results["unhealthy_services"] += 1
                logger.warning("[WARN] %s: Unhealthy", name)

                restart_result = self.restart_service(name)
                if restart_result.get("success"):
                    results["restarts_performed"] += 1
                    logger.info("[RESTART] %s: Restarted successfully", name)
                else:
                    error_msg = restart_result.get("error", "Unknown error")
                    logger.error("[RESTART-FAIL] %s: %s", name, error_msg)

        self.health_log.append(results)
        return results

    def get_system_summary(self) -> dict[str, Any]:
        """Get overall system health summary."""
        if not self.service_status:
            return {"status": "unknown", "services": 0}

        healthy_count = sum(1 for s in self.service_status.values() if s.get("overall_healthy"))
        total = len(self.service_status)
        health_pct = round((healthy_count / total) * 100, 1) if total else 0.0

        service_uptimes = {name: ("running" if s.get("overall_healthy") else "down") for name, s in self.service_status.items()}

        return {
            "overall_health": "healthy" if healthy_count == total else "degraded",
            "healthy_services": healthy_count,
            "total_services": total,
            "health_percentage": health_pct,
            "service_uptimes": service_uptimes,
            "last_check": datetime.now(timezone.utc).isoformat(),
        }

    def get_health_history(self, limit: int | None = None) -> list[dict[str, Any]]:
        """Get health check history."""
        if limit is None:
            limit = _get_health_history_default_limit()
        return self.health_log[-limit:]

    def get_restart_history(self, limit: int | None = None) -> list[dict[str, Any]]:
        """Get restart history."""
        if limit is None:
            limit = _get_restart_history_default_limit()
        return self.restart_history[-limit:]

    def run_continuous_monitoring(self) -> None:
        """Run continuous monitoring loop (Ctrl+C to stop)."""
        logger.info("Starting continuous monitoring...")
        logger.info(
            "Check interval: %s sec | Max restart attempts: %s",
            self.check_interval,
            self.max_restart_attempts,
        )

        while True:
            try:
                results = self.monitor_all_services()
                summary = self.get_system_summary()

                logger.info(
                    "System Summary: %s | Healthy %d/%d (%.1f%%) | Restarts this pass: %d",
                    summary["overall_health"].upper(),
                    summary["healthy_services"],
                    summary["total_services"],
                    summary["health_percentage"],
                    results["restarts_performed"],
                )
                # Regular check interval - sync sleep OK for standalone monitoring script
                time.sleep(self.check_interval)

            except KeyboardInterrupt:
                logger.info("Monitoring stopped by user")
                break
            except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
                logger.exception("Monitoring error: %s", e)
                # Continue with same interval on error
                time.sleep(self.check_interval)


# Convenience functions
def check_service_health(service_name: str) -> dict[str, Any]:
    """Check health of a specific service."""
    return TradingWatchdog().check_service_health(service_name)


def restart_service(service_name: str) -> dict[str, Any]:
    """Restart a specific service."""
    return TradingWatchdog().restart_service(service_name)


def monitor_services() -> dict[str, Any]:
    """Run a single monitoring pass across all services."""
    return TradingWatchdog().monitor_all_services()


def get_system_health() -> dict[str, Any]:
    """Get system health summary."""
    return TradingWatchdog().get_system_summary()


# ------------------------------------------------------------------------------
# Configuration helpers (live config)
# ------------------------------------------------------------------------------
def _get_check_interval() -> int:
    """Get check interval from live configuration."""
    if _mystic_config is not None:
        try:
            value = getattr(_mystic_config, "watchdog", None)
            if value and hasattr(value, "check_interval_sec"):
                interval = value.check_interval_sec
                if isinstance(interval, int) and interval > 0:
                    return interval
        except (AttributeError, ValueError, TypeError):
            pass
    # Fallback to environment variable
    try:
        value = int(os.getenv("WATCHDOG_CHECK_INTERVAL_SEC", "60"))
        return max(1, value)
    except (ValueError, TypeError):
        return 60


def _get_max_restart_attempts() -> int:
    """Get max restart attempts from live configuration."""
    if _mystic_config is not None:
        try:
            value = getattr(_mystic_config, "watchdog", None)
            if value and hasattr(value, "max_restart_attempts"):
                attempts = value.max_restart_attempts
                if isinstance(attempts, int) and attempts > 0:
                    return attempts
        except (AttributeError, ValueError, TypeError):
            pass
    # Fallback to environment variable
    try:
        value = int(os.getenv("WATCHDOG_MAX_RESTART_ATTEMPTS", "3"))
        return max(1, value)
    except (ValueError, TypeError):
        return 3


def _get_default_host() -> str:
    """Get default host from live configuration."""
    if _mystic_config is not None:
        try:
            value = getattr(_mystic_config, "watchdog", None)
            if value and hasattr(value, "default_host"):
                host = value.default_host
                if isinstance(host, str) and host:
                    return host.strip()
        except (AttributeError, ValueError, TypeError):
            pass
    # Fallback to environment variable
    host = os.getenv("WATCHDOG_DEFAULT_HOST", "127.0.0.1").strip()
    return host if host else "127.0.0.1"


def _get_default_scheme() -> str:
    """Get default scheme from live configuration."""
    if _mystic_config is not None:
        try:
            value = getattr(_mystic_config, "watchdog", None)
            if value and hasattr(value, "default_scheme"):
                scheme = value.default_scheme
                if isinstance(scheme, str) and scheme in ("http", "https"):
                    return scheme
        except (AttributeError, ValueError, TypeError):
            pass
    # Fallback to environment variable
    scheme = os.getenv("WATCHDOG_DEFAULT_SCHEME", "http").strip().lower()
    return scheme if scheme in ("http", "https") else "http"


def _get_default_http_timeout() -> float:
    """Get default HTTP timeout from live configuration."""
    if _mystic_config is not None:
        try:
            value = getattr(_mystic_config, "watchdog", None)
            if value and hasattr(value, "default_http_timeout_sec"):
                timeout = value.default_http_timeout_sec
                if isinstance(timeout, (int, float)) and timeout > 0:
                    return float(timeout)
        except (AttributeError, ValueError, TypeError):
            pass
    # Fallback to environment variable
    try:
        value = float(os.getenv("WATCHDOG_DEFAULT_HTTP_TIMEOUT_SEC", "3.0"))
        return max(0.1, value)
    except (ValueError, TypeError):
        return 3.0


def _get_default_acceptable_status() -> list[int]:
    """Get default acceptable HTTP status codes from live configuration."""
    if _mystic_config is not None:
        try:
            value = getattr(_mystic_config, "watchdog", None)
            if value and hasattr(value, "default_acceptable_status"):
                status = value.default_acceptable_status
                if isinstance(status, list) and all(isinstance(x, int) for x in status):
                    return status
        except (AttributeError, ValueError, TypeError):
            pass
    # Fallback to environment variable or default range
    status_str = os.getenv("WATCHDOG_DEFAULT_ACCEPTABLE_STATUS", "").strip()
    if status_str:
        try:
            return [int(x.strip()) for x in status_str.split(",") if x.strip()]
        except (ValueError, TypeError):
            pass
    return list(range(200, 300))


def _get_port_check_timeout() -> float:
    """Get port check timeout from live configuration."""
    if _mystic_config is not None:
        try:
            value = getattr(_mystic_config, "watchdog", None)
            if value and hasattr(value, "port_check_timeout_sec"):
                timeout = value.port_check_timeout_sec
                if isinstance(timeout, (int, float)) and timeout > 0:
                    return float(timeout)
        except (AttributeError, ValueError, TypeError):
            pass
    # Fallback to environment variable
    try:
        value = float(os.getenv("WATCHDOG_PORT_CHECK_TIMEOUT_SEC", "1.5"))
        return max(0.1, value)
    except (ValueError, TypeError):
        return 1.5


def _get_cpu_check_interval() -> float:
    """Get CPU check interval from live configuration."""
    if _mystic_config is not None:
        try:
            value = getattr(_mystic_config, "watchdog", None)
            if value and hasattr(value, "cpu_check_interval_sec"):
                interval = value.cpu_check_interval_sec
                if isinstance(interval, (int, float)) and interval > 0:
                    return float(interval)
        except (AttributeError, ValueError, TypeError):
            pass
    # Fallback to environment variable
    try:
        value = float(os.getenv("WATCHDOG_CPU_CHECK_INTERVAL_SEC", "0.15"))
        return max(0.01, value)
    except (ValueError, TypeError):
        return 0.15


def _get_restart_delay_sec() -> float:
    """Get restart delay from live configuration."""
    if _mystic_config is not None:
        try:
            value = getattr(_mystic_config, "watchdog", None)
            if value and hasattr(value, "restart_delay_sec"):
                delay = value.restart_delay_sec
                if isinstance(delay, (int, float)) and delay >= 0:
                    return float(delay)
        except (AttributeError, ValueError, TypeError):
            pass
    # Fallback to environment variable
    try:
        value = float(os.getenv("WATCHDOG_RESTART_DELAY_SEC", "1.5"))
        return max(0.0, value)
    except (ValueError, TypeError):
        return 1.5


def _get_post_restart_window_sec() -> float:
    """Get post-restart window from live configuration."""
    if _mystic_config is not None:
        try:
            value = getattr(_mystic_config, "watchdog", None)
            if value and hasattr(value, "post_restart_window_sec"):
                window = value.post_restart_window_sec
                if isinstance(window, (int, float)) and window > 0:
                    return float(window)
        except (AttributeError, ValueError, TypeError):
            pass
    # Fallback to environment variable
    try:
        value = float(os.getenv("WATCHDOG_POST_RESTART_WINDOW_SEC", "10"))
        return max(0.1, value)
    except (ValueError, TypeError):
        return 10.0


def _get_health_check_retry_step_sec() -> float:
    """Get health check retry step from live configuration."""
    if _mystic_config is not None:
        try:
            value = getattr(_mystic_config, "watchdog", None)
            if value and hasattr(value, "health_check_retry_step_sec"):
                step = value.health_check_retry_step_sec
                if isinstance(step, (int, float)) and step > 0:
                    return float(step)
        except (AttributeError, ValueError, TypeError):
            pass
    # Fallback to environment variable
    try:
        value = float(os.getenv("WATCHDOG_HEALTH_CHECK_RETRY_STEP_SEC", "2"))
        return max(0.1, value)
    except (ValueError, TypeError):
        return 2.0


def _get_process_wait_timeout_sec() -> float:
    """Get process wait timeout from live configuration."""
    if _mystic_config is not None:
        try:
            value = getattr(_mystic_config, "watchdog", None)
            if value and hasattr(value, "process_wait_timeout_sec"):
                timeout = value.process_wait_timeout_sec
                if isinstance(timeout, (int, float)) and timeout > 0:
                    return float(timeout)
        except (AttributeError, ValueError, TypeError):
            pass
    # Fallback to environment variable
    try:
        value = float(os.getenv("WATCHDOG_PROCESS_WAIT_TIMEOUT_SEC", "10"))
        return max(0.1, value)
    except (ValueError, TypeError):
        return 10.0


def _get_health_history_default_limit() -> int:
    """Get health history default limit from live configuration."""
    if _mystic_config is not None:
        try:
            value = getattr(_mystic_config, "watchdog", None)
            if value and hasattr(value, "health_history_default_limit"):
                limit = value.health_history_default_limit
                if isinstance(limit, int) and limit > 0:
                    return limit
        except (AttributeError, ValueError, TypeError):
            pass
    # Fallback to environment variable
    try:
        value = int(os.getenv("WATCHDOG_HEALTH_HISTORY_DEFAULT_LIMIT", "100"))
        return max(1, value)
    except (ValueError, TypeError):
        return 100


def _get_restart_history_default_limit() -> int:
    """Get restart history default limit from live configuration."""
    if _mystic_config is not None:
        try:
            value = getattr(_mystic_config, "watchdog", None)
            if value and hasattr(value, "restart_history_default_limit"):
                limit = value.restart_history_default_limit
                if isinstance(limit, int) and limit > 0:
                    return limit
        except (AttributeError, ValueError, TypeError):
            pass
    # Fallback to environment variable
    try:
        value = int(os.getenv("WATCHDOG_RESTART_HISTORY_DEFAULT_LIMIT", "50"))
        return max(1, value)
    except (ValueError, TypeError):
        return 50


if __name__ == "__main__":
    # One quick pass + loop
    wd = TradingWatchdog()
    wd.monitor_all_services()
    wd.run_continuous_monitoring()
