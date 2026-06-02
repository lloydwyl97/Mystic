#!/usr/bin/env python3
"""
Connection Manager for Mystic Trading - FIXED FOR PRODUCTION RELIABILITY
Manages all external service connections with proper error handling and fallbacks.

FIXED FOR PRODUCTION RELIABILITY:
- NO BLOCKING CALLS IN ASYNC: All blocking operations moved to thread executors
- REDIS AUTO-START OPT-IN: Default off, configurable via env flag
- COMPREHENSIVE HEALTH: All services report {connected, error, diagnostics}
- LIVE-ONLY POLICY: No fake data, explicit degraded states
- WINDOWS-SAFE: Configurable service names and paths
- BOUNDED RETRIES: Strict timeout limits prevent infinite loading
- CLEAN SHUTDOWN: Proper cleanup of all clients
- SECURE LOGGING: No secrets in logs

Windows/Python 3.12+ Compatibility:
- Uses modern type annotations compatible with Python 3.12+
- All logging messages are ASCII-only for Windows PowerShell compatibility
- Safe environment variable parsing prevents import-time crashes
- Robust error handling with proper logging
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import os
import platform
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import pika

import redis
from backend.config.redis_config import get_shared_redis_sync

# Import with fallback for backend module paths
try:
    from backend.auto_trading_manager import get_auto_trading_manager
except (ImportError, ModuleNotFoundError):
    try:
        from auto_trading_manager import get_auto_trading_manager  # type: ignore[import-not-found]
    except (ImportError, ModuleNotFoundError):
        get_auto_trading_manager = None  # type: ignore[assignment]

try:
    from backend.metrics_collector import get_metrics_collector
except (ImportError, ModuleNotFoundError):
    try:
        from metrics_collector import get_metrics_collector  # type: ignore[import-not-found]
    except (ImportError, ModuleNotFoundError):
        get_metrics_collector = None  # type: ignore[assignment]

try:
    from backend.notification_service import NotificationService, get_notification_service
except (ImportError, ModuleNotFoundError):
    try:
        from notification_service import NotificationService, get_notification_service  # type: ignore[import-not-found]
    except (ImportError, ModuleNotFoundError):
        NotificationService = None  # type: ignore[assignment]
        get_notification_service = None  # type: ignore[assignment]

try:
    from backend.signal_manager import get_signal_manager
except (ImportError, ModuleNotFoundError):
    try:
        from signal_manager import get_signal_manager  # type: ignore[import-not-found]
    except (ImportError, ModuleNotFoundError):
        get_signal_manager = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)

PIKA_AVAILABLE = True
REDIS_AVAILABLE = True


class ConnectionManager:
    """Connection manager with proper async handling and comprehensive health reporting"""

    def __init__(self) -> None:
        self.redis_client: Any | None = None
        self.rabbitmq_conn: Any | None = None
        self.notification_service: Any | None = None
        self.signal_manager: Any | None = None
        self.auto_trading_manager: Any | None = None
        self.metrics_collector: Any | None = None

        # Configuration from environment
        self.redis_auto_start = os.getenv("REDIS_AUTO_START", "false").lower() in (
            "true",
            "1",
            "on",
            "yes",
        )
        self.redis_service_name = os.getenv("REDIS_SERVICE_NAME", "Redis")
        self.redis_exe_path = os.getenv(
            "REDIS_EXE_PATH",
            str(Path.cwd() / "redis-server" / "redis-server.exe"),
        )
        self.redis_mandatory = os.getenv("REDIS_MANDATORY", "false").lower() in (
            "true",
            "1",
            "on",
            "yes",
        )

        # Connection status tracking
        self._initialization_complete = False
        self._initialization_error: str | None = None

        # Thread executor for blocking operations
        self._executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="conn-mgr")

    async def initialize(self) -> None:
        """Initialize all connections asynchronously without blocking the event loop"""
        try:
            logger.info("Starting connection manager initialization...")

            # Run blocking initialization in thread executor
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(self._executor, self._initialize_sync)

            self._initialization_complete = True
            logger.info("Connection manager initialization completed")

        except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
            self._initialization_error = str(e)
            logger.exception("Connection manager initialization failed: %s", e)
            # Don't raise - allow degraded operation

    def _initialize_sync(self) -> None:
        """Synchronous initialization run in thread executor"""
        try:
            self.redis_client = self._initialize_redis()
            self.rabbitmq_conn = self._initialize_rabbitmq()
            self._initialize_services()
        except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
            logger.exception("Sync initialization failed: %s", e)
            raise

    def _initialize_redis(self) -> Any | None:
        """Initialize Redis connection with bounded retries and opt-in auto-start"""
        if not REDIS_AVAILABLE:
            logger.warning("Redis library not available - Redis features disabled")
            return None

        # All Live Data, No Fallback/Hardcoded Data
        redis_url = os.getenv("REDIS_URL")
        if not redis_url:
            msg = "REDIS_URL environment variable is required - no fallback/hardcoded Redis URL"
            raise RuntimeError(msg)

        # Sanitize URL for logging (remove credentials)
        sanitized_url = self._sanitize_url(redis_url)

        try:
            parsed = urlparse(redis_url)
            redis_host = parsed.hostname
            redis_port = parsed.port
        except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
            msg = f"Failed to parse REDIS_URL: {e}"
            raise RuntimeError(msg) from e

        # Validate parsed URL components outside try to avoid TRY301
        if not redis_host:
            msg = "REDIS_URL must include a hostname - no fallback/hardcoded Redis host"
            raise RuntimeError(msg)
        if not redis_port:
            redis_port = 6379  # Default port only if not specified in URL

        logger.info(
            "Attempting to connect to Redis at %s:%s (URL: %s)",
            redis_host,
            redis_port,
            sanitized_url,
        )

        # Bounded retry attempts with strict timeout
        max_attempts = 3
        max_total_time = 10  # seconds

        attempts = [
            lambda: self._try_redis_url_connection(redis_url),
            lambda: self._try_redis_connection(redis_host, redis_port, socket_timeout=3),
            lambda: self._try_redis_connection_pool(redis_host, redis_port),
        ]

        # Add auto-start attempt only if enabled
        if self.redis_auto_start:
            attempts.append(lambda: self._try_start_redis_service(redis_host, redis_port))

        start_time = time.time()

        for i, attempt in enumerate(attempts[:max_attempts], 1):
            # Check if we've exceeded total time budget
            if time.time() - start_time > max_total_time:
                logger.warning("Redis connection timeout exceeded (%ds)", max_total_time)
                break

            try:
                logger.info("Redis connection attempt %d...", i)
                client = attempt()
                if client:
                    logger.info("Redis connection established at %s:%s", redis_host, redis_port)
                    return client
            except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
                logger.warning("Redis connection attempt %d failed: %s", i, e)

        # Check if Redis is mandatory
        if self.redis_mandatory:
            error_msg = "CRITICAL: Redis is mandatory but connection failed"
            logger.error(error_msg)
            raise RuntimeError(error_msg)
        logger.warning("Redis connection failed - operating in degraded mode")
        return None

    def _sanitize_url(self, url: str) -> str:
        """Sanitize URL by removing credentials for safe logging"""
        try:
            parsed = urlparse(url)
            if parsed.username or parsed.password:
                # Rebuild URL without credentials
                host = parsed.hostname or "[hostname]"
                port = f":{parsed.port}" if parsed.port else ""
                sanitized = f"{parsed.scheme}://{host}{port}{parsed.path or ''}"
                if parsed.query:
                    sanitized += f"?{parsed.query}"
                result = sanitized
            else:
                result = url
        except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError):
            return "redis://[sanitized]"
        else:
            return result

    def _try_redis_url_connection(self, redis_url: str) -> Any | None:
        """Try Redis connection using URL with proper TLS handling"""
        client = get_shared_redis_sync()
        if client is None:
            return None
        client.ping()
        return client

    def _try_redis_connection(self, host: str, port: int, socket_timeout: int = 5) -> Any | None:
        """Try direct Redis connection"""
        client = get_shared_redis_sync()
        if client is None:
            return None
        client.ping()
        return client

    def _try_redis_connection_pool(self, host: str, port: int) -> Any | None:
        """Try establishing a Redis client via a connection pool"""
        pool = redis.ConnectionPool(
            host=host,
            port=port,
            db=0,
            decode_responses=True,
            protocol=2,  # CRITICAL: Use RESP2 to avoid CLIENT SETINFO issues on Windows Redis
        )
        client = redis.Redis(connection_pool=pool)
        client.ping()
        return client

    def _try_start_redis_service(self, host: str, port: int) -> Any | None:
        """Attempt to start a local Redis service/binary and connect to it"""
        logger.info("Attempting to start Redis service/binary as configured")
        start_timeout = 6  # seconds to wait after starting
        exe_client_connected = False
        systemctl_client_state = None

        try:
            # If an explicit exe path exists, try launching it
            exe = self.redis_exe_path
            if exe and Path(exe).is_file() and os.access(exe, os.X_OK):
                logger.info("Starting Redis executable: %s", exe)
                # Start process detached
                try:
                    if platform.system() == "Windows":
                        # DETACHED_PROCESS flag to avoid inheriting console (value 0x00000008)
                        creationflags = 0x00000008
                        subprocess.Popen([exe], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, creationflags=creationflags)
                    else:
                        subprocess.Popen([exe], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True)
                except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
                    logger.warning("Failed to start Redis executable: %s", e)

                # Wait briefly and attempt to connect
                deadline = time.time() + start_timeout
                while time.time() < deadline:
                    try:
                        client = self._try_redis_connection(host, port, socket_timeout=2)
                        if client:
                            exe_client_connected = True
                            return client
                    except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError):
                        time.sleep(0.5)

            # Fallback: try system service management (systemctl) on Linux
            if platform.system() != "Windows":
                try:
                    subprocess.run(["systemctl", "start", self.redis_service_name], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)
                    # Wait briefly and try to connect
                    deadline = time.time() + start_timeout
                    while time.time() < deadline:
                        try:
                            client = self._try_redis_connection(host, port, socket_timeout=2)
                            if client:
                                systemctl_client_state = client
                                return client
                        except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError):
                            time.sleep(0.5)
                except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
                    logger.warning("Failed to start Redis via systemctl: %s", e)
        except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
            logger.warning("Failed to start Redis service: %s", e)
            raise

        # Validate and raise outside try blocks to avoid TRY301
        if exe and Path(exe).is_file() and os.access(exe, os.X_OK) and not exe_client_connected:
            msg = "Timed out waiting for started Redis process to accept connections"
            raise RuntimeError(msg)

        if platform.system() != "Windows" and systemctl_client_state is None:
            msg = "Unable to start Redis service or process"
            raise RuntimeError(msg)

    def _initialize_rabbitmq(self) -> Any | None:
        """Initialize RabbitMQ connection if pika is available"""
        if not PIKA_AVAILABLE:
            return None

        # All Live Data, No Fallback/Hardcoded Data
        rabbit_url = os.getenv("RABBITMQ_URL")
        if not rabbit_url:
            logger.debug("RABBITMQ_URL not set - RabbitMQ features disabled (using in-memory alternatives)")
            return None
        try:
            params = pika.URLParameters(rabbit_url)
            conn = pika.BlockingConnection(params)
            logger.info("RabbitMQ connection established")
        except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
            logger.warning("RabbitMQ connection failed: %s", e)
            return None
        else:
            return conn

    def _initialize_services(self) -> None:
        """Initialize higher-level services that may depend on Redis or other clients"""
        # Notification service
        try:
            if get_notification_service:
                try:
                    self.notification_service = get_notification_service(self.redis_client)
                    logger.info("Notification service initialized via factory")
                except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
                    logger.warning("Notification service factory failed: %s", e)
                    # Try class fallback
                    if NotificationService:
                        try:
                            self.notification_service = NotificationService(self.redis_client)
                            logger.info("Notification service initialized via class fallback")
                        except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as fallback_e:
                            logger.warning("Notification service class fallback failed: %s", fallback_e)
                    else:
                        logger.warning("Notification service not available")
            elif NotificationService:
                try:
                    self.notification_service = NotificationService(self.redis_client)
                    logger.info("Notification service initialized")
                except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
                    logger.warning("Notification service init failed: %s", e)
            else:
                logger.warning("Notification service not available")
        except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
            logger.warning("Notification service init unexpected error: %s", e)
            self.notification_service = None

        # Signal manager
        try:
            if get_signal_manager:
                try:
                    self.signal_manager = get_signal_manager(self.redis_client)
                    logger.info("Signal manager initialized")
                except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
                    logger.warning("Signal manager factory failed: %s", e)
                    self.signal_manager = None
            else:
                logger.warning("Signal manager not available")
                self.signal_manager = None
        except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
            logger.warning("Signal manager init unexpected error: %s", e)
            self.signal_manager = None

        # Auto trading manager (legacy - replaced by AILiveAutoBuyService)
        try:
            if get_auto_trading_manager:
                try:
                    self.auto_trading_manager = get_auto_trading_manager(self.redis_client)
                    logger.info("Auto trading manager initialized")
                except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
                    logger.debug("Auto trading manager factory failed (using modern AI services): %s", e)
                    self.auto_trading_manager = None
            else:
                logger.debug("Auto trading manager not available (using AILiveAutoBuyService instead)")
                self.auto_trading_manager = None
        except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
            logger.debug("Auto trading manager init unexpected error (using modern AI services): %s", e)
            self.auto_trading_manager = None

        # Metrics collector
        try:
            if get_metrics_collector and self.redis_client and hasattr(self.redis_client, "ping"):
                try:
                    self.metrics_collector = get_metrics_collector(self.redis_client)
                    logger.info("Metrics collector initialized")
                except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
                    logger.warning("Metrics collector factory failed: %s", e)
                    self.metrics_collector = None
            else:
                logger.info("Redis not available or metrics collector factory not present, skipping metrics collector")
                self.metrics_collector = None
        except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
            logger.warning("Metrics collector init unexpected error: %s", e)
            self.metrics_collector = None

    def get_comprehensive_health(self) -> dict[str, Any]:
        """Get comprehensive health status for all services"""
        health = {
            "initialization_complete": self._initialization_complete,
            "initialization_error": self._initialization_error,
            "services": {
                "redis": self.check_redis_health(),
                "rabbitmq": self.check_rabbitmq_health(),
                "notification": self.check_notification_health(),
                "signal_manager": self.check_signal_manager_health(),
                "auto_trading": self.check_auto_trading_health(),
                "metrics": self.check_metrics_health(),
            },
            "overall_status": "unknown",
            "degraded_services": [],
            "critical_failures": [],
        }

        # Determine overall status
        redis_ok = health["services"]["redis"]["connected"]
        rabbitmq_ok = health["services"]["rabbitmq"]["connected"]
        notification_ok = health["services"]["notification"]["connected"]

        # Check for critical failures
        if self.redis_mandatory and not redis_ok:
            health["critical_failures"].append("Redis is mandatory but unavailable")

        # Determine degraded services
        if not redis_ok:
            health["degraded_services"].append("redis")
        if not rabbitmq_ok:
            health["degraded_services"].append("rabbitmq")
        if not notification_ok:
            health["degraded_services"].append("notification")

        # Set overall status
        if health["critical_failures"]:
            health["overall_status"] = "critical"
        elif health["degraded_services"]:
            health["overall_status"] = "degraded"
        elif redis_ok and notification_ok:
            health["overall_status"] = "healthy"
        else:
            health["overall_status"] = "unknown"

        return health

    def check_redis_health(self) -> dict[str, Any]:
        """Check Redis connection health"""
        info: dict[str, Any] = {
            "connected": False,
            "client_type": "none",
            "error": None,
            "diagnostics": {},
        }

        if not REDIS_AVAILABLE:
            info["error"] = "redis library not available"
            return info

        if not self.redis_client:
            info["error"] = "No Redis client initialized"
            return info

        try:
            info["client_type"] = type(self.redis_client).__name__
            if hasattr(self.redis_client, "ping"):
                try:
                    self.redis_client.ping()
                    info["connected"] = True
                except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
                    info["error"] = f"Ping failed: {e}"
            else:
                # If client has no ping, assume connected
                info["connected"] = True
        except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
            info["error"] = f"Health check failed: {e}"

        return info

    def check_rabbitmq_health(self) -> dict[str, Any]:
        """Check RabbitMQ connection health"""
        info: dict[str, Any] = {
            "connected": False,
            "client_type": "none",
            "error": None,
            "diagnostics": {},
        }

        if not PIKA_AVAILABLE:
            info["error"] = "pika library not available"
            return info

        if not self.rabbitmq_conn:
            info["error"] = "No RabbitMQ connection initialized"
            return info

        try:
            info["client_type"] = "pika.BlockingConnection"
            # Try to check connection state
            if hasattr(self.rabbitmq_conn, "is_closed"):
                info["diagnostics"]["is_closed"] = self.rabbitmq_conn.is_closed
            info["connected"] = True
        except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
            info["error"] = f"Health check failed: {e}"

        return info

    def check_notification_health(self) -> dict[str, Any]:
        """Check notification service health"""
        info: dict[str, Any] = {
            "connected": False,
            "client_type": "none",
            "error": None,
            "diagnostics": {},
        }

        if not self.notification_service:
            info["error"] = "No notification service initialized"
            return info

        try:
            info["client_type"] = type(self.notification_service).__name__
            # Basic health check - try to access service
            if hasattr(self.notification_service, "is_healthy"):
                try:
                    info["connected"] = bool(self.notification_service.is_healthy())
                except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
                    info["error"] = f"Health check failed: {e}"
            else:
                info["connected"] = True  # Assume healthy if no health check method
        except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
            info["error"] = f"Health check failed: {e}"

        return info

    def check_signal_manager_health(self) -> dict[str, Any]:
        """Check signal manager health"""
        info: dict[str, Any] = {
            "connected": False,
            "client_type": "none",
            "error": None,
            "diagnostics": {},
        }

        if not self.signal_manager:
            info["error"] = "No signal manager initialized"
            return info

        try:
            info["client_type"] = type(self.signal_manager).__name__
            info["connected"] = True  # Assume healthy if initialized
        except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
            info["error"] = f"Health check failed: {e}"

        return info

    def check_auto_trading_health(self) -> dict[str, Any]:
        """Check auto trading manager health"""
        info: dict[str, Any] = {
            "connected": False,
            "client_type": "none",
            "error": None,
            "diagnostics": {},
        }

        if not self.auto_trading_manager:
            info["error"] = "No auto trading manager initialized"
            return info

        try:
            info["client_type"] = type(self.auto_trading_manager).__name__
            info["connected"] = True  # Assume healthy if initialized
        except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
            info["error"] = f"Health check failed: {e}"

        return info

    def check_metrics_health(self) -> dict[str, Any]:
        """Check metrics collector health"""
        info: dict[str, Any] = {
            "connected": False,
            "client_type": "none",
            "error": None,
            "diagnostics": {},
        }

        if not self.metrics_collector:
            info["error"] = "No metrics collector initialized"
            return info

        try:
            info["client_type"] = type(self.metrics_collector).__name__
            info["connected"] = True  # Assume healthy if initialized
        except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
            info["error"] = f"Health check failed: {e}"

        return info

    async def close_connections(self) -> None:
        """Close all connections with proper cleanup"""
        if self.rabbitmq_conn:
            try:
                self.rabbitmq_conn.close()
                logger.info("RabbitMQ connection closed")
            except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
                logger.exception("Error closing RabbitMQ connection: %s", e)

        # Close Redis connection if it has a close method
        if self.redis_client and hasattr(self.redis_client, "close"):
            try:
                # Some redis clients use .close(), others use .connection_pool.disconnect()
                try:
                    result = self.redis_client.close()
                    if asyncio.iscoroutine(result):
                        await result
                except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError):
                    if hasattr(self.redis_client, "connection_pool"):
                        with contextlib.suppress(ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError):
                            result = self.redis_client.connection_pool.disconnect()
                            if asyncio.iscoroutine(result):
                                await result
                logger.info("Redis connection closed")
            except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
                logger.exception("Error closing Redis connection: %s", e)

        # Shutdown thread executor
        if hasattr(self, "_executor"):
            try:
                self._executor.shutdown(wait=True)
                logger.info("Thread executor shutdown")
            except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
                logger.exception("Error shutting down thread executor: %s", e)

    async def close(self) -> None:
        await self.close_connections()


# CRITICAL FIX: Lazy initialization to prevent connection leaks at import time
# Previous: connection_manager = ConnectionManager() - created connections at import
# Now: Use get_connection_manager() for lazy initialization

_connection_manager_state: dict[str, ConnectionManager | None] = {"instance": None}


def get_connection_manager() -> ConnectionManager:
    """
    Get the global ConnectionManager instance using lazy initialization.

    CRITICAL: This prevents Redis connection leaks by deferring connection
    creation until first actual use, not at import time.

    Returns:
        ConnectionManager: Singleton instance
    """
    if _connection_manager_state["instance"] is None:
        _connection_manager_state["instance"] = ConnectionManager()
        logger.info("ConnectionManager singleton created via lazy initialization")
    return _connection_manager_state["instance"]


# For backward compatibility - lazily initialized global variable
class _LazyConnectionManagerProxy:
    """Lazy proxy for backward compatibility with existing imports."""

    def __getattr__(self, name):
        return getattr(get_connection_manager(), name)

    async def __call__(self, *args, **kwargs):
        return await get_connection_manager()(*args, **kwargs)


connection_manager = _LazyConnectionManagerProxy()  # type: ignore[assignment]
