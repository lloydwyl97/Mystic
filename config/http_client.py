"""
Optimized HTTP client configuration for Mystic Trading Platform.
Connected to live configuration and integrated with HTTP services.
"""

import os

import aiohttp
from aiohttp import ClientTimeout

# Import live configuration
try:
    from backend.config_bridge import get_mystic_config

    _mystic_config = get_mystic_config()
except (ImportError, AttributeError, ValueError, TypeError, RuntimeError):
    _mystic_config = None


def _get_connector_limit() -> int:
    """Get connector limit from live configuration."""
    try:
        value = int(os.getenv("HTTP_CONNECTOR_LIMIT", "100"))
        return max(1, value)
    except (ValueError, TypeError):
        return 100


def _get_connector_limit_per_host() -> int:
    """Get connector limit per host from live configuration."""
    try:
        value = int(os.getenv("HTTP_CONNECTOR_LIMIT_PER_HOST", "20"))
        return max(1, value)
    except (ValueError, TypeError):
        return 20


def _get_dns_cache_ttl() -> int:
    """Get DNS cache TTL from live configuration."""
    try:
        value = int(os.getenv("HTTP_DNS_CACHE_TTL", "300"))
        return max(1, value)
    except (ValueError, TypeError):
        return 300


def _get_timeout_total() -> float:
    """Get total timeout from live configuration."""
    try:
        value = float(os.getenv("HTTP_TIMEOUT_TOTAL", "30"))
        return max(1.0, value)
    except (ValueError, TypeError):
        return 30.0


def _get_timeout_connect() -> float:
    """Get connection timeout from live configuration."""
    try:
        value = float(os.getenv("HTTP_TIMEOUT_CONNECT", "5"))
        return max(1.0, value)
    except (ValueError, TypeError):
        return 5.0


def _get_timeout_sock_connect() -> float:
    """Get socket connection timeout from live configuration."""
    try:
        value = float(os.getenv("HTTP_TIMEOUT_SOCK_CONNECT", "5"))
        return max(1.0, value)
    except (ValueError, TypeError):
        return 5.0


def _get_timeout_sock_read() -> float:
    """Get socket read timeout from live configuration."""
    try:
        value = float(os.getenv("HTTP_TIMEOUT_SOCK_READ", "15"))
        return max(1.0, value)
    except (ValueError, TypeError):
        return 15.0


def _get_retry_total() -> int:
    """Get retry total from live configuration."""
    try:
        value = int(os.getenv("HTTP_RETRY_TOTAL", "3"))
        return max(0, value)
    except (ValueError, TypeError):
        return 3


def _get_retry_delay() -> float:
    """Get retry delay from live configuration."""
    try:
        value = float(os.getenv("HTTP_RETRY_DELAY", "1.0"))
        return max(0.0, value)
    except (ValueError, TypeError):
        return 1.0


def _get_connector_settings() -> dict[str, int | bool | None]:
    """Get connector settings from live configuration."""
    return {
        "limit": _get_connector_limit(),
        "limit_per_host": _get_connector_limit_per_host(),
        "enable_cleanup_closed": True,
        "force_close": False,
        "use_dns_cache": True,
        "ttl_dns_cache": _get_dns_cache_ttl(),
        "ssl": None,
    }


def _get_timeout_settings() -> dict[str, float]:
    """Get timeout settings from live configuration."""
    return {
        "total": _get_timeout_total(),
        "connect": _get_timeout_connect(),
        "sock_connect": _get_timeout_sock_connect(),
        "sock_read": _get_timeout_sock_read(),
    }


def _get_retry_settings() -> dict[str, int | float | list[int]]:
    """Get retry settings from live configuration."""
    return {
        "total": _get_retry_total(),
        "retry_delay": _get_retry_delay(),
        "status_forcelist": [500, 502, 503, 504],
    }


# Connector settings (dynamically loaded from config)
CONNECTOR_SETTINGS = _get_connector_settings()

# Timeout settings (dynamically loaded from config)
TIMEOUT_SETTINGS = _get_timeout_settings()

# Retry settings (dynamically loaded from config)
RETRY_SETTINGS = _get_retry_settings()

# Shared global connector for connection pooling
_SHARED_CONNECTOR: aiohttp.TCPConnector | None = None


def get_shared_connector() -> aiohttp.TCPConnector:
    """Get or create a shared connection pool"""
    global _SHARED_CONNECTOR
    if _SHARED_CONNECTOR is None or _SHARED_CONNECTOR.closed:
        # Reload settings to ensure live config is used
        settings = _get_connector_settings()
        _SHARED_CONNECTOR = aiohttp.TCPConnector(**settings)
    return _SHARED_CONNECTOR


async def get_client_session(timeout_settings: dict[str, float] | None = None) -> aiohttp.ClientSession:
    """Get optimized aiohttp client session using connection pooling"""
    # Use provided timeout settings or reload from live config
    timeout_dict = timeout_settings if timeout_settings is not None else _get_timeout_settings()
    timeout = ClientTimeout(**timeout_dict)

    # Create session with shared connector
    connector = get_shared_connector()
    return aiohttp.ClientSession(connector=connector, timeout=timeout)


async def close_shared_connector() -> None:
    """Close shared connector gracefully"""
    global _SHARED_CONNECTOR
    if _SHARED_CONNECTOR and not _SHARED_CONNECTOR.closed:
        await _SHARED_CONNECTOR.close()
        _SHARED_CONNECTOR = None
