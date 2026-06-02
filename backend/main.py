# CRITICAL: Force IPv4 BEFORE any other imports (Binance US requirement)
import socket as _socket

_original_getaddrinfo = _socket.getaddrinfo


def _force_ipv4_getaddrinfo(*args, **kwargs):
    """Filter to only return IPv4 addresses."""
    responses = _original_getaddrinfo(*args, **kwargs)
    ipv4_responses = [r for r in responses if r[0] == _socket.AF_INET]
    return ipv4_responses if ipv4_responses else responses


_socket.getaddrinfo = _force_ipv4_getaddrinfo

from backend.app_factory import create_app

# Create the app first
app = create_app()

# Orchestrator will be started manually after app startup

# ServiceManager initializes inside backend.app_factory lifespan to ensure async wiring happens there.
