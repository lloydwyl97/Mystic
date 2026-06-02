"""
Force IPv4 for all network connections.
Import this module at the very start of any script that uses network connections.
"""

import socket

_original_getaddrinfo = socket.getaddrinfo


def _ipv4_only_getaddrinfo(*args, **kwargs):
    """Filter to only return IPv4 addresses."""
    responses = _original_getaddrinfo(*args, **kwargs)
    ipv4_responses = [r for r in responses if r[0] == socket.AF_INET]
    return ipv4_responses if ipv4_responses else responses


# Apply the patch
socket.getaddrinfo = _ipv4_only_getaddrinfo
