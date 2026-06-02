import socket

try:
    import httpx.packages.urllib3.util.connection as urllib3_cn
except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError):
    # Fallback if httpx does not vendor urllib3
    import urllib3.util.connection as urllib3_cn


def _ipv4_only():
    # Force urllib3 / requests to use IPv4 addresses
    def allowed_gai_family():
        return socket.AF_INET

    # Monkeypatch the urllib3 connection module to prefer IPv4.
    # Use setattr to be safe in environments where direct assignment might fail.
    urllib3_cn.allowed_gai_family = allowed_gai_family


_ipv4_only()
