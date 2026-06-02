"""
HTTP client adapters.
"""

from .base import HTTPClient

try:
    from .httpx_client import HttpxHTTPClient
except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError):
    # If httpx or the httpx-based client isn't available, don't fail importing the package.
    HttpxHTTPClient = None

__all__ = ["HTTPClient"]
if HttpxHTTPClient is not None:
    __all__.append("HttpxHTTPClient")
