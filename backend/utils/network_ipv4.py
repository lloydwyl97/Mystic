"""
Single shared IPv4-only DNS patch (Binance.US requires IPv4; IPv6 requests get
``{"code":-71012,"msg":"IPv6 not supported"}``).

Previously seven separate modules (backend/main.py, backend/app_factory.py,
backend/services/order_book_collector.py, backend/services/live_trading_service.py,
backend/services/binance_user_stream.py, backend/services/binance_scalp/market_reader.py,
backend/services/binance_scalp/strategies/kline_cache.py) each independently
monkey-patched ``socket.getaddrinfo`` at import time, each capturing its own
"original" reference. Import order therefore determined which module's
"original" was actually the true original vs. an already-patched wrapper —
fragile and impossible to reason about or restore cleanly.

This module is the one shared bootstrap location. ``ensure_ipv4_only()`` is
idempotent: calling it from multiple modules (in any import order) patches
``socket.getaddrinfo`` exactly once and is a safe no-op afterward.
"""

from __future__ import annotations

import logging
import socket as _socket

logger = logging.getLogger(__name__)

_PATCH_MARKER = "_mystic_ipv4_only_patch"


def ensure_ipv4_only() -> None:
    """Idempotently force all DNS resolution to IPv4 (AF_INET) only.

    Falls back to the unfiltered result if IPv4 lookup fails or returns
    nothing, so a genuinely IPv6-only host (never expected for Binance.US)
    does not hard-fail name resolution.
    """
    if getattr(_socket.getaddrinfo, _PATCH_MARKER, False):
        return  # already patched by an earlier import — no-op

    original_getaddrinfo = _socket.getaddrinfo

    def _ipv4_only_getaddrinfo(host, port, family=0, type=0, proto=0, flags=0):  # noqa: A002 (match socket signature)
        try:
            results = original_getaddrinfo(host, port, _socket.AF_INET, type, proto, flags)
            if results:
                return results
        except OSError:
            pass
        # Fall back to the caller's requested family (or unfiltered) rather than
        # hard-failing resolution for a host that genuinely has no IPv4 record.
        return original_getaddrinfo(host, port, family, type, proto, flags)

    setattr(_ipv4_only_getaddrinfo, _PATCH_MARKER, True)
    _socket.getaddrinfo = _ipv4_only_getaddrinfo
    logger.info("NETWORK_IPV4_ONLY: socket.getaddrinfo patched once (shared bootstrap)")


__all__ = ["ensure_ipv4_only"]
