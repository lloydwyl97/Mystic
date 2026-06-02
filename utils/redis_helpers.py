"""
Redis Helpers - Live Configuration Only

Lightweight helpers to normalize Redis return types across sync/async clients
and different configurations (bytes vs str), so downstream JSON parsing and
string operations are safe.
All configuration values come from live config - no hardcoded values.

Usage:
- Use to_str(...) for single-value reads like lpop/get
- Use to_str_list([...]) for list reads like lrange
"""

from __future__ import annotations

import os
from collections.abc import Iterable
from typing import Any

# Import live configuration
try:
    from backend.config_bridge import get_mystic_config

    _mystic_config = get_mystic_config()
except (ImportError, AttributeError, ValueError, TypeError, RuntimeError):
    _mystic_config = None

# --- Live Configuration Helpers -------------------------------------------------------------------


def _get_redis_encoding() -> str:
    """Get Redis encoding from live config."""
    if _mystic_config:
        try:
            value = _mystic_config
            if hasattr(value, "redis") and hasattr(value.redis, "encoding"):
                encoding = value.redis.encoding
                if isinstance(encoding, str) and encoding:
                    return encoding.strip()
        except (AttributeError, ValueError, TypeError):
            pass

    encoding = os.getenv("REDIS_ENCODING", "").strip()
    if encoding:
        return encoding

    return "utf-8"


def _get_redis_decode_errors() -> str:
    """Get Redis decode error handling mode from live config."""
    if _mystic_config:
        try:
            value = _mystic_config
            if hasattr(value, "redis") and hasattr(value.redis, "decode_errors"):
                errors = value.redis.decode_errors
                if isinstance(errors, str) and errors:
                    return errors.strip()
        except (AttributeError, ValueError, TypeError):
            pass

    errors = os.getenv("REDIS_DECODE_ERRORS", "").strip()
    if errors:
        return errors

    return "ignore"


def to_str(value: Any) -> str | None:
    """Return value as str, decoding bytes with configured encoding; None stays None.

    This avoids bytes passed into json.loads or string operations.
    """
    if value is None:
        return None
    if isinstance(value, bytes):
        encoding = _get_redis_encoding()
        decode_errors = _get_redis_decode_errors()
        try:
            return value.decode(encoding)
        except (UnicodeDecodeError, AttributeError, TypeError):
            # Fallback with configured error handling to avoid hard failures on rare bad bytes
            return value.decode(encoding, errors=decode_errors)
    if isinstance(value, str):
        return value
    # For non-bytes, non-str (rare for Redis), coerce to str explicitly
    return str(value)


def to_str_list(values: Any) -> list[str]:
    """Normalize a Redis list result (possibly list[bytes|str]) to list[str]."""
    if values is None:
        return []
    if not isinstance(values, Iterable) or isinstance(values, str | bytes):
        # Defensive: unexpected scalar; coerce to single-element list
        s = to_str(values)
        return [s] if s is not None else []
    result: list[str] = []
    for item in values:
        s = to_str(item)
        if s is not None:
            result.append(s)
    return result
