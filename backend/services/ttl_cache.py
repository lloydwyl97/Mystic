"""
Simple TTL (Time-To-Live) Cache - LIVE data caching with expiration
"""

import time
from typing import Any


class TTLCache:
    """Simple time-based cache with automatic expiration"""

    def __init__(self, ttl_seconds: int = 300) -> None:
        """
        Initialize TTL cache

        Args:
            ttl_seconds: Time-to-live in seconds (default 300 = 5 minutes)
        """
        self.ttl_seconds = ttl_seconds
        self._cache: dict[str, tuple[Any, float]] = {}

    def get(self, key: str) -> Any | None:
        """Get value from cache if not expired"""
        if key not in self._cache:
            return None

        value, timestamp = self._cache[key]

        # Check if expired
        if time.time() - timestamp > self.ttl_seconds:
            del self._cache[key]
            return None

        return value

    def set(self, key: str, value: Any) -> None:
        """Set value in cache with current timestamp"""
        self._cache[key] = (value, time.time())

    def delete(self, key: str) -> None:
        """Delete value from cache"""
        if key in self._cache:
            del self._cache[key]

    def clear(self) -> None:
        """Clear all cached values"""
        self._cache.clear()

    def cleanup_expired(self) -> int:
        """
        Proactively remove expired entries from cache.

        Returns:
            Number of entries removed
        """
        current_time = time.time()
        expired_keys = [key for key, (_, timestamp) in self._cache.items() if current_time - timestamp > self.ttl_seconds]

        for key in expired_keys:
            del self._cache[key]

        return len(expired_keys)

    def __contains__(self, key: str) -> bool:
        """Check if key exists and is not expired"""
        return self.get(key) is not None

    def __getitem__(self, key: str) -> Any:
        """Get value using [] notation"""
        value = self.get(key)
        if value is None:
            raise KeyError(key)
        return value

    def __setitem__(self, key: str, value: Any) -> None:
        """Set value using [] notation"""
        self.set(key, value)
