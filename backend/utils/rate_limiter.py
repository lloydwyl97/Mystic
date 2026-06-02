#!/usr/bin/env python3
"""
Rate Limiter for API Operations
DEFECT-024 FIX: Prevent excessive order placement
"""

import time
from collections import deque
from threading import Lock


class RateLimiter:
    """Token bucket rate limiter for API operations"""

    def __init__(self, max_requests: int = 10, window_seconds: float = 60.0):
        """
        Initialize rate limiter

        Args:
            max_requests: Maximum requests allowed in the time window
            window_seconds: Time window in seconds
        """
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.requests: deque = deque(maxlen=max_requests)
        self.lock = Lock()

    def is_allowed(self) -> bool:
        """
        Check if a request is allowed under rate limits

        Returns:
            True if request is allowed, False if rate limited
        """
        with self.lock:
            current_time = time.time()

            # Remove old requests outside the window
            while self.requests and self.requests[0] < current_time - self.window_seconds:
                self.requests.popleft()

            # Check if we're under the limit
            if len(self.requests) < self.max_requests:
                self.requests.append(current_time)
                return True

            return False

    def wait_time(self) -> float:
        """
        Get seconds to wait before next request is allowed

        Returns:
            Seconds to wait (0 if request allowed now)
        """
        with self.lock:
            if len(self.requests) < self.max_requests:
                return 0.0

            current_time = time.time()
            oldest_request = self.requests[0]
            wait_until = oldest_request + self.window_seconds

            return max(0.0, wait_until - current_time)

    def reset(self) -> None:
        """Reset the rate limiter"""
        with self.lock:
            self.requests.clear()


# Global rate limiters for different operation types
_order_rate_limiter = RateLimiter(max_requests=10, window_seconds=60.0)
_cancel_rate_limiter = RateLimiter(max_requests=20, window_seconds=60.0)


def get_order_rate_limiter() -> RateLimiter:
    """Get the global order placement rate limiter"""
    return _order_rate_limiter


def get_cancel_rate_limiter() -> RateLimiter:
    """Get the global order cancellation rate limiter"""
    return _cancel_rate_limiter
