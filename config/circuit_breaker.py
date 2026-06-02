"""
Optimized circuit breaker pattern for Mystic Trading Platform.
Connected to live configuration and integrated with circuit breaker service.
"""

import asyncio
import functools
import os
import time
from collections.abc import Awaitable, Callable
from enum import Enum
from typing import Any, TypeVar

T = TypeVar("T")

# Import live configuration
try:
    from backend.config_bridge import get_mystic_config

    _mystic_config = get_mystic_config()
except (ImportError, AttributeError, ValueError, TypeError, RuntimeError):
    _mystic_config = None


class CircuitState(Enum):
    CLOSED = 0  # Normal operation, requests pass through
    OPEN = 1  # Circuit is open, fast-fail requests
    HALF_OPEN = 2  # Testing if service is recovered


def _get_default_failure_threshold() -> int:
    """Get default failure threshold from live configuration."""
    try:
        value = int(os.getenv("CIRCUIT_BREAKER_FAILURE_THRESHOLD", "5"))
        return max(1, value)
    except (ValueError, TypeError):
        return 5


def _get_default_reset_timeout() -> float:
    """Get default reset timeout from live configuration."""
    try:
        value = float(os.getenv("CIRCUIT_BREAKER_RESET_TIMEOUT", "30"))
        return max(1.0, value)
    except (ValueError, TypeError):
        return 30.0


class AsyncCircuitBreaker:
    """Circuit breaker for async operations"""

    def __init__(
        self,
        name: str,
        failure_threshold: int | None = None,
        reset_timeout: float | None = None,
    ):
        self.name = name
        self.failure_threshold = failure_threshold if failure_threshold is not None else _get_default_failure_threshold()
        self.reset_timeout = reset_timeout if reset_timeout is not None else _get_default_reset_timeout()
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.last_failure_time = 0.0
        self.lock = asyncio.Lock()

    async def __call__(self, func: Callable[..., Awaitable[T]], *args, **kwargs) -> T:
        """Execute the function with circuit breaker protection"""
        await self._check_state()

        if self.state == CircuitState.OPEN:
            msg = f"Circuit {self.name} is OPEN"
            raise CircuitOpenError(msg)

        try:
            result = await func(*args, **kwargs)
        except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
            # Failure occurred
            await self._on_failure(e)
            raise
        else:
            # Success, reset failure count if in half-open state
            if self.state == CircuitState.HALF_OPEN:
                await self._reset()
            return result

    async def _check_state(self) -> None:
        """Check and update circuit state if needed"""
        if self.state == CircuitState.OPEN and time.time() - self.last_failure_time >= self.reset_timeout:
            async with self.lock:
                if self.state == CircuitState.OPEN:
                    # Try half-open state
                    self.state = CircuitState.HALF_OPEN

    async def _on_failure(self, _exception: Exception) -> None:
        """Handle a failure"""
        async with self.lock:
            self.last_failure_time = time.time()
            if self.state == CircuitState.HALF_OPEN:
                # If failed during half-open, immediately open the circuit
                self.state = CircuitState.OPEN
            elif self.state == CircuitState.CLOSED:
                # Increment failure counter
                self.failure_count += 1
                if self.failure_count >= self.failure_threshold:
                    self.state = CircuitState.OPEN

    async def _reset(self) -> None:
        """Reset the circuit breaker to closed state"""
        async with self.lock:
            self.failure_count = 0
            self.state = CircuitState.CLOSED


class CircuitOpenError(Exception):
    """Exception raised when circuit is open"""


def circuit_breaker(name: str, **circuit_kwargs):
    """Decorator to apply circuit breaker pattern to an async function"""
    breaker = AsyncCircuitBreaker(name, **circuit_kwargs)

    def decorator(func):
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            return await breaker(func, *args, **kwargs)

        return wrapper

    return decorator


# Track all circuit breakers
_circuit_breakers: dict[str, AsyncCircuitBreaker] = {}


def get_breaker(name: str, **circuit_kwargs) -> AsyncCircuitBreaker:
    """Get or create a circuit breaker by name"""
    if name not in _circuit_breakers:
        _circuit_breakers[name] = AsyncCircuitBreaker(name, **circuit_kwargs)
    return _circuit_breakers[name]


async def get_circuit_status() -> dict[str, Any]:
    """Get status of all circuit breakers"""
    return {
        name: {
            "state": breaker.state.name,
            "failure_count": breaker.failure_count,
            "last_failure": breaker.last_failure_time,
            "failure_threshold": breaker.failure_threshold,
            "reset_timeout": breaker.reset_timeout,
        }
        for name, breaker in _circuit_breakers.items()
    }
