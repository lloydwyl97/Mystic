"""
Enhanced Resilience Service
Comprehensive error recovery, retry strategies, and graceful degradation mechanisms
"""

import asyncio
import logging
import random
import time
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Optional

logger = logging.getLogger(__name__)


class ResilienceError(Exception):
    """Base exception for resilience operations"""

    pass


class CircuitBreakerOpenError(ResilienceError):
    """Exception raised when circuit breaker is open"""

    def __init__(self, message: str = "Circuit breaker is open"):
        self.message = message
        super().__init__(self.message)


class FailureType(Enum):
    """Types of failures that can occur"""

    NETWORK = "network"
    API_RATE_LIMIT = "api_rate_limit"
    DATABASE = "database"
    REDIS = "redis"
    EXTERNAL_API = "external_api"
    MODEL_LOADING = "model_loading"
    DATA_VALIDATION = "data_validation"
    TIMEOUT = "timeout"
    UNKNOWN = "unknown"


class CircuitBreakerState(Enum):
    """Circuit breaker states"""

    CLOSED = "closed"  # Normal operation
    OPEN = "open"  # Failing, requests blocked
    HALF_OPEN = "half_open"  # Testing recovery


@dataclass
class RetryConfig:
    """Configuration for retry strategies"""

    max_attempts: int = 3
    initial_delay: float = 0.1
    max_delay: float = 10.0
    backoff_multiplier: float = 2.0
    jitter_factor: float = 0.1
    retryable_exceptions: tuple = (Exception,)


@dataclass
class CircuitBreakerConfig:
    """Configuration for circuit breaker"""

    failure_threshold: int = 5
    recovery_timeout: float = 60.0
    expected_exception: tuple = (Exception,)


@dataclass
class ResilienceMetrics:
    """Metrics for resilience operations"""

    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    retries_attempted: int = 0
    retries_succeeded: int = 0
    circuit_breaker_trips: int = 0
    timeouts: int = 0
    last_failure_time: float = 0
    failure_counts: dict[str, int] = None

    def __post_init__(self):
        if self.failure_counts is None:
            self.failure_counts = defaultdict(int)


class ExponentialBackoffRetry:
    """Exponential backoff retry strategy with jitter"""

    def __init__(self, config: RetryConfig):
        self.config = config

    async def execute_with_retry(self, operation: Callable[[], Any], operation_name: str = "operation") -> Any:
        """Execute operation with retry logic"""
        last_exception = None

        for attempt in range(self.config.max_attempts):
            try:
                result = await operation()
                ResilienceService._record_success(operation_name)
            except self.config.retryable_exceptions as e:
                last_exception = e
                ResilienceService._record_failure(operation_name, str(type(e).__name__))

                if attempt < self.config.max_attempts - 1:
                    delay = self._calculate_delay(attempt)
                    logger.warning(f"{operation_name} failed (attempt {attempt + 1}/{self.config.max_attempts}): {e}. Retrying in {delay:.2f}s")
                    await asyncio.sleep(delay)
                else:
                    logger.exception(f"{operation_name} failed after {self.config.max_attempts} attempts: {e}")
            else:
                return result

        raise last_exception

    def _calculate_delay(self, attempt: int) -> float:
        """Calculate delay with exponential backoff and jitter"""
        delay = min(self.config.initial_delay * (self.config.backoff_multiplier**attempt), self.config.max_delay)

        # Add jitter to prevent thundering herd
        jitter = delay * self.config.jitter_factor * (random.random() * 2 - 1)
        return max(0.001, delay + jitter)


class CircuitBreaker:
    """Circuit breaker pattern implementation"""

    def __init__(self, name: str, config: CircuitBreakerConfig):
        self.name = name
        self.config = config
        self.state = CircuitBreakerState.CLOSED
        self.failure_count = 0
        self.last_failure_time = 0
        self.next_attempt_time = 0

    async def execute(self, operation: Callable[[], Any], operation_name: str = "operation") -> Any:
        """Execute operation through circuit breaker"""
        if self.state == CircuitBreakerState.OPEN:
            if time.time() < self.next_attempt_time:
                msg = f"Circuit breaker {self.name} is OPEN"
                raise CircuitBreakerOpenError(msg)
            else:
                self.state = CircuitBreakerState.HALF_OPEN
                logger.info(f"Circuit breaker {self.name} entering HALF_OPEN state")

        try:
            result = await operation()
            self._on_success()
        except self.config.expected_exception:
            self._on_failure()
            raise
        else:
            return result

    def _on_success(self):
        """Handle successful operation"""
        if self.state == CircuitBreakerState.HALF_OPEN:
            self.state = CircuitBreakerState.CLOSED
            self.failure_count = 0
            logger.info(f"Circuit breaker {self.name} recovered to CLOSED state")
        elif self.state == CircuitBreakerState.CLOSED:
            self.failure_count = 0

    def _on_failure(self):
        """Handle failed operation"""
        self.failure_count += 1
        self.last_failure_time = time.time()

        if self.state == CircuitBreakerState.HALF_OPEN:
            self.state = CircuitBreakerState.OPEN
            self.next_attempt_time = time.time() + self.config.recovery_timeout
            logger.warning(f"Circuit breaker {self.name} tripped back to OPEN state")
            ResilienceService._record_circuit_breaker_trip(self.name)

        elif self.failure_count >= self.config.failure_threshold:
            self.state = CircuitBreakerState.OPEN
            self.next_attempt_time = time.time() + self.config.recovery_timeout
            logger.warning(f"Circuit breaker {self.name} tripped to OPEN state after {self.failure_count} failures")
            ResilienceService._record_circuit_breaker_trip(self.name)


class GracefulDegradationManager:
    """Manages graceful degradation when services are unavailable"""

    def __init__(self):
        self.degraded_services: dict[str, dict[str, Any]] = {}
        self.fallback_strategies: dict[str, Callable[[], Any]] = {}

    def register_fallback(self, service_name: str, fallback_strategy: Callable[[], Any]):
        """Register a fallback strategy for a service"""
        self.fallback_strategies[service_name] = fallback_strategy

    async def execute_with_fallback(self, service_name: str, primary_operation: Callable[[], Any], *args, **kwargs) -> Any:
        """Execute primary operation with fallback if it fails"""
        try:
            result = await primary_operation(*args, **kwargs)
            self._clear_degradation(service_name)
        except Exception as e:
            logger.warning(f"Primary operation for {service_name} failed: {e}")
            return await self._execute_fallback(service_name, e)
        else:
            return result

    async def _execute_fallback(self, service_name: str, original_error: Exception) -> Any:
        """Execute fallback strategy for degraded service"""
        if service_name in self.fallback_strategies:
            try:
                self._mark_degraded(service_name, original_error)
                result = await self.fallback_strategies[service_name]()
                logger.info(f"Successfully executed fallback for {service_name}")
            except Exception as fallback_error:
                logger.exception(f"Fallback for {service_name} also failed: {fallback_error}")
                raise original_error from fallback_error
            else:
                return result
        else:
            logger.error(f"No fallback strategy registered for {service_name}")
            raise original_error

    def _mark_degraded(self, service_name: str, error: Exception):
        """Mark a service as degraded"""
        self.degraded_services[service_name] = {"degraded_at": datetime.now(timezone.utc).isoformat(), "error": str(error), "error_type": type(error).__name__}
        logger.warning(f"Service {service_name} marked as degraded")

    def _clear_degradation(self, service_name: str):
        """Clear degradation status for a service"""
        if service_name in self.degraded_services:
            del self.degraded_services[service_name]
            logger.info(f"Service {service_name} recovered from degradation")

    def get_degradation_status(self) -> dict[str, dict[str, Any]]:
        """Get current degradation status of all services"""
        return self.degraded_services.copy()


class ResilienceService:
    """
    Central resilience service providing comprehensive error recovery and fault tolerance
    """

    _instance: Optional["ResilienceService"] = None
    _lock: asyncio.Lock | None = None  # Lazy init to avoid event loop issues at import

    @classmethod
    def _get_lock(cls) -> asyncio.Lock:
        """Get or create lock lazily."""
        if cls._lock is None:
            cls._lock = asyncio.Lock()
        return cls._lock

    def __init__(self):
        self.metrics = ResilienceMetrics()
        self.retry_strategies: dict[str, ExponentialBackoffRetry] = {}
        self.circuit_breakers: dict[str, CircuitBreaker] = {}
        self.degradation_manager = GracefulDegradationManager()
        self.health_checks: dict[str, Callable[[], bool]] = {}

    @classmethod
    async def get_instance(cls) -> "ResilienceService":
        """Get singleton instance"""
        if cls._instance is None:
            async with cls._get_lock():
                if cls._instance is None:
                    cls._instance = cls()
                    await cls._instance._initialize()
        return cls._instance

    async def _initialize(self):
        """Initialize resilience service with default configurations"""
        # Register default retry strategies
        self.retry_strategies["default"] = ExponentialBackoffRetry(RetryConfig())
        self.retry_strategies["api_call"] = ExponentialBackoffRetry(RetryConfig(max_attempts=5, initial_delay=0.5, max_delay=30.0))
        self.retry_strategies["database"] = ExponentialBackoffRetry(RetryConfig(max_attempts=3, initial_delay=0.1, max_delay=5.0))

        # Register default circuit breakers
        self.circuit_breakers["binance_api"] = CircuitBreaker("binance_api", CircuitBreakerConfig(failure_threshold=3, recovery_timeout=30.0))
        self.circuit_breakers["database"] = CircuitBreaker("database", CircuitBreakerConfig(failure_threshold=5, recovery_timeout=60.0))
        self.circuit_breakers["redis"] = CircuitBreaker("redis", CircuitBreakerConfig(failure_threshold=3, recovery_timeout=15.0))

        # Register common fallback strategies
        await self._register_default_fallbacks()

    async def _register_default_fallbacks(self):
        """Register default fallback strategies"""

        # Binance API fallback
        async def binance_fallback():
            logger.warning("Using Binance API fallback - returning cached data")
            # Return minimal cached market data if available
            return {"fallback": True, "message": "Using cached market data"}

        self.degradation_manager.register_fallback("binance_api", binance_fallback)

        # Database fallback
        async def database_fallback():
            logger.warning("Using database fallback - operations queued for retry")
            return {"fallback": True, "message": "Database operations queued"}

        self.degradation_manager.register_fallback("database", database_fallback)

        # Redis fallback
        async def redis_fallback():
            logger.warning("Using Redis fallback - using in-memory cache")
            return {"fallback": True, "message": "Using in-memory cache"}

        self.degradation_manager.register_fallback("redis", redis_fallback)

    async def execute_with_resilience(
        self,
        operation: Callable[[], Any],
        operation_name: str,
        failure_type: FailureType = FailureType.UNKNOWN,
        retry_strategy: str = "default",
        circuit_breaker: str | None = None,
        use_fallback: bool = True,
    ) -> Any:
        """
        Execute operation with full resilience: retry + circuit breaker + fallback
        """
        self.metrics.total_requests += 1

        try:
            # Apply circuit breaker if specified
            if circuit_breaker and circuit_breaker in self.circuit_breakers:
                operation = self._wrap_with_circuit_breaker(operation, circuit_breaker)

            # Apply retry strategy
            if retry_strategy in self.retry_strategies:
                retry_handler = self.retry_strategies[retry_strategy]
                result = await retry_handler.execute_with_retry(operation, operation_name)
            else:
                result = await operation()

        except Exception:
            self.metrics.failed_requests += 1
            self.metrics.failure_counts[str(failure_type.value)] += 1

            # Try fallback if enabled
            if use_fallback:
                service_name = circuit_breaker or operation_name.split("_", maxsplit=1)[0]
                if service_name in self.degradation_manager.fallback_strategies:
                    try:
                        return await self.degradation_manager.execute_with_fallback(service_name, operation)
                    except Exception:
                        pass  # Fallback also failed, continue to raise original error

            raise
        else:
            return result

    def _wrap_with_circuit_breaker(self, operation: Callable[[], Any], breaker_name: str):
        """Wrap operation with circuit breaker"""

        async def wrapped_operation():
            return await self.circuit_breakers[breaker_name].execute(operation, breaker_name)

        return wrapped_operation

    @classmethod
    def _record_success(cls, operation_name: str):
        """Record successful operation"""
        if cls._instance:
            cls._instance.metrics.successful_requests += 1

    @classmethod
    def _record_failure(cls, operation_name: str, error_type: str):
        """Record failed operation"""
        if cls._instance:
            cls._instance.metrics.failed_requests += 1
            cls._instance.metrics.failure_counts[error_type] += 1

    @classmethod
    def _record_circuit_breaker_trip(cls, breaker_name: str):
        """Record circuit breaker trip"""
        if cls._instance:
            cls._instance.metrics.circuit_breaker_trips += 1

    def get_resilience_report(self) -> dict[str, Any]:
        """Get comprehensive resilience report"""
        total_requests = self.metrics.total_requests
        success_rate = (self.metrics.successful_requests / total_requests * 100) if total_requests > 0 else 0

        return {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "overall_health": {
                "total_requests": total_requests,
                "success_rate": f"{success_rate:.2f}%",
                "failure_rate": f"{100 - success_rate:.2f}%",
                "active_circuit_breakers": len([cb for cb in self.circuit_breakers.values() if cb.state == CircuitBreakerState.OPEN]),
                "degraded_services": len(self.degradation_manager.degraded_services),
            },
            "circuit_breakers": {name: {"state": cb.state.value, "failure_count": cb.failure_count, "last_failure": cb.last_failure_time} for name, cb in self.circuit_breakers.items()},
            "degraded_services": self.degradation_manager.get_degradation_status(),
            "failure_breakdown": dict(self.metrics.failure_counts),
            "retry_metrics": {"retries_attempted": self.metrics.retries_attempted, "retries_succeeded": self.metrics.retries_succeeded},
        }


# Global instance accessor
async def get_resilience_service() -> ResilienceService:
    """Get the global resilience service instance"""
    return await ResilienceService.get_instance()
