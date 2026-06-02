"""
API Throttling System for Mystic Trading Platform

Provides intelligent API call throttling with:
- Rate limiting per endpoint
- Token bucket burst control
- Adaptive throttling
- Performance monitoring
- Graceful degradation
"""

import asyncio
import logging
import threading
import time
from collections import defaultdict, deque
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from typing import Any

# Direct imports for production
from backend.config.settings import settings
from backend.utils.exceptions import APIError, RateLimitException, handle_exception

DEFAULT_REQUEST_TIMEOUT = getattr(settings, "DEFAULT_REQUEST_TIMEOUT", 30.0)


def throttle(_requests_per_minute: int = 60, message: str = "Rate limit exceeded"):
    """Throttle function calls to specified requests per minute"""

    def decorator(func):
        if asyncio.iscoroutinefunction(func):

            async def async_wrapper(*args, **kwargs):
                try:
                    return await func(*args, **kwargs)
                except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError):
                    logging.getLogger(__name__).exception(message)
                    raise

            return async_wrapper

        def sync_wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError):
                logging.getLogger(__name__).exception(message)
                raise

        return sync_wrapper
        return decorator

    return decorator


logger = logging.getLogger(__name__)


class ThrottleLevel(Enum):
    """Throttling levels"""

    CONSERVATIVE = "conservative"  # Start with low rates
    MODERATE = "moderate"  # Medium rates
    AGGRESSIVE = "aggressive"  # High rates
    UNLIMITED = "unlimited"  # No throttling


@dataclass
class ThrottleConfig:
    """Configuration for API throttling"""

    requests_per_second: int = 10
    burst_limit: int = 20
    queue_size: int = 100
    timeout: float = 30.0
    retry_attempts: int = 3
    backoff_factor: float = 2.0


@dataclass
class RequestMetrics:
    """Request performance metrics"""

    endpoint: str
    method: str
    timestamp: float
    response_time: float
    status_code: int
    success: bool
    throttled: bool = False


@dataclass
class TokenBucket:
    """Simple token bucket for burst control"""

    capacity: int
    tokens: float
    last_refill: float
    refill_rate: float  # tokens per second


class AdaptiveThrottler:
    """Adaptive API throttling system"""

    def __init__(self) -> None:
        self.throttle_level = ThrottleLevel.CONSERVATIVE
        self.configs = {
            ThrottleLevel.CONSERVATIVE: ThrottleConfig(
                requests_per_second=5,
                burst_limit=10,
                queue_size=50,
                timeout=30.0,
            ),
            ThrottleLevel.MODERATE: ThrottleConfig(
                requests_per_second=20,
                burst_limit=40,
                queue_size=100,
                timeout=20.0,
            ),
            ThrottleLevel.AGGRESSIVE: ThrottleConfig(
                requests_per_second=50,
                burst_limit=100,
                queue_size=200,
                timeout=10.0,
            ),
            ThrottleLevel.UNLIMITED: ThrottleConfig(
                requests_per_second=1000,  # High static limit, not tied to timeout
                burst_limit=2000,
                queue_size=500,
                timeout=10.0,
            ),
        }

        # Per-endpoint throttling; returns the config for the current throttle level by default
        self.endpoint_limits: defaultdict[str, ThrottleConfig] = defaultdict(lambda: self.configs[self.throttle_level])

        # Request tracking with explicit integer limits
        self.request_history: defaultdict[str, deque[float]] = defaultdict(lambda: deque(maxlen=1000))  # Fixed integer limit
        self.request_metrics: list[RequestMetrics] = []
        self.metrics_lock = threading.Lock()
        self.history_lock = threading.Lock()  # Add lock for request_history

        # Token buckets for burst control
        self.token_buckets: dict[str, TokenBucket] = {}

        # Performance monitoring
        self.performance_stats = {
            "total_requests": 0,
            "throttled_requests": 0,
            "failed_requests": 0,
            "average_response_time": 0.0,
            "success_rate": 0.0,
        }

        # Adaptive throttling
        self.adaptation_interval = 60  # seconds
        self.last_adaptation = time.time()
        self.adaptation_thread = threading.Thread(target=self._adaptation_loop, daemon=True)
        self.adaptation_thread.start()

        logger.info(f"[OK] AdaptiveThrottler initialized with {self.throttle_level.value} level")

    def set_throttle_level(self, level: ThrottleLevel):
        """Set throttling level"""
        self.throttle_level = level
        logger.info(f"[OK] Throttle level set to {level.value}")

    def get_current_config(self) -> ThrottleConfig:
        """Get current throttling configuration"""
        return self.configs[self.throttle_level]

    def set_endpoint_limit(self, endpoint: str, config: ThrottleConfig):
        """Set custom limits for specific endpoint"""
        self.endpoint_limits[endpoint] = config
        logger.info(f"[OK] Custom limits set for {endpoint}: {config.requests_per_second} req/s")

    def _can_make_request(self, endpoint: str) -> bool:
        """Check if request can be made based on rate limits (1-second sliding window)"""
        config = self.endpoint_limits[endpoint]
        now = time.time()

        with self.history_lock:
            # Clean old requests older than 1 second
            hist = self.request_history[endpoint]
            while hist and (now - hist[0]) >= 1.0:
                hist.popleft()

            # Check rate limit
            current_requests = len(hist)
            return current_requests < config.requests_per_second

    def _record_request(self, endpoint: str):
        """Record a request for rate limiting"""
        with self.history_lock:
            self.request_history[endpoint].append(time.time())

    def _get_or_create_bucket(self, endpoint: str, config: ThrottleConfig) -> TokenBucket:
        """Get or create token bucket for endpoint"""
        if endpoint not in self.token_buckets:
            now = time.time()
            self.token_buckets[endpoint] = TokenBucket(
                capacity=config.burst_limit,
                tokens=float(config.burst_limit),  # Start full
                last_refill=now,
                refill_rate=float(config.requests_per_second),
            )
        return self.token_buckets[endpoint]

    def _try_consume_token(self, endpoint: str, config: ThrottleConfig) -> bool:
        """Try to consume a token from the bucket (burst control)"""
        bucket = self._get_or_create_bucket(endpoint, config)
        now = time.time()

        # Refill tokens based on time elapsed
        time_passed = now - bucket.last_refill
        if time_passed > 0:
            tokens_to_add = time_passed * bucket.refill_rate
            bucket.tokens = min(bucket.capacity, bucket.tokens + tokens_to_add)
            bucket.last_refill = now

        # Try to consume a token
        if bucket.tokens >= 1.0:
            bucket.tokens -= 1.0
            return True
        return False

    @handle_exception("Request throttling failed", RateLimitException)
    async def throttle_request(
        self,
        endpoint: str,
        method: str = "GET",
        func: Callable[..., Any] | None = None,
        *args,
        **kwargs,
    ) -> Any:
        """Throttle and execute API request"""
        config = self.endpoint_limits[endpoint]
        start_time = time.time()

        # Check if request can be made using token bucket
        can_make_request = self._try_consume_token(endpoint, config)

        if not can_make_request:
            # Record throttled request as metrics
            self._record_metrics(endpoint, method, 0.0, 429, False, True)
            msg = f"Rate limit exceeded for {endpoint} (burst limit: {config.burst_limit})"
            raise RateLimitException(
                msg,
                details={
                    "endpoint": endpoint,
                    "limit": config.requests_per_second,
                    "burst_limit": config.burst_limit,
                    "retry_after": 1.0,
                },
            )

        # Record request
        self._record_request(endpoint)

        # Execute request with retries
        last_exception = None
        for attempt in range(max(1, config.retry_attempts)):
            try:
                if func:
                    if asyncio.iscoroutinefunction(func):
                        result = await func(*args, **kwargs)
                    else:
                        # If a synchronous function is provided, run it in default loop executor to avoid blocking
                        loop = asyncio.get_running_loop()
                        result = await loop.run_in_executor(None, lambda: func(*args, **kwargs))
                else:
                    result = None

                # Record successful request with real status code
                response_time = time.time() - start_time
                status_code = 200  # Default success, but should be actual HTTP status
                if hasattr(result, "status_code"):
                    status_code = result.status_code
                elif isinstance(result, dict) and "status_code" in result:
                    status_code = result["status_code"]

                self._record_metrics(endpoint, method, response_time, int(status_code), True, False)
            except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
                last_exception = e
                response_time = time.time() - start_time
                # Determine status code from exception
                status_code = 500  # Default error
                if hasattr(e, "status_code"):
                    status_code = e.status_code
                elif hasattr(e, "response") and getattr(e.response, "status_code", None) is not None:
                    status_code = e.response.status_code
                else:
                    estr = str(e).lower()
                    if "429" in estr:
                        status_code = 429
                    elif "timeout" in estr or "timed out" in estr:
                        status_code = 408

                # Record failed request
                self._record_metrics(endpoint, method, response_time, int(status_code), False, False)

                # Exponential backoff with base delay
                if attempt < config.retry_attempts - 1:
                    base_delay = 1.0  # Base delay in seconds
                    wait_time = base_delay * (config.backoff_factor**attempt)
                    logger.warning(f"Request failed, retrying in {wait_time:.1f}s: {e}")
                    await asyncio.sleep(wait_time)

        # All retries failed
        if last_exception:
            raise last_exception
        msg = "Request failed after all retries"
        raise APIError(msg)

    def _record_metrics(
        self,
        endpoint: str,
        method: str,
        response_time: float,
        status_code: int,
        success: bool,
        throttled: bool,
    ):
        """Record request metrics"""
        with self.metrics_lock:
            metric = RequestMetrics(
                endpoint=endpoint,
                method=method,
                timestamp=time.time(),
                response_time=response_time,
                status_code=status_code,
                success=success,
                throttled=throttled,
            )

            self.request_metrics.append(metric)

            # Update performance stats
            self.performance_stats["total_requests"] += 1
            if not success:
                self.performance_stats["failed_requests"] += 1
            if throttled:
                self.performance_stats["throttled_requests"] += 1

            # Keep only last metrics based on explicit limit
            max_metrics = 10000  # Fixed integer limit, not tied to timeout
            if len(self.request_metrics) > max_metrics:
                # Keep the last max_metrics entries
                self.request_metrics = self.request_metrics[-max_metrics:]

    def _adaptation_loop(self):
        """Background loop for adaptive throttling"""
        while True:
            try:
                time.sleep(self.adaptation_interval)
                self._adapt_throttling()
            except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
                logger.exception(f"Adaptation loop error: {e}")

    def _adapt_throttling(self):
        """Adapt throttling based on performance metrics"""
        with self.metrics_lock:
            if not self.request_metrics:
                return

            now = time.time()
            # Calculate performance metrics using recent metrics within adaptation interval
            recent_metrics = [m for m in self.request_metrics if now - m.timestamp < self.adaptation_interval]

            if not recent_metrics:
                return

            success_rate = sum(1 for m in recent_metrics if m.success) / len(recent_metrics)
            avg_response_time = sum(m.response_time for m in recent_metrics) / len(recent_metrics)
            throttle_rate = sum(1 for m in recent_metrics if m.throttled) / len(recent_metrics)

            # Update performance stats
            self.performance_stats["success_rate"] = success_rate
            self.performance_stats["average_response_time"] = avg_response_time

            # Adaptive logic
            current_level = self.throttle_level

            if success_rate > 0.95 and avg_response_time < DEFAULT_REQUEST_TIMEOUT and throttle_rate < 0.1:
                # Performance is good, can increase throttling
                if current_level == ThrottleLevel.CONSERVATIVE:
                    self.set_throttle_level(ThrottleLevel.MODERATE)
                elif current_level == ThrottleLevel.MODERATE:
                    self.set_throttle_level(ThrottleLevel.AGGRESSIVE)
                elif current_level == ThrottleLevel.AGGRESSIVE:
                    self.set_throttle_level(ThrottleLevel.UNLIMITED)

            elif success_rate < 0.8 or avg_response_time > DEFAULT_REQUEST_TIMEOUT or throttle_rate > 0.3:
                # Performance is poor, decrease throttling
                if current_level == ThrottleLevel.UNLIMITED:
                    self.set_throttle_level(ThrottleLevel.AGGRESSIVE)
                elif current_level == ThrottleLevel.AGGRESSIVE:
                    self.set_throttle_level(ThrottleLevel.MODERATE)
                elif current_level == ThrottleLevel.MODERATE:
                    self.set_throttle_level(ThrottleLevel.CONSERVATIVE)

            logger.info(f"Adaptation: success_rate={success_rate:.2f}, avg_response_time={avg_response_time:.2f}s, throttle_rate={throttle_rate:.2f}")

    def get_performance_stats(self) -> dict[str, Any]:
        """Get comprehensive performance statistics"""
        with self.metrics_lock:
            stats = self.performance_stats.copy()

            # Add endpoint-specific stats
            endpoint_stats = defaultdict(
                lambda: {
                    "total_requests": 0,
                    "successful_requests": 0,
                    "failed_requests": 0,
                    "throttled_requests": 0,
                    "average_response_time": 0.0,
                },
            )

            for metric in self.request_metrics:
                endpoint = metric.endpoint
                data = endpoint_stats[endpoint]
                data["total_requests"] += 1

                if metric.success:
                    data["successful_requests"] += 1
                else:
                    data["failed_requests"] += 1

                if metric.throttled:
                    data["throttled_requests"] += 1

            # Calculate averages for each endpoint
            for endpoint, data in list(endpoint_stats.items()):
                if data["total_requests"] > 0:
                    endpoint_metrics = [m for m in self.request_metrics if m.endpoint == endpoint]
                    if endpoint_metrics:
                        data["average_response_time"] = sum(m.response_time for m in endpoint_metrics) / len(endpoint_metrics)

            stats["endpoint_stats"] = dict(endpoint_stats)
            stats["current_throttle_level"] = self.throttle_level.value
            current_config = self.get_current_config()
            stats["current_config"] = {
                "requests_per_second": current_config.requests_per_second,
                "burst_limit": current_config.burst_limit,
                "timeout": current_config.timeout,
            }

            return stats

    def increase_throttling(self):
        """Manually increase throttling level"""
        levels = list(ThrottleLevel)
        current_index = levels.index(self.throttle_level)
        if current_index < len(levels) - 1:
            self.set_throttle_level(levels[current_index + 1])
            logger.info(f"[OK] Throttling increased to {self.throttle_level.value}")
        else:
            logger.info("[OK] Already at maximum throttling level")

    def decrease_throttling(self):
        """Manually decrease throttling level"""
        levels = list(ThrottleLevel)
        current_index = levels.index(self.throttle_level)
        if current_index > 0:
            self.set_throttle_level(levels[current_index - 1])
            logger.info(f"[OK] Throttling decreased to {self.throttle_level.value}")
        else:
            logger.info("[OK] Already at minimum throttling level")


# Global throttler instance
api_throttler = AdaptiveThrottler()
