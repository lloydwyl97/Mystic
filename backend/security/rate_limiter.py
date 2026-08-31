"""
Advanced Rate Limiter for Mystic Trading Platform (single file)

Features:
- Per-endpoint configs (rpm/rps/burst), strategies: sliding_window or
  token_bucket
- Client identity: header (X-Client-ID) > authenticated user > IP fallback
- Local, thread-safe implementation with optional Redis for distributed limits
- Suspicious activity & auto-blocking (temporary), structured stats
- Helpful helpers for FastAPI (optional): dependency + install function
- Conservative defaults; degrades gracefully if Redis/FastAPI not installed

Notes:
- Sliding window uses timestamp deques (local) or Redis INCR/EXPIRE buckets.
- Token bucket is local-only for simplicity; Redis falls back to sliding
  window.
- All public methods are safe to call from sync FastAPI routes (no
  await needed).
"""

from __future__ import annotations

import logging
import os
import threading
import time
from collections import defaultdict, deque
from dataclasses import dataclass
from functools import partial
from typing import Any

logger = logging.getLogger(__name__)

# Optional FastAPI imports
try:
    from fastapi import HTTPException, Request  # type: ignore[import-not-found]
except (ImportError, ModuleNotFoundError, AttributeError):
    HTTPException = None  # type: ignore[assignment]
    Request = None  # type: ignore[assignment]

# Optional Redis (sync client for atomic counters)
try:
    from redis import Redis  # type: ignore[import-not-found]

    from backend.config.redis_config import get_redis_client
except (ImportError, ModuleNotFoundError, AttributeError):
    redis = None  # type: ignore[assignment]
    Redis = None  # type: ignore[assignment]

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
SLIDING_WINDOW_SIZE_SEC = 60
TOKEN_BUCKET_CAPACITY = 100
TOKEN_BUCKET_FILL_RATE = TOKEN_BUCKET_CAPACITY / 60.0  # tokens/sec for
# 1-min refill

# Security thresholds
MAX_FAILED_ATTEMPTS = 5
BLOCK_DURATION_SEC = 300  # 5 min
SUSPICIOUS_ACTIVITY_THRESHOLD_RPM = 50  # if total rpm > this -> suspicious

# Redis defaults (override via env)
REDIS_URL = os.getenv("RATE_LIMITER_REDIS_URL")
REDIS_HOST = os.getenv("RATE_LIMITER_REDIS_HOST")
REDIS_PORT = os.getenv("RATE_LIMITER_REDIS_PORT")
REDIS_DB = os.getenv("RATE_LIMITER_REDIS_DB")

# HTTP headers (standard-ish)
HDR_LIMIT = "X-RateLimit-Limit"
HDR_REMAINING = "X-RateLimit-Remaining"
HDR_RESET = "X-RateLimit-Reset"
HDR_RETRY_AFTER = "Retry-After"


# -----------------------------------------------------------------------------
# Models
# -----------------------------------------------------------------------------
@dataclass(frozen=True)
class RateLimitConfig:
    """Rate limit configuration for endpoint."""

    endpoint: str
    requests_per_minute: int
    requests_per_second: int
    burst_limit: int
    window_size: int
    strategy: str = "sliding_window"  # 'sliding_window' | 'token_bucket'


@dataclass
class RateLimitViolation:
    """Rate limit violation record."""

    client_id: str
    endpoint: str
    timestamp: float
    violation_type: str  # 'rate_limit' | 'burst_limit' | etc.
    request_count: int
    limit: int


# -----------------------------------------------------------------------------
# Security Monitor
# -----------------------------------------------------------------------------
class SecurityMonitor:
    """Security monitoring for rate limit violations."""

    def __init__(self) -> None:
        self.violations: deque[RateLimitViolation] = deque(maxlen=1000)
        self.blocked_clients: dict[str, float] = {}  # client_id -> unblock
        self.suspicious_counts: dict[str, int] = defaultdict(int)
        self.security_alerts: deque[dict[str, Any]] = deque(maxlen=100)
        self._lock = threading.Lock()

    def record_violation(
        self,
        client_id: str,
        endpoint: str,
        violation_type: str,
        request_count: int,
        limit: int,
    ) -> None:
        """Record a rate limit violation."""
        v = RateLimitViolation(
            client_id=client_id,
            endpoint=endpoint,
            timestamp=time.time(),
            violation_type=violation_type,
            request_count=request_count,
            limit=limit,
        )
        with self._lock:
            self.violations.append(v)
            key = f"{client_id}:{endpoint}"
            self.suspicious_counts[key] += 1
            if self.suspicious_counts[key] >= MAX_FAILED_ATTEMPTS:
                self.blocked_clients[client_id] = time.time() + BLOCK_DURATION_SEC
                self._alert(
                    client_id,
                    ("Client temporarily blocked due to repeated rate-limit violations"),
                )

    def is_client_blocked(self, client_id: str) -> bool:
        """Check if client is blocked."""
        with self._lock:
            until = self.blocked_clients.get(client_id)
            if until is None:
                return False
            if time.time() < until:
                return True
            # Expired
            self.blocked_clients.pop(client_id, None)
            return False

    def _alert(self, client_id: str, reason: str) -> None:
        """Create security alert."""
        alert = {
            "client_id": client_id,
            "reason": reason,
            "timestamp": time.time(),
            "severity": "high",
        }
        self.security_alerts.append(alert)
        logger.warning("Security alert: %s (client=%s)", reason, client_id)

    def stats(self) -> dict[str, Any]:
        """Get security statistics."""
        with self._lock:
            return {
                "total_violations": len(self.violations),
                "blocked_clients": len(self.blocked_clients),
                "security_alerts": len(self.security_alerts),
                "suspicious_patterns": dict(self.suspicious_counts),
            }

    def clear_expired_blocks(self) -> None:
        """Clear expired block entries."""
        with self._lock:
            now = time.time()
            expired = [cid for cid, until in self.blocked_clients.items() if until <= now]
            for cid in expired:
                del self.blocked_clients[cid]


# -----------------------------------------------------------------------------
# Sliding Window (local)
# -----------------------------------------------------------------------------
class SlidingWindowRateLimiter:
    """Local sliding window rate limiter."""

    def __init__(self) -> None:
        # key -> deque[timestamps]
        self._window: dict[str, deque[float]] = defaultdict(deque)
        self._lock = threading.Lock()

    def _prune(self, dq: deque[float], window_size: int, now: float) -> None:
        """Prune old entries from deque."""
        while dq and (now - dq[0]) > window_size:
            dq.popleft()

    def hit(self, key: str, limit: int, window_size: int) -> tuple[bool, int, float]:
        """
        Record a hit into sliding window.

        Returns (allowed, remaining, reset_epoch).
        """
        now = time.time()
        with self._lock:
            dq = self._window[key]
            self._prune(dq, window_size, now)
            count = len(dq)
            if count < limit:
                dq.append(now)
                remaining = limit - (count + 1)
                reset = (dq[0] + window_size) if dq else (now + window_size)
                return True, max(0, remaining), reset
            # denied
            reset = dq[0] + window_size if dq else now + window_size
            return False, 0, reset

    def remaining(self, key: str, limit: int, window_size: int) -> tuple[int, float]:
        """Get remaining requests and reset epoch."""
        now = time.time()
        with self._lock:
            dq = self._window[key]
            self._prune(dq, window_size, now)
            count = len(dq)
            remaining = max(0, limit - count)
            reset = (dq[0] + window_size) if dq else (now + window_size)
            return remaining, reset


# -----------------------------------------------------------------------------
# Token Bucket (local)
# -----------------------------------------------------------------------------
class TokenBucketRateLimiter:
    """Local token bucket rate limiter."""

    def __init__(
        self,
        capacity: int = TOKEN_BUCKET_CAPACITY,
        rate: float = TOKEN_BUCKET_FILL_RATE,
    ) -> None:
        self.capacity = float(capacity)
        self.rate = float(rate)  # tokens per second
        self._tokens: dict[str, float] = defaultdict(partial(float, capacity))
        self._updated: dict[str, float] = defaultdict(time.time)
        self._lock = threading.Lock()

    def hit(self, key: str, tokens_required: int = 1) -> tuple[bool, float]:
        """
        Consume tokens from the bucket.

        Returns (allowed, remaining_tokens).
        """
        now = time.time()
        with self._lock:
            last = self._updated[key]
            self._tokens[key] = min(
                self.capacity,
                self._tokens[key] + (now - last) * self.rate,
            )
            self._updated[key] = now

            if self._tokens[key] >= tokens_required:
                self._tokens[key] -= tokens_required
                return True, self._tokens[key]
            return False, self._tokens[key]

    def remaining(self, key: str) -> float:
        """Get remaining tokens."""
        now = time.time()
        with self._lock:
            last = self._updated[key]
            self._tokens[key] = min(
                self.capacity,
                self._tokens[key] + (now - last) * self.rate,
            )
            self._updated[key] = now
            return self._tokens[key]


# -----------------------------------------------------------------------------
# Redis-backed helpers (distributed sliding window via INCR/EXPIRE)
# -----------------------------------------------------------------------------
class RedisCounters:
    """Small helper for distributed window counting; gracefully disabled if
    Redis not available."""

    def __init__(self) -> None:
        self.client: Redis | None = None
        if Redis is None:
            logger.info("Redis not installed; distributed rate limiting disabled.")
            return
        try:
            # Use shared Redis pool
            self.client = get_redis_client()

            if self.client:
                self.client.ping()
                logger.info("Redis connection established for rate limiting.")
        except Exception as e:
            logger.warning(
                "Redis connection failed; falling back to local limits: %s",
                e,
            )
            self.client = None

    def minute_bucket_key(self, key: str) -> str:
        """Generate minute bucket key."""
        # rotate every 60s
        return f"rl:minute:{key}:{int(time.time() // 60)}"

    def second_bucket_key(self, key: str) -> str:
        """Generate second bucket key."""
        return f"rl:second:{key}:{int(time.time())}"

    def incr_with_ttl(self, key: str, ttl: int) -> int:
        """Increment counter with TTL. Returns count or -1 if unavailable."""
        if not self.client:
            return -1
        pipe = self.client.pipeline()
        pipe.incr(key)
        pipe.expire(key, ttl)
        results = pipe.execute()  # type: ignore[attr-defined]
        if not isinstance(results, list) or len(results) < 2:
            return -1
        count = results[0]
        if count is None:
            return -1
        return int(count)

    def get(self, key: str) -> int:
        """Get counter value. Returns count or -1 if unavailable."""
        if not self.client:
            return -1
        val = self.client.get(key)
        if val is None:
            return -1
        try:
            return int(val)
        except (ValueError, TypeError):
            return -1


# -----------------------------------------------------------------------------
# Advanced Rate Limiter
# -----------------------------------------------------------------------------
class AdvancedRateLimiter:
    """Advanced rate limiter with per-endpoint configs."""

    def __init__(self) -> None:
        self._sliding = SlidingWindowRateLimiter()
        self._bucket = TokenBucketRateLimiter()
        self._security = SecurityMonitor()
        self._endpoint_cfg: dict[str, RateLimitConfig] = {}
        # request timestamps for suspicious rpm detection
        self._client_times: dict[str, deque[float]] = defaultdict(lambda: deque(maxlen=1000))
        self._lock = threading.Lock()
        self._redis = RedisCounters()

    # ---- Configuration ----------------------------------------------------
    def add_endpoint_config(self, cfg: RateLimitConfig) -> None:
        """Add endpoint rate limit configuration."""
        with self._lock:
            self._endpoint_cfg[cfg.endpoint] = cfg
        logger.info(
            "RateLimit config set: %s (rpm=%s rps=%s strategy=%s)",
            cfg.endpoint,
            cfg.requests_per_minute,
            cfg.requests_per_second,
            cfg.strategy,
        )

    def get_config(self, endpoint: str) -> RateLimitConfig | None:
        """Get rate limit config for endpoint."""
        with self._lock:
            return self._endpoint_cfg.get(endpoint)

    # ---- Identity & accounting ----------------------------------------------
    @staticmethod
    def client_id_from_request(request: Any) -> str:
        """Extract client ID from request."""
        # Header first
        try:
            v = request.headers.get("X-Client-ID")  # type: ignore[attr-defined]
            if v:
                return f"h_{v}"
        except Exception:
            pass

        # Authenticated user
        try:
            user = getattr(request, "user", None)
            if user:
                uid = getattr(user, "id", None)
                if uid:
                    return f"u_{uid}"
        except Exception:
            pass

        # IP fallback
        try:
            ip = request.client.host  # type: ignore[attr-defined]
            if ip:
                return f"ip_{ip}"
        except Exception:
            pass

        return "ip_unknown"

    def _record_client_hit(self, client_id: str) -> None:
        """Record client request hit."""
        now = time.time()
        dq = self._client_times[client_id]
        dq.append(now)
        # keep last 60s only
        cutoff = now - 60.0
        while dq and dq[0] < cutoff:
            dq.popleft()

    def _is_suspicious(self, client_id: str) -> bool:
        """Check if client activity is suspicious."""
        dq = self._client_times[client_id]
        return len(dq) > SUSPICIOUS_ACTIVITY_THRESHOLD_RPM

    # ---- Core check ---------------------------------------------------------
    def is_rate_limited(self, request: Any, endpoint: str) -> tuple[bool, dict[str, Any]]:
        """
        Returns (limited, info).

        info contains headers-friendly metadata:
        {
            "remaining": int,
            "limit": int,
            "reset_epoch": float,
            "retry_after": int | None,
            "blocked": bool,
            "reason": str
        }
        """
        client_id = self.client_id_from_request(request)
        cfg = self.get_config(endpoint)

        if not cfg:
            # No config for endpoint - allow
            return False, {
                "remaining": 0,
                "limit": 0,
                "reset_epoch": time.time() + 60,
                "retry_after": None,
                "blocked": False,
                "reason": "ok",
            }

        key_base = f"{client_id}:{endpoint}"

        # Blocked client check
        if self._security.is_client_blocked(client_id):
            info = {
                "remaining": 0,
                "limit": 0,
                "reset_epoch": time.time() + BLOCK_DURATION_SEC,
                "retry_after": BLOCK_DURATION_SEC,
                "blocked": True,
                "reason": ("Client temporarily blocked due to repeated violations"),
            }
            return True, info

        # Global accounting for suspicious RPM
        self._record_client_hit(client_id)
        if self._is_suspicious(client_id):
            self._security.record_violation(
                client_id,
                endpoint,
                "suspicious_activity",
                len(self._client_times[client_id]),
                SUSPICIOUS_ACTIVITY_THRESHOLD_RPM,
            )
            info = {
                "remaining": 0,
                "limit": SUSPICIOUS_ACTIVITY_THRESHOLD_RPM,
                "reset_epoch": time.time() + 60,
                "retry_after": 60,
                "blocked": False,
                "reason": "Suspicious activity detected",
            }
            return True, info

        # 1) Per-second throttle (rps) using sliding second window
        sec_key = f"{key_base}:rps"
        if self._redis.client:
            rkey = self._redis.second_bucket_key(sec_key)
            count = self._redis.incr_with_ttl(rkey, ttl=2)
            # when redis unavailable, count = -1 (skip)
            if count >= 0 and count > cfg.requests_per_second:
                remaining = max(0, cfg.requests_per_second - count)
                reset_epoch = int(time.time()) + 1
                self._security.record_violation(
                    client_id,
                    endpoint,
                    "burst_limit",
                    int(count),
                    cfg.requests_per_second,
                )
                return True, {
                    "remaining": remaining,
                    "limit": cfg.requests_per_second,
                    "reset_epoch": reset_epoch,
                    "retry_after": reset_epoch - int(time.time()),
                    "blocked": False,
                    "reason": "Per-second rate limit exceeded",
                }
        else:
            allowed, remaining, reset_epoch = self._sliding.hit(sec_key, cfg.requests_per_second, 1)
            if not allowed:
                self._security.record_violation(
                    client_id,
                    endpoint,
                    "burst_limit",
                    cfg.requests_per_second - remaining,
                    cfg.requests_per_second,
                )
                return True, {
                    "remaining": remaining,
                    "limit": cfg.requests_per_second,
                    "reset_epoch": reset_epoch,
                    "retry_after": max(0, int(reset_epoch - time.time())),
                    "blocked": False,
                    "reason": "Per-second rate limit exceeded",
                }

        # 2) Main strategy (per-minute)
        minute_key = f"{key_base}:rpm"
        if cfg.strategy == "sliding_window" or self._redis.client:
            # Prefer Redis window if available (distributed)
            if self._redis.client:
                rkey = self._redis.minute_bucket_key(minute_key)
                count = self._redis.incr_with_ttl(rkey, ttl=SLIDING_WINDOW_SIZE_SEC + 2)
                if count >= 0 and count > cfg.requests_per_minute:
                    remaining = max(0, cfg.requests_per_minute - count)
                    reset_epoch = int(time.time() // 60) * 60 + 60
                    self._security.record_violation(
                        client_id,
                        endpoint,
                        "rate_limit",
                        int(count),
                        cfg.requests_per_minute,
                    )
                    return True, {
                        "remaining": remaining,
                        "limit": cfg.requests_per_minute,
                        "reset_epoch": reset_epoch,
                        "retry_after": max(0, reset_epoch - int(time.time())),
                        "blocked": False,
                        "reason": "Per-minute rate limit exceeded",
                    }
                remaining = max(0, cfg.requests_per_minute - count)
                reset_epoch = int(time.time() // 60) * 60 + 60
            else:
                allowed, remaining, reset_epoch = self._sliding.hit(minute_key, cfg.requests_per_minute, cfg.window_size)
                if not allowed:
                    self._security.record_violation(
                        client_id,
                        endpoint,
                        "rate_limit",
                        cfg.requests_per_minute - remaining,
                        cfg.requests_per_minute,
                    )
                    return True, {
                        "remaining": remaining,
                        "limit": cfg.requests_per_minute,
                        "reset_epoch": reset_epoch,
                        "retry_after": max(0, int(reset_epoch - time.time())),
                        "blocked": False,
                        "reason": "Per-minute rate limit exceeded",
                    }
        else:
            # token bucket (local)
            allowed, remaining_tokens = self._bucket.hit(minute_key, 1)
            if not allowed:
                # Estimate time until 1 token
                deficit = 1.0 - remaining_tokens
                retry_after = max(
                    1,
                    int(deficit / max(1e-6, self._bucket.rate)),
                )
                reset_epoch = time.time() + retry_after
                self._security.record_violation(
                    client_id,
                    endpoint,
                    "rate_limit",
                    int(TOKEN_BUCKET_CAPACITY - remaining_tokens),
                    TOKEN_BUCKET_CAPACITY,
                )
                return True, {
                    "remaining": int(remaining_tokens),
                    "limit": TOKEN_BUCKET_CAPACITY,
                    "reset_epoch": reset_epoch,
                    "retry_after": retry_after,
                    "blocked": False,
                    "reason": "Token bucket limit exceeded",
                }
            remaining = int(self._bucket.remaining(minute_key))
            reset_epoch = time.time() + 60  # approximate

        # Success
        return False, {
            "remaining": int(remaining),
            "limit": cfg.requests_per_minute,
            "reset_epoch": reset_epoch,
            "retry_after": None,
            "blocked": False,
            "reason": "ok",
        }

    # ---- Introspection ----------------------------------------------------
    def get_rate_limit_info(self, client_id: str, endpoint: str) -> dict[str, Any] | None:
        """Get rate limit info for client and endpoint."""
        cfg = self.get_config(endpoint)
        if not cfg:
            return None

        # best-effort remaining:
        if self._redis.client:
            rkey = self._redis.minute_bucket_key(f"{client_id}:{endpoint}:rpm")
            used = self._redis.get(rkey)
            remaining = max(0, cfg.requests_per_minute - used) if used >= 0 else cfg.requests_per_minute
            reset_epoch = int(time.time() // 60) * 60 + 60
        else:
            remaining, reset_epoch = self._sliding.remaining(
                f"{client_id}:{endpoint}:rpm",
                cfg.requests_per_minute,
                cfg.window_size,
            )

        return {
            "endpoint": endpoint,
            "strategy": cfg.strategy,
            "limit_per_minute": cfg.requests_per_minute,
            "limit_per_second": cfg.requests_per_second,
            "burst_limit": cfg.burst_limit,
            "remaining_requests": int(remaining),
            "window_size": cfg.window_size,
            "reset_epoch": reset_epoch,
            "is_blocked": self._security.is_client_blocked(client_id),
            "redis_enabled": bool(self._redis.client),
        }

    def stats(self) -> dict[str, Any]:
        """Get rate limiter statistics."""
        with self._lock:
            active_clients = len(self._client_times)
            total_requests_last_min = sum(len(dq) for dq in self._client_times.values())
            return {
                "endpoint_configs": len(self._endpoint_cfg),
                "active_clients": active_clients,
                "requests_last_minute": total_requests_last_min,
                "redis_enabled": bool(self._redis.client),
                "security": self._security.stats(),
            }

    def clear_old_data(self, max_age_hours: int = 24) -> None:
        """Clear old data. Local-only housekeeping; Redis keys expire."""
        cutoff = time.time() - max_age_hours * 3600
        with self._lock:
            for cid, dq in list(self._client_times.items()):
                while dq and dq[0] < cutoff:
                    dq.popleft()
                if not dq:
                    del self._client_times[cid]
        self._security.clear_expired_blocks()
        logger.info(
            "Rate limiter: cleared local data older than %sh",
            max_age_hours,
        )

    # ---- FastAPI helpers (optional) -----------------------------------------
    def fastapi_dependency(self, endpoint: str):  # pragma: no cover
        """
        Use as: `Depends(rate_limiter.fastapi_dependency("/api/foo"))`.

        Raises HTTPException(429) and sets rate-limit headers when limited.
        """
        if HTTPException is None:

            def _noop_dep(request: Any):
                limited, info = self.is_rate_limited(request, endpoint)
                if limited:
                    reason = info.get("reason")
                    reason_str = reason if reason else "Too Many Requests"
                    msg = f"429 Too Many Requests: {reason_str}"
                    raise RuntimeError(msg)

            return _noop_dep

        def _dep(request: Request):  # type: ignore[name-defined]
            limited, info = self.is_rate_limited(request, endpoint)
            if limited:
                headers = {
                    HDR_LIMIT: str(info.get("limit", 0)),
                    HDR_REMAINING: str(max(0, int(info.get("remaining", 0)))),
                    HDR_RESET: str(int(info.get("reset_epoch", time.time()))),
                }
                retry_after = info.get("retry_after")
                if retry_after is not None:
                    headers[HDR_RETRY_AFTER] = str(int(retry_after))
                reason = info.get("reason")
                reason_str = reason if reason else "Too Many Requests"
                raise HTTPException(
                    status_code=429,
                    detail=reason_str,
                    headers=headers,
                )
            # You can set headers on successful path via response object
            # in route if needed
            return True

        return _dep


# Global instance
rate_limiter = AdvancedRateLimiter()

__all__ = [
    "AdvancedRateLimiter",
    "RateLimitConfig",
    "RateLimitViolation",
    "SecurityMonitor",
    "SlidingWindowRateLimiter",
    "TokenBucketRateLimiter",
    "rate_limiter",
]
