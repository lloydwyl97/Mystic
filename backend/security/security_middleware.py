"""
Security Middleware for Mystic Trading Platform (single-file edition)

Integrates:
- Rate limiting (per-minute + burst)
- JWT authN (bearer) + basic authZ by endpoint family
- Secure error handling with sanitization
- Secure logging & audit hooks
- Security monitoring facade

This file is self-contained. It does NOT import other local modules.
It provides the same public globals:
  - security_middleware
  - security_decorator
  - security_monitor

Env (optional):
  JWT_SECRET            default: None (must be set)
  JWT_ALGORITHM         default: None (must be set)
  RATE_LIMIT_RPM        default: None (must be set)
  RATE_LIMIT_BURST      default: None (must be set)
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
from collections import defaultdict, deque
from typing import Any

import jwt
from fastapi import Request, Response
from fastapi.responses import JSONResponse
from jwt import ExpiredSignatureError, InvalidTokenError

logger = logging.getLogger(__name__)

# =============================================================================
# Utilities: Sanitizer
# =============================================================================

_SENSITIVE_KEYS = [
    "password",
    "token",
    "key",
    "secret",
    "auth",
    "credential",
    "api_key",
    "private_key",
    "access_token",
    "refresh_token",
    "ssn",
    "credit_card",
    "account_number",
    "pin",
]


class _Sanitizer:
    """Sanitizer for sensitive data."""

    def __init__(self) -> None:
        self._replacements = {
            "password": "***PASSWORD***",
            "token": "***TOKEN***",
            "key": "***KEY***",
            "secret": "***SECRET***",
            "auth": "***AUTH***",
            "credential": "***CREDENTIAL***",
            "api_key": "***API_KEY***",
            "private_key": "***PRIVATE_KEY***",
            "access_token": "***ACCESS_TOKEN***",
            "refresh_token": "***REFRESH_TOKEN***",
            "ssn": "***SSN***",
            "credit_card": "***CREDIT_CARD***",
            "account_number": "***ACCOUNT_NUMBER***",
            "pin": "***PIN***",
        }

    def message(self, msg: str) -> str:
        """Sanitize message string."""
        if not msg:
            return msg
        out = msg
        # Keyword redactions (case-insensitive word boundary)
        for k, r in self._replacements.items():
            out = re.sub(rf"\b{re.escape(k)}\b", r, out, flags=re.IGNORECASE)
        # CC numbers
        out = re.sub(
            r"\b\d{4}[- ]?\d{4}[- ]?\d{4}[- ]?\d{4}\b",
            "***CREDIT_CARD***",
            out,
        )
        # SSN
        out = re.sub(r"\b\d{3}-\d{2}-\d{4}\b", "***SSN***", out)
        # Long tokens
        out = re.sub(r"\b[a-zA-Z0-9_-]{32,}\b", "***TOKEN***", out)
        # Emails (keep domain)
        out = re.sub(
            r"\b([a-zA-Z0-9._%+-]+)@([a-zA-Z0-9.-]+\.[A-Za-z]{2,})\b",
            r"***@\2",
            out,
        )
        # IPs (keep first octet)
        return re.sub(
            r"\b(\d{1,3})\.(\d{1,3})\.(\d{1,3})\.(\d{1,3})\b",
            r"\1.***.***.***",
            out,
        )

    def dict(self, data: dict[str, Any]) -> dict[str, Any]:
        """Sanitize dictionary data."""
        if not isinstance(data, dict):
            return data  # type: ignore[return-value]
        out: dict[str, Any] = {}
        for k, v in data.items():
            if isinstance(v, dict):
                out[k] = self.dict(v)
            elif isinstance(v, list):
                out[k] = [self.dict(x) if isinstance(x, dict) else self.message(str(x)) for x in v]
            elif isinstance(v, str):
                if any(p in k.lower() for p in _SENSITIVE_KEYS):
                    out[k] = "***SENSITIVE***"
                else:
                    out[k] = self.message(v)
            else:
                out[k] = v
        return out


# =============================================================================
# Secure Logger (lightweight)
# =============================================================================


class _SecureLogger:
    """Lightweight secure logger."""

    def __init__(self) -> None:
        self._san = _Sanitizer()
        # Console handler if none
        root = logging.getLogger()
        handler_check = any(isinstance(h, logging.StreamHandler) for h in root.handlers)
        if not handler_check:
            sh = logging.StreamHandler()
            sh.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
            root.addHandler(sh)
        if root.level == logging.WARNING:
            root.setLevel(logging.INFO)

        # in-memory counters for stats
        self._security_events: deque[dict[str, Any]] = deque(maxlen=500)

    def _build(self, msg: str, kwargs: dict[str, Any]) -> str:
        """Build sanitized log message."""
        smsg = self._san.message(msg)
        skw = self._san.dict(kwargs) if kwargs else {}
        return f"{smsg} - {json.dumps(skw, separators=(',', ':'))}" if skw else smsg

    def log_info(self, message: str, **kwargs: Any) -> None:
        """Log info message."""
        logging.getLogger().info(self._build(message, kwargs))

    def log_warning(self, message: str, **kwargs: Any) -> None:
        """Log warning message."""
        logging.getLogger().warning(self._build(message, kwargs))

    def log_error(
        self,
        message: str,
        error: Exception | None = None,
        **kwargs: Any,
    ) -> None:
        """Log error message."""
        base = self._san.message(message)
        if error:
            base += f" - Error: {self._san.message(str(error))}"
        skw = self._san.dict(kwargs) if kwargs else {}
        suffix = f" - {json.dumps(skw, separators=(',', ':'))}" if skw else ""
        logging.getLogger().error(base + suffix)

    def log_security(
        self,
        message: str,
        severity: str = "medium",
        **kwargs: Any,
    ) -> None:
        """Log security event."""
        entry = {
            "ts": time.time(),
            "sev": severity,
            "msg": self._san.message(message),
            "extra": self._san.dict(kwargs),
        }
        self._security_events.append(entry)
        logging.getLogger().warning(self._build(f"[SECURITY-{severity.upper()}] {message}", kwargs))

    def log_access_attempt(
        self,
        user_id: str,
        action: str,
        resource: str,
        success: bool,
        client_ip: str | None = None,
    ) -> None:
        """Log access attempt."""
        ip_str = client_ip if client_ip else "-"
        self.log_info(
            "Access attempt",
            user_id=user_id,
            action=action,
            resource=resource,
            success=bool(success),
            client_ip=ip_str,
        )

    def log_authentication_attempt(
        self,
        user_id: str,
        method: str,
        success: bool,
        client_ip: str | None = None,
    ) -> None:
        """Log authentication attempt."""
        ip_str = client_ip if client_ip else "-"
        self.log_info(
            "Authentication attempt",
            user_id=user_id,
            method=method,
            success=bool(success),
            client_ip=ip_str,
        )

    def get_logging_stats(self) -> dict[str, Any]:
        """Get logging statistics."""
        recent = list(self._security_events)[-10:] if self._security_events else []
        return {
            "security_events": len(self._security_events),
            "recent_security": recent,
        }


secure_logger = _SecureLogger()

# =============================================================================
# Error Handler (lightweight)
# =============================================================================


class _ErrorHandler:
    """Lightweight error handler."""

    def __init__(self) -> None:
        self._san = _Sanitizer()
        self._log: deque[dict[str, Any]] = deque(maxlen=1000)

    def handle_error(
        self,
        error: Exception,
        client_id: str | None = None,
        endpoint: str | None = None,
        include_details: bool = False,
    ) -> tuple[dict[str, Any], int]:
        """Handle error and return response body and status code."""
        msg = str(error).lower()
        code = 500
        body = {
            "error": True,
            "error_code": "INTERNAL_ERROR",
            "message": "Internal server error",
            "timestamp": time.time(),
        }
        if "validation" in msg or "invalid" in msg:
            code = 400
            body["error_code"] = "VALIDATION_ERROR"
            body["message"] = "Invalid request parameters"
        elif "timeout" in msg or "connection" in msg or "temporarily unavailable" in msg:
            code = 503
            body["error_code"] = "CONNECTION_ERROR"
            body["message"] = "Service temporarily unavailable"

        if include_details:
            body["details"] = {
                "error_type": type(error).__name__,
                "message": self._san.message(str(error)),
            }

        # Log compact record
        rec = {
            "ts": time.time(),
            "type": type(error).__name__,
            "msg": self._san.message(str(error)),
            "client": client_id,
            "endpoint": endpoint,
            "status": code,
        }
        self._log.append(rec)
        log_level = logging.ERROR if code >= 500 else logging.WARNING
        logging.getLogger().log(log_level, f"Handled error: {rec}")
        return body, code

    def get_error_stats(self) -> dict[str, Any]:
        """Get error statistics."""
        counts: dict[str, int] = defaultdict(int)
        recent = 0
        now = time.time()
        for r in self._log:
            counts[r["type"]] += 1
            if now - r["ts"] < 3600:
                recent += 1
        return {
            "total_errors": len(self._log),
            "error_types": dict(counts),
            "recent_errors": recent,
        }


error_handler = _ErrorHandler()

# =============================================================================
# Rate Limiter (sliding window + burst)
# =============================================================================


class _RateLimiter:
    """Rate limiter with sliding window and burst protection."""

    def __init__(self) -> None:
        rpm_str = os.getenv("RATE_LIMIT_RPM")
        burst_str = os.getenv("RATE_LIMIT_BURST")

        if rpm_str:
            try:
                self.rpm = int(rpm_str)
            except (ValueError, TypeError):
                self.rpm = None
        else:
            self.rpm = None

        if burst_str:
            try:
                self.burst = int(burst_str)
            except (ValueError, TypeError):
                self.burst = None
        else:
            self.burst = None

        self._minute: dict[str, deque[float]] = defaultdict(deque)
        self._second: dict[str, deque[float]] = defaultdict(deque)

        def _nested_int_counts() -> defaultdict:
            return defaultdict(int)

        self._counts: dict[str, dict[str, int]] = defaultdict(_nested_int_counts)

    def _cid(self, request: Request) -> str:
        """Extract client ID from request."""
        cid = request.headers.get("X-Client-ID")
        if cid:
            return cid
        if hasattr(request, "user") and getattr(request, "user", None):
            try:
                return f"user_{request.user.id}"
            except Exception:
                pass
        client_host = request.client.host if request.client else None
        if client_host:
            return f"ip_{client_host}"
        return "ip_unknown"

    def _now(self) -> float:
        """Get current timestamp."""
        return time.time()

    def is_rate_limited(self, request: Request, endpoint: str) -> tuple[bool, dict[str, Any]]:
        """Check if request is rate limited."""
        if self.rpm is None or self.burst is None:
            # No rate limits configured - allow all
            return False, {"remaining_requests": 0, "reset_time": 0}

        cid = self._cid(request)
        now = self._now()

        k_min = f"{cid}:{endpoint}:min"
        k_sec = f"{cid}:{endpoint}:sec"

        # prune old timestamps
        w_min = self._minute[k_min]
        while w_min and now - w_min[0] > 60.0:
            w_min.popleft()
        w_sec = self._second[k_sec]
        while w_sec and now - w_sec[0] > 1.0:
            w_sec.popleft()

        allowed_min = len(w_min) < self.rpm
        allowed_sec = len(w_sec) < self.burst

        # increment counters (we count even if limited to drive
        # suspicious detection externally if needed)
        self._counts[cid][endpoint] += 1

        if not (allowed_min and allowed_sec):
            remaining = max(0, self.rpm - len(w_min))
            retry_after = 1 if not allowed_sec else 60
            return True, {
                "blocked": False,
                "reason": "Rate limit exceeded",
                "retry_after": retry_after,
                "remaining_requests": remaining,
            }

        # record the call
        w_min.append(now)
        w_sec.append(now)

        remaining = max(0, self.rpm - len(w_min))
        return False, {
            "remaining_requests": remaining,
            "reset_time": int(now + 60),
        }

    def get_rate_limit_info(self, client_id: str, endpoint: str) -> dict[str, Any]:
        """Get rate limit info for client and endpoint."""
        if self.rpm is None or self.burst is None:
            return {
                "endpoint": endpoint,
                "strategy": "none",
                "limit_per_minute": 0,
                "limit_per_second": 0,
                "burst_limit": 0,
                "remaining_requests": 0,
                "window_size": 60,
                "reset_time": 0,
                "is_blocked": False,
            }

        # Compute roughly from the per-minute window
        k_min = f"{client_id}:{endpoint}:min"
        now = self._now()
        w_min = self._minute[k_min]
        while w_min and now - w_min[0] > 60.0:
            w_min.popleft()
        remaining = max(0, self.rpm - len(w_min))
        reset = int(now + 60)
        return {
            "endpoint": endpoint,
            "strategy": "sliding_window+burst",
            "limit_per_minute": self.rpm,
            "limit_per_second": self.burst,
            "burst_limit": self.burst,
            "remaining_requests": remaining,
            "window_size": 60,
            "reset_time": reset,
            "is_blocked": False,
        }

    def get_rate_limit_stats(self) -> dict[str, Any]:
        """Get rate limit statistics."""
        total = sum(sum(d.values()) for d in self._counts.values())
        return {
            "rpm": self.rpm,
            "burst": self.burst,
            "active_clients": len(self._counts),
            "total_requests": total,
        }


rate_limiter = _RateLimiter()

# =============================================================================
# Authentication (JWT)
# =============================================================================


class _AuthManager:
    """JWT authentication manager."""

    def __init__(self) -> None:
        self.secret = os.getenv("JWT_SECRET")
        self.alg = os.getenv("JWT_ALGORITHM")
        self._failed: dict[str, int] = defaultdict(int)

    async def verify_token(self, token: str) -> dict[str, Any]:
        """Verify JWT token."""
        if not self.secret or not self.alg:
            return {
                "valid": False,
                "reason": "JWT configuration missing",
            }

        try:
            payload = jwt.decode(token, self.secret, algorithms=[self.alg])
            # expected optional fields: user_id, roles, permissions
            user_id = payload.get("user_id")
            if not user_id:
                user_id = payload.get("sub")
            if not user_id:
                user_id = None

            roles = payload.get("roles")
            if not isinstance(roles, list):
                roles = []

            perms = payload.get("permissions")
            if not isinstance(perms, list):
                perms = []
        except ExpiredSignatureError:
            return {"valid": False, "reason": "Token expired"}
        except InvalidTokenError:
            return {"valid": False, "reason": "Invalid token"}
        except Exception as e:
            logger.exception(f"Error validating token: {e}")
            return {"valid": False, "error": str(e)}
        else:
            return {
                "valid": True,
                "user_info": {
                    "user_id": user_id,
                    "roles": roles,
                    "permissions": perms,
                },
            }

    def get_auth_stats(self) -> dict[str, Any]:
        """Get authentication statistics."""
        return {
            "failed_attempts": dict(self._failed),
            "uses_jwt": True,
            "algorithm": self.alg,
        }


auth_manager = _AuthManager()

# =============================================================================
# Security Middleware (public API)
# =============================================================================


class SecurityMiddleware:
    """Comprehensive security middleware for FastAPI"""

    def __init__(self) -> None:
        self.secure_logger = secure_logger
        self.error_handler = error_handler
        self.rate_limiter = rate_limiter
        self.auth_manager = auth_manager

    async def process_request(self, request: Request) -> Response | None:
        """Process incoming request with security checks"""
        start_time = time.time()
        client_ip = self._get_client_ip(request)
        endpoint = request.url.path

        try:
            # 1) Rate limit
            limited, info = self.rate_limiter.is_rate_limited(request, endpoint)
            if limited:
                return self._create_rate_limit_response(info)

            # 2) AuthN (protected endpoints)
            if self._is_protected_endpoint(endpoint):
                auth_result = await self._authenticate_request(request)
                if not auth_result["authenticated"]:
                    return self._create_unauthorized_response(auth_result["reason"])

                # 3) AuthZ
                if not self._authorize_request(auth_result["user_info"], endpoint):
                    return self._create_forbidden_response()

                # attach user to state
                request.state.user = auth_result["user_info"]

            # 4) Access log
            user_obj = getattr(request.state, "user", None)
            user_id_val = user_obj.get("user_id") if isinstance(user_obj, dict) else None
            user_id = user_id_val if user_id_val else "anonymous"
            self.secure_logger.log_access_attempt(
                user_id=user_id,
                action=request.method,
                resource=endpoint,
                success=True,
                client_ip=client_ip,
            )

            # 5) Timing log (pre-handler perspective)
            proc_ms = time.time() - start_time
            self.secure_logger.log_info(
                f"Request processed: {request.method} {endpoint}",
                processing_time=round(proc_ms, 6),
                client_ip=client_ip,
            )
        except Exception as e:
            # Error path
            body, status = self.error_handler.handle_error(
                e,
                client_id=self._get_client_id(request),
                endpoint=endpoint,
                include_details=False,
            )
            self.secure_logger.log_error(
                "Security middleware error",
                error=e,
                endpoint=endpoint,
                client_ip=client_ip,
            )
            return JSONResponse(status_code=status, content=body)

    async def process_response(self, request: Request, response: Response) -> Response:
        """Add security + rate-limit headers"""
        # Security headers
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
        response.headers["Content-Security-Policy"] = "default-src 'self'"

        # Rate headers
        client_id = self._get_client_id(request)
        endpoint = request.url.path
        info = self.rate_limiter.get_rate_limit_info(client_id, endpoint)
        limit_val = info.get("limit_per_minute")
        remaining_val = info.get("remaining_requests")
        reset_val = info.get("reset_time")

        if limit_val is not None:
            response.headers["X-RateLimit-Limit"] = str(limit_val)
        if remaining_val is not None:
            response.headers["X-RateLimit-Remaining"] = str(remaining_val)
        if reset_val is not None:
            response.headers["X-RateLimit-Reset"] = str(int(reset_val))

        return response

    # ---- Helpers -----------------------------------------------------------
    def _get_client_ip(self, request: Request) -> str:
        """Extract client IP from request."""
        xf = request.headers.get("X-Forwarded-For")
        if xf:
            return xf.split(",")[0].strip()
        xr = request.headers.get("X-Real-IP")
        if xr:
            return xr
        client_host = request.client.host if request.client else None
        if client_host:
            return client_host
        return "unknown"

    def _get_client_id(self, request: Request) -> str:
        """Extract client ID from request."""
        cid = request.headers.get("X-Client-ID")
        if cid:
            return cid
        if hasattr(request.state, "user") and request.state.user:
            user_id_val = request.state.user.get("user_id")
            if user_id_val:
                return f"user_{user_id_val}"
        ip = self._get_client_ip(request)
        return f"ip_{ip}"

    def _is_protected_endpoint(self, endpoint: str) -> bool:
        """Check if endpoint requires protection."""
        protected = [
            "/api/auth/",
            "/api/admin/",
            "/api/user/",
            "/api/trading/",
            "/api/portfolio/",
            "/api/strategies/",
        ]
        # Exclude WebSocket endpoints from protection to allow upgrades
        if endpoint.startswith("/ws") or "/ws" in endpoint:
            return False
        return any(p in endpoint for p in protected)

    async def _authenticate_request(self, request: Request) -> dict[str, Any]:
        """Authenticate request."""
        try:
            auth_header = request.headers.get("Authorization")
            if not auth_header:
                return {
                    "authenticated": False,
                    "reason": "No authorization header provided",
                }
            if not auth_header.startswith("Bearer "):
                return {
                    "authenticated": False,
                    "reason": "Invalid authorization header format",
                }
            token = auth_header[7:]
            vr = await self.auth_manager.verify_token(token)
            if not vr.get("valid"):
                reason = vr.get("reason")
                reason_str = reason if reason else "Invalid token"
                return {
                    "authenticated": False,
                    "reason": reason_str,
                }
            return {"authenticated": True, "user_info": vr["user_info"]}
        except Exception as e:
            logger.exception(f"Authentication error: {e}")
            return {
                "authenticated": False,
                "reason": "Authentication failed",
            }

    def _authorize_request(self, user_info: dict[str, Any], endpoint: str) -> bool:
        """Authorize request based on permissions."""
        # Simple mapping: require presence of a keyword permission
        perms_raw = user_info.get("permissions")
        perms = set(perms_raw) if isinstance(perms_raw, list) else set()
        needed = set(self._get_required_permissions(endpoint))
        return needed.issubset(perms)

    def _get_required_permissions(self, endpoint: str) -> list[str]:
        """Get required permissions for endpoint."""
        if "/api/admin/" in endpoint:
            return ["admin"]
        if "/api/trading/" in endpoint:
            return ["trading"]
        if "/api/portfolio/" in endpoint:
            return ["portfolio"]
        return []

    # ---- Responses ---------------------------------------------------------
    def _create_rate_limit_response(self, info: dict[str, Any]) -> JSONResponse:
        """Create rate limit response."""
        reason = info.get("reason")
        reason_str = reason if reason else "Rate limit exceeded"
        retry_val = info.get("retry_after")
        retry_after = retry_val if retry_val is not None else 60
        remaining_val = info.get("remaining_requests")
        remaining = remaining_val if remaining_val is not None else 0

        return JSONResponse(
            status_code=429,
            content={
                "error": True,
                "error_code": "RATE_LIMIT_EXCEEDED",
                "message": reason_str,
                "retry_after": retry_after,
                "remaining_requests": remaining,
            },
            headers={
                "Retry-After": str(retry_after),
                "X-RateLimit-Remaining": str(remaining),
            },
        )

    def _create_unauthorized_response(self, reason: str) -> JSONResponse:
        """Create unauthorized response."""
        return JSONResponse(
            status_code=401,
            content={
                "error": True,
                "error_code": "UNAUTHORIZED",
                "message": reason,
            },
        )

    def _create_forbidden_response(self) -> JSONResponse:
        """Create forbidden response."""
        return JSONResponse(
            status_code=403,
            content={
                "error": True,
                "error_code": "FORBIDDEN",
                "message": ("Insufficient permissions to access this resource"),
            },
        )


# =============================================================================
# Decorators & Monitor (facades)
# =============================================================================


class SecurityDecorator:
    """Placeholders to allow future per-endpoint decorators without
    changing call-sites."""

    def __init__(self, security_middleware: SecurityMiddleware) -> None:
        self.middleware = security_middleware

    def require_auth(self, _required_permissions: list[str] | None = None):
        """Decorator for requiring authentication."""

        def decorator(func):
            async def wrapper(*args, **kwargs):
                # Real integration would inspect request and enforce;
                # here we pass-through.
                return await func(*args, **kwargs)

            return wrapper

        return decorator

    def rate_limited(self, _requests_per_minute: int = 100):
        """Decorator for rate limiting."""

        def decorator(func):
            async def wrapper(*args, **kwargs):
                return await func(*args, **kwargs)

            return wrapper

        return decorator


class SecurityMonitor:
    """Aggregates security stats across sub-systems."""

    def __init__(self, security_middleware: SecurityMiddleware) -> None:
        self.middleware = security_middleware
        self._events: deque[dict[str, Any]] = deque(maxlen=1000)

    def log_security_event(
        self,
        event_type: str,
        severity: str,
        description: str,
        **kwargs,
    ):
        """Log security event."""
        evt = {
            "event_type": event_type,
            "severity": severity,
            "description": description,
            "timestamp": time.time(),
            **kwargs,
        }
        self._events.append(evt)
        self.middleware.secure_logger.log_security(description, severity=severity, **kwargs)

    def get_security_stats(self) -> dict[str, Any]:
        """Get security statistics."""
        return {
            "rate_limiting": (self.middleware.rate_limiter.get_rate_limit_stats()),
            "authentication": (self.middleware.auth_manager.get_auth_stats()),
            "error_handling": (self.middleware.error_handler.get_error_stats()),
            "logging": (self.middleware.secure_logger.get_logging_stats()),
            "security_events": len(self._events),
        }


# =============================================================================
# Globals (public)
# =============================================================================

security_middleware = SecurityMiddleware()
security_decorator = SecurityDecorator(security_middleware)
security_monitor = SecurityMonitor(security_middleware)
