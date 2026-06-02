"""
Secure Error Handler for Mystic Trading Platform (single file)

Goals:
- Prevent sensitive info leakage
- Consistent JSON error envelopes with HTTP status codes
- Lightweight security incident detection (auth errors, SQLi/path
  traversal, burst errors)
- Structured logging, safe sanitization (keys + values), and log
  retention
- Optional FastAPI helpers (won't error if FastAPI isn't installed)

Notes:
- Pure-Python and thread-safe; degrades gracefully without external
  services.
- Does not downcase messages or blindly nuke content; uses targeted
  redaction with regex.
"""

from __future__ import annotations

import hashlib
import logging
import re
import secrets
import threading
import time
from collections import defaultdict, deque
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

# Direct imports for production
from fastapi import FastAPI, Request  # type: ignore[import-not-found]
from fastapi.responses import JSONResponse  # type: ignore[import-not-found]

from backend.utils.exceptions import MysticException  # type: ignore[import-not-found]

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
# Keys that should always be redacted if they appear as dict keys or
# header names
SENSITIVE_KEYS = {
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
    "authorization",
    "x-api-key",
    "x-auth-token",
}

# Regex patterns for sensitive values (case-insensitive)
SENSITIVE_VALUE_PATTERNS: list[re.Pattern] = [
    re.compile(
        r"(?:bearer\s+)?[a-z0-9-_]{20,}\.[a-z0-9-_]{10,}\."
        r"[a-z0-9-_]{10,}",
        re.IGNORECASE,
    ),  # JWT-ish
    re.compile(
        r"(?:^|[^a-z0-9])[a-f0-9]{32}(?:[^a-z0-9]|$)",
        re.IGNORECASE,
    ),  # 32-hex
    re.compile(r"sk_live_[a-z0-9]{24,}", re.IGNORECASE),  # stripe-ish
    re.compile(
        r"(?<![A-Za-z])[A-Za-z0-9_\-]{24,}(?![A-Za-z])",
        re.IGNORECASE,
    ),  # generic long tokens
]

# Regex to redact obvious filesystem paths
PATH_PATTERN = re.compile(
    r"((?:[A-Za-z]:\\|/)(?:[^\s\"'<>]{1,256}))",  # windows drive or
    # absolute unix, up to 256 chars
    re.UNICODE,
)

# Simple SQLi-ish and traversal indicators (kept conservative to reduce
# false positives)
SQLI_HINTS = [
    r"(?i)\bunion\s+select\b",
    r"(?i)\bdrop\s+table\b",
    r"(?i)\binsert\s+into\b",
    r"(?i)\bupdate\s+\w+\s+set\b",
    r"(?i)\bor\s+1\s*=\s*1\b",
    r"(?i)\bsleep\(\s*\d+\s*\)",
]
SQLI_REGEXES = [re.compile(p) for p in SQLI_HINTS]

TRAVERSAL_HINTS = [
    re.compile(r"\.\./"),
    re.compile(r"\.\.\\"),
    re.compile(r"(?i)/etc/passwd"),
    re.compile(r"(?i)\\windows\\system32"),
]

# Error log / incident configuration
MAX_ERROR_LOG_SIZE = 1000
ERROR_RETENTION_HOURS = 24
# Burst threshold: #errors per client within WINDOW seconds
SECURITY_BURST_THRESHOLD = 10
SECURITY_BURST_WINDOW_SEC = 60

# Redaction tokens
REDACTED = "***REDACTED***"
REDACTED_PASSWORD = "***PASSWORD***"
REDACTED_TOKEN = "***TOKEN***"
REDACTED_KEY = "***KEY***"
REDACTED_SECRET = "***SECRET***"

# Max message length we'll ever emit back to clients (avoid log injection
# / bloat)
MAX_PUBLIC_MESSAGE_LEN = 500

# Generic security-safe error message
GENERIC_ERROR_MESSAGE = "Internal server error"


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------
@dataclass
class SecurityIncident:
    """Security incident record."""

    incident_id: str
    incident_type: str  # e.g., "authentication_error"
    severity: str  # 'low', 'medium', 'high', 'critical'
    description: str
    timestamp: float
    client_id: str | None = None
    endpoint: str | None = None
    error_details: dict[str, Any] | None = None


# ---------------------------------------------------------------------------
# Sanitizer
# ---------------------------------------------------------------------------
class ErrorSanitizer:
    """Sanitizes messages and dicts to prevent sensitive info disclosure."""

    def __init__(self) -> None:
        self._sensitive_keys = {k.lower() for k in SENSITIVE_KEYS}

    def sanitize_message(self, message: str) -> str:
        """Redact sensitive patterns and paths; preserve general semantics."""
        if not message:
            return GENERIC_ERROR_MESSAGE

        msg = str(message)

        # Redact file-system paths
        msg = PATH_PATTERN.sub(REDACTED, msg)

        # Redact value patterns (JWTs, long tokens, etc.)
        for rx in SENSITIVE_VALUE_PATTERNS:
            if rx.search(msg):
                msg = rx.sub(REDACTED, msg)

        # Limit size
        if len(msg) > MAX_PUBLIC_MESSAGE_LEN:
            msg = msg[:MAX_PUBLIC_MESSAGE_LEN] + "…"

        # Very generic fallback if message still contains obvious
        # "traceback" markers
        if "Traceback (most recent call last):" in message or "stack trace" in message.lower():
            return GENERIC_ERROR_MESSAGE

        return msg

    def sanitize_kv(self, key: str, value: Any) -> Any:
        """Sanitize a (key, value) pair based on key semantics and value
        content."""
        k = key.lower()
        if k in self._sensitive_keys:
            return REDACTED
        if isinstance(value, str):
            return self.sanitize_message(value)
        if isinstance(value, Mapping):
            return self.sanitize_dict(value)  # type: ignore[arg-type]
        if isinstance(value, list):
            return [self.sanitize_kv(k, v) for v in value]
        return value

    def sanitize_dict(self, data: Mapping[str, Any]) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for k, v in data.items():
            out[k] = self.sanitize_kv(k, v)
        return out

    def scrub_headers(self, headers: Mapping[str, str]) -> dict[str, str]:
        """Redact common sensitive HTTP headers for safe logging."""
        return {k: (REDACTED if k.lower() in self._sensitive_keys else self.sanitize_message(v)) for k, v in headers.items()}


# ---------------------------------------------------------------------------
# Security Incident Detector
# ---------------------------------------------------------------------------
class SecurityIncidentDetector:
    """Detects suspicious patterns and burst errors."""

    def __init__(self) -> None:
        self._incidents: deque[SecurityIncident] = deque(maxlen=100)
        self._error_counts_by_type: dict[str, int] = defaultdict(int)
        self._client_error_times: dict[str, deque[float]] = defaultdict(lambda: deque(maxlen=256))
        self._lock = threading.Lock()

    def _new_incident_id(self) -> str:
        # Short, random, stable ID for correlation
        return secrets.token_hex(8)

    def detect(
        self,
        error: Exception,
        *,
        client_id: str | None = None,
        endpoint: str | None = None,
        message: str | None = None,
    ) -> SecurityIncident | None:
        now = time.time()
        etype = type(error).__name__
        emsg = (message if message is not None else str(error)) or ""

        incident: SecurityIncident | None = None

        with self._lock:
            self._error_counts_by_type[etype] += 1

            # Burst detection per client
            if client_id:
                times = self._client_error_times[client_id]
                times.append(now)
                # Keep only last WINDOW seconds
                while times and (now - times[0]) > SECURITY_BURST_WINDOW_SEC:
                    times.popleft()
                if len(times) > SECURITY_BURST_THRESHOLD:
                    incident = SecurityIncident(
                        incident_id=self._new_incident_id(),
                        incident_type="excessive_errors",
                        severity="medium",
                        description=(f"Excessive errors from client {client_id} in last {SECURITY_BURST_WINDOW_SEC}s"),
                        timestamp=now,
                        client_id=client_id,
                        endpoint=endpoint,
                        error_details={
                            "count_window": len(times),
                            "window_sec": SECURITY_BURST_WINDOW_SEC,
                        },
                    )

            # Auth errors
            lowered = emsg.lower()
            if any(
                w in lowered
                for w in (
                    "unauthorized",
                    "forbidden",
                    "invalid token",
                    "invalid signature",
                )
            ):
                incident = SecurityIncident(
                    incident_id=self._new_incident_id(),
                    incident_type="authentication_error",
                    severity="medium",
                    description=f"Authentication/authorization error: {etype}",
                    timestamp=now,
                    client_id=client_id,
                    endpoint=endpoint,
                    error_details={"error_type": etype},
                )

            # SQLi-like patterns
            if any(rx.search(emsg) for rx in SQLI_REGEXES):
                incident = SecurityIncident(
                    incident_id=self._new_incident_id(),
                    incident_type="sql_injection_attempt",
                    severity="high",
                    description=(f"Potential SQL injection attempt: {etype}"),
                    timestamp=now,
                    client_id=client_id,
                    endpoint=endpoint,
                    error_details={"sample": emsg[:180]},
                )

            # Path traversal indicators
            if any(rx.search(emsg) for rx in TRAVERSAL_HINTS):
                incident = SecurityIncident(
                    incident_id=self._new_incident_id(),
                    incident_type="path_traversal_attempt",
                    severity="high",
                    description=(f"Potential path traversal attempt: {etype}"),
                    timestamp=now,
                    client_id=client_id,
                    endpoint=endpoint,
                    error_details={"sample": emsg[:180]},
                )

            if incident:
                self._incidents.append(incident)
                logger.warning("Security incident detected: %s", incident.description)

        return incident

    # Stats
    def get_security_stats(self) -> dict[str, Any]:
        now = time.time()
        with self._lock:
            recent = [i for i in self._incidents if (now - i.timestamp) < 3600]
            per_client_rate = {c: len([t for t in times if now - t < SECURITY_BURST_WINDOW_SEC]) for c, times in self._client_error_times.items()}
            return {
                "total_incidents": len(self._incidents),
                "error_counts_by_type": dict(self._error_counts_by_type),
                "recent_incidents_last_hour": len(recent),
                "per_client_burst_last_window": per_client_rate,
            }

    @property
    def incidents(self) -> list[SecurityIncident]:
        with self._lock:
            return list(self._incidents)


# ---------------------------------------------------------------------------
# Secure Error Handler
# ---------------------------------------------------------------------------
class SecureErrorHandler:
    """Centralized error handling, sanitization, logging, and incident
    detection."""

    def __init__(self) -> None:
        self.sanitizer = ErrorSanitizer()
        self.detector = SecurityIncidentDetector()
        self._error_log: deque[dict[str, Any]] = deque(maxlen=MAX_ERROR_LOG_SIZE)
        self._lock = threading.Lock()

    # Public API: returns (payload, http_status)
    def handle_error(
        self,
        error: Exception,
        *,
        client_id: str | None = None,
        endpoint: str | None = None,
        include_details: bool = False,
        context: Mapping[str, Any] | None = None,
    ) -> tuple[dict[str, Any], int]:
        """Main entry point. Produces a sanitized JSON error payload and HTTP
        status."""
        try:
            # Incident detection (non-blocking)
            incident = self.detector.detect(
                error,
                client_id=client_id,
                endpoint=endpoint,
                message=str(error),
            )

            # Structured logging (sanitized)
            self._log_error(error, client_id, endpoint, incident, context=context)

            # Specific exception handling
            if isinstance(error, MysticException):
                out = self._handle_mystic_exception(error, include_details)
                return out

            lowered = str(error).lower()
            if "validation" in lowered or "invalid" in lowered:
                out = self._handle_validation_error(error, include_details)
                return out

            if any(
                w in lowered
                for w in (
                    "connection",
                    "timeout",
                    "timed out",
                    "temporarily unavailable",
                )
            ):
                out = self._handle_connection_error(error, include_details)
                return out

            out = self._handle_generic_error(error, include_details)
        except Exception as e:
            logger.exception(f"Error handling error: {e}")
            return ({"error": True, "message": "Internal server error"}, 500)
        else:
            return out

    # ---- Response builders -------------------------------------------
    def _base_envelope(
        self,
        *,
        code: str,
        msg: str,
        _status: int,
        include_details: bool,
        error: Exception,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "error": True,
            "error_code": code,
            "message": self.sanitizer.sanitize_message(msg),
            "timestamp": time.time(),
        }
        if include_details:
            # provide stable correlation id for logs
            err_id = hashlib.sha1(f"{type(error).__name__}:{msg}".encode()).hexdigest()[:12]
            payload["details"] = {
                "error_type": type(error).__name__,
                "error_id": err_id,
            }
        return payload

    def _handle_mystic_exception(self, error: MysticException, include_details: bool) -> tuple[dict[str, Any], int]:
        code = getattr(error, "error_code", None)
        if not code:
            code = "MYSTIC_ERROR"

        status = getattr(error, "status_code", None)
        if status is None:
            status = 400

        msg = str(error)
        if not msg:
            msg = "Mystic error"

        return self._base_envelope(
            code=code,
            msg=msg,
            status=status,
            include_details=include_details,
            error=error,
        ), status

    def _handle_validation_error(self, error: Exception, include_details: bool) -> tuple[dict[str, Any], int]:
        return self._base_envelope(
            code="VALIDATION_ERROR",
            msg="Invalid request parameters",
            status=400,
            include_details=include_details,
            error=error,
        ), 400

    def _handle_connection_error(self, error: Exception, include_details: bool) -> tuple[dict[str, Any], int]:
        return self._base_envelope(
            code="CONNECTION_ERROR",
            msg="Service temporarily unavailable",
            status=503,
            include_details=include_details,
            error=error,
        ), 503

    def _handle_generic_error(self, error: Exception, include_details: bool) -> tuple[dict[str, Any], int]:
        return self._base_envelope(
            code="INTERNAL_ERROR",
            msg=GENERIC_ERROR_MESSAGE,
            status=500,
            include_details=include_details,
            error=error,
        ), 500

    # ---- Logging & retention -----------------------------------------
    def _log_error(
        self,
        error: Exception,
        client_id: str | None,
        endpoint: str | None,
        incident: SecurityIncident | None,
        *,
        context: Mapping[str, Any] | None = None,
    ) -> None:
        info = {
            "error_type": type(error).__name__,
            "error_message": self.sanitizer.sanitize_message(str(error)),
            "client_id": client_id,
            "endpoint": endpoint,
            "timestamp": time.time(),
            "has_incident": bool(incident),
        }
        if context:
            try:
                info["context"] = self.sanitizer.sanitize_dict(context)  # type: ignore[arg-type]
            except Exception:
                info["context"] = {"_error": "context_sanitize_failed"}

        with self._lock:
            self._error_log.append(info)

        # System log (structured)
        if incident:
            logger.error("Handled error with incident | %s", info)
        else:
            logger.warning("Handled error | %s", info)

    # ---- Stats & housekeeping -----------------------------------------------
    def get_error_stats(self) -> dict[str, Any]:
        cutoff = time.time() - 3600
        with self._lock:
            types = defaultdict(int)
            recent = 0
            for e in self._error_log:
                types[e["error_type"]] += 1
                if e["timestamp"] >= cutoff:
                    recent += 1
            return {
                "total_errors": len(self._error_log),
                "error_types": dict(types),
                "recent_errors_last_hour": recent,
                "security": self.detector.get_security_stats(),
            }

    def clear_old_errors(self, max_age_hours: int = ERROR_RETENTION_HOURS) -> None:
        cutoff_time = time.time() - (max_age_hours * 3600)
        with self._lock:
            self._error_log = deque(
                (e for e in self._error_log if e["timestamp"] > cutoff_time),
                maxlen=MAX_ERROR_LOG_SIZE,
            )
        logger.info("Cleared old error logs older than %sh", max_age_hours)

    # ---- Optional FastAPI helpers -------------------------------------------
    def to_fastapi_response(
        self,
        error: Exception,
        *,
        client_id: str | None = None,
        endpoint: str | None = None,
        include_details: bool = False,
        context: Mapping[str, Any] | None = None,
    ):
        """Return a FastAPI JSONResponse. No-op if FastAPI not installed."""
        if JSONResponse is None:
            # Fallback structure to avoid import errors
            payload, status = self.handle_error(
                error,
                client_id=client_id,
                endpoint=endpoint,
                include_details=include_details,
                context=context,
            )
            return payload, status
        payload, status = self.handle_error(
            error,
            client_id=client_id,
            endpoint=endpoint,
            include_details=include_details,
            context=context,
        )
        return JSONResponse(content=payload, status_code=status)

    def install_fastapi_handlers(self, app: FastAPI, *, include_details: bool = False) -> None:  # type: ignore[name-defined]
        """Register global exception handlers on a FastAPI app."""
        if FastAPI is None or JSONResponse is None:
            return

        @app.exception_handler(MysticException)  # type: ignore[misc]
        async def _mystic_exc_handler(request: Request, exc: MysticException):  # type: ignore[name-defined]
            return self.to_fastapi_response(
                exc,
                client_id=self._client_id_of(request),
                endpoint=str(request.url),
                include_details=include_details,
            )

        @app.exception_handler(Exception)  # type: ignore[misc]
        async def _generic_exc_handler(
            request: Request,
            exc: Exception,
        ):  # type: ignore[name-defined]
            return self.to_fastapi_response(
                exc,
                client_id=self._client_id_of(request),
                endpoint=str(request.url),
                include_details=False,
            )

    @staticmethod
    def _client_id_of(request: Any) -> str:
        try:
            # Prefer IP; fall back to UA hash
            ip = request.client.host if request and request.client else None
            if ip:
                return ip
            ua = request.headers.get("user-agent")
            if ua:
                return hashlib.md5(ua.encode("utf-8")).hexdigest()[:8]
        except Exception:
            return "unknown"
        else:
            return "unknown"


# Global instance
error_handler = SecureErrorHandler()
