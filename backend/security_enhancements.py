from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
import secrets
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from functools import wraps
from pathlib import Path
from typing import Any

import structlog
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC

# backend may or may not be required depending on cryptography version
try:
    from cryptography.hazmat.backends import default_backend  # type: ignore[import-not-found]
except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError):
    default_backend = None  # type: ignore[assignment]

logger = structlog.get_logger()


@dataclass
class SecurityEvent:
    event_id: str
    event_type: str
    user_id: str | None
    ip_address: str
    endpoint: str
    method: str
    timestamp: datetime
    details: dict[str, Any]
    severity: str
    status: str


@dataclass
class APIKey:
    key_id: str
    user_id: str
    key_hash: str
    permissions: list[str]
    created_at: datetime
    expires_at: datetime | None
    last_used: datetime | None
    is_active: bool
    rate_limit: int


class EncryptionManager:
    def __init__(self, master_key: str | bytes | None = None, passphrase: str | None = None) -> None:
        key_env = master_key if master_key is not None else os.getenv("MASTER_ENCRYPTION_KEY")
        if isinstance(key_env, str):
            key_bytes = key_env.encode()
        elif isinstance(key_env, (bytes, bytearray)):
            key_bytes = bytes(key_env)
        else:
            key_bytes = None

        self.key_derivation_salt = (os.getenv("KEY_DERIVATION_SALT") or secrets.token_hex(16)).encode()

        if key_bytes:
            try:
                self.fernet = Fernet(key_bytes)
                self.master_key = key_bytes
            except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
                # All Live Data, No Fallback/Hardcoded Data
                pp = passphrase or os.getenv("MASTER_ENCRYPTION_PASSPHRASE")
                if not pp:
                    msg = "MASTER_ENCRYPTION_PASSPHRASE environment variable is required - no fallback/hardcoded passphrase"
                    raise RuntimeError(msg) from e
                derived = self.derive_key(pp, self.key_derivation_salt)
                self.fernet = Fernet(derived)
                self.master_key = derived
        else:
            pp = passphrase or os.getenv("MASTER_ENCRYPTION_PASSPHRASE")
            if pp:
                derived = self.derive_key(pp, self.key_derivation_salt)
                self.fernet = Fernet(derived)
                self.master_key = derived
            else:
                gen = Fernet.generate_key()
                self.fernet = Fernet(gen)
                self.master_key = gen

    def encrypt_data(self, data: str) -> str:
        enc = self.fernet.encrypt(data.encode())
        return base64.b64encode(enc).decode()

    def decrypt_data(self, encrypted_data: str) -> str:
        enc_bytes = base64.b64decode(encrypted_data.encode())
        dec = self.fernet.decrypt(enc_bytes)
        return dec.decode()

    def derive_key(self, password: str, salt: bytes | None = None) -> bytes:
        salt = salt or self.key_derivation_salt
        # PBKDF2HMAC signature changed across cryptography versions; handle both possibilities
        try:
            if default_backend is not None:
                kdf = PBKDF2HMAC(algorithm=hashes.SHA256(), length=32, salt=salt, iterations=200_000, backend=default_backend())
            else:
                kdf = PBKDF2HMAC(algorithm=hashes.SHA256(), length=32, salt=salt, iterations=200_000)
        except TypeError:
            # fallback if keywords/parameters differ
            kdf = PBKDF2HMAC(hashes.SHA256(), 32, salt, 200000, default_backend()) if default_backend is not None else PBKDF2HMAC(hashes.SHA256(), 32, salt, 200000)
        return base64.urlsafe_b64encode(kdf.derive(password.encode()))

    def hash_api_key(self, api_key: str) -> str:
        return hashlib.sha256(api_key.encode()).hexdigest()

    def verify_api_key(self, api_key: str, stored_hash: str) -> bool:
        return hmac.compare_digest(hashlib.sha256(api_key.encode()).hexdigest(), stored_hash)


class RateLimiter:
    def __init__(self) -> None:
        self.rate_limits: dict[str, dict[str, Any]] = {}
        self.request_counts: dict[str, list[datetime]] = {}

    def add_rate_limit(self, endpoint: str, max_requests: int, window_seconds: int = 60):
        self.rate_limits[endpoint] = {
            "max_requests": max_requests,
            "window_seconds": window_seconds,
        }

    def is_rate_limited(self, endpoint: str, identifier: str) -> bool:
        cfg = self.rate_limits.get(endpoint)
        if not cfg:
            return False
        key = f"{endpoint}:{identifier}"
        now = datetime.now(timezone.utc)
        window = cfg["window_seconds"]
        cutoff = now - timedelta(seconds=window)
        bucket = self.request_counts.setdefault(key, [])
        bucket[:] = [t for t in bucket if t > cutoff]
        if len(bucket) >= cfg["max_requests"]:
            return True
        bucket.append(now)
        return False

    def get_remaining_requests(self, endpoint: str, identifier: str) -> int:
        cfg = self.rate_limits.get(endpoint)
        if not cfg:
            return 1_000_000_000
        key = f"{endpoint}:{identifier}"
        now = datetime.now(timezone.utc)
        cutoff = now - timedelta(seconds=cfg["window_seconds"])
        bucket = self.request_counts.get(key, [])
        recent = [t for t in bucket if t > cutoff]
        return max(0, int(cfg["max_requests"] - len(recent)))


class AuditLogger:
    def __init__(self, log_file: str = "audit.log") -> None:
        self.log_file = log_file
        self.events: list[SecurityEvent] = []
        self.max_events = 10_000

    def log_event(
        self,
        event_type: str,
        user_id: str | None,
        ip_address: str,
        endpoint: str,
        method: str,
        details: dict[str, Any],
        severity: str = "low",
        status: str = "success",
    ):
        ev = SecurityEvent(
            event_id=secrets.token_hex(8),
            event_type=event_type,
            user_id=user_id,
            ip_address=ip_address,
            endpoint=endpoint,
            method=method,
            timestamp=datetime.now(timezone.utc),
            details=details,
            severity=severity,
            status=status,
        )
        self.events.append(ev)
        if len(self.events) > self.max_events:
            self.events.pop(0)
        self._write_to_file(ev)
        logger.info(
            "security_event",
            event_id=ev.event_id,
            type=ev.event_type,
            user_id=ev.user_id,
            ip=ev.ip_address,
            endpoint=ev.endpoint,
            severity=ev.severity,
            status=ev.status,
        )

    def _write_to_file(self, event: SecurityEvent):
        try:
            log_file_path = Path(self.log_file)
            log_file_path.parent.mkdir(parents=True, exist_ok=True)
            with log_file_path.open("a", encoding="utf-8") as f:
                f.write(
                    json.dumps(
                        {
                            "event_id": event.event_id,
                            "event_type": event.event_type,
                            "user_id": event.user_id,
                            "ip_address": event.ip_address,
                            "endpoint": event.endpoint,
                            "method": event.method,
                            "timestamp": event.timestamp.isoformat(),
                            "details": event.details,
                            "severity": event.severity,
                            "status": event.status,
                        }
                    )
                    + "\n"
                )
        except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
            logger.exception("audit_write_failed", error=str(e))

    def get_events(
        self,
        event_type: str | None = None,
        user_id: str | None = None,
        severity: str | None = None,
        start_time: datetime | None = None,
        end_time: datetime | None = None,
    ) -> list[SecurityEvent]:
        evs = self.events
        if event_type:
            evs = [e for e in evs if e.event_type == event_type]
        if user_id:
            evs = [e for e in evs if e.user_id == user_id]
        if severity:
            evs = [e for e in evs if e.severity == severity]
        if start_time:
            evs = [e for e in evs if e.timestamp >= start_time]
        if end_time:
            evs = [e for e in evs if e.timestamp <= end_time]
        return evs

    def get_security_summary(self) -> dict[str, Any]:
        now = datetime.now(timezone.utc)
        last_24h = now - timedelta(hours=24)
        recent = [e for e in self.events if e.timestamp >= last_24h]
        return {
            "total_events_24h": len(recent),
            "failed_attempts_24h": len([e for e in recent if e.status == "failure"]),
            "blocked_attempts_24h": len([e for e in recent if e.status == "blocked"]),
            "high_severity_events_24h": len([e for e in recent if e.severity in ("high", "critical")]),
            "unique_ips_24h": len({e.ip_address for e in recent}),
            "unique_users_24h": len({e.user_id for e in recent if e.user_id}),
            "most_active_endpoints": self._most_active_endpoints(recent),
            "security_score": self._security_score(recent),
        }

    def _most_active_endpoints(self, events: list[SecurityEvent]) -> list[dict[str, Any]]:
        counts: dict[str, int] = {}
        for e in events:
            counts[e.endpoint] = counts.get(e.endpoint, 0) + 1
        return [{"endpoint": ep, "count": c} for ep, c in sorted(counts.items(), key=lambda x: x[1], reverse=True)[:10]]

    def _security_score(self, events: list[SecurityEvent]) -> float:
        if not events:
            return 100.0
        n = len(events)
        failed = sum(1 for e in events if e.status == "failure")
        blocked = sum(1 for e in events if e.status == "blocked")
        severe = sum(1 for e in events if e.severity in ("high", "critical"))
        failure_rate = failed / n
        severity_penalty = severe / n
        base = 100.0 * (1.0 - failure_rate) * (1.0 - severity_penalty)
        bonus = min(10.0, blocked * 0.5)
        return float(min(100.0, max(0.0, base + bonus)))


class APIKeyManager:
    def __init__(self, encryption_manager: EncryptionManager) -> None:
        self.encryption_manager = encryption_manager
        self.api_keys: dict[str, APIKey] = {}
        self.user_permissions: dict[str, list[str]] = {}

    def generate_api_key(self, user_id: str, permissions: list[str], expires_in_days: int | None = None) -> str:
        key_id = secrets.token_hex(16)
        api_key_val = secrets.token_urlsafe(32)
        expires_at = None
        if expires_in_days:
            expires_at = datetime.now(timezone.utc) + timedelta(days=expires_in_days)
        obj = APIKey(
            key_id=key_id,
            user_id=user_id,
            key_hash=self.encryption_manager.hash_api_key(api_key_val),
            permissions=permissions,
            created_at=datetime.now(timezone.utc),
            expires_at=expires_at,
            last_used=None,
            is_active=True,
            rate_limit=100,
        )
        self.api_keys[key_id] = obj
        return f"{key_id}.{api_key_val}"

    def validate_api_key(self, api_key: str) -> APIKey | None:
        try:
            key_id, key_val = api_key.split(".", 1)
        except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError):
            return None
        obj = self.api_keys.get(key_id)
        if not obj or not obj.is_active:
            return None
        if obj.expires_at and obj.expires_at < datetime.now(timezone.utc):
            return None
        if not self.encryption_manager.verify_api_key(key_val, obj.key_hash):
            return None
        obj.last_used = datetime.now(timezone.utc)
        return obj

    def has_permission(self, api_key: str, permission: str) -> bool:
        obj = self.validate_api_key(api_key)
        return bool(obj and permission in obj.permissions)

    def revoke_api_key(self, key_id: str):
        if key_id in self.api_keys:
            self.api_keys[key_id].is_active = False

    def get_user_keys(self, user_id: str) -> list[APIKey]:
        return [k for k in self.api_keys.values() if k.user_id == user_id]


class SecurityMiddleware:
    def __init__(self) -> None:
        self.encryption_manager = EncryptionManager()
        self.rate_limiter = RateLimiter()
        self.audit_logger = AuditLogger()
        self.api_key_manager = APIKeyManager(self.encryption_manager)
        self.rate_limiter.add_rate_limit("/api/v1/trading", 60, 60)
        self.rate_limiter.add_rate_limit("/api/v1/portfolio", 120, 60)
        self.rate_limiter.add_rate_limit("/api/v1/market-data", 300, 60)

    def require_api_key(self, required_permissions: list[str] | None = None):
        def decorator(func):
            @wraps(func)
            async def wrapper(*args, **kwargs):
                request = kwargs.get("request")
                if request is None:
                    msg = "Request object not found"
                    raise ValueError(msg)
                api_key = request.headers.get("X-API-Key")
                if not api_key:
                    self.audit_logger.log_event(
                        event_type="api_key_missing",
                        user_id=None,
                        ip_address=request.client.host,
                        endpoint=request.url.path,
                        method=request.method,
                        details={"error": "API key required"},
                        severity="medium",
                        status="failure",
                    )
                    msg = "API key required"
                    raise ValueError(msg)
                obj = self.api_key_manager.validate_api_key(api_key)
                if not obj:
                    self.audit_logger.log_event(
                        event_type="api_key_invalid",
                        user_id=None,
                        ip_address=request.client.host,
                        endpoint=request.url.path,
                        method=request.method,
                        details={"error": "Invalid API key"},
                        severity="medium",
                        status="failure",
                    )
                    msg = "Invalid API key"
                    raise ValueError(msg)
                if required_permissions:
                    for p in required_permissions:
                        if not self.api_key_manager.has_permission(api_key, p):
                            self.audit_logger.log_event(
                                event_type="permission_denied",
                                user_id=obj.user_id,
                                ip_address=request.client.host,
                                endpoint=request.url.path,
                                method=request.method,
                                details={"required_permissions": required_permissions},
                                severity="high",
                                status="failure",
                            )
                            msg = f"Permission denied: {p}"
                            raise ValueError(msg)
                if self.rate_limiter.is_rate_limited(request.url.path, obj.user_id):
                    self.audit_logger.log_event(
                        event_type="rate_limit_exceeded",
                        user_id=obj.user_id,
                        ip_address=request.client.host,
                        endpoint=request.url.path,
                        method=request.method,
                        details={"rate_limit": "exceeded"},
                        severity="medium",
                        status="blocked",
                    )
                    msg = "Rate limit exceeded"
                    raise ValueError(msg)
                self.audit_logger.log_event(
                    event_type="api_access",
                    user_id=obj.user_id,
                    ip_address=request.client.host,
                    endpoint=request.url.path,
                    method=request.method,
                    details={"permissions": obj.permissions},
                    severity="low",
                    status="success",
                )
                return await func(*args, **kwargs)

            return wrapper

        return decorator

    def encrypt_sensitive_data(self, data: dict[str, Any]) -> dict[str, Any]:
        out = dict(data)
        for field in ("api_key", "password", "secret", "private_key"):
            if field in out and out[field] is not None:
                out[field] = self.encryption_manager.encrypt_data(str(out[field]))
        return out

    def get_security_status(self) -> dict[str, Any]:
        return {
            "encryption": {
                "master_key_configured": bool(self.encryption_manager.master_key),
                "key_derivation_salt_configured": bool(self.encryption_manager.key_derivation_salt),
            },
            "rate_limiting": {
                "endpoints_protected": len(self.rate_limiter.rate_limits),
                "active_limits": list(self.rate_limiter.rate_limits.keys()),
            },
            "audit_logging": {
                "total_events": len(self.audit_logger.events),
                "log_file": self.audit_logger.log_file,
            },
            "api_keys": {
                "total_keys": len(self.api_key_manager.api_keys),
                "active_keys": len([k for k in self.api_key_manager.api_keys.values() if k.is_active]),
            },
            "security_summary": self.audit_logger.get_security_summary(),
        }


security_middleware = SecurityMiddleware()
