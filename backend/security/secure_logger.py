"""
Secure Logger for Mystic Trading Platform (single file)

Features:
- Sensitive data filtering/redaction (messages + kwargs + dicts)
- Optional log-line encryption (Fernet) with rotation & retention
- Security event monitoring (keywords & suspicious patterns)
- Audit trail logger (access/auth/data/security) with rotation
- Ready-to-use helper methods (info/warn/error/security/access/auth)
- Graceful fallbacks if optional deps (cryptography) are missing

Environment overrides (optional):
- SECURE_LOG_DIR                default: "logs"
- SECURE_LOG_ENCRYPTION         default: "1" (enable) | "0" (disable)
- SECURE_LOG_KEY                default: (auto) Fernet key; if missing,
  encryption uses an in-memory key
- SECURE_LOG_ROTATE_MAX_BYTES   default: 10_485_760  (10MB)
- SECURE_LOG_ROTATE_BACKUPS     default: 5
- SECURE_LOG_RETENTION_DAYS     default: 30

Notes:
- If cryptography is not installed and encryption is enabled, logging
  falls back to plaintext with a warning.
- Encrypted lines are Fernet tokens (base64 text), one token per line.
"""

from __future__ import annotations

import json
import logging
import logging.handlers
import os
import re
import threading
import time
from collections import defaultdict, deque
from pathlib import Path
from typing import Any

# -----------------------------------------------------------------------------
# Optional crypto for encryption
# -----------------------------------------------------------------------------
try:
    from cryptography.fernet import Fernet  # type: ignore[import-not-found]
except (ImportError, ModuleNotFoundError, AttributeError):
    Fernet = None  # type: ignore[assignment]

# -----------------------------------------------------------------------------
# Module logger
# -----------------------------------------------------------------------------
logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------------
# Defaults & config
# -----------------------------------------------------------------------------
DEFAULT_LOG_DIR = os.getenv("SECURE_LOG_DIR")
LOG_ENCRYPTION_ENABLED = os.getenv("SECURE_LOG_ENCRYPTION", "1") == "1"
LOG_RETENTION_DAYS_STR = os.getenv("SECURE_LOG_RETENTION_DAYS")
ROTATE_MAX_BYTES_STR = os.getenv("SECURE_LOG_ROTATE_MAX_BYTES")
ROTATE_BACKUPS_STR = os.getenv("SECURE_LOG_ROTATE_BACKUPS")

# Parse environment variables with validation
if LOG_RETENTION_DAYS_STR:
    try:
        LOG_RETENTION_DAYS = int(LOG_RETENTION_DAYS_STR)
    except (ValueError, TypeError):
        LOG_RETENTION_DAYS = None
else:
    LOG_RETENTION_DAYS = None

if ROTATE_MAX_BYTES_STR:
    try:
        ROTATE_MAX_BYTES = int(ROTATE_MAX_BYTES_STR)
    except (ValueError, TypeError):
        ROTATE_MAX_BYTES = None
else:
    ROTATE_MAX_BYTES = None

if ROTATE_BACKUPS_STR:
    try:
        ROTATE_BACKUPS = int(ROTATE_BACKUPS_STR)
    except (ValueError, TypeError):
        ROTATE_BACKUPS = None
else:
    ROTATE_BACKUPS = None

SECURITY_LOG_LEVEL = logging.WARNING
AUDIT_LOG_LEVEL = logging.INFO

SENSITIVE_FIELDS = [
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


# -----------------------------------------------------------------------------
# Sanitizer
# -----------------------------------------------------------------------------
class LogSanitizer:
    """Sanitizes log messages and dict payloads to remove sensitive
    information."""

    def __init__(self) -> None:
        self.sensitive_patterns = SENSITIVE_FIELDS
        self.replacements = {
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
        self.sensitive_regex = re.compile(
            r"\b(" + "|".join(map(re.escape, self.sensitive_patterns)) + r")\b",
            re.IGNORECASE,
        )

    def sanitize_message(self, message: str) -> str:
        """Sanitize message string."""
        if not message:
            return message

        sanitized = message

        # Replace exact keywords with placeholders (case-insensitive)
        for pat, repl in self.replacements.items():
            sanitized = re.sub(
                rf"\b{re.escape(pat)}\b",
                repl,
                sanitized,
                flags=re.IGNORECASE,
            )

        # Credit card numbers
        sanitized = re.sub(
            r"\b\d{4}[- ]?\d{4}[- ]?\d{4}[- ]?\d{4}\b",
            "***CREDIT_CARD***",
            sanitized,
        )

        # Social security numbers
        sanitized = re.sub(r"\b\d{3}-\d{2}-\d{4}\b", "***SSN***", sanitized)

        # Long tokens / API keys
        sanitized = re.sub(r"\b[a-zA-Z0-9_-]{32,}\b", "***TOKEN***", sanitized)

        # Email addresses - keep domain
        sanitized = re.sub(
            r"\b([a-zA-Z0-9._%+-]+)@([a-zA-Z0-9.-]+\.[A-Za-z]{2,})\b",
            r"***@\2",
            sanitized,
        )

        # IP addresses - keep first octet
        return re.sub(
            r"\b(\d{1,3})\.(\d{1,3})\.(\d{1,3})\.(\d{1,3})\b",
            r"\1.***.***.***",
            sanitized,
        )

    def sanitize_dict(self, data: dict[str, Any]) -> dict[str, Any]:
        """Sanitize dictionary data."""
        if not isinstance(data, dict):
            return data  # type: ignore[return-value]

        out: dict[str, Any] = {}
        for k, v in data.items():
            if isinstance(v, dict):
                out[k] = self.sanitize_dict(v)
            elif isinstance(v, list):
                out[k] = [self.sanitize_dict(x) if isinstance(x, dict) else self.sanitize_message(str(x)) for x in v]
            elif isinstance(v, str):
                if any(p in k.lower() for p in self.sensitive_patterns):
                    out[k] = "***SENSITIVE***"
                else:
                    out[k] = self.sanitize_message(v)
            else:
                out[k] = v
        return out


# -----------------------------------------------------------------------------
# Security Log Filter
# -----------------------------------------------------------------------------
class SecurityLogFilter(logging.Filter):
    """Captures security-relevant records and suspicious patterns."""

    def __init__(self, name: str = "") -> None:
        super().__init__(name)
        self.security_events: deque[dict[str, Any]] = deque(maxlen=1000)
        self.suspicious_patterns: dict[str, int] = defaultdict(int)
        self._lock = threading.Lock()

    def filter(self, record: logging.LogRecord) -> bool:
        """Filter log records for security events."""
        msg = str(record.getMessage())
        # Keywords that flag security interest
        keywords = [
            "authentication",
            "authorization",
            "login",
            "logout",
            "password",
            "token",
            "session",
            "access",
            "permission",
            "security",
            "audit",
            "mfa",
            "2fa",
        ]
        if any(k in msg.lower() for k in keywords):
            self._record_security_event(record)

        # Suspicious markers
        if self._is_suspicious(msg):
            self._record_suspicious(record)

        return True

    def _record_security_event(self, record: logging.LogRecord) -> None:
        """Record security event."""
        with self._lock:
            self.security_events.append(
                {
                    "ts": time.time(),
                    "level": record.levelname,
                    "message": record.getMessage(),
                    "module": record.module,
                    "function": record.funcName,
                },
            )

    @staticmethod
    def _is_suspicious(message: str) -> bool:
        """Check if message contains suspicious patterns."""
        patterns = [
            "failed login",
            "invalid password",
            "unauthorized access",
            "suspicious activity",
            "security violation",
            "multiple attempts",
            "bruteforce",
            "csrf",
            "xss",
        ]
        return any(p in message.lower() for p in patterns)

    def _record_suspicious(self, record: logging.LogRecord) -> None:
        """Record suspicious pattern."""
        with self._lock:
            key = record.getMessage()[:80]
            self.suspicious_patterns[key] += 1

    def get_security_stats(self) -> dict[str, Any]:
        """Get security statistics."""
        with self._lock:
            recent_events = list(self.security_events)[-10:] if self.security_events else []
            return {
                "total_security_events": len(self.security_events),
                "suspicious_patterns": dict(self.suspicious_patterns),
                "recent_events": recent_events,
            }


# -----------------------------------------------------------------------------
# Encrypted Rotating File Handler
# -----------------------------------------------------------------------------
class EncryptedRotatingFileHandler(logging.handlers.RotatingFileHandler):
    """
    Rotating file handler that optionally encrypts each formatted log line
    using Fernet.

    - Writes Fernet tokens as base64 text (one token per line) when enabled
      and key is available.
    - If encryption disabled/unavailable, writes plaintext.
    """

    def __init__(
        self,
        filename: str,
        maxBytes: int | None = None,
        backupCount: int | None = None,
        encrypt: bool = LOG_ENCRYPTION_ENABLED,
        fernet: Fernet | None = None,  # type: ignore[name-defined]
        **kwargs: Any,
    ) -> None:
        max_bytes_val = maxBytes if maxBytes is not None else (ROTATE_MAX_BYTES if ROTATE_MAX_BYTES else 5242880)
        backup_count_val = backupCount if backupCount is not None else (ROTATE_BACKUPS if ROTATE_BACKUPS else 2)
        super().__init__(
            filename,
            maxBytes=max_bytes_val,
            backupCount=backup_count_val,
            **kwargs,
        )
        self._encrypt = bool(encrypt and fernet is not None)
        self._fernet = fernet
        if encrypt and fernet is None:
            logging.getLogger(__name__).warning("Log encryption requested, but 'cryptography' is not available; falling back to plaintext.")

        # Serialize writes across threads/processes
        self._emit_lock = threading.Lock()

    def emit(self, record: logging.LogRecord) -> None:
        """Emit log record."""
        try:
            msg = self.format(record)
            line = self._encrypt_line(msg) if self._encrypt else (msg + self.terminator)

            with self._emit_lock:
                if self.shouldRollover(record):
                    self.doRollover()
                # Write as text; encrypted lines are base64 strings
                self.stream.write(line)
                self.flush()
        except Exception:
            self.handleError(record)

    def _encrypt_line(self, text: str) -> str:
        """Encrypt log line."""
        try:
            assert self._fernet is not None
            token = self._fernet.encrypt(text.encode("utf-8"))
            return token.decode("utf-8") + self.terminator
        except Exception:
            # Fallback to plaintext if encryption fails for a line
            return text + self.terminator


# -----------------------------------------------------------------------------
# Audit Logger
# -----------------------------------------------------------------------------
class AuditLogger:
    """Audit logger writing structured compliance events to
    logs/audit.log"""

    def __init__(
        self,
        log_dir: str | None = None,
        rotate_bytes: int | None = None,
        backups: int | None = None,
    ) -> None:
        log_dir_val = log_dir if log_dir else (DEFAULT_LOG_DIR if DEFAULT_LOG_DIR else "logs")
        rotate_bytes_val = rotate_bytes if rotate_bytes is not None else (ROTATE_MAX_BYTES if ROTATE_MAX_BYTES else 5242880)
        backups_val = backups if backups is not None else (ROTATE_BACKUPS if ROTATE_BACKUPS else 2)
        Path(log_dir_val).mkdir(parents=True, exist_ok=True)
        path = str(Path(log_dir_val) / "audit.log")
        self._handler = logging.handlers.RotatingFileHandler(path, maxBytes=rotate_bytes_val, backupCount=backups_val)
        self._handler.setLevel(AUDIT_LOG_LEVEL)
        self._handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
        self._log = logging.getLogger("audit")
        self._log.setLevel(AUDIT_LOG_LEVEL)
        handler_check = any(isinstance(h, logging.handlers.RotatingFileHandler) and getattr(h, "baseFilename", "") == self._handler.baseFilename for h in self._log.handlers)
        if not handler_check:
            self._log.addHandler(self._handler)

        # In-memory ring buffers
        self.audit_events: deque[dict[str, Any]] = deque(maxlen=10000)
        self.access_logs: deque[dict[str, Any]] = deque(maxlen=5000)
        self.auth_logs: deque[dict[str, Any]] = deque(maxlen=5000)
        self.data_access_logs: deque[dict[str, Any]] = deque(maxlen=5000)
        self.security_events: deque[dict[str, Any]] = deque(maxlen=2000)
        self._lock = threading.Lock()

    def _store(self, store: deque[dict[str, Any]], event: dict[str, Any]) -> None:
        """Store event in deque."""
        with self._lock:
            store.append(event)
            self.audit_events.append(event)

    def log_access(
        self,
        user_id: str,
        action: str,
        resource: str,
        success: bool,
        client_ip: str | None = None,
    ) -> None:
        """Log access event."""
        ev = {
            "ts": time.time(),
            "type": "access",
            "user_id": user_id,
            "action": action,
            "resource": resource,
            "success": success,
            "ip": client_ip,
        }
        self._store(self.access_logs, ev)
        ip_str = client_ip if client_ip else "-"
        self._log.info(
            "ACCESS %s %s %s %s IP=%s",
            user_id,
            action,
            resource,
            "SUCCESS" if success else "FAILED",
            ip_str,
        )

    def log_authentication(
        self,
        user_id: str,
        method: str,
        success: bool,
        client_ip: str | None = None,
    ) -> None:
        """Log authentication event."""
        ev = {
            "ts": time.time(),
            "type": "authentication",
            "user_id": user_id,
            "method": method,
            "success": success,
            "ip": client_ip,
        }
        self._store(self.auth_logs, ev)
        ip_str = client_ip if client_ip else "-"
        self._log.info(
            "AUTH %s %s %s IP=%s",
            user_id,
            method,
            "SUCCESS" if success else "FAILED",
            ip_str,
        )

    def log_data_access(
        self,
        user_id: str,
        data_type: str,
        operation: str,
        success: bool,
        record_count: int | None = None,
    ) -> None:
        """Log data access event."""
        ev = {
            "ts": time.time(),
            "type": "data_access",
            "user_id": user_id,
            "data_type": data_type,
            "operation": operation,
            "success": success,
            "records": record_count,
        }
        self._store(self.data_access_logs, ev)
        count_str = str(record_count) if record_count is not None else "-"
        self._log.info(
            "DATA_ACCESS %s %s %s %s records=%s",
            user_id,
            operation,
            data_type,
            "SUCCESS" if success else "FAILED",
            count_str,
        )

    def log_security_event(
        self,
        event_type: str,
        severity: str,
        description: str,
        user_id: str | None = None,
        client_ip: str | None = None,
    ) -> None:
        """Log security event."""
        ev = {
            "ts": time.time(),
            "type": "security",
            "event": event_type,
            "severity": severity,
            "desc": description,
            "user_id": user_id,
            "ip": client_ip,
        }
        self._store(self.security_events, ev)
        user_str = user_id if user_id else "-"
        ip_str = client_ip if client_ip else "-"
        self._log.warning(
            "SECURITY %s %s %s user=%s ip=%s",
            event_type,
            severity,
            description,
            user_str,
            ip_str,
        )

    def get_audit_stats(self) -> dict[str, Any]:
        """Get audit statistics."""
        with self._lock:
            recent_events = list(self.audit_events)[-20:] if self.audit_events else []
            return {
                "total_audit_events": len(self.audit_events),
                "access_logs": len(self.access_logs),
                "auth_logs": len(self.auth_logs),
                "data_access_logs": len(self.data_access_logs),
                "security_events": len(self.security_events),
                "recent_events": recent_events,
            }


# -----------------------------------------------------------------------------
# Secure Logger
# -----------------------------------------------------------------------------
class SecureLogger:
    """High-level secure logger: sanitization, encryption, rotation, audit
    & stats."""

    def __init__(self) -> None:
        self.sanitizer = LogSanitizer()
        self.security_filter = SecurityLogFilter()
        log_dir_val = DEFAULT_LOG_DIR if DEFAULT_LOG_DIR else "logs"
        rotate_bytes_val = ROTATE_MAX_BYTES if ROTATE_MAX_BYTES else 5242880
        backups_val = ROTATE_BACKUPS if ROTATE_BACKUPS else 2
        self.audit_logger = AuditLogger(log_dir_val, rotate_bytes_val, backups_val)
        self._encrypt_enabled = LOG_ENCRYPTION_ENABLED
        self._fernet = self._init_fernet() if self._encrypt_enabled else None

        # One-time setup
        self._setup_logging()

    # ---- Crypto ------------------------------------------------------------
    def _init_fernet(self) -> Fernet | None:  # type: ignore[name-defined]
        """Initialize Fernet encryption."""
        if Fernet is None:
            return None
        key = os.getenv("SECURE_LOG_KEY")
        if key:
            try:
                return Fernet(key.encode("utf-8"))
            except Exception:
                logger.warning("SECURE_LOG_KEY invalid; generating ephemeral key.")
        # Generate ephemeral key (not persisted)
        try:
            fk = Fernet.generate_key()
            logger.info("Generated ephemeral Fernet key for log encryption (not persisted).")
            return Fernet(fk)
        except Exception as e:  # pragma: no cover
            logger.warning(
                "Failed to create Fernet key; disabling encryption: %s",
                e,
            )
            return None

    def set_encryption_key(self, key: str) -> bool:
        """Set/replace encryption key at runtime. Returns True on success."""
        if Fernet is None:
            logger.warning("Cannot set encryption key: cryptography not installed.")
            return False
        try:
            self._fernet = Fernet(key.encode("utf-8"))
            self._encrypt_enabled = True
            logger.info("SecureLogger: encryption key set.")
        except Exception as e:
            logger.exception("Invalid encryption key: %s", e)
            return False
        else:
            return True

    # ---- Setup ------------------------------------------------------------
    def _setup_logging(self) -> None:
        """Setup logging handlers and configuration."""
        # Attach security filter to root
        root = logging.getLogger()
        root.addFilter(self.security_filter)

        # Ensure log dir
        log_dir_val = DEFAULT_LOG_DIR if DEFAULT_LOG_DIR else "logs"
        Path(log_dir_val).mkdir(parents=True, exist_ok=True)

        # Handlers: encrypted file + console
        file_path = str(Path(log_dir_val) / "secure.log")
        rotate_bytes_val = ROTATE_MAX_BYTES if ROTATE_MAX_BYTES else 5242880
        backups_val = ROTATE_BACKUPS if ROTATE_BACKUPS else 2
        enc_handler = EncryptedRotatingFileHandler(
            filename=file_path,
            maxBytes=rotate_bytes_val,
            backupCount=backups_val,
            encrypt=self._encrypt_enabled,
            fernet=self._fernet,
        )
        enc_handler.setLevel(logging.INFO)
        enc_handler.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))

        console = logging.StreamHandler()
        console.setLevel(logging.INFO)
        console.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))

        # Install handlers if not already present (avoid duplication across
        # reloads)
        def _has_handler(t: type) -> bool:
            return any(isinstance(h, t) for h in root.handlers)

        if not _has_handler(EncryptedRotatingFileHandler):
            root.addHandler(enc_handler)
        if not _has_handler(logging.StreamHandler):
            root.addHandler(console)

        # Default root level
        if root.level == logging.WARNING:
            root.setLevel(logging.INFO)

    # ---- Redaction helpers ------------------------------------------------
    def _build_log_message(self, message: str, kwargs: dict[str, Any]) -> str:
        """Build sanitized log message."""
        smsg = self.sanitizer.sanitize_message(message)
        skw = self.sanitizer.sanitize_dict(kwargs) if kwargs else {}
        return f"{smsg} - {json.dumps(skw, separators=(',', ':'))}" if skw else smsg

    # ---- Public API -------------------------------------------------------
    def log_info(self, message: str, **kwargs: Any) -> None:
        """Log info message."""
        logging.getLogger().info(self._build_log_message(message, kwargs))

    def log_warning(self, message: str, **kwargs: Any) -> None:
        """Log warning message."""
        logging.getLogger().warning(self._build_log_message(message, kwargs))

    def log_error(
        self,
        message: str,
        error: Exception | None = None,
        **kwargs: Any,
    ) -> None:
        """Log error message."""
        smsg = self.sanitizer.sanitize_message(message)
        skw = self.sanitizer.sanitize_dict(kwargs) if kwargs else {}
        if error:
            emsg = self.sanitizer.sanitize_message(str(error))
            base = f"{smsg} - Error: {emsg}"
        else:
            base = smsg
        suffix = f" - {json.dumps(skw, separators=(',', ':'))}" if skw else ""
        logging.getLogger().error(base + suffix)

    def log_security(
        self,
        message: str,
        severity: str = "medium",
        **kwargs: Any,
    ) -> None:
        """Log security event."""
        smsg = self.sanitizer.sanitize_message(message)
        skw = self.sanitizer.sanitize_dict(kwargs) if kwargs else {}
        suffix = f" - {json.dumps(skw, separators=(',', ':'))}" if skw else ""
        formatted = f"[SECURITY-{severity.upper()}] {smsg}" + suffix
        logging.getLogger().log(SECURITY_LOG_LEVEL, formatted)
        # Also write to audit trail (only pass safe fields)
        user_id = None
        client_ip = None
        if isinstance(skw, dict):
            user_id = skw.get("user_id")
            client_ip = skw.get("client_ip")
        self.audit_logger.log_security_event(
            event_type="security_log",
            severity=severity,
            description=smsg,
            user_id=user_id,
            client_ip=client_ip,
        )

    def log_access_attempt(
        self,
        user_id: str,
        action: str,
        resource: str,
        success: bool,
        client_ip: str | None = None,
    ) -> None:
        """Log access attempt."""
        self.audit_logger.log_access(user_id, action, resource, success, client_ip)
        status = "SUCCESS" if success else "FAILED"
        ip_str = client_ip if client_ip else "-"
        logging.getLogger().info(
            self._build_log_message(
                "Access attempt",
                {
                    "user_id": user_id,
                    "action": action,
                    "resource": resource,
                    "status": status,
                    "client_ip": ip_str,
                },
            )
        )

    def log_authentication_attempt(
        self,
        user_id: str,
        method: str,
        success: bool,
        client_ip: str | None = None,
    ) -> None:
        """Log authentication attempt."""
        self.audit_logger.log_authentication(user_id, method, success, client_ip)
        status = "SUCCESS" if success else "FAILED"
        ip_str = client_ip if client_ip else "-"
        logging.getLogger().info(
            self._build_log_message(
                "Authentication attempt",
                {
                    "user_id": user_id,
                    "method": method,
                    "status": status,
                    "client_ip": ip_str,
                },
            )
        )

    # ---- Stats & Maintenance ----------------------------------------------
    def get_logging_stats(self) -> dict[str, Any]:
        """Get logging statistics."""
        log_dir_val = DEFAULT_LOG_DIR if DEFAULT_LOG_DIR else "logs"
        rotate_bytes_val = ROTATE_MAX_BYTES if ROTATE_MAX_BYTES else 5242880
        backups_val = ROTATE_BACKUPS if ROTATE_BACKUPS else 2
        retention_days_val = LOG_RETENTION_DAYS if LOG_RETENTION_DAYS else 30
        return {
            "security_stats": self.security_filter.get_security_stats(),
            "audit_stats": self.audit_logger.get_audit_stats(),
            "log_encryption_enabled": bool(self._encrypt_enabled and self._fernet is not None),
            "log_dir": log_dir_val,
            "rotation": {
                "max_bytes": rotate_bytes_val,
                "backups": backups_val,
            },
            "retention_days": retention_days_val,
        }

    def cleanup_old_logs(self, max_age_days: int | None = None) -> None:
        """Clean up old log files."""
        age_days = max_age_days if max_age_days is not None else (LOG_RETENTION_DAYS if LOG_RETENTION_DAYS else 30)
        cutoff = time.time() - (age_days * 86400)
        log_dir_val = DEFAULT_LOG_DIR if DEFAULT_LOG_DIR else "logs"
        if Path(log_dir_val).exists():
            log_path = Path(log_dir_val)
            for path in log_path.iterdir():
                if path.is_file():
                    try:
                        if path.stat().st_mtime < cutoff:
                            path.unlink()
                            logger.info("Removed old log file: %s", path.name)
                    except Exception:
                        continue
        logger.info("Cleaned up old log files")

    # ---- Uvicorn/Starlette helpers (optional) ---------------------------
    def install_uvicorn_intercept(self) -> None:
        """
        Route uvicorn access/error logs through our handlers/filters
        (optional).

        Call once during app startup, if using FastAPI/Uvicorn.
        """
        for ln in ("uvicorn", "uvicorn.error", "uvicorn.access"):
            lg = logging.getLogger(ln)
            lg.setLevel(logging.INFO)
            for h in logging.getLogger().handlers:
                h_base = getattr(h, "baseFilename", None)
                handler_check = not any(getattr(h2, "baseFilename", None) == h_base for h2 in lg.handlers)
                if handler_check:
                    lg.addHandler(h)  # mirror root handlers
            lg.addFilter(self.security_filter)


# -----------------------------------------------------------------------------
# Global instance
# -----------------------------------------------------------------------------
secure_logger = SecureLogger()
