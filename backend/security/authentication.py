"""
API Authentication System for Mystic Trading Platform (single file)

Features:
- JWT token mint/verify with proper expiration handling
- Role-based access control (RBAC)
- API key generation + storage (hashed) in Redis (if available)
- Session management (Redis-backed when available)
- Basic security monitoring (failed logins, lockouts)

Notes:
- No fake or placeholder data is emitted. If a backing service (e.g., Redis)
  is not available, the module returns None/empty.
- Password hashing uses SHA-256 here to avoid extra deps; consider
  bcrypt/argon2 in production.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import logging
import secrets
import threading
import time
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any

import jwt

from backend.config import settings

# Direct imports for production
from backend.config.redis_config import get_redis_client
from redis import Redis  # type: ignore[import-not-found]

logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------------
# Config
# -----------------------------------------------------------------------------
JWT_SECRET_KEY = settings.security.secret_key
JWT_ALGORITHM = settings.security.algorithm
# Use minutes as configured; DON'T integer-divide to hours (would become 0!)
expire_minutes = getattr(settings.security, "access_token_expire_minutes", None)
if expire_minutes is None:
    expire_minutes = 60
JWT_EXPIRATION_MINUTES = int(expire_minutes)

API_KEY_LENGTH_BYTES = 32  # token_urlsafe() uses bytes -> ~43 chars
SESSION_TIMEOUT = 3600  # seconds
MAX_LOGIN_ATTEMPTS = 5
ACCOUNT_LOCKOUT_DURATION = 900  # seconds


# -----------------------------------------------------------------------------
# Models
# -----------------------------------------------------------------------------
@dataclass
class User:
    """User model."""

    user_id: str
    username: str
    email: str
    roles: list[str]
    permissions: list[str]
    is_active: bool = True
    created_at: float = 0.0
    last_login: float = 0.0


@dataclass
class APIToken:
    """API token model."""

    token_id: str
    user_id: str
    token_hash: str
    permissions: list[str]
    is_active: bool = True
    created_at: float = 0.0
    last_used: float = 0.0


# -----------------------------------------------------------------------------
# Token + API key management
# -----------------------------------------------------------------------------
class TokenManager:
    """Manages JWT tokens and API keys."""

    def __init__(self) -> None:
        self.secret_key = JWT_SECRET_KEY
        self.algorithm = JWT_ALGORITHM
        self.exp_delta = timedelta(minutes=JWT_EXPIRATION_MINUTES)
        self.blacklisted_tokens: set[str] = set()
        self._lock = threading.Lock()

    def generate_jwt_token(self, user_id: str, roles: list[str], permissions: list[str]) -> str:
        """Generate JWT token."""
        now = datetime.now(timezone.utc)
        payload: dict[str, Any] = {
            "sub": user_id,
            "user_id": user_id,
            "roles": roles,
            "permissions": permissions,
            "iat": int(now.timestamp()),
            "exp": int((now + self.exp_delta).timestamp()),
            "jti": secrets.token_hex(12),
        }
        return jwt.encode(payload, self.secret_key, algorithm=self.algorithm)

    def verify_jwt_token(self, token: str) -> dict[str, Any] | None:
        """Verify JWT and return payload dict, or None if invalid/expired/
        blacklisted."""
        try:
            with self._lock:
                if token in self.blacklisted_tokens:
                    return None

            return jwt.decode(
                token,
                self.secret_key,
                algorithms=[self.algorithm],
                options={"require": ["exp", "iat"]},
            )
        except jwt.ExpiredSignatureError:
            return None
        except jwt.InvalidTokenError:
            return None
        except Exception as e:
            logger.exception(f"Token verification error: {e}")
            return None

    def blacklist_token(self, token: str) -> None:
        """Blacklist a token."""
        with self._lock:
            self.blacklisted_tokens.add(token)

    def generate_api_key(self) -> str:
        """Generate API key."""
        # ~43-char URL-safe string
        return secrets.token_urlsafe(API_KEY_LENGTH_BYTES)

    @staticmethod
    def hash_api_key(api_key: str) -> str:
        """Hash API key."""
        return hashlib.sha256(api_key.encode("utf-8")).hexdigest()


# -----------------------------------------------------------------------------
# RBAC
# -----------------------------------------------------------------------------
class RoleBasedAccessControl:
    """Role-based access control system."""

    def __init__(self) -> None:
        self.roles_permissions: dict[str, list[str]] = {
            "admin": [
                "read:all",
                "write:all",
                "delete:all",
                "admin:all",
                "trading:all",
                "analytics:all",
                "system:all",
            ],
            "trader": [
                "read:trading",
                "write:trading",
                "read:portfolio",
                "write:portfolio",
                "read:analytics",
            ],
            "analyst": [
                "read:analytics",
                "write:analytics",
                "read:market_data",
                "read:reports",
            ],
            "viewer": [
                "read:portfolio",
                "read:market_data",
                "read:reports",
            ],
            "api_user": [
                "read:market_data",
                "read:portfolio",
                "read:analytics",
            ],
        }

        self.resource_permissions: dict[str, list[str]] = {
            "/api/market-data": ["read:market_data"],
            "/api/trading": ["write:trading"],
            "/api/portfolio": [
                "read:portfolio",
                "write:portfolio",
            ],
            "/api/analytics": [
                "read:analytics",
                "write:analytics",
            ],
            "/api/admin": ["admin:all"],
            "/api/system": ["system:all"],
        }

    @staticmethod
    def check_permission(user_permissions: list[str], required_permissions: list[str]) -> bool:
        """Check if user has required permissions."""
        if not user_permissions or not required_permissions:
            return False
        return any(perm in user_permissions for perm in required_permissions)

    def get_user_permissions(self, roles: list[str]) -> list[str]:
        """Get permissions for roles."""
        permissions: list[str] = []
        for role in roles:
            role_perms = self.roles_permissions.get(role)
            if isinstance(role_perms, list):
                permissions.extend(role_perms)
        return sorted(set(permissions))

    def get_resource_permissions(self, resource: str) -> list[str]:
        """Get permissions required for resource."""
        return self.resource_permissions.get(resource, [])

    def add_role_permissions(self, role: str, permissions: list[str]) -> None:
        """Add permissions to role."""
        if role not in self.roles_permissions:
            self.roles_permissions[role] = []
        self.roles_permissions[role].extend(permissions)
        self.roles_permissions[role] = sorted(set(self.roles_permissions[role]))


# -----------------------------------------------------------------------------
# Authentication Manager
# -----------------------------------------------------------------------------
class AuthenticationManager:
    """Main authentication manager; Redis-backed when available."""

    def __init__(self) -> None:
        self.token_manager = TokenManager()
        self.rbac = RoleBasedAccessControl()
        self.redis_client: Redis | None = None  # type: ignore[type-arg]
        self.failed_attempts: dict[str, int] = defaultdict(int)
        self.lockout_times: dict[str, float] = defaultdict(float)
        self._init_redis()

    # ---------------------------- Redis init ---------------------------------
    def _init_redis(self) -> None:
        """Initialize Redis connection via shared pool."""
        try:
            # Use shared Redis pool for synchronous client
            self.redis_client = get_redis_client()
            # Connection check
            self.redis_client.ping()
            logger.info("Redis connection established via shared pool")
        except Exception as e:
            logger.exception(f"Redis connection failed: {e}")
            self.redis_client = None

    # ---------------------------- Auth Core -------------------------
    def authenticate_user(
        self,
        username: str,
        password: str,
        client_ip: str | None = None,
    ) -> dict[str, Any] | None:
        """Username/password authentication. Returns token bundle or None."""
        try:
            if self._is_account_locked(username):
                logger.warning(f"Account locked: {username}")
                return None

            if not self.redis_client:
                logger.error("Redis not available for authentication")
                return None

            roles: list[str] = []
            permissions: list[str] = []

            key = f"user:{username}"
            user = self.redis_client.hgetall(key)
            if not user:
                self._record_failed_login(username, client_ip)
                return None

            is_active = user.get("is_active")
            if is_active != "true":
                self._record_failed_login(username, client_ip)
                return None

            stored_hash = user.get("password_hash")
            if not stored_hash:
                self._record_failed_login(username, client_ip)
                return None

            password_hash = hashlib.sha256(password.encode("utf-8")).hexdigest()
            if password_hash != stored_hash:
                self._record_failed_login(username, client_ip)
                return None

            # Roles + permissions are JSON lists
            roles_json = user.get("roles")
            if roles_json:
                try:
                    parsed_roles = json.loads(roles_json)
                    if isinstance(parsed_roles, list):
                        roles = parsed_roles
                except (ValueError, TypeError, json.JSONDecodeError):
                    pass

            permissions_json = user.get("permissions")
            if permissions_json:
                try:
                    parsed_perms = json.loads(permissions_json)
                    if isinstance(parsed_perms, list):
                        permissions = parsed_perms
                except (ValueError, TypeError, json.JSONDecodeError):
                    pass

            if not permissions and roles:
                permissions = self.rbac.get_user_permissions(roles)

            token = self.token_manager.generate_jwt_token(username, roles, permissions)
            self._create_session(
                token,
                {
                    "user_id": username,
                    "roles": roles,
                    "permissions": permissions,
                    "client_ip": client_ip,
                    "created_at": time.time(),
                },
            )
            self._log_auth_ok(username, client_ip)
            out_obj = {
                "token": token,
                "user_id": username,
                "roles": roles,
                "permissions": permissions,
                "expires_in": JWT_EXPIRATION_MINUTES * 60,
            }
        except Exception as e:
            logger.exception(f"Authentication error: {e}")
            return None
        else:
            return out_obj

    def authenticate_api_key(self, api_key: str) -> dict[str, Any] | None:
        """API key auth via Redis (hash lookup)."""
        try:
            if not self.redis_client:
                return None
            digest = self.token_manager.hash_api_key(api_key)
            key = f"api_token:{digest}"
            data = self.redis_client.hgetall(key)
            if not data:
                return None

            is_active = data.get("is_active")
            if is_active != "true":
                return None

            user_id = data.get("user_id")
            if not user_id:
                return None

            permissions: list[str] = []
            permissions_json = data.get("permissions")
            if permissions_json:
                try:
                    parsed_perms = json.loads(permissions_json)
                    if isinstance(parsed_perms, list):
                        permissions = parsed_perms
                except (ValueError, TypeError, json.JSONDecodeError):
                    pass

            # Touch last_used
            self.redis_client.hset(key, "last_used", str(time.time()))

            response: dict[str, Any] = {
                "user_id": user_id,
                "auth_type": "api_key",
            }

            if permissions:
                response["permissions"] = permissions

            out_obj = response
        except Exception as e:
            logger.exception(f"API key authentication error: {e}")
            return None
        else:
            return out_obj

    def verify_token(self, token: str) -> dict[str, Any] | None:
        """Verify JWT and session. Returns claims+session or None."""
        try:
            claims = self.token_manager.verify_jwt_token(token)
            if not claims:
                return None
            sess = self._get_session(token)
            if not sess:
                return None

            user_id = claims.get("user_id")
            if not user_id:
                user_id = claims.get("sub")

            response: dict[str, Any] = {
                "user_id": user_id,
                "session_data": sess,
            }

            roles = claims.get("roles")
            if isinstance(roles, list):
                response["roles"] = roles

            permissions = claims.get("permissions")
            if isinstance(permissions, list):
                response["permissions"] = permissions

            out_obj = response
        except Exception as e:
            logger.exception(f"Token verification error: {e}")
            return None
        else:
            return out_obj

    def check_access(self, user_permissions: list[str], resource: str) -> bool:
        """Check if user has access to resource."""
        required = self.rbac.get_resource_permissions(resource)
        return self.rbac.check_permission(user_permissions, required)

    # ---------------------------- API Keys --------------------------
    def create_api_key(self, user_id: str, permissions: list[str]) -> str:
        """Return a freshly minted API key (plain).

        Hash is stored in Redis if available.
        """
        try:
            if not self.redis_client:
                logger.error("Redis not available for API key creation")
                return ""

            api_key = self.token_manager.generate_api_key()
            digest = self.token_manager.hash_api_key(api_key)
            data = {
                "user_id": user_id,
                "token_hash": digest,
                "permissions": json.dumps(permissions),
                "is_active": "true",
                "created_at": str(time.time()),
                "last_used": str(time.time()),
            }
            for k, v in data.items():
                self.redis_client.hset(f"api_token:{digest}", k, v)
            out_key = api_key
        except Exception as e:
            logger.exception(f"API key creation error: {e}")
            return ""
        else:
            return out_key

    def revoke_api_key(self, token_hash_or_id: str) -> bool:
        """Deactivate API key by hash; returns True if updated."""
        try:
            if not self.redis_client:
                return False
            # Accept either "api_token:<hash>" or just "<hash>"
            key = token_hash_or_id if token_hash_or_id.startswith("api_token:") else f"api_token:{token_hash_or_id}"

            if not self.redis_client.exists(key):
                return False
            self.redis_client.hset(key, "is_active", "false")
            out_bool = True
        except Exception as e:
            logger.exception(f"API key revocation error: {e}")
            return False
        else:
            return out_bool

    # ---------------------------- Sessions --------------------------
    def logout(self, token: str) -> None:
        """Blacklist token + remove session."""
        try:
            self.token_manager.blacklist_token(token)
            self._remove_session(token)
            logger.info("User logged out")
        except Exception as e:
            logger.exception(f"Logout error: {e}")

    def _create_session(self, token: str, data: dict[str, Any]) -> None:
        """Store session in Redis (JSON fields).

        Expires after SESSION_TIMEOUT.
        """
        try:
            if not self.redis_client:
                return

            key = f"session:{token}"
            payload: dict[str, str] = {}

            user_id = data.get("user_id")
            if user_id:
                payload["user_id"] = str(user_id)

            roles = data.get("roles")
            if isinstance(roles, list):
                payload["roles"] = json.dumps(roles)

            permissions = data.get("permissions")
            if isinstance(permissions, list):
                payload["permissions"] = json.dumps(permissions)

            client_ip = data.get("client_ip")
            if client_ip:
                payload["client_ip"] = str(client_ip)

            created_at = data.get("created_at")
            if created_at:
                payload["created_at"] = str(float(created_at))
            else:
                payload["created_at"] = str(time.time())

            for k, v in payload.items():
                self.redis_client.hset(key, k, v)
            self.redis_client.expire(key, SESSION_TIMEOUT)
        except Exception as e:
            logger.exception(f"Session creation error: {e}")

    def _get_session(self, token: str) -> dict[str, Any] | None:
        """Get session from Redis."""
        try:
            if not self.redis_client:
                return None
            key = f"session:{token}"
            raw = self.redis_client.hgetall(key)
            if not raw:
                return None

            # Decode types
            out: dict[str, Any] = {}

            user_id = raw.get("user_id")
            if user_id:
                out["user_id"] = user_id

            client_ip = raw.get("client_ip")
            if client_ip:
                out["client_ip"] = client_ip

            created_at_str = raw.get("created_at")
            if created_at_str:
                with contextlib.suppress(ValueError, TypeError):
                    out["created_at"] = float(created_at_str)

            roles_json = raw.get("roles")
            if roles_json:
                try:
                    parsed_roles = json.loads(roles_json)
                    if isinstance(parsed_roles, list):
                        out["roles"] = parsed_roles
                    else:
                        out["roles"] = []
                except (ValueError, TypeError, json.JSONDecodeError):
                    out["roles"] = []
            else:
                out["roles"] = []

            permissions_json = raw.get("permissions")
            if permissions_json:
                try:
                    parsed_perms = json.loads(permissions_json)
                    if isinstance(parsed_perms, list):
                        out["permissions"] = parsed_perms
                    else:
                        out["permissions"] = []
                except (ValueError, TypeError, json.JSONDecodeError):
                    out["permissions"] = []
            else:
                out["permissions"] = []
        except Exception as e:
            logger.exception(f"Session retrieval error: {e}")
            return None
        else:
            return out

    def _remove_session(self, token: str) -> None:
        """Remove session from Redis."""
        try:
            if self.redis_client:
                self.redis_client.delete(f"session:{token}")
        except Exception as e:
            logger.exception(f"Session removal error: {e}")

    # ---------------------------- Security --------------------------
    def _is_account_locked(self, username: str) -> bool:
        """Check if account is locked."""
        if username not in self.lockout_times:
            return False
        locked_at = self.lockout_times[username]
        if time.time() - locked_at < ACCOUNT_LOCKOUT_DURATION:
            return True
        # reset after lockout expires
        del self.lockout_times[username]
        self.failed_attempts[username] = 0
        return False

    def _record_failed_login(self, username: str, client_ip: str | None) -> None:
        """Record failed login attempt."""
        self.failed_attempts[username] += 1
        if self.failed_attempts[username] >= MAX_LOGIN_ATTEMPTS:
            self.lockout_times[username] = time.time()
            logger.warning(f"Account locked for user: {username}")

        ip_str = client_ip if client_ip else "unknown"
        logger.warning(f"Failed login for user: {username} from IP: {ip_str}")

    @staticmethod
    def _log_auth_ok(user_id: str, client_ip: str | None) -> None:
        """Log successful authentication."""
        ip_str = client_ip or "unknown"
        logger.info(f"Auth OK for user: {user_id} from IP: {ip_str}")

    def get_auth_stats(self) -> dict[str, Any]:
        """Get authentication statistics."""
        return {
            "failed_attempts": dict(self.failed_attempts),
            "lockout_times": dict(self.lockout_times),
            "redis_connected": bool(self.redis_client),
        }


# -----------------------------------------------------------------------------
# Singleton accessor
# -----------------------------------------------------------------------------
auth_manager = AuthenticationManager()


# -----------------------------------------------------------------------------
# Convenience helpers for FastAPI dependencies (optional)
# -----------------------------------------------------------------------------
def extract_bearer_token(authorization: str | None) -> str:
    """Extract raw token from 'Authorization: Bearer <token>'."""
    if not authorization:
        return ""
    parts = authorization.split(" ", 1)
    if len(parts) != 2:
        return ""
    if parts[0].lower() != "bearer":
        return ""
    return parts[1].strip()


def require_permissions(user_permissions: list[str], required: list[str]) -> bool:
    """Quick RBAC gate usable inside route handlers."""
    return auth_manager.rbac.check_permission(user_permissions, required)
