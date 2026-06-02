import logging
import os
import time
import uuid
from datetime import datetime, timedelta, timezone
from typing import Any

import jwt
from fastapi import HTTPException, WebSocket, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from backend.utils.password_utils import hash_password, verify_password

# Initialize logger
logger = logging.getLogger(__name__)


# JWT configuration with validation
def _get_jwt_secret():
    """Get JWT secret with validation"""
    jwt_secret = os.getenv("JWT_SECRET")
    if not jwt_secret or jwt_secret.strip() == "":
        msg = "JWT_SECRET environment variable must be set and non-empty"
        raise RuntimeError(msg)
    if len(jwt_secret) < 32:
        logger.warning("JWT_SECRET is shorter than 32 characters - consider using a stronger secret")
    return jwt_secret


# Lazy initialization - only check when actually needed
JWT_SECRET = None

JWT_ALGORITHM = "HS256"
JWT_ISSUER = os.getenv("JWT_ISSUER", "mystic-trading")
JWT_AUDIENCE = os.getenv("JWT_AUDIENCE", "mystic-api")
ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "30"))
REFRESH_TOKEN_EXPIRE_DAYS = int(os.getenv("REFRESH_TOKEN_EXPIRE_DAYS", "7"))

# API Token configuration - REMOVED: Not used anywhere in the application
# MYSTIC_API_TOKEN = os.getenv("MYSTIC_API_TOKEN", "").strip()
# if not MYSTIC_API_TOKEN:
#     logger.warning("MYSTIC_API_TOKEN not set - API token authentication is disabled")

# Brute force protection
MAX_LOGIN_ATTEMPTS = int(os.getenv("MAX_LOGIN_ATTEMPTS", "5"))
LOCKOUT_DURATION_MINUTES = int(os.getenv("LOCKOUT_DURATION_MINUTES", "15"))

# Security scheme for HTTP authentication
security = HTTPBearer(auto_error=False)

# In-memory tracking for brute force protection (in production, use Redis)
login_attempts = {}
lockouts = {}


class AuthenticationError(Exception):
    """Custom exception for authentication errors"""

    def __init__(self, message: str, status_code: int = status.HTTP_401_UNAUTHORIZED) -> None:
        self.message = message
        self.status_code = status_code
        super().__init__(message)


class BruteForceError(AuthenticationError):
    """Exception for brute force protection"""

    def __init__(self, message: str = "Too many login attempts") -> None:
        super().__init__(message, status.HTTP_429_TOO_MANY_REQUESTS)


class UserLockedError(AuthenticationError):
    """Exception for locked user accounts"""

    def __init__(self, message: str = "Account temporarily locked") -> None:
        super().__init__(message, status.HTTP_423_LOCKED)


def create_access_token(data: dict[str, Any], token_type: str = "access") -> str:
    """Create a new JWT access token with comprehensive claims"""
    now = datetime.now(timezone.utc)
    expire = now + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)

    to_encode = {
        **data,
        "exp": int(expire.timestamp()),  # Standardize on epoch timestamps
        "iat": int(now.timestamp()),
        "nbf": int(now.timestamp()),
        "iss": JWT_ISSUER,
        "aud": JWT_AUDIENCE,
        "typ": token_type,
        "jti": str(uuid.uuid4()),  # Unique token ID for replay protection
    }

    encoded_jwt = jwt.encode(to_encode, _get_jwt_secret(), algorithm=JWT_ALGORITHM)
    logger.debug(f"Created {token_type} token for user: {data.get('sub', 'unknown')}")
    return encoded_jwt


def create_refresh_token(data: dict[str, Any]) -> str:
    """Create a new JWT refresh token"""
    now = datetime.now(timezone.utc)
    expire = now + timedelta(days=REFRESH_TOKEN_EXPIRE_DAYS)

    to_encode = {
        **data,
        "exp": int(expire.timestamp()),
        "iat": int(now.timestamp()),
        "nbf": int(now.timestamp()),
        "iss": JWT_ISSUER,
        "aud": JWT_AUDIENCE,
        "typ": "refresh",
        "jti": str(uuid.uuid4()),
    }

    encoded_jwt = jwt.encode(to_encode, _get_jwt_secret(), algorithm=JWT_ALGORITHM)
    logger.debug(f"Created refresh token for user: {data.get('sub', 'unknown')}")
    return encoded_jwt


# Direct imports for production
# Database functions - stubs for now
async def get_user_by_credentials(_username: str):
    """Get user by credentials - stub implementation"""
    return


async def create_user(_user_data: dict):
    """Create user - stub implementation"""
    return


DATABASE_AVAILABLE = True


def verify_token(token: str) -> dict[str, Any]:
    """Verify a JWT token and return the payload with comprehensive validation"""
    try:
        # Decode with full validation
        payload = jwt.decode(
            token,
            _get_jwt_secret(),
            algorithms=[JWT_ALGORITHM],
            issuer=JWT_ISSUER,
            audience=JWT_AUDIENCE,
            options={"verify_exp": True, "verify_iss": True, "verify_aud": True},
        )
    except jwt.ExpiredSignatureError as e:
        logger.warning("Token verification failed: token expired")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token has expired",
        ) from e
    except jwt.InvalidIssuerError as e:
        logger.warning("Token verification failed: invalid issuer")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token issuer",
        ) from e
    except jwt.InvalidAudienceError as e:
        logger.warning("Token verification failed: invalid audience")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token audience",
        ) from e
    except (jwt.InvalidTokenError, jwt.DecodeError) as e:
        logger.warning(f"Token verification failed: {e}")
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token") from e

    # Validate token type after decoding - move outside try to avoid TRY301
    if payload.get("typ") not in ["access", "refresh"]:
        msg = "Invalid token type"
        raise jwt.InvalidTokenError(msg)

    logger.debug(f"Token verified for user: {payload.get('sub', 'unknown')}")
    return payload


def _check_brute_force_protection(identifier: str) -> None:
    """Check if user/IP is locked out due to brute force attempts"""
    current_time = time.time()

    # Check if identifier is currently locked out
    if identifier in lockouts:
        lockout_until = lockouts[identifier]
        if current_time < lockout_until:
            remaining_time = int((lockout_until - current_time) / 60)
            logger.warning(f"Brute force lockout active for {identifier}, {remaining_time} minutes remaining")
            msg = f"Account locked for {remaining_time} minutes"
            raise UserLockedError(msg)
        # Lockout expired, remove it
        del lockouts[identifier]
        login_attempts.pop(identifier, None)


def _record_failed_login(identifier: str) -> None:
    """Record a failed login attempt"""
    current_time = time.time()

    if identifier not in login_attempts:
        login_attempts[identifier] = []

    # Clean old attempts (older than lockout duration)
    cutoff_time = current_time - (LOCKOUT_DURATION_MINUTES * 60)
    login_attempts[identifier] = [attempt_time for attempt_time in login_attempts[identifier] if attempt_time > cutoff_time]

    # Add current attempt
    login_attempts[identifier].append(current_time)

    # Check if we should lock out
    if len(login_attempts[identifier]) >= MAX_LOGIN_ATTEMPTS:
        lockout_until = current_time + (LOCKOUT_DURATION_MINUTES * 60)
        lockouts[identifier] = lockout_until
        logger.warning(f"Brute force lockout activated for {identifier}")
        raise BruteForceError


def _clear_login_attempts(identifier: str) -> None:
    """Clear failed login attempts for successful login"""
    login_attempts.pop(identifier, None)
    lockouts.pop(identifier, None)


async def get_current_user(
    credentials: HTTPAuthorizationCredentials | None = None,
) -> dict[str, Any]:
    """Get the current user from the JWT token"""
    if not credentials:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authorization credentials required",
        )

    token = credentials.credentials
    return verify_token(token)


# REMOVED: require_api_token function - not used anywhere in the application
# async def require_api_token(
#     x_api_token: str | None = Header(default=None, alias="X-API-Token"),
# ) -> None:
#     """Optional API token guard with improved logging.
#
#     If environment variable MYSTIC_API_TOKEN is set, require requests to include
#     header X-API-Token with the exact same value. Otherwise, allow the request.
#     """
#     if MYSTIC_API_TOKEN:
#         if not x_api_token or x_api_token.strip() != MYSTIC_API_TOKEN:
#             logger.warning("API token authentication failed - invalid or missing token")
#             raise HTTPException(
#                 status_code=status.HTTP_401_UNAUTHORIZED,
#                 detail="Invalid or missing API token",
#             )
#         logger.debug("API token authentication successful")
#     else:
#         logger.debug("API token authentication disabled - no token configured")


async def verify_websocket_token(
    websocket: WebSocket,
) -> dict[str, Any] | None:
    """Verify the token from WebSocket connection with comprehensive support"""
    try:
        token = None

        # Try to get token from Authorization header first
        auth_header = websocket.headers.get("authorization", "")
        if auth_header.startswith("Bearer "):
            token = auth_header[7:]  # Remove "Bearer " prefix
            logger.debug("WebSocket token found in Authorization header")

        # Fallback to query parameter
        if not token:
            token = websocket.query_params.get("token")
            if token:
                logger.debug("WebSocket token found in query parameters")

        if not token:
            logger.warning("WebSocket connection rejected: no token provided")
            await websocket.close(code=status.WS_1008_POLICY_VIOLATION)
            return None

        # Verify token
        payload = verify_token(token)

        # Additional WebSocket-specific validation
        if payload.get("typ") != "access":
            logger.warning("WebSocket connection rejected: invalid token type")
            await websocket.close(code=status.WS_1008_POLICY_VIOLATION)
            return None

        logger.info(f"WebSocket connection authenticated for user: {payload.get('sub', 'unknown')}")
    except HTTPException as e:
        logger.warning(f"WebSocket token verification failed: {e.detail}")
        await websocket.close(code=status.WS_1008_POLICY_VIOLATION)
        return None
    except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
        logger.exception(f"WebSocket token verification error: {e}")
        await websocket.close(code=status.WS_1008_POLICY_VIOLATION)
        return None
    else:
        return payload


async def authenticate_user(username: str, password: str, client_ip: str = "unknown") -> dict[str, Any]:
    """Authenticate a user with username and password including brute force protection"""
    identifier = f"{username}:{client_ip}"  # Use username + IP for brute force tracking

    # Check brute force protection and validate database availability before try block
    _check_brute_force_protection(identifier)

    if not DATABASE_AVAILABLE:
        logger.error("Authentication failed: database not available")
        _record_failed_login(identifier)
        msg = "Authentication service unavailable"
        raise AuthenticationError(msg)

    try:
        # Get user from database
        user = await get_user_by_credentials(username)
    except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
        logger.exception(f"Error fetching user {username}: {e}")
        _record_failed_login(identifier)
        msg = f"Authentication failed: {e!s}"
        raise AuthenticationError(msg) from e

    # Validate user exists and password - move outside try to avoid TRY301
    if not user:
        logger.warning(f"Authentication failed: user not found - {username}")
        _record_failed_login(identifier)
        msg = "Invalid credentials"
        raise AuthenticationError(msg)

    # Verify password (assuming synchronous function)
    if not verify_password(password, user.get("password_hash", "")):
        logger.warning(f"Authentication failed: invalid password - {username}")
        _record_failed_login(identifier)
        msg = "Invalid credentials"
        raise AuthenticationError(msg)

    try:
        # Clear failed attempts on successful login
        _clear_login_attempts(identifier)

        # Create tokens
        token_data = {"sub": username, "user_id": user["id"]}
        access_token = create_access_token(token_data)
        refresh_token = create_refresh_token(token_data)

        logger.info(f"User authenticated successfully: {username}")

        return {
            "access_token": access_token,
            "refresh_token": refresh_token,
            "user_id": user["id"],
            "username": username,
        }

    except (BruteForceError, UserLockedError):
        # Re-raise brute force exceptions
        raise
    except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
        logger.exception(f"Authentication error for {username}: {e}")
        _record_failed_login(identifier)
        msg = f"Authentication failed: {e!s}"
        raise AuthenticationError(msg) from e


async def register_user(username: str, password: str, email: str) -> dict[str, Any]:
    """Register a new user with comprehensive validation"""
    # Validate database availability and input before try block
    if not DATABASE_AVAILABLE:
        logger.error("User registration failed: database not available")
        msg = "Registration service unavailable"
        raise AuthenticationError(msg)

    # Basic validation
    if not username or len(username.strip()) < 3:
        msg = "Username must be at least 3 characters long"
        raise AuthenticationError(msg)

    if not password or len(password) < 8:
        msg = "Password must be at least 8 characters long"
        raise AuthenticationError(msg)

    if not email or "@" not in email:
        msg = "Valid email address required"
        raise AuthenticationError(msg)

    try:
        # Check if user already exists
        existing_user = await get_user_by_credentials(username)
    except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
        logger.exception(f"Error checking existing user {username}: {e}")
        msg = f"Registration failed: {e!s}"
        raise AuthenticationError(msg) from e

    # Validate user doesn't already exist - move outside try to avoid TRY301
    if existing_user:
        logger.warning(f"Registration failed: user already exists - {username}")
        msg = "User already exists"
        raise AuthenticationError(msg)

    try:
        # Hash password and create user
        password_hash = hash_password(password)
        user_data = {
            "username": username.strip(),
            "email": email.strip().lower(),
            "password_hash": password_hash,
        }

        user = await create_user(user_data)

        # Create tokens for new user
        token_data = {"sub": username, "user_id": user["id"]}
        access_token = create_access_token(token_data)
        refresh_token = create_refresh_token(token_data)

        logger.info(f"User registered successfully: {username}")

        return {
            "user_id": user["id"],
            "username": username,
            "access_token": access_token,
            "refresh_token": refresh_token,
        }

    except AuthenticationError:
        # Re-raise authentication exceptions
        raise
    except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
        logger.exception(f"Registration error for {username}: {e}")
        msg = f"Registration failed: {e!s}"
        raise AuthenticationError(msg) from e


async def refresh_access_token(refresh_token: str) -> dict[str, Any]:
    """Refresh an access token using a valid refresh token"""
    # Verify refresh token (HTTPException will propagate naturally)
    payload = verify_token(refresh_token)

    # Ensure this is a refresh token - validate outside try to avoid TRY301
    if payload.get("typ") != "refresh":
        msg = "Invalid token type for refresh"
        raise AuthenticationError(msg)

    try:
        # Create new access token
        token_data = {"sub": payload.get("sub"), "user_id": payload.get("user_id")}

        new_access_token = create_access_token(token_data)

        logger.info(f"Access token refreshed for user: {payload.get('sub', 'unknown')}")
    except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
        logger.exception(f"Token refresh error: {e}")
        msg = f"Token refresh failed: {e!s}"
        raise AuthenticationError(msg) from e
    else:
        return {"access_token": new_access_token, "token_type": "bearer"}
