"""
Password Utilities - Live Configuration Only

Secure password hashing and verification using bcrypt via passlib.
All configuration values come from live environment variables - no hardcoded values.

Usage:
- hash_password(password) -> returns bcrypt hash string
- verify_password(password, password_hash) -> returns bool
"""

from __future__ import annotations

import logging
import os

# Lazy import for passlib - may not be available in all deployments
try:
    from passlib.context import CryptContext
except (ImportError, ModuleNotFoundError, AttributeError, ValueError, TypeError, RuntimeError):
    CryptContext = None  # type: ignore[assignment, misc]

logger = logging.getLogger(__name__)

# Global password context - initialized on first use
_password_context: CryptContext | None = None


def _get_password_context() -> CryptContext:
    """Get or create the password hashing context with live configuration."""
    global _password_context

    if _password_context is not None:
        return _password_context

    if CryptContext is None:
        msg = "passlib[bcrypt] is required for password hashing. Install with: pip install 'passlib[bcrypt]'"
        raise RuntimeError(msg)

    # Get bcrypt rounds from environment (live config)
    # Default to 12 rounds (secure default), but allow override via env var
    bcrypt_rounds_str = os.getenv("BCRYPT_ROUNDS", "12")
    try:
        bcrypt_rounds = int(bcrypt_rounds_str)
        # Validate reasonable range (4-31 for bcrypt)
        if bcrypt_rounds < 4:
            logger.warning(f"BCRYPT_ROUNDS ({bcrypt_rounds}) is too low, using minimum 4")
            bcrypt_rounds = 4
        elif bcrypt_rounds > 31:
            logger.warning(f"BCRYPT_ROUNDS ({bcrypt_rounds}) is too high, using maximum 31")
            bcrypt_rounds = 31
    except (ValueError, TypeError):
        logger.warning(f"Invalid BCRYPT_ROUNDS value '{bcrypt_rounds_str}', using default 12")
        bcrypt_rounds = 12

    # Create context with live configuration
    # Use bcrypt as primary scheme, with automatic migration support
    _password_context = CryptContext(
        schemes=["bcrypt"],
        bcrypt__rounds=bcrypt_rounds,
        deprecated="auto",  # Automatically handle deprecated schemes
    )

    logger.info(f"Password context initialized with bcrypt (rounds={bcrypt_rounds})")
    return _password_context


def hash_password(password: str) -> str:
    """
    Hash a password using bcrypt with live configuration.

    Args:
        password: Plain text password to hash

    Returns:
        Bcrypt hash string (includes salt and rounds)

    Raises:
        RuntimeError: If passlib[bcrypt] is not available
        ValueError: If password is empty or invalid
    """
    if not password or not isinstance(password, str):
        msg = "Password must be a non-empty string"
        raise ValueError(msg)

    if not password.strip():
        msg = "Password cannot be empty or whitespace only"
        raise ValueError(msg)

    try:
        context = _get_password_context()
        hashed = context.hash(password)
    except (AttributeError, TypeError) as e:
        msg = f"Password hashing failed: {e}"
        raise RuntimeError(msg) from e
    else:
        return hashed


def verify_password(password: str, password_hash: str) -> bool:
    """
    Verify a password against a stored hash using bcrypt.

    Args:
        password: Plain text password to verify
        password_hash: Stored bcrypt hash string

    Returns:
        True if password matches hash, False otherwise

    Raises:
        RuntimeError: If passlib[bcrypt] is not available
        ValueError: If inputs are invalid
    """
    if not password or not isinstance(password, str):
        return False

    if not password_hash or not isinstance(password_hash, str):
        return False

    if not password_hash.strip():
        return False

    try:
        context = _get_password_context()
        is_valid = context.verify(password, password_hash)
    except (AttributeError, TypeError, ValueError) as e:
        logger.warning(f"Password verification error: {e}")
        return False
    else:
        return is_valid
