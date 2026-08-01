"""
Redis Helpers - Live Configuration Only

Lightweight helpers to normalize Redis return types across sync/async clients
and different configurations (bytes vs str), so downstream JSON parsing and
string operations are safe.
All configuration values come from live config - no hardcoded values.

Usage:
- Use to_str(...) for single-value reads like lpop/get
- Use to_str_list([...]) for list reads like lrange
- Use WriterLock for single-writer enforcement per Redis keyspace
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import os
import platform
import time
from collections.abc import Iterable
from typing import Any

# Import live configuration
try:
    from backend.config_bridge import get_mystic_config

    _mystic_config = get_mystic_config()
except (ImportError, AttributeError, ValueError, TypeError, RuntimeError):
    _mystic_config = None

# --- Live Configuration Helpers -------------------------------------------------------------------


def _get_redis_encoding() -> str:
    """Get Redis encoding from live config."""
    if _mystic_config:
        try:
            value = _mystic_config
            if hasattr(value, "redis") and hasattr(value.redis, "encoding"):
                encoding = value.redis.encoding
                if isinstance(encoding, str) and encoding:
                    return encoding.strip()
        except (AttributeError, ValueError, TypeError):
            pass

    encoding = os.getenv("REDIS_ENCODING", "").strip()
    if encoding:
        return encoding

    return "utf-8"


def _get_redis_decode_errors() -> str:
    """Get Redis decode error handling mode from live config."""
    if _mystic_config:
        try:
            value = _mystic_config
            if hasattr(value, "redis") and hasattr(value.redis, "decode_errors"):
                errors = value.redis.decode_errors
                if isinstance(errors, str) and errors:
                    return errors.strip()
        except (AttributeError, ValueError, TypeError):
            pass

    errors = os.getenv("REDIS_DECODE_ERRORS", "").strip()
    if errors:
        return errors

    return "ignore"


def to_str(value: Any) -> str | None:
    """Return value as str, decoding bytes with configured encoding; None stays None.

    This avoids bytes passed into json.loads or string operations.
    """
    if value is None:
        return None
    if isinstance(value, bytes):
        encoding = _get_redis_encoding()
        decode_errors = _get_redis_decode_errors()
        try:
            return value.decode(encoding)
        except (UnicodeDecodeError, AttributeError, TypeError):
            # Fallback with configured error handling to avoid hard failures on rare bad bytes
            return value.decode(encoding, errors=decode_errors)
    if isinstance(value, str):
        return value
    # For non-bytes, non-str (rare for Redis), coerce to str explicitly
    return str(value)


def to_str_list(values: Any) -> list[str]:
    """Normalize a Redis list result (possibly list[bytes|str]) to list[str]."""
    if values is None:
        return []
    if not isinstance(values, Iterable) or isinstance(values, str | bytes):
        # Defensive: unexpected scalar; coerce to single-element list
        s = to_str(values)
        return [s] if s is not None else []
    result: list[str] = []
    for item in values:
        s = to_str(item)
        if s is not None:
            result.append(s)
    return result


def _is_async_redis(client: Any) -> bool:
    """Check if a Redis client is an async (redis.asyncio) client."""
    try:
        import redis.asyncio as aioredis

        return isinstance(client, aioredis.Redis)
    except (ImportError, AttributeError):
        return False


# --- Writer Lock System for Single-Writer Enforcement -----------------------------------------------

logger = logging.getLogger(__name__)


class WriterLockError(Exception):
    """Raised when a writer lock cannot be acquired due to another process holding it."""

    pass


def _lock_holder_dead_on_same_host(existing_lock_str: str) -> bool:
    """
    True if ``existing_lock_str`` parses as ``{pid}@{hostname}|...`` where
    hostname matches this machine and PID is no longer alive (stale unclean shutdown).
    Never True for another host — avoids stealing from a live Redis consumer elsewhere.
    """
    if not existing_lock_str or "|" not in existing_lock_str:
        return False
    meta = existing_lock_str.split("|", 1)[0].strip()
    if "@" not in meta:
        return False
    try:
        pid_s, hostname = meta.split("@", 1)
        holder_pid = int(pid_s.strip())
    except ValueError:
        return False
    here = platform.node()
    if hostname.strip() != here:
        return False
    try:
        os.kill(holder_pid, 0)
        return False  # Alive (or zombie still visible — do not steal)
    except ProcessLookupError:
        return True
    except PermissionError:
        return False


class WriterLock:
    """
    Redis-based writer lock for single-writer enforcement per keyspace.

    Prevents duplicate services from writing to the same Redis key patterns.
    Each writer role can only have one active instance across all processes.

    Usage:
        async with WriterLock("ai_signal_writer", redis_client):
            # Only this process can write canonical ai_signal:*:* keys (example):
            await redis.hmset("ai_signal:day:BTCUSDT", {...})
    """

    LOCK_TTL_SECONDS = 30  # Lock expires after 30 seconds
    REFRESH_INTERVAL_SECONDS = 10  # Refresh every 10 seconds

    def __init__(self, role: str, redis_client: Any, exit_on_failure: bool = True):
        """
        Initialize writer lock.

        Args:
            role: Writer role (e.g., "ai_signal_writer", "market_data_writer")
            redis_client: Redis client (sync or async)
            exit_on_failure: Whether to exit process if lock acquisition fails
        """
        self.role = role
        self.redis = redis_client
        self.exit_on_failure = exit_on_failure
        self.lock_key = f"writer:{role}"
        self.writer_id = f"{os.getpid()}@{platform.node()}"
        self.start_time = int(time.time())
        self.is_locked = False
        self._refresh_task: asyncio.Task | None = None

        # Create lock value with metadata
        self.lock_value = f"{self.writer_id}|{self.start_time}"

    async def acquire(self, *, _retry_after_stale_clear: bool = False) -> bool:
        """
        Acquire the writer lock with retry logic for slow Windows+WSL2 connections.

        Returns:
            True if lock acquired successfully, False otherwise
        """
        # CRITICAL FIX: Windows + WSL2 async Redis connections are slow on first attempt
        # Retry up to 3 times with exponential backoff to handle transient connection issues
        max_retries = 3
        retry_delay = 2.0  # Start with 2 second delay

        for attempt in range(max_retries):
            try:
                # Try to acquire lock with SET NX EX
                if _is_async_redis(self.redis):
                    success = await self.redis.set(
                        self.lock_key,
                        self.lock_value,
                        nx=True,
                        ex=self.LOCK_TTL_SECONDS,
                    )
                else:

                    def _sync_set():
                        return self.redis.set(self.lock_key, self.lock_value, nx=True, ex=self.LOCK_TTL_SECONDS)

                    success = await asyncio.to_thread(_sync_set)

                # If we got here, connection succeeded - break out of retry loop
                break

            except (ConnectionError, TimeoutError, OSError) as conn_error:
                if attempt < max_retries - 1:
                    logger.warning(f"[WRITER-LOCK] Connection attempt {attempt + 1}/{max_retries} failed: {conn_error}")
                    logger.warning(f"[WRITER-LOCK] Retrying in {retry_delay}s...")
                    await asyncio.sleep(retry_delay)
                    retry_delay *= 2  # Exponential backoff
                    continue
                else:
                    # Final attempt failed - re-raise
                    logger.exception(f"[WRITER-LOCK] All {max_retries} connection attempts failed")
                    raise

        try:
            if success:
                self.is_locked = True
                logger.info(f"Acquired writer lock {self.role}: {self.writer_id}")

                # Start refresh task
                if _is_async_redis(self.redis):
                    self._refresh_task = asyncio.create_task(self._refresh_loop())

                return True
            else:
                # Lock already held - check if it's our own lock (re-entrant case)
                try:
                    if _is_async_redis(self.redis):
                        existing_lock = await self.redis.get(self.lock_key)
                    else:
                        existing_lock = await asyncio.to_thread(self.redis.get, self.lock_key)

                    existing_lock_str = to_str(existing_lock)

                    # Check if this is our own lock (re-entrant acquisition)
                    # Lock format: "{pid}@{hostname}|{timestamp}"
                    if existing_lock_str and existing_lock_str.startswith(f"{self.writer_id}|"):
                        # This is our own lock - treat as successful re-acquisition
                        logger.info(f"[WRITER-LOCK] Re-acquired own lock {self.role}: {self.writer_id}")
                        self.is_locked = True
                        # Refresh the lock with new timestamp
                        if _is_async_redis(self.redis):
                            await self.redis.set(self.lock_key, self.lock_value, ex=self.LOCK_TTL_SECONDS)
                        else:
                            await asyncio.to_thread(lambda: self.redis.set(self.lock_key, self.lock_value, ex=self.LOCK_TTL_SECONDS))
                        return True

                    if existing_lock_str and not _retry_after_stale_clear and _lock_holder_dead_on_same_host(existing_lock_str):
                        logger.warning(
                            "[WRITER-LOCK] Stale holder (dead PID on %s); clearing %s before retry once",
                            platform.node(),
                            self.lock_key,
                        )
                        try:
                            if _is_async_redis(self.redis):
                                await self.redis.delete(self.lock_key)
                            else:
                                await asyncio.to_thread(self.redis.delete, self.lock_key)
                        except Exception as del_err:
                            logger.warning("[WRITER-LOCK] Failed to delete stale lock: %s", del_err)
                            return False
                        return await self.acquire(_retry_after_stale_clear=True)

                    logger.error(f"[WRITER-LOCK] FAILED to acquire {self.role} lock")
                    logger.error(f"[WRITER-LOCK] Lock held by: {existing_lock_str}")
                    logger.error(f"[WRITER-LOCK] This process: {self.writer_id}")

                    if self.exit_on_failure:
                        logger.error("[WRITER-LOCK] Raising WriterLockError (duplicate writer blocked)")
                        raise WriterLockError(f"Writer lock {self.role} already held by {existing_lock_str}")

                except WriterLockError:
                    raise
                except Exception as check_err:
                    logger.exception(f"[WRITER-LOCK] Failed to check existing lock: {check_err}")

                return False

        except WriterLockError:
            raise
        except Exception as e:
            logger.exception(f"[WRITER-LOCK] Exception acquiring {self.role} lock: {e}")
            if self.exit_on_failure:
                raise WriterLockError(f"Failed to acquire writer lock {self.role}: {e}") from e
            return False

    async def release(self) -> None:
        """Release the writer lock."""
        if not self.is_locked:
            return

        # Cancel refresh task
        if self._refresh_task and not self._refresh_task.done():
            self._refresh_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._refresh_task

        # Atomic release: only delete if we still own the lock
        try:
            release_script = """
if redis.call('get', KEYS[1]) == ARGV[1] then
    return redis.call('del', KEYS[1])
else
    return 0
end
"""
            result = await self.redis.eval(release_script, 1, self.lock_key, self.lock_value)

            if result:
                logger.info(f"[WRITER-LOCK] Released {self.role} lock: {self.writer_id}")
            else:
                logger.warning(f"[WRITER-LOCK] Lock {self.role} was not owned by us, skipped release")
        except Exception as e:
            logger.warning(f"[WRITER-LOCK] Failed to release {self.role} lock: {e}")
        finally:
            self.is_locked = False

    async def _refresh_loop(self) -> None:
        """Background task to refresh the lock periodically."""
        while self.is_locked:
            try:
                await asyncio.sleep(self.REFRESH_INTERVAL_SECONDS)

                if not self.is_locked:
                    break

                # Atomic refresh: only extend TTL if we still own the lock
                refresh_script = """
if redis.call('get', KEYS[1]) == ARGV[1] then
    return redis.call('expire', KEYS[1], ARGV[2])
else
    return 0
end
"""
                result = await self.redis.eval(refresh_script, 1, self.lock_key, self.lock_value, str(self.LOCK_TTL_SECONDS))

                if not result:
                    # Try to self-heal if TTL briefly expired and no other owner took the lock.
                    current_raw = await self.redis.get(self.lock_key)
                    current_lock = to_str(current_raw) or ""
                    if not current_lock:
                        reacquired = await self.redis.set(
                            self.lock_key,
                            self.lock_value,
                            nx=True,
                            ex=self.LOCK_TTL_SECONDS,
                        )
                        if reacquired:
                            logger.warning(
                                "[WRITER-LOCK] Re-acquired %s lock after transient expiry: %s",
                                self.role,
                                self.writer_id,
                            )
                            continue

                    if current_lock.startswith(f"{self.writer_id}|"):
                        # Same process metadata but stale value/TTL edge case; refresh forcefully.
                        await self.redis.set(self.lock_key, self.lock_value, ex=self.LOCK_TTL_SECONDS)
                        logger.warning(
                            "[WRITER-LOCK] Refreshed %s lock with current owner metadata: %s",
                            self.role,
                            self.writer_id,
                        )
                        continue

                    logger.warning(
                        "[WRITER-LOCK] Lost %s lock ownership (holder=%s); stopping refresh loop",
                        self.role,
                        current_lock or "none",
                    )
                    self.is_locked = False
                    break

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.warning(f"[WRITER-LOCK] Refresh loop error for {self.role}: {e}")
                await asyncio.sleep(1)  # Brief pause before retry

    async def __aenter__(self):
        """Async context manager entry."""
        success = await self.acquire()
        if not success:
            raise RuntimeError(f"Failed to acquire writer lock for {self.role}")
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.release()


def create_writer_payload(writer_role: str, data: dict[str, Any]) -> dict[str, Any]:
    """
    Add writer metadata to a Redis payload.

    Args:
        writer_role: The role of the writer (e.g., "ai_signal_writer")
        data: The payload data

    Returns:
        Enhanced payload with writer metadata
    """
    writer_id = f"{os.getpid()}@{platform.node()}"

    enhanced_data = data.copy()
    enhanced_data.update({"writer_role": writer_role, "writer_id": writer_id, "writer_timestamp": int(time.time())})

    return enhanced_data


def verify_writer_payload(expected_role: str, payload: dict[str, Any], redis_client: Any) -> bool:
    """
    Verify that a payload came from the expected writer role.

    Args:
        expected_role: The expected writer role
        payload: The payload to verify
        redis_client: Redis client for error counting

    Returns:
        True if payload is valid, False otherwise
    """
    try:
        actual_role = payload.get("writer_role")

        if actual_role != expected_role:
            # Increment mismatch counter
            try:
                if _is_async_redis(redis_client):
                    _incr_task = asyncio.create_task(redis_client.incr("alerts:writer_mismatch"))
                else:
                    redis_client.incr("alerts:writer_mismatch")
            except Exception:
                logger.debug("Redis alerts:writer_mismatch incr failed", exc_info=True)
                pass  # Don't fail verification on counter error

            return False

        return True

    except Exception as ex:
        logger.debug("verify_writer_topology failed: %s", ex)
        return False


# Exclusive canonical writers. Each role must hold a WriterLock for its process
# lifetime and is therefore expected under the ``writer:*`` Redis keyspace.
WRITER_ROLES = {
    "MARKET_DATA": "market_data_writer",
    "AI_SIGNALS": "ai_signal_writer",
    "DECISION_ROUTER": "decision_router",
}

# The Binance weight limiter is deliberately multi-process. Its token updates use
# Redis WATCH/MULTI transactions, so representing it as an exclusive writer would
# either block legitimate consumers or make topology report a permanently missing
# lock.
SHARED_ATOMIC_WRITER_ROLES = {
    "RATE_LIMITER": "binance_weight_limiter",
}
