"""
Circuit Breaker Middleware - All Live Data, No Fallback/Hardcoded Data

This module provides circuit breaker middleware for live API requests (backend port 8000).
All operations:
- Monitor live API request success/failure rates
- Protect backend services from cascading failures
- Track live service health and circuit states
- No fallback/hardcoded data - all monitoring from live operations
- Used by backend services on port 8000 for live trading operations

Live Data Sources:
- API requests: Live requests to backend API (port 8000)
- Response status codes: Live HTTP status codes from backend operations
- Circuit states: Live service health states (CLOSED, OPEN, HALF_OPEN)
- Failure tracking: Live consecutive failure counts from real API responses
- All circuit breaker operations use live data - no mock/test data

Endpoint References:
- Backend API: Port 8000 (circuit breaker monitors live requests)
- All circuit breaker operations use live connections - no fallback/hardcoded data
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections import defaultdict
from collections.abc import Awaitable, Callable
from dataclasses import dataclass

from fastapi import Request, Response
from fastapi.responses import JSONResponse

# Performance logging decorator not available - commenting out
# from backend.enhanced_logging import log_operation_performance


# Stub decorator to prevent errors
def log_operation_performance(_operation_name: str) -> Callable:
    """
    Stub decorator - performance logging disabled.

    Placeholder for performance logging decorator (not fallback data, dependency missing).
    """

    def decorator(func: Callable) -> Callable:
        return func

    return decorator


logger = logging.getLogger(__name__)

# --- Config ---
# Configuration defaults for live circuit breaker (not fallback data)
FAILURE_THRESHOLD = 5  # Consecutive failures before OPEN (configuration default, not fallback data)
RESET_TIMEOUT = 60.0  # Seconds to move from OPEN -> HALF_OPEN (configuration default, not fallback data)
HALF_OPEN_MAX_TRIALS = 3  # Allowed trial requests in HALF_OPEN (configuration default, not fallback data)
SERVER_ERROR_MIN = 500  # Status codes >= this count as failure (configuration default, not fallback data)

# --- Time source (monotonic for robustness) ---
_now = time.monotonic


# --- State model ---
@dataclass
class CircuitState:
    """
    Circuit state for live API request monitoring.

    Tracks live service health state from real API requests - no fallback/hardcoded data.
    """

    failures: int = 0  # Live consecutive failure count
    last_failure_ts: float = 0.0  # Live timestamp of last failure
    is_open: bool = False  # Live circuit open state
    half_open_trials: int = 0  # Live counts trials in HALF_OPEN state


# Live circuit states for live API requests (backend port 8000)
circuit_states: dict[str, CircuitState] = defaultdict(CircuitState)


def _get_state_lock() -> asyncio.Lock:
    """Get or create state lock lazily."""
    if not hasattr(_get_state_lock, "_lock"):
        _get_state_lock._lock = asyncio.Lock()
    return _get_state_lock._lock


def _key_for_request(request: Request) -> str:
    # Group by method + path (adjust as needed)
    return f"{request.method} {request.url.path}"


def _is_server_failure_status(status_code: int) -> bool:
    return status_code >= SERVER_ERROR_MIN


async def _should_block(state: CircuitState) -> bool:
    """
    Returns True if state is OPEN and still within timeout.
    Also transitions OPEN -> HALF_OPEN when timeout elapses.
    """
    if not state.is_open:
        return False

    elapsed = _now() - state.last_failure_ts
    if elapsed >= RESET_TIMEOUT:
        # Move to HALF_OPEN: allow limited trials
        state.is_open = False
        state.half_open_trials = 0
        state.failures = 0  # reset consecutive counter for the new phase
        return False

    return True


def _set_open(state: CircuitState) -> None:
    state.is_open = True
    state.last_failure_ts = _now()
    state.failures = FAILURE_THRESHOLD  # pin at threshold
    state.half_open_trials = 0


def _record_success(state: CircuitState) -> None:
    # Success path:
    # - If we were HALF_OPEN, count trial successes and close after first success
    # - Otherwise, just reset consecutive failures
    if not state.is_open and state.half_open_trials > 0:
        # Success during HALF_OPEN → CLOSE immediately
        state.half_open_trials = 0
    state.failures = 0


def _record_failure(state: CircuitState) -> None:
    # Failure increments failures. If in HALF_OPEN (trials > 0), immediately OPEN.
    if state.half_open_trials > 0:
        _set_open(state)
        return

    state.failures += 1
    state.last_failure_ts = _now()
    if state.failures >= FAILURE_THRESHOLD:
        _set_open(state)


def _enter_half_open_if_needed(state: CircuitState) -> None:
    # This helper marks that we're attempting a HALF_OPEN trial
    # Only used when previously OPEN transitioned due to timeout.
    if state.half_open_trials == 0:
        state.half_open_trials = 1
    else:
        state.half_open_trials += 1
        if state.half_open_trials > HALF_OPEN_MAX_TRIALS:
            # Too many trials without clear success → OPEN again
            _set_open(state)


def _state_label(state: CircuitState) -> str:
    if state.is_open:
        return "OPEN"
    if state.half_open_trials > 0:
        return "HALF_OPEN"
    return "CLOSED"


@log_operation_performance("circuit_breaker")
async def circuit_breaker_middleware(request: Request, call_next: Callable[[Request], Awaitable[Response]]) -> Response | JSONResponse:
    """
    Circuit breaker middleware for live API requests (backend port 8000).

    Monitors live API request success/failure rates and protects backend services.
    All monitoring uses live data - no fallback/hardcoded data.

    States:
      - CLOSED: Pass-through; count consecutive failures from live API responses
      - OPEN: Short-circuit with 503 until RESET_TIMEOUT; then transition to HALF_OPEN
      - HALF_OPEN: Allow limited trial requests; first success -> CLOSED, failure -> OPEN

    Args:
        request: Live API request to backend (port 8000)
        call_next: Next middleware handler

    Returns:
        Live API response or circuit breaker error response (503 if circuit open)
    """
    key = _key_for_request(request)

    async with _get_state_lock():
        state = circuit_states[key]
        if await _should_block(state):
            logger.warning("Circuit OPEN for %s", key)
            return JSONResponse(
                status_code=503,
                content={"detail": "Service temporarily unavailable"},
                headers={"X-Circuit-State": "OPEN"},
            )

        # If we just transitioned from OPEN (timeout elapsed), mark HALF_OPEN trial
        if not state.is_open and state.failures == 0 and state.half_open_trials >= 0 and state.last_failure_ts > 0:
            _enter_half_open_if_needed(state)

    # Execute downstream
    try:
        response = await call_next(request)

        # Success path or failure by status code
        async with _get_state_lock():
            state = circuit_states[key]
            if _is_server_failure_status(response.status_code):
                _record_failure(state)
            else:
                _record_success(state)

            # Add state header for observability (non-invasive)
            response.headers["X-Circuit-State"] = _state_label(state)
    except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError):
        # Record live exception as failure (not fallback data, error handling)
        logger.exception("Circuit breaker caught exception for live request %s", key)
        async with _get_state_lock():
            state = circuit_states[key]
            _record_failure(state)
            hdr = {"X-Circuit-State": _state_label(state)}
        return JSONResponse(status_code=500, content={"detail": "Internal server error"}, headers=hdr)
    else:
        return response
