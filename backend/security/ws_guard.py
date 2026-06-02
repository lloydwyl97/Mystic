# backend/security/ws_guard.py
import os
import re

LOCAL_ORIGIN_RE = re.compile(r"^https?://(localhost|127\.0\.0\.1)(:\d+)?$", re.IGNORECASE)


def is_dev_mode() -> bool:
    """Check if running in development mode."""
    env_val = os.getenv("ENV")
    debug_val = os.getenv("DEBUG")

    env_check = False
    if env_val:
        env_check = env_val.lower() == "dev"

    debug_check = False
    if debug_val:
        debug_check = debug_val.lower() in {
            "1",
            "true",
            "yes",
            "on",
        }

    return env_check or debug_check


def origin_is_local(origin: str | None) -> bool:
    """Check if origin is localhost."""
    if not origin:
        return False
    return bool(LOCAL_ORIGIN_RE.match(origin.strip()))


def extract_ws_token(headers, query_params) -> str | None:
    """Extract WebSocket token from headers or query parameters."""
    # Headers are plain dict-like in Starlette/FastAPI
    token = headers.get("X-WS-Token")
    if token:
        return token
    return query_params.get("token")


def require_ws_auth(_path: str, headers, query_params):
    """Require WebSocket authentication."""
    origin = headers.get("Origin")
    if is_dev_mode() and origin_is_local(origin):
        origin_str = origin if origin else "unknown"
        return True, f"dev-local origin={origin_str}"

    expected = os.getenv("WS_TOKEN")
    provided = extract_ws_token(headers, query_params)

    if not expected:
        return False, "WS_TOKEN not configured"
    if not provided:
        return False, "token missing"
    if provided != expected:
        return False, "token mismatch"
    return True, "token ok"
