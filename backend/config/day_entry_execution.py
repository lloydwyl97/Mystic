"""DAY entry execution mode. Fail closed. Never select legacy immediate BUY."""

from __future__ import annotations

import os
from typing import Final

TRAILING_BUY_MODE: Final[str] = "trailing_buy"
ENTRY_AUTHORITY_TRAILING_BUY: Final[str] = "DAY_TRAILING_BUY_CONFIRMED"
VALID_ENTRY_MODES: Final[frozenset[str]] = frozenset({TRAILING_BUY_MODE})
MODE_ENV: Final[str] = "DAY_ENTRY_EXECUTION_MODE"
MAX_WAIT_ENV: Final[str] = "DAY_TRAILING_BUY_MAX_WAIT_SECONDS"
DEFAULT_MAX_WAIT_SECONDS: Final[int] = 900
BOOK_STALE_ENV: Final[str] = "DAY_BOOK_STALE_SEC"
DEFAULT_BOOK_STALE_SEC: Final[float] = 30.0


def _resolve_book_stale_sec() -> float:
    raw = str(os.getenv(BOOK_STALE_ENV, "") or "").strip()
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return DEFAULT_BOOK_STALE_SEC
    return value if value > 0 else DEFAULT_BOOK_STALE_SEC


BOOK_STALE_SEC: Final[float] = _resolve_book_stale_sec()


def raw_day_entry_execution_mode() -> str:
    return str(os.getenv(MODE_ENV, "") or "").strip().lower()


def trailing_buy_mode_status() -> tuple[bool, str, str]:
    """Return (ok, error_code, mode). Missing or invalid mode is not trailing_buy."""
    mode = raw_day_entry_execution_mode()
    if not mode:
        return False, "DAY_ENTRY_EXECUTION_MODE_MISSING", mode
    if mode not in VALID_ENTRY_MODES:
        return False, f"DAY_ENTRY_EXECUTION_MODE_INVALID:{mode}", mode
    return True, "", mode


def trailing_buy_mode_active() -> bool:
    ok, _err, _mode = trailing_buy_mode_status()
    return ok


def is_trailing_buy_confirmed(entry_authority: object) -> bool:
    return str(entry_authority or "") == ENTRY_AUTHORITY_TRAILING_BUY


def trailing_buy_max_wait_seconds() -> int:
    raw = str(os.getenv(MAX_WAIT_ENV, str(DEFAULT_MAX_WAIT_SECONDS)) or "").strip()
    try:
        value = int(float(raw))
    except (TypeError, ValueError):
        return DEFAULT_MAX_WAIT_SECONDS
    return max(1, value)


__all__ = [
    "BOOK_STALE_ENV",
    "BOOK_STALE_SEC",
    "DEFAULT_BOOK_STALE_SEC",
    "DEFAULT_MAX_WAIT_SECONDS",
    "ENTRY_AUTHORITY_TRAILING_BUY",
    "MAX_WAIT_ENV",
    "MODE_ENV",
    "TRAILING_BUY_MODE",
    "VALID_ENTRY_MODES",
    "is_trailing_buy_confirmed",
    "raw_day_entry_execution_mode",
    "trailing_buy_max_wait_seconds",
    "trailing_buy_mode_active",
    "trailing_buy_mode_status",
]
