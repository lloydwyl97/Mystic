"""
Mystic single-source-of-truth trading mode.

This module is the ONE place that resolves whether Mystic is running in
PAPER or LIVE. Paper and live use the same trading brain, same symbols,
same indicators, same sell rules, same cooldown rules, same learning,
same logs, and same dust cleanup behavior. The only thing the mode
selects is the execution adapter.

Operator contract:
    Environment variable: MYSTIC_TRADING_MODE
    Allowed values:       paper | live
    Missing / invalid:    fail closed (RuntimeError on the first hard read)

Legacy aliases ``EXECUTION_MODE`` and ``TRADING_MODE`` are honored only as
a back-compat tier under ``MYSTIC_TRADING_MODE``; when both are present
they MUST agree, otherwise resolution fails closed.
"""

from __future__ import annotations

import logging
import os
from enum import Enum
from typing import Final

logger = logging.getLogger(__name__)

ENV_VAR_PRIMARY: Final[str] = "MYSTIC_TRADING_MODE"
ENV_VAR_LEGACY: Final[tuple[str, ...]] = ("EXECUTION_MODE", "TRADING_MODE")
_ALLOWED: Final[frozenset[str]] = frozenset({"paper", "live"})


class TradingMode(str, Enum):
    PAPER = "paper"
    LIVE = "live"


class TradingModeError(RuntimeError):
    """Raised when the configured trading mode is missing or invalid."""


def _read_raw_mode() -> tuple[str, str]:
    """Return (raw_value, source_var_name). raw_value is "" when unset."""
    primary = os.getenv(ENV_VAR_PRIMARY, "")
    if primary:
        return primary.strip().lower(), ENV_VAR_PRIMARY
    for var in ENV_VAR_LEGACY:
        raw = os.getenv(var, "")
        if raw:
            return raw.strip().lower(), var
    return "", ENV_VAR_PRIMARY


def _consistency_check(primary_mode: str) -> None:
    """If any legacy alias is set, it MUST equal the resolved primary mode."""
    for var in ENV_VAR_LEGACY:
        raw = os.getenv(var, "")
        if not raw:
            continue
        if raw.strip().lower() != primary_mode:
            raise TradingModeError(f"trading mode mismatch: {ENV_VAR_PRIMARY}={primary_mode!r} but {var}={raw!r}. Refusing to run with ambiguous mode.")


def resolve_trading_mode() -> TradingMode:
    """
    Resolve the active trading mode (paper/live).

    Fails closed:
      * Missing -> TradingModeError
      * Unknown value -> TradingModeError
      * MYSTIC_TRADING_MODE disagrees with EXECUTION_MODE/TRADING_MODE
        when both are set -> TradingModeError
    """
    raw, source = _read_raw_mode()
    if not raw:
        raise TradingModeError(f"trading mode is not configured. Set {ENV_VAR_PRIMARY}=paper or {ENV_VAR_PRIMARY}=live. Mystic refuses to default silently.")
    if raw not in _ALLOWED:
        raise TradingModeError(f"invalid trading mode {raw!r} (from {source}). Allowed values: {sorted(_ALLOWED)}.")
    _consistency_check(raw)
    return TradingMode(raw)


def is_live() -> bool:
    return resolve_trading_mode() is TradingMode.LIVE


def is_paper() -> bool:
    return resolve_trading_mode() is TradingMode.PAPER


def log_trading_mode_at_startup() -> TradingMode:
    """
    Hard-read the trading mode and log it. Intended to be called once during
    backend bootstrap. Re-raises ``TradingModeError`` if the configuration is
    invalid so the process fails closed instead of drifting into a default.
    """
    mode = resolve_trading_mode()
    src_value, src_var = _read_raw_mode()
    logger.warning(
        "MYSTIC_TRADING_MODE_RESOLVED mode=%s source=%s raw=%s",
        mode.value,
        src_var,
        src_value,
    )
    if mode is TradingMode.LIVE:
        logger.warning("=" * 60)
        logger.warning("MYSTIC TRADING MODE = LIVE (Binance.US real money)")
        logger.warning("=" * 60)
    else:
        logger.warning("MYSTIC TRADING MODE = PAPER (simulated execution)")
    return mode
