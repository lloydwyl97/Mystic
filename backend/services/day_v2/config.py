"""DAY V2 configuration.

All configuration keys are namespaced with DAY_V2_ to ensure complete
isolation from legacy DAY_ and SCALP_ variables. No defaults inherit from
existing engine config. Every variable must be independently set.

Fails closed: if any env var is present but not parseable, raises ValueError.
If DAY_V2_ENABLED is not explicitly 'true', get_day_v2_config() raises.
"""

import os


def _read_bool(key: str, default: bool) -> bool:
    """Parse a boolean env var. Accepts 'true'/'false' (case-insensitive)."""
    raw = os.environ.get(key)
    if raw is None:
        return default
    v = raw.strip().lower()
    if v == "true":
        return True
    if v == "false":
        return False
    raise ValueError(f"DAY V2 config error: {key!r} must be 'true' or 'false', got {raw!r}")


def _read_int(key: str, default: int) -> int:
    """Parse an integer env var."""
    raw = os.environ.get(key)
    if raw is None:
        return default
    try:
        return int(raw.strip())
    except ValueError as err:
        raise ValueError(f"DAY V2 config error: {key!r} must be an integer, got {raw!r}") from err


def _read_float(key: str, default: float) -> float:
    """Parse a float env var."""
    raw = os.environ.get(key)
    if raw is None:
        return default
    try:
        return float(raw.strip())
    except ValueError as err:
        raise ValueError(f"DAY V2 config error: {key!r} must be a float, got {raw!r}") from err


# ---------------------------------------------------------------------------
# Configuration values — evaluated at import time so import errors surface early.
# ---------------------------------------------------------------------------

DAY_V2_ENABLED: bool = _read_bool("DAY_V2_ENABLED", default=False)

# Bar durations
DAY_V2_PRIMARY_BAR_SECONDS: int = _read_int("DAY_V2_PRIMARY_BAR_SECONDS", default=900)  # 15m
DAY_V2_CONTEXT_BAR_SECONDS: int = _read_int("DAY_V2_CONTEXT_BAR_SECONDS", default=3600)  # 1H
DAY_V2_REGIME_BAR_SECONDS: int = _read_int("DAY_V2_REGIME_BAR_SECONDS", default=14400)  # 4H

# Hold ceiling
DAY_V2_MAX_HOLD_MINUTES: int = _read_int("DAY_V2_MAX_HOLD_MINUTES", default=300)  # 5H initial ceiling

# Winner protection — trail only after meaningful development
DAY_V2_WINNER_PROTECTION_MIN_MFE_PCT: float = _read_float("DAY_V2_WINNER_PROTECTION_MIN_MFE_PCT", default=0.008)  # 0.8%

# Structural invalidation — require N closed 15m bars before firing
DAY_V2_STRUCTURAL_INVALIDATION_BARS_CLOSED: int = _read_int("DAY_V2_STRUCTURAL_INVALIDATION_BARS_CLOSED", default=3)

# Universe — immutable, not overridable via env
DAY_V2_UNIVERSE: tuple[str, ...] = ("BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT")

# Catastrophic stop multiplier (ATR multiples)
DAY_V2_CATASTROPHIC_ATR_MULTIPLIER: float = _read_float("DAY_V2_CATASTROPHIC_ATR_MULTIPLIER", default=3.0)


def get_day_v2_config() -> dict:
    """Return all DAY V2 config values as a dict.

    Raises RuntimeError if DAY_V2_ENABLED is not explicitly True.
    This ensures shadow operation is an explicit opt-in.
    """
    if not DAY_V2_ENABLED:
        raise RuntimeError("DAY_V2_ENABLED is not set to 'true'. DAY V2 config is disabled. Set DAY_V2_ENABLED=true in environment to enable shadow operation.")
    return {
        "DAY_V2_ENABLED": DAY_V2_ENABLED,
        "DAY_V2_PRIMARY_BAR_SECONDS": DAY_V2_PRIMARY_BAR_SECONDS,
        "DAY_V2_CONTEXT_BAR_SECONDS": DAY_V2_CONTEXT_BAR_SECONDS,
        "DAY_V2_REGIME_BAR_SECONDS": DAY_V2_REGIME_BAR_SECONDS,
        "DAY_V2_MAX_HOLD_MINUTES": DAY_V2_MAX_HOLD_MINUTES,
        "DAY_V2_WINNER_PROTECTION_MIN_MFE_PCT": DAY_V2_WINNER_PROTECTION_MIN_MFE_PCT,
        "DAY_V2_STRUCTURAL_INVALIDATION_BARS_CLOSED": DAY_V2_STRUCTURAL_INVALIDATION_BARS_CLOSED,
        "DAY_V2_UNIVERSE": DAY_V2_UNIVERSE,
        "DAY_V2_CATASTROPHIC_ATR_MULTIPLIER": DAY_V2_CATASTROPHIC_ATR_MULTIPLIER,
    }
