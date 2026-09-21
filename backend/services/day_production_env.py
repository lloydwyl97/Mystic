"""Load DAY configuration the same way production does.

Production (`backend/app_factory.py`) calls `load_dotenv(repo/.env, override=False)`
before trading modules read env-backed constants. Replay and analysis must use
that path — not `trading_economics` module defaults — when an env file is given.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

# Keys that change DAY replay economics / cadence. Secrets are never required.
_DAY_ENV_KEYS = (
    "DAY_TARGET_NOTIONAL_PER_SLOT_USD",
    "DAY_MAX_DEPLOYED_USD",
    "DAY_BASE_NOTIONAL_PER_SLOT_USD",
    "DAY_NOTIONAL_MULT",
    "DAY_MAX_OPEN_SLOTS",
    "DAY_MAX_BUYS_PER_BAR",
    "DAY_INTACT_4H_MAX_POSITIONS",
    "DAY_AI_SIGNAL_LOOP_SEC",
    "DAY_PRIMARY_BAR_SECONDS",
    "DAY_PATH_AWARE_EXIT",
    "DAY_STALL_EXIT_ENABLED",
    "DAY_GIVEBACK_EXIT_ENABLED",
    "ESTIMATED_ROUNDTRIP_COST",
    "ESTIMATED_ROUNDTRIP_COST_PCT",
    "COOLDOWN_SECONDS_AFTER_SELL",
    "POST_SELL_COOLDOWN_WALL_SEC",
    "MAX_OPEN_POSITIONS",
    "TRADING_MODE",
    "EXECUTION_MODE",
)


def production_dotenv_path(repo_root: str | Path | None = None) -> Path:
    if repo_root is None:
        repo_root = Path(__file__).resolve().parents[2]
    return Path(repo_root) / ".env"


def load_production_env(
    env_path: str | Path | None = None,
    *,
    override: bool = False,
) -> Path:
    """Same load as `backend/app_factory.py`: dotenv, no override by default."""
    path = Path(env_path) if env_path is not None else production_dotenv_path()
    load_dotenv(path, override=override)
    return path


def apply_env_map(values: dict[str, str], *, override: bool = True) -> None:
    """Apply an already-sanitized key map (Ocean dump without secrets)."""
    for key, value in values.items():
        if not override and os.getenv(key) not in (None, ""):
            continue
        os.environ[str(key)] = str(value)


def snapshot_day_env() -> dict[str, Any]:
    return {key: os.getenv(key) for key in _DAY_ENV_KEYS}


def runtime_float(name: str, default: float) -> float:
    raw = os.getenv(name, "")
    if not raw:
        return float(default)
    try:
        return float(raw)
    except (TypeError, ValueError):
        return float(default)


def runtime_int(name: str, default: int) -> int:
    raw = os.getenv(name, "")
    if not raw:
        return int(default)
    try:
        return int(raw)
    except (TypeError, ValueError):
        return int(default)
