"""
Optional DAY entry quality gates — default OFF (observation-only telemetry elsewhere).

Env:
  DAY_REQUIRE_SETUP_CREDIT — when true, BUY requires setup_credit > 0 (strong_setup).
  DAY_ENTRY_RS_FLOOR — when set, ctx_rs_btc must be >= this value ([-1, 1] scale).
  DAY_ENTRY_RS_RANK_MAX — when set (1-4), RS rank among top-4 basket must be <= this value.
"""

from __future__ import annotations

import os


def _flag(name: str, default: str = "false") -> bool:
    return os.getenv(name, default).strip().lower() in ("1", "true", "yes", "on")


def day_require_setup_credit_enabled() -> bool:
    return _flag("DAY_REQUIRE_SETUP_CREDIT", "false")


def day_entry_rs_floor() -> float | None:
    raw = (os.getenv("DAY_ENTRY_RS_FLOOR") or "").strip()
    if not raw:
        return None
    try:
        return float(raw)
    except ValueError:
        return None


def day_entry_rs_rank_max() -> int | None:
    raw = (os.getenv("DAY_ENTRY_RS_RANK_MAX") or "").strip()
    if not raw:
        return None
    try:
        v = int(float(raw))
        return v if 1 <= v <= 4 else None
    except ValueError:
        return None


def day_entry_gates_enforced() -> bool:
    """True when any hard entry-quality gate is active."""
    return (
        day_require_setup_credit_enabled()
        or day_entry_rs_floor() is not None
        or day_entry_rs_rank_max() is not None
    )


def day_entry_gates_config_snapshot() -> dict[str, object]:
    return {
        "require_setup_credit": day_require_setup_credit_enabled(),
        "rs_floor": day_entry_rs_floor(),
        "rs_rank_max": day_entry_rs_rank_max(),
        "enforced": day_entry_gates_enforced(),
    }


__all__ = [
    "day_entry_gates_config_snapshot",
    "day_entry_gates_enforced",
    "day_entry_rs_floor",
    "day_entry_rs_rank_max",
    "day_require_setup_credit_enabled",
]
