"""Explicit live AI strategy contracts (DAY-only)."""

from __future__ import annotations

import os
from dataclasses import dataclass
from enum import Enum
from pathlib import Path


class LiveStrategyId(str, Enum):
    DAY = "day"


@dataclass(frozen=True)
class LiveStrategyContract:
    """Single live strategy definition — clocks, artifacts, attribution."""

    strategy_id: str
    redis_signal_loop_seconds: int
    artifact_subdir: str  # under models/active/<subdir>/
    attribution_log_prefix: str


def parse_enabled_live_strategies() -> tuple[str, ...]:
    """Runtime strategy universe is DAY-only."""
    return (LiveStrategyId.DAY.value,)


def contract_for(strategy_id: str) -> LiveStrategyContract:
    sid = strategy_id.strip().lower()
    if sid == LiveStrategyId.DAY.value:
        return LiveStrategyContract(
            strategy_id=sid,
            redis_signal_loop_seconds=int(os.getenv("DAY_AI_SIGNAL_LOOP_SEC", "120")),
            artifact_subdir="day",
            attribution_log_prefix="DAY_AI",
        )
    raise ValueError(f"unknown live strategy_id={strategy_id!r}")


def redis_ai_signal_key(strategy_id: str, symbol_bus: str) -> str:
    """Canonical Redis ML signal key: ai_signal:<strategy_id>:<BUS_SYMBOL>."""
    bus = symbol_bus.strip().upper().replace("/", "")
    sid = strategy_id.strip().lower()
    return f"ai_signal:{sid}:{bus}"


# Match all live ML strategy/symbol keys (two-segment namespace after prefix).
REDIS_ML_SIGNAL_SCAN_PATTERN: str = "ai_signal:*:*"


def per_coin_artifact_file(models_active: Path | str, strategy_id: str, symbol_bus: str) -> Path:
    """models/active/<strategy>/<SYMBOL>_direction.pkl"""
    sid = strategy_id.strip().lower()
    sym = symbol_bus.strip().upper().replace("/", "")
    root = Path(models_active)
    return root / sid / f"{sym}_direction.pkl"


def live_ai_strict_startup() -> bool:
    """If True, missing any enabled strategy x symbol artifact aborts generator start (audit gates)."""
    return os.getenv("LIVE_AI_STRICT", "false").strip().lower() in ("1", "true", "yes", "on")


def live_ai_min_feature_version() -> int:
    """Legacy global floor. Prefer ``live_ai_min_feature_version_for_strategy``."""
    try:
        v = int(os.getenv("LIVE_AI_MIN_FEATURE_VERSION", "3"))
    except ValueError:
        return 3
    return max(1, min(5, v))


def live_ai_min_feature_version_for_strategy(strategy_id: str) -> int:
    """DAY full-MTF+v5 context contract."""
    try:
        return max(1, min(5, int(os.getenv("LIVE_AI_MIN_FEATURE_VERSION_DAY", "5"))))
    except ValueError:
        return 5


def live_ai_min_feature_versions_map(enabled: tuple[str, ...]) -> dict[str, int]:
    return {s.strip().lower(): live_ai_min_feature_version_for_strategy(s) for s in enabled}


def live_ai_fail_closed_without_context() -> bool:
    """When True and artifact is v2, missing ai_context hash skips emit (no neutral fill)."""
    return os.getenv("LIVE_AI_FAIL_CLOSED_CONTEXT", "true").strip().lower() in ("1", "true", "yes", "on")


def train_strategy_ids() -> tuple[str, ...]:
    """Training strategy universe is DAY-only."""
    return (LiveStrategyId.DAY.value,)


def parse_canonical_ml_signal_key(redis_key: str) -> tuple[str | None, str | None]:
    """
    Parse canonical ``ai_signal:<strategy_id>:<BUS>`` or legacy ``ai_signal:<BUS>``.
    Returns (strategy_id_or_None, symbol_bus_upper).
    """
    s = (redis_key or "").strip()
    if not s.startswith("ai_signal:"):
        return None, None
    rest = s[len("ai_signal:") :].strip()
    if not rest:
        return None, None
    if ":" not in rest:
        # Legacy ``ai_signal:BTCUSDT`` (no strategy segment). Live ML must not emit this shape.
        bus = rest.upper().replace("/", "")
        return None, bus if bus else None
    sid, _, bus = rest.partition(":")
    sid = sid.strip().lower()
    bus = bus.strip().upper().replace("/", "")
    if sid == "day" and bus:
        return sid, bus
    return None, None


def is_legacy_bus_only_ml_signal_key(redis_key: str) -> bool:
    """True if key looks like legacy ``ai_signal:<BUS>`` without strategy segment."""
    sid, bus = parse_canonical_ml_signal_key(redis_key)
    return bus is not None and sid is None


__all__ = [
    "REDIS_ML_SIGNAL_SCAN_PATTERN",
    "LiveStrategyContract",
    "LiveStrategyId",
    "contract_for",
    "is_legacy_bus_only_ml_signal_key",
    "live_ai_fail_closed_without_context",
    "live_ai_min_feature_version",
    "live_ai_min_feature_version_for_strategy",
    "live_ai_min_feature_versions_map",
    "live_ai_strict_startup",
    "parse_canonical_ml_signal_key",
    "parse_enabled_live_strategies",
    "per_coin_artifact_file",
    "redis_ai_signal_key",
    "train_strategy_ids",
]
