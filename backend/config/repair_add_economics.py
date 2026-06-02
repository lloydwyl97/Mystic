"""
Controlled one-time DAY position repair-add economics.

Repair add: a single optional add to an existing open top-4 position when
drawdown + AI confirmation + allocation caps allow. Not repeated averaging down.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Final

logger = logging.getLogger(__name__)


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name, "")
    if not raw:
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on")


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name, "")
    if not raw:
        return float(default)
    try:
        return float(raw)
    except (TypeError, ValueError):
        logger.warning("REPAIR_ADD env %s=%r invalid; using %s", name, raw, default)
        return float(default)


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name, "")
    if not raw:
        return int(default)
    try:
        return int(raw)
    except (TypeError, ValueError):
        logger.warning("REPAIR_ADD env %s=%r invalid; using %s", name, raw, default)
        return int(default)


REPAIR_ADD_ENABLED: Final[bool] = _env_bool("REPAIR_ADD_ENABLED", True)
MAX_REPAIR_ADDS_PER_POSITION: Final[int] = _env_int("MAX_REPAIR_ADDS_PER_POSITION", 1)
REPAIR_ADD_TRIGGER_NET_PNL: Final[float] = _env_float("REPAIR_ADD_TRIGGER_NET_PNL", -0.015)
REPAIR_ADD_MIN_CONFIDENCE: Final[float] = _env_float("REPAIR_ADD_MIN_CONFIDENCE", 0.80)
REPAIR_ADD_MAX_TOTAL_SYMBOL_ALLOCATION_PCT: Final[float] = _env_float("REPAIR_ADD_MAX_TOTAL_SYMBOL_ALLOCATION_PCT", 0.25)
REPAIR_ADD_SIZE_PCT_OF_ORIGINAL: Final[float] = _env_float("REPAIR_ADD_SIZE_PCT_OF_ORIGINAL", 1.0)
REPAIR_ADD_COOLDOWN_SEC: Final[int] = _env_int("REPAIR_ADD_COOLDOWN_SEC", 21600)
REPAIR_ADD_MIN_RECOVERY_IMPROVEMENT_PCT: Final[float] = _env_float("REPAIR_ADD_MIN_RECOVERY_IMPROVEMENT_PCT", 0.003)
REPAIR_ADD_REQUIRED_FEATURE_VERSION: Final[int] = _env_int("REPAIR_ADD_REQUIRED_FEATURE_VERSION", 5)
REPAIR_ADD_REQUIRED_FEATURE_DIM: Final[int] = _env_int("REPAIR_ADD_REQUIRED_FEATURE_DIM", 145)


@dataclass(frozen=True)
class RepairAddEconomicsSnapshot:
    enabled: bool
    max_adds_per_position: int
    trigger_net_pnl: float
    min_confidence: float
    max_symbol_allocation_pct: float
    size_pct_of_original: float
    cooldown_sec: int
    min_recovery_improvement_pct: float


def get_repair_add_economics() -> RepairAddEconomicsSnapshot:
    return RepairAddEconomicsSnapshot(
        enabled=REPAIR_ADD_ENABLED,
        max_adds_per_position=MAX_REPAIR_ADDS_PER_POSITION,
        trigger_net_pnl=REPAIR_ADD_TRIGGER_NET_PNL,
        min_confidence=REPAIR_ADD_MIN_CONFIDENCE,
        max_symbol_allocation_pct=REPAIR_ADD_MAX_TOTAL_SYMBOL_ALLOCATION_PCT,
        size_pct_of_original=REPAIR_ADD_SIZE_PCT_OF_ORIGINAL,
        cooldown_sec=REPAIR_ADD_COOLDOWN_SEC,
        min_recovery_improvement_pct=REPAIR_ADD_MIN_RECOVERY_IMPROVEMENT_PCT,
    )


def log_repair_add_economics_at_startup() -> RepairAddEconomicsSnapshot:
    snap = get_repair_add_economics()
    logger.warning(
        "REPAIR_ADD_ECONOMICS enabled=%s max_adds=%s trigger_net_pnl=%s min_conf=%s max_alloc_pct=%s size_pct_original=%s cooldown_sec=%s min_recovery_improve_pct=%s",
        snap.enabled,
        snap.max_adds_per_position,
        snap.trigger_net_pnl,
        snap.min_confidence,
        snap.max_symbol_allocation_pct,
        snap.size_pct_of_original,
        snap.cooldown_sec,
        snap.min_recovery_improvement_pct,
    )
    return snap
