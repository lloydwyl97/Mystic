"""Canonical primary signal / training clocks for DAY strategy."""

from __future__ import annotations

import os
from typing import Final

DAY_PRIMARY_BAR_SECONDS: Final[int] = int(os.getenv("DAY_PRIMARY_BAR_SECONDS", "900"))
DAY_PRIMARY_CCXT_TF: Final[str] = os.getenv("DAY_PRIMARY_CCXT_TF", "15m")

EXECUTION_REFINE_CCXT_TF: Final[str] = "1m"


def primary_bar_seconds_for_strategy(strategy_id: str) -> int:
    _sid = (strategy_id or "").strip().lower()
    return max(60, DAY_PRIMARY_BAR_SECONDS)


def primary_ccxt_timeframe(strategy_id: str) -> str:
    _sid = (strategy_id or "").strip().lower()
    return DAY_PRIMARY_CCXT_TF.strip() or "15m"


__all__ = [
    "DAY_PRIMARY_BAR_SECONDS",
    "DAY_PRIMARY_CCXT_TF",
    "EXECUTION_REFINE_CCXT_TF",
    "primary_bar_seconds_for_strategy",
    "primary_ccxt_timeframe",
]
