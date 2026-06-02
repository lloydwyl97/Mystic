"""
Day-trading model contract helpers (backward-compat module).

Authoritative timeframe list lives in ``backend.config.day_active_timeframes``;
``DAY_HTF_CCXT_ORDER`` now mirrors ``DAY_ACTIVE_TIMEFRAMES`` (not legacy 4-slot HTF only).
Training label cadence for day continues to anchor on ``4h`` via ``trade_worthiness_timing``.
"""

from __future__ import annotations

import os
from typing import Final

from backend.config.day_active_timeframes import DAY_ACTIVE_TIMEFRAMES

DAY_HTF_CCXT_ORDER: Final[tuple[str, ...]] = DAY_ACTIVE_TIMEFRAMES

DAY_HTF_MIN_BARS_1H: Final[int] = int(os.getenv("DAY_HTF_MIN_BARS_1H", "120"))
DAY_HTF_MIN_BARS_4H: Final[int] = int(os.getenv("DAY_HTF_MIN_BARS_4H", "80"))
DAY_HTF_MIN_BARS_1D: Final[int] = int(os.getenv("DAY_HTF_MIN_BARS_1D", "60"))
DAY_HTF_MIN_BARS_1W: Final[int] = int(os.getenv("DAY_HTF_MIN_BARS_1W", "24"))

DAY_HTF_FETCH_1H: Final[int] = int(os.getenv("DAY_HTF_FETCH_1H", "1000"))
DAY_HTF_FETCH_4H: Final[int] = int(os.getenv("DAY_HTF_FETCH_4H", "500"))
DAY_HTF_FETCH_1D: Final[int] = int(os.getenv("DAY_HTF_FETCH_1D", "400"))
DAY_HTF_FETCH_1W: Final[int] = int(os.getenv("DAY_HTF_FETCH_1W", "120"))

DAY_HTF_BLOCK_DIM: Final[int] = int(os.getenv("DAY_HTF_BLOCK_DIM", "31"))

FEATURE_VERSION_DAY_HTF: Final[int] = int(os.getenv("FEATURE_VERSION_DAY_HTF", "5"))


def day_htf_min_bars_for_tf(tf: str) -> int:
    t = (tf or "").strip().lower()
    if t == "1h":
        return max(20, DAY_HTF_MIN_BARS_1H)
    if t == "4h":
        return max(15, DAY_HTF_MIN_BARS_4H)
    if t == "1d":
        return max(10, DAY_HTF_MIN_BARS_1D)
    if t in ("1w", "1W"):
        return max(8, DAY_HTF_MIN_BARS_1W)
    return 40


def day_htf_fetch_limit(tf: str) -> int:
    t = (tf or "").strip().lower()
    if t == "1h":
        return min(1000, max(200, DAY_HTF_FETCH_1H))
    if t == "4h":
        return min(1000, max(100, DAY_HTF_FETCH_4H))
    if t == "1d":
        return min(1000, max(80, DAY_HTF_FETCH_1D))
    if t in ("1w", "1W"):
        return min(1000, max(52, DAY_HTF_FETCH_1W))
    return 500


__all__ = [
    "DAY_ACTIVE_TIMEFRAMES",
    "DAY_HTF_BLOCK_DIM",
    "DAY_HTF_CCXT_ORDER",
    "FEATURE_VERSION_DAY_HTF",
    "day_htf_fetch_limit",
    "day_htf_min_bars_for_tf",
]
