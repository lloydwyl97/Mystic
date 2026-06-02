"""Canonical trade-worthiness time contract for DAY labels."""

from __future__ import annotations

import os

from backend.config.ai_primary_clock import primary_bar_seconds_for_strategy

TRAINING_LABEL_BAR_SECONDS = 60


def day_label_grid_seconds() -> int:
    return max(300, int(os.getenv("DAY_LABEL_GRID_SECONDS", str(4 * 3600))))


_MAX_HOLD_MIN_BY_CCXT: dict[str, int] = {
    "BTC/USDT": 480,
    "ETH/USDT": 480,
    "SOL/USDT": 420,
    "XRP/USDT": 420,
}
_DEFAULT_MAX_HOLD_MIN = 420


def _bus_to_ccxt_pair(trading_symbol_bus: str) -> str:
    s = (trading_symbol_bus or "").strip().upper().replace("-", "")
    if "/" in s:
        return s
    if s.endswith("USDT") and len(s) > 4:
        return f"{s[:-4]}/USDT"
    return f"{s}/USDT"


def max_hold_seconds_for_symbol(trading_symbol_bus: str) -> int:
    ccxt = _bus_to_ccxt_pair(trading_symbol_bus)
    max_hold_min = _MAX_HOLD_MIN_BY_CCXT.get(ccxt, _DEFAULT_MAX_HOLD_MIN)
    return int(max_hold_min) * 60


def max_hold_seconds_for_day_labels() -> int:
    return max(300, int(os.getenv("DAY_LABEL_MAX_HOLD_MINUTES", "480")) * 60)


def primary_label_bar_seconds_for_strategy(strategy_id: str) -> int:
    _sid = (strategy_id or "").strip().lower()
    return day_label_grid_seconds()


def label_horizon_primary_bars_for_strategy(strategy_id: str, trading_symbol_bus: str) -> int:
    gsec = day_label_grid_seconds()
    raw = os.getenv("DAY_RF_LABEL_LOOKAHEAD")
    if raw is not None and str(raw).strip() != "":
        return max(2, int(raw))
    secs = max_hold_seconds_for_day_labels()
    return max(2, int(secs // gsec))


def label_horizon_bars_for_symbol(trading_symbol_bus: str) -> int:
    return label_horizon_primary_bars_for_strategy("day", trading_symbol_bus)


def label_horizon_bars_for_strategy(strategy_id: str, trading_symbol_bus: str) -> int:
    return label_horizon_primary_bars_for_strategy("day", trading_symbol_bus)
