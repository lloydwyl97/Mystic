"""Account-level operator labels. The API process is not the DAY submitter."""

from __future__ import annotations

import os
from typing import Any

from backend.config.day_entry_execution import raw_day_entry_execution_mode, trailing_buy_mode_active


def _env_mode(*keys: str) -> str:
    for key in keys:
        raw = str(os.getenv(key) or "").strip().lower()
        if raw:
            return raw
    return ""


def account_execution_is_live() -> bool:
    """DAY account authority, not whether this API worker holds a live client."""
    mode = _env_mode("MYSTIC_TRADING_MODE", "EXECUTION_MODE", "TRADING_MODE")
    return mode == "live"


def scalp_is_paper() -> bool:
    """Same authority the SCALP_V2 live loop uses to arm live entries."""
    try:
        from backend.services.binance_scalp.config import get_scalp_config
        from backend.services.binance_scalp.structural_mode import live_entry_enabled

        return not live_entry_enabled(get_scalp_config().resolved_structural_mode())
    except Exception:
        return True


def account_operator_labels(*, live_client_present: bool = False) -> dict[str, Any]:
    day_live = account_execution_is_live()
    scalp_label = "SCALP PAPER" if scalp_is_paper() else "SCALP LIVE"
    return {
        "mode": "LIVE" if day_live else "PAPER",
        "account_execution_live": day_live,
        "day_mode_display": "DAY LIVE" if day_live else "DAY PAPER",
        "scalp_mode_display": scalp_label,
        "operator_mode_labels": {
            "day": "DAY LIVE" if day_live else "DAY PAPER",
            "scalp": scalp_label,
        },
        "live_service_connected": bool(day_live or live_client_present),
        "real_orders_enabled": bool(day_live),
        "day_entry_execution_mode": raw_day_entry_execution_mode() or ("trailing_buy" if trailing_buy_mode_active() else ""),
        "trailing_buy_execution_mode": bool(trailing_buy_mode_active()),
    }
