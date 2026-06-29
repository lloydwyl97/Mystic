"""
Legacy position metadata helpers only (reporting/history).

Does NOT block DAY trading. Inventory recovery freeze was removed.
"""

from __future__ import annotations

from typing import Any

from backend.utils.symbols import normalize_symbol


def is_day_top4_symbol(symbol: str) -> bool:
    from backend.config.trading_universe import DAY_TRADE_SYMBOLS

    return normalize_symbol(symbol).replace("/", "").upper() in DAY_TRADE_SYMBOLS


def apply_legacy_tags_from_thesis(pos: Any, thesis_payload: dict[str, Any]) -> None:
    """Restore router/legacy flags from thesis_json for reporting."""
    opened = bool(thesis_payload.get("opened_under_router"))
    legacy = bool(thesis_payload.get("legacy_pre_regime_router"))
    if opened:
        pos.opened_under_router = True
        pos.legacy_pre_regime_router = False
    elif legacy or is_day_top4_symbol(getattr(pos, "symbol", "")):
        pos.legacy_pre_regime_router = True
        pos.opened_under_router = False


def thesis_json_for_position(pos: Any) -> dict[str, Any]:
    return {
        "entry_thesis": str(getattr(pos, "entry_thesis", "") or ""),
        "thesis_score": float(getattr(pos, "thesis_score", 0.0) or 0.0),
        "thesis_invalid_level": float(getattr(pos, "thesis_invalid_level", 0.0) or 0.0),
        "thesis_target_level": float(getattr(pos, "thesis_target_level", 0.0) or 0.0),
        "entry_vwap": float(getattr(pos, "entry_vwap", 0.0) or 0.0),
        "thesis_trend_tf": str(getattr(pos, "thesis_trend_tf", "") or ""),
        "day_route_regime_at_entry": str(getattr(pos, "day_route_regime_at_entry", "") or ""),
        "price_structure_regime_at_entry": str(getattr(pos, "price_structure_regime_at_entry", "") or ""),
        "max_hold_min": int(getattr(pos, "max_hold_min", 0) or 0),
        "trail_pct": float(getattr(pos, "trail_pct", 0.0) or 0.0),
        "legacy_pre_regime_router": bool(getattr(pos, "legacy_pre_regime_router", False)),
        "opened_under_router": bool(getattr(pos, "opened_under_router", False)),
    }


def mark_new_regime_entry(pos: Any) -> None:
    pos.opened_under_router = True
    pos.legacy_pre_regime_router = False
