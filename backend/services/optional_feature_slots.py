"""Optional 124-block sentiment/fundamental slots — inactive is OK, not a health defect."""

from __future__ import annotations

OPTIONAL_SENTIMENT_FUNDAMENTAL_SLOTS: frozenset[str] = frozenset(
    {
        "put_call_ratio",
        "vix",
        "market_cap",
        "supply",
        "circulating_supply",
        "max_supply",
        "market_dominance",
    }
)


def is_optional_slot(name: str) -> bool:
    return (name or "").strip().lower() in OPTIONAL_SENTIMENT_FUNDAMENTAL_SLOTS


__all__ = ["OPTIONAL_SENTIMENT_FUNDAMENTAL_SLOTS", "is_optional_slot"]
