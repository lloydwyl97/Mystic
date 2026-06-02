"""
Signal Action Utilities

Single source of truth for extracting and normalizing signal actions (buy/sell/hold).
"""

from __future__ import annotations

from typing import Any


def normalize_signal_action(action: str | None) -> str:
    """
    Normalize a signal action string to canonical form.

    Args:
        action: Raw action string like "BUY", "buy", "SELL", "HOLD", etc.

    Returns:
        Normalized lowercase action: "buy", "sell", or "hold"
    """
    if action is None:
        return "hold"

    action_str = str(action).strip().lower()

    if action_str in ("buy", "long", "bullish", "1", "true"):
        return "buy"
    elif action_str in ("sell", "short", "bearish", "-1"):
        return "sell"
    else:
        return "hold"


def extract_canonical_action(signal: dict[str, Any]) -> str:
    """
    Extract the canonical action from a signal dictionary.

    Checks multiple possible field names in priority order:
    - side
    - action
    - prediction
    - recommendation
    - signal

    Args:
        signal: Signal dictionary with action field

    Returns:
        Normalized action: "buy", "sell", or "hold"
    """
    if not signal or not isinstance(signal, dict):
        return "hold"

    # Priority order for action fields
    action_fields = ["side", "action", "prediction", "recommendation", "signal"]

    for field in action_fields:
        value = signal.get(field)
        if value is not None:
            normalized = normalize_signal_action(str(value))
            if normalized in ("buy", "sell"):
                return normalized

    return "hold"
