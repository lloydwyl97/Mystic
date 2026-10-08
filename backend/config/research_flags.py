"""Explicit switches for research that is not live trading authority.

Both default off. Live DAY and SCALP read Binance.US books, the active
learner, and the safety and accounting path without these switches.
"""

from __future__ import annotations

import os


def _enabled(name: str) -> bool:
    return os.getenv(name, "false").strip().lower() in {"1", "true", "yes", "on"}


def policy_research_enabled() -> bool:
    """Prospective policy episodes and the research model zoo."""
    return _enabled("POLICY_RESEARCH_ENABLED")


def cross_venue_research_enabled() -> bool:
    """Coinbase and Kraken price discovery. Not an execution venue."""
    return _enabled("CROSS_VENUE_RESEARCH_ENABLED")
