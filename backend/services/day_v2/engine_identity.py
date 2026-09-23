"""Engine authority boundaries for Mystic trading engines.

The _AUTHORITY_TABLE is the single auditable authority boundary. Changing
any engine from SHADOW to LIVE (or LIVE to SHADOW) requires an explicit git
commit with a clear message. It must never happen automatically.

Promotion history:
  2026-09-21  DAY_V2_LIVE promoted from SHADOW to LIVE.
              Qualification: event-driven replay @ a88479a showed:
              Validation PF 1.18, Untouched-OOS PF 1.60, both green.
              Replay script: scripts/research/day_v2_replay.py
              Live activation gated by DAY_V2_ENABLED=true env var.

  2026-09-22  SCALP_V2_LIVE promoted from SCALP_V2_CANDIDATE to LIVE.
              Qualification: 24 paper-trade proof with positive expectancy
              and evidence-based exit calibration (giveback/stall recovery
              rates measured on real Binance.US fills).
              engine_id stored on positions and paper_trades is "SCALP_V2".

Authority notes:
  LEGACY_DAY_LIVE  — exit-only for legacy positions opened before DAY V2.
                     Must NEVER create new entries. LIVE authority is kept
                     so has_live_authority() returns True for accounting.
  SCALP_V2_LIVE    — full LIVE authority: creates entries AND manages exits.
  DAY_V2_LIVE      — full LIVE authority: creates entries AND manages exits.
                     Gated at application layer by DAY_V2_ENABLED env var.
"""

import time
from dataclasses import dataclass, field
from enum import Enum


class EngineId(str, Enum):
    """Unique identity for each engine in the trading system."""

    # Exit-only for legacy positions. Never creates new entries.
    LEGACY_DAY_LIVE = "LEGACY_DAY_LIVE"
    # Retired candidate label (SCALP V2 is now SCALP_V2_LIVE).
    SCALP_V2_CANDIDATE = "SCALP_V2_CANDIDATE"
    DAY_V2_SHADOW = "DAY_V2_SHADOW"
    # Promoted 2026-09-21 after qualifying event-driven replay.
    # engine_id stored on positions and intents is the string "DAY_V2".
    DAY_V2_LIVE = "DAY_V2"
    # Promoted 2026-09-22 after 24-trade paper proof with positive expectancy.
    # engine_id stored on positions and paper_trades is the string "SCALP_V2".
    SCALP_V2_LIVE = "SCALP_V2"


class AuthorityLevel(str, Enum):
    """Authority level granted to an engine."""

    LIVE = "LIVE"
    SHADOW = "SHADOW"
    DISABLED = "DISABLED"


# Static authority table — the single source of truth.
# Changing an engine's level requires an explicit git commit.
# This must never happen automatically.
_AUTHORITY_TABLE: dict[EngineId, AuthorityLevel] = {
    # Exit-only for legacy positions; kept LIVE so has_live_authority() is
    # accurate for accounting and position management.
    EngineId.LEGACY_DAY_LIVE: AuthorityLevel.LIVE,
    # Retired candidate label — superseded by SCALP_V2_LIVE.
    EngineId.SCALP_V2_CANDIDATE: AuthorityLevel.SHADOW,
    EngineId.DAY_V2_SHADOW: AuthorityLevel.SHADOW,
    # DAY_V2_LIVE has LIVE authority.  Gated at the application layer by
    # the DAY_V2_ENABLED env var — when False the live signal and entry
    # modules are not called. The table entry itself is always LIVE so
    # that has_live_authority() returns accurate results for audit.
    EngineId.DAY_V2_LIVE: AuthorityLevel.LIVE,
    # SCALP_V2_LIVE has LIVE authority.  Creates real Binance.US limit orders
    # and manages exits through the portfolio engine (execute_buy_fifo path).
    EngineId.SCALP_V2_LIVE: AuthorityLevel.LIVE,
}

# Frozen sets for fast membership checks.
LIVE_ENGINE_IDS: frozenset[EngineId] = frozenset({EngineId.LEGACY_DAY_LIVE, EngineId.DAY_V2_LIVE, EngineId.SCALP_V2_LIVE})
SHADOW_ENGINE_IDS: frozenset[EngineId] = frozenset({EngineId.SCALP_V2_CANDIDATE, EngineId.DAY_V2_SHADOW})

# Canonical engine_id string constants.
# Use these in all trade-recording code paths so the string and the enum never
# drift apart.  Import from here rather than hard-coding the literal string.
DAY_V2_ENGINE_ID: str = EngineId.DAY_V2_LIVE.value  # "DAY_V2"
SCALP_V2_ENGINE_ID: str = EngineId.SCALP_V2_LIVE.value  # "SCALP_V2"
LEGACY_DAY_LIVE_ENGINE_ID: str = EngineId.LEGACY_DAY_LIVE.value  # "LEGACY_DAY_LIVE"


def get_authority(engine_id: EngineId) -> AuthorityLevel:
    """Return the AuthorityLevel for the given engine."""
    return _AUTHORITY_TABLE[engine_id]


def has_live_authority(engine_id: EngineId) -> bool:
    """Return True iff the engine has LIVE authority."""
    return _AUTHORITY_TABLE[engine_id] == AuthorityLevel.LIVE


def assert_no_live_authority(engine_id: EngineId) -> None:
    """Raise PermissionError if the engine has LIVE authority.

    Call this at every boundary that must not route to real order execution.
    Live engines: LEGACY_DAY_LIVE (exit-only), DAY_V2_LIVE, SCALP_V2_LIVE.
    """
    if has_live_authority(engine_id):
        raise PermissionError(
            f"Engine {engine_id!r} has LIVE authority and must not be used in "
            "shadow/research code paths. Live engines (LEGACY_DAY_LIVE, "
            "DAY_V2_LIVE, SCALP_V2_LIVE) must never be instantiated from "
            "shadow or research code paths."
        )


def is_shadow_eligible(engine_id: EngineId) -> bool:
    """Return True iff the engine is SHADOW-eligible (read-only research allowed)."""
    return _AUTHORITY_TABLE[engine_id] == AuthorityLevel.SHADOW


@dataclass
class AuthorityBoundary:
    """Immutable record of an authority check at a specific boundary."""

    engine_id: EngineId
    caller_context: str
    checked_at: float = field(default_factory=time.time)


def create_authority_boundary(engine_id: EngineId, caller_context: str) -> AuthorityBoundary:
    """Assert no live authority and return a boundary record.

    Raises PermissionError if engine_id has LIVE authority.
    """
    assert_no_live_authority(engine_id)
    return AuthorityBoundary(
        engine_id=engine_id,
        caller_context=caller_context,
        checked_at=time.time(),
    )
