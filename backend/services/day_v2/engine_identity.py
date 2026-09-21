"""DAY V2 engine authority boundaries.

The _AUTHORITY_TABLE is the single auditable authority boundary. Changing
LEGACY_DAY_LIVE to SHADOW or promoting a candidate to LIVE requires an
explicit git commit with a clear message. It must never happen automatically.
"""

import time
from dataclasses import dataclass, field
from enum import Enum


class EngineId(str, Enum):
    """Unique identity for each engine in the trading system."""

    LEGACY_DAY_LIVE = "LEGACY_DAY_LIVE"
    SCALP_V2_CANDIDATE = "SCALP_V2_CANDIDATE"
    DAY_V2_SHADOW = "DAY_V2_SHADOW"


class AuthorityLevel(str, Enum):
    """Authority level granted to an engine."""

    LIVE = "LIVE"
    SHADOW = "SHADOW"
    DISABLED = "DISABLED"


# Static authority table — the single source of truth.
# Changing LEGACY_DAY_LIVE to SHADOW or promoting a candidate to LIVE
# requires an explicit git commit with a clear message.
# This must never happen automatically.
_AUTHORITY_TABLE: dict[EngineId, AuthorityLevel] = {
    EngineId.LEGACY_DAY_LIVE: AuthorityLevel.LIVE,
    EngineId.SCALP_V2_CANDIDATE: AuthorityLevel.SHADOW,
    EngineId.DAY_V2_SHADOW: AuthorityLevel.SHADOW,
}

# Frozen sets for fast membership checks.
LIVE_ENGINE_IDS: frozenset[EngineId] = frozenset({EngineId.LEGACY_DAY_LIVE})
SHADOW_ENGINE_IDS: frozenset[EngineId] = frozenset({EngineId.SCALP_V2_CANDIDATE, EngineId.DAY_V2_SHADOW})


def get_authority(engine_id: EngineId) -> AuthorityLevel:
    """Return the AuthorityLevel for the given engine."""
    return _AUTHORITY_TABLE[engine_id]


def has_live_authority(engine_id: EngineId) -> bool:
    """Return True iff the engine has LIVE authority."""
    return _AUTHORITY_TABLE[engine_id] == AuthorityLevel.LIVE


def assert_no_live_authority(engine_id: EngineId) -> None:
    """Raise PermissionError if the engine has LIVE authority.

    Call this at every boundary that must not route to real order execution.
    """
    if has_live_authority(engine_id):
        raise PermissionError(
            f"Engine {engine_id!r} has LIVE authority and must not be used in "
            "shadow/research code paths. Only LEGACY_DAY_LIVE may have live "
            "authority and it must never be instantiated from the day_v2 package."
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
