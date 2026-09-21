"""DAY V2 Opportunity State Machine.

An 'opportunity' is a bounded context of market structure that may support
one or more entries. It is NOT the same as a position. A single opportunity
may include at most one re-entry after the first position closes, and only
if the original structure is still intact (structural_reset confirmed).

Key design rules:
- Wall-clock time alone does NOT advance any state.
- Closing a position does NOT automatically create a new opportunity.
- ACTIVE_NOT_ENTERED and POSITION_CLOSED_OPPORTUNITY_ACTIVE both require
  structural evidence to progress to POSITION_OPEN.
- Only one Opportunity per symbol may be FORMING, ACTIVE_NOT_ENTERED,
  POSITION_OPEN, or POSITION_CLOSED_OPPORTUNITY_ACTIVE at a time.
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any


class OpportunityState(str, Enum):
    NO_OPPORTUNITY = "NO_OPPORTUNITY"
    FORMING = "FORMING"
    ACTIVE_NOT_ENTERED = "ACTIVE_NOT_ENTERED"
    POSITION_OPEN = "POSITION_OPEN"
    POSITION_CLOSED_OPPORTUNITY_ACTIVE = "POSITION_CLOSED_OPPORTUNITY_ACTIVE"
    EXHAUSTED = "EXHAUSTED"
    RESET_ELIGIBLE = "RESET_ELIGIBLE"


# States that block a second opportunity for the same symbol.
_EXCLUSIVE_STATES: frozenset[OpportunityState] = frozenset(
    {
        OpportunityState.FORMING,
        OpportunityState.ACTIVE_NOT_ENTERED,
        OpportunityState.POSITION_OPEN,
        OpportunityState.POSITION_CLOSED_OPPORTUNITY_ACTIVE,
    }
)

# Valid state transitions: from_state -> set of allowed to_states
_ALLOWED_TRANSITIONS: dict[OpportunityState, set[OpportunityState]] = {
    OpportunityState.NO_OPPORTUNITY: {OpportunityState.FORMING},
    OpportunityState.FORMING: {
        OpportunityState.ACTIVE_NOT_ENTERED,
        OpportunityState.EXHAUSTED,
    },
    OpportunityState.ACTIVE_NOT_ENTERED: {
        OpportunityState.POSITION_OPEN,
        OpportunityState.EXHAUSTED,
    },
    OpportunityState.POSITION_OPEN: {
        OpportunityState.POSITION_CLOSED_OPPORTUNITY_ACTIVE,
    },
    OpportunityState.POSITION_CLOSED_OPPORTUNITY_ACTIVE: {
        OpportunityState.POSITION_OPEN,
        OpportunityState.EXHAUSTED,
    },
    OpportunityState.EXHAUSTED: {OpportunityState.RESET_ELIGIBLE},
    OpportunityState.RESET_ELIGIBLE: {OpportunityState.FORMING},
}


class InvalidOpportunityTransition(Exception):  # noqa: N818
    """Raised when an illegal state transition is attempted."""

    def __init__(
        self,
        from_state: OpportunityState,
        to_state: OpportunityState,
        reason: str,
    ) -> None:
        self.from_state = from_state
        self.to_state = to_state
        self.reason = reason
        super().__init__(f"Invalid transition {from_state} -> {to_state}: {reason}")


@dataclass
class OpportunityId:
    """Deterministic identifier for a market opportunity context."""

    symbol: str
    setup_family: str  # e.g. "MOMENTUM", "REVERSION", "BREAKOUT"
    regime: str  # e.g. "bull", "bear", "neutral"
    structural_anchor: str  # e.g. "4H_HIGH_20260920T16"
    formation_bar: str  # ISO timestamp of the forming 15m bar (rounded to 15m boundary)

    @property
    def canonical_id(self) -> str:
        """Deterministic string from all fields. Stable across restarts."""
        return f"{self.symbol}:{self.setup_family}:{self.regime}:{self.structural_anchor}:{self.formation_bar}"

    @classmethod
    def from_signal(
        cls,
        symbol: str,
        setup_family: str,
        regime: str,
        structural_anchor: str,
        bar_timestamp: str,
    ) -> OpportunityId:
        """Construct from a signal observation."""
        return cls(
            symbol=symbol,
            setup_family=setup_family,
            regime=regime,
            structural_anchor=structural_anchor,
            formation_bar=bar_timestamp,
        )

    def to_dict(self) -> dict[str, str]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, str]) -> OpportunityId:
        return cls(**d)


@dataclass
class Opportunity:
    """Live tracking entity for a bounded market opportunity context."""

    opportunity_id: OpportunityId
    state: OpportunityState
    created_at: float
    state_entered_at: float
    position_entry_count: int = 0
    last_position_closed_at: float | None = None
    exhaustion_reason: str = ""
    reset_eligible_at: float | None = None

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["state"] = self.state.value
        d["opportunity_id"] = self.opportunity_id.to_dict()
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Opportunity:
        opp_id = OpportunityId.from_dict(d["opportunity_id"])
        return cls(
            opportunity_id=opp_id,
            state=OpportunityState(d["state"]),
            created_at=d["created_at"],
            state_entered_at=d["state_entered_at"],
            position_entry_count=d.get("position_entry_count", 0),
            last_position_closed_at=d.get("last_position_closed_at"),
            exhaustion_reason=d.get("exhaustion_reason", ""),
            reset_eligible_at=d.get("reset_eligible_at"),
        )


class OpportunityStateMachine:
    """Enforces valid transitions for a single Opportunity."""

    def transition(
        self,
        opportunity: Opportunity,
        new_state: OpportunityState,
        *,
        reason: str,
        structural_reset: bool = False,
        now: float | None = None,
    ) -> Opportunity:
        """Apply a state transition and return the mutated Opportunity.

        Raises InvalidOpportunityTransition on illegal transitions.

        Additional guards beyond the base graph:
        - POSITION_CLOSED_OPPORTUNITY_ACTIVE -> POSITION_OPEN requires structural_reset=True
        - EXHAUSTED -> RESET_ELIGIBLE requires structural_reset=True
        """
        from_state = opportunity.state
        allowed = _ALLOWED_TRANSITIONS.get(from_state, set())

        if new_state not in allowed:
            raise InvalidOpportunityTransition(from_state, new_state, reason)

        # Extra structural evidence requirements
        if from_state == OpportunityState.POSITION_CLOSED_OPPORTUNITY_ACTIVE and new_state == OpportunityState.POSITION_OPEN and not structural_reset:
            raise InvalidOpportunityTransition(
                from_state,
                new_state,
                f"Re-entry from POSITION_CLOSED_OPPORTUNITY_ACTIVE requires structural_reset=True. Reason supplied: {reason}",
            )

        if from_state == OpportunityState.EXHAUSTED and new_state == OpportunityState.RESET_ELIGIBLE and not structural_reset:
            raise InvalidOpportunityTransition(
                from_state,
                new_state,
                f"EXHAUSTED -> RESET_ELIGIBLE requires structural_reset=True. Reason supplied: {reason}",
            )

        ts = now if now is not None else time.time()
        opportunity.state = new_state
        opportunity.state_entered_at = ts

        if new_state == OpportunityState.POSITION_OPEN:
            opportunity.position_entry_count += 1

        if new_state == OpportunityState.POSITION_CLOSED_OPPORTUNITY_ACTIVE:
            opportunity.last_position_closed_at = ts

        if new_state == OpportunityState.EXHAUSTED:
            opportunity.exhaustion_reason = reason

        if new_state == OpportunityState.RESET_ELIGIBLE:
            opportunity.reset_eligible_at = ts

        return opportunity


class OpportunityRegistry:
    """Manages per-symbol Opportunity lifecycle."""

    def __init__(self) -> None:
        self._opps: dict[str, Opportunity] = {}
        self._sm = OpportunityStateMachine()

    def get_or_create(self, symbol: str, opportunity_id: OpportunityId) -> Opportunity:
        """Return existing opportunity for symbol, or create one in NO_OPPORTUNITY."""
        if symbol not in self._opps:
            now = time.time()
            self._opps[symbol] = Opportunity(
                opportunity_id=opportunity_id,
                state=OpportunityState.NO_OPPORTUNITY,
                created_at=now,
                state_entered_at=now,
            )
        return self._opps[symbol]

    def transition(
        self,
        symbol: str,
        new_state: OpportunityState,
        *,
        reason: str,
        structural_reset: bool = False,
        now: float | None = None,
    ) -> Opportunity:
        """Transition the opportunity for symbol to new_state.

        Enforces that only one opportunity per symbol may be in an exclusive state.
        When transitioning to FORMING and there is already an exclusive-state
        opportunity for that symbol, the existing one must first be EXHAUSTED.
        """
        if symbol not in self._opps:
            raise KeyError(f"No opportunity registered for {symbol!r}. Call get_or_create first.")

        opp = self._opps[symbol]

        # Guard: cannot create a second active opportunity if one already exists.
        if new_state == OpportunityState.FORMING and opp.state in _EXCLUSIVE_STATES:
            raise InvalidOpportunityTransition(
                opp.state,
                new_state,
                f"Symbol {symbol!r} already has an active opportunity in state {opp.state!r}. Exhaust it before forming a new one.",
            )

        return self._sm.transition(
            opp,
            new_state,
            reason=reason,
            structural_reset=structural_reset,
            now=now,
        )

    def is_entry_eligible(self, symbol: str) -> tuple[bool, str]:
        """Return (eligible, reason).

        True only if:
        - State is ACTIVE_NOT_ENTERED (first entry), OR
        - State is POSITION_CLOSED_OPPORTUNITY_ACTIVE AND structural_reset
          would be confirmed (caller must pass structural_reset=True to transition).

        This method only checks state; the caller must separately confirm
        structural evidence before calling transition with structural_reset=True.
        """
        if symbol not in self._opps:
            return False, f"No opportunity registered for {symbol!r}"

        opp = self._opps[symbol]
        if opp.state == OpportunityState.ACTIVE_NOT_ENTERED:
            return True, "opportunity is ACTIVE_NOT_ENTERED — first entry eligible"
        if opp.state == OpportunityState.POSITION_CLOSED_OPPORTUNITY_ACTIVE:
            return (
                True,
                "opportunity is POSITION_CLOSED_OPPORTUNITY_ACTIVE — re-entry eligible if structural_reset confirmed by caller",
            )
        return False, f"state {opp.state!r} is not entry-eligible"

    def all_opportunities(self) -> dict[str, Opportunity]:
        """Return all registered opportunities (reference, not copy)."""
        return dict(self._opps)

    def to_dict(self) -> dict[str, Any]:
        return {symbol: opp.to_dict() for symbol, opp in self._opps.items()}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> OpportunityRegistry:
        registry = cls()
        for symbol, opp_dict in d.items():
            registry._opps[symbol] = Opportunity.from_dict(opp_dict)
        return registry
